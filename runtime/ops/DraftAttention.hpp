#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/LaneRows.hpp"

#include <cstdint>
#include <span>
#include <memory>
#include <vector>

namespace splash::ops {

struct DraftAttentionShape final {
  uint32_t hiddenSize = 0;
  uint32_t dynamicSize = 0;
  uint32_t qkvSize = 0;
  uint32_t attentionSize = 0;
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;

  bool operator==(const DraftAttentionShape &) const = default;
};

struct DraftAttentionWorkspace final {
  uint64_t convolutionBytes = 0;
  uint64_t qkvBytes = 0;
  uint64_t groupedQueriesBytes = 0;
  uint64_t queryKeysBytes = 0;
  uint64_t queryValuesBytes = 0;
};

// Constructed only by DraftAttention::plan so the shape, physical rows and
// workspace cannot disagree. The grouped-queries tensor also carries the
// split partials of the attention core behind the query rows, so the core
// needs no device scratch beyond these tensors.
class DraftAttentionPlan final {
public:
  [[nodiscard]] DraftAttentionShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t lanes() const noexcept { return lanes_; }
  [[nodiscard]] DraftAttentionWorkspace workspace() const noexcept;

private:
  DraftAttentionPlan(DraftAttentionShape shape, uint32_t lanes)
      : shape_(shape), lanes_(lanes) {}

  DraftAttentionShape shape_;
  uint32_t lanes_;

  friend class DraftAttention;
};

enum class DraftConvolutionStage : uint8_t { Prepare, Residual };

struct DraftConvolutionBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer weights;
  metal::MetalBuffer residual;
  metal::MetalBuffer output;
};

struct DraftPrepareBuffers final {
  metal::MetalBuffer qkv;
  metal::MetalBuffer groupedQueries;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
};

struct DraftDecodeAttentionBuffers final {
  // The prepared query rows, grouped by query head, and the split partials
  // behind them. The attention leaves its rows in the query rows' place, in
  // [row][query head][dimension] order, the output projection's input.
  metal::MetalBuffer groupedQueries;
  std::span<const metal::MetalBuffer> persistentKeys;
  std::span<const metal::MetalBuffer> persistentValues;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
};

// DSpark's global layer K/V as segments (metal/abi/DraftAttention.h), reached
// through the table of their GPU addresses: every key segment's, then every
// value segment's. Segment s holds the rows from s * segmentTokens, the last
// of them fewer; together they hold `rows` rows of 8 KV heads x 128 bfloat16.
struct DraftSegmentedContext final {
  std::shared_ptr<const std::vector<metal::MetalBuffer>> table;
  uint32_t segmentTokens = 0;
  uint32_t rows = 0;

  // Builds the table of `keys` and `values`, which hold `rows` rows in
  // segments of `segmentTokens`; std::invalid_argument for segments that do not.
  [[nodiscard]] static DraftSegmentedContext make(std::span<const metal::MetalBuffer> keys,
                                                  std::span<const metal::MetalBuffer> values,
                                                  uint32_t segmentTokens, uint32_t rows);
  // Segments of each of keys and values.
  [[nodiscard]] uint32_t segments() const noexcept {
    return table ? static_cast<uint32_t>(table->size() / 2) : 0;
  }
  // The segment count `rows` rows take at `segmentTokens` a segment.
  [[nodiscard]] static constexpr uint32_t segmentCount(uint32_t rows, uint32_t segmentTokens) noexcept {
    return segmentTokens ? static_cast<uint32_t>((uint64_t{rows} + segmentTokens - 1) / segmentTokens) : 0;
  }
  // Throws std::invalid_argument unless the table holds a buffer of the
  // bytes of each segment's rows, key segments then value segments.
  void validate() const;
};

class DraftAttention final {
public:
  [[nodiscard]] static DraftAttentionPlan plan(DraftAttentionShape shape,
                                               uint32_t lanes);

  static void addConvolution(metal::CommandGraph &graph,
                             DraftConvolutionBuffers buffers,
                             const DraftAttentionPlan &plan,
                             DraftConvolutionStage stage);
  static void addPrepare(metal::CommandGraph &graph,
                         DraftPrepareBuffers buffers,
                         const DraftAttentionPlan &plan);
  static void addDecode(metal::CommandGraph &graph,
                        DraftDecodeAttentionBuffers buffers,
                        std::span<const uint32_t> cacheLengths,
                        const DraftAttentionPlan &plan);
  static void addDSparkFusion(metal::CommandGraph &graph, metal::MetalBuffer captured,
                              metal::MetalBuffer fuse, metal::MetalBuffer norm, metal::MetalBuffer output,
                              uint32_t rows, uint32_t consumer);
  static void addDSparkHeadPrepare(metal::CommandGraph &graph, metal::MetalBuffer input,
                                   metal::MetalBuffer norm, metal::MetalBuffer output, uint32_t rows,
                                   uint32_t heads, uint32_t startPosition);
  // The bytes of the workspace a DSpark block attention's split partials take (DSPARK_ATTENTION_MAX_SPLITS).
  [[nodiscard]] static uint64_t dsparkAttentionWorkspaceBytes();
  static void addDSparkAttention(metal::CommandGraph &graph, metal::MetalBuffer queries,
                                  metal::MetalBuffer contextKeys, metal::MetalBuffer contextValues,
                                  metal::MetalBuffer blockKeys, metal::MetalBuffer blockValues,
                                  metal::MetalBuffer output, metal::MetalBuffer workspace, uint32_t contextRows,
                                  uint32_t contextFirst, uint32_t contextCapacity);
  static void addDSparkBlockInput(metal::CommandGraph &graph, metal::MetalBuffer anchor,
                                   metal::MetalBuffer mask, metal::MetalBuffer hidden, metal::MetalBuffer residual);
  static void addDSparkResidualNorm(metal::CommandGraph &graph, metal::MetalBuffer input,
                                    metal::MetalBuffer residual, metal::MetalBuffer norm,
                                    metal::MetalBuffer output);
  static void addDSparkSwiGlu(metal::CommandGraph &graph, metal::MetalBuffer gate,
                              metal::MetalBuffer up, metal::MetalBuffer output);
  static void addDSparkContextStore(metal::CommandGraph &graph, metal::MetalBuffer keys,
                                    metal::MetalBuffer values, metal::MetalBuffer persistentKeys,
                                    metal::MetalBuffer persistentValues, uint32_t rows,
                                    uint32_t startPosition, uint32_t basePosition, uint32_t capacity);
  static void addDSparkCommitGather(metal::CommandGraph &graph, metal::MetalBuffer captured,
                                     metal::MetalBuffer retained, metal::MetalBuffer laneRows,
                                     metal::MetalBuffer output, uint32_t packedRows, uint32_t laneCap);
  static void addDSparkCommitStore(metal::CommandGraph &graph, metal::MetalBuffer keys,
                                    metal::MetalBuffer values, metal::MetalBuffer retained,
                                    metal::MetalBuffer persistentKeys, metal::MetalBuffer persistentValues,
                                    uint32_t laneCap, uint32_t startPosition, uint32_t basePosition,
                                    uint32_t capacity);
  // The segmented global layer of DSpark (SPLASH_DRAFT_GLOBAL_SEGMENT_TOKENS,
  // metal/abi/DraftAttention.h): the same operations as the ring's above, over
  // rows of positions counted from `basePosition` that cross segments,
  // reaching every segment through its address table. Rows read or written
  // stay below context.rows.
  static void addDSparkSegmentedAttention(metal::CommandGraph &graph, metal::MetalBuffer queries,
                                          const DraftSegmentedContext &context, metal::MetalBuffer blockKeys,
                                          metal::MetalBuffer blockValues, metal::MetalBuffer output,
                                          metal::MetalBuffer workspace, uint32_t contextRows, uint32_t contextFirst);
  static void addDSparkSegmentedContextStore(metal::CommandGraph &graph, metal::MetalBuffer keys,
                                             metal::MetalBuffer values, const DraftSegmentedContext &context,
                                             uint32_t rows, uint32_t startPosition, uint32_t basePosition);
  static void addDSparkSegmentedCommitStore(metal::CommandGraph &graph, metal::MetalBuffer keys,
                                            metal::MetalBuffer values, metal::MetalBuffer retained,
                                            const DraftSegmentedContext &context, uint32_t laneCap,
                                            uint32_t startPosition, uint32_t basePosition);
  // The context writers. A row of contextKv holds its keys, then its values
  // (kvHeads * headDimension each); its normalized, rotated keys and its
  // values go to its position's slot of the ring.
  static void addContextPrefill(
      metal::CommandGraph &graph, metal::MetalBuffer contextKv,
      metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
      metal::MetalBuffer ropeSin, metal::MetalBuffer keys,
      metal::MetalBuffer values, uint32_t tokens, uint32_t startPosition,
      DraftAttentionShape shape);
  // Each lane's retained rows of a decode step (a GPU mask of its rows, which
  // acceptance writes) among its rows of the step (laneRows, SplashLaneRows,
  // which the step's proposer writes): the path's context rows among the
  // lane's of contextKv, its i-th at the lane's start position plus i,
  // rotated by the lane's RoPE row i (a lane's RoPE rows hold its verify
  // block's positions, SPLASH_TARGET_VERIFY_ROWS a lane).
  static void addContextCommit(
      metal::CommandGraph &graph, metal::MetalBuffer contextKv,
      metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
      metal::MetalBuffer ropeSin,
      std::span<const metal::MetalBuffer> persistentKeys,
      std::span<const metal::MetalBuffer> persistentValues,
      metal::MetalBuffer retainedRows, metal::MetalBuffer laneRows,
      std::span<const uint32_t> startPositions, const StepRows &step,
      DraftAttentionShape shape);

  // A context window: for each slot of a ring, the bf16 context row of
  // `width` values the slot's keys and values are computed from, in 4-bit
  // codes of 64-value groups with each group's fp16 scale and minimum
  // (metal/abi/DraftAttention.h). Positions map to slots as in the rings.
  [[nodiscard]] static uint64_t contextWindowBytes(uint32_t width);
  // Stores `rows` context rows in the slots of positions startPosition on.
  static void addWindowStore(metal::CommandGraph &graph,
                             metal::MetalBuffer context,
                             metal::MetalBuffer window, uint32_t rows,
                             uint32_t startPosition, uint32_t width);
  // Stores the first `rows` (at most a verify block's) of the context rows a
  // lane retained of its rows of a decode step of `contextRows` context rows
  // (`retained`, its acceptance's mask of them, metal/abi/VerifyTree.h, among
  // its rows `laneRows`, SplashLaneRows), the path's i-th at position
  // startPosition + i, as addWindowStore does.
  static void addWindowCommit(metal::CommandGraph &graph,
                              metal::MetalBuffer context,
                              metal::MetalBuffer window,
                              metal::MetalBuffer retained,
                              metal::MetalBuffer laneRows,
                              uint32_t contextRows, uint32_t rows,
                              uint32_t startPosition, uint32_t width);
  // Loads the rows of positions startPosition on back into `rows` rows.
  static void addWindowLoad(metal::CommandGraph &graph,
                            metal::MetalBuffer window,
                            metal::MetalBuffer context, uint32_t rows,
                            uint32_t startPosition, uint32_t width);
};

} // namespace splash::ops
