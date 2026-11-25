// Draft sliding-window attention against a direct CPU reference. The kernel
// streams the 2048-slot ring in ring-aligned 128-token tiles over a fixed
// number of splits and combines their partials in split order; these cases
// cover an unaligned wrap, a wrap that starts exactly one slot before the ring
// end, the short prefix before any wrap, aligned wraps, windows that fill
// exactly one or several splits, and the last slot of the ring, so that tile
// bounds, the window mask, empty splits and the empty-row softmax guard are
// all exercised.
#include "ChainRows.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/DraftAttention.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::metal::CommandGraph;
using namespace splash::ops;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::ulpBf16;
using splash::test::mostRows;
using splash::test::rowsEnd;

constexpr uint32_t kRows = 8;
// A lane's RoPE rows, its verify block's positions, of which its draft's rows
// take the first kRows.
constexpr uint32_t kRopeRows = SPLASH_TARGET_VERIFY_ROWS;

// Lanes' rows of a step as its proposer writes them on the device
// (SplashLaneRows), and the step of them: their rows' end and most rows.
MetalBuffer laneRowsBuffer(MetalBackend &backend, std::span<const LaneRows> lanes) {
  MetalBuffer buffer =
      backend.allocateBuffer(lanes.size() * sizeof(LaneRows), BufferStorage::Shared, "draft lane rows");
  std::ranges::copy(lanes, static_cast<LaneRows *>(buffer.contents()));
  return buffer;
}
StepRows stepOf(std::span<const LaneRows> lanes) {
  return {static_cast<uint32_t>(lanes.size()), static_cast<uint32_t>(rowsEnd(lanes)), mostRows(lanes)};
}
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQueryHeadsPerKv = 4;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kWindow = 2048;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
// The shipped split count and the fp32 partial one split leaves per (lane,
// head) behind the grouped queries: 32 rows x (128 + max + sum).
constexpr uint32_t kSplits = SPLASH_DRAFT_ATTENTION_SPLITS;
constexpr uint64_t kPartialBytes = uint64_t{32} * 130 * sizeof(float);
constexpr uint32_t kAttention = kKvHeads * kQueryHeadsPerKv * kHeadDim;
constexpr uint32_t kGroupRows = kQueryHeadsPerKv * kRows;
constexpr float kScale = 0.08838834765F;

// The drafts of Qwen3.8-27B, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B and
// Qwen3.5-122B-A10B: one convolution kernel takes every hidden width.
constexpr std::array kShapes{
    DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2560, 640, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{3072, 768, 6144, 4096, 32, 8, 128}};

using splash::test::rejects;
using splash::test::require;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(count * sizeof(uint16_t), BufferStorage::Shared,
                             label);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = floatToBf16(random.unit());
  return buffer;
}

// Reference for one lane and kv head: every row attends to the ring window it
// can see plus all eight current rows, exactly as the kernel's mask defines.
void referenceRows(const uint16_t *queries, const uint16_t *keys,
                   const uint16_t *values, const uint16_t *queryKeys,
                   const uint16_t *queryValues, uint32_t cacheLength,
                   std::vector<float> &output) {
  const uint32_t commonStart =
      cacheLength >= kWindow - 1 ? cacheLength - (kWindow - 1) : 0;
  const uint32_t oldCount = cacheLength - commonStart;
  std::vector<double> scores(oldCount + kRows);
  std::vector<double> accumulated(kHeadDim);
  for (uint32_t row = 0; row < kGroupRows; ++row) {
    const uint32_t proposal = row % kRows;
    const uint32_t queryPosition = cacheLength + proposal;
    const uint32_t rowStart =
        queryPosition >= kWindow - 1 ? queryPosition - (kWindow - 1) : 0;
    const uint32_t hiddenPrefix = rowStart - commonStart;
    const uint16_t *query = queries + uint64_t{row} * kHeadDim;
    double best = -INFINITY;
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      double score = -INFINITY;
      if (key >= oldCount) {
        const uint32_t current = key - oldCount;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(bf16ToFloat(query[d])) *
                 bf16ToFloat(queryKeys[current * kHeadDim + d]);
        }
        score = dot * kScale;
      } else if (key >= hiddenPrefix) {
        const uint32_t slot = (commonStart + key) % kWindow;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(bf16ToFloat(query[d])) *
                 bf16ToFloat(keys[uint64_t{slot} * kHeadDim + d]);
        }
        score = dot * kScale;
      }
      scores[key] = score;
      best = std::max(best, score);
    }
    double sum = 0.0;
    std::fill(accumulated.begin(), accumulated.end(), 0.0);
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      if (scores[key] == -INFINITY)
        continue;
      const double probability = std::exp(scores[key] - best);
      sum += probability;
      for (uint32_t d = 0; d < kHeadDim; ++d) {
        const float value =
            key >= oldCount
                ? bf16ToFloat(queryValues[uint64_t{d} * kRows + key - oldCount])
                : bf16ToFloat(values[uint64_t{d} * kWindow + (commonStart + key) % kWindow]);
        accumulated[d] += probability * value;
      }
    }
    for (uint32_t d = 0; d < kHeadDim; ++d)
      output[uint64_t{row} * kHeadDim + d] =
          static_cast<float>(accumulated[d] / sum);
  }
}

// Where the attention leaves row `row` of a KV head's 32 reference rows, its
// four query heads' eight proposal rows each, in its lane's rows: they are in
// [proposal row][query head][dim] order, the output projection's input.
constexpr uint64_t attentionOffset(uint32_t head, uint64_t row) {
  return (row % kRows * (kKvHeads * kQueryHeadsPerKv) + head * kQueryHeadsPerKv + row / kRows) * kHeadDim;
}

void runCase(MetalBackend &backend, uint32_t lanes, DraftAttentionShape shape,
             const std::array<uint32_t, kLanes> &cacheLengths) {
  Random random(0x5eed0000ULL + cacheLengths[0]);
  const uint64_t ringElements = uint64_t{kKvHeads} * kWindow * kHeadDim;
  // The grouped-queries tensor is sized by the plan so the split partials
  // behind the query rows end exactly at the allocation under validation.
  MetalBuffer queries = randomBfloat(
      backend, DraftAttention::plan(shape, lanes).workspace().groupedQueriesBytes / 2,
      random, "draft queries");
  std::vector<MetalBuffer> keys;
  std::vector<MetalBuffer> values;
  // Each tensor follows the last head's ring with one 128-token tile of bf16
  // NaN. Shader validation does not see the MPP loads of a tile that reads
  // past the ring, but a masked NaN value still reaches its row as
  // 0 x NaN, which the finiteness check below rejects.
  const uint64_t tailElements = uint64_t{128} * kHeadDim;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    keys.push_back(randomBfloat(backend, ringElements + tailElements, random,
                                "draft keys"));
    values.push_back(randomBfloat(backend, ringElements + tailElements, random,
                                  "draft values"));
    for (const MetalBuffer &tensor : {keys.back(), values.back()})
      std::fill_n(static_cast<uint16_t *>(tensor.contents()) + ringElements,
                  tailElements, uint16_t{0x7FC0});
  }
  MetalBuffer queryKeys = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kRows * kHeadDim, random,
      "draft query keys");
  MetalBuffer queryValues = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kHeadDim * kRows, random,
      "draft query values");
  std::vector<uint16_t> input(
      static_cast<const uint16_t *>(queries.contents()),
      static_cast<const uint16_t *>(queries.contents()) +
          uint64_t{lanes} * kRows * kAttention);

  CommandGraph graph;
  DraftAttention::addDecode(graph,
      {queries, keys, values, queryKeys, queryValues},
      std::span(cacheLengths).first(lanes), DraftAttention::plan(shape, lanes));
  const auto dispatches = graph.dispatches();
  require(dispatches.size() == 2 &&
              dispatches[0].threadgroups.x == kKvHeads &&
              dispatches[0].threadgroups.y == lanes &&
              dispatches[0].threadgroups.z == kSplits &&
              dispatches[1].threadgroups.x == kKvHeads &&
              dispatches[1].threadgroups.y == lanes &&
              dispatches[1].threadgroups.z == 1,
          "draft attention core dispatch changed");
  static_cast<void>(backend.submitCommandAsync(dispatches).wait());
  // A cache length for each of the plan's lanes, no fewer.
  CommandGraph mismatched;
  rejects(
      [&] {
        DraftAttention::addDecode(mismatched,
            {queries, keys, values, queryKeys, queryValues},
            std::span(cacheLengths).first(lanes - 1),
            DraftAttention::plan(shape, lanes));
      },
      "invalid draft attention geometry", "fewer cache lengths than lanes were accepted");
  require(mismatched.empty(), "mismatched cache lengths encoded a graph");

  const auto *output = static_cast<const uint16_t *>(queries.contents());
  std::vector<float> reference(uint64_t{kGroupRows} * kHeadDim);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    // Every head: the final one's ring is the one the NaN tile follows, and
    // together the eight write every value of the lane's rows.
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      const uint64_t laneOffset = uint64_t{lane} * kRows * kAttention;
      const uint64_t queryOffset =
          laneOffset + uint64_t{head} * kGroupRows * kHeadDim;
      referenceRows(
          input.data() + queryOffset,
          static_cast<const uint16_t *>(keys[lane].contents()) +
              uint64_t{head} * kWindow * kHeadDim,
          static_cast<const uint16_t *>(values[lane].contents()) +
              uint64_t{head} * kWindow * kHeadDim,
          static_cast<const uint16_t *>(queryKeys.contents()) +
              (uint64_t{lane} * kKvHeads + head) * kRows * kHeadDim,
          static_cast<const uint16_t *>(queryValues.contents()) +
              (uint64_t{lane} * kKvHeads + head) * kHeadDim * kRows,
          cacheLengths[lane], reference);
      for (uint64_t index = 0; index < reference.size(); ++index) {
        const uint64_t row = index / kHeadDim;
        const uint64_t dim = index % kHeadDim;
        const float actual = bf16ToFloat(output[laneOffset + attentionOffset(head, row) + dim]);
        const float expected = reference[index];
        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) >
                0.02F + 0.02F * std::fabs(expected)) {
          std::cerr << "lane " << lane << " head " << head << " length "
                    << cacheLengths[lane] << " row " << row
                    << " dim " << dim << ": " << actual
                    << " vs " << expected << '\n';
          throw std::runtime_error("draft attention diverged from reference");
        }
      }
    }
  }
}

void planGeometry() {
  for (const auto shape : kShapes) {
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      const uint64_t rows = uint64_t{lanes} * kRows;
      const auto plan = DraftAttention::plan(shape, lanes);
      const auto workspace = plan.workspace();
      require(plan.lanes() == lanes && plan.shape() == shape &&
                  workspace.convolutionBytes == rows * shape.hiddenSize * 2 &&
                  workspace.qkvBytes == rows * 6144 * 2 &&
                  workspace.groupedQueriesBytes ==
                      rows * 4096 * 2 +
                          uint64_t{lanes} * kKvHeads * kSplits * kPartialBytes &&
                  workspace.queryKeysBytes == rows * 8 * 128 * 2 &&
                  workspace.queryValuesBytes == rows * 8 * 128 * 2,
              "draft plan padded lanes or changed tensor storage");
    }
    rejects([&] { (void)DraftAttention::plan(shape, 0); }, "invalid draft batch width",
            "a draft plan of no lanes was accepted");
    rejects([&] { (void)DraftAttention::plan(shape, kLanes + 1); }, "invalid draft batch width",
            "a draft plan wider than a batch was accepted");
  }
  auto unsupported = kShapes[0];
  unsupported.queryHeads = 16;
  rejects([&] { (void)DraftAttention::plan(unsupported, 1); },
          "unsupported compiled draft attention shape",
          "a draft shape without a compiled kernel was accepted");
  // A hidden width of a partial convolution group.
  const DraftAttentionShape partialGroup{2056, 512, 6144, 4096, 32, 8, 128};
  rejects([&] { (void)DraftAttention::plan(partialGroup, 1); },
          "unsupported compiled draft attention shape",
          "a draft of a partial convolution group was accepted");
  rejects([&] { (void)DraftAttention::plan({}, 1); }, "unsupported compiled draft attention shape",
          "an empty draft shape was accepted");
}

void fillDyadic(const MetalBuffer &buffer, uint32_t multiplier, uint32_t modulus) {
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
    values[i] = floatToBf16((int((i * multiplier) % modulus) - int(modulus / 2)) / 8.0F);
}

void surroundingPhases(MetalBackend &backend, DraftAttentionShape shape,
                       uint32_t lanes) {
  const auto plan = DraftAttention::plan(shape, lanes);
  const auto workspace = plan.workspace();
  const uint64_t rows = uint64_t{lanes} * kRows;
  auto allocate = [&](uint64_t bytes) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, "draft phases");
  };
  const auto input = allocate(workspace.convolutionBytes);
  const auto dynamic = allocate(rows * shape.dynamicSize * 2);
  const auto weights = allocate(uint64_t{4} * shape.hiddenSize * 2);
  const auto residual = allocate(workspace.convolutionBytes);
  const auto output = allocate(workspace.convolutionBytes);
  fillDyadic(input, 7, 23);
  fillDyadic(dynamic, 5, 17);
  fillDyadic(weights, 3, 13);
  fillDyadic(residual, 11, 19);
  const auto *in = static_cast<const uint16_t *>(input.contents());
  const auto *dyn = static_cast<const uint16_t *>(dynamic.contents());
  const auto *base = static_cast<const uint16_t *>(weights.contents());
  const auto *res = static_cast<const uint16_t *>(residual.contents());
  const uint32_t channelsPerGroup = 16;
  const uint32_t convolutionGroups = shape.hiddenSize / channelsPerGroup;

  Random random(0x5eed1234 + shape.hiddenSize + lanes);
  const auto qkv = randomBfloat(backend, workspace.qkvBytes / 2, random, "QKV");
  const auto queries = allocate(workspace.groupedQueriesBytes);
  const auto queryKeys = allocate(workspace.queryKeysBytes);
  const auto queryValues = allocate(workspace.queryValuesBytes);
  const auto queryNorm = allocate(kHeadDim * 2);
  const auto keyNorm = allocate(kHeadDim * 2);
  std::fill_n(static_cast<uint16_t *>(queryNorm.contents()), kHeadDim,
              floatToBf16(1));
  std::fill_n(static_cast<uint16_t *>(keyNorm.contents()), kHeadDim,
              floatToBf16(1));
  // A lane's RoPE rows are its verify block's, kRopeRows from lane x
  // kRopeRows, of which its draft's rows take the first kRows.
  const uint64_t ropeRows = uint64_t{lanes} * kRopeRows;
  const auto ropeCos = allocate(ropeRows * kHeadDim / 2 * sizeof(float));
  const auto ropeSin = allocate(ropeRows * kHeadDim / 2 * sizeof(float));
  for (uint64_t i = 0; i < ropeRows * kHeadDim / 2; ++i) {
    static_cast<float *>(ropeCos.contents())[i] = std::cos(float(i) * 0.01F);
    static_cast<float *>(ropeSin.contents())[i] = std::sin(float(i) * 0.01F);
  }
  std::vector<uint16_t> originalQkv(
      static_cast<const uint16_t *>(qkv.contents()),
      static_cast<const uint16_t *>(qkv.contents()) + workspace.qkvBytes / 2);
  for (const auto stage : {DraftConvolutionStage::Prepare,
                          DraftConvolutionStage::Residual}) {
    std::memset(output.contents(), 0xFF, output.sizeBytes());
    CommandGraph graph;
    DraftAttention::addConvolution(graph,
        {input, dynamic, weights, residual, output}, plan, stage);
    const bool finish = stage == DraftConvolutionStage::Residual;
    std::array<uint32_t, 2> params{};
    std::memcpy(params.data(), graph.dispatches()[0].bytes[0].data, sizeof(params));
    require(graph.dispatches()[0].threadgroups.x == (kRows * shape.hiddenSize + 255) / 256 &&
                params[0] == (finish ? 1U : 0U) && params[1] == shape.hiddenSize,
            "convolution dispatch does not cover each element once");
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *actual = static_cast<const uint16_t *>(output.contents());
    const uint32_t kind = finish ? 1 : 0;
    for (uint64_t row = 0; row < rows; ++row) {
      for (uint32_t channel = 0; channel < shape.hiddenSize; ++channel) {
        const uint64_t index = row * shape.hiddenSize + channel;
        const uint32_t group = channel / channelsPerGroup;
        float value = bf16ToFloat(in[index]) *
            (bf16ToFloat(base[(kind * 2) * shape.hiddenSize + channel]) +
             bf16ToFloat(dyn[row * shape.dynamicSize +
                             (kind * 2) * convolutionGroups + group]));
        if (row % kRows != 0)
          value += bf16ToFloat(in[index - shape.hiddenSize]) *
              (bf16ToFloat(base[(kind * 2 + 1) * shape.hiddenSize + channel]) +
               bf16ToFloat(dyn[row * shape.dynamicSize +
                               (kind * 2 + 1) * convolutionGroups + group]));
        if (finish)
          value += bf16ToFloat(res[index]);
        require(actual[index] == floatToBf16(value),
                "draft convolution differed from exact CPU arithmetic");
      }
    }
  }

  CommandGraph graph;
  DraftAttention::addPrepare(graph,
      {qkv, queries, queryNorm, keyNorm, ropeCos, ropeSin, queryKeys,
       queryValues}, plan);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  require(std::equal(originalQkv.begin(), originalQkv.end(),
                     static_cast<const uint16_t *>(qkv.contents())),
          "draft prepare wrote its QKV input");
  // Every query and key head RMS-normalized (unit norm weights), rounded to
  // bf16 and rotated, and every value head copied. A rotated value is within
  // an ulp of the fp64 rotation of the bf16-rounded norms plus an ulp of the
  // larger input, which covers fp32 rounding a norm to its other neighbour.
  constexpr uint32_t kPacked = 6144, kPairs = kHeadDim / 2;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < kRows; ++row) {
      const uint64_t laneRow = uint64_t{lane} * kRows + row;
      const uint16_t *packedRow = originalQkv.data() + laneRow * kPacked;
      for (uint32_t head = 0; head < shape.queryHeads + kKvHeads; ++head) {
        const bool query = head < shape.queryHeads;
        const uint32_t h = query ? head : head - shape.queryHeads;
        const uint16_t *source = packedRow + (query ? 0 : kAttention) + h * kHeadDim;
        const auto *prepared = static_cast<const uint16_t *>(
            (query ? queries : queryKeys).contents()) +
            ((uint64_t{lane} * (query ? shape.queryHeads : kKvHeads) + h) * kRows + row) * kHeadDim;
        double squares = 0;
        for (uint32_t d = 0; d < kHeadDim; ++d)
          squares += double(bf16ToFloat(source[d])) * bf16ToFloat(source[d]);
        const double inverse = 1 / std::sqrt(squares / kHeadDim + SPLASH_RMS_EPSILON);
        for (uint32_t d = 0; d < kPairs; ++d) {
          const double first = bf16ToFloat(
              floatToBf16(float(bf16ToFloat(source[d]) * inverse)));
          const double second = bf16ToFloat(
              floatToBf16(float(bf16ToFloat(source[d + kPairs]) * inverse)));
          const uint64_t ropeRow = uint64_t{lane} * kRopeRows + row;
          const double c = static_cast<const float *>(ropeCos.contents())[ropeRow * kPairs + d];
          const double s = static_cast<const float *>(ropeSin.contents())[ropeRow * kPairs + d];
          const double rotated[2] = {first * c - second * s, second * c + first * s};
          for (uint32_t half = 0; half < 2; ++half)
            require(std::fabs(bf16ToFloat(prepared[d + half * kPairs]) - rotated[half]) <=
                        ulpBf16(float(rotated[half])) +
                            ulpBf16(float(std::max(std::fabs(first), std::fabs(second)))),
                    "draft prepare differs from the fp64 norm and rotation");
        }
        if (!query)
          for (uint32_t d = 0; d < kHeadDim; ++d)
            require(static_cast<const uint16_t *>(queryValues.contents())
                            [((uint64_t{lane} * kKvHeads + h) * kHeadDim + d) * kRows + row] ==
                        packedRow[kAttention + kKvHeads * kHeadDim + h * kHeadDim + d],
                    "draft prepare value copy differs");
      }
    }

  // The attention over those prepared rows, every lane's ring empty: it only
  // reads the prepared keys and values, and it replaces the grouped query
  // rows with its own rows. A ring of NaN shows that an empty ring is unread.
  const auto snapshot = [](const MetalBuffer &buffer, uint64_t bytes) {
    const auto *begin = static_cast<const uint16_t *>(buffer.contents());
    return std::vector<uint16_t>(begin, begin + bytes / 2);
  };
  const auto preparedQueries = snapshot(queries, rows * kAttention * 2);
  const auto preparedKeys = snapshot(queryKeys, workspace.queryKeysBytes);
  const auto preparedValues = snapshot(queryValues, workspace.queryValuesBytes);
  const uint64_t ringBytes = uint64_t{kKvHeads} * kWindow * kHeadDim * 2;
  const auto ring = allocate(ringBytes);
  auto *const empty = static_cast<uint16_t *>(ring.contents());
  std::fill_n(empty, ringBytes / 2, uint16_t{0x7FC0});
  std::array<MetalBuffer, kLanes> rings;
  rings.fill(ring);
  const std::array<uint32_t, kLanes> lengths{};
  CommandGraph attention;
  DraftAttention::addDecode(attention, {queries, rings, rings, queryKeys, queryValues},
                            std::span(lengths).first(lanes), plan);
  static_cast<void>(backend.submitCommandAsync(attention.dispatches()).wait());
  require(preparedKeys == snapshot(queryKeys, workspace.queryKeysBytes) &&
              preparedValues == snapshot(queryValues, workspace.queryValuesBytes),
          "draft attention changed its prepared keys or values");
  const auto *attended = static_cast<const uint16_t *>(queries.contents());
  std::vector<float> reference(uint64_t{kGroupRows} * kHeadDim);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      const uint64_t kvHead = uint64_t{lane} * kKvHeads + head;
      referenceRows(preparedQueries.data() + kvHead * kGroupRows * kHeadDim, empty, empty,
                    preparedKeys.data() + kvHead * kRows * kHeadDim,
                    preparedValues.data() + kvHead * kHeadDim * kRows, 0, reference);
      for (uint64_t index = 0; index < reference.size(); ++index) {
        const float actual = bf16ToFloat(attended[uint64_t{lane} * kRows * kAttention +
                                                  attentionOffset(head, index / kHeadDim) + index % kHeadDim]);
        require(std::isfinite(actual) &&
                    std::fabs(actual - reference[index]) <= 0.02F + 0.02F * std::fabs(reference[index]),
                "draft attention over the prepared rows differs from the CPU reference");
      }
    }
}

// The context writers against a CPU ring: a row's key is RMS-normalized,
// scaled by the key norm and rotated, its value copied, into slot
// position % 2048 of each KV head's ring (keys [head][slot][dim], values
// [head][dim][slot]); every other slot keeps its bits. The prefill writes
// its rows from a start position, the commit each lane's retained verify
// rows (at most eight). The buffers hold exactly what the writers read.
void contextWriters(MetalBackend &backend, DraftAttentionShape shape) {
  // A context row holds its keys, then its values.
  constexpr uint32_t kRowWidth = 2048, kKeyColumn = 0, kValueColumn = 1024;
  constexpr uint16_t kUntouched = 0xC2C2;
  const uint64_t ringElements = uint64_t{kKvHeads} * kWindow * kHeadDim;
  Random random(0xc0de0000ULL + shape.hiddenSize);
  const MetalBuffer keyNorm =
      randomBfloat(backend, kHeadDim, random, "draft key norm");
  const auto *norm = static_cast<const uint16_t *>(keyNorm.contents());
  const auto ring = [&] {
    MetalBuffer buffer = backend.allocateBuffer(
        ringElements * 2, BufferStorage::Shared, "draft ring");
    std::fill_n(static_cast<uint16_t *>(buffer.contents()), ringElements,
                kUntouched);
    return buffer;
  };
  const auto table = [&](uint64_t rows, bool sine) {
    MetalBuffer buffer =
        backend.allocateBuffer(rows * kHeadDim / 2 * sizeof(float),
                               BufferStorage::Shared, "draft rope");
    for (uint64_t i = 0; i < rows * kHeadDim / 2; ++i)
      static_cast<float *>(buffer.contents())[i] =
          sine ? std::sin(float(i) * 0.37F) : std::cos(float(i) * 0.37F);
    return buffer;
  };
  // Compares one lane's rings with `rows` rows of `kv` and the RoPE tables
  // written from `start` on.
  const auto check = [&](const MetalBuffer &keys, const MetalBuffer &values,
                         const uint16_t *kv, const float *cosines,
                         const float *sines, uint32_t rows, uint32_t start) {
    std::vector<uint16_t> wantValues(ringElements, kUntouched);
    std::vector<float> wantKeys(ringElements, NAN);
    for (uint32_t row = 0; row < rows; ++row) {
      const uint32_t slot = (start + row) % kWindow;
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        const uint16_t *key =
            kv + uint64_t{row} * kRowWidth + kKeyColumn + head * kHeadDim;
        const uint16_t *value =
            kv + uint64_t{row} * kRowWidth + kValueColumn + head * kHeadDim;
        float square = 0.0F;
        for (uint32_t d = 0; d < kHeadDim; ++d)
          square += bf16ToFloat(key[d]) * bf16ToFloat(key[d]);
        const float inverse =
            1.0F / std::sqrt(square / kHeadDim + float(SPLASH_RMS_EPSILON));
        std::array<float, kHeadDim> normalized;
        for (uint32_t d = 0; d < kHeadDim; ++d)
          normalized[d] = bf16ToFloat(
              floatToBf16(bf16ToFloat(key[d]) * inverse * bf16ToFloat(norm[d])));
        float *out =
            wantKeys.data() + (uint64_t{head} * kWindow + slot) * kHeadDim;
        for (uint32_t d = 0; d < kHeadDim / 2; ++d) {
          const float c = cosines[uint64_t{row} * kHeadDim / 2 + d],
                      s = sines[uint64_t{row} * kHeadDim / 2 + d];
          out[d] = normalized[d] * c - normalized[d + kHeadDim / 2] * s;
          out[d + kHeadDim / 2] =
              normalized[d + kHeadDim / 2] * c + normalized[d] * s;
        }
        for (uint32_t d = 0; d < kHeadDim; ++d)
          wantValues[(uint64_t{head} * kHeadDim + d) * kWindow + slot] =
              value[d];
      }
    }
    const auto *gotKeys = static_cast<const uint16_t *>(keys.contents());
    const auto *gotValues = static_cast<const uint16_t *>(values.contents());
    for (uint64_t i = 0; i < ringElements; ++i) {
      const float expected = wantKeys[i],
                  actual = bf16ToFloat(gotKeys[i]);
      require(gotValues[i] == wantValues[i] &&
                  (std::isnan(expected)
                       ? gotKeys[i] == kUntouched
                       : std::fabs(actual - expected) <=
                             0.02F + 0.02F * std::fabs(expected)),
              "draft context writer diverged from the CPU ring");
    }
  };

  // Prefill: 37 rows from position 6130 (slot 2034), across the ring's end.
  constexpr uint32_t kTokens = 37, kStart = 6130;
  const MetalBuffer kv = randomBfloat(backend, uint64_t{kTokens} * kRowWidth,
                                      random, "draft context kv");
  const MetalBuffer ropeCos = table(kTokens, false),
                    ropeSin = table(kTokens, true);
  const MetalBuffer keys = ring(), values = ring();
  CommandGraph prefill;
  DraftAttention::addContextPrefill(prefill, kv, keyNorm, ropeCos, ropeSin,
                                    keys, values, kTokens, kStart, shape);
  static_cast<void>(backend.submitCommandAsync(prefill.dispatches()).wait());
  check(keys, values, static_cast<const uint16_t *>(kv.contents()),
        static_cast<const float *>(ropeCos.contents()),
        static_cast<const float *>(ropeSin.contents()), kTokens, kStart);

  // Commit: three lanes retaining the first 8, 3 and (clamped) 8 of their
  // verify rows.
  constexpr uint32_t kCommitLanes = 3;
  const std::array<uint32_t, kCommitLanes> starts{2044, 0, 4101};
  const uint32_t retained[kCommitLanes] = {8, 3, 12};
  const uint32_t retainedMasks[kCommitLanes] = {0xFF, 0x7, 0xFFF};
  const MetalBuffer laneKv =
      randomBfloat(backend, uint64_t{kCommitLanes} * kRows * kRowWidth, random,
                   "draft lane kv");
  const MetalBuffer laneCos = table(kCommitLanes * kRopeRows, false),
                    laneSin = table(kCommitLanes * kRopeRows, true);
  MetalBuffer retainedRows = backend.allocateBuffer(
      sizeof(retainedMasks), BufferStorage::Shared, "draft retained rows");
  std::memcpy(retainedRows.contents(), retainedMasks, sizeof(retainedMasks));
  std::array<MetalBuffer, kLanes> laneKeys, laneValues;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    laneKeys[lane] = lane < kCommitLanes ? ring() : laneKeys[0];
    laneValues[lane] = lane < kCommitLanes ? ring() : laneValues[0];
  }
  const std::array<LaneRows, kCommitLanes> blocks{{{0, kRows}, {kRows, kRows}, {2 * kRows, kRows}}};
  const MetalBuffer laneRows = laneRowsBuffer(backend, blocks);
  CommandGraph commit;
  DraftAttention::addContextCommit(commit, laneKv, keyNorm, laneCos, laneSin,
                                   laneKeys, laneValues, retainedRows, laneRows,
                                   starts, stepOf(blocks), shape);
  static_cast<void>(backend.submitCommandAsync(commit.dispatches()).wait());
  for (uint32_t lane = 0; lane < kCommitLanes; ++lane)
    check(laneKeys[lane], laneValues[lane],
          static_cast<const uint16_t *>(laneKv.contents()) +
              uint64_t{lane} * kRows * kRowWidth,
          static_cast<const float *>(laneCos.contents()) +
              uint64_t{lane} * kRopeRows * kHeadDim / 2,
          static_cast<const float *>(laneSin.contents()) +
              uint64_t{lane} * kRopeRows * kHeadDim / 2,
          std::min(retained[lane], kRows), starts[lane]);

  CommandGraph invalid;
  // The start positions name the lanes: none, or more than a batch, commit
  // nothing.
  const std::array<uint32_t, kLanes + 1> overfull{};
  for (const std::span<const uint32_t> positions :
       {std::span<const uint32_t>(), std::span<const uint32_t>(overfull)})
    rejects(
        [&] {
          DraftAttention::addContextCommit(invalid, laneKv, keyNorm, laneCos,
                                           laneSin, laneKeys, laneValues,
                                           retainedRows, laneRows, positions,
                                           stepOf(blocks), shape);
        },
        "invalid draft batch width",
        "a context commit of no lanes or more than a batch was accepted");
  require(invalid.empty(),
          "invalid draft context write partially encoded a graph");
}

// Lanes whose rows of a step hold different row counts, packed one after
// another among the context rows (5, 8, 1 and 6 rows from rows 0, 5, 13 and
// 14), commit their retained rows (the first 4, all 8, the 1, and 9 clamped
// to 6) as each lane does alone from row 0: the same rings, bit for bit. A
// lane's RoPE rows are its verify block's, SPLASH_TARGET_VERIFY_ROWS a lane,
// whichever rows it holds.
void packedContextCommit(MetalBackend &backend, DraftAttentionShape shape) {
  constexpr uint32_t kRowWidth = 2048;
  constexpr uint16_t kUntouched = 0xC2C2;
  const uint64_t ringElements = uint64_t{kKvHeads} * kWindow * kHeadDim;
  Random random(0x7061636bULL + shape.hiddenSize);
  const std::array<LaneRows, kLanes> packed{{{0, 5}, {5, 8}, {13, 1}, {14, 6}, {20, 3}, {23, 2}, {25, 7}, {32, 4}}};
  const std::array<uint32_t, kLanes> starts{2044, 0, 4101, 300, 6140, 17, 1023, 8190},
      retained{0xF, 0xFF, 0x1, 0x1FF, 0x7, 0x3, 0x7F, 0xF};
  const MetalBuffer keyNorm = randomBfloat(backend, kHeadDim, random, "draft key norm");
  const MetalBuffer kv = randomBfloat(backend, uint64_t{36} * kRowWidth, random, "draft packed kv");
  const auto table = [&](bool sine) {
    const uint64_t count = uint64_t{kLanes} * kRopeRows * kHeadDim / 2;
    MetalBuffer result = backend.allocateBuffer(count * sizeof(float), BufferStorage::Shared, "draft rope");
    for (uint64_t i = 0; i < count; ++i)
      static_cast<float *>(result.contents())[i] = sine ? std::sin(float(i) * 0.29F) : std::cos(float(i) * 0.29F);
    return result;
  };
  const MetalBuffer ropeCos = table(false), ropeSin = table(true);
  const auto buffer = [&](uint64_t bytes, const void *from) {
    MetalBuffer result = backend.allocateBuffer(bytes, BufferStorage::Shared, "draft packed commit");
    if (from)
      std::memcpy(result.contents(), from, bytes);
    else
      std::fill_n(static_cast<uint16_t *>(result.contents()), bytes / 2, kUntouched);
    return result;
  };
  const auto rings = [&](uint32_t count) {
    std::array<MetalBuffer, kLanes> result;
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      result[lane] = lane < count ? buffer(ringElements * 2, nullptr) : result[0];
    return result;
  };
  const std::array<MetalBuffer, kLanes> keys = rings(kLanes), values = rings(kLanes);
  CommandGraph commit;
  DraftAttention::addContextCommit(commit, kv, keyNorm, ropeCos, ropeSin, keys, values,
                                   buffer(sizeof(retained), retained.data()), laneRowsBuffer(backend, packed), starts,
                                   stepOf(packed), shape);
  static_cast<void>(backend.submitCommandAsync(commit.dispatches()).wait());
  const uint64_t ropeLaneBytes = uint64_t{kRopeRows} * kHeadDim / 2 * sizeof(float);
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    const LaneRows block = packed[lane];
    const std::array<MetalBuffer, kLanes> laneKeys = rings(1), laneValues = rings(1);
    const std::array<LaneRows, 1> own{{{0, block.rows}}};
    CommandGraph alone;
    DraftAttention::addContextCommit(
        alone,
        buffer(uint64_t{block.rows} * kRowWidth * 2,
               static_cast<const uint16_t *>(kv.contents()) + uint64_t{block.first} * kRowWidth),
        keyNorm, buffer(ropeLaneBytes, static_cast<const uint8_t *>(ropeCos.contents()) + lane * ropeLaneBytes),
        buffer(ropeLaneBytes, static_cast<const uint8_t *>(ropeSin.contents()) + lane * ropeLaneBytes), laneKeys,
        laneValues, buffer(sizeof(uint32_t), &retained[lane]), laneRowsBuffer(backend, own),
        std::span(starts).subspan(lane, 1), stepOf(own), shape);
    static_cast<void>(backend.submitCommandAsync(alone.dispatches()).wait());
    require(std::memcmp(keys[lane].contents(), laneKeys[0].contents(), ringElements * 2) == 0 &&
                std::memcmp(values[lane].contents(), laneValues[0].contents(), ringElements * 2) == 0,
            "packed lane " + std::to_string(lane) + " committed other context rows than alone");
  }
  // Lane 3 committed its six rows, clamped from nine, and lane 0 four of its
  // five: the slot after each lane's last keeps its bits.
  const auto touched = [&](const MetalBuffer &ring, uint32_t position) {
    return static_cast<const uint16_t *>(ring.contents())[uint64_t{position % kWindow} * kHeadDim] != kUntouched;
  };
  require(touched(keys[3], starts[3] + 5) && !touched(keys[3], starts[3] + 6) && touched(keys[0], starts[0] + 3) &&
              !touched(keys[0], starts[0] + 4),
          "packed lanes committed other rows than they retained");

  // A lane that retains a tree's path, rows 0, 2, 5 and 6 of its 8 rows from
  // row 3 of the context rows (the second lane of a step, after a lane of 3
  // rows), commits those rows at its next four positions, the path's i-th
  // with its RoPE row i: the rings of a commit of the four rows alone,
  // gathered from row 0, bit for bit.
  constexpr std::array<uint32_t, 4> path{0, 2, 5, 6};
  const std::array<uint32_t, 2> pathMasks{0b1, 0b1100101};
  const std::array<LaneRows, 2> tree{{{0, 3}, {3, 8}}};
  const std::array<MetalBuffer, kLanes> pathKeys = rings(2), pathValues = rings(2);
  CommandGraph followed;
  DraftAttention::addContextCommit(followed, kv, keyNorm, ropeCos, ropeSin, pathKeys, pathValues,
                                   buffer(sizeof(pathMasks), pathMasks.data()), laneRowsBuffer(backend, tree),
                                   std::span(starts).first(2), stepOf(tree), shape);
  static_cast<void>(backend.submitCommandAsync(followed.dispatches()).wait());
  std::vector<uint16_t> gathered(path.size() * kRowWidth);
  for (uint32_t i = 0; i < path.size(); ++i)
    std::memcpy(gathered.data() + uint64_t{i} * kRowWidth,
                static_cast<const uint16_t *>(kv.contents()) + uint64_t{tree[1].first + path[i]} * kRowWidth,
                kRowWidth * 2);
  const uint32_t prefix = (1u << path.size()) - 1;
  const std::array<MetalBuffer, kLanes> chainKeys = rings(1), chainValues = rings(1);
  const std::array<LaneRows, 1> contiguous{{{0, uint32_t(path.size())}}};
  CommandGraph chain;
  DraftAttention::addContextCommit(
      chain, buffer(gathered.size() * 2, gathered.data()), keyNorm,
      buffer(ropeLaneBytes, static_cast<const uint8_t *>(ropeCos.contents()) + ropeLaneBytes),
      buffer(ropeLaneBytes, static_cast<const uint8_t *>(ropeSin.contents()) + ropeLaneBytes), chainKeys, chainValues,
      buffer(sizeof(prefix), &prefix), laneRowsBuffer(backend, contiguous), std::span(starts).subspan(1, 1),
      stepOf(contiguous), shape);
  static_cast<void>(backend.submitCommandAsync(chain.dispatches()).wait());
  require(std::memcmp(pathKeys[1].contents(), chainKeys[0].contents(), ringElements * 2) == 0 &&
              std::memcmp(pathValues[1].contents(), chainValues[0].contents(), ringElements * 2) == 0 &&
              touched(pathKeys[1], starts[1] + 3) && !touched(pathKeys[1], starts[1] + 4),
          "a retained path committed other context rows than its own, in order");
}

// The context window's reference for one group of 64 values: the codes and
// the fp16 scale and minimum prefill_draft_context_store writes, and the bf16
// values prefill_draft_context_load reads back from them.
struct WindowGroup final {
  std::array<uint8_t, 32> codes{};
  _Float16 scale = 0, minimum = 0;
  std::array<uint16_t, 64> loaded{};
};

WindowGroup referenceWindowGroup(const uint16_t *values) {
  float low = INFINITY, high = -INFINITY;
  for (uint32_t i = 0; i < 64; ++i) {
    low = std::min(low, bf16ToFloat(values[i]));
    high = std::max(high, bf16ToFloat(values[i]));
  }
  WindowGroup result;
  result.scale = _Float16((high - low) / 15.0F);
  result.minimum = _Float16(low);
  const float scale = float(result.scale), minimum = float(result.minimum);
  for (uint32_t i = 0; i < 64; ++i) {
    const float steps = (bf16ToFloat(values[i]) - minimum) / scale;
    const uint32_t code = scale > 0 ? uint32_t(std::clamp(std::nearbyint(steps), 0.0F, 15.0F)) : 0;
    result.codes[i / 2] |= uint8_t(code << (4 * (i % 2)));
    result.loaded[i] = floatToBf16(std::fma(float(code), scale, minimum));
  }
  return result;
}

// Stores 37 rows that wrap from the ring's last slots to its first, and every
// slot of a second window, and loads them back, all rows and a run inside:
// codes, scales and minimums match the reference byte for byte, the loaded
// rows bit for bit, and slots no store reached keep their bytes. The rows
// hold a constant group (zero scale), a wide one and random ones.
void contextWindow(MetalBackend &backend, uint32_t width) {
  constexpr uint8_t kUntouched = 0xA5;
  const uint32_t groups = width / 64;
  const uint64_t windowBytes = DraftAttention::contextWindowBytes(width);
  require(windowBytes == uint64_t{kWindow} * (width / 2 + groups * 4), "context window bytes");
  Random random(0xc0de1000ULL + width);
  for (const auto [rows, start] : {std::pair{37U, 6130U}, std::pair{kWindow, 0U}}) {
    const MetalBuffer input = randomBfloat(backend, uint64_t{rows} * width, random, "draft context rows");
    auto *values = static_cast<uint16_t *>(input.contents());
    std::fill_n(values, 64, floatToBf16(0.75F));
    for (uint32_t i = 0; i < 64; ++i)
      values[uint64_t{width} + 64 + i] = floatToBf16((float(i) - 31.5F) * 9.375F);
    MetalBuffer window = backend.allocateBuffer(windowBytes, BufferStorage::Shared, "draft context window");
    std::memset(window.contents(), kUntouched, windowBytes);
    MetalBuffer loaded = backend.allocateBuffer(uint64_t{rows} * width * 2, BufferStorage::Shared,
                                                "draft context loaded rows");
    constexpr uint32_t kRunBegin = 5, kRunRows = 16;
    MetalBuffer run = backend.allocateBuffer(uint64_t{kRunRows} * width * 2, BufferStorage::Shared,
                                             "draft context loaded run");
    CommandGraph graph;
    DraftAttention::addWindowStore(graph, input, window, rows, start, width);
    DraftAttention::addWindowLoad(graph, window, loaded, rows, start, width);
    DraftAttention::addWindowLoad(graph, window, run, kRunRows, start + kRunBegin, width);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

    const auto *bytes = static_cast<const uint8_t *>(window.contents());
    const auto *scales = reinterpret_cast<const _Float16 *>(bytes + uint64_t{kWindow} * (width / 2));
    const auto *loadedValues = static_cast<const uint16_t *>(loaded.contents());
    const auto *runValues = static_cast<const uint16_t *>(run.contents());
    std::vector<bool> written(kWindow);
    for (uint32_t row = 0; row < rows; ++row) {
      const uint32_t slot = (start + row) % kWindow;
      written[slot] = true;
      for (uint32_t group = 0; group < groups; ++group) {
        const uint64_t offset = uint64_t{row} * width + group * 64;
        const WindowGroup want = referenceWindowGroup(values + offset);
        require(std::equal(want.codes.begin(), want.codes.end(),
                           bytes + uint64_t{slot} * (width / 2) + group * 32),
                "context window codes");
        const _Float16 *scale = scales + (uint64_t{slot} * groups + group) * 2;
        require(std::bit_cast<uint16_t>(scale[0]) == std::bit_cast<uint16_t>(want.scale) &&
                    std::bit_cast<uint16_t>(scale[1]) == std::bit_cast<uint16_t>(want.minimum),
                "context window scale and minimum");
        require(std::equal(want.loaded.begin(), want.loaded.end(), loadedValues + offset),
                "context window loaded rows");
        if (row >= kRunBegin && row < kRunBegin + kRunRows)
          require(std::equal(want.loaded.begin(), want.loaded.end(),
                             runValues + (uint64_t{row - kRunBegin}) * width + group * 64),
                  "context window loaded run");
      }
    }
    for (uint32_t slot = 0; slot < kWindow; ++slot) {
      if (written[slot]) continue;
      const uint8_t *codes = bytes + uint64_t{slot} * (width / 2);
      const auto *groupBytes = reinterpret_cast<const uint8_t *>(scales + uint64_t{slot} * groups * 2);
      require(std::all_of(codes, codes + width / 2, [](uint8_t b) { return b == kUntouched; }) &&
                  std::all_of(groupBytes, groupBytes + groups * 4, [](uint8_t b) { return b == kUntouched; }),
              "a context window store reached a slot outside its rows");
    }
  }
  CommandGraph invalid;
  const MetalBuffer rows = backend.allocateBuffer(uint64_t{kWindow + 1} * width * 2, BufferStorage::Shared, "rows");
  const MetalBuffer window = backend.allocateBuffer(windowBytes, BufferStorage::Shared, "window");
  rejects([&] { DraftAttention::addWindowStore(invalid, rows, window, 0, 0, width); },
          "a context window holds one ring's rows", "an empty context window store was accepted");
  rejects([&] { DraftAttention::addWindowLoad(invalid, window, rows, kWindow + 1, 0, width); },
          "a context window holds one ring's rows", "a load of more rows than a ring was accepted");
  rejects([&] { static_cast<void>(DraftAttention::contextWindowBytes(width + 32)); },
          "draft context rows are whole 64-value groups", "a partial context group was accepted");
  require(invalid.empty(), "an invalid context window transfer encoded a dispatch");
}

// A decode step's window commit of k rows stores the first k rows a lane
// retained of its eight context rows, from row 2 of a step's ten, the path's
// in order, as a store of those rows gathered does, in the slots of their
// positions across the ring's wrap, and leaves every other slot as it was:
// of every chain's first rows, and none, and of a tree's path, rows 0, 2, 5
// and 6, for every k, the whole path (a cycle's commit) and fewer of its rows
// (a capture whose page boundary falls inside the path).
void contextWindowCommit(MetalBackend &backend, uint32_t width) {
  constexpr uint32_t start = kWindow - 4, first = 2, stepRows = first + kRows;
  const uint64_t windowBytes = DraftAttention::contextWindowBytes(width);
  Random random(0xc0de3000ULL + width);
  const MetalBuffer rows = randomBfloat(backend, uint64_t{stepRows} * width, random, "draft context rows");
  const MetalBuffer retained = backend.allocateBuffer(4, BufferStorage::Shared, "draft retained rows");
  const LaneRows lane{first, kRows};
  MetalBuffer laneRows = backend.allocateBuffer(sizeof(LaneRows), BufferStorage::Shared, "draft lane rows");
  std::memcpy(laneRows.contents(), &lane, sizeof(lane));
  const auto allocate = [&](uint64_t bytes, const char *label) {
    MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
    std::memset(buffer.contents(), 0xA5, bytes);
    return buffer;
  };
  std::vector<uint32_t> paths;
  for (uint32_t count = 0; count <= kRows; ++count)
    paths.push_back((1U << count) - 1);
  paths.push_back(0b1100101);
  const MetalBuffer committed = allocate(windowBytes, "committed window"),
                    stored = allocate(windowBytes, "stored window");
  for (const uint32_t path : paths) {
    *static_cast<uint32_t *>(retained.contents()) = path;
    // The path's rows, gathered in order.
    const uint32_t count = static_cast<uint32_t>(std::popcount(path));
    const MetalBuffer gathered = allocate(std::max<uint64_t>(count, 1) * width * 2, "gathered rows");
    uint32_t at = 0;
    for (uint32_t row = 0; row < kRows; ++row)
      if (path >> row & 1)
        std::memcpy(static_cast<uint16_t *>(gathered.contents()) + uint64_t{at++} * width,
                    static_cast<const uint16_t *>(rows.contents()) + uint64_t{first + row} * width, width * 2);
    for (uint32_t k = 1; k <= kRows; ++k) {
      std::memset(committed.contents(), 0xA5, windowBytes);
      std::memset(stored.contents(), 0xA5, windowBytes);
      const uint32_t kept = std::min(k, count);
      CommandGraph graph;
      DraftAttention::addWindowCommit(graph, rows, committed, retained, laneRows, stepRows, k, start, width);
      if (kept)
        DraftAttention::addWindowStore(graph, gathered, stored, kept, start, width);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      require(std::memcmp(committed.contents(), stored.contents(), windowBytes) == 0,
              "a window commit of the first " + std::to_string(k) + " of retained rows " + std::to_string(path) +
                  " is not their store");
    }
  }
}

// A state captured at s inside a command over [a, b) starts from the lane's
// window as the command finds it, copied, and stores the command's rows
// [a, s) into it as the lane's window takes them: the slots of its positions
// [s - 2048, s) are, byte for byte, those one store of those rows leaves,
// while the lane's window, which goes on to rows [s, b), has lost the oldest
// once b passes 2048. Before the window fills and once the ring wraps.
void contextWindowCapturedInside(MetalBackend &backend, uint32_t width) {
  const uint64_t windowBytes = DraftAttention::contextWindowBytes(width);
  const uint64_t rowBytes = uint64_t{width} * 2;
  const uint64_t codeBytes = width / 2, groupBytes = width / 64 * 4;
  Random random(0xc0de2000ULL + width);
  for (const auto [a, s, b] : {std::array{1000U, 1500U, 2900U}, std::array{5000U, 5500U, 6900U}}) {
    const MetalBuffer rows = randomBfloat(backend, uint64_t{b} * width, random, "draft context rows");
    const auto store = [&](CommandGraph &graph, uint32_t begin, uint32_t end, const MetalBuffer &window) {
      DraftAttention::addWindowStore(graph, backend.view(rows, begin * rowBytes, (end - begin) * rowBytes),
                                     window, end - begin, begin, width);
    };
    const auto allocate = [&](const char *label) {
      MetalBuffer window = backend.allocateBuffer(windowBytes, BufferStorage::Shared, label);
      std::memset(window.contents(), 0xA5, windowBytes);
      return window;
    };
    const MetalBuffer lane = allocate("lane window"), captured = allocate("captured window"), whole = allocate("whole window");
    CommandGraph earlier;
    store(earlier, a - std::min(a, kWindow), a, lane);
    static_cast<void>(backend.submitCommandAsync(earlier.dispatches()).wait());
    std::memcpy(captured.contents(), lane.contents(), windowBytes);
    const std::vector<uint8_t> before(static_cast<const uint8_t *>(lane.contents()),
                                      static_cast<const uint8_t *>(lane.contents()) + windowBytes);
    CommandGraph command, reference;
    store(command, a, b, lane);
    store(command, a, s, captured);
    store(reference, s - std::min(s, kWindow), s, whole);
    static_cast<void>(backend.submitCommandAsync(command.dispatches()).wait());
    static_cast<void>(backend.submitCommandAsync(reference.dispatches()).wait());
    const auto slot = [&](const void *window, uint64_t position, bool codes) {
      const uint64_t index = position % kWindow;
      return static_cast<const uint8_t *>(window) +
             (codes ? index * codeBytes : kWindow * codeBytes + index * groupBytes);
    };
    uint64_t lost = 0;
    for (uint64_t position = s - std::min(s, kWindow); position < s; ++position)
      for (const bool codes : {true, false}) {
        const uint64_t bytes = codes ? codeBytes : groupBytes;
        require(std::memcmp(slot(captured.contents(), position, codes), slot(whole.contents(), position, codes),
                            bytes) == 0,
                "a captured window's slot is not the one a store of its rows leaves");
        if (position + kWindow < b) {
          require(std::memcmp(slot(captured.contents(), position, codes), slot(before.data(), position, codes),
                              bytes) == 0,
                  "a captured window's overwritten slot is not the lane's before the command");
          lost += std::memcmp(slot(lane.contents(), position, codes), slot(whole.contents(), position, codes),
                              bytes) != 0;
        } else {
          require(std::memcmp(slot(captured.contents(), position, codes), slot(lane.contents(), position, codes),
                              bytes) == 0,
                  "a captured window's slot is not the lane's after the command");
        }
      }
    require(lost != 0, "the lane's window kept the rows its later rows overwrite");
  }
}

// Each buffer the draft phases reach, at its extent and one element short,
// for three lanes of eight rows: the convolution's rows, the lanes' dynamic
// weights and the taps' base weights; the prepare's q|k|v rows, its grouped
// query rows, the current rows' keys and values, the norms and the RoPE rows;
// the attention's query rows, followed by the split partials, and each lane's
// rings; the context writers' key and value rows, norm, RoPE rows, rings and
// the commit's retained rows; the context windows' rows, windows and a window
// commit's retained rows. A lane's RoPE rows are its verify block's, of which
// its draft takes the first kRows.
void bufferExtents(MetalBackend &backend, DraftAttentionShape shape) {
  constexpr uint32_t lanes = 3, contextTokens = 37;
  const auto plan = DraftAttention::plan(shape, lanes);
  const uint64_t rows = uint64_t{lanes} * kRows, hidden = rows * shape.hiddenSize * 2;
  const uint64_t queryRows = rows * kAttention * 2, current = rows * kKvHeads * kHeadDim * 2;
  const uint64_t ring = uint64_t{kKvHeads} * kWindow * kHeadDim * 2, norm = kHeadDim * 2;
  const uint64_t ropeRow = kHeadDim / 2 * 4, contextRow = uint64_t{shape.qkvSize - shape.attentionSize} * 2;
  using Extents = std::initializer_list<splash::test::BufferExtent>;
  using Buffers = std::vector<MetalBuffer>;
  splash::test::requireExtents(backend,
                               Extents{{0, hidden, 2, "draft convolution input"},
                                       {1, rows * shape.dynamicSize * 2, 2, "draft dynamic convolution"},
                                       {2, uint64_t{4} * shape.hiddenSize * 2, 2, "draft convolution weight"},
                                       {3, hidden, 2, "draft convolution residual"},
                                       {4, hidden, 2, "draft convolution output"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 DraftAttention::addConvolution(graph, {b[0], b[1], b[2], b[3], b[4]}, plan,
                                                                DraftConvolutionStage::Residual);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, rows * shape.qkvSize * 2, 2, "draft q/k/v"},
                                       {1, queryRows, 2, "draft grouped queries"},
                                       {2, norm, 2, "draft query norm"},
                                       {3, norm, 2, "draft key norm"},
                                       {4, (uint64_t{lanes - 1} * kRopeRows + kRows) * ropeRow, 4,
                                        "draft RoPE cosine"},
                                       {5, (uint64_t{lanes - 1} * kRopeRows + kRows) * ropeRow, 4,
                                        "draft RoPE sine"},
                                       {6, current, 2, "draft query keys"},
                                       {7, current, 2, "draft query values"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 DraftAttention::addPrepare(graph, {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]},
                                                            plan);
                               });
  // Every lane's rings follow the query rows and current keys and values; a
  // batch binds a ring for each of its lanes, those past the plan's unread.
  const auto rings = [](const Buffers &b, size_t first) {
    std::array<MetalBuffer, kLanes> result;
    for (uint32_t lane = 0; lane < kLanes; ++lane) result[lane] = b[first + std::min(lane, lanes - 1)];
    return result;
  };
  std::vector<splash::test::BufferExtent> extents{
      {0, queryRows + uint64_t{lanes} * kKvHeads * kSplits * kPartialBytes, 4, "draft grouped queries"},
      {1, current, 2, "draft query keys"},
      {2, current, 2, "draft query values"}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    extents.push_back({3 + lane, ring, 2, "draft key ring"});
    extents.push_back({3 + lanes + lane, ring, 2, "draft value ring"});
  }
  const std::array<uint32_t, lanes> lengths{0, 2047, 6000};
  splash::test::requireExtents(backend, extents, [&](CommandGraph &graph, const Buffers &b) {
    DraftAttention::addDecode(graph, {b[0], rings(b, 3), rings(b, 3 + lanes), b[1], b[2]}, lengths, plan);
  });
  splash::test::requireExtents(backend,
                               Extents{{0, contextTokens * contextRow, 2, "draft context K/V"},
                                       {1, norm, 2, "draft key norm"},
                                       {2, contextTokens * ropeRow, 4, "draft RoPE cosine"},
                                       {3, contextTokens * ropeRow, 4, "draft RoPE sine"},
                                       {4, ring, 2, "draft key ring"},
                                       {5, ring, 2, "draft value ring"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 DraftAttention::addContextPrefill(graph, b[0], b[1], b[2], b[3], b[4], b[5],
                                                                   contextTokens, 6130, shape);
                               });
  extents = {{0, rows * contextRow, 2, "draft context K/V"},
             {1, norm, 2, "draft key norm"},
             {2, (uint64_t{lanes - 1} * kRopeRows + kRows) * ropeRow, 4, "draft RoPE cosine"},
             {3, (uint64_t{lanes - 1} * kRopeRows + kRows) * ropeRow, 4, "draft RoPE sine"},
             {4, uint64_t{lanes} * 4, 4, "draft retained rows"},
             {5, uint64_t{lanes} * sizeof(LaneRows), sizeof(LaneRows), "draft lane rows"}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    extents.push_back({6 + lane, ring, 2, "draft key ring"});
    extents.push_back({6 + lanes + lane, ring, 2, "draft value ring"});
  }
  const std::array<uint32_t, lanes> starts{2044, 0, 4101};
  splash::test::requireExtents(backend, extents, [&](CommandGraph &graph, const Buffers &b) {
    DraftAttention::addContextCommit(graph, b[0], b[1], b[2], b[3], rings(b, 6), rings(b, 6 + lanes), b[4], b[5],
                                     starts, {lanes, lanes * kRows, kRows}, shape);
  });
  const uint64_t windowBytes = DraftAttention::contextWindowBytes(shape.hiddenSize);
  const uint64_t contextRows = uint64_t{contextTokens} * shape.hiddenSize * 2;
  splash::test::requireExtents(backend,
                               Extents{{0, contextRows, 2, "draft context rows"},
                                       {1, windowBytes, 1, "draft context window"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 DraftAttention::addWindowStore(graph, b[0], b[1], contextTokens, 2040,
                                                                shape.hiddenSize);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, windowBytes, 1, "draft context window"},
                                       {1, contextRows, 2, "draft context rows"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 DraftAttention::addWindowLoad(graph, b[0], b[1], contextTokens, 2040,
                                                               shape.hiddenSize);
                               });
  splash::test::requireExtents(backend,
                               Extents{{0, uint64_t{kRows} * shape.hiddenSize * 2, 2, "draft context rows"},
                                       {1, windowBytes, 1, "draft context window"},
                                       {2, 4, 4, "draft retained rows"},
                                       {3, sizeof(LaneRows), sizeof(LaneRows), "draft lane rows"}},
                               [&](CommandGraph &graph, const Buffers &b) {
                                 DraftAttention::addWindowCommit(graph, b[0], b[1], b[2], b[3], kRows, kRows, 2044,
                                                                 shape.hiddenSize);
                               });
}

void dsparkFusion(MetalBackend &backend) {
  constexpr uint32_t width = 2304, rows = 2;
  Random random(0x5eed);
  MetalBuffer captured = randomBfloat(backend, rows * 4 * width, random, "DSpark taps");
  MetalBuffer fuse = randomBfloat(backend, 16, random, "DSpark fuse");
  MetalBuffer norm = randomBfloat(backend, width, random, "DSpark norm");
  MetalBuffer output = backend.allocateBuffer(rows * width * 2, BufferStorage::Shared, "DSpark context");
  const auto *taps = static_cast<const uint16_t *>(captured.contents());
  const auto *weights = static_cast<const uint16_t *>(fuse.contents());
  const auto *scale = static_cast<const uint16_t *>(norm.contents());
  for (uint32_t consumer = 0; consumer < 4; ++consumer) {
    CommandGraph graph;
    DraftAttention::addDSparkFusion(graph, captured, fuse, norm, output, rows, consumer);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    const auto *actual = static_cast<const uint16_t *>(output.contents());
    float alpha[4], mass = 0.0F, maxWeight = -INFINITY;
    for (uint32_t tap = 0; tap < 4; ++tap)
      maxWeight = std::max(maxWeight, bf16ToFloat(weights[consumer * 4 + tap]));
    for (uint32_t tap = 0; tap < 4; ++tap) {
      alpha[tap] = std::exp(bf16ToFloat(weights[consumer * 4 + tap]) - maxWeight);
      mass += alpha[tap];
    }
    for (float &a : alpha) a = bf16ToFloat(floatToBf16(a / mass));
    for (uint32_t row = 0; row < rows; ++row) {
      std::vector<float> mixed(width);
      float squares = 0.0F;
      for (uint32_t col = 0; col < width; ++col) {
        for (uint32_t tap = 0; tap < 4; ++tap)
          mixed[col] += bf16ToFloat(taps[(row * 4 + tap) * width + col]) * alpha[tap];
        squares += mixed[col] * mixed[col];
      }
      const float inv = 1.0F / std::sqrt(squares / width + 1e-6F);
      for (uint32_t col = 0; col < width; ++col) {
        const float expected = mixed[col] * inv * bf16ToFloat(scale[col]);
        require(std::fabs(bf16ToFloat(actual[row * width + col]) - expected) < 0.02F,
                "DSpark weighted context and norm");
      }
    }
  }
}

void dsparkAttention(MetalBackend &backend) {
  constexpr uint32_t dim = 128, kv = 8 * dim, q = 24 * dim, contextRows = 3;
  const auto buffer = [&](uint64_t count, const char *name) {
    return backend.allocateBuffer(count * 2, BufferStorage::Shared, name);
  };
  MetalBuffer query = buffer(8 * q, "DSpark query");
  MetalBuffer contextKey = buffer(contextRows * kv, "DSpark context key");
  MetalBuffer contextValue = buffer(contextRows * kv, "DSpark context value");
  MetalBuffer blockKey = buffer(8 * kv, "DSpark block key");
  MetalBuffer blockValue = buffer(8 * kv, "DSpark block value");
  MetalBuffer result = buffer(8 * q, "DSpark attention result");
  std::memset(query.contents(), 0, query.sizeBytes());
  std::memset(contextKey.contents(), 0, contextKey.sizeBytes());
  std::memset(blockKey.contents(), 0, blockKey.sizeBytes());
  auto *cv = static_cast<uint16_t *>(contextValue.contents());
  auto *bv = static_cast<uint16_t *>(blockValue.contents());
  for (uint32_t row = 0; row < contextRows; ++row)
    for (uint32_t head = 0; head < 8; ++head)
      for (uint32_t d = 0; d < dim; ++d)
        cv[row * kv + head * dim + d] = floatToBf16(float(row + head));
  for (uint32_t row = 0; row < 8; ++row)
    for (uint32_t head = 0; head < 8; ++head)
      for (uint32_t d = 0; d < dim; ++d)
        bv[row * kv + head * dim + d] = floatToBf16(float(row + head));
  CommandGraph graph;
  const MetalBuffer workspace = backend.allocateBuffer(DraftAttention::dsparkAttentionWorkspaceBytes(),
                                                       BufferStorage::Shared, "DSpark attention workspace");
  DraftAttention::addDSparkAttention(graph, query, contextKey, contextValue, blockKey, blockValue,
                                    result, workspace, contextRows, 2, 0);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto *values = static_cast<const uint16_t *>(result.contents());
  for (uint32_t row = 0; row < 8; ++row)
    for (uint32_t head = 0; head < 24; ++head)
      for (uint32_t d = 0; d < dim; ++d) {
        const float expected = (30.0F + 9 * float(head / 3)) / 9;
        require(std::fabs(bf16ToFloat(values[row * q + head * dim + d]) - expected) < 0.04F,
                "DSpark block is bidirectional and context is windowed");
      }
}

// Random queries, keys and values over a context of several splits, against a CPU softmax: each output element
// of each query row and head is its softmax-weighted value sum over the context's rows from `first` and the block's
// own 8 rows.
void dsparkAttentionSplits(MetalBackend &backend) {
  constexpr uint32_t dim = 128, kv = 8 * dim, q = 24 * dim, contextRows = 700, first = 37;
  Random random(0xd5a7);
  MetalBuffer query = randomBfloat(backend, 8 * q, random, "DSpark query");
  MetalBuffer contextKey = randomBfloat(backend, contextRows * kv, random, "DSpark context key");
  MetalBuffer contextValue = randomBfloat(backend, contextRows * kv, random, "DSpark context value");
  MetalBuffer blockKey = randomBfloat(backend, 8 * kv, random, "DSpark block key");
  MetalBuffer blockValue = randomBfloat(backend, 8 * kv, random, "DSpark block value");
  MetalBuffer result = backend.allocateBuffer(8 * q * 2, BufferStorage::Shared, "DSpark attention result");
  const MetalBuffer workspace = backend.allocateBuffer(DraftAttention::dsparkAttentionWorkspaceBytes(),
                                                       BufferStorage::Shared, "DSpark attention workspace");
  CommandGraph graph;
  DraftAttention::addDSparkAttention(graph, query, contextKey, contextValue, blockKey, blockValue, result, workspace,
                                    contextRows, first, 0);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto value = [](const MetalBuffer &b, uint64_t i) {
    return double(bf16ToFloat(static_cast<const uint16_t *>(b.contents())[i]));
  };
  double worst = 0;
  for (uint32_t row = 0; row < 8; ++row)
    for (uint32_t head = 0; head < 24; ++head) {
      const uint32_t group = head / 3;
      std::vector<double> scores;
      const auto keyAt = [&](uint32_t key, uint32_t d) {
        return key < contextRows - first ? value(contextKey, uint64_t{first + key} * kv + group * dim + d)
                                         : value(blockKey, uint64_t{key - (contextRows - first)} * kv + group * dim + d);
      };
      const auto valueAt = [&](uint32_t key, uint32_t d) {
        return key < contextRows - first
                   ? value(contextValue, uint64_t{first + key} * kv + group * dim + d)
                   : value(blockValue, uint64_t{key - (contextRows - first)} * kv + group * dim + d);
      };
      const uint32_t keys = contextRows - first + 8;
      double maximum = -1e300;
      for (uint32_t key = 0; key < keys; ++key) {
        double dot = 0;
        for (uint32_t d = 0; d < dim; ++d) dot += value(query, uint64_t{row} * q + head * dim + d) * keyAt(key, d);
        scores.push_back(dot / std::sqrt(double(dim)));
        maximum = std::max(maximum, scores.back());
      }
      double total = 0;
      for (double &s : scores) total += (s = std::exp(s - maximum));
      for (uint32_t d = 0; d < dim; ++d) {
        double expected = 0;
        for (uint32_t key = 0; key < keys; ++key) expected += scores[key] / total * valueAt(key, d);
        worst = std::max(worst, std::fabs(value(result, uint64_t{row} * q + head * dim + d) - expected));
      }
    }
  require(worst < 0.01, "DSpark split attention matches a CPU softmax over several splits");
}

void dsparkHeadPrepare(MetalBackend &backend) {
  constexpr uint32_t heads = 8, dim = 128;
  MetalBuffer input = backend.allocateBuffer(heads * dim * 2, BufferStorage::Shared, "DSpark head input");
  MetalBuffer norm = backend.allocateBuffer(dim * 2, BufferStorage::Shared, "DSpark head norm");
  MetalBuffer output = backend.allocateBuffer(heads * dim * 2, BufferStorage::Shared, "DSpark head output");
  std::fill_n(static_cast<uint16_t *>(input.contents()), heads * dim, floatToBf16(1.0F));
  std::fill_n(static_cast<uint16_t *>(norm.contents()), dim, floatToBf16(1.0F));
  CommandGraph graph;
  DraftAttention::addDSparkHeadPrepare(graph, input, norm, output, 1, heads, 2);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto *values = static_cast<const uint16_t *>(output.contents());
  for (uint32_t h = 0; h < heads; ++h) {
    require(std::fabs(bf16ToFloat(values[h * dim]) - (std::cos(2.0F) - std::sin(2.0F))) < 0.01F,
            "DSpark NeoX first half");
    require(std::fabs(bf16ToFloat(values[h * dim + 64]) - (std::sin(2.0F) + std::cos(2.0F))) < 0.01F,
            "DSpark NeoX second half");
  }
}

void dsparkContextRing(MetalBackend &backend) {
  constexpr uint32_t width = 1024, rows = 5, capacity = 3;
  const auto alloc = [&](uint32_t count) {
    return backend.allocateBuffer(count * width * 2, BufferStorage::Shared, "DSpark ring");
  };
  MetalBuffer keys = alloc(rows), values = alloc(rows), ringKeys = alloc(capacity), ringValues = alloc(capacity);
  auto *k = static_cast<uint16_t *>(keys.contents());
  auto *v = static_cast<uint16_t *>(values.contents());
  for (uint32_t row = 0; row < rows; ++row) {
    std::fill_n(k + row * width, width, floatToBf16(float(row)));
    std::fill_n(v + row * width, width, floatToBf16(float(row + 10)));
  }
  CommandGraph graph;
  DraftAttention::addDSparkContextStore(graph, keys, values, ringKeys, ringValues, rows, 7, 0, capacity);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto *rk = static_cast<const uint16_t *>(ringKeys.contents());
  const auto *rv = static_cast<const uint16_t *>(ringValues.contents());
  for (uint32_t row = 2; row < rows; ++row) {
    const uint32_t slot = (7 + row) % capacity;
    for (uint32_t col = 0; col < width; ++col)
      require(rk[slot * width + col] == k[row * width + col] &&
              rv[slot * width + col] == v[row * width + col], "DSpark context ring wrap");
  }
}

void dsparkCommit(MetalBackend &backend) {
  constexpr uint32_t width = 9216, rows = 6, cap = 5, kv = 1024, slots = 8;
  const auto alloc = [&](uint64_t bytes) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, "DSpark commit");
  };
  MetalBuffer captured = alloc(rows * width * 2), compact = alloc((cap + 1) * width * 2);
  MetalBuffer retained = alloc(4), lanes = alloc(sizeof(LaneRows));
  MetalBuffer keys = alloc(cap * kv * 2), values = alloc(cap * kv * 2);
  MetalBuffer ringKeys = alloc(slots * kv * 2), ringValues = alloc(slots * kv * 2);
  auto *source = static_cast<uint16_t *>(captured.contents());
  for (uint32_t i = 0; i < rows; ++i) std::fill_n(source + i * width, width, floatToBf16(float(i)));
  *static_cast<uint32_t *>(retained.contents()) = 0b10101;
  *static_cast<LaneRows *>(lanes.contents()) = {1, cap};
  auto *k = static_cast<uint16_t *>(keys.contents());
  auto *v = static_cast<uint16_t *>(values.contents());
  for (uint32_t i = 0; i < cap; ++i) {
    std::fill_n(k + i * kv, kv, floatToBf16(float(i)));
    std::fill_n(v + i * kv, kv, floatToBf16(float(i + 10)));
  }
  std::memset(ringKeys.contents(), 0, ringKeys.sizeBytes());
  std::memset(ringValues.contents(), 0, ringValues.sizeBytes());
  CommandGraph graph;
  DraftAttention::addDSparkCommitGather(graph, captured, retained, lanes, compact, rows, cap);
  DraftAttention::addDSparkCommitStore(graph, keys, values, retained, ringKeys, ringValues, cap, 7, 0, slots);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const auto *out = static_cast<const uint16_t *>(compact.contents());
  for (uint32_t i = 0; i <= cap; ++i) {
    const uint32_t expected = i < 3 ? 1 + 2 * i : 5;
    for (uint32_t col = 0; col < width; ++col)
      require(out[i * width + col] == floatToBf16(float(expected)), "DSpark accepted capture order");
  }
  const auto *stored = static_cast<const uint16_t *>(ringKeys.contents());
  for (uint32_t i = 0; i < 3; ++i)
    require(stored[((7 + i) % slots) * kv] == floatToBf16(float(i)), "DSpark accepted KV commit");
  require(stored[2 * kv] == 0, "DSpark rejected KV is not committed");
}

// A segmented global layer of `rows` rows in segments of `segmentTokens`, the
// last shorter, which the tests keep small so that attention, prefill stores
// and commits cross segment boundaries (production's segments hold
// SPLASH_DRAFT_GLOBAL_SEGMENT_TOKENS rows).
struct SegmentedFixture final {
  std::vector<MetalBuffer> keys, values;
  DraftSegmentedContext context;
  uint32_t segmentTokens, rows;

  SegmentedFixture(MetalBackend &backend, uint32_t tokens, uint32_t total) : segmentTokens(tokens), rows(total) {
    for (uint32_t first = 0; first < rows; first += segmentTokens) {
      const uint64_t bytes = uint64_t{std::min(segmentTokens, rows - first)} * 1024 * 2;
      keys.push_back(backend.allocateBuffer(bytes, BufferStorage::Shared, "DSpark key segment"));
      values.push_back(backend.allocateBuffer(bytes, BufferStorage::Shared, "DSpark value segment"));
      std::memset(keys.back().contents(), 0, bytes);
      std::memset(values.back().contents(), 0, bytes);
    }
    context = DraftSegmentedContext::make(keys, values, segmentTokens, rows);
  }
  // The 1024 values of global row `at` of the keys, or the values.
  uint16_t *row(bool value, uint32_t at) const {
    const MetalBuffer &segment = (value ? values : keys)[at / segmentTokens];
    return static_cast<uint16_t *>(segment.contents()) + uint64_t{at % segmentTokens} * 1024;
  }
};

void dsparkSegmentedContext(MetalBackend &backend) {
  const MetalBuffer workspace = backend.allocateBuffer(DraftAttention::dsparkAttentionWorkspaceBytes(),
                                                       BufferStorage::Shared, "DSpark attention workspace");
  constexpr uint32_t kv = 1024, q = 3072, rows = 8, segmentTokens = 3;
  require(DraftSegmentedContext::segmentCount(rows, segmentTokens) == 3 &&
              DraftSegmentedContext::segmentCount(6, 3) == 2 && DraftSegmentedContext::segmentCount(1, 3) == 1,
          "DSpark segment count");

  // Attention over segments is bitwise the attention over one buffer holding
  // the same rows: a window of rows 1 to 7 reads segments 0, 1 and 2.
  {
    Random random(0x5e6);
    SegmentedFixture fixture(backend, segmentTokens, rows);
    MetalBuffer flatKeys = randomBfloat(backend, uint64_t{rows} * kv, random, "DSpark flat keys");
    MetalBuffer flatValues = randomBfloat(backend, uint64_t{rows} * kv, random, "DSpark flat values");
    for (uint32_t row = 0; row < rows; ++row) {
      std::memcpy(fixture.row(false, row), static_cast<uint16_t *>(flatKeys.contents()) + row * kv, kv * 2);
      std::memcpy(fixture.row(true, row), static_cast<uint16_t *>(flatValues.contents()) + row * kv, kv * 2);
    }
    MetalBuffer query = randomBfloat(backend, 8 * q, random, "DSpark query");
    MetalBuffer blockKey = randomBfloat(backend, 8 * kv, random, "DSpark block key");
    MetalBuffer blockValue = randomBfloat(backend, 8 * kv, random, "DSpark block value");
    for (const auto [contextRows, first] : {std::pair{7u, 1u}, {8u, 0u}, {2u, 2u}, {0u, 0u}, {4u, 0u}}) {
      MetalBuffer expected = backend.allocateBuffer(8 * q * 2, BufferStorage::Shared, "DSpark flat attention");
      MetalBuffer actual = backend.allocateBuffer(8 * q * 2, BufferStorage::Shared, "DSpark segmented attention");
      CommandGraph graph;
      DraftAttention::addDSparkAttention(graph, query, flatKeys, flatValues, blockKey, blockValue, expected,
                                         workspace, contextRows, first, 0);
      DraftAttention::addDSparkSegmentedAttention(graph, query, fixture.context, blockKey, blockValue, actual,
                                                  workspace, contextRows, first);
      (void)backend.submitCommandAsync(graph.dispatches()).wait();
      require(std::memcmp(expected.contents(), actual.contents(), 8 * q * 2) == 0,
              "DSpark segmented attention matches one buffer across segments");
    }
  }

  // A prefill store of five rows at positions 4 to 8 of a global layer based
  // at 2 lands in rows 2 to 6: segment 0's last row, all of segment 1, and
  // segment 2's first, and touches nothing else.
  {
    Random random(0x57);
    SegmentedFixture fixture(backend, segmentTokens, rows);
    constexpr uint32_t stored = 5, start = 4, base = 2;
    MetalBuffer keys = randomBfloat(backend, uint64_t{stored} * kv, random, "DSpark stored keys");
    MetalBuffer values = randomBfloat(backend, uint64_t{stored} * kv, random, "DSpark stored values");
    CommandGraph graph;
    DraftAttention::addDSparkSegmentedContextStore(graph, keys, values, fixture.context, stored, start, base);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    for (uint32_t row = 0; row < rows; ++row) {
      const bool written = row >= start - base && row < start - base + stored;
      for (uint32_t col = 0; col < kv; ++col) {
        const uint16_t key = written ? static_cast<uint16_t *>(keys.contents())[(row - (start - base)) * kv + col] : 0;
        const uint16_t value =
            written ? static_cast<uint16_t *>(values.contents())[(row - (start - base)) * kv + col] : 0;
        require(fixture.row(false, row)[col] == key && fixture.row(true, row)[col] == value,
                "DSpark segmented prefill store crosses segments");
      }
    }
    CommandGraph invalid;
    rejects([&] { DraftAttention::addDSparkSegmentedContextStore(invalid, keys, values, fixture.context, stored,
                                                                 start + 2, base); },
            "invalid DSpark segmented context store", "a store past the last segment was accepted");
    rejects([&] { DraftAttention::addDSparkSegmentedContextStore(invalid, keys, values, fixture.context, stored,
                                                                 base - 1, base); },
            "invalid DSpark segmented context store", "a store before the global base was accepted");
    require(invalid.empty(), "a rejected segmented store encoded a dispatch");
  }

  // An accepted-row commit writes only the rows its retained mask counts:
  // three of five, at global rows 2, 3 and 4, from segment 0 into segment 1.
  {
    Random random(0xc0);
    SegmentedFixture fixture(backend, segmentTokens, rows);
    constexpr uint32_t cap = 5, start = 2;
    MetalBuffer keys = randomBfloat(backend, uint64_t{cap} * kv, random, "DSpark commit keys");
    MetalBuffer values = randomBfloat(backend, uint64_t{cap} * kv, random, "DSpark commit values");
    MetalBuffer retained = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, "DSpark retained");
    *static_cast<uint32_t *>(retained.contents()) = 0b10101;
    CommandGraph graph;
    DraftAttention::addDSparkSegmentedCommitStore(graph, keys, values, retained, fixture.context, cap, start, 0);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    for (uint32_t row = 0; row < rows; ++row) {
      const bool written = row >= start && row < start + 3;
      for (uint32_t col = 0; col < kv; ++col) {
        const uint16_t key = written ? static_cast<uint16_t *>(keys.contents())[(row - start) * kv + col] : 0;
        const uint16_t value = written ? static_cast<uint16_t *>(values.contents())[(row - start) * kv + col] : 0;
        require(fixture.row(false, row)[col] == key && fixture.row(true, row)[col] == value,
                "DSpark segmented commit crosses segments and skips rejected rows");
      }
    }
    CommandGraph invalid;
    rejects([&] { DraftAttention::addDSparkSegmentedCommitStore(invalid, keys, values, retained, fixture.context,
                                                                cap, rows - cap + 1, 0); },
            "invalid DSpark segmented context commit", "a commit past the last segment was accepted");
  }

  // Segments that do not hold the rows, or hold too many to address, are
  // refused before any dispatch.
  {
    SegmentedFixture fixture(backend, segmentTokens, rows);
    rejects([&] { (void)DraftSegmentedContext::make({fixture.keys.data(), 2}, {fixture.values.data(), 2},
                                                    segmentTokens, rows); },
            "segmented context holds", "too few segments for the rows were accepted");
    rejects([&] { (void)DraftSegmentedContext::make(fixture.keys, fixture.values, segmentTokens, rows + 1); },
            "buffer holds", "a short last segment was accepted");
    rejects([&] { (void)DraftSegmentedContext::make(fixture.keys, fixture.values, segmentTokens, 0); },
            "invalid DSpark segmented context", "an empty segmented context was accepted");
    rejects([&] { (void)DraftSegmentedContext::make(fixture.keys, {fixture.values.data(), 2}, segmentTokens, rows); },
            "segments differ", "unpaired key and value segments were accepted");
  }
}

// What a rewind and a restore leave of DSpark's global layer. Rows are stored
// by position and read only up to the lane's committed length, so a rewind
// needs no clearing: the rows past the rewound length are stale and never
// read, and the rows a later cycle commits overwrite them. A restore sets the
// global base, the position of row 0, and what the layer held of another
// request, or of the same one before, is out of the attention's reach.
void dsparkSegmentedRewindAndRestore(MetalBackend &backend) {
  const MetalBuffer workspace = backend.allocateBuffer(DraftAttention::dsparkAttentionWorkspaceBytes(),
                                                       BufferStorage::Shared, "DSpark attention workspace");
  constexpr uint32_t kv = 1024, q = 3072, rows = 12, segmentTokens = 5;
  Random random(0x4e77);
  MetalBuffer query = randomBfloat(backend, 8 * q, random, "DSpark query");
  MetalBuffer blockKey = randomBfloat(backend, 8 * kv, random, "DSpark block key");
  MetalBuffer blockValue = randomBfloat(backend, 8 * kv, random, "DSpark block value");
  using Source = std::tuple<const MetalBuffer *, uint32_t, uint32_t>;
  // One buffer of `count` rows of each source from its `first`, in order.
  const auto join = [&](std::initializer_list<Source> sources) {
    uint32_t total = 0;
    for (const auto &[buffer, first, count] : sources) total += count;
    MetalBuffer flat = backend.allocateBuffer(uint64_t{total} * kv * 2, BufferStorage::Shared, "DSpark joined rows");
    uint32_t at = 0;
    for (const auto &[buffer, first, count] : sources) {
      std::memcpy(static_cast<uint16_t *>(flat.contents()) + uint64_t{at} * kv,
                  static_cast<const uint16_t *>(buffer->contents()) + uint64_t{first} * kv, uint64_t{count} * kv * 2);
      at += count;
    }
    return flat;
  };
  // True where attention over the first `contextRows` rows of the segments is
  // bitwise attention over those rows held in one buffer.
  const auto matches = [&](const SegmentedFixture &fixture, const MetalBuffer &flatKeys,
                           const MetalBuffer &flatValues, uint32_t contextRows) {
    MetalBuffer expected = backend.allocateBuffer(8 * q * 2, BufferStorage::Shared, "DSpark flat attention");
    MetalBuffer actual = backend.allocateBuffer(8 * q * 2, BufferStorage::Shared, "DSpark segmented attention");
    CommandGraph graph;
    DraftAttention::addDSparkAttention(graph, query, flatKeys, flatValues, blockKey, blockValue, expected,
                                       workspace, contextRows, 0, 0);
    DraftAttention::addDSparkSegmentedAttention(graph, query, fixture.context, blockKey, blockValue, actual,
                                                workspace, contextRows, 0);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    return std::memcmp(expected.contents(), actual.contents(), 8 * q * 2) == 0;
  };
  const auto store = [&](const SegmentedFixture &fixture, const MetalBuffer &keys, const MetalBuffer &values,
                         uint32_t count, uint32_t start, uint32_t base) {
    CommandGraph graph;
    DraftAttention::addDSparkSegmentedContextStore(graph, keys, values, fixture.context, count, start, base);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
  };
  const auto commit = [&](const SegmentedFixture &fixture, const MetalBuffer &keys, const MetalBuffer &values,
                          uint32_t retainedMask, uint32_t cap, uint32_t start, uint32_t base) {
    MetalBuffer retained = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, "DSpark retained");
    *static_cast<uint32_t *>(retained.contents()) = retainedMask;
    CommandGraph graph;
    DraftAttention::addDSparkSegmentedCommitStore(graph, keys, values, retained, fixture.context, cap, start, base);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
  };
  const auto poison = [&](const SegmentedFixture &fixture, uint32_t at) {
    std::memset(fixture.row(false, at), 0xFF, kv * 2);
    std::memset(fixture.row(true, at), 0xFF, kv * 2);
  };

  // Rewind: ten rows are committed, the lane rewinds to position 7, and the
  // next cycle commits two rows of four there.
  {
    SegmentedFixture fixture(backend, segmentTokens, rows);
    MetalBuffer keys = randomBfloat(backend, uint64_t{10} * kv, random, "DSpark first keys");
    MetalBuffer values = randomBfloat(backend, uint64_t{10} * kv, random, "DSpark first values");
    store(fixture, keys, values, 10, 0, 0);
    require(matches(fixture, keys, values, 10), "DSpark segmented rows before a rewind match one buffer");
    // Rewound to 7: rows 7 to 9 are stale, and attention over 7 rows reads none of them.
    poison(fixture, 7);
    poison(fixture, 8);
    poison(fixture, 9);
    require(matches(fixture, keys, values, 7), "a rewound DSpark global layer read a row past its length");
    MetalBuffer newKeys = randomBfloat(backend, uint64_t{4} * kv, random, "DSpark cycle keys");
    MetalBuffer newValues = randomBfloat(backend, uint64_t{4} * kv, random, "DSpark cycle values");
    commit(fixture, newKeys, newValues, 0b11, 4, 7, 0);
    const MetalBuffer joinedKeys = join({{&keys, 0, 7}, {&newKeys, 0, 2}});
    const MetalBuffer joinedValues = join({{&values, 0, 7}, {&newValues, 0, 2}});
    require(matches(fixture, joinedKeys, joinedValues, 9),
            "a DSpark cycle after a rewind did not attend to its committed rows");
    require(static_cast<uint16_t *>(fixture.row(false, 9))[0] == 0xFFFF &&
                static_cast<uint16_t *>(fixture.row(true, 9))[kv - 1] == 0xFFFF,
            "a DSpark commit wrote a row it did not retain");
  }

  // Restore: a window of eight rows, positions 6 to 13, makes the base 6; a
  // cycle then commits four rows at 14, up to the layer's last row.
  {
    SegmentedFixture fixture(backend, segmentTokens, rows);
    MetalBuffer keys = randomBfloat(backend, uint64_t{8} * kv, random, "DSpark window keys");
    MetalBuffer values = randomBfloat(backend, uint64_t{8} * kv, random, "DSpark window values");
    store(fixture, keys, values, 8, 6, 6);
    require(matches(fixture, keys, values, 8), "a restored DSpark window is not rows 0 to 7 of the global layer");
    MetalBuffer newKeys = randomBfloat(backend, uint64_t{4} * kv, random, "DSpark cycle keys");
    MetalBuffer newValues = randomBfloat(backend, uint64_t{4} * kv, random, "DSpark cycle values");
    commit(fixture, newKeys, newValues, 0b1111, 4, 14, 6);
    require(matches(fixture, join({{&keys, 0, 8}, {&newKeys, 0, 4}}), join({{&values, 0, 8}, {&newValues, 0, 4}}), 12),
            "a DSpark cycle after a restore did not attend to the window and its commit");
    CommandGraph invalid;
    MetalBuffer retained = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, "DSpark retained");
    *static_cast<uint32_t *>(retained.contents()) = 1;
    MetalBuffer extraKeys = randomBfloat(backend, uint64_t{5} * kv, random, "DSpark extra keys");
    MetalBuffer extraValues = randomBfloat(backend, uint64_t{5} * kv, random, "DSpark extra values");
    rejects([&] { DraftAttention::addDSparkSegmentedCommitStore(invalid, extraKeys, extraValues, retained,
                                                                fixture.context, 5, 14, 6); },
            "invalid DSpark segmented context commit", "a commit past the restored layer's capacity was accepted");
    require(invalid.empty(), "a rejected commit encoded a dispatch");

    // The lane serves another request restored at base 20: its window of
    // eight rows replaces rows 0 to 7, and the first request's rows 8 to 11
    // left in the layer are never read.
    MetalBuffer otherKeys = randomBfloat(backend, uint64_t{8} * kv, random, "DSpark other keys");
    MetalBuffer otherValues = randomBfloat(backend, uint64_t{8} * kv, random, "DSpark other values");
    store(fixture, otherKeys, otherValues, 8, 20, 20);
    poison(fixture, 8);
    poison(fixture, 9);
    poison(fixture, 10);
    poison(fixture, 11);
    require(matches(fixture, otherKeys, otherValues, 8),
            "a lane's next restore read rows its earlier request left in the global layer");
  }
}


} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-attention METALLIB");
    planGeometry();
    MetalBackend backend(argv[1]);
    dsparkFusion(backend);
    dsparkAttention(backend);
    dsparkAttentionSplits(backend);
    dsparkHeadPrepare(backend);
    dsparkContextRing(backend);
    dsparkCommit(backend);
    dsparkSegmentedContext(backend);
    dsparkSegmentedRewindAndRestore(backend);
    for (const auto shape : kShapes) {
      bufferExtents(backend, shape);
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        surroundingPhases(backend, shape, lanes);
      contextWriters(backend, shape);
      packedContextCommit(backend, shape);
      contextWindow(backend, shape.hiddenSize);
      contextWindowCapturedInside(backend, shape.hiddenSize);
      contextWindowCommit(backend, shape.hiddenSize);
      runCase(backend, 1, shape, {0, 0, 0, 0});
      runCase(backend, 2, shape, {2048, 2047, 0, 0});
      runCase(backend, 3, shape, {2100, 4094, 6143, 0});
      runCase(backend, 4, shape, {262137, 4094, 500, 6143});
      runCase(backend, 4, shape, {512, 513, 1024, 1536});
      runCase(backend, 4, shape, {2046, 2049, 4095, 4096});
      runCase(backend, 2, shape, {8191, 262144, 0, 0});
      runCase(backend, kLanes, shape, {262137, 4094, 500, 6143, 512, 2049, 0, 8191});
    }
    std::cout << "draft_attention_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_attention_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
