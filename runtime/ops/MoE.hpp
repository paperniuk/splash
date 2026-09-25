#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace splash::ops {

// The shared expert is an expert every row visits, so it has the routed
// experts' intermediate width and runs through the same grouped tiles.
struct MoeShape final {
  uint32_t hiddenSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return hiddenSize && hiddenSize % 256 == 0 && experts && experts <= 256 &&
           expertsPerToken && expertsPerToken <= experts &&
           expertIntermediateSize && expertIntermediateSize % 256 == 0;
  }
  // Routed experts followed by the shared expert.
  [[nodiscard]] constexpr uint32_t routesPerToken() const noexcept {
    return expertsPerToken + 1;
  }
};

// All weights for one sparse MoE block. The model package owns the buffers;
// this value only exposes semantic projections to the operator. The shared
// expert is a one-expert slab.
struct MoeWeights final {
  Q8Projection router;
  ExpertQ4Projection expertGate;
  ExpertQ4Projection expertUp;
  ExpertQ4Projection expertDown;
  ExpertQ4Projection sharedGate;
  ExpertQ4Projection sharedUp;
  ExpertQ4Projection sharedDown;
  Q8Projection sharedExpertGate;
};

// Grouped-row scratch. Routes are sorted by expert into tiles of tileRows
// rows; every routed expert may leave one partially filled tile and no tile
// is empty, and the shared expert fills one tile per tileRows rows. A split
// prefill plan also parks the gate projection in expertOutput before the
// down pass overwrites it, so that field spans the wider of the two widths.
inline constexpr uint32_t kMoeDecodeTileRows = 8;
inline constexpr uint32_t kMoePrefillTileRows = 32;
// Router score tiles: 8 x 32 for short chunks, 32 x 128 for longer chunks.
// The measured Apple10 crossover is about 26 rows per GPU core, with a
// 20-core fallback when the core count is unknown. Both tiles preserve scores.
struct MoeRouteTile final {
  uint32_t rows;
  uint32_t experts;
};
inline constexpr uint32_t kMoeRouteWideRows = 512;
inline constexpr uint32_t kMoeRouteRowsPerCore = 26;

[[nodiscard]] constexpr uint32_t moeRouteWideRows(uint32_t gpuCores) noexcept {
  return gpuCores ? gpuCores * kMoeRouteRowsPerCore : kMoeRouteWideRows;
}

[[nodiscard]] constexpr MoeRouteTile
moeRouteTile(uint32_t rows, uint32_t wideRows = kMoeRouteWideRows) noexcept {
  return rows >= wideRows ? MoeRouteTile{32, 128} : MoeRouteTile{8, 32};
}

[[nodiscard]] constexpr uint32_t moeMaximumTiles(uint32_t rows, MoeShape shape,
                                                 uint32_t tileRows) noexcept {
  const uint32_t routed = rows * shape.expertsPerToken;
  return std::min(routed / tileRows + shape.experts, routed) +
         (rows + tileRows - 1) / tileRows;
}

struct MoeWorkspace final {
  uint64_t selectedExpertsBytes = 0;
  uint64_t routingWeightsBytes = 0;
  uint64_t tileDescriptorsBytes = 0;
  uint64_t tileCountBytes = 0;
  uint64_t groupedRoutesBytes = 0;
  uint64_t routeRowsBytes = 0;
  uint64_t groupedInputBytes = 0;
  uint64_t expertIntermediateBytes = 0;
  uint64_t expertOutputBytes = 0;
};

// Both configurations consume affine Q4 expert slabs in StorageN=256 order.
// The tile applies to grouping, gather and both expert projections together;
// changing it never changes the physical rows in a command. M8 plans and
// decode plans run the fused gate/up tile; the M32 prefill plan runs the
// experts as three N256 passes (gate, up with the silu gate, down) whose
// tiles shrink to the descriptor's live rows, bit-identical to the fused
// tile.
enum class MoeExpertTile : uint8_t { M8 = 8, M32 = 32 };

// Simdgroups per 8-row expert tile: a device policy the execution plans set,
// not a tuned choice. Eight is the shipped N128 tile for both projections.
// Four halves the threadgroup to 128 threads and runs gate/up at N128 and
// down at N256, for Apple9 decode plans: family 9 has no per-core matrix
// unit, and a decode expert grid leaves it latency-bound at low occupancy.
// Measured on a 40-core Apple9 GPU at the 35B shape (H=2048, E=256,
// top_k=8, I=512), ms per layer at rows 8/16/24/32: gate/up
// 0.332/0.551/0.728/0.859 -> 0.314/0.503/0.618/0.699 (1.06x-1.23x), down
// 0.157/0.274/0.363/0.419 -> 0.137/0.226/0.297/0.335 (1.15x-1.25x). Apple10
// variants had mixed results across shapes, so family 10
// keeps eight. Smaller Apple9 core counts still need performance validation;
// this family gate does not establish their optimum.
// The 32-row tiles always run eight simdgroups. Either choice
// writes bit-identical outputs and needs the same workspace; only the down
// pass's column grid changes.
enum class MoeExpertSimdgroups : uint8_t { Eight = 8, Four = 4 };

// Families below 9 are rejected at startup; 10 and later keep the shipped
// tile, as does an unknown family.
[[nodiscard]] constexpr MoeExpertSimdgroups
moeDecodeSimdgroups(uint32_t appleGpuFamily) noexcept {
  return appleGpuFamily == 9 ? MoeExpertSimdgroups::Four
                             : MoeExpertSimdgroups::Eight;
}

// Expert tile implementation: a device policy the execution plans set, not a
// tuned choice. Mpp is the shipped tensor-operation tile. Register is the
// Apple7/8 tile (moe_mma.metal: exact half Q4 weights, fp32 inputs and
// accumulation, in-kernel input sums), whose gate/up pass is fused at either
// row count, so its prefill plans never split the experts. Its outputs match
// the MPP tiles up to summation order, not bitwise.
enum class MoeExpertKernel : uint8_t { Mpp = 0, Register = 1 };

struct MoeConfig final {
  MoeExpertTile expertTile = MoeExpertTile::M32;
  // Rows from which the router uses the 32-row scores tile; the execution
  // plans derive it from the GPU core count.
  uint32_t routeWideRows = kMoeRouteWideRows;
  // Simdgroups of the 8-row expert tiles; the execution plans derive it from
  // the GPU family for decode plans and keep eight for prefill plans.
  MoeExpertSimdgroups m8Simdgroups = MoeExpertSimdgroups::Eight;
  MoeExpertKernel kernel = MoeExpertKernel::Mpp;
  bool operator==(const MoeConfig &) const = default;
};

// Constructed by the operator so workspace sizing and encoding use the same plan.
class MoePlan final {
public:
  [[nodiscard]] MoeShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] MoeConfig config() const noexcept { return config_; }
  [[nodiscard]] uint32_t tileRows() const noexcept {
    return static_cast<uint32_t>(config_.expertTile);
  }
  [[nodiscard]] bool splitExperts() const noexcept { return splitExperts_; }
  [[nodiscard]] uint32_t maximumTiles() const noexcept { return maximumTiles_; }
  [[nodiscard]] const MoeWorkspace &workspace() const noexcept {
    return workspace_;
  }

private:
  friend struct MoE;
  MoePlan(MoeShape shape, uint32_t rows, MoeConfig config, bool prefill);

  MoeShape shape_;
  uint32_t rows_;
  MoeConfig config_;
  bool splitExperts_;
  uint32_t maximumTiles_;
  MoeWorkspace workspace_;
};

struct MoeBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer residual;
  metal::MetalBuffer output;
  // rows * routesPerToken() routes: expert ids and routing weights.
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  // Sized by moeMaximumTiles(): tile descriptors, one tile count, the route
  // at each grouped row, each route's grouped row, and the grouped rows'
  // inputs, intermediates and outputs. The router parks its bf16 scores in
  // groupedInput until the gather claims it.
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
};

// Routes and executes grouped experts from immutable weight views.
struct MoE final {
  [[nodiscard]] static MoePlan prefillPlan(
      MoeShape shape, uint32_t rows,
      MoeConfig config = {MoeExpertTile::M32});
  [[nodiscard]] static MoePlan decodePlan(
      MoeShape shape, uint32_t lanes,
      MoeConfig config = {MoeExpertTile::M8});
  // Bounded precompiled candidates, shipped baseline first. ExecutionPlans
  // supplies the device's router threshold to every expert-tile candidate
  // and its 8-row tile simdgroups to the decode candidates.
  [[nodiscard]] static std::array<MoePlan, 2>
  prefillCandidates(MoeShape shape, uint32_t rows, uint32_t routeWideRows,
                    MoeExpertKernel kernel = MoeExpertKernel::Mpp);
  [[nodiscard]] static std::array<MoePlan, 2>
  decodeCandidates(MoeShape shape, uint32_t lanes, uint32_t routeWideRows,
                   MoeExpertSimdgroups m8Simdgroups = MoeExpertSimdgroups::Eight,
                   MoeExpertKernel kernel = MoeExpertKernel::Mpp);
  static void add(metal::CommandGraph &graph, const MoeBuffers &buffers,
                  const MoeWeights &weights, const MoePlan &plan);
};

} // namespace splash::ops
