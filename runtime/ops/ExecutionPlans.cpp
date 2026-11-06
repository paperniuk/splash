#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = SPLASH_TARGET_VERIFY_ROWS;

template <typename Workspace, size_t N>
void include(Workspace &bound, const Workspace &required,
             const std::array<uint64_t Workspace::*, N> &fields,
             uint32_t lanes = 1) {
  for (auto field : fields) {
    const uint64_t bytes = required.*field;
    bound.*field = std::max(bound.*field,
                           bytes / lanes + uint64_t{bytes % lanes != 0});
  }
}

} // namespace

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device)
    : linear_(device),
      attentionTile_(linear_.calibration().simdgroupMma ? AttentionTile::Register : AttentionTile::Mpp) {}

bool ExecutionPlans::hyperNormStaged(uint32_t rows) const noexcept {
  // Apple7/8 only, and only while the rows' threadgroups have the cores to
  // themselves: past that the staged kernel runs a second wave where the
  // quarters kernel doesn't slow down. Its 96 norms of a Flash-Next Q2_0 step
  // on an M2 Max (38 cores), quarters -> staged ms: 16 rows 2.73 -> 2.10,
  // 32 rows 2.80 -> 2.27, 33 rows 2.82 -> 2.24, 34 rows 2.80 -> 3.01, 64 rows
  // 2.90 -> 3.13. Three quarters of the cores keeps a margin below that edge.
  return calibration().simdgroupMma && rows * 4 <= calibration().cores * 3;
}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t historyTokens, uint32_t queryHeads, kv::Layout layout,
    AttentionVariant variant) const {
  return PagedAttention::prefillPlan(rows, historyTokens, queryHeads, layout,
                                     calibration().attention.prefillPagesPerSplit, variant, attentionTile_,
                                     calibration().cores);
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens, AttentionVariant variant) const {
  return PagedAttention::verifyPlan(lanes, queryHeads, layout, historyTokens, variant, attentionTile_);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  return DraftAttention::plan(shape, lanes);
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, const MoeFormats &formats, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, selectMoe(calibration(), shape, formats, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, const MoeFormats &formats, uint32_t lanes) const {
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, lanes,
                         selectMoe(calibration(), shape, formats, lanes * kDecodeRows, MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// The verify bound is linear in the lanes: one lane's is every width's share.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::verifyWorkspace(1, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  return DraftAttention::plan(shape, 1).workspace();
}

MoeWorkspace ExecutionPlans::moePrefillWorkspace(MoeShape shape, const MoeFormats &formats,
                                                 uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  auto bound = moePrefill(shape, formats, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows)
    include(bound, moePrefill(shape, formats, rows).workspace(), kMoeWorkspaceFields);
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape, const MoeFormats &formats) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, moeDecode(shape, formats, lanes).workspace(), kMoeWorkspaceFields, lanes);
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops
