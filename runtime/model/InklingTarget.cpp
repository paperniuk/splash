#include "model/InklingTarget.hpp"

#include "model/Inkling.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace splash::model {
namespace {

using Layout = InklingLayout;
constexpr uint32_t kVisits = 48;
constexpr uint32_t kRestarts[] = {Layout::coreBegin + (Layout::coreEnd - Layout::coreBegin),
                                  Layout::coreBegin + 2 * (Layout::coreEnd - Layout::coreBegin)};
constexpr uint32_t kCoreVisitsEnd = Layout::coreBegin + (Layout::coreEnd - Layout::coreBegin) * Layout::passes;

// A pass after the first restarts the core from the prelude's stream: its
// first visit of the core's first layer.
constexpr bool restarts(uint32_t visit) noexcept {
  return std::ranges::find(kRestarts, visit) != std::end(kRestarts);
}
// The core's later passes read messages of the pass before.
constexpr bool messaging(uint32_t visit) noexcept { return visit >= kRestarts[0] && visit < kCoreVisitsEnd; }
constexpr bool boundary(uint32_t layer) noexcept { return layer % Layout::blockPeriod == 0; }

// The block states a visit's depth attention reads before its own block,
// slots 0 on: the prelude's block alone once a pass restarts.
constexpr uint32_t blocksAt(uint32_t visit) noexcept {
  uint32_t count = 0;
  for (uint32_t at = 0; at <= visit; ++at) {
    if (restarts(at)) count = 1;
    if (at < visit && boundary(Layout::visitLayer(at))) ++count;
  }
  return count;
}
static_assert(blocksAt(0) == 0 && blocksAt(1) == 1 && blocksAt(7) == 2 && blocksAt(13) == 3 && blocksAt(18) == 1 &&
              blocksAt(19) == 2 && blocksAt(42) == 3 && blocksAt(43) == 4);

ops::LayoutRows promptRows(std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows) {
  uint32_t longest = 0;
  for (const QwenTargetPrefillSequence &sequence : sequences) longest = std::max(longest, sequence.rows);
  return {static_cast<uint32_t>(sequences.size()), rows, longest};
}

} // namespace

InklingTarget::InklingTarget(const InklingWeights &weights, const QwenTargetGeometry &geometry,
                             metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                             uint32_t prefillChunks)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights),
      steps_(geometry_, backend, prefillChunks) {
  if (weights.layers.size() != Layout::storedLayers || geometry_.layers != kVisits)
    throw std::invalid_argument("Inkling weights do not match their execution geometry");
}

QwenTargetScratchBytes InklingTarget::allocatedScratchBytes() const noexcept { return steps_.allocatedBytes(); }

std::vector<metal::MetalBuffer> InklingTarget::addPrefill(metal::CommandGraph &graph,
                                                          std::span<const QwenTargetPrefillChunk> chunks,
                                                          const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                          QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  if (aneFfn) throw std::invalid_argument("Inkling's experts do not run on the Neural Engine");
  std::vector<PrefillStep> steps = prefillSteps<PrefillStep>(graph, chunks, kv, spilled, layers, nullptr);
  for (uint32_t index = 0; index < steps.size(); ++index) {
    PrefillStep &step = steps[index];
    if (step.spilled.active()) throw std::invalid_argument("Inkling attends no spilled K/V");
    step.scratch = step.decode ? steps_.verifyScratch() : steps_.prefillScratch(index);
    step.capacity = step.decode ? ExecutionLimits::maximumBatchWidth * ExecutionLimits::targetVerifyRows
                                : ExecutionLimits::prefillTokenBudget;
  }
  return addPrefillLayers(std::span(steps), *this, layers);
}

void InklingTarget::addPrefillBegin(PrefillStep &step) const {
  ops::Normalization::addRms(step.graph, step.buffers.hidden[0], weights_.embeddingNorm, step.scratch.x,
                             geometry_.hiddenSize, step.rows);
}

const ops::BlockMoeWeights *InklingTarget::moe(uint32_t visit) const {
  return &weights_.layers[Layout::visitLayer(visit)].moe;
}

void InklingTarget::addConvolution(PrefillStep &step, uint32_t visit, uint32_t offset,
                                   const ops::InklingConvolution &convolution, ops::InklingRows input,
                                   ops::InklingRows dynamicTaps, metal::MetalBuffer output,
                                   const ops::InklingResidual &residual) const {
  const uint32_t channels = convolution.channels, rows = step.rows;
  const uint32_t historyOffset = visit * InklingSteps::kVisitWidth + offset;
  std::vector<metal::MetalBuffer> historyIn, historyOut;
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    historyIn.push_back(sequence.auxiliaryIn);
    if (!step.decode) historyOut.push_back(sequence.auxiliaryOut);
  }
  const ops::InklingRows in = input, out{std::move(output), channels, channels, 0};
  // A residual's prompt rows go to the history captures below from the input alone.
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    ops::Inkling::addConvolution(step.graph, in, historyIn, {}, historyOffset, InklingSteps::kHistoryWidth,
                                 convolution, dynamicTaps, out, step.buffers.laneRows,
                                 {d->step.lanes, d->step.rows, d->step.laneCap}, d->tree,
                                 {step.scratch.tape, convolution.channels, InklingSteps::kHistoryWidth, historyOffset},
                                 residual);
    return;
  }
  ops::Inkling::addConvolution(step.graph, in, historyIn, historyOut, historyOffset, InklingSteps::kHistoryWidth,
                               convolution, dynamicTaps, out, step.buffers.laneRows,
                               promptRows(step.sequences, rows), {}, {}, residual);
  for (const QwenTargetPrefillSequence &sequence : step.sequences)
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      ops::Inkling::addConvolutionHistory(
          step.graph,
          {rowsOf<uint16_t>(input.buffer, sequence.rowBegin, capture.row, input.stride), channels, input.stride,
           input.offset},
          sequence.auxiliaryIn, capture.auxiliary, historyOffset, InklingSteps::kHistoryWidth, channels,
          convolution.taps, capture.row);
}

void InklingTarget::addPrefillLayer(PrefillStep &step, uint32_t visit) const {
  const uint32_t layer = Layout::visitLayer(visit);
  const InklingLayerWeights &w = weights_.layers[layer];
  const InklingScratch &h = step.scratch;
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t rows = step.rows, d = geometry_.hiddenSize, cap = step.capacity;
  const bool sliding = Layout::isSlidingLayer(layer), messages = messaging(visit);
  const uint32_t blocks = blocksAt(visit);
  const auto slot = [&](uint32_t index) { return ops::InklingRows{h.blocks, d, 4 * d, index * d}; };
  const ops::InklingRows x{h.x, d, d, 0};

  // A later pass's messages are its pass before's: the core's middle block
  // and its output, read before the stream restarts from the prelude's,
  // which the core's first block holds.
  if (restarts(visit)) {
    ops::RowCopy::add(step.graph, h.blocks, {0, 4 * d, 2 * d}, h.bank, {0, 2 * d, 0}, rows, d);
    ops::RowCopy::add(step.graph, h.x, {0, d, 0}, h.bank, {0, 2 * d, d}, rows, d);
    ops::RowCopy::add(step.graph, h.blocks, {0, 4 * d, d}, h.x, {0, d, 0}, rows, d);
  }
  if (messages) {
    const std::array<ops::InklingRows, 1> bank{ops::InklingRows{h.bank, d, 2 * d, 0}};
    const uint64_t queryBytes = uint64_t{4} * d * sizeof(float);
    ops::Inkling::addDepthMix(step.graph, bank, {h.bank, d, 2 * d, d}, false,
                              backend_.view(weights_.messageQueries, (layer - Layout::coreBegin) * queryBytes,
                                            queryBytes),
                              4, {h.messages, 4 * d, 4 * d, 0}, h.depthSums, d, rows);
  }

  // The attention's inputs: a sliding layer's one mix, a full layer's three,
  // each through its message gate in a later pass, then its norm.
  const uint32_t queries = sliding ? 1 : 3;
  if (!blocks) {
    ops::Inkling::addSum(step.graph, x, false, {}, false, {h.mixed, d, 3 * d, 0}, true, d, rows);
  } else {
    std::vector<ops::InklingRows> states;
    for (uint32_t index = 0; index < blocks; ++index) states.push_back(slot(index));
    ops::Inkling::addDepthMix(step.graph, states, x, false, w.attentionInputQuery, queries,
                              {h.mixed, 3 * d, 3 * d, 0}, h.depthSums, d, rows);
  }
  // The parts' gates in one pair of dispatches and their norms in one: part
  // p's mix (the one mix of a sliding layer), message and outputs p steps on.
  const uint32_t parts = sliding && !messages ? 1 : 3;
  ops::InklingRows source{h.mixed, d, 3 * d, 0};
  uint32_t sourceStep = queries == 1 ? 0 : d;
  if (messages) {
    std::array<ops::InklingGateWeights, 3> gates;
    for (uint32_t part = 0; part < parts; ++part) {
      const InklingControllerWeights &c = w.controllers[part];
      gates[part] = {c.down, c.forget, c.gain};
    }
    const ops::InklingRows gated{h.parts, d, 3 * d, 0};
    ops::Inkling::addMessageGates(step.graph, source, {h.messages, d, 4 * d, 0}, std::span(gates).first(parts), 64,
                                  {h.gateLow, 64, 64, 0}, gated,
                                  {.x = sourceStep, .message = d, .low = 64 * cap, .output = d}, d, rows);
    source = gated;
    sourceStep = d;
  }
  const std::array<ops::NormWeights, 3> norms =
      sliding ? std::array{w.inputNorm, w.inputNorm, w.inputNorm} : std::array{w.queryNorm, w.keyNorm, w.valueNorm};
  ops::Inkling::addNorms(step.graph, source, std::span(norms).first(parts), {h.normed, d, d, 0},
                         {.x = sourceStep, .output = cap * d}, d, rows);
  const auto normed = [&](uint32_t part) {
    // Every row of the part's region: a projection's tile may hold more rows
    // than the step's (LinearPlan::storageRows).
    return rowsOf<uint16_t>(h.normed, (parts == 1 ? 0 : part) * cap, cap, d);
  };

  // Projections. A full layer's convolution taps add each row's own, bf16
  // as its projection rounds them.
  // A sliding layer's k and v of its one input are columns of its packed rows,
  // one projection with q, r and gate; of inputs of their own (parts), they
  // project apart.
  const bool fused = sliding && parts == 1;
  linear.addProjection(step.graph, normed(0), sliding && !fused ? w.queryRelativeGateAlone : w.queryRelativeGate,
                       h.packed, rows, b.linearScratch);
  ops::addGgufFloat(step.graph, normed(0), w.sink, h.sinks, rows, geometry_.attentionQueryHeads, 0,
                    ops::FloatOutput::BFloat16,
                    linear.ggufFloatTile(rows, geometry_.attentionQueryHeads, w.sink.formatId));
  if (!fused) {
    linear.addProjection(step.graph, normed(1), w.key, h.key, rows, b.linearScratch);
    linear.addProjection(step.graph, normed(2), w.value, h.value, rows, b.linearScratch);
  }
  const ops::InklingRows keys = fused ? ops::InklingRows{h.packed, 1024, INKLING_PACKED_WIDTH, INKLING_KEY_OFFSET}
                                      : ops::InklingRows{h.key, 1024, 1024, 0};
  const ops::InklingRows values = fused ? ops::InklingRows{h.packed, 1024, INKLING_PACKED_WIDTH, INKLING_VALUE_OFFSET}
                                        : ops::InklingRows{h.value, 1024, 1024, 0};
  const uint32_t kv = geometry_.attentionKvHeads * geometry_.attentionHeadDimension;
  const uint32_t taps = geometry_.attentionKvHeads * Layout::convolutionTaps;
  ops::InklingRows keyTaps{}, valueTaps{};
  if (!sliding) {
    const auto tapsOf = [&](metal::MetalBuffer input, const InklingConvolutionWeights &conv, metal::MetalBuffer out) {
      ops::addGgufFloat(step.graph, std::move(input), conv.weightProjection, h.rawTaps, rows, taps, 0,
                        ops::FloatOutput::BFloat16, linear.ggufFloatTile(rows, taps, conv.weightProjection.formatId));
      ops::Inkling::addSum(step.graph, {h.rawTaps, taps, taps, 0}, false, {}, false, {out, taps, taps, 0}, true, taps,
                           rows);
      return ops::InklingRows{std::move(out), taps, taps, 0};
    };
    keyTaps = tapsOf(normed(1), w.keyConvolution, h.keyTaps);
    valueTaps = tapsOf(normed(2), w.valueConvolution, h.valueTaps);
  }
  const auto convolution = [&](const metal::MetalBuffer &weights, uint32_t channels, bool dynamic) {
    return ops::InklingConvolution{channels, Layout::convolutionTaps, weights, dynamic,
                                   dynamic ? geometry_.attentionKvHeads : 0};
  };
  addConvolution(step, visit, InklingSteps::kKeyOffset, convolution(w.keyConvolution.taps, kv, !sliding), keys,
                 keyTaps, h.convolvedKey);
  addConvolution(step, visit, InklingSteps::kValueOffset, convolution(w.valueConvolution.taps, kv, !sliding), values,
                 valueTaps, h.convolvedValue);

  // Attention over the visit's own KV layer.
  const std::span<const kv::ChunkedPrefillParams> sequences = step.attentionSequences();
  const ops::AttentionRows attentionRows{b.laneRows, rows, step.decode ? step.decode->tree : metal::MetalBuffer{}};
  const std::span<const metal::MetalBuffer> pageTables = std::span(step.pageTables).first(sequences.size());
  ops::PagedAttention::addInklingPreparation(step.graph, h.packed, h.convolvedKey, h.convolvedValue,
                                             w.queryScale, w.relativeScale, h.preparedQuery, h.preparedRelative,
                                             step.kvLayers[visit], pageTables, sequences, attentionRows,
                                             geometry_.kvLayout, !sliding);
  std::array<ops::AttentionHistory, ExecutionLimits::maximumBatchWidth> histories{};
  for (size_t index = 0; index < sequences.size(); ++index)
    histories[index] = {step.kvLayers[visit], pageTables[index], {}};
  ops::PagedAttention::addInklingAttention(step.graph, h.preparedQuery, h.preparedRelative, w.relativeProjection,
                                           h.sinks, h.packed, h.attention, h.workspace, h.workspaceSplits, sequences,
                                           std::span(histories).first(sequences.size()), attentionRows,
                                           geometry_.kvLayout, sliding);
  linear.addProjection(step.graph, h.attention, w.output, h.branch, rows, b.linearScratch);
  // The residual, f32: the branch's convolution plus the stream, but at a
  // block boundary, which keeps its input as a block instead.
  const ops::InklingRows residual{h.residual, d, d, 0};
  addConvolution(step, visit, InklingSteps::kBranchOffset, convolution(w.attentionConvolution, d, false),
                 {h.branch, d, d, 0}, {}, h.residual, {.addend = boundary(layer) ? ops::InklingRows{} : x, .outputFloat = true});
  if (boundary(layer))
    ops::RowCopy::add(step.graph, h.x, {0, d, 0}, h.blocks, {0, 4 * d, layer / Layout::blockPeriod * d}, rows, d);
  std::vector<ops::InklingRows> states;
  for (uint32_t index = 0; index < blocks + boundary(layer); ++index) states.push_back(slot(index));
  const ops::InklingRows hidden{h.hidden, d, d, 0};
  ops::Inkling::addDepthMix(step.graph, states, residual, true, w.mlpInputQuery, 1, hidden, h.depthSums, d, rows);
  if (messages) {
    const InklingControllerWeights &c = w.controllers[3];
    ops::Inkling::addMessageGate(step.graph, hidden, {h.messages, d, 4 * d, 3 * d}, {c.down, c.forget, c.gain}, 64,
                                 {h.gateLow, 64, 64, 0}, hidden, d, rows);
  }
  ops::Inkling::addNorm(step.graph, hidden, w.postAttentionNorm, {b.normalized, d, d, 0}, d, rows);
  step.ffn = {.input = b.normalized, .residual = h.zeros, .output = h.expertOutput, .scratch = b.moe};
  if (const QwenTargetDecodeBuffers *dec = step.decode) {
    step.ffn.lanes = dec->step.lanes;
    step.ffn.laneRows = b.laneRows;
  }
  ops::MoE::addRouting(step.graph, step.ffn, w.moe, *step.moe);
  ++step.attentionLayer;
}

void InklingTarget::addPrefillExperts(PrefillStep &step, uint32_t visit) const {
  const InklingLayerWeights &w = weights_.layers[Layout::visitLayer(visit)];
  const InklingScratch &h = step.scratch;
  const uint32_t d = geometry_.hiddenSize;
  ops::MoE::addRouted(step.graph, step.ffn, w.moe, *step.moe);
  // The visit's output stream, bf16(residual + bf16(conv)).
  addConvolution(step, visit, InklingSteps::kExpertOffset,
                 {d, Layout::convolutionTaps, w.mlpConvolution, false, 0}, {h.expertOutput, d, d, 0}, {}, h.x,
                 {.addend = {h.residual, d, d, 0}, .addendFloat = true});
  if (visit + 1 < kVisits) addCaptures(step, visit, h.x);
}

// The draft reads the stream after a captured visit (DSpark's tap entering
// the next visit) and, of the last visit, the head's normed rows (its tap
// after the final norm): a decode step's every row, a prompt's spans.
void InklingTarget::addCaptures(PrefillStep &step, uint32_t visit, const metal::MetalBuffer &rows) const {
  const std::optional<uint32_t> slot = geometry_.captureSlot(visit);
  if (!slot) return;
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    addCapture(step.graph, *slot, rows, 0, d->capturedTargetHidden, 0, step.rows);
    return;
  }
  forEachPrefillCapture(step, visit, [&](uint32_t s, uint32_t source, uint32_t destination, uint32_t count) {
    addCapture(step.graph, s, rows, source, step.buffers.captured, destination, count);
  });
}

void InklingTarget::addFinalNorm(PrefillStep &step, metal::MetalBuffer output) const {
  const InklingScratch &h = step.scratch;
  const uint32_t d = geometry_.hiddenSize;
  std::vector<ops::InklingRows> states;
  for (uint32_t index = 0; index < 4; ++index) states.push_back({h.blocks, d, 4 * d, index * d});
  ops::Inkling::addDepthMix(step.graph, states, {h.x, d, d, 0}, false, weights_.finalInputQuery, 1,
                            {h.hidden, d, d, 0}, h.depthSums, d, step.rows);
  ops::Inkling::addNorm(step.graph, {h.hidden, d, d, 0}, weights_.finalNorm, {std::move(output), d, d, 0}, d,
                        step.rows);
}

void InklingTarget::addLogitCap(metal::CommandGraph &graph, metal::MetalBuffer logits, uint32_t rows) const {
  ops::Inkling::addLogitCap(graph, std::move(logits), Layout::logitSoftcap, Layout::unpaddedVocabulary,
                            geometry_.vocabularySize, rows);
}

// A prompt's rows end in the head's normed rows, which addHeadBatch projects;
// a decode step's own head projects every row.
metal::MetalBuffer InklingTarget::addPrefillEnd(PrefillStep &step) const {
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    addFinalNorm(step, d->finalHidden);
    addCaptures(step, kVisits - 1, d->finalHidden);
    const ops::Linear &linear = operators_.linear();
    linear.addProjection(step.graph, d->finalHidden, vocabularyProjection(), d->logits, step.rows,
                         step.buffers.linearScratch);
    addLogitCap(step.graph, d->logits, step.rows);
    return d->finalHidden;
  }
  addFinalNorm(step, step.buffers.hidden[1]);
  addCaptures(step, kVisits - 1, step.buffers.hidden[1]);
  return step.buffers.hidden[1];
}

void InklingTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer,
                                 metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const {
  operators_.linear().addProjection(graph, std::move(hidden), vocabularyProjection(), logits, rows, scratch);
  addLogitCap(graph, std::move(logits), rows);
}

// Every visit's convolution histories after each lane's accepted path, from
// the tape of its verify's inputs, in one dispatch.
void InklingTarget::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                   const ops::StepRows &step) const {
  QwenTarget::addStateCommit(graph, buffers, step);
  const InklingScratch scratch = steps_.verifyScratch();
  ops::Inkling::addConvolutionCommit(
      graph, {scratch.tape, InklingSteps::kHistoryWidth, InklingSteps::kHistoryWidth, 0}, 1,
      InklingSteps::kHistoryWidth,
      InklingSteps::kHistoryWidth, std::span(buffers.auxiliaryIn).first(step.lanes),
      std::span(buffers.auxiliaryOut).first(step.lanes), 0, InklingSteps::kHistoryWidth, Layout::convolutionTaps,
      buffers.retainedRows, buffers.laneRows, {step.lanes, step.rows, step.laneCap});
}

// A reply's state at a page boundary: one lane's histories after the first
// `rows` rows of its accepted path, from the same tape as its commit.
void InklingTarget::addAuxiliaryCapture(metal::CommandGraph &graph, QwenTargetCaptureBuffers buffers, uint32_t rows,
                                        const ops::StepRows &step) const {
  const InklingScratch scratch = steps_.verifyScratch();
  const std::array<metal::MetalBuffer, 1> in{std::move(buffers.auxiliaryIn)}, out{std::move(buffers.auxiliaryOut)};
  ops::Inkling::addConvolutionCommit(
      graph, {scratch.tape, InklingSteps::kHistoryWidth, InklingSteps::kHistoryWidth, 0}, 1,
      InklingSteps::kHistoryWidth, InklingSteps::kHistoryWidth, in, out, 0, InklingSteps::kHistoryWidth,
      Layout::convolutionTaps, std::move(buffers.retainedRows), std::move(buffers.laneRows),
      {1, step.rows, step.laneCap}, rows);
}

// The pipeline of the history capture inside a prompt's rows, which
// addConvolution runs within the visit that writes the inputs; the runtime
// only prepares this one (Runtime::warmupPrefill).
void InklingTarget::addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                                     metal::MetalBuffer captured) const {
  const uint32_t d = geometry_.hiddenSize;
  ops::Inkling::addConvolutionHistory(graph, {steps_.prefillScratch(0).branch, d, d, 0}, std::move(auxiliaryIn),
                                      std::move(captured), InklingSteps::kBranchOffset, InklingSteps::kHistoryWidth,
                                      d, Layout::convolutionTaps, 1);
}

std::unique_ptr<QwenTarget> makeQwenTarget(const InklingWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t prefillChunks) {
  return std::make_unique<InklingTarget>(weights, geometry, backend, operators, prefillChunks);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const InklingLayout &, const QwenTargetGeometry &geometry,
                                              uint32_t prefillChunks, uint64_t) {
  return InklingSteps::scratchBytes(geometry, prefillChunks);
}

} // namespace splash::model
