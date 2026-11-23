#include "model/DSparkDraft.hpp"
#include "model/DraftCheckpoint.hpp"

#include <algorithm>
#include <cstring>

namespace splash::model {

DSparkDraftWeights loadDSparkDraftWeights(metal::MetalBackend &backend, DSparkCheckpointLoader &files) {
  using G = DSparkGeometry;
  constexpr uint64_t bf16 = 2;
  const uint64_t before = backend.memoryStats().allocatedBytes;
  DSparkDraftWeights result;
  result.files.reserve(G::layers + 1);
  for (uint32_t i = 0; i < G::layers; ++i) {
    WeightFile f = files.layer(i);
    DSparkLayerWeights &w = result.layers[i];
    w.contextNorm = f.section(G::hidden * bf16, "context-norm");
    w.inputNorm = f.section(G::hidden * bf16, "input-norm");
    w.query = f.section(G::attention * G::hidden * bf16, "query");
    w.key = f.section(G::kv * G::hidden * bf16, "key");
    w.value = f.section(G::kv * G::hidden * bf16, "value");
    w.contextKey = f.section(G::kv * G::hidden * bf16, "context-key");
    w.contextValue = f.section(G::kv * G::hidden * bf16, "context-value");
    w.queryNorm = f.section(G::head * bf16, "query-norm");
    w.keyNorm = f.section(G::head * bf16, "key-norm");
    w.contextKeyNorm = f.section(G::head * bf16, "context-key-norm");
    w.output = f.section(G::hidden * G::attention * bf16, "attention-output");
    w.postAttentionNorm = f.section(G::hidden * bf16, "post-attention-norm");
    w.gate = f.section(G::intermediate * G::hidden * bf16, "gate");
    w.up = f.section(G::intermediate * G::hidden * bf16, "up");
    w.down = f.section(G::hidden * G::intermediate * bf16, "down");
    f.finish();
    result.files.push_back(f.record());
  }
  WeightFile f = files.model();
  result.fuse = f.section(4 * 4 * bf16, "fuse");
  result.mask = f.section(G::hidden * bf16, "mask");
  result.finalNorm = f.section(G::hidden * bf16, "final-norm");
  result.markovNorm = f.section(G::hidden * bf16, "markov-norm");
  result.markovEmbedding = f.section(uint64_t{G::vocabulary} * G::rank * bf16, "markov-embedding");
  result.recurrent = f.section(3 * G::rank * (2 * G::rank + 2 * G::hidden) * bf16, "recurrent");
  result.recurrentBias = f.section(3 * G::rank * bf16, "recurrent-bias");
  result.hiddenOut = f.section(G::hidden * G::rank * bf16, "hidden-out");
  result.hiddenOutBias = f.section(G::hidden * bf16, "hidden-out-bias");
  f.finish();
  result.files.push_back(f.record());
  result.actualAllocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  return result;
}

void foldDSparkCorrection(metal::MetalBackend &backend, const ops::Linear &linear, const ops::Projection &head,
                          DSparkDraftWeights &weights) {
  using G = DSparkGeometry;
  if (head.inputSize != G::hidden || head.outputSize != G::vocabulary ||
      head.destination != ops::FloatOutput::Float32 || !weights.hiddenOut || !weights.hiddenOutBias)
    throw std::invalid_argument("invalid DSpark fold head");
  constexpr uint32_t inputRows = G::rank + 1, chunk = 32;
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const auto shared = [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  };
  weights.foldWeight = shared(uint64_t{G::vocabulary} * G::rank * 2, "dspark-fold-weight");
  weights.foldBias = shared(uint64_t{G::vocabulary} * sizeof(float), "dspark-fold-bias");
  {
    // The fold's inputs, then the head's products of them a chunk of rows at a
    // time, each stored before the next overwrites the products. The head
    // projects them as it projects hidden rows (a rotated head rotates them),
    // so the products are what a hidden correction's logits are.
    const ops::ProjectionShape shape = head.shape();
    const uint32_t storage = std::max(chunk, linear.storageRows(chunk, shape));
    const metal::MetalBuffer inputs = shared(uint64_t{inputRows + storage} * G::hidden * 2, "dspark-fold-inputs");
    const metal::MetalBuffer product = shared(uint64_t{storage} * G::vocabulary * sizeof(float), "dspark-fold-product");
    const ops::LinearScratchSize size = linear.scratchSize(shape, chunk);
    const auto scratchBuffer = [&](uint64_t bytes) {
      return bytes ? shared(bytes, "dspark-fold-scratch") : metal::MetalBuffer{};
    };
    ops::LinearScratch scratch;
    scratch.input = scratchBuffer(size.input);
    scratch.sums = scratchBuffer(size.sums);
    scratch.partials = scratchBuffer(size.partials);
    scratch.counters = scratchBuffer(size.counters);
    scratch.rotated = scratchBuffer(size.rotated);
    metal::CommandGraph prepare;
    ops::DraftSelector::addDSparkFoldInputs(prepare, weights.hiddenOut, weights.hiddenOutBias, inputs);
    (void)backend.submitCommandAsync(prepare.dispatches()).wait();
    for (uint32_t start = 0; start < inputRows; start += chunk) {
      const uint32_t rows = std::min(chunk, inputRows - start);
      const uint64_t offset = uint64_t{start} * G::hidden * 2;
      metal::CommandGraph graph;
      linear.addProjection(graph, backend.view(inputs, offset, inputs.sizeBytes() - offset), head, product, rows,
                           scratch);
      ops::DraftSelector::addDSparkFoldStore(graph, product, weights.foldWeight, weights.foldBias, start, rows,
                                             G::vocabulary);
      (void)backend.submitCommandAsync(graph.dispatches()).wait();
    }
  }
  weights.actualAllocatedBytes += metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
}

namespace {

ops::Projection bf16Projection(uint32_t output, uint32_t input, metal::MetalBuffer weights) {
  ops::BlockWeights blocks;
  blocks.segments.push_back(ops::QuantizedSegment::rows(GGUF_FMT_BF16, output, input, std::move(weights)));
  return ops::Projection(output, input, std::move(blocks));
}

} // namespace

DSparkDraft::DSparkDraft(const DSparkDraftWeights &weights, metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      attentionWorkspace_(backend.allocateBuffer(ops::DraftAttention::dsparkAttentionWorkspaceBytes(),
                                                 metal::BufferStorage::Private, "DSpark attention workspace")) {
  for (uint32_t layer = 0; layer < DSparkGeometry::layers; ++layer) {
    contextKeys_[layer] = bf16Projection(DSparkGeometry::kv, DSparkGeometry::hidden, weights.layers[layer].contextKey);
    contextValues_[layer] = bf16Projection(DSparkGeometry::kv, DSparkGeometry::hidden, weights.layers[layer].contextValue);
    auto &p = layers_[layer];
    const auto &w = weights.layers[layer];
    p.query = bf16Projection(DSparkGeometry::attention, DSparkGeometry::hidden, w.query);
    p.key = bf16Projection(DSparkGeometry::kv, DSparkGeometry::hidden, w.key);
    p.value = bf16Projection(DSparkGeometry::kv, DSparkGeometry::hidden, w.value);
    p.output = bf16Projection(DSparkGeometry::hidden, DSparkGeometry::attention, w.output);
    p.gate = bf16Projection(DSparkGeometry::intermediate, DSparkGeometry::hidden, w.gate);
    p.up = bf16Projection(DSparkGeometry::intermediate, DSparkGeometry::hidden, w.up);
    p.down = bf16Projection(DSparkGeometry::hidden, DSparkGeometry::intermediate, w.down);
  }
}

void DSparkDraft::addContext(metal::CommandGraph &graph, const DSparkContextBuffers &buffers,
                            uint32_t rows, uint32_t startPosition, uint32_t globalBase) const {
  if (!rows || globalBase > startPosition) throw std::invalid_argument("invalid DSpark context span");
  const auto &linear = operators_.draftLinear();
  for (uint32_t layer = 0; layer < DSparkGeometry::layers; ++layer) {
    const auto &w = weights_.layers[layer];
    ops::DraftAttention::addDSparkFusion(graph, buffers.captured, weights_.fuse, w.contextNorm, buffers.fused,
                                        rows, layer);
    linear.add(graph, {.input = buffers.fused, .output = buffers.rawKeys, .scratch = buffers.scratch},
               contextKeys_[layer], linear.plan(contextKeys_[layer], rows));
    linear.add(graph, {.input = buffers.fused, .output = buffers.projectedValues, .scratch = buffers.scratch},
               contextValues_[layer], linear.plan(contextValues_[layer], rows));
    ops::DraftAttention::addDSparkHeadPrepare(graph, buffers.rawKeys, w.contextKeyNorm, buffers.preparedKeys,
                                             rows, 8, startPosition);
    if (layer == DSparkGeometry::globalLayer) {
      ops::DraftAttention::addDSparkSegmentedContextStore(graph, buffers.preparedKeys, buffers.projectedValues,
                                                          buffers.global, rows, startPosition, globalBase);
      continue;
    }
    const uint32_t capacity = static_cast<uint32_t>(buffers.keys[layer].sizeBytes() / (1024 * 2));
    ops::DraftAttention::addDSparkContextStore(graph, buffers.preparedKeys, buffers.projectedValues,
                                               buffers.keys[layer], buffers.values[layer], rows,
                                               startPosition, 0, capacity);
  }
  constexpr uint64_t rowBytes = uint64_t{4} * DSparkGeometry::hidden * 2;
  const auto last = backend_.view(buffers.captured, uint64_t{rows - 1} * rowBytes, rowBytes);
  ops::DraftAttention::addDSparkFusion(graph, last, weights_.fuse, weights_.markovNorm, buffers.markovFeature, 1, 3);
}

void DSparkDraft::addContextCommit(metal::CommandGraph &graph, const DSparkContextBuffers &b,
                                   metal::MetalBuffer compact, metal::MetalBuffer retained, metal::MetalBuffer laneRows,
                                   uint32_t packedRows, uint32_t laneCap, uint32_t startPosition,
                                   uint32_t globalBase) const {
  if (globalBase > startPosition) throw std::invalid_argument("invalid DSpark committed context");
  ops::DraftAttention::addDSparkCommitGather(graph, b.captured, retained, laneRows, compact, packedRows, laneCap);
  const auto &linear = operators_.draftLinear();
  for (uint32_t layer = 0; layer < DSparkGeometry::layers; ++layer) {
    const auto &w = weights_.layers[layer];
    ops::DraftAttention::addDSparkFusion(graph, compact, weights_.fuse, w.contextNorm, b.fused, laneCap, layer);
    linear.add(graph, {.input = b.fused, .output = b.rawKeys, .scratch = b.scratch},
               contextKeys_[layer], linear.plan(contextKeys_[layer], laneCap));
    linear.add(graph, {.input = b.fused, .output = b.projectedValues, .scratch = b.scratch},
               contextValues_[layer], linear.plan(contextValues_[layer], laneCap));
    ops::DraftAttention::addDSparkHeadPrepare(graph, b.rawKeys, w.contextKeyNorm, b.preparedKeys,
                                             laneCap, 8, startPosition);
    if (layer == DSparkGeometry::globalLayer) {
      ops::DraftAttention::addDSparkSegmentedCommitStore(graph, b.preparedKeys, b.projectedValues, retained,
                                                         b.global, laneCap, startPosition, globalBase);
      continue;
    }
    const uint32_t capacity = static_cast<uint32_t>(b.keys[layer].sizeBytes() / (1024 * 2));
    ops::DraftAttention::addDSparkCommitStore(graph, b.preparedKeys, b.projectedValues, retained,
                                              b.keys[layer], b.values[layer], laneCap, startPosition, 0, capacity);
  }
  constexpr uint64_t rowBytes = uint64_t{4} * DSparkGeometry::hidden * 2;
  const auto last = backend_.view(compact, uint64_t{laneCap} * rowBytes, rowBytes);
  ops::DraftAttention::addDSparkFusion(graph, last, weights_.fuse, weights_.markovNorm, b.markovFeature, 1, 3);
}

void DSparkDraft::addBlock(metal::CommandGraph &graph, const DSparkBlockBuffers &b,
                           uint32_t position, uint32_t localBase, uint32_t globalBase) const {
  using G = DSparkGeometry;
  if (localBase > position || globalBase > position) throw std::invalid_argument("invalid DSpark context range");
  const auto &linear = operators_.draftLinear();
  const auto project = [&](const ops::Projection &p, metal::MetalBuffer input, metal::MetalBuffer output) {
    linear.add(graph, {.input = std::move(input), .output = std::move(output), .scratch = b.scratch},
               p, linear.plan(p, G::block));
  };
  ops::DraftAttention::addDSparkBlockInput(graph, b.anchor, weights_.mask, b.hidden[0], b.residual);
  for (uint32_t layer = 0; layer < G::layers; ++layer) {
    const auto &w = weights_.layers[layer];
    const auto &p = layers_[layer];
    if (layer == 0)
      ops::Normalization::addRms(graph, b.hidden[0], {w.inputNorm, false}, b.normalized, G::hidden, G::block);
    else
      ops::DraftAttention::addDSparkResidualNorm(graph, b.hidden[layer & 1], b.residual, w.inputNorm, b.normalized);
    project(p.query, b.normalized, b.rawQuery);
    project(p.key, b.normalized, b.rawKey);
    project(p.value, b.normalized, b.value);
    ops::DraftAttention::addDSparkHeadPrepare(graph, b.rawQuery, w.queryNorm, b.query, G::block, 24, position);
    ops::DraftAttention::addDSparkHeadPrepare(graph, b.rawKey, w.keyNorm, b.key, G::block, 8, position);
    const uint32_t first = layer == G::globalLayer ? 0 : std::max(localBase, position > 1024 ? position - 1024 : 0);
    // The global layer's rows are those of the committed positions from its
    // base up to `position`; rows beyond, left by rows a rewind dropped, are
    // never read.
    if (layer == G::globalLayer)
      ops::DraftAttention::addDSparkSegmentedAttention(graph, b.query, b.global, b.key, b.value, b.attention,
                                                       attentionWorkspace_, position - globalBase, first);
    else
      ops::DraftAttention::addDSparkAttention(graph, b.query, b.contextKeys[layer], b.contextValues[layer],
                                              b.key, b.value, b.attention, attentionWorkspace_, position, first,
                                              1024);
    project(p.output, b.attention, b.projected);
    ops::DraftAttention::addDSparkResidualNorm(graph, b.projected, b.residual, w.postAttentionNorm, b.normalized);
    project(p.gate, b.normalized, b.gate);
    project(p.up, b.normalized, b.up);
    ops::DraftAttention::addDSparkSwiGlu(graph, b.gate, b.up, b.intermediate);
    project(p.down, b.intermediate, b.hidden[(layer + 1) & 1]);
  }
  ops::DraftAttention::addDSparkResidualNorm(graph, b.hidden[G::layers & 1], b.residual, weights_.finalNorm,
                                              b.finalHidden);
}

void DSparkDraft::addChain(metal::CommandGraph &graph, ops::DSparkMarkovBuffers b,
                           const ops::Projection &head, ops::LinearScratch scratch,
                           uint32_t anchor, uint32_t proposals) const {
  using G = DSparkGeometry;
  if (!proposals || proposals >= G::block || head.inputSize != G::hidden ||
      head.outputSize != G::vocabulary || head.destination != ops::FloatOutput::Float32)
    throw std::invalid_argument("invalid DSpark proposal head or count");
  if (!weights_.foldWeight || !weights_.foldBias)
    throw std::invalid_argument("DSpark correction is not folded through the head");
  b.recurrentWeights = weights_.recurrent;
  b.recurrentBias = weights_.recurrentBias;
  b.embedding = weights_.markovEmbedding;
  b.foldWeight = weights_.foldWeight;
  b.foldBias = weights_.foldBias;
  ops::DraftSelector::addDSparkContext(graph, b, proposals);
  // The base logits of the requested proposal rows alone, in one batch: the
  // block's eight rows run as trained, but its anchor row and the rows past
  // the proposals reach no head.
  const auto &target = operators_.linear();
  const uint64_t rowBytes = uint64_t{G::hidden} * 2;
  const auto draftRows = backend_.view(b.hidden, rowBytes, b.hidden.sizeBytes() - rowBytes);
  target.add(graph, {.input = draftRows, .output = b.baseLogits, .scratch = scratch},
             head, target.plan(head, proposals));
  // A position's Markov step, then its scan of every legal token's base logit
  // plus the folded correction, whose argmax is the next step's previous token.
  for (uint32_t position = 0; position < proposals; ++position) {
    ops::DraftSelector::addDSparkStep(graph, b, position, anchor);
    ops::DraftSelector::addDSparkSelect(graph, b, position, G::vocabulary, G::unpaddedVocabulary);
  }
}

} // namespace splash::model
