#include "model/Inkling.hpp"
#include "metal/abi/PagedAttention.h"

#include "model/ExpertStore.hpp"
#include "model/GgufPreparation.hpp"
#include "model/InklingImage.hpp"
#include "model/QwenTargetLoader.hpp"

namespace splash::model {
namespace {

struct Images final {
  explicit Images(const std::filesystem::path &directory) : checkpoint(directory) {
    for (uint32_t layer = 0; layer < InklingLayout::storedLayers; ++layer)
      plans.push_back(planInklingLayer(checkpoint, layer));
    plans.push_back(planInklingHead(checkpoint));
    plans.push_back(planInklingEmbedding(checkpoint));
  }
  SafetensorsCheckpoint checkpoint;
  std::vector<gguf::Image> plans;
};

} // namespace

QwenTargetGeometry qwenTargetGeometry(const InklingWeights &weights, std::span<const uint32_t> captureLayers) {
  QwenTargetGeometry geometry(weights.layout, captureLayers);
  geometry.kvLayout = weights.layout.kvLayout();
  geometry.stateLayout = weights.layout.gdnStateLayout();
  std::vector<const ops::BlockMoeWeights *> blocks;
  for (const auto &layer : weights.layers) {
    for (const ops::Projection *projection :
         {&layer.queryRelativeGate, &layer.queryRelativeGateAlone, &layer.key, &layer.value, &layer.output})
      if (!projection->blocks().segments.empty()) geometry.include(*projection);
    blocks.push_back(&layer.moe);
  }
  geometry.finish(blocks, weights.logitsProjection);
  return geometry;
}

InklingWeights loadInklingWeights(metal::MetalBackend &backend, InklingLayout layout, WeightImages &images,
                                  const std::filesystem::path &directory, const ExpertStoreOptions &options) {
  const uint64_t baseline = backend.memoryStats().allocatedBytes;
  if (options.memoryBytes) throw WeightStoreError("Inkling expert streaming is not supported");
  auto planned = std::make_shared<Images>(directory);
  const auto open = [&](uint32_t index) {
    return loadImage(backend, images, "target/", {planned, &planned->plans[index]},
                     [planned] { planned->checkpoint.checkUnchanged(); });
  };
  InklingWeights result;
  result.layout = layout;
  result.layers.reserve(InklingLayout::storedLayers);
  const metal::MetalBuffer slots = residentExpertSlots(backend, layout.experts);
  const BlockTargetFormat format{false, ops::GdnHeadOrder::Grouped};
  constexpr uint64_t bf16 = 2, f32 = 4, hidden = 2304;
  for (uint32_t i = 0; i < InklingLayout::storedLayers; ++i) {
    WeightFile file = open(i);
    auto &layer = result.layers.emplace_back();
    const bool sliding = InklingLayout::isSlidingLayer(i);
    if (sliding) layer.inputNorm = format.norm(file, hidden, "input-norm");
    else {
      layer.queryNorm = format.norm(file, hidden, "query-input-norm");
      layer.keyNorm = format.norm(file, hidden, "key-input-norm");
      layer.valueNorm = format.norm(file, hidden, "value-input-norm");
    }
    // The attention's packed rows (INKLING_PACKED_WIDTH): a sliding layer's
    // q|r|gate|k|v as one tensor, a full layer's q|r|gate with its k and v apart.
    if (sliding) {
      layer.queryRelativeGate = format.projection(file, INKLING_PACKED_WIDTH, hidden, "query-relative-gate-key-value");
      // Rows [first, first + count) of the fused tensor, whole 64-row plane
      // tiles of every plane.
      const ops::QuantizedSegment &fused = layer.queryRelativeGate.blocks().segments.front();
      const QuantFormat &f = kQuantFormats[fused.formatId];
      const uint64_t groups = hidden / 32;
      const auto rowsOf = [&](uint32_t first, uint32_t count) {
        const auto plane = [&](const metal::MetalBuffer &buffer, uint64_t rowBytes) {
          return rowBytes ? backend.view(buffer, first * rowBytes, count * rowBytes) : metal::MetalBuffer{};
        };
        ops::BlockWeights blocks;
        blocks.segments = {ops::QuantizedSegment::planes(fused.formatId, count, hidden,
                                                         plane(fused.plane0, groups * f.plane0_bytes),
                                                         plane(fused.plane1, groups * f.plane1_bytes),
                                                         plane(fused.meta, groups / f.meta_groups * f.meta_bytes))};
        return blocks;
      };
      layer.queryRelativeGateAlone = ops::Projection(INKLING_PACKED_WIDTH, hidden, rowsOf(0, INKLING_KEY_OFFSET));
      layer.key = ops::Projection(1024, hidden, rowsOf(INKLING_KEY_OFFSET, 1024));
      layer.value = ops::Projection(1024, hidden, rowsOf(INKLING_VALUE_OFFSET, 1024));
    } else {
      const ops::Projection packed = format.projection(file, INKLING_KEY_OFFSET, hidden, "query-relative-gate");
      layer.queryRelativeGate = ops::Projection(INKLING_PACKED_WIDTH, hidden, packed.blocks());
      layer.key = format.projection(file, 1024, hidden, "key");
      layer.value = format.projection(file, 1024, hidden, "value");
    }
    layer.output = format.projection(file, hidden, 3072, "output");
    layer.sink = readFloatSegment(file, "sink");
    layer.queryScale = file.section(24 * 128 * bf16, "query-scale");
    layer.relativeScale = file.section(24 * 16 * bf16, "relative-scale");
    layer.relativeProjection = file.section(16 * 1024 * bf16, "relative-projection");
    for (auto *conv : {&layer.keyConvolution, &layer.valueConvolution}) {
      conv->taps = file.section(1024 * 4 * bf16, "convolution-taps");
      if (!sliding) conv->weightProjection = readFloatSegment(file, "convolution-projection");
    }
    layer.attentionConvolution = file.section(hidden * 4 * bf16, "attention-convolution");
    layer.mlpConvolution = file.section(hidden * 4 * bf16, "mlp-convolution");
    if (i) layer.attentionInputQuery = file.section((sliding ? 1 : 3) * hidden * f32, "attention-input-query");
    layer.mlpInputQuery = file.section(hidden * f32, "mlp-input-query");
    if (i >= InklingLayout::coreBegin && i < InklingLayout::coreEnd)
      for (auto &controller : layer.controllers) {
        controller.down = file.section(64 * 2 * hidden * bf16, "controller-down");
        controller.forget = file.section(hidden * 64 * bf16, "controller-forget");
        controller.gain = file.section(hidden * 64 * bf16, "controller-gain");
      }
    layer.postAttentionNorm = format.norm(file, hidden, "post-attention-norm");
    layer.moe.router = readFloatSegment(file, "router");
    layer.moe.correctionBias = file.section(128 * f32, "correction-bias");
    layer.moe.gate.routed = readQuantizedSegment(file, "expert-gate");
    layer.moe.up.routed = readQuantizedSegment(file, "expert-up");
    layer.moe.down.routed = readQuantizedSegment(file, "expert-down");
    layer.moe.slotTable = slots;
    layer.moe.slots = layout.experts;
    layer.moe.chunks = ops::residentExpertChunks(layer.moe);
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = open(24);
    result.finalNorm = format.norm(file, hidden, "final-norm");
    result.finalInputQuery = file.section(hidden * f32, "final-input-query");
    result.messageQueries = file.section(12 * 4 * hidden * f32, "message-queries");
    result.logitsProjection = format.projection(file, layout.vocabularySize, hidden, "logits");
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = open(25);
    result.embeddingNorm = format.norm(file, hidden, "embedding-norm");
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, hidden);
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(baseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
