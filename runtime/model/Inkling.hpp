#pragma once

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace splash::model {

struct InklingLayout final : QwenHybridLayout {
  static constexpr std::string_view modelType = "inkling2_loop";
  static constexpr std::string_view ggufArchitecture = "inkling2_loop";
  static constexpr bool loadsSafetensors = true;
  static constexpr uint32_t storedLayers = 24;
  static constexpr uint32_t coreBegin = 6, coreEnd = 18, passes = 3;
  static constexpr uint32_t blockPeriod = 6, relativeDimension = 16, relativeExtent = 1024;
  static constexpr uint32_t convolutionTaps = 4, unpaddedVocabulary = 200058;
  static constexpr float logitSoftcap = 30.0F, logScalingFloor = 8192.0F, logScalingAlpha = 0.1F;
  static constexpr float qkClip = 22.627416997969522F, relativeClip = 8.0F;

  [[nodiscard]] static constexpr uint32_t visitLayer(uint32_t visit) noexcept {
    return visit < coreBegin ? visit : visit < coreBegin + (coreEnd - coreBegin) * passes
        ? coreBegin + (visit - coreBegin) % (coreEnd - coreBegin)
        : coreEnd + visit - (coreBegin + (coreEnd - coreBegin) * passes);
  }
  [[nodiscard]] static constexpr bool isSlidingLayer(uint32_t layer) noexcept {
    return (layer + 1) % blockPeriod != 0;
  }
  static constexpr uint64_t slidingMask() noexcept;

  constexpr InklingLayout();
  bool operator==(const InklingLayout &) const = default;
};

constexpr uint64_t InklingLayout::slidingMask() noexcept {
  uint64_t mask = 0;
  for (uint32_t visit = 0; visit < 48; ++visit)
    if (isSlidingLayer(visitLayer(visit))) mask |= uint64_t{1} << visit;
  return mask;
}

constexpr InklingLayout::InklingLayout()
    : QwenHybridLayout{{.nativeContextTokens = 1'048'576,
                        .layers = 48,
                        .hiddenSize = 2304,
                        .vocabularySize = 201024,
                        .packedFullWidth = 8192,
                        .attentionWidth = 3072,
                        .attentionQueryHeads = 24,
                        .attentionKvHeads = 8,
                        .attentionHeadDimension = 128,
                        // Unread: its own loader (planInklingLayer) takes its
                        // norms as stored, and it has no q or k norm.
                        .attentionNormMean = 1.0F,
                        .familyAttention = true,
                        .rotaryPairs = 0,
                        .slidingWindow = 1024,
                        .slidingLayers = slidingMask(),
                        .unrotatedLayers = (uint64_t{1} << 48) - 1,
                        .fullAttentionPeriod = 1,
                        // Its last unused special token, <|unused_200056|>:
                        // its vocabulary keeps no mask token of its own.
                        .maskToken = 200056,
                        .draftReadsLayerInputs = true,
                        .stopTokens = {200006},
                        .ffnKind = QwenFfnKind::SparseMoe,
                        .experts = 128,
                        .expertsPerToken = 8,
                        .expertIntermediateSize = 1152,
                        .moeRouter = ops::MoeRouter::SigmoidCorrected,
                        .sharedExpertSlices = 0,
                        .sharedExpertGate = false,
                        .moeRouteScale = 8.0F,
                        .moeActivation = ops::MoeActivation::SoftcappedSwiGlu,
                        .moeGateSoftcap = 4.0F,
                        .moeUpSoftcap = 25.0F,
                        .familyStateBytes = uint64_t{48} * 3 * (1024 + 1024 + 2304 + 2304) * 2},
                       "Inkling-2-Mini"} {}


static_assert(InklingLayout::visitLayer(5) == 5 && InklingLayout::visitLayer(6) == 6 &&
              InklingLayout::visitLayer(18) == 6 && InklingLayout::visitLayer(30) == 6 &&
              InklingLayout::visitLayer(42) == 18 && InklingLayout::visitLayer(47) == 23);
static_assert(InklingLayout{}.familyStateBytes == 1'916'928);
struct InklingControllerWeights final {
  metal::MetalBuffer down, forget, gain;
};

struct InklingConvolutionWeights final {
  metal::MetalBuffer taps;
  ops::QuantizedSegment weightProjection;
};

struct InklingLayerWeights final {
  ops::Projection queryRelativeGate, key, value, output;
  // A sliding layer's q|r|gate rows alone, for a visit whose k and v read rows
  // of their own: like key and value, views of queryRelativeGate's planes.
  ops::Projection queryRelativeGateAlone;
  ops::QuantizedSegment sink;
  ops::NormWeights inputNorm, queryNorm, keyNorm, valueNorm, postAttentionNorm;
  metal::MetalBuffer queryScale, relativeScale, relativeProjection;
  InklingConvolutionWeights keyConvolution, valueConvolution;
  metal::MetalBuffer attentionConvolution, mlpConvolution;
  std::array<InklingControllerWeights, 4> controllers;
  metal::MetalBuffer attentionInputQuery, mlpInputQuery;
  ops::BlockMoeWeights moe;
};

struct InklingWeights final : QwenTargetWeightsBase {
  InklingLayout layout;
  std::vector<InklingLayerWeights> layers;
  ops::NormWeights embeddingNorm, finalNorm;
  metal::MetalBuffer finalInputQuery, messageQueries;
};

struct ExpertStoreOptions;
class WeightImages;
enum class TargetSource : uint8_t;
// The target's weights from its safetensors checkpoint or its GGUF (`source`)
// in `directory`, whose images hold the same sections either way.
[[nodiscard]] InklingWeights loadInklingWeights(metal::MetalBackend &backend, InklingLayout layout,
                                                WeightImages &images, const std::filesystem::path &directory,
                                                TargetSource source, const ExpertStoreOptions &options);

[[nodiscard]] inline const ops::BlockMoeWeights *qwenMoeBlock(const InklingLayerWeights &layer) noexcept {
  return &layer.moe;
}
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const InklingWeights &weights,
                                                    std::span<const uint32_t> captureLayers);

[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const InklingWeights &weights,
                                                        const QwenTargetGeometry &geometry,
                                                        metal::MetalBackend &backend,
                                                        const ops::ExecutionPlans &operators,
                                                        uint32_t prefillChunks);
// The step scratch its target keeps itself (model/InklingSteps.hpp).
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const InklingLayout &layout,
                                                            const QwenTargetGeometry &geometry,
                                                            uint32_t prefillChunks, uint64_t tableRowBytes);

} // namespace splash::model
