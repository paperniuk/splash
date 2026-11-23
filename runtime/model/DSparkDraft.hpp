#pragma once

#include "model/WeightStore.hpp"
#include "model/StateLayout.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/DraftSelector.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace splash::model {

class DSparkCheckpointLoader;

struct DSparkGeometry final {
  static constexpr uint32_t layers = 3;
  static constexpr uint32_t hidden = 2304;
  static constexpr uint32_t attention = 3072;
  static constexpr uint32_t kv = 1024;
  static constexpr uint32_t intermediate = 9216;
  static constexpr uint32_t head = 128;
  static constexpr uint32_t vocabulary = 201024;
  static constexpr uint32_t unpaddedVocabulary = 200058;
  static constexpr uint32_t rank = 256;
  static constexpr uint32_t block = 8;
  // Each tap entering visit n is the output of visit n-1; tap 48 is the
  // final normalized output captured after visit 47.
  static constexpr std::array<uint32_t, 4> captureLayers{4, 22, 40, 47};
  // The one layer that attends to the whole context: its K/V are rows from the
  // request's global base in segments (ops::DraftSegmentedContext), where the
  // other layers' are rings of SPLASH_DRAFT_SLIDING_WINDOW rows.
  static constexpr uint32_t globalLayer = 1;
  [[nodiscard]] static constexpr DraftStateLayout stateLayout(uint32_t capacity) noexcept {
    return {layers, 8, head, 4 * hidden, 1024, capacity, globalLayer, hidden};
  }
};

struct DSparkDraftLayout final {
  uint32_t contextCapacity = 0;
  static constexpr uint32_t hiddenSize = DSparkGeometry::hidden;
  static constexpr uint32_t vocabularySize = DSparkGeometry::vocabulary;
  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return DSparkGeometry::stateLayout(contextCapacity);
  }
  [[nodiscard]] constexpr std::span<const uint32_t> targetLayerIds() const noexcept {
    return DSparkGeometry::captureLayers;
  }
  [[nodiscard]] constexpr uint32_t targetHiddenSize() const noexcept { return 4 * hiddenSize; }
  bool operator==(const DSparkDraftLayout &) const = default;
};

struct DSparkLayerWeights final {
  metal::MetalBuffer contextNorm, inputNorm, query, key, value, contextKey, contextValue;
  metal::MetalBuffer queryNorm, keyNorm, contextKeyNorm, output, postAttentionNorm;
  metal::MetalBuffer gate, up, down;
};

struct DSparkDraftWeights final {
  std::array<DSparkLayerWeights, DSparkGeometry::layers> layers;
  metal::MetalBuffer fuse, mask, finalNorm, markovNorm, markovEmbedding;
  metal::MetalBuffer recurrent, recurrentBias, hiddenOut, hiddenOutBias;
  // The Markov correction folded through the target head
  // (foldDSparkCorrection): bf16 [vocabulary][rank] and fp32 [vocabulary].
  metal::MetalBuffer foldWeight, foldBias;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

[[nodiscard]] DSparkDraftWeights loadDSparkDraftWeights(metal::MetalBackend &backend,
                                                        DSparkCheckpointLoader &files);

// The bytes of DSparkDraftWeights::foldWeight and foldBias, each in whole
// 16 KiB host pages as Metal allocates them, which the draft's planned
// weights count beside its images.
[[nodiscard]] constexpr uint64_t dsparkFoldBytes() noexcept {
  constexpr uint64_t page = 16384;
  const auto pages = [](uint64_t bytes) { return (bytes + page - 1) / page * page; };
  return pages(uint64_t{DSparkGeometry::vocabulary} * DSparkGeometry::rank * 2) +
         pages(uint64_t{DSparkGeometry::vocabulary} * sizeof(float));
}

// Folds the Markov head's hidden output (hiddenOut and its bias) through the
// target's vocabulary `head`, whichever format it is stored in, into
// weights.foldWeight and foldBias: for token t, fold_weight[t][r] =
// sum_k head[t][k] hiddenOut[k][r] rounded to bf16 and fold_bias[t] =
// sum_k head[t][k] hiddenOutBias[k], fp32, accumulated by the head's own
// projection of the hidden output's columns and its bias as inputs. A proposal
// step then corrects every token's logit by fold_bias[t] + fold_weight[t] .
// latent. Runs on the GPU and waits for it.
void foldDSparkCorrection(metal::MetalBackend &backend, const ops::Linear &linear, const ops::Projection &head,
                          DSparkDraftWeights &weights);

struct DSparkContextBuffers final {
  metal::MetalBuffer captured;
  metal::MetalBuffer fused;
  metal::MetalBuffer rawKeys;
  metal::MetalBuffer preparedKeys;
  metal::MetalBuffer projectedValues;
  std::array<metal::MetalBuffer, DSparkGeometry::layers> keys;
  std::array<metal::MetalBuffer, DSparkGeometry::layers> values;
  // Rings keys and values hold every layer but DSparkGeometry::globalLayer,
  // whose segments are `global`.
  ops::DraftSegmentedContext global;
  metal::MetalBuffer markovFeature;
  ops::LinearScratch scratch;
};

struct DSparkBlockBuffers final {
  metal::MetalBuffer anchor;
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer residual, normalized, rawQuery, rawKey, query, key, value;
  metal::MetalBuffer attention, projected, gate, up, intermediate, finalHidden;
  std::array<metal::MetalBuffer, DSparkGeometry::layers> contextKeys;
  std::array<metal::MetalBuffer, DSparkGeometry::layers> contextValues;
  ops::DraftSegmentedContext global;
  ops::LinearScratch scratch;
};

class DSparkDraft final {
public:
  DSparkDraft(const DSparkDraftWeights &weights, metal::MetalBackend &backend,
              const ops::ExecutionPlans &operators);
  void addContext(metal::CommandGraph &graph, const DSparkContextBuffers &buffers,
                  uint32_t rows, uint32_t startPosition, uint32_t globalBase) const;
  void addContextCommit(metal::CommandGraph &graph, const DSparkContextBuffers &buffers,
                        metal::MetalBuffer compact, metal::MetalBuffer retained, metal::MetalBuffer laneRows,
                        uint32_t packedRows, uint32_t laneCap, uint32_t startPosition, uint32_t globalBase) const;
  void addBlock(metal::CommandGraph &graph, const DSparkBlockBuffers &buffers,
                uint32_t position, uint32_t localBase, uint32_t globalBase) const;
  void addChain(metal::CommandGraph &graph, ops::DSparkMarkovBuffers buffers,
                const ops::Projection &head, ops::LinearScratch scratch, uint32_t anchor,
                uint32_t proposals) const;

private:
  const DSparkDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  struct LayerProjections final {
    ops::Projection query, key, value, output, gate, up, down;
  };
  std::array<LayerProjections, DSparkGeometry::layers> layers_;
  std::array<ops::Projection, DSparkGeometry::layers> contextKeys_;
  std::array<ops::Projection, DSparkGeometry::layers> contextValues_;
  // The block attention's split partials, every layer's and lane's in turn.
  metal::MetalBuffer attentionWorkspace_;
};

} // namespace splash::model
