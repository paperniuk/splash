#pragma once

// Inkling 2's target (model/Inkling.hpp): 48 visits of 24 stored layers,
// the core's twelve run in three passes, over a bf16 stream whose blocks and
// messages its depth attention reads (ops::Inkling), each visit with its own
// KV layer and short-convolution histories. A decode step is a prefill over
// its verify rows; its commit folds every visit's convolution inputs along
// each lane's accepted path into the lane's history in one dispatch.

#include "model/InklingSteps.hpp"
#include "model/QwenTarget.hpp"
#include "ops/Inkling.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

struct InklingWeights;
struct InklingLayerWeights;

class InklingTarget final : public QwenTarget {
public:
  InklingTarget(const InklingWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                const ops::ExecutionPlans &operators, uint32_t prefillChunks);

  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override;
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                      const ops::StepRows &step) const override;
  [[nodiscard]] bool capturesAuxiliary() const noexcept override { return true; }
  void addAuxiliaryCapture(metal::CommandGraph &graph, QwenTargetCaptureBuffers buffers, uint32_t rows,
                           const ops::StepRows &step) const override;
  void addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                        metal::MetalBuffer captured) const override;

private:
  friend class QwenTarget;

  struct PrefillStep : QwenTarget::PrefillStep {
    InklingScratch scratch{};
    uint32_t capacity = 0;
  };

  // The layers' hooks addPrefillLayers calls (QwenTarget.hpp): every visit
  // attends; a visit's routing ends addPrefillLayer and its experts, its
  // last convolution and the stream's update addPrefillExperts.
  void addPrefillBegin(PrefillStep &step) const;
  [[nodiscard]] bool attends(uint32_t) const noexcept { return true; }
  void addPrefillLayer(PrefillStep &step, uint32_t visit) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t visit) const;
  void addPrefillExperts(PrefillStep &step, uint32_t visit) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  // One of a visit's four short convolutions of `input` (its rows' bf16
  // inputs of `channels`) into `output`, from history value `offset` of the
  // visit's histories: a prompt sequence's histories in and out and the
  // states captured inside it, a decode lane's in alone, its rows' inputs
  // into the step's tape, from which its commit writes the next.
  void addConvolution(PrefillStep &step, uint32_t visit, uint32_t offset, const ops::InklingConvolution &convolution,
                      ops::InklingRows input, ops::InklingRows dynamicTaps, metal::MetalBuffer output,
                      const ops::InklingResidual &residual = {}) const;
  // The draft's captures of visit `visit`'s rows `rows` (addCaptures).
  void addCaptures(PrefillStep &step, uint32_t visit, const metal::MetalBuffer &rows) const;
  // The head's normed rows of the step's last visit into `output`.
  void addFinalNorm(PrefillStep &step, metal::MetalBuffer output) const;
  void addLogitCap(metal::CommandGraph &graph, metal::MetalBuffer logits, uint32_t rows) const;

  const InklingWeights &weights_;
  InklingSteps steps_;
};

} // namespace splash::model
