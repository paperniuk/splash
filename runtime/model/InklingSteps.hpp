#pragma once

// The scratch an Inkling 2 step needs beyond the buffers every Qwen target
// binds, which its target keeps (model/InklingTarget.hpp): the residual
// stream between visits and the states its depth attention reads, a visit's
// sublayer inputs, projections and convolution outputs, the attention's
// prepared rows and workspace, and a decode step's convolution inputs of
// every visit, which its commit folds into each lane's history.

#include "model/QwenTarget.hpp"

#include <cstdint>

namespace splash::model {

// A step's rows' tensors (a prefill chunk's or a decode step's). Carried
// across a command's layers and, for a prefill chunk, between the commands
// of one prompt's passes over the layers: the stream `x` (bf16), the block
// states (bf16, four of hidden values a row: the inputs of layers 0, 6, 12
// and 18), the message bank (bf16, two a row) and the attention branch's
// residual (f32), which a visit's MoE reads after the layer's expert load.
// The rest a visit writes and reads within itself. A decode step's
// convolution inputs (`tape`) hold every visit's, InklingSteps::kHistoryWidth
// values a row.
struct InklingScratch final {
  metal::MetalBuffer x;
  metal::MetalBuffer blocks;
  metal::MetalBuffer bank;
  metal::MetalBuffer residual;
  metal::MetalBuffer mixed;
  metal::MetalBuffer messages;
  metal::MetalBuffer parts;
  metal::MetalBuffer normed;
  metal::MetalBuffer packed;
  metal::MetalBuffer key;
  metal::MetalBuffer value;
  metal::MetalBuffer sinks;
  metal::MetalBuffer rawTaps;
  metal::MetalBuffer gateLow;
  metal::MetalBuffer depthSums;
  metal::MetalBuffer keyTaps;
  metal::MetalBuffer valueTaps;
  metal::MetalBuffer convolvedKey;
  metal::MetalBuffer convolvedValue;
  metal::MetalBuffer preparedQuery;
  metal::MetalBuffer preparedRelative;
  metal::MetalBuffer attention;
  metal::MetalBuffer branch;
  metal::MetalBuffer convolved;
  metal::MetalBuffer hidden;
  metal::MetalBuffer expertOutput;
  metal::MetalBuffer zeros;
  metal::MetalBuffer workspace;
  metal::MetalBuffer tape;
  uint32_t workspaceSplits = 0;
};

class InklingSteps final {
public:
  // A visit's convolution inputs in a row of the decode tape and of a state
  // cell's history: key 1024, value 1024, attention branch 2304, expert
  // output 2304; 48 visits of them.
  static constexpr uint32_t kVisitWidth = 1024 + 1024 + 2304 + 2304;
  static constexpr uint32_t kHistoryWidth = 48 * kVisitWidth;
  static constexpr uint32_t kKeyOffset = 0, kValueOffset = 1024, kBranchOffset = 2048, kExpertOffset = 4352;
  // The splits of a decode step's attention, whose rows alone give the GPU
  // too little work: one for a prefill chunk.
  static constexpr uint32_t kDecodeSplits = 64;

  InklingSteps(const QwenTargetGeometry &geometry, metal::MetalBackend &backend, uint32_t prefillChunks);
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry,
                                                           uint32_t prefillChunks);
  [[nodiscard]] QwenTargetScratchBytes allocatedBytes() const noexcept;
  // Chunk `index`'s scratch of a prefill command.
  [[nodiscard]] InklingScratch prefillScratch(uint32_t index) const;
  // A decode step's scratch, for its most rows.
  [[nodiscard]] InklingScratch verifyScratch() const;

private:
  struct Placed final {
    metal::MetalBuffer InklingScratch::*field;
    uint64_t rowBytes;
    uint64_t fixedBytes;
    bool carried;
    uint64_t offset;
  };
  [[nodiscard]] InklingScratch view(const metal::MetalBuffer &buffer, const std::vector<Placed> &tensors,
                                    uint32_t rows, uint32_t chunk, uint32_t splits) const;

  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  uint32_t prefillChunks_ = 0;
  metal::MetalBuffer prefill_;
  metal::MetalBuffer decode_;
  std::vector<Placed> prefillTensors_;
  std::vector<Placed> decodeTensors_;
};

} // namespace splash::model
