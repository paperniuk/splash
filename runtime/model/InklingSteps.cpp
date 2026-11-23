#include "model/InklingSteps.hpp"

#include "metal/abi/PagedAttention.h"
#include "model/StateLayout.hpp"
#include "ops/Inkling.hpp"
#include "ops/PagedAttention.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace splash::model {
namespace {

constexpr uint32_t kChunkRows = ExecutionLimits::prefillTokenBudget;
// A decode step's most rows: every lane's verify block, packed.
constexpr uint32_t kStepRows = ExecutionLimits::maximumBatchWidth * ExecutionLimits::targetVerifyRows;

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept { return elements * sizeof(T); }

struct Tensor final {
  metal::MetalBuffer InklingScratch::*field;
  uint64_t rowBytes;
  uint64_t fixedBytes;
  bool carried;
};

// A step's tensors: per row, but a decode step's tape (every visit's
// convolution inputs) and the attention's workspace, sized by the step's
// rows and splits.
std::vector<Tensor> tensors(const QwenTargetGeometry &g, bool decode) {
  const uint64_t d = g.hiddenSize;
  const uint64_t q = uint64_t{g.attentionQueryHeads} * g.attentionHeadDimension;
  const uint64_t kv = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
  const uint32_t splits = decode ? InklingSteps::kDecodeSplits : 1;
  std::vector<Tensor> result{
      {&InklingScratch::x, bytesFor<uint16_t>(d), 0, true},
      {&InklingScratch::blocks, bytesFor<uint16_t>(4 * d), 0, true},
      {&InklingScratch::bank, bytesFor<uint16_t>(2 * d), 0, true},
      {&InklingScratch::residual, bytesFor<float>(d), 0, true},
      {&InklingScratch::mixed, bytesFor<float>(3 * d), 0, false},
      {&InklingScratch::messages, bytesFor<float>(4 * d), 0, false},
      {&InklingScratch::parts, bytesFor<float>(3 * d), 0, false},
      {&InklingScratch::normed, bytesFor<uint16_t>(3 * d), 0, false},
      {&InklingScratch::packed, bytesFor<uint16_t>(INKLING_PACKED_WIDTH), 0, false},
      {&InklingScratch::key, bytesFor<uint16_t>(kv), 0, false},
      {&InklingScratch::value, bytesFor<uint16_t>(kv), 0, false},
      {&InklingScratch::sinks, bytesFor<uint16_t>(g.attentionQueryHeads), 0, false},
      {&InklingScratch::keyTaps, bytesFor<float>(uint64_t{g.attentionKvHeads} * 4), 0, false},
      {&InklingScratch::rawTaps, bytesFor<uint16_t>(uint64_t{g.attentionKvHeads} * 4), 0, false},
      {&InklingScratch::gateLow, bytesFor<float>(3 * 64), 0, false},
      {&InklingScratch::depthSums, ops::Inkling::depthSumsBytes(g.hiddenSize, 1), 0, false},
      {&InklingScratch::valueTaps, bytesFor<float>(uint64_t{g.attentionKvHeads} * 4), 0, false},
      {&InklingScratch::convolvedKey, bytesFor<uint16_t>(kv), 0, false},
      {&InklingScratch::convolvedValue, bytesFor<uint16_t>(kv), 0, false},
      {&InklingScratch::preparedQuery, bytesFor<uint16_t>(q), 0, false},
      {&InklingScratch::preparedRelative, bytesFor<uint16_t>(uint64_t{g.attentionQueryHeads} * 16), 0, false},
      {&InklingScratch::attention, bytesFor<uint16_t>(q), 0, false},
      {&InklingScratch::branch, bytesFor<uint16_t>(d), 0, false},
      {&InklingScratch::convolved, bytesFor<uint16_t>(d), 0, false},
      {&InklingScratch::hidden, bytesFor<float>(d), 0, false},
      {&InklingScratch::expertOutput, bytesFor<uint16_t>(d), 0, false},
      {&InklingScratch::zeros, bytesFor<uint16_t>(d), 0, false},
      {&InklingScratch::workspace, ops::PagedAttention::inklingWorkspaceBytes(1, splits), 0, false},
  };
  if (decode) result.push_back({&InklingScratch::tape, bytesFor<uint16_t>(InklingSteps::kHistoryWidth), 0, false});
  return result;
}

uint64_t bytesOf(const Tensor &tensor, uint64_t rows) noexcept { return tensor.rowBytes * rows + tensor.fixedBytes; }

struct Sizes final {
  uint64_t prefill = 0;
  uint64_t decode = 0;
};

Sizes sizes(const QwenTargetGeometry &g, uint32_t chunks) {
  Sizes result;
  for (const Tensor &tensor : tensors(g, false))
    result.prefill += alignUp(bytesOf(tensor, kChunkRows)) * (tensor.carried ? chunks : 1);
  for (const Tensor &tensor : tensors(g, true)) result.decode += alignUp(bytesOf(tensor, kStepRows));
  return result;
}

} // namespace

InklingSteps::InklingSteps(const QwenTargetGeometry &geometry, metal::MetalBackend &backend, uint32_t prefillChunks)
    : geometry_(geometry), backend_(backend), prefillChunks_(prefillChunks) {
  if (!prefillChunks_) throw std::invalid_argument("invalid Inkling step geometry");
  const Sizes bytes = sizes(geometry_, prefillChunks_);
  prefill_ = backend_.allocateBuffer(bytes.prefill, metal::BufferStorage::Shared, "inkling-prefill-scratch");
  decode_ = backend_.allocateBuffer(bytes.decode, metal::BufferStorage::Shared, "inkling-decode-scratch");
  uint64_t cursor = 0;
  for (const Tensor &tensor : tensors(geometry_, false)) {
    prefillTensors_.push_back({tensor.field, tensor.rowBytes, tensor.fixedBytes, tensor.carried, cursor});
    cursor += alignUp(bytesOf(tensor, kChunkRows)) * (tensor.carried ? prefillChunks_ : 1);
  }
  cursor = 0;
  for (const Tensor &tensor : tensors(geometry_, true)) {
    decodeTensors_.push_back({tensor.field, tensor.rowBytes, tensor.fixedBytes, tensor.carried, cursor});
    cursor += alignUp(bytesOf(tensor, kStepRows));
  }
  // The rows the MoE adds a block's output to.
  for (const metal::MetalBuffer &zeros : {prefillScratch(0).zeros, verifyScratch().zeros})
    std::memset(zeros.contents(), 0, zeros.sizeBytes());
}

QwenTargetScratchBytes InklingSteps::scratchBytes(const QwenTargetGeometry &geometry, uint32_t prefillChunks) {
  const Sizes bytes = sizes(geometry, prefillChunks);
  return {bytes.prefill, bytes.decode};
}

QwenTargetScratchBytes InklingSteps::allocatedBytes() const noexcept {
  return {prefill_.sizeBytes(), decode_.sizeBytes()};
}

InklingScratch InklingSteps::view(const metal::MetalBuffer &buffer, const std::vector<Placed> &tensors, uint32_t rows,
                                  uint32_t chunk, uint32_t splits) const {
  InklingScratch scratch{.workspaceSplits = splits};
  for (const Placed &tensor : tensors) {
    const uint64_t bytes = tensor.rowBytes * rows + tensor.fixedBytes;
    scratch.*tensor.field = backend_.view(buffer, tensor.offset + (tensor.carried ? chunk * alignUp(bytes) : 0), bytes);
  }
  return scratch;
}

InklingScratch InklingSteps::prefillScratch(uint32_t index) const {
  if (index >= prefillChunks_) throw std::out_of_range("Inkling prefill chunk");
  return view(prefill_, prefillTensors_, kChunkRows, index, 1);
}

InklingScratch InklingSteps::verifyScratch() const { return view(decode_, decodeTensors_, kStepRows, 0, kDecodeSplits); }

} // namespace splash::model
