#include "DenseAttention.hpp"

#include "metal/abi/DenseAttention.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

// The bytes of `tensor` the kernel reaches over `rows` rows of `width`
// dimensions of every head.
void requireTensor(const HeadTensor &tensor, uint32_t heads, uint32_t rows, uint32_t width, const char *what) {
  const uint64_t last = uint64_t{heads - 1} * tensor.headStride + uint64_t{rows - 1} * tensor.rowStride + width;
  requireBytes(tensor.buffer, last * 2, what);
}

// The Apple7/8 kernel (shared/vision_mma.metal) stages keys and values in
// 16-byte runs of eight bf16, so their view offsets and strides must align.
constexpr uint32_t kMmaRunElements = 8;

void requireRuns(const HeadTensor &tensor, const char *what) {
  if (tensor.buffer.offsetBytes() % (kMmaRunElements * sizeof(uint16_t)))
    throw std::invalid_argument(std::string(what) + " view offset must be a multiple of 16 bytes");
  if (tensor.headStride % kMmaRunElements || tensor.rowStride % kMmaRunElements)
    throw std::invalid_argument(std::string(what) + " strides must be multiples of " +
                                std::to_string(kMmaRunElements) + " elements");
}

} // namespace

DenseAttention::DenseAttention(const DeviceCapabilities &device) noexcept : mma_(device.appleGpuFamily < 9) {}

uint32_t DenseAttention::paddedQueries(uint32_t rows) const noexcept {
  return roundUp(rows, mma_ ? SPLASH_DENSE_ATTENTION_MMA_ROWS : SPLASH_DENSE_ATTENTION_ROWS);
}

uint32_t DenseAttention::paddedKeys(uint32_t rows) noexcept { return roundUp(rows, SPLASH_DENSE_ATTENTION_KEYS); }

void DenseAttention::add(metal::CommandGraph &graph, const DenseAttentionTensors &tensors, uint32_t heads,
                         uint32_t dimensions, uint32_t rows, float scale) const {
  const uint32_t padded = splash_dense_attention_padded(dimensions);
  requireTensor(tensors.queries, heads, paddedQueries(rows), padded, "dense attention query");
  requireTensor(tensors.keys, heads, paddedKeys(rows), padded, "dense attention key");
  requireTensor(tensors.values, heads, paddedKeys(rows), dimensions, "dense attention value");
  requireTensor(tensors.output, heads, rows, dimensions, "dense attention output");
  if (mma_) {
    requireRuns(tensors.keys, "dense attention key");
    requireRuns(tensors.values, "dense attention value");
  }
  const DenseAttentionParams params{rows,
                                    scale,
                                    tensors.queries.headStride,
                                    tensors.keys.headStride,
                                    tensors.values.headStride,
                                    tensors.output.headStride,
                                    tensors.queries.rowStride,
                                    tensors.keys.rowStride,
                                    tensors.values.rowStride,
                                    tensors.output.rowStride};
  const uint32_t tileRows = mma_ ? SPLASH_DENSE_ATTENTION_MMA_ROWS : SPLASH_DENSE_ATTENTION_ROWS;
  const uint32_t threads = mma_ ? SPLASH_DENSE_ATTENTION_MMA_THREADS : SPLASH_DENSE_ATTENTION_THREADS;
  graph.add((mma_ ? "dense_attention_mma_" : "dense_attention_") + std::to_string(dimensions),
            {tensors.queries.buffer, tensors.keys.buffer, tensors.values.buffer, tensors.output.buffer}, params,
            {paddedQueries(rows) / tileRows, heads, 1}, {threads, 1, 1});
}

} // namespace splash::ops
