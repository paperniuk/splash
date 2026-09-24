#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ops/PagedAttention.hpp"
#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace splash::kv;

namespace {

constexpr uint32_t kQueryHeads = 24;
constexpr uint32_t kQueryHeadsPerKvHead = 6;
constexpr std::string_view kChunkedPrefillStorePipeline =
    "prefill_attention_q8_store";
constexpr std::string_view kPrefillAttentionSplitPipeline =
    "prefill_attention_q8_split";
constexpr std::string_view kPrefillAttentionRegisterSplitPipeline =
    "prefill_attention_q8_split_sgf";
constexpr std::string_view kPrefillAttentionReducePipeline =
    "prefill_attention_q8_reduce";
// The MPP split or the Apple7/8 register split, with the shared reduce.
struct AttentionPipelines {
  id<MTLComputePipelineState> split;
  id<MTLComputePipelineState> reduce;
  splash::ops::AttentionTile tile = splash::ops::AttentionTile::Mpp;
};

uint64_t attentionIndex(uint32_t stride, uint32_t head, uint32_t row,
                        uint32_t dimension) {
  uint32_t kvHead = head / kQueryHeadsPerKvHead;
  uint32_t localHead = head % kQueryHeadsPerKvHead;
  return ((uint64_t{kvHead} * stride + row) * kQueryHeadsPerKvHead +
          localHead) *
             kHeadDimension +
         dimension;
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:std::max<uint64_t>(bytes, 1)
                          options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal buffer allocation failed");
  std::memset(result.contents, 0, result.length);
  return result;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         const char *name) {
  id<MTLFunction> function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!function)
    throw std::runtime_error(std::string("missing Metal kernel: ") + name);
  NSError *error = nil;
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!result)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return result;
}

void finish(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::string message = "Metal command failed";
    if (command.error) {
      message += ": ";
      message += command.error.localizedDescription.UTF8String;
    }
    throw std::runtime_error(message);
  }
}

float keyPattern(uint32_t token, uint32_t head, uint32_t dimension,
                 uint32_t variant = 0) {
  int32_t centered = int32_t((uint64_t{token + 13 * variant} * 37 +
                               head * 101 + dimension * 17 +
                               uint64_t{token + variant} * dimension * 3) %
                              2003) -
                     1001;
  return float(centered) / 2002.0f;
}

float valuePattern(uint32_t token, uint32_t head, uint32_t dimension,
                   uint32_t variant = 0) {
  int32_t centered = int32_t((uint64_t{token + 17 * variant} * 53 +
                               head * 79 + dimension * 29 +
                               uint64_t{token + variant} * dimension * 5) %
                              2011) -
                     1005;
  return float(centered) / 1005.0f;
}

float queryPattern(uint32_t row, uint32_t head, uint32_t dimension) {
  int32_t centered = int32_t((uint64_t{row} * 43 + head * 67 +
                               dimension * 11 +
                               uint64_t{head} * dimension * 7) %
                              1019) -
                     509;
  return float(centered) / 2036.0f;
}

struct Case {
  Q8ChunkedPrefillParams params;
  std::vector<uint32_t> pageTable;
  id<MTLBuffer> pageTableBuffer;
  id<MTLBuffer> q8Keys;
  id<MTLBuffer> keyScales;
  id<MTLBuffer> q8Values;
  id<MTLBuffer> valueScales;
  id<MTLBuffer> chunkKeys;
  id<MTLBuffer> chunkValues;
  id<MTLBuffer> queries;
  id<MTLBuffer> output;
  id<MTLBuffer> partials;
  id<MTLBuffer> statistics;
};

uint64_t keyChunkIndex(uint32_t stride, uint32_t head, uint32_t token,
                       uint32_t dimension) {
  return (uint64_t{head} * stride + token) * kHeadDimension + dimension;
}

uint64_t valueChunkIndex(uint32_t stride, uint32_t head, uint32_t token,
                         uint32_t dimension) {
  return (uint64_t{head} * kHeadDimension + dimension) * stride + token;
}

Case makeCase(id<MTLDevice> device, uint32_t committed, uint32_t chunk,
              uint32_t stride) {
  Case result;
  uint32_t pages = (committed + chunk + kPageTokens - 1) / kPageTokens;
  uint32_t physicalPages = pages * 2 + 1;
  result.params = {committed, chunk, stride, pages, physicalPages, 0, 0, 0};
  require(chunkedPrefillValid(result.params), "invalid generated params");
  result.pageTable.resize(pages);
  for (uint32_t logical = 0; logical < pages; ++logical)
    result.pageTable[logical] = logical * 2 + 1;
  require(chunkedPrefillPageTableInRange(result.params, result.pageTable),
          "invalid generated page table");
  result.pageTableBuffer = makeBuffer(device, pages * sizeof(uint32_t));
  std::memcpy(result.pageTableBuffer.contents, result.pageTable.data(),
              pages * sizeof(uint32_t));
  result.q8Keys =
      makeBuffer(device, uint64_t{physicalPages} * kKeyDataBytesPerLayerPage);
  result.keyScales = makeBuffer(
      device, uint64_t{physicalPages} * kKeyScaleBytesPerLayerPage);
  result.q8Values = makeBuffer(
      device, uint64_t{physicalPages} * kValueDataBytesPerLayerPage);
  result.valueScales = makeBuffer(
      device, uint64_t{physicalPages} * kValueScaleBytesPerLayerPage);
  uint64_t chunkElements = uint64_t{kKvHeads} * stride * kHeadDimension;
  result.chunkKeys = makeBuffer(device, chunkElements * sizeof(BFloat16Bits));
  result.chunkValues =
      makeBuffer(device, chunkElements * sizeof(BFloat16Bits));
  uint64_t queryElements = uint64_t{kQueryHeads} * stride * kHeadDimension;
  result.queries = makeBuffer(device, queryElements * sizeof(BFloat16Bits));
  result.output = makeBuffer(device, queryElements * sizeof(BFloat16Bits));
  const auto workspace = splash::ops::PagedAttention::prefillWorkspace(
      chunk, kQueryHeads, {1, kKvHeads, kHeadDimension},
      {splash::ops::PrefillSplitMultiplier::Two});
  result.partials = makeBuffer(device, workspace.partialsBytes);
  result.statistics = makeBuffer(device, workspace.statisticsBytes);
  return result;
}

void fillHistory(Case &data) {
  auto *keys = static_cast<int8_t *>(data.q8Keys.contents);
  auto *keyScales = static_cast<float *>(data.keyScales.contents);
  auto *values = static_cast<int8_t *>(data.q8Values.contents);
  auto *valueScales = static_cast<float *>(data.valueScales.contents);
  std::vector<float> pageKeys;
  std::vector<float> pageValues;
  auto quantized = std::make_unique<Q8LayerPage>();
  uint32_t committedPages =
      (data.params.committed_tokens + kPageTokens - 1) / kPageTokens;
  for (uint32_t logicalPage = 0; logicalPage < committedPages; ++logicalPage) {
    uint32_t valid = std::min(kPageTokens,
                              data.params.committed_tokens -
                                  logicalPage * kPageTokens);
    pageKeys.assign(uint64_t{valid} * kKvHeads * kHeadDimension, 0.0f);
    pageValues.assign(uint64_t{valid} * kKvHeads * kHeadDimension, 0.0f);
    for (uint32_t token = 0; token < valid; ++token) {
      uint32_t global = logicalPage * kPageTokens + token;
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        for (uint32_t dimension = 0; dimension < kHeadDimension;
             ++dimension) {
          uint64_t index = logicalIndex(token, head, dimension);
          pageKeys[index] = bfloat16ToFloat(
              floatToBFloat16(keyPattern(global, head, dimension)));
          pageValues[index] = bfloat16ToFloat(
              floatToBFloat16(valuePattern(global, head, dimension)));
        }
      }
    }
    quantizeLayerPage(pageKeys, pageValues, valid, *quantized);
    uint32_t physical = data.pageTable[logicalPage];
    std::copy(quantized->keys.begin(), quantized->keys.end(),
              keys + uint64_t{physical} * kElementsPerLayerPage);
    std::copy(quantized->keyScales.begin(), quantized->keyScales.end(),
              keyScales +
                  uint64_t{physical} * kScalesPerTensorLayerPage);
    std::copy(quantized->values.begin(), quantized->values.end(),
              values + uint64_t{physical} * kElementsPerLayerPage);
    std::copy(quantized->valueScales.begin(), quantized->valueScales.end(),
              valueScales +
                  uint64_t{physical} * kScalesPerTensorLayerPage);
  }
}

void fillCurrent(Case &data, uint32_t variant = 0, uint32_t queryOffset = 0) {
  auto *keys = static_cast<BFloat16Bits *>(data.chunkKeys.contents);
  auto *values = static_cast<BFloat16Bits *>(data.chunkValues.contents);
  auto *queries = static_cast<BFloat16Bits *>(data.queries.contents);
  std::memset(keys, 0, data.chunkKeys.length);
  std::memset(values, 0, data.chunkValues.length);
  for (uint32_t head = 0; head < kKvHeads; ++head) {
    for (uint32_t token = 0; token < data.params.chunk_tokens; ++token) {
      uint32_t global = data.params.committed_tokens + token;
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        keys[keyChunkIndex(data.params.chunk_stride, head, token, dimension)] =
            floatToBFloat16(keyPattern(global, head, dimension, variant));
        values[valueChunkIndex(data.params.chunk_stride, head, token,
                               dimension)] =
            floatToBFloat16(valuePattern(global, head, dimension, variant));
      }
    }
  }
  for (uint32_t head = 0; head < kQueryHeads; ++head) {
    for (uint32_t token = 0; token < data.params.chunk_tokens; ++token) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        queries[attentionIndex(data.params.chunk_stride, head, token,
                               dimension)] =
            floatToBFloat16(queryPattern(queryOffset + token, head, dimension));
      }
    }
  }
}

void encodeStore(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> pipeline, const Case &data) {
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:data.chunkKeys offset:0 atIndex:0];
  [encoder setBuffer:data.chunkValues offset:0 atIndex:1];
  [encoder setBuffer:data.q8Keys offset:0 atIndex:2];
  [encoder setBuffer:data.keyScales offset:0 atIndex:3];
  [encoder setBuffer:data.q8Values offset:0 atIndex:4];
  [encoder setBuffer:data.valueScales offset:0 atIndex:5];
  [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:6];
  [encoder setBytes:&data.params length:sizeof(data.params) atIndex:7];
  [encoder dispatchThreadgroups:MTLSizeMake(
                                     2 * kKvHeads * data.params.chunk_tokens,
                                     1, 1)
             threadsPerThreadgroup:MTLSizeMake(kHeadDimension, 1, 1)];
}

void encodeAttention(id<MTLComputeCommandEncoder> encoder,
                     const AttentionPipelines &pipelines, const Case &data,
                     splash::ops::PrefillAttentionConfig config = {},
                     const Q8PrefillAttentionParams *overrideParams = nullptr) {
  config.tile = pipelines.tile;
  const auto plan = splash::ops::PagedAttention::prefillPlan(
      data.params.chunk_tokens, kQueryHeads, {1, kKvHeads, kHeadDimension},
      data.params.committed_tokens, config);
  require(plan.workspace.partialsBytes <= data.partials.length &&
              plan.workspace.statisticsBytes <= data.statistics.length,
          "attention splits exceed the shared arena");
  const Q8PrefillAttentionParams params = overrideParams ? *overrideParams :
      Q8PrefillAttentionParams{data.params.committed_tokens, data.params.chunk_tokens,
                              data.params.chunk_stride, data.params.page_table_entries,
                              data.params.physical_page_count, plan.splits, 0, 0};
  [encoder setComputePipelineState:pipelines.split];
  [encoder setBuffer:data.queries offset:0 atIndex:0];
  [encoder setBuffer:data.q8Keys offset:0 atIndex:1];
  [encoder setBuffer:data.keyScales offset:0 atIndex:2];
  [encoder setBuffer:data.q8Values offset:0 atIndex:3];
  [encoder setBuffer:data.valueScales offset:0 atIndex:4];
  [encoder setBuffer:data.partials offset:0 atIndex:5];
  [encoder setBuffer:data.statistics offset:0 atIndex:6];
  [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:7];
  [encoder setBytes:&params length:sizeof(params) atIndex:8];
  require(plan.splitThreads.x <= pipelines.split.maxTotalThreadsPerThreadgroup,
          "prefill split threadgroup exceeds its pipeline");
  [encoder dispatchThreadgroups:MTLSizeMake(plan.splitGroups.x, plan.splitGroups.y,
                                           plan.splitGroups.z)
             threadsPerThreadgroup:MTLSizeMake(plan.splitThreads.x, plan.splitThreads.y,
                                               plan.splitThreads.z)];
  [encoder setComputePipelineState:pipelines.reduce];
  [encoder setBuffer:data.partials offset:0 atIndex:0];
  [encoder setBuffer:data.statistics offset:0 atIndex:1];
  [encoder setBuffer:data.output offset:0 atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(plan.reduceGroups.x, plan.reduceGroups.y,
                                           plan.reduceGroups.z)
             threadsPerThreadgroup:MTLSizeMake(kHeadDimension, 1, 1)];
}

float loadKey(const Case &data, uint32_t token, uint32_t head,
              uint32_t dimension) {
  uint32_t physical = data.pageTable[token / kPageTokens];
  uint32_t pageToken = token % kPageTokens;
  float scale = static_cast<const float *>(data.keyScales.contents)[
      uint64_t{physical} * kScalesPerTensorLayerPage +
      keyScaleIndex(head, pageToken)];
  return float(static_cast<const int8_t *>(data.q8Keys.contents)[
             uint64_t{physical} * kElementsPerLayerPage +
             keyDataIndex(head, pageToken, dimension)]) *
         scale;
}

float loadValue(const Case &data, uint32_t token, uint32_t head,
                uint32_t dimension) {
  uint32_t physical = data.pageTable[token / kPageTokens];
  uint32_t pageToken = token % kPageTokens;
  float scale = static_cast<const float *>(data.valueScales.contents)[
      uint64_t{physical} * kScalesPerTensorLayerPage +
      valueScaleIndex(head, pageToken)];
  return float(static_cast<const int8_t *>(data.q8Values.contents)[
             uint64_t{physical} * kElementsPerLayerPage +
             valueDataIndex(head, pageToken, dimension)]) *
         scale;
}

std::vector<BFloat16Bits> cpuAttention(const Case &data,
                                      std::span<const uint32_t> rows,
                                      std::span<const uint32_t> heads) {
  std::vector<BFloat16Bits> output(
      uint64_t{kQueryHeads} * data.params.chunk_stride * kHeadDimension);
  const auto *queries =
      static_cast<const BFloat16Bits *>(data.queries.contents);
  for (uint32_t queryHead : heads) {
    uint32_t kvHead = queryHead / kQueryHeadsPerKvHead;
    for (uint32_t row : rows) {
      uint32_t visible = data.params.committed_tokens + row + 1;
      std::vector<float> scores(visible);
      float maximum = -std::numeric_limits<float>::infinity();
      for (uint32_t token = 0; token < visible; ++token) {
        float score = 0.0f;
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          float query = bfloat16ToFloat(
              queries[attentionIndex(data.params.chunk_stride, queryHead, row,
                                     dimension)]);
          score += query * loadKey(data, token, kvHead, dimension);
        }
        scores[token] = score * 0.0625f;
        maximum = std::max(maximum, scores[token]);
      }
      float denominator = 0.0f;
      for (float &score : scores) {
        const float probability = std::exp(score - maximum);
        denominator += probability;
        // The BF16 probability is identical for every output dimension.
        // Preserve each dimension's token accumulation order while avoiding
        // another exponential evaluation for every dimension.
        score = bfloat16ToFloat(floatToBFloat16(probability));
      }
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        float value = 0.0f;
        for (uint32_t token = 0; token < visible; ++token)
          value += scores[token] * loadValue(data, token, kvHead, dimension);
        output[attentionIndex(data.params.chunk_stride, queryHead, row,
                              dimension)] =
            floatToBFloat16(value / denominator);
      }
    }
  }
  return output;
}

void validateAttention(const Case &data,
                       std::span<const uint32_t> rows = {},
                       std::span<const uint32_t> heads = {}) {
  std::vector<uint32_t> allRows;
  std::array<uint32_t, kQueryHeads> allHeads{};
  if (rows.empty()) {
    allRows.resize(data.params.chunk_tokens);
    std::iota(allRows.begin(), allRows.end(), 0U);
    rows = allRows;
  }
  if (heads.empty()) {
    std::iota(allHeads.begin(), allHeads.end(), 0U);
    heads = allHeads;
  }
  std::vector<BFloat16Bits> expected = cpuAttention(data, rows, heads);
  const auto *actual = static_cast<const BFloat16Bits *>(data.output.contents);
  double dot = 0.0;
  double actualSquared = 0.0;
  double expectedSquared = 0.0;
  float maximumAbsolute = 0.0f;
  std::string worst;
  for (uint32_t head : heads) {
    for (uint32_t row : rows) {
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        uint64_t index = attentionIndex(data.params.chunk_stride, head, row,
                                        dimension);
        float observed = bfloat16ToFloat(actual[index]);
        float reference = bfloat16ToFloat(expected[index]);
        if (!std::isfinite(observed)) {
          throw std::runtime_error(
              "non-finite attention output: head=" +
              std::to_string(head) + " row=" + std::to_string(row) +
              " dimension=" + std::to_string(dimension));
        }
        if (std::abs(observed - reference) > maximumAbsolute) {
          maximumAbsolute = std::abs(observed - reference);
          worst = " at head=" + std::to_string(head) +
                  " row=" + std::to_string(row) +
                  " dimension=" + std::to_string(dimension) + " (" +
                  std::to_string(observed) + " vs " +
                  std::to_string(reference) + ")";
        }
        dot += double(observed) * reference;
        actualSquared += double(observed) * observed;
        expectedSquared += double(reference) * reference;
      }
    }
  }
  double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  if (!(cosine > 0.999) || !(maximumAbsolute < 0.035f)) {
    throw std::runtime_error(
        "prefill attention mismatch (committed=" +
        std::to_string(data.params.committed_tokens) +
        " chunk=" + std::to_string(data.params.chunk_tokens) +
        "): cosine=" + std::to_string(cosine) +
        " maximum_absolute=" + std::to_string(maximumAbsolute) + worst);
  }
}

std::vector<uint32_t> chunkReferenceRows(const Case &data) {
  const uint32_t count = data.params.chunk_tokens;
  const uint32_t committed = data.params.committed_tokens;
  std::vector<uint32_t> rows{0, count / 2, count - 1};
  const auto add = [&](int64_t row) {
    if (row >= 0 && row < count)
      rows.push_back(static_cast<uint32_t>(row));
  };
  // Both sides of a query-tile boundary and the first/last Page32 causal
  // boundaries in this chunk. Keep long-history scalar work bounded to nine
  // rows; the caller still checks every active output and inactive guard.
  add(kQ8PrefillAttentionTileRows - 1);
  add(kQ8PrefillAttentionTileRows);
  const uint32_t firstPageEnd = kPageTokens - 1 - committed % kPageTokens;
  add(firstPageEnd);
  add(firstPageEnd + 1);
  const int64_t lastPageStart =
      int64_t{(committed + count - 1) / kPageTokens * kPageTokens} - committed;
  add(lastPageStart - 1);
  add(lastPageStart);
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

std::vector<int8_t> expectedRow(const Case &data, bool valueTensor,
                                uint32_t chunkToken, uint32_t variant,
                                float &scale) {
  std::vector<float> values(kHeadDimension);
  std::vector<int8_t> quantized(kHeadDimension);
  uint32_t global = data.params.committed_tokens + chunkToken;
  uint32_t head = 0;
  float maximum = 0.0f;
  for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
    float source = valueTensor
                       ? valuePattern(global, head, dimension, variant)
                       : keyPattern(global, head, dimension, variant);
    values[dimension] =
        bfloat16ToFloat(floatToBFloat16(source));
    maximum = std::max(maximum, std::abs(values[dimension]));
  }
  scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
  for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
    quantized[dimension] = maximum == 0.0f
                              ? 0
                              : static_cast<int8_t>(std::clamp(
                                    int(std::nearbyint(values[dimension] *
                                                       127.0f / maximum)),
                                    -127, 127));
  }
  return quantized;
}

void validateStoredRow(const Case &data, bool valueTensor,
                       uint32_t chunkToken, uint32_t variant) {
  float expectedScale = 0.0f;
  std::vector<int8_t> expected =
      expectedRow(data, valueTensor, chunkToken, variant, expectedScale);
  uint32_t logical = data.params.committed_tokens + chunkToken;
  uint32_t physical = data.pageTable[logical / kPageTokens];
  uint32_t pageToken = logical % kPageTokens;
  const auto *scales = static_cast<const float *>(
      valueTensor ? data.valueScales.contents : data.keyScales.contents);
  float observedScale = scales[
      uint64_t{physical} * kScalesPerTensorLayerPage +
      (valueTensor ? valueScaleIndex(0, pageToken)
                   : keyScaleIndex(0, pageToken))];
  require(std::abs(observedScale - expectedScale) <= 2e-7f,
          "stored Q8 row scale differs");
  const auto *stored = static_cast<const int8_t *>(
      valueTensor ? data.q8Values.contents : data.q8Keys.contents);
  for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
    uint64_t index = uint64_t{physical} * kElementsPerLayerPage +
                     (valueTensor
                          ? valueDataIndex(0, pageToken, dimension)
                          : keyDataIndex(0, pageToken, dimension));
    require(stored[index] == expected[dimension],
            "stored Q8 row payload differs");
  }
}

void testAttentionAndDirectStore(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> store,
                                 const AttentionPipelines &attention) {
  Case data = makeCase(device, 133, 32, 32);
  fillHistory(data);
  fillCurrent(data);
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  encodeStore(encoder, store, data);
  encodeAttention(encoder, attention, data);
  [encoder endEncoding];
  finish(command);
  validateStoredRow(data, false, 0, 0);
  validateStoredRow(data, true, 31, 0);
  validateAttention(data);
}

// A chunk ending mid-page over a long history, several query tiles over a
// short history, a ragged tile over three balanced splits, and a long chunk
// whose baseline uses one balanced history split.
void testAttentionGeometries(id<MTLDevice> device, id<MTLCommandQueue> queue,
                             id<MTLComputePipelineState> store,
                             const AttentionPipelines &attention) {
  struct Geometry {
    uint32_t committed;
    uint32_t chunk;
    uint32_t stride;
  };
  for (const Geometry &geometry : {Geometry{70, 77, 96}, Geometry{2500, 32, 32},
                                   Geometry{17, 96, 96}, Geometry{100, 77, 96},
                                   Geometry{5, 1056, 1056}}) {
    Case data = makeCase(device, geometry.committed, geometry.chunk,
                         geometry.stride);
    fillHistory(data);
    fillCurrent(data);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encodeStore(encoder, store, data);
    encodeAttention(encoder, attention, data);
    [encoder endEncoding];
    finish(command);
    validateAttention(data);
  }
}

// Keep the logical query, stored Q8 representation and causal row fixed while
// checking each split multiplier and unaligned host chunk against a bounded
// scalar Q8 oracle. Different partitionings can legitimately round differently.
void testChunkAndSplitReference(id<MTLDevice> device, id<MTLCommandQueue> queue,
                               id<MTLComputePipelineState> store,
                               const AttentionPipelines &attention) {
  using namespace splash::ops;
  constexpr uint32_t history = 4093, rows = 2048;
  Case data = makeCase(device, history, rows, rows);
  fillHistory(data);
  const std::array whole{rows};
  const std::array fragmented{1U, 31U, 33U, 511U, 1472U};
  std::array<uint32_t, 2 * kKvHeads> referenceHeads{};
  for (uint32_t kvHead = 0; kvHead < kKvHeads; ++kvHead) {
    referenceHeads[2 * kvHead] = kvHead * kQueryHeadsPerKvHead;
    referenceHeads[2 * kvHead + 1] = (kvHead + 1) * kQueryHeadsPerKvHead - 1;
  }
  for (const auto config :
       {PrefillAttentionConfig{PrefillSplitMultiplier::One},
        PrefillAttentionConfig{PrefillSplitMultiplier::Two}})
    for (std::span<const uint32_t> chunks :
         {std::span<const uint32_t>(whole), std::span<const uint32_t>(fragmented)}) {
      uint32_t offset = 0;
      for (uint32_t chunk : chunks) {
        data.params.committed_tokens = history + offset;
        data.params.chunk_tokens = chunk;
        fillCurrent(data, 0, offset);
        std::memset(data.output.contents, 0xff, data.output.length);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        encodeStore(encoder, store, data);
        encodeAttention(encoder, attention, data, config);
        [encoder endEncoding];
        finish(command);
        const auto referenceRows = chunkReferenceRows(data);
        validateAttention(data, referenceRows, referenceHeads);
        const auto *output = static_cast<const BFloat16Bits *>(data.output.contents);
        const std::vector<BFloat16Bits> first(
            output, output + data.output.length / sizeof(BFloat16Bits));
        const auto validateCoverage = [&] {
          for (uint32_t head = 0; head < kQueryHeads; ++head)
            for (uint32_t row = 0; row < rows; ++row)
              for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
                const auto value = output[attentionIndex(rows, head, row, dimension)];
                if (row < chunk) {
                  require(std::isfinite(bfloat16ToFloat(value)), "chunked output is nonfinite");
                } else {
                  require(value == 0xffff, "prefill split overwrote an inactive output row");
                }
              }
        };
        validateCoverage();
        std::memset(data.output.contents, 0xff, data.output.length);
        command = [queue commandBuffer];
        encoder = [command computeCommandEncoder];
        encodeAttention(encoder, attention, data, config);
        [encoder endEncoding];
        finish(command);
        validateCoverage();
        size_t differences = 0;
        float maximumError = 0;
        uint32_t worstRow = 0, worstHead = 0;
        double dot = 0, firstSquared = 0, repeatedSquared = 0;
        for (uint32_t head = 0; head < kQueryHeads; ++head)
          for (uint32_t row = 0; row < chunk; ++row)
            for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
              const auto index = attentionIndex(rows, head, row, dimension);
              const float left = bfloat16ToFloat(first[index]);
              const float right = bfloat16ToFloat(output[index]);
              differences += first[index] != output[index];
              if (const float error = std::abs(left - right); error > maximumError) {
                maximumError = error;
                worstRow = row;
                worstHead = head;
              }
              dot += double(left) * right;
              firstSquared += double(left) * left;
              repeatedSquared += double(right) * right;
            }
        const double cosine = dot / std::sqrt(firstSquared * repeatedSquared);
        require(maximumError < 0.035f && cosine > 0.999,
                "repeated attention exceeds numerical tolerance");
        if (differences) {
          std::cout << "attention repeat: history=" << data.params.committed_tokens
                    << " chunk=" << chunk << " split_multiplier="
                    << static_cast<uint32_t>(config.splitMultiplier)
                    << " differences=" << differences
                    << " maximum_absolute_error=" << maximumError
                    << " cosine=" << cosine << '\n';
          // Expand the independent reference sample to include the worst
          // difference. Keep its aggregate cosine domain: a near-zero vector
          // alone can have low cosine despite negligible absolute error.
          std::vector<uint32_t> checkedRows(referenceRows);
          std::vector<uint32_t> checkedHeads(referenceHeads.begin(), referenceHeads.end());
          checkedRows.push_back(worstRow);
          checkedHeads.push_back(worstHead);
          for (auto *indices : {&checkedRows, &checkedHeads}) {
            std::sort(indices->begin(), indices->end());
            indices->erase(std::unique(indices->begin(), indices->end()), indices->end());
          }
          validateAttention(data, checkedRows, checkedHeads);
        }
        offset += chunk;
      }
      require(offset == rows, "chunk reference test changed total logical rows");
      // All even physical pages are unleased: direct store must preserve them.
      for (uint32_t page = 0; page < data.params.physical_page_count; page += 2) {
        const auto checkZero = [&](id<MTLBuffer> buffer, uint64_t pageBytes) {
          const auto *bytes = static_cast<const uint8_t *>(buffer.contents) + page * pageBytes;
          require(std::all_of(bytes, bytes + pageBytes, [](uint8_t value) { return value == 0; }),
                  "noncontiguous unleased Q8 page was overwritten");
        };
        checkZero(data.q8Keys, kKeyDataBytesPerLayerPage);
        checkZero(data.q8Values, kValueDataBytesPerLayerPage);
        checkZero(data.keyScales, kKeyScaleBytesPerLayerPage);
        checkZero(data.valueScales, kValueScaleBytesPerLayerPage);
      }
    }
}

void testInvalidAttentionParams(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                const AttentionPipelines &attention) {
  Case data = makeCase(device, 33, 17, 32);
  const auto plan = splash::ops::PagedAttention::prefillPlan(
      data.params.chunk_tokens, kQueryHeads, {1, kKvHeads, kHeadDimension},
      data.params.committed_tokens);
  const Q8PrefillAttentionParams valid{
      data.params.committed_tokens, data.params.chunk_tokens,
      data.params.chunk_stride, data.params.page_table_entries,
      data.params.physical_page_count, plan.splits, 0, 0};
  for (uint32_t invalidField = 0; invalidField < 4; ++invalidField) {
    auto params = valid;
    if (invalidField == 0) params.split_count = 0;
    if (invalidField == 1) params.split_count = 33;
    if (invalidField == 2) params.reserved0 = 1;
    if (invalidField == 3) params.reserved1 = 1;
    for (id<MTLBuffer> buffer : {data.partials, data.statistics, data.output})
      std::memset(buffer.contents, 0xa5, buffer.length);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encodeAttention(encoder, attention, data, {}, &params);
    [encoder endEncoding];
    finish(command);
    for (id<MTLBuffer> buffer : {data.partials, data.statistics, data.output}) {
      const auto *bytes = static_cast<const uint8_t *>(buffer.contents);
      require(std::all_of(bytes, bytes + buffer.length,
                          [](uint8_t value) { return value == 0xa5; }),
              "invalid prefill split count or reserved field wrote scratch/output");
    }
  }
}

void testCommitIndexOverwrite(id<MTLDevice> device,
                              id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> store) {
  Case data = makeCase(device, 127, 8, 32);
  fillCurrent(data, 1);
  id<MTLCommandBuffer> first = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [first computeCommandEncoder];
  encodeStore(encoder, store, data);
  [encoder endEncoding];
  finish(first);

  // Accept only three candidates. The logical commit index moves 127 -> 130;
  // the five rejected physical rows remain invisible.
  data.params.committed_tokens = 130;
  data.params.chunk_tokens = 5;
  fillCurrent(data, 2);
  id<MTLCommandBuffer> second = [queue commandBuffer];
  encoder = [second computeCommandEncoder];
  encodeStore(encoder, store, data);
  [encoder endEncoding];
  finish(second);

  // The accepted rows 127..129 still contain variant 1. Temporarily express
  // their original command coordinates for the common row oracle.
  Q8ChunkedPrefillParams secondParams = data.params;
  data.params.committed_tokens = 127;
  data.params.chunk_tokens = 8;
  for (uint32_t token = 0; token < 3; ++token) {
    validateStoredRow(data, false, token, 1);
    validateStoredRow(data, true, token, 1);
  }
  data.params = secondParams;
  for (uint32_t token = 0; token < 5; ++token) {
    validateStoredRow(data, false, token, 2);
    validateStoredRow(data, true, token, 2);
  }
}

void testBatchedVerifyStore(id<MTLDevice> device, id<MTLCommandQueue> queue,
                            id<MTLComputePipelineState> store) {
  constexpr uint32_t lanes = 4;
  constexpr uint32_t committed = 31;
  constexpr uint32_t rows = 8;
  constexpr uint32_t stride = 32;
  constexpr uint32_t physicalPages = lanes * 2;
  constexpr uint64_t laneElements =
      uint64_t{kKvHeads} * stride * kHeadDimension;

  id<MTLBuffer> chunkKeys =
      makeBuffer(device, lanes * laneElements * sizeof(BFloat16Bits));
  id<MTLBuffer> chunkValues =
      makeBuffer(device, lanes * laneElements * sizeof(BFloat16Bits));
  id<MTLBuffer> q8Keys = makeBuffer(
      device, uint64_t{physicalPages} * kKeyDataBytesPerLayerPage);
  id<MTLBuffer> keyScales = makeBuffer(
      device, uint64_t{physicalPages} * kKeyScaleBytesPerLayerPage);
  id<MTLBuffer> q8Values = makeBuffer(
      device, uint64_t{physicalPages} * kValueDataBytesPerLayerPage);
  id<MTLBuffer> valueScales = makeBuffer(
      device, uint64_t{physicalPages} * kValueScaleBytesPerLayerPage);

  std::array<Q8ChunkedPrefillParams, lanes> params{};
  std::array<std::array<uint32_t, 2>, lanes> tables{};
  std::array<id<MTLBuffer>, lanes> tableBuffers{};
  auto *keys = static_cast<BFloat16Bits *>(chunkKeys.contents);
  auto *values = static_cast<BFloat16Bits *>(chunkValues.contents);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params[lane] = {committed, rows, stride, 2, physicalPages, 0, 0, 0};
    tables[lane] = {lane * 2, lane * 2 + 1};
    tableBuffers[lane] = makeBuffer(device, 2 * sizeof(uint32_t));
    std::memcpy(tableBuffers[lane].contents, tables[lane].data(),
                2 * sizeof(uint32_t));
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      for (uint32_t token = 0; token < rows; ++token) {
        const uint32_t global = committed + token;
        for (uint32_t dimension = 0; dimension < kHeadDimension;
             ++dimension) {
          const uint64_t base = uint64_t{lane} * laneElements;
          keys[base + keyChunkIndex(stride, head, token, dimension)] =
              floatToBFloat16(
                  keyPattern(global, head, dimension, lane + 1));
          values[base + valueChunkIndex(stride, head, token, dimension)] =
              floatToBFloat16(
                  valuePattern(global, head, dimension, lane + 1));
        }
      }
    }
  }

  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:store];
  [encoder setBuffer:chunkKeys offset:0 atIndex:0];
  [encoder setBuffer:chunkValues offset:0 atIndex:1];
  [encoder setBuffer:q8Keys offset:0 atIndex:2];
  [encoder setBuffer:keyScales offset:0 atIndex:3];
  [encoder setBuffer:q8Values offset:0 atIndex:4];
  [encoder setBuffer:valueScales offset:0 atIndex:5];
  for (uint32_t lane = 0; lane < lanes; ++lane)
    [encoder setBuffer:tableBuffers[lane] offset:0 atIndex:6 + lane];
  [encoder setBytes:params.data() length:sizeof(params) atIndex:10];
  [encoder dispatchThreadgroups:MTLSizeMake(
                                     lanes * 2 * rows * kKvHeads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kHeadDimension, 1, 1)];
  [encoder endEncoding];
  finish(command);

  const auto *storedKeys = static_cast<const int8_t *>(q8Keys.contents);
  const auto *storedValues = static_cast<const int8_t *>(q8Values.contents);
  const auto *storedKeyScales =
      static_cast<const float *>(keyScales.contents);
  const auto *storedValueScales =
      static_cast<const float *>(valueScales.contents);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (uint32_t token : {0U, rows - 1}) {
      const uint32_t global = committed + token;
      const uint32_t page = tables[lane][global / kPageTokens];
      const uint32_t pageToken = global % kPageTokens;
      for (bool valueTensor : {false, true}) {
        float maximum = 0.0f;
        std::array<float, kHeadDimension> source{};
        for (uint32_t dimension = 0; dimension < kHeadDimension;
             ++dimension) {
          const float raw = valueTensor
                                ? valuePattern(global, 0, dimension, lane + 1)
                                : keyPattern(global, 0, dimension, lane + 1);
          source[dimension] = bfloat16ToFloat(floatToBFloat16(raw));
          maximum = std::max(maximum, std::abs(source[dimension]));
        }
        const float expectedScale = maximum / 127.0f;
        const float observedScale =
            (valueTensor ? storedValueScales : storedKeyScales)[
                uint64_t{page} * kScalesPerTensorLayerPage +
                (valueTensor ? valueScaleIndex(0, pageToken)
                             : keyScaleIndex(0, pageToken))];
        require(std::abs(expectedScale - observedScale) <= 2e-7f,
                "batched verify store scale differs");
        for (uint32_t dimension = 0; dimension < kHeadDimension;
             ++dimension) {
          const int8_t expected = static_cast<int8_t>(std::clamp(
              int(std::nearbyint(source[dimension] * 127.0f / maximum)),
              -127, 127));
          const uint64_t index =
              uint64_t{page} * kElementsPerLayerPage +
              (valueTensor ? valueDataIndex(0, pageToken, dimension)
                           : keyDataIndex(0, pageToken, dimension));
          require((valueTensor ? storedValues : storedKeys)[index] == expected,
                  "batched verify store payload differs");
        }
      }
    }
  }
}

void testContract() {
  static_assert(sizeof(Q8PrefillAttentionParams) == 32);
  static_assert(offsetof(Q8PrefillAttentionParams, committed_tokens) == 0);
  static_assert(offsetof(Q8PrefillAttentionParams, rows) == 4);
  static_assert(offsetof(Q8PrefillAttentionParams, chunk_stride) == 8);
  static_assert(offsetof(Q8PrefillAttentionParams, page_table_entries) == 12);
  static_assert(offsetof(Q8PrefillAttentionParams, physical_page_count) == 16);
  static_assert(offsetof(Q8PrefillAttentionParams, split_count) == 20);
  static_assert(offsetof(Q8PrefillAttentionParams, reserved0) == 24);
  static_assert(offsetof(Q8PrefillAttentionParams, reserved1) == 28);
  Q8ChunkedPrefillParams params{129, 8, 32, 5, 8, 0, 0, 0};
  require(chunkedPrefillValid(params),
          "partial committed page must be a valid direct-Q8 input");
  require(chunkedPrefillRequiredPages(params) == 5,
          "page plan must include speculative destination slots");
  params.page_table_entries = 4;
  require(chunkedPrefillValidationError(params) == "page_table_too_short",
          "short page table was accepted");

  Q8ChunkedPrefillParams finalCycle{
      splash::kv::kMaximumLogicalTokens - 1,
      splash::kv::kQ8VerifyMaximumRows,
      32,
      (splash::kv::kMaximumPhysicalTokens + kPageTokens - 1) /
          kPageTokens,
      (splash::kv::kMaximumPhysicalTokens + kPageTokens - 1) /
          kPageTokens,
      0,
      0,
      0};
  require(chunkedPrefillValid(finalCycle),
          "final fixed-eight verification rows exceeded physical KV scratch");
  ++finalCycle.committed_tokens;
  require(chunkedPrefillValidationError(finalCycle) == "context_out_of_range",
          "physical KV scratch exceeded its fixed seven-row allowance");
}

void run(const char *libraryPath) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device)
    throw std::runtime_error("Metal device unavailable");
  NSError *error = nil;
  NSURL *url = [NSURL
      fileURLWithPath:[NSString stringWithUTF8String:libraryPath]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLComputePipelineState> store =
      makePipeline(device, library, kChunkedPrefillStorePipeline.data());
  id<MTLComputePipelineState> reduce =
      makePipeline(device, library, kPrefillAttentionReducePipeline.data());
  id<MTLComputePipelineState> verifyStore =
      makePipeline(device, library, "verify_attention_q8_store");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  testContract();
  using splash::ops::AttentionTile;
  for (const auto [name, tile] :
       {std::pair{kPrefillAttentionSplitPipeline, AttentionTile::Mpp},
        std::pair{kPrefillAttentionRegisterSplitPipeline, AttentionTile::Register}}) {
    const AttentionPipelines attention{makePipeline(device, library, name.data()),
                                       reduce, tile};
    std::cout << "prefill attention tile: " << name << '\n';
    testAttentionAndDirectStore(device, queue, store, attention);
    testAttentionGeometries(device, queue, store, attention);
    testChunkAndSplitReference(device, queue, store, attention);
    testInvalidAttentionParams(device, queue, attention);
  }
  testCommitIndexOverwrite(device, queue, store);
  testBatchedVerifyStore(device, queue, verifyStore);
  std::cout << "q8_chunked_prefill_metal_test: ok\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::runtime_error(
            "usage: q8_chunked_prefill_metal_test <metallib>");
      run(argv[1]);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "q8_chunked_prefill_metal_test: " << error.what() << '\n';
      return 1;
    }
  }
}
