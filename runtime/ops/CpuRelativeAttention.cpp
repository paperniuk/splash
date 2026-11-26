#include "ops/CpuRelativeAttention.hpp"

#include "metal/abi/RelativeAttention.h"

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::ops {

namespace {

constexpr uint32_t kGroup = CpuRelativeAttention::kQueryHeads / CpuRelativeAttention::kKvHeads;
constexpr uint32_t kD = CpuRelativeAttention::kHeadDimension;
constexpr uint32_t kHeads = CpuRelativeAttention::kKvHeads;
constexpr uint32_t kStates = SPLASH_RELATIVE_ATTENTION_STATES;
constexpr uint32_t kExtent = SPLASH_RELATIVE_ATTENTION_EXTENT;
constexpr uint32_t kPage = kv::kPageTokens;
// A matrix tile is 32 fused rows by 32 keys; a block of keys is 16 pages, which
// a task packs once for its six tiles.
constexpr uint32_t kTile = 32;
constexpr uint32_t kBlockPages = 16;
constexpr uint32_t kBlock = kBlockPages * kPage;
constexpr uint32_t kTaskFused = CpuRelativeAttention::kTaskRows * kGroup;
constexpr uint32_t kTaskTiles = (kTaskFused + kTile - 1) / kTile;
constexpr uint32_t kWidth = CpuRelativeAttention::kQueryHeads * kD;
static_assert(kPage == kTile && kD == 128 && kStates == 16, "the panels are shaped for 32 keys and 128 dimensions");
// 32 consecutive fused rows span at most 12 rows, so a tile's rows' backward
// distances to one key differ by at most 11: the bias table's shifted rows
// start this far before distance zero.
constexpr uint32_t kBiasLead = 16;
constexpr uint32_t kBiasRows = kExtent + kBiasLead;
// P is held as fp16 times 2^15: the values' scales are below 0.1, so the
// product stays under 3,000 and rounds to fp16 exactly down to 2e-9.
constexpr float kProbabilityScale = 32768.0f;
constexpr float kScoreScale = 0.08838834764831845f;

// ---- The matrix units -------------------------------------------------------

template <unsigned Op> inline void amx(uint64_t operand) {
  __asm__ volatile(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(Op), "r"(operand) : "memory");
}
inline void amxSet() { __asm__ volatile("nop\nnop\nnop\n.word (0x201000 + (17 << 5) + 0)" : : : "memory"); }
inline void amxClear() { __asm__ volatile("nop\nnop\nnop\n.word (0x201000 + (17 << 5) + 1)" : : : "memory"); }
inline void loadX(const void *p) { amx<0>(reinterpret_cast<uint64_t>(p)); }
inline void loadY(const void *p) { amx<1>(reinterpret_cast<uint64_t>(p)); }
// Z rows 2r and 2r + 1 as 32 floats at `p`.
inline void storeZPair(void *p, unsigned r) {
  amx<5>(reinterpret_cast<uint64_t>(p) | (uint64_t{2 * r} << 56) | (1ull << 62));
}
// Z (f32, a 32 x 32 tile) += X (32 fp16) outer Y (32 fp16); the first product of
// a tile overwrites Z (bit 27), so Z is never cleared. Z[2 j + (i & 1)][i / 2]
// holds X[i] * Y[j], so a pair store gives row j's even then odd lanes of X.
constexpr uint64_t kFmaFirst = (1ull << 62) | (1ull << 27), kFmaNext = 1ull << 62;
inline void fma16(uint64_t mode) { amx<15>(mode); }

// ---- NEON helpers -----------------------------------------------------------

// e^x for x <= 0, relative error 2e-7 (degree 6); x below -87 gives 0, as -inf.
inline float32x4_t expNonPositive(float32x4_t x) {
  const uint32x4_t underflow = vcltq_f32(x, vdupq_n_f32(-87.0f));
  x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
  const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504f));
  float32x4_t r = vfmsq_n_f32(x, n, 0.693359375f);
  r = vfmsq_n_f32(r, n, -2.12194440e-4f);
  float32x4_t y = vdupq_n_f32(1.0f / 720);
  y = vfmaq_f32(vdupq_n_f32(1.0f / 120), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.0f / 24), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.0f / 6), y, r);
  y = vfmaq_f32(vdupq_n_f32(0.5f), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.0f), y, r);
  y = vfmaq_f32(vdupq_n_f32(1.0f), y, r);
  const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  y = vmulq_f32(y, vreinterpretq_f32_s32(e));
  return vbslq_f32(underflow, vdupq_n_f32(0.0f), y);
}

// Round to nearest even to bf16's 8 bits, kept as an f32 (finite values).
inline float32x4_t roundBf16(float32x4_t x) {
  uint32x4_t u = vreinterpretq_u32_f32(x);
  const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
  u = vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7fff), lsb));
  return vreinterpretq_f32_u32(vandq_u32(u, vdupq_n_u32(0xffff0000u)));
}
inline uint16_t toBf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if (x != x) return 0x7fc0;
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float fromBf16(uint16_t x) {
  const uint32_t u = uint32_t{x} << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// An 8 x 8 byte matrix transposed: out[c] holds column c of the rows in[0..7].
inline void transpose8(const int8x8_t *in, int8x8_t *out) {
  const int8x8x2_t a = vzip_s8(in[0], in[1]), b = vzip_s8(in[2], in[3]);
  const int8x8x2_t c = vzip_s8(in[4], in[5]), d = vzip_s8(in[6], in[7]);
  const int16x4x2_t e0 = vzip_s16(vreinterpret_s16_s8(a.val[0]), vreinterpret_s16_s8(b.val[0]));
  const int16x4x2_t e1 = vzip_s16(vreinterpret_s16_s8(a.val[1]), vreinterpret_s16_s8(b.val[1]));
  const int16x4x2_t e2 = vzip_s16(vreinterpret_s16_s8(c.val[0]), vreinterpret_s16_s8(d.val[0]));
  const int16x4x2_t e3 = vzip_s16(vreinterpret_s16_s8(c.val[1]), vreinterpret_s16_s8(d.val[1]));
  const int32x2x2_t f0 = vzip_s32(vreinterpret_s32_s16(e0.val[0]), vreinterpret_s32_s16(e2.val[0]));
  const int32x2x2_t f1 = vzip_s32(vreinterpret_s32_s16(e0.val[1]), vreinterpret_s32_s16(e2.val[1]));
  const int32x2x2_t f2 = vzip_s32(vreinterpret_s32_s16(e1.val[0]), vreinterpret_s32_s16(e3.val[0]));
  const int32x2x2_t f3 = vzip_s32(vreinterpret_s32_s16(e1.val[1]), vreinterpret_s32_s16(e3.val[1]));
  out[0] = vreinterpret_s8_s32(f0.val[0]);
  out[1] = vreinterpret_s8_s32(f0.val[1]);
  out[2] = vreinterpret_s8_s32(f1.val[0]);
  out[3] = vreinterpret_s8_s32(f1.val[1]);
  out[4] = vreinterpret_s8_s32(f2.val[0]);
  out[5] = vreinterpret_s8_s32(f2.val[1]);
  out[6] = vreinterpret_s8_s32(f3.val[0]);
  out[7] = vreinterpret_s8_s32(f3.val[1]);
}
inline float16x8_t widen(int8x8_t codes) { return vcvtq_f16_s16(vmovl_s8(codes)); }

// A page's keys of one head, int8 [32 token][128], to the fp16 panel [128
// dimension][32 token] the matrix units read.
void packKeys(const int8_t *keys, __fp16 *panel) {
  for (uint32_t tb = 0; tb < kPage / 8; ++tb)
    for (uint32_t db = 0; db < kD / 8; ++db) {
      int8x8_t rows[8], columns[8];
      for (uint32_t i = 0; i < 8; ++i) rows[i] = vld1_s8(keys + (tb * 8 + i) * kD + db * 8);
      transpose8(rows, columns);
      for (uint32_t m = 0; m < 8; ++m)
        vst1q_f16(reinterpret_cast<float16_t *>(panel + (db * 8 + m) * kPage + tb * 8), widen(columns[m]));
    }
}

// A page's values of one head, int8 [128 dimension][32 token], to the fp16
// panel [32 token][128 dimension].
void packValues(const int8_t *values, __fp16 *panel) {
  for (uint32_t tb = 0; tb < kPage / 8; ++tb)
    for (uint32_t db = 0; db < kD / 8; ++db) {
      int8x8_t rows[8], columns[8];
      for (uint32_t i = 0; i < 8; ++i) rows[i] = vld1_s8(values + (db * 8 + i) * kPage + tb * 8);
      transpose8(rows, columns);
      for (uint32_t m = 0; m < 8; ++m)
        vst1q_f16(reinterpret_cast<float16_t *>(panel + (tb * 8 + m) * kD + db * 8), widen(columns[m]));
    }
}

// ---- Pages ------------------------------------------------------------------

struct PageRef final {
  const int8_t *keys = nullptr;
  const float *keyScales = nullptr;
  const int8_t *values = nullptr;
  const float *valueScales = nullptr;
};

std::vector<PageRef> resolvePages(const CpuRelativeAttention::Rows &in, uint32_t pages) {
  std::vector<kv::HostExtent> extents(in.extents.begin(), in.extents.end());
  std::sort(extents.begin(), extents.end(),
            [](const kv::HostExtent &a, const kv::HostExtent &b) { return a.gpuAddress < b.gpuAddress; });
  std::vector<PageRef> result(pages);
  const uint64_t pagesInExtent = in.layer.extent_pages;
  for (uint32_t page = 0; page < pages; ++page) {
    const SplashKvPage entry = in.pageTable[page];
    const uint64_t address = entry & ~uint64_t{SPLASH_KV_PAGE_INDEX_MASK}, index = entry & SPLASH_KV_PAGE_INDEX_MASK;
    const auto at = std::lower_bound(extents.begin(), extents.end(), address,
                                     [](const kv::HostExtent &e, uint64_t a) { return e.gpuAddress < a; });
    if (!address || at == extents.end() || at->gpuAddress != address || !at->contents)
      throw std::invalid_argument("a CPU attention page is in no host extent");
    const std::byte *region = at->contents + in.layer.offset;
    result[page] = {reinterpret_cast<const int8_t *>(region + index * in.dataBytes),
                    reinterpret_cast<const float *>(region + pagesInExtent * in.dataBytes + index * in.scaleBytes),
                    reinterpret_cast<const int8_t *>(region + pagesInExtent * (in.dataBytes + in.scaleBytes) +
                                                     index * in.dataBytes),
                    reinterpret_cast<const float *>(region + pagesInExtent * (2 * in.dataBytes + in.scaleBytes) +
                                                    index * in.scaleBytes)};
  }
  return result;
}

// ---- A worker's memory ------------------------------------------------------

struct Aligned final {
  void operator()(void *p) const noexcept { std::free(p); }
};
template <class T> using Buffer = std::unique_ptr<T[], Aligned>;
template <class T> Buffer<T> allocate(size_t count) {
  size_t bytes = (count * sizeof(T) + 127) / 128 * 128;
  return Buffer<T>(static_cast<T *>(std::aligned_alloc(128, bytes)));
}

// One matrix tile of 32 fused rows of a task.
struct Tile final {
  Buffer<__fp16> queries = allocate<__fp16>(kD * kTile);
  // [row][slice][pair order]: a slice's 32 dimensions as the pair stores give
  // them, the even ones then the odd ones.
  Buffer<float> accumulator = allocate<float>(kTile * kD);
  // The tile's bias, [shifted distance][row], built at its first biased block.
  Buffer<float> bias = allocate<float>(size_t{kBiasRows} * kTile);
  alignas(16) float maximum[kTile], sum[kTile];
  alignas(16) uint32_t position[kTile];
  uint32_t valid = 0;
  uint32_t first = 0;
  bool biasReady = false;
};

struct Worker final {
  Buffer<__fp16> keyPanel = allocate<__fp16>(size_t{kBlockPages} * kD * kPage);
  Buffer<__fp16> valuePanel = allocate<__fp16>(size_t{kBlock} * kD);
  Buffer<float> scores = allocate<float>(size_t{kBlock} * kTile);
  Buffer<__fp16> probabilities = allocate<__fp16>(size_t{kBlock} * kTile);
  Buffer<float> product = allocate<float>(kTile * kTile);
  Buffer<float> rowBias = allocate<float>(kExtent);
  alignas(16) float keyScale[kBlock], valueScale[kBlock];
  std::vector<Tile> tiles = std::vector<Tile>(kTaskTiles);
};

// ---- A task -----------------------------------------------------------------

struct Context final {
  const CpuRelativeAttention::Rows &in;
  std::vector<PageRef> pages;
  // The bias table by state: [state][distance].
  std::vector<float> biasByState;
  std::atomic<uint32_t> next{0};
  uint32_t rowBlocks = 0;
  uint32_t tasks = 0;
};

// The bias of every distance below the extent for each of a tile's fused rows,
// shifted so that a key's row of it is one vector over the tile's rows:
// bias[distance + kBiasLead - (row's position - the tile's first)][row], zero
// where the distance is outside [0, extent).
void buildBias(Worker &w, Tile &tile, const Context &ctx, uint32_t head, uint32_t fused0) {
  const CpuRelativeAttention::Rows &in = ctx.in;
  std::memset(tile.bias.get(), 0, size_t{kBiasRows} * kTile * sizeof(float));
  for (uint32_t i = 0; i < tile.valid; ++i) {
    const uint32_t f = fused0 + i, row = f / kGroup;
    const float *states = in.states + ((uint64_t{head} * in.stride + row) * kGroup + f % kGroup) * kStates;
    for (uint32_t e = 0; e < kExtent; e += 4) {
      float32x4_t sum = vdupq_n_f32(0.0f);
      for (uint32_t k = 0; k < kStates; ++k)
        sum = vfmaq_n_f32(sum, vld1q_f32(ctx.biasByState.data() + size_t{k} * kExtent + e), states[k]);
      vst1q_f32(w.rowBias.get() + e, sum);
    }
    const uint32_t lead = in.committed + row - tile.position[0];
    for (uint32_t e = 0; e < kExtent; ++e) tile.bias[size_t{e + kBiasLead - lead} * kTile + i] = w.rowBias[e];
  }
  tile.biasReady = true;
}

// Scores of 32 fused rows against `tiles` pages of keys, into scores[key][row].
void scoreTile(const __fp16 *queries, const __fp16 *keys, uint32_t tiles, float *scores) {
  for (uint32_t n = 0; n < tiles; ++n) {
    const __fp16 *panel = keys + size_t{n} * kD * kPage;
    uint64_t mode = kFmaFirst;
    for (uint32_t d = 0; d < kD; ++d) {
      loadX(queries + d * kTile);
      loadY(panel + d * kPage);
      fma16(mode);
      mode = kFmaNext;
    }
    float *to = scores + size_t{n} * kPage * kTile;
    for (uint32_t t = 0; t < kPage; ++t) storeZPair(to + t * kTile, t);
  }
}

// The scores scaled (key scale, then 1/sqrt(128), each rounded as the GPU's),
// biased and masked, and each row's maximum over the block's keys.
template <bool Biased, bool Masked>
void scaleScores(Worker &w, const Tile &tile, uint32_t first, uint32_t keys, float32x4_t *maximum) {
  const float32x4_t none = vdupq_n_f32(-INFINITY);
  for (uint32_t q = 0; q < kTile / 4; ++q) maximum[q] = none;
  for (uint32_t t = 0; t < keys; ++t) {
    float *row = w.scores.get() + size_t{t} * kTile;
    const uint32_t key = first + t;
    const float *bias = nullptr;
    if constexpr (Biased) {
      const int64_t shifted = int64_t{tile.position[0]} - key + kBiasLead;
      if (shifted >= 0 && shifted < kBiasRows) bias = tile.bias.get() + size_t{static_cast<uint32_t>(shifted)} * kTile;
    }
    for (uint32_t q = 0; q < kTile / 4; ++q) {
      float32x4_t s = vmulq_n_f32(vmulq_n_f32(vld1q_f32(row + 4 * q), w.keyScale[t]), kScoreScale);
      if constexpr (Biased)
        if (bias) s = vaddq_f32(s, vld1q_f32(bias + 4 * q));
      if constexpr (Masked) s = vbslq_f32(vcleq_u32(vdupq_n_u32(key), vld1q_u32(tile.position + 4 * q)), s, none);
      vst1q_f32(row + 4 * q, s);
      maximum[q] = vmaxq_f32(maximum[q], s);
    }
  }
}

// The softmax of a block into fp16 panels of P = bf16(p * value scale), the row
// maxima and sums advanced, and the accumulator's factors for the rows whose
// maximum grew; then P times the values into the accumulator.
void softmaxAndValues(Worker &w, Tile &tile, uint32_t keys, const float32x4_t *blockMaximum) {
  const float32x4_t none = vdupq_n_f32(-INFINITY), zero = vdupq_n_f32(0.0f);
  float32x4_t next[kTile / 4], safe[kTile / 4], sums[kTile / 4];
  alignas(16) float factor[kTile];
  for (uint32_t q = 0; q < kTile / 4; ++q) {
    const float32x4_t previous = vld1q_f32(tile.maximum + 4 * q);
    next[q] = vmaxq_f32(previous, blockMaximum[q]);
    // A row with no key yet keeps factor 1, as the GPU's page loop does.
    const uint32x4_t unchanged = vorrq_u32(vceqq_f32(next[q], none), vceqq_f32(next[q], previous));
    const float32x4_t scale = expNonPositive(vsubq_f32(previous, next[q]));
    vst1q_f32(factor + 4 * q, vbslq_f32(unchanged, vdupq_n_f32(1.0f), scale));
    safe[q] = vbslq_f32(vceqq_f32(next[q], none), zero, next[q]);
    vst1q_f32(tile.maximum + 4 * q, next[q]);
    sums[q] = zero;
  }
  for (uint32_t t = 0; t < keys; ++t) {
    const float *row = w.scores.get() + size_t{t} * kTile;
    const float32x4_t valueScale = vdupq_n_f32(w.valueScale[t]);
    for (uint32_t q = 0; q < kTile / 4; q += 2) {
      float16x4_t half[2];
      for (uint32_t k = 0; k < 2; ++k) {
        const float32x4_t s = vld1q_f32(row + 4 * (q + k));
        const float32x4_t p = expNonPositive(vsubq_f32(s, safe[q + k]));
        sums[q + k] = vaddq_f32(sums[q + k], p);
        const float32x4_t scaled = vmulq_n_f32(roundBf16(vmulq_f32(p, valueScale)), kProbabilityScale);
        // A masked key's scale may be unwritten memory: its probability is zero.
        const uint32x4_t seen = vcgtq_f32(s, none);
        half[k] = vcvt_f16_f32(vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(scaled), seen)));
      }
      vst1q_f16(reinterpret_cast<float16_t *>(w.probabilities.get() + size_t{t} * kTile + 4 * q),
                vcombine_f16(half[0], half[1]));
    }
  }
  for (uint32_t q = 0; q < kTile / 4; ++q)
    vst1q_f32(tile.sum + 4 * q,
              vaddq_f32(vmulq_f32(vld1q_f32(tile.sum + 4 * q), vld1q_f32(factor + 4 * q)), sums[q]));
  for (uint32_t slice = 0; slice < kD / kTile; ++slice) {
    uint64_t mode = kFmaFirst;
    for (uint32_t t = 0; t < keys; ++t) {
      loadX(w.valuePanel.get() + size_t{t} * kD + slice * kTile);
      loadY(w.probabilities.get() + size_t{t} * kTile);
      fma16(mode);
      mode = kFmaNext;
    }
    for (uint32_t r = 0; r < kTile; ++r) storeZPair(w.product.get() + r * kTile, r);
    for (uint32_t r = 0; r < kTile; ++r) {
      float *accumulator = tile.accumulator.get() + r * kD + slice * kTile;
      const float *product = w.product.get() + r * kTile;
      for (uint32_t k = 0; k < kTile; k += 4)
        vst1q_f32(accumulator + k, vfmaq_n_f32(vld1q_f32(product + k), vld1q_f32(accumulator + k), factor[r]));
    }
  }
}

// A row's output: the sink in the maximum and the sum, the quotient rounded to
// bf16, then times its gate's sigmoid, rounded again.
void finishRows(const Tile &tile, const Context &ctx, uint32_t head, uint32_t fused0, uint16_t *hidden) {
  const CpuRelativeAttention::Rows &in = ctx.in;
  for (uint32_t i = 0; i < tile.valid; ++i) {
    const uint32_t f = fused0 + i, row = f / kGroup, queryHead = head * kGroup + f % kGroup;
    const float sink = in.sinks[(uint64_t{head} * in.stride + row) * kGroup + f % kGroup];
    const float maximum = std::max(tile.maximum[i], sink);
    const float weight = std::exp(tile.maximum[i] - maximum);
    const float weighted = weight * tile.sum[i];
    const float denominator = weighted + std::exp(sink - maximum);
    const float *accumulator = tile.accumulator.get() + i * kD;
    const uint16_t *gates = in.projection + uint64_t{row} * in.projectionStride + in.gateColumn + queryHead * kD;
    uint16_t *out = hidden + uint64_t{row} * kWidth + queryHead * kD;
    for (uint32_t d = 0; d < kD; ++d) {
      const uint32_t slice = d / kTile, lane = d % kTile;
      const float value = accumulator[slice * kTile + (lane % 2 ? kTile / 2 : 0) + lane / 2];
      const float numerator = weight * value / kProbabilityScale;
      const float quotient = fromBf16(toBf16(denominator > 0.0f ? numerator / denominator : 0.0f));
      const float gate = 1.0f / (1.0f + std::exp2(-1.44269504089f * fromBf16(gates[d])));
      out[d] = toBf16(quotient * gate);
    }
  }
}

void runTask(Worker &w, const Context &ctx, uint32_t head, uint32_t rowBlock) {
  const CpuRelativeAttention::Rows &in = ctx.in;
  const uint32_t row0 = rowBlock * CpuRelativeAttention::kTaskRows;
  const uint32_t rows = std::min(CpuRelativeAttention::kTaskRows, in.rows - row0);
  const uint32_t fused0 = row0 * kGroup, fusedCount = rows * kGroup;
  const uint32_t tiles = (fusedCount + kTile - 1) / kTile;
  const uint32_t last = in.committed + row0 + rows - 1;
  for (uint32_t index = 0; index < tiles; ++index) {
    Tile &tile = w.tiles[index];
    tile.first = fused0 + index * kTile;
    tile.valid = std::min(kTile, fusedCount - index * kTile);
    tile.biasReady = false;
    std::memset(tile.accumulator.get(), 0, kTile * kD * sizeof(float));
    for (uint32_t i = 0; i < kTile; ++i) {
      const bool valid = i < tile.valid;
      const uint32_t f = tile.first + std::min(i, tile.valid - 1), row = f / kGroup;
      tile.position[i] = in.committed + row;
      tile.maximum[i] = -INFINITY;
      tile.sum[i] = 0.0f;
      // Lane i of a query panel: the pair stores give rows 0 to 15 at the even
      // lanes and 16 to 31 at the odd ones.
      const uint32_t lane = i < 16 ? 2 * i : 2 * (i - 16) + 1;
      const uint16_t *query = in.queries + ((uint64_t{head} * in.stride + row) * kGroup + f % kGroup) * kD;
      for (uint32_t d = 0; d < kD; ++d)
        tile.queries[d * kTile + lane] = valid ? static_cast<__fp16>(fromBf16(query[d])) : static_cast<__fp16>(0.0f);
    }
  }
  const uint32_t keys = last + 1;
  for (uint32_t first = 0; first < keys; first += kBlock) {
    const uint32_t pages = std::min(kBlockPages, (keys - first + kPage - 1) / kPage), count = pages * kPage;
    for (uint32_t n = 0; n < pages; ++n) {
      const PageRef &page = ctx.pages[first / kPage + n];
      packKeys(page.keys + head * kPage * kD, w.keyPanel.get() + size_t{n} * kD * kPage);
      packValues(page.values + head * kD * kPage, w.valuePanel.get() + size_t{n} * kPage * kD);
      std::memcpy(w.keyScale + n * kPage, page.keyScales + head * kPage, kPage * sizeof(float));
      std::memcpy(w.valueScale + n * kPage, page.valueScales + head * kPage, kPage * sizeof(float));
    }
    for (uint32_t index = 0; index < tiles; ++index) {
      Tile &tile = w.tiles[index];
      const uint32_t lowest = tile.position[0];
      const bool masked = first + count - 1 > lowest, biased = uint64_t{first} + count - 1 + kExtent > lowest;
      if (biased && !tile.biasReady) buildBias(w, tile, ctx, head, tile.first);
      scoreTile(tile.queries.get(), w.keyPanel.get(), pages, w.scores.get());
      float32x4_t maximum[kTile / 4];
      if (biased)
        masked ? scaleScores<true, true>(w, tile, first, count, maximum)
               : scaleScores<true, false>(w, tile, first, count, maximum);
      else
        masked ? scaleScores<false, true>(w, tile, first, count, maximum)
               : scaleScores<false, false>(w, tile, first, count, maximum);
      softmaxAndValues(w, tile, count, maximum);
    }
  }
  for (uint32_t index = 0; index < tiles; ++index) finishRows(w.tiles[index], ctx, head, w.tiles[index].first, in.hidden);
}

void work(void *raw, size_t) {
  Context &ctx = *static_cast<Context *>(raw);
  thread_local std::unique_ptr<Worker> worker;
  if (!worker) worker = std::make_unique<Worker>();
  amxSet();
  for (uint32_t task = ctx.next.fetch_add(1); task < ctx.tasks; task = ctx.next.fetch_add(1)) {
    // Later row blocks see the most keys: they go first.
    runTask(*worker, ctx, task % kHeads, ctx.rowBlocks - 1 - task / kHeads);
  }
  amxClear();
}

} // namespace

bool CpuRelativeAttention::available() noexcept {
  uint32_t family = 0, sme = 0;
  size_t size = sizeof(family);
  if (sysctlbyname("hw.cpufamily", &family, &size, nullptr, 0)) return false;
  size = sizeof(sme);
  if (!sysctlbyname("hw.optional.arm.FEAT_SME", &sme, &size, nullptr, 0) && sme) return false;
  constexpr uint32_t kM1 = 0x1b588bb3, kM2 = 0xda33d83d;
  return family == kM1 || family == kM2;
}

const CpuRelativeAttention::Settings &CpuRelativeAttention::settings() {
  static const Settings value = [] {
    Settings result;
    if (const char *share = std::getenv("SPLASH_CPU_ATTN_SHARE")) result.share = std::clamp(std::atof(share), 0.0, 0.9);
    if (const char *threads = std::getenv("SPLASH_CPU_ATTN_THREADS"))
      result.threads = static_cast<uint32_t>(std::clamp(std::atoi(threads), 1, 32));
    const auto flag = [](const char *name) {
      const char *text = std::getenv(name);
      return text && std::string_view(text) == "1";
    };
    result.shadow = flag("SPLASH_CPU_ATTN_SHADOW");
    result.timing = flag("SPLASH_CPU_ATTN_TIMING");
    // The shadow compares at the share's rows, so it needs one.
    if (result.shadow && result.share <= 0.0) result.share = 0.3;
    return result;
  }();
  return value;
}

uint32_t CpuRelativeAttention::rowsFor(double share, uint32_t committed, uint32_t rows) noexcept {
  if (share <= 0.0 || committed < kMinimumHistory || rows < kMinimumRows) return 0;
  const double c = committed, target = share * rows * (c + rows / 2.0);
  const double taken = -c + std::sqrt(c * c + 2.0 * target);
  const uint32_t tile = SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS;
  return std::min(static_cast<uint32_t>(taken) / tile * tile, (rows - 1) / tile * tile);
}

CpuRelativeAttention::Difference CpuRelativeAttention::compare(const uint16_t *expected, const uint16_t *actual,
                                                               uint64_t count) noexcept {
  // bf16 values as integers that grow with the value: the distance of two is
  // the number of bf16 values between them.
  const auto order = [](uint16_t bits) { return bits & 0x8000 ? 0x8000 - int32_t{bits & 0x7fff} : 0x8000 + int32_t{bits}; };
  Difference result;
  double squares = 0.0, norm = 0.0;
  uint64_t far = 0;
  for (uint64_t i = 0; i < count; ++i) {
    const auto steps = static_cast<uint32_t>(std::abs(order(expected[i]) - order(actual[i])));
    result.largestSteps = std::max(result.largestSteps, steps);
    far += steps >= 2;
    const double a = fromBf16(expected[i]), b = fromBf16(actual[i]);
    squares += (a - b) * (a - b);
    norm += a * a;
  }
  result.relative = norm > 0.0 ? std::sqrt(squares / norm) : 0.0;
  result.farShare = count ? double(far) / double(count) : 0.0;
  return result;
}

void CpuRelativeAttention::run(const Rows &in, uint32_t threads) {
  if (!in.rows) return;
  const uint32_t pages = (in.committed + in.rows + kPage - 1) / kPage;
  if (!in.queries || !in.states || !in.sinks || !in.projection || !in.bias || !in.pageTable || !in.hidden ||
      in.pageTableEntries < pages || in.stride < in.rows || !in.dataBytes || !in.scaleBytes)
    throw std::invalid_argument("invalid CPU attention rows");
  Context ctx{in, resolvePages(in, pages), std::vector<float>(size_t{kStates} * kExtent)};
  for (uint32_t distance = 0; distance < kExtent; ++distance)
    for (uint32_t k = 0; k < kStates; ++k) ctx.biasByState[size_t{k} * kExtent + distance] = in.bias[distance * kStates + k];
  ctx.rowBlocks = (in.rows + kTaskRows - 1) / kTaskRows;
  ctx.tasks = ctx.rowBlocks * kHeads;
  threads = std::max(1U, std::min(threads, ctx.tasks));
  dispatch_apply_f(threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx, work);
}

} // namespace splash::ops
