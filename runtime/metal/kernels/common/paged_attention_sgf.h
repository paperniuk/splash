#pragma once

#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/sgmatrix.h"

// Apple7/8 (M1/M2) register-matrix paged attention over INT8 or BF16 KV (the
// Register tile), for prefill and verify. These GPUs have no bfloat
// arithmetic, and the MPP tensor path spends most of a long-context prefill
// chunk or verify cycle here. Operands:
//   K, V  INT8: the cache as exact half: byte ^ 0x80 is the value plus 128,
//         0x6400 | v is 1024 + v, and subtracting 1152 leaves the value;
//         BF16: the stored values widened to exact fp32;
//   Q     bfloat widened to exact fp32;
//   P     the (value-scaled) probability rounded to bfloat exactly as the MPP
//         tile rounds it, then widened to fp32;
//   C     fp32, so every product is exact and only summation order differs.
// Each simdgroup owns eight fused rows (one query row group of the tile) and
// runs the whole online softmax for them in registers: S^T = K Q^T leaves the
// scores in exactly the fragment layout PV = V^T P^T takes as its B operand,
// so no threadgroup memory or barrier is needed. The tile's G simdgroups read
// the same K/V page. Partials and statistics use the MPP tile's layout, so
// the shared prefill and verify reduces are unchanged. Both scale placements
// of the MPP tile compute score x key scale x 1/sqrt(D) in that order, as
// this tile does; BF16 has no scales, as in the MPP tile. Pages are reached
// through their table entries (kv_extent.h), as in the MPP tile.
namespace q8sgf {
using sgmatrix::Lane;
using sgmatrix::lane_map;
using sgmatrix::te;

// c += a x b for exact half cache values and fp32 queries/probabilities.
__attribute__((always_inline)) inline void mma(thread float2 &c, half2 a, float2 b) {
  simdgroup_half8x8 A;
  simdgroup_float8x8 B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

// c += a x b for exact fp32 (widened bfloat) cache values.
__attribute__((always_inline)) inline void mma(thread float2 &c, float2 a, float2 b) {
  simdgroup_float8x8 A, B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

// Bytes 0 and 2 of w as exact half.
__attribute__((always_inline)) inline half2 int8_pair(uint w) {
  return as_type<half2>(((w & 0x00FF00FFu) ^ 0x00800080u) | 0x64006400u) -
         half2(1152.0h);
}

// Two bfloat values packed low/high, widened to exact fp32.
__attribute__((always_inline)) inline float2 bfloat_pair(uint p) {
  return float2(as_type<float>(p << 16), as_type<float>(p & 0xFFFF0000u));
}

// The low (high = false) or high bfloat of words a and b, widened to fp32.
__attribute__((always_inline)) inline float2 bfloat_halves(uint a, uint b, bool high) {
  return high ? float2(as_type<float>(a & 0xFFFF0000u), as_type<float>(b & 0xFFFF0000u))
              : float2(as_type<float>(a << 16), as_type<float>(b << 16));
}

// The reduction index k of an MMA may name any element, as long as A and B
// agree. Both products permute theirs so that a lane's two k columns (fn and
// fn + 1) over consecutive steps are one contiguous load, with the pair for a
// step in elements 0 and 2 (even step) or 1 and 3 (odd step) of a group of
// four.
// Token of k row k at PV step s (four steps span a page), which is also the
// token of score row k in S^T tile s.
inline uint page_token(uint k, uint s) {
  return 8 * (k >> 1) + 4 * (s >> 1) + (s & 1) + 2 * (k & 1);
}
// Offset inside a 64-dimension block of k row k at QK step t (eight steps).
inline uint block_dimension(uint k, uint t) {
  return 16 * (k >> 1) + 4 * (t >> 1) + (t & 1) + 2 * (k & 1);
}

// A lane's sixteen key elements of one 64-dimension block: dimensions
// 8 fn .. 8 fn + 15, consumed two per QK step.
template <typename CacheElement> struct KeyRun;
template <> struct KeyRun<int8_t> {
  uint4 words;
  void load(device const int8_t *keys) {
    words = *reinterpret_cast<device const uint4 *>(keys);
  }
  half2 step(uint t) const { return int8_pair(words[t >> 1] >> (8 * (t & 1))); }
};
template <> struct KeyRun<bfloat> {
  uint4 low, high;
  void load(device const bfloat *keys) {
    low = reinterpret_cast<device const uint4 *>(keys)[0];
    high = reinterpret_cast<device const uint4 *>(keys)[1];
  }
  float2 step(uint t) const {
    const uint4 words = t < 4 ? low : high;
    const uint j = 2 * ((t >> 1) & 1);
    return bfloat_halves(words[j], words[j + 1], t & 1);
  }
};

// A lane's eight value elements of one dimension: tokens 4 fn .. 4 fn + 7 of
// the page, consumed two per PV step.
template <typename CacheElement> struct ValueRun;
template <> struct ValueRun<int8_t> {
  uint2 words;
  void load(device const int8_t *values) {
    words = *reinterpret_cast<device const uint2 *>(values);
  }
  void load(threadgroup const int8_t *values) {
    words = *reinterpret_cast<threadgroup const uint2 *>(values);
  }
  half2 step(uint s) const {
    return int8_pair((s < 2 ? words.x : words.y) >> (8 * (s & 1)));
  }
};
template <> struct ValueRun<bfloat> {
  uint4 words;
  void load(device const bfloat *values) {
    words = *reinterpret_cast<device const uint4 *>(values);
  }
  void load(threadgroup const bfloat *values) {
    words = *reinterpret_cast<threadgroup const uint4 *>(values);
  }
  float2 step(uint s) const {
    return s < 2 ? bfloat_halves(words.x, words.y, s & 1)
                 : bfloat_halves(words.z, words.w, s & 1);
  }
};

// One tile of Rows query rows x G heads over its split's pages, as
// splash_paged_attention_tile: row r of the tile sees committed_tokens + r + 1
// tokens, and rows past active_rows repeat the last active row's mask. Rows is
// the eight-row tile's, or one part's (splash_attention_tile_part), so that a
// group of sixteen runs as two parts of 64 fused rows, eight simdgroups each.
// HeadDim 128 (Flint's layouts) halves the QK blocks and O^T tiles; Window, as
// the MPP tile's, attends a row at position p only the keys p - Window to p.
template <uint KVHeads, uint QueryHeadsPerKVHead, typename CacheElement, uint Rows = 8,
          uint HeadDim = SPLASH_KV_HEAD_DIMENSION, uint Window = 0>
__attribute__((always_inline)) inline void tile(
    device const bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    uint sg, uint lane, threadgroup uint *query_stage) {
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr uint G = QueryHeadsPerKVHead;
  constexpr uint M = Rows * G;
  constexpr uint N = SplashKvPageTokens;
  constexpr uint D = HeadDim;
  static_assert(N == 32 && (D == 256 || D == 128),
                "fragment maps assume Page32 and D of 256 or 128");
  static_assert(SPLASH_TARGET_VERIFY_ROWS == 8 && SPLASH_PREFILL_ATTENTION_TILE_ROWS == 8 &&
                    8 % Rows == 0 && M % 8 == 0,
                "one simdgroup per eight fused rows of an eight-row tile or one of its parts");
  const uint visible = committed_tokens + active_rows;
  const uint pages = splash_attention_pages(visible);
  const uint first_page = splash_attention_first_page<Window>(committed_tokens);
  const uint per_split = splash_attention_pages_per_split(pages - first_page, splits);
  const uint page_begin = first_page + split * per_split;
  if (page_begin >= pages)
    return;
  const uint page_end = min(pages, page_begin + per_split);
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  // This lane's fused rows are row and row + 1, the columns of its fragments.
  const uint row = sg * 8 + fn;
  const uint2 limit(
      min(visible, committed_tokens + min(row / G, active_rows - 1) + 1),
      min(visible, committed_tokens + min((row + 1) / G, active_rows - 1) + 1));
  // A row's first key with a window: its causal end less Window keys.
  const uint2 window_start(splash_attention_window_start<Window>(limit.x - 1),
                           splash_attention_window_start<Window>(limit.y - 1));

  // Q^T fragments for all D/4 QK steps, bfloat pairs (row, row + 1), staged
  // in this lane's own threadgroup words (step-major, so lanes never share a
  // bank) to keep them out of registers.
  constexpr uint kQueryWords = 32 * (D / 8);
  threadgroup uint *queries = query_stage + sg * kQueryWords;
  {
    device const ushort *q0 =
        reinterpret_cast<device const ushort *>(tile_queries) + row * D;
    device const ushort *q1 = q0 + D;
#pragma unroll
    for (uint b = 0; b < D / 64; ++b)
#pragma unroll
      for (uint t = 0; t < 8; ++t) {
        const uint d = 64 * b + block_dimension(fm, t);
        queries[(b * 8 + t) * 32 + lane] = uint(q0[d]) | (uint(q1[d]) << 16);
      }
  }
  // O^T tiles: dimension 8 d + fm of rows (row, row + 1).
  float2 output[D / 8];
#pragma unroll
  for (uint d = 0; d < D / 8; ++d)
    output[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);
  // The four tokens of this lane's score rows: base + {0, 1, 4, 5}.
  const uint scale_base = page_token(fm, 0);
  const SplashKvAddressing<KVHeads, D, CacheElement> addressing(kv, kv_head);

  for (uint page = page_begin; page < page_end; ++page) {
    const SplashKvPageTensors<CacheElement> tensors = addressing.page(page_table[page]);
    device const CacheElement *keys = tensors.keys;
    device const CacheElement *values = tensors.values;
    float key_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float value_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if constexpr (Quantized) {
      device const float *key_scales = tensors.key_scales + scale_base;
      device const float *value_scales = tensors.value_scales + scale_base;
      const float2 k01 = *reinterpret_cast<device const float2 *>(key_scales);
      const float2 k45 = *reinterpret_cast<device const float2 *>(key_scales + 4);
      const float2 v01 = *reinterpret_cast<device const float2 *>(value_scales);
      const float2 v45 = *reinterpret_cast<device const float2 *>(value_scales + 4);
      key_scale[0] = k01.x, key_scale[1] = k01.y, key_scale[2] = k45.x, key_scale[3] = k45.y;
      value_scale[0] = v01.x, value_scale[1] = v01.y, value_scale[2] = v45.x,
      value_scale[3] = v45.y;
    }

    // S^T tile s: tokens page_token(fm, s) x rows (row, row + 1), each tile
    // summing its (b, t) steps in order.
    float2 scores[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s)
      scores[s] = float2(0.0f);
#pragma unroll
    for (uint b = 0; b < D / 64; ++b)
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        KeyRun<CacheElement> run;
        run.load(keys + page_token(fm, s) * D + 64 * b + 8 * fn);
#pragma unroll
        for (uint t = 0; t < 8; ++t)
          mma(scores[s], run.step(t), bfloat_pair(queries[(b * 8 + t) * 32 + lane]));
      }

    // Online softmax, as splash_attention_page_softmax: key scale, then
    // 1/sqrt(D); masked tokens never contribute, whatever their cache holds.
    // 1/sqrt(D): 0.0625 = 2^-4 (D of 256); 0.08838834764831845 = 2^-3.5 (128).
    constexpr float kScoreScale = D == 256 ? 0.0625f : 0.08838834764831845f;
    const uint token_start = page * N;
    float2 local_max(-INFINITY);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = token_start + page_token(fm, s);
      float2 score = scores[s];
      if constexpr (Quantized)
        score *= key_scale[s];
      score *= kScoreScale;
      scores[s] = float2(token < limit.x && token >= window_start.x ? score.x : -INFINITY,
                         token < limit.y && token >= window_start.y ? score.y : -INFINITY);
      local_max = max(local_max, scores[s]);
    }
    // Lane bits 1, 2 and 4 select fm: the other 28 tokens of these rows.
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next_max = max(row_max, local_max);
    float2 probability[4];
    float2 local_sum(0.0f);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = token_start + page_token(fm, s);
      probability[s] =
          float2(token < limit.x && token >= window_start.x
                     ? fast::exp(scores[s].x - next_max.x) : 0.0f,
                 token < limit.y && token >= window_start.y
                     ? fast::exp(scores[s].y - next_max.y) : 0.0f);
      local_sum += probability[s];
    }
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 16);
    const float2 scale(
        next_max.x == -INFINITY || next_max.x == row_max.x
            ? 1.0f : fast::exp(row_max.x - next_max.x),
        next_max.y == -INFINITY || next_max.y == row_max.y
            ? 1.0f : fast::exp(row_max.y - next_max.y));
    row_sum = row_sum * scale + local_sum;
    row_max = next_max;
    if (simd_any(scale.x != 1.0f || scale.y != 1.0f)) {
#pragma unroll
      for (uint d = 0; d < D / 8; ++d)
        output[d] *= scale;
    }
    // P^T tile s is the score tile's own fragment: B[k = fm][n = fn].
    float2 weights[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = token_start + page_token(fm, s);
      weights[s] = float2(
          token < limit.x && token >= window_start.x
              ? float(bfloat(probability[s].x * value_scale[s])) : 0.0f,
          token < limit.y && token >= window_start.y
              ? float(bfloat(probability[s].y * value_scale[s])) : 0.0f);
    }

    // O^T tile d += V^T (dimensions 8 d + fm, tokens) x P^T, each output tile
    // summing its four token steps in order. The steps are spelled out: as a
    // nested loop inside the unrolled dimension loop, the compiler leaves the
    // step index dynamic, which costs up to 10% in verify and 20% in BF16.
    device const CacheElement *value_row = values + fm * N + 4 * fn;
#pragma unroll
    for (uint d = 0; d < D / 8; ++d) {
      ValueRun<CacheElement> run;
      run.load(value_row + d * 8 * N);
      mma(output[d], run.step(0), weights[0]);
      mma(output[d], run.step(1), weights[1]);
      mma(output[d], run.step(2), weights[2]);
      mma(output[d], run.step(3), weights[3]);
    }
  }

  device float *target = partials + slot * M * D;
#pragma unroll
  for (uint d = 0; d < D / 8; ++d) {
    target[row * D + 8 * d + fm] = output[d].x;
    target[(row + 1) * D + 8 * d + fm] = output[d].y;
  }
  if (fm == 0) {
    device float *stat = statistics + (slot * M + row) * 2;
    stat[0] = row_max.x;
    stat[1] = row_sum.x;
    stat[2] = row_max.y;
    stat[3] = row_sum.y;
  }
}
} // namespace q8sgf
