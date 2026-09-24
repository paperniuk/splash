#pragma once

#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/q4_sgmatrix.h"

// Apple7/8 (M1/M2) register-matrix paged attention over INT8 KV (the
// Register tile), for prefill and verify. These GPUs have no bfloat
// arithmetic, and the MPP tensor path spends most of a long-context prefill
// chunk or verify cycle here. Operands:
//   K, V  the INT8 cache as exact half: byte ^ 0x80 is the value plus 128,
//         0x6400 | v is 1024 + v, and subtracting 1152 leaves the value;
//   Q     bfloat widened to exact fp32;
//   P     the value-scaled probability rounded to bfloat exactly as the MPP
//         tile rounds it, then widened to fp32;
//   C     fp32, so every product is exact and only summation order differs.
// Each simdgroup owns eight fused rows (one query row group of the tile) and
// runs the whole online softmax for them in registers: S^T = K Q^T leaves the
// scores in exactly the fragment layout PV = V^T P^T takes as its B operand,
// so no threadgroup memory or barrier is needed. The tile's G simdgroups read
// the same K/V page. Partials and statistics use the MPP tile's layout, so
// the shared prefill and verify reduces are unchanged. Both scale placements
// of the MPP tile compute score x key scale x 1/sqrt(D) in that order, as
// this tile does.
namespace q8sgf {
using q4sg::Lane;
using q4sg::lane_map;
using q4sg::te;

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

// Bytes 0 and 2 of w as exact half.
__attribute__((always_inline)) inline half2 int8_pair(uint w) {
  return as_type<half2>(((w & 0x00FF00FFu) ^ 0x00800080u) | 0x64006400u) -
         half2(1152.0h);
}

// Two bfloat values packed low/high, widened to exact fp32.
__attribute__((always_inline)) inline float2 bfloat_pair(uint p) {
  return float2(as_type<float>(p << 16), as_type<float>(p & 0xFFFF0000u));
}

// The reduction index k of an MMA may name any element, as long as A and B
// agree. Both products permute theirs so that a lane's two k columns (fn and
// fn + 1) over consecutive steps are one contiguous load, with the pair for a
// step in bytes 0 and 2 (even step) or 1 and 3 (odd step) of a word.
// Token of k row k at PV step s (four steps span a page), which is also the
// token of score row k in S^T tile s.
inline uint page_token(uint k, uint s) {
  return 8 * (k >> 1) + 4 * (s >> 1) + (s & 1) + 2 * (k & 1);
}
// Offset inside a 64-dimension block of k row k at QK step t (eight steps).
inline uint block_dimension(uint k, uint t) {
  return 16 * (k >> 1) + 4 * (t >> 1) + (t & 1) + 2 * (k & 1);
}

// One tile of eight query rows x G heads over its split's pages, as
// splash_paged_attention_tile: row r of the tile sees committed_tokens + r + 1
// tokens, and rows past active_rows repeat the last active row's mask.
template <uint KVHeads, uint QueryHeadsPerKVHead>
__attribute__((always_inline)) inline void tile(
    device const bfloat *tile_queries, device const int8_t *cache_keys,
    device const float *key_scales, device const int8_t *cache_values,
    device const float *value_scales, device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    uint sg, uint lane) {
  constexpr uint G = QueryHeadsPerKVHead;
  constexpr uint Rows = 8;
  constexpr uint M = Rows * G;
  constexpr uint N = SplashQ8PageTokens;
  constexpr uint D = SplashQ8HeadDimension;
  static_assert(N == 32 && D == 256, "fragment maps assume Page32 and D = 256");
  static_assert(SPLASH_TARGET_VERIFY_ROWS == Rows &&
                    SPLASH_PREFILL_ATTENTION_TILE_ROWS == Rows,
                "one simdgroup per G fused rows of an eight-row tile");
  const uint visible = committed_tokens + active_rows;
  const uint pages = splash_attention_pages(visible);
  const uint per_split = splash_attention_pages_per_split(pages, splits);
  const uint page_begin = split * per_split;
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

  // Q^T fragments for all 32 QK steps, bfloat pairs (row, row + 1).
  uint queries[4][8];
  {
    device const ushort *q0 =
        reinterpret_cast<device const ushort *>(tile_queries) + row * D;
    device const ushort *q1 = q0 + D;
#pragma unroll
    for (uint b = 0; b < 4; ++b)
#pragma unroll
      for (uint t = 0; t < 8; ++t) {
        const uint d = 64 * b + block_dimension(fm, t);
        queries[b][t] = uint(q0[d]) | (uint(q1[d]) << 16);
      }
  }
  // O^T tiles: dimension 8 d + fm of rows (row, row + 1).
  float2 output[32];
#pragma unroll
  for (uint d = 0; d < 32; ++d)
    output[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);
  // The four tokens of this lane's score rows: base + {0, 1, 4, 5}.
  const uint scale_base = page_token(fm, 0);

  for (uint page = page_begin; page < page_end; ++page) {
    const uint physical = page_table[page];
    device const int8_t *keys =
        cache_keys + splash_q8_key_index<KVHeads>(physical, kv_head, 0, 0);
    device const int8_t *values =
        cache_values + splash_q8_value_index<KVHeads>(physical, kv_head, 0, 0);
    const ulong scale_index =
        splash_q8_scale_index<KVHeads>(physical, kv_head, 0) + scale_base;
    const float2 k01 = *reinterpret_cast<device const float2 *>(key_scales + scale_index);
    const float2 k45 = *reinterpret_cast<device const float2 *>(key_scales + scale_index + 4);
    const float2 v01 = *reinterpret_cast<device const float2 *>(value_scales + scale_index);
    const float2 v45 = *reinterpret_cast<device const float2 *>(value_scales + scale_index + 4);
    const float key_scale[4] = {k01.x, k01.y, k45.x, k45.y};
    const float value_scale[4] = {v01.x, v01.y, v45.x, v45.y};

    // S^T tile s: tokens page_token(fm, s) x rows (row, row + 1).
    float2 scores[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s)
      scores[s] = float2(0.0f);
#pragma unroll
    for (uint b = 0; b < 4; ++b)
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        const uint4 words = *reinterpret_cast<device const uint4 *>(
            keys + page_token(fm, s) * D + 64 * b + 8 * fn);
#pragma unroll
        for (uint t = 0; t < 8; ++t)
          mma(scores[s], int8_pair(words[t >> 1] >> (8 * (t & 1))),
              bfloat_pair(queries[b][t]));
      }

    // Online softmax, as splash_attention_page_softmax: key scale, then
    // 1/sqrt(D); masked tokens never contribute, whatever their cache holds.
    const uint token_start = page * N;
    float2 local_max(-INFINITY);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = token_start + page_token(fm, s);
      float2 score = scores[s] * key_scale[s];
      score *= 0.0625f;
      scores[s] = float2(token < limit.x ? score.x : -INFINITY,
                         token < limit.y ? score.y : -INFINITY);
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
          float2(token < limit.x ? fast::exp(scores[s].x - next_max.x) : 0.0f,
                 token < limit.y ? fast::exp(scores[s].y - next_max.y) : 0.0f);
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
      for (uint d = 0; d < 32; ++d)
        output[d] *= scale;
    }
    // P^T tile s is the score tile's own fragment: B[k = fm][n = fn].
    float2 weights[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = token_start + page_token(fm, s);
      weights[s] = float2(
          token < limit.x ? float(bfloat(probability[s].x * value_scale[s])) : 0.0f,
          token < limit.y ? float(bfloat(probability[s].y * value_scale[s])) : 0.0f);
    }

    // O^T tile d += V^T (dimensions 8 d + fm, tokens) x P^T.
    device const int8_t *value_row = values + fm * N + 4 * fn;
#pragma unroll
    for (uint d = 0; d < 32; ++d) {
      const uint2 words =
          *reinterpret_cast<device const uint2 *>(value_row + d * 8 * N);
      mma(output[d], int8_pair(words.x), weights[0]);
      mma(output[d], int8_pair(words.x >> 8), weights[1]);
      mma(output[d], int8_pair(words.y), weights[2]);
      mma(output[d], int8_pair(words.y >> 8), weights[3]);
    }
  }

  device float *target = partials + slot * M * D;
#pragma unroll
  for (uint d = 0; d < 32; ++d) {
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
