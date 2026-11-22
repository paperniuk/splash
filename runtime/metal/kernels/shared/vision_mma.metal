// Vision encoder GEMMs and attention for Apple7/8 (M1/M2) on simdgroup MMAs:
// the bindings, grids and outputs of vision_gemm_* and vision_attention
// (vision.metal), whose MPP matmul2d runs far below the MMA rate there. Both
// operands are bf16 widened to exact fp32 and accumulate in fp32, so every
// product is exact and only summation order differs from the MPP kernels.

#include "metal/abi/Vision.h"
#include "metal/kernels/common/sgmatrix.h"
#include <metal_stdlib>

using namespace metal;

namespace vision_mma {
using sgmatrix::Lane;
using sgmatrix::lane_map;
using sgmatrix::te;

enum Activation { None, GeluTanh, GeluErf };

// c += a x b on fp32 fragments.
__attribute__((always_inline)) inline void mma(thread float2 &c, float2 a, float2 b) {
  simdgroup_float8x8 A, B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

// Element e of four bf16 packed in a uint2, widened to exact fp32.
__attribute__((always_inline)) inline float widen(uint2 w, uint e) {
  const uint word = e < 2 ? w.x : w.y;
  return as_type<float>((e & 1) ? word & 0xFFFF0000u : word << 16);
}

// Abramowitz & Stegun 7.1.26, as vision_erf.
inline float erf_approx(float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float polynomial = t * (0.254829592f + t * (-0.284496736f +
      t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}

// output[M, N] = input[M, K] x weights[N, K]^T + bias with the MPP kernel's
// epilogue. Eight simdgroups of 32 rows x 32 columns (SGM x SGN) cover the
// TileM x TileN tile; each computes C^T fragments (A = weights, B = input^T)
// and walks K 32 inputs at a time. K order within a block: B row fm at step j
// is input 4 fm + j, so lane (fm, fn) reads one 8-byte run of each of its two
// rows; A column fn at step j is input 4 fn + j, one 16-byte run of its
// weight row for both columns fn and fn + 1.
template <ushort TileM, ushort TileN, Activation Act, bool AddResidual, typename Output>
inline void gemm(device const bfloat *input, device const bfloat *weights,
                 device const bfloat *bias, device Output *output,
                 device const Output *residual, uint N, uint K, uint2 tg,
                 uint sg, uint lane) {
  constexpr uint FM = 4, FN = 4, SGN = TileN / (FN * 8);
  static_assert((TileM / (FM * 8)) * SGN == 8, "eight simdgroups per tile");
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint row0 = tg.x * TileM + (sg / SGN) * FM * 8 + fn;
  const uint col0 = tg.y * TileN + (sg % SGN) * FN * 8 + fm;
  device const bfloat *x = input + ulong(row0) * K + 4 * fm;
  device const bfloat *w = weights + ulong(col0) * K + 4 * fn;

  float2 acc[FN][FM];
#pragma unroll
  for (uint i = 0; i < FN; ++i)
#pragma unroll
    for (uint j = 0; j < FM; ++j) acc[i][j] = float2(0.0f);

  for (uint k = 0; k < K; k += 32) {
    uint2 xv[FM][2];
    uint4 wv[FN];
#pragma unroll
    for (uint j = 0; j < FM; ++j)
#pragma unroll
      for (uint e = 0; e < 2; ++e)
        xv[j][e] = *reinterpret_cast<device const uint2 *>(x + ulong(8 * j + e) * K + k);
#pragma unroll
    for (uint i = 0; i < FN; ++i)
      wv[i] = *reinterpret_cast<device const uint4 *>(w + ulong(8 * i) * K + k);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      float2 b[FM];
#pragma unroll
      for (uint j = 0; j < FM; ++j) b[j] = float2(widen(xv[j][0], s), widen(xv[j][1], s));
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        const float2 a(widen(wv[i].xy, s), widen(wv[i].zw, s));
#pragma unroll
        for (uint j = 0; j < FM; ++j) mma(acc[i][j], a, b[j]);
      }
    }
  }

#pragma unroll
  for (uint i = 0; i < FN; ++i) {
    const uint n = col0 + 8 * i;
    const float b = float(bias[n]);
#pragma unroll
    for (uint j = 0; j < FM; ++j)
#pragma unroll
      for (uint e = 0; e < 2; ++e) {
        const ulong o = ulong(row0 + 8 * j + e) * N + n;
        float value = acc[i][j][e] + b;
        if (Act == GeluTanh)
          value = 0.5f * value * (1.0f + precise::tanh(0.7978845608028654f *
                                                        (value + 0.044715f * value * value * value)));
        else if (Act == GeluErf)
          value = 0.5f * value * (1.0f + erf_approx(value * 0.7071067811865476f));
        if (AddResidual) value += float(residual[o]);
        output[o] = Output(value);
      }
  }
}

// Non-causal flash attention of 64 query rows of one head, as
// vision_attention: eight simdgroups of eight rows, keys in blocks of 32
// staged in threadgroup memory as fp32. S^T = K Q^T leaves each lane's scores
// in the fragment layout PV = V^T P^T takes as its B operand, so the online
// softmax runs in registers; P is rounded to bf16 as the MPP kernel rounds
// its probabilities, and the row sums keep the unrounded exponents.
constexpr constant uint kRows = SPLASH_VISION_ATTENTION_MMA_ROWS, kKeys = 32, kHeadDim = 72, kQkDim = 80;
constexpr constant uint kKeyStride = 84, kValueStride = 76;  // bank-spread rows
constexpr constant uint kThreads = SPLASH_VISION_ATTENTION_MMA_THREADS;
static_assert(kThreads / 32 * 8 == kRows, "eight rows per simdgroup");

inline void attention(device const bfloat *queries, device const bfloat *keys,
                      device const bfloat *values, device bfloat *output,
                      constant VisionAttentionParams &p, uint2 tg, uint sg, uint lane,
                      threadgroup float *stage_k, threadgroup float *stage_v) {
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint thread_index = sg * 32 + lane;
  const uint head = tg.y, row = tg.x * kRows + sg * 8 + fn;
  const ulong plane = ulong(head) * p.padded_tokens;
  device const bfloat *q = queries + (plane + row) * kQkDim + fm;
  device const bfloat *k = keys + plane * kQkDim;
  device const bfloat *v = values + plane * kHeadDim;

  // Q^T fragments: dimension 8 t + fm of rows (row, row + 1).
  float2 query[kQkDim / 8];
#pragma unroll
  for (uint t = 0; t < kQkDim / 8; ++t) query[t] = float2(q[8 * t], q[kQkDim + 8 * t]);
  // O^T fragments: dimension 8 d + fm of rows (row, row + 1).
  float2 out[kHeadDim / 8];
#pragma unroll
  for (uint d = 0; d < kHeadDim / 8; ++d) out[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);

  const uint context = p.tokens;
  for (uint base = 0; base < context; base += kKeys) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 32 x 80 keys and 32 x 72 values: 320 and 288 runs of eight bf16.
    for (uint u = thread_index; u < kKeys * (kQkDim + kHeadDim) / 8; u += kThreads) {
      const bool key = u < kKeys * kQkDim / 8;
      const uint e = 8 * (key ? u : u - kKeys * kQkDim / 8);
      const uint dim = key ? kQkDim : kHeadDim;
      const uint4 w = *reinterpret_cast<device const uint4 *>(
          (key ? k + ulong(base) * kQkDim : v + ulong(base) * kHeadDim) + e);
      threadgroup float *dst = key ? stage_k + (e / dim) * kKeyStride + e % dim
                                   : stage_v + (e / dim) * kValueStride + e % dim;
      *reinterpret_cast<threadgroup float4 *>(dst) =
          float4(widen(w.xy, 0), widen(w.xy, 1), widen(w.xy, 2), widen(w.xy, 3));
      *reinterpret_cast<threadgroup float4 *>(dst + 4) =
          float4(widen(w.zw, 0), widen(w.zw, 1), widen(w.zw, 2), widen(w.zw, 3));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S^T tile s: keys base + 8 s + fm x rows (row, row + 1).
    float2 score[kKeys / 8];
#pragma unroll
    for (uint s = 0; s < kKeys / 8; ++s) {
      score[s] = float2(0.0f);
      threadgroup const float *kr = stage_k + (8 * s + fm) * kKeyStride + fn;
#pragma unroll
      for (uint t = 0; t < kQkDim / 8; ++t)
        mma(score[s], *reinterpret_cast<threadgroup const float2 *>(kr + 8 * t), query[t]);
    }

    float2 local_max(-INFINITY);
#pragma unroll
    for (uint s = 0; s < kKeys / 8; ++s) {
      const bool live = base + 8 * s + fm < context;
      score[s] = live ? score[s] * p.scale : float2(-INFINITY);
      local_max = max(local_max, score[s]);
    }
    // Lane bits 1, 2 and 4 select fm: the other keys of these rows.
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next_max = max(row_max, local_max);
    float2 local_sum(0.0f);
#pragma unroll
    for (uint s = 0; s < kKeys / 8; ++s) {
      const float2 e = float2(fast::exp(score[s].x - next_max.x), fast::exp(score[s].y - next_max.y));
      local_sum += e;
      score[s] = float2(float(bfloat(e.x)), float(bfloat(e.y)));
    }
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 16);
    const float2 scale(row_max.x == -INFINITY ? 0.0f : fast::exp(row_max.x - next_max.x),
                       row_max.y == -INFINITY ? 0.0f : fast::exp(row_max.y - next_max.y));
    row_sum = row_sum * scale + local_sum;
    row_max = next_max;

    // O^T tile d = O^T tile d x scale + V^T (dimensions 8 d + fm) x P^T.
#pragma unroll
    for (uint d = 0; d < kHeadDim / 8; ++d) {
      float2 partial(0.0f);
      threadgroup const float *vr = stage_v + fn * kValueStride + 8 * d + fm;
#pragma unroll
      for (uint s = 0; s < kKeys / 8; ++s)
        mma(partial, float2(vr[8 * s * kValueStride], vr[(8 * s + 1) * kValueStride]), score[s]);
      out[d] = out[d] * scale + partial;
    }
  }

  device bfloat *o = output + (plane + row) * kHeadDim + fm;
#pragma unroll
  for (uint d = 0; d < kHeadDim / 8; ++d) {
    o[8 * d] = bfloat(out[d].x / row_sum.x);
    o[kHeadDim + 8 * d] = bfloat(out[d].y / row_sum.y);
  }
}

} // namespace vision_mma

#define VISION_GEMM_MMA_KERNEL(name, tile_m, tile_n, activation, add_residual, output_type) \
  kernel void name(device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]], \
                   device bfloat *bias [[buffer(2)]], device output_type *output [[buffer(3)]], \
                   device output_type *residual [[buffer(4)]],                              \
                   constant VisionGemmParams &params [[buffer(5)]],                         \
                   uint2 tg [[threadgroup_position_in_grid]],                               \
                   uint sg [[simdgroup_index_in_threadgroup]],                              \
                   uint lane [[thread_index_in_simdgroup]]) {                               \
    vision_mma::gemm<tile_m, tile_n, vision_mma::activation, add_residual, output_type>(    \
        input, weights, bias, output, residual, params.output_size, params.input_size, tg,  \
        sg, lane);                                                                          \
  }

VISION_GEMM_MMA_KERNEL(vision_gemm_mma_m64n128, 64, 128, None, false, bfloat)
// The fp32 residual stream of vision_gemm_m64n128_residual.
VISION_GEMM_MMA_KERNEL(vision_gemm_mma_m64n128_residual, 64, 128, None, true, float)
VISION_GEMM_MMA_KERNEL(vision_gemm_mma_m64n128_gelu_tanh, 64, 128, GeluTanh, false, bfloat)
VISION_GEMM_MMA_KERNEL(vision_gemm_mma_m32n256, 32, 256, None, false, bfloat)
VISION_GEMM_MMA_KERNEL(vision_gemm_mma_m32n256_gelu_erf, 32, 256, GeluErf, false, bfloat)
#undef VISION_GEMM_MMA_KERNEL

kernel void vision_attention_mma(device bfloat *queries [[buffer(0)]],
                                 device bfloat *keys [[buffer(1)]],
                                 device bfloat *values [[buffer(2)]],
                                 device bfloat *output [[buffer(3)]],
                                 constant VisionAttentionParams &params [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float stage_k[vision_mma::kKeys * vision_mma::kKeyStride];
  threadgroup float stage_v[vision_mma::kKeys * vision_mma::kValueStride];
  vision_mma::attention(queries, keys, values, output, params, tg, sg, lane, stage_k, stage_v);
}
