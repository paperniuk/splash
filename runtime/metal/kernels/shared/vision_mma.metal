// Apple7/8 dense projections and attention on fp32 simdgroup matrices.
// The bf16 operands widen exactly; the reduction order differs from MPP.

#include "metal/abi/DenseAttention.h"
#include "metal/abi/DenseLinear.h"
#include "metal/kernels/common/activation.h"
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
                 constant DenseLinearParams &p, uint2 tg,
                 uint sg, uint lane) {
  const uint N = p.output_size, K = p.input_size;
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
        const uint row = row0 + 8 * j + e;
        if (row >= p.rows) continue;
        const ulong o = ulong(row) * N + n;
        float value = acc[i][j][e] + b;
        if constexpr (Act == GeluTanh) value = splash_gelu_tanh(value);
        if constexpr (Act == GeluErf) value = splash_gelu_erf(value);
        if constexpr (AddResidual) value += float(output[o]);
        output[o] = Output(value);
      }
  }
}

// Non-causal flash attention of 64 query rows of one head, as
// dense_attention: eight simdgroups of eight rows, keys in blocks of 32
// staged in threadgroup memory as fp32. S^T = K Q^T leaves each lane's scores
// in the fragment layout PV = V^T P^T takes as its B operand, so the online
// softmax runs in registers; P is rounded to bf16 as the MPP kernel rounds
// its probabilities, and the row sums keep the unrounded exponents.
constexpr constant uint kRows = SPLASH_DENSE_ATTENTION_MMA_ROWS, kKeys = 32, kHeadDim = 72, kQkDim = 80;
constexpr constant uint kKeyStride = 84;  // bank-spread rows
// Values stage as pairs of rows interleaved, row 2 i + r of dimension c at
// i kPairStride + 2 c + r: the two rows of a dimension PV takes are one
// float2, half the threadgroup loads (and validated checks) of two scalars.
constexpr constant uint kPairStride = 2 * kHeadDim;
constexpr constant uint kThreads = SPLASH_DENSE_ATTENTION_MMA_THREADS;
static_assert(kThreads / 32 * 8 == kRows, "eight rows per simdgroup");
// Keys and values stage as 16-byte runs of eight bf16, 320 runs of keys and
// 144 pairs of runs of values a block: a run never crosses a row, and lands
// on 16-byte aligned threadgroup floats. DenseAttention::add refuses key and
// value strides that are not whole runs.
constexpr constant uint kKeyRuns = kKeys * kQkDim / 8, kValuePairRuns = kKeys / 2 * kHeadDim / 8;
static_assert(kQkDim % 8 == 0 && kHeadDim % 8 == 0, "rows of whole runs");
static_assert(kKeyStride % 4 == 0 && kPairStride % 4 == 0, "float4-aligned staged rows");

inline void attention(device const bfloat *queries, device const bfloat *keys,
                      device const bfloat *values, device bfloat *output,
                      constant DenseAttentionParams &p, uint2 tg, uint sg, uint lane,
                      threadgroup float *stage_k, threadgroup float *stage_v) {
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint thread_index = sg * 32 + lane;
  const uint head = tg.y, row = tg.x * kRows + sg * 8 + fn;
  device const bfloat *q = queries + ulong(head) * p.query_head_stride + ulong(row) * p.query_row_stride + fm;
  device const bfloat *k = keys + ulong(head) * p.key_head_stride;
  device const bfloat *v = values + ulong(head) * p.value_head_stride;

  // Q^T fragments: dimension 8 t + fm of rows (row, row + 1).
  float2 query[kQkDim / 8];
#pragma unroll
  for (uint t = 0; t < kQkDim / 8; ++t) query[t] = float2(q[8 * t], q[p.query_row_stride + 8 * t]);
  // O^T fragments: dimension 8 d + fm of rows (row, row + 1).
  float2 out[kHeadDim / 8];
#pragma unroll
  for (uint d = 0; d < kHeadDim / 8; ++d) out[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);

  const uint context = p.rows;
  for (uint base = 0; base < context; base += kKeys) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One 16-byte load a run: eight scalar loads cost validated runs past
    // the 120 s command limit at the largest grid.
    for (uint u = thread_index; u < kKeyRuns + kValuePairRuns; u += kThreads) {
      if (u < kKeyRuns) {
        const uint row = 8 * u / kQkDim, column = 8 * u % kQkDim;
        const uint4 w = *reinterpret_cast<device const uint4 *>(k + ulong(base + row) * p.key_row_stride + column);
        threadgroup float *dst = stage_k + row * kKeyStride + column;
        *reinterpret_cast<threadgroup float4 *>(dst) =
            float4(widen(w.xy, 0), widen(w.xy, 1), widen(w.xy, 2), widen(w.xy, 3));
        *reinterpret_cast<threadgroup float4 *>(dst + 4) =
            float4(widen(w.zw, 0), widen(w.zw, 1), widen(w.zw, 2), widen(w.zw, 3));
      } else {
        const uint pair = 8 * (u - kKeyRuns) / kHeadDim, column = 8 * (u - kKeyRuns) % kHeadDim;
        device const bfloat *src = v + ulong(base + 2 * pair) * p.value_row_stride + column;
        const uint4 a = *reinterpret_cast<device const uint4 *>(src);
        const uint4 b = *reinterpret_cast<device const uint4 *>(src + p.value_row_stride);
        threadgroup float4 *dst = reinterpret_cast<threadgroup float4 *>(stage_v + pair * kPairStride + 2 * column);
        dst[0] = float4(widen(a.xy, 0), widen(b.xy, 0), widen(a.xy, 1), widen(b.xy, 1));
        dst[1] = float4(widen(a.xy, 2), widen(b.xy, 2), widen(a.xy, 3), widen(b.xy, 3));
        dst[2] = float4(widen(a.zw, 0), widen(b.zw, 0), widen(a.zw, 1), widen(b.zw, 1));
        dst[3] = float4(widen(a.zw, 2), widen(b.zw, 2), widen(a.zw, 3), widen(b.zw, 3));
      }
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
      // Rows fn + 8 s and fn + 8 s + 1, the pair fn / 2 + 4 s, of dimension 8 d + fm.
      threadgroup const float2 *vr =
          reinterpret_cast<threadgroup const float2 *>(stage_v + (fn / 2) * kPairStride) + 8 * d + fm;
#pragma unroll
      for (uint s = 0; s < kKeys / 8; ++s)
        mma(partial, vr[4 * s * kHeadDim], score[s]);
      out[d] = out[d] * scale + partial;
    }
  }

  if (row >= p.rows) return;
  device bfloat *o = output + ulong(head) * p.output_head_stride + ulong(row) * p.output_row_stride + fm;
#pragma unroll
  for (uint d = 0; d < kHeadDim / 8; ++d) {
    o[8 * d] = bfloat(out[d].x / row_sum.x);
    if (row + 1 < p.rows) o[p.output_row_stride + 8 * d] = bfloat(out[d].y / row_sum.y);
  }
}

} // namespace vision_mma

#define DENSE_MMA_GEMM(name, tile_m, tile_n, activation, add_residual, output_type) \
  kernel void name(device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]], \
                   device bfloat *bias [[buffer(2)]], device output_type *output [[buffer(3)]], \
                   constant DenseLinearParams &params [[buffer(4)]], \
                   uint2 tg [[threadgroup_position_in_grid]], \
                   uint sg [[simdgroup_index_in_threadgroup]], \
                   uint lane [[thread_index_in_simdgroup]]) { \
    vision_mma::gemm<tile_m, tile_n, vision_mma::activation, add_residual, output_type>( \
        input, weights, bias, output, params, tg, sg, lane); \
  }

#define DENSE_MMA_TILE(name, rows, columns) \
  DENSE_MMA_GEMM(name, rows, columns, None, false, bfloat) \
  DENSE_MMA_GEMM(name##_residual, rows, columns, None, true, float) \
  DENSE_MMA_GEMM(name##_gelu_tanh, rows, columns, GeluTanh, false, bfloat) \
  DENSE_MMA_GEMM(name##_gelu_erf, rows, columns, GeluErf, false, bfloat)

DENSE_MMA_TILE(dense_linear_mma_m64n128, 64, 128)
DENSE_MMA_TILE(dense_linear_mma_m32n256, 32, 256)
#undef DENSE_MMA_TILE
#undef DENSE_MMA_GEMM

kernel void dense_attention_mma_72(device bfloat *queries [[buffer(0)]],
                                 device bfloat *keys [[buffer(1)]],
                                 device bfloat *values [[buffer(2)]],
                                 device bfloat *output [[buffer(3)]],
                                 constant DenseAttentionParams &params [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float stage_k[vision_mma::kKeys * vision_mma::kKeyStride];
  threadgroup float stage_v[vision_mma::kKeys / 2 * vision_mma::kPairStride];
  vision_mma::attention(queries, keys, values, output, params, tg, sg, lane, stage_k, stage_v);
}
