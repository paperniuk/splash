#include "metal/abi/Linear.h"
#include <metal_stdlib>
using namespace metal;

// Register-matrix Q4 prefill projection for Apple7/8 (M1/M2), whose MPP
// matmul2d path runs the uint4b x bfloat tile far below the GPU's simdgroup
// MMA rate. A threadgroup of four simdgroups owns one 32-row prefill tile and
// 64 output columns; each simdgroup computes 32 columns x 16 rows as 4 x 2
// 8x8 fragments and walks K one 64-input quant group at a time:
//   dot     q x^T for the group with 8x8x8 MMAs, A = q as exact half,
//           B = x as exact fp32 (bf16 widened), fp32 accumulation;
//   affine  acc += scale * dot + bias * sum(x), the shipped per-group form.
// Every q*x product is exact, as in the fp32 decode tile. Both operands are
// loaded straight from device memory into each lane's fragment registers, so
// the kernel uses no threadgroup memory and no barriers; residency is bounded
// only by registers.
//
// K order within a group: row r of every B fragment at k-step j is physical
// k = 8 r + j, so lane (fm, fn) reads one contiguous 16-byte run of each of
// its two activation rows. The matching A lane (fm, fn) needs nibble j of
// both words of its eight-byte weight chunk; two repacks per group place each
// pair 16 bits apart for the decode tile's shift-and-mask. Bindings match the
// MPP prefill kernels of the same epilogue, including the 32-row input sums.
namespace q4mm {

constexpr constant uint kRows = 32;       // prompt rows per threadgroup
constexpr constant uint kGroup = 64;      // inputs per quant group
constexpr constant uint kStorageN = 256;  // weight tile width of the packing
constexpr constant uint FN = 4, FM = 2;   // fragments per simdgroup: columns, rows
constexpr constant uint SGN = 2, SGM = 2; // simdgroups per threadgroup: columns, rows
constexpr constant uint kColumns = SGN * FN * 8;
constexpr constant uint kSimdgroups = SGN * SGM;
static_assert(SGM * FM * 8 == kRows && kStorageN % kColumns == 0 && kColumns % kGroup == 0);

enum class Epilogue { Affine, Residual, UpWithGate };

template <Epilogue E>
inline void project(device const bfloat *input, device const uchar *weights,
                    device const bfloat *scales, device const bfloat *biases,
                    device bfloat *output, device const bfloat *auxiliary,
                    device const float *sums, uint N, uint K, uint2 tg,
                    uint sg, uint lane) {
  const uint groups = K / kGroup;
  const uint n0 = tg.y * kColumns;
  const uint tile = n0 / kStorageN;
  const uint sgn = sg % SGN, sgm = sg / SGN;
  // simdgroup_matrix lane layout: thread_elements() are M[fm][fn], M[fm][fn + 1].
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint nBase = sgn * FN * 8;
  const uint rowBase = sgm * FM * 8 + fn;  // tile row of element 0 of fragment 0
  const uint nLocal = (n0 + nBase) % kStorageN + fm;
  device const uchar *laneWeights = weights +
      (ulong(tile) * groups * kStorageN + nLocal) * (kGroup / 2) + (fn / 2) * 8;
  device const bfloat *laneScales = scales + ulong(tile) * groups * kStorageN + nLocal;
  device const bfloat *laneBiases = biases + ulong(tile) * groups * kStorageN + nLocal;
  input += (ulong(tg.x) * kRows + rowBase) * K + 8 * fm;
  sums += ulong(tg.x) * kRows * groups + rowBase;
  output += ulong(tg.x) * kRows * N;
  auxiliary += ulong(tg.x) * kRows * N;

  float2 acc[FN][FM];
#pragma unroll
  for (uint i = 0; i < FN; ++i)
#pragma unroll
    for (uint j = 0; j < FM; ++j) acc[i][j] = float2(0);

  uint2 w[FN];
#pragma unroll
  for (uint i = 0; i < FN; ++i)
    w[i] = *reinterpret_cast<device const uint2 *>(laneWeights + i * 8 * 32);
  for (uint g = 0; g < groups; ++g) {
    // x[j][e]: the eight bf16 of fragment j's row fn + e for all eight k-steps.
    uint4 x[FM][2];
#pragma unroll
    for (uint j = 0; j < FM; ++j)
#pragma unroll
      for (uint e = 0; e < 2; ++e)
        x[j][e] = *reinterpret_cast<device const uint4 *>(
            input + ulong(j * 8 + e) * K + g * kGroup);
    // Nibble j of words x and y, 16 bits apart: low halves serve j < 4.
    uint2 pairs[FN];
#pragma unroll
    for (uint i = 0; i < FN; ++i)
      pairs[i] = uint2((w[i].x & 0xFFFFu) | (w[i].y << 16),
                       (w[i].x >> 16) | (w[i].y & 0xFFFF0000u));
    // Accumulators stay plain float2 registers; matrices exist only per MMA.
    float2 dot[FN][FM];
#pragma unroll
    for (uint i = 0; i < FN; ++i)
#pragma unroll
      for (uint j = 0; j < FM; ++j) dot[i][j] = float2(0);
#pragma unroll
    for (uint kb = 0; kb < 8; ++kb) {
      simdgroup_float8x8 b[FM];
#pragma unroll
      for (uint j = 0; j < FM; ++j) {
        const uint shift = (kb & 1) ? 0 : 16;
        reinterpret_cast<thread float2 &>(b[j].thread_elements()) =
            float2(as_type<float>((x[j][0][kb >> 1] << shift) & 0xFFFF0000u),
                   as_type<float>((x[j][1][kb >> 1] << shift) & 0xFFFF0000u));
      }
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        const uint word = kb < 4 ? pairs[i].x : pairs[i].y;
        const uint nibbles = (word >> (4 * (kb & 3))) & 0x000F000Fu;
        // 0x6400 is half 1024, whose ulp is 1: the OR adds q exactly.
        simdgroup_half8x8 a;
        reinterpret_cast<thread half2 &>(a.thread_elements()) =
            as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h);
#pragma unroll
        for (uint j = 0; j < FM; ++j) {
          simdgroup_float8x8 c, d;
          reinterpret_cast<thread float2 &>(c.thread_elements()) = dot[i][j];
          simdgroup_multiply_accumulate(d, a, b[j], c);
          dot[i][j] = reinterpret_cast<thread float2 &>(d.thread_elements());
        }
      }
    }
    float scale[FN], bias[FN];
#pragma unroll
    for (uint i = 0; i < FN; ++i) {
      scale[i] = float(laneScales[ulong(g) * kStorageN + i * 8]);
      bias[i] = float(laneBiases[ulong(g) * kStorageN + i * 8]);
    }
    if (g + 1 < groups) {
#pragma unroll
      for (uint i = 0; i < FN; ++i)
        w[i] = *reinterpret_cast<device const uint2 *>(
            laneWeights + (ulong(g + 1) * kStorageN + i * 8) * 32);
    }
#pragma unroll
    for (uint j = 0; j < FM; ++j) {
      const float2 sum = float2(sums[g * kRows + j * 8], sums[g * kRows + j * 8 + 1]);
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        acc[i][j] = fma(dot[i][j], scale[i], acc[i][j]);
        acc[i][j] = fma(sum, bias[i], acc[i][j]);
      }
    }
  }

#pragma unroll
  for (uint i = 0; i < FN; ++i) {
    const uint n = n0 + nBase + i * 8 + fm;
#pragma unroll
    for (uint j = 0; j < FM; ++j) {
#pragma unroll
      for (uint e = 0; e < 2; ++e) {
        const uint m = rowBase + j * 8 + e;
        float value = float(bfloat(acc[i][j][e]));
        if (E == Epilogue::UpWithGate) {
          const float gate = float(auxiliary[m * N + n]);
          value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
        } else if (E == Epilogue::Residual) {
          value += float(auxiliary[m * N + n]);
        }
        output[m * N + n] = bfloat(value);
      }
    }
  }
}

// Row sums of the fused up output: the next (down) projection's input sums,
// in the prefill 32-row tile layout. Reads back this threadgroup's own stores.
inline void write_output_sums(device const bfloat *output, device float *outputSums,
                              uint N, uint2 tg, uint sg, uint lane) {
  constexpr uint Groups = kColumns / kGroup;
  threadgroup_barrier(mem_flags::mem_device);
  output += ulong(tg.x) * kRows * N;
  outputSums += ulong(tg.x) * kRows * (N / kGroup);
  const uint n0 = tg.y * kColumns;
  for (uint task = sg; task < kRows * Groups; task += kSimdgroups) {
    const uint row = task / Groups, local = task % Groups;
    const uint origin = row * N + n0 + local * kGroup + lane;
    const float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (lane == 0) outputSums[(n0 / kGroup + local) * kRows + row] = sum;
  }
}

} // namespace q4mm

#define Q4MM_THREADS                                                           \
  uint2 tg [[threadgroup_position_in_grid]],                                   \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]

kernel void prefill_linear_q4_mma64(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4PrefillParams &p [[buffer(6)]], Q4MM_THREADS) {
  q4mm::project<q4mm::Epilogue::Affine>(input, weights, scales, biases, output, output,
                                        sums, p.output_size, p.input_size, tg, sg, lane);
}

kernel void prefill_linear_q4_mma64_residual(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], constant Q4PrefillParams &p [[buffer(7)]],
    Q4MM_THREADS) {
  q4mm::project<q4mm::Epilogue::Residual>(input, weights, scales, biases, output, residual,
                                          sums, p.output_size, p.input_size, tg, sg, lane);
}

kernel void prefill_linear_q4_mma64_up_silu_sums(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], device float *outputSums [[buffer(7)]],
    constant Q4PrefillParams &p [[buffer(8)]], Q4MM_THREADS) {
  q4mm::project<q4mm::Epilogue::UpWithGate>(input, weights, scales, biases, output, gate,
                                            sums, p.output_size, p.input_size, tg, sg, lane);
  q4mm::write_output_sums(output, outputSums, p.output_size, tg, sg, lane);
}

#undef Q4MM_THREADS
