#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include <metal_stdlib>
using namespace metal;

// Register-matrix grouped expert tiles for Apple7/8 (M1/M2), whose MPP
// matmul2d path runs the uint4b x bfloat expert tiles far below the GPU's
// simdgroup MMA rate. The arithmetic is the dense prefill_linear_q4_mma64
// tile's: A = q as exact half (nibble | 0x6400 minus 1024), B = the grouped
// input widened from bf16 to exact fp32, fp32 accumulation, then per quant
// group acc += scale * dot + bias * sum(x). Every q*x product is exact; only
// the summation order differs from the MPP tiles.
//
// Unlike the dense tile, the input sums are not precomputed: each lane already
// holds eight inputs of each of its rows per group, and three xor shuffles over
// the lanes that share a row finish the group sum.
//
// A threadgroup is four simdgroups. The 32-row tile arranges them 2 x 2
// (columns x rows), each simdgroup owning 16 rows as two 8-row fragments; the
// 8-row tile runs all four across columns. Simdgroups whose rows are all past
// the descriptor's live rows return at once, so a ragged prefill tile or a
// sparse decode tile pays for its live row fragments only. Gate/up fuses the
// two projections of the same columns (fragments 0-1 gate, 2-3 up) and writes
// silu(gate) * up with both rounded to bf16, as the shipped tiles do; the down
// pass is one affine projection. Bindings match moe_expert_{gate_up,down}_q4_m8.
namespace moemma {

constexpr constant uint kGroup = 64;      // inputs per quant group
constexpr constant uint kStorageN = 256;  // weight tile width of the packing
constexpr constant uint FN = 4;           // column fragments per simdgroup
constexpr constant uint kSimdgroups = 4;

enum class Pass { GateUp, Down };

template <uint Rows, Pass P>
inline void expert(device const bfloat *grouped_input,
                   device const MoeTileDescriptor *tiles,
                   device const uint *tile_count, device uchar *packed_0,
                   device uchar *packed_1, device uchar *shared_0,
                   device uchar *shared_1, device bfloat *output,
                   constant MoeExpertParams &params, uint2 tg, uint sg,
                   uint lane) {
  static_assert(Rows == 8 || Rows == 32, "8- and 32-row expert tiles");
  constexpr uint SGM = Rows == 32 ? 2 : 1;  // simdgroups across rows
  constexpr uint SGN = kSimdgroups / SGM;   // simdgroups across columns
  constexpr uint FM = Rows / (8 * SGM);     // row fragments per simdgroup
  constexpr bool gateUp = P == Pass::GateUp;
  constexpr uint sgColumns = gateUp ? FN / 2 * 8 : FN * 8;
  constexpr uint tgColumns = SGN * sgColumns;
  static_assert(kStorageN % tgColumns == 0, "a tile never straddles a packed 256-column tile");

  if (tg.y >= *tile_count)
    return;
  const MoeTileDescriptor descriptor = tiles[tg.y];
  const uint sgn = sg % SGN, sgm = sg / SGN;
  const uint firstRow = sgm * FM * 8;
  if (firstRow >= descriptor.rows)
    return;
  const uint N = params.output_size, K = params.input_size, groups = K / kGroup;
  const MoeQ4Slab slab_0 =
      moe_q4_slab(packed_0, shared_0, descriptor.expert, params.experts,
                  params.expert_stride_bytes_0, N, K);
  MoeQ4Slab slab_1 = slab_0;
  if constexpr (gateUp)
    slab_1 = moe_q4_slab(packed_1, shared_1, descriptor.expert, params.experts,
                         params.expert_stride_bytes_1, N, K);

  // simdgroup_matrix lane layout: thread_elements() are M[fm][fn], M[fm][fn + 1].
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint n0 = tg.x * tgColumns + sgn * sgColumns;
  const uint tile = n0 / kStorageN;

  // Fragment i: gate/up pairs fragments i and i + 2 on one column.
  device const uchar *laneWeights[FN];
  device const bfloat *laneScales[FN];
  device const bfloat *laneBiases[FN];
#pragma unroll
  for (uint i = 0; i < FN; ++i) {
    const bool up = gateUp && i >= FN / 2;
    const MoeQ4Slab slab = up ? slab_1 : slab_0;
    const uint n = n0 + (gateUp ? i % (FN / 2) : i) * 8 + fm;
    const ulong column = ulong(tile) * groups * kStorageN + n % kStorageN;
    laneWeights[i] = slab.weights + column * (kGroup / 2) + (fn / 2) * 8;
    laneScales[i] = slab.scales + column;
    laneBiases[i] = slab.biases + column;
  }
  const uint rowBase = firstRow + fn;
  device const bfloat *input =
      grouped_input + (ulong(tg.y) * Rows + rowBase) * K + 8 * fm;
  bool live[FM];
#pragma unroll
  for (uint j = 0; j < FM; ++j) live[j] = firstRow + j * 8 < descriptor.rows;

  float2 acc[FN][FM];
#pragma unroll
  for (uint i = 0; i < FN; ++i)
#pragma unroll
    for (uint j = 0; j < FM; ++j) acc[i][j] = float2(0);

  uint2 w[FN];
#pragma unroll
  for (uint i = 0; i < FN; ++i)
    w[i] = *reinterpret_cast<device const uint2 *>(laneWeights[i]);
  for (uint g = 0; g < groups; ++g) {
    // x[j][e]: the eight bf16 of fragment j's row fn + e for all eight k-steps.
    uint4 x[FM][2];
    float2 sum[FM];
#pragma unroll
    for (uint j = 0; j < FM; ++j) {
#pragma unroll
      for (uint e = 0; e < 2; ++e) {
        x[j][e] = live[j] ? *reinterpret_cast<device const uint4 *>(
                                input + ulong(j * 8 + e) * K + g * kGroup)
                          : uint4(0);
        float s = 0;
#pragma unroll
        for (uint v = 0; v < 4; ++v)
          s += as_type<float>(x[j][e][v] << 16) + as_type<float>(x[j][e][v] & 0xFFFF0000u);
        sum[j][e] = s;
      }
      // Lane bits 1, 2 and 4 select fm: the other 56 inputs of these rows.
      sum[j] += simd_shuffle_xor(sum[j], 2);
      sum[j] += simd_shuffle_xor(sum[j], 4);
      sum[j] += simd_shuffle_xor(sum[j], 16);
    }
    // Nibble j of words x and y, 16 bits apart: low halves serve j < 4.
    uint2 pairs[FN];
#pragma unroll
    for (uint i = 0; i < FN; ++i)
      pairs[i] = uint2((w[i].x & 0xFFFFu) | (w[i].y << 16),
                       (w[i].x >> 16) | (w[i].y & 0xFFFF0000u));
    float2 dot[FN][FM];
#pragma unroll
    for (uint i = 0; i < FN; ++i)
#pragma unroll
      for (uint j = 0; j < FM; ++j) dot[i][j] = float2(0);
#pragma unroll
    for (uint kb = 0; kb < 8; ++kb) {
      simdgroup_float8x8 b[FM];
      const uint shift = (kb & 1) ? 0 : 16;
#pragma unroll
      for (uint j = 0; j < FM; ++j)
        reinterpret_cast<thread float2 &>(b[j].thread_elements()) =
            float2(as_type<float>((x[j][0][kb >> 1] << shift) & 0xFFFF0000u),
                   as_type<float>((x[j][1][kb >> 1] << shift) & 0xFFFF0000u));
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
          if (!live[j])
            continue;
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
      scale[i] = float(laneScales[i][ulong(g) * kStorageN]);
      bias[i] = float(laneBiases[i][ulong(g) * kStorageN]);
    }
    if (g + 1 < groups) {
#pragma unroll
      for (uint i = 0; i < FN; ++i)
        w[i] = *reinterpret_cast<device const uint2 *>(
            laneWeights[i] + ulong(g + 1) * kStorageN * (kGroup / 2));
    }
#pragma unroll
    for (uint j = 0; j < FM; ++j)
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        acc[i][j] = fma(dot[i][j], scale[i], acc[i][j]);
        acc[i][j] = fma(sum[j], bias[i], acc[i][j]);
      }
  }

  device bfloat *tileOutput = output + ulong(tg.y) * Rows * N;
#pragma unroll
  for (uint j = 0; j < FM; ++j) {
    if (!live[j])
      continue;
#pragma unroll
    for (uint e = 0; e < 2; ++e) {
      const uint m = rowBase + j * 8 + e;
      if (gateUp) {
#pragma unroll
        for (uint i = 0; i < FN / 2; ++i) {
          const float gate = float(bfloat(acc[i][j][e]));
          const float up = float(bfloat(acc[i + FN / 2][j][e]));
          const float value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
          tileOutput[ulong(m) * N + n0 + i * 8 + fm] = bfloat(value);
        }
      } else {
#pragma unroll
        for (uint i = 0; i < FN; ++i)
          tileOutput[ulong(m) * N + n0 + i * 8 + fm] = bfloat(acc[i][j][e]);
      }
    }
  }
}

} // namespace moemma

#define MOE_MMA_THREADS                                                        \
  uint2 group [[threadgroup_position_in_grid]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]]

#define MOE_MMA_KERNELS(Rows)                                                  \
  kernel void moe_expert_gate_up_q4_mma_m##Rows(                               \
      device bfloat *grouped_input [[buffer(0)]],                              \
      device const MoeTileDescriptor *tiles [[buffer(1)]],                     \
      device const uint *tile_count [[buffer(2)]],                             \
      device uchar *gate_packed [[buffer(3)]],                                 \
      device uchar *up_packed [[buffer(4)]],                                   \
      device uchar *shared_gate [[buffer(5)]],                                 \
      device uchar *shared_up [[buffer(6)]],                                   \
      device bfloat *output [[buffer(7)]],                                     \
      constant MoeExpertParams &params [[buffer(8)]], MOE_MMA_THREADS) {       \
    moemma::expert<Rows, moemma::Pass::GateUp>(                                \
        grouped_input, tiles, tile_count, gate_packed, up_packed, shared_gate, \
        shared_up, output, params, group, simd_group, simd_lane);              \
  }                                                                            \
  kernel void moe_expert_down_q4_mma_m##Rows(                                  \
      device bfloat *grouped_input [[buffer(0)]],                              \
      device const MoeTileDescriptor *tiles [[buffer(1)]],                     \
      device const uint *tile_count [[buffer(2)]],                             \
      device uchar *down_packed [[buffer(3)]],                                 \
      device uchar *shared_down [[buffer(4)]],                                 \
      device bfloat *output [[buffer(5)]],                                     \
      constant MoeExpertParams &params [[buffer(6)]], MOE_MMA_THREADS) {       \
    moemma::expert<Rows, moemma::Pass::Down>(                                  \
        grouped_input, tiles, tile_count, down_packed, down_packed,            \
        shared_down, shared_down, output, params, group, simd_group,           \
        simd_lane);                                                            \
  }

MOE_MMA_KERNELS(8)
MOE_MMA_KERNELS(32)
#undef MOE_MMA_KERNELS
#undef MOE_MMA_THREADS
