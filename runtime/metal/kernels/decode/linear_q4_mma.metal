#include "metal/kernels/common/q4_sgmatrix.h"

// Apple7/8 (M1/M2) register-matrix Q4 decode projection (the SimdgroupF32
// tile). These GPUs have no bfloat arithmetic and issue simdgroup matrix
// products on the ordinary ALU pipes, so the kernel is bound by instructions
// per MMA rather than by weight bandwidth:
//   A  the weights as exact half: nibble | 0x6400 is 1024 + q (ulp 1), and
//      subtracting 1024 leaves q;
//   B  the shared X^T table widened from bfloat to exact fp32;
//   C  fp32, so every q*x product is exact and only the summation order
//      differs from the sequential tiles.
// One threadgroup covers every request lane of a batch: each unpacked weight
// fragment feeds one MMA per lane, instead of a threadgroup per lane unpacking
// the same weights again. A simdgroup computes 16 columns (8 for each of the
// gate/up streams) for all L eight-row tiles. The table, row sums, K split and
// its partial-sum workspace follow decode_linear_q4_sg; lanes' tiles are
// consecutive in every buffer.
namespace q4sgf {
using namespace q4sg;
enum class Epilogue { Affine, Residual, GateUp };

// c += a x b for exact half weights and fp32 activations.
__attribute__((always_inline)) inline void mma(thread float2 &c, half2 a, float2 b) {
  simdgroup_half8x8 A;
  simdgroup_float8x8 B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

template <Epilogue E, uint L>
__attribute__((always_inline)) inline void decode(device const bfloat *table, device const uchar *w0,
                   device const bfloat *sc0, device const bfloat *bi0,
                   device bfloat *out, device const float *sums,
                   coherent(device) device float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1,
                   device const bfloat *sc1, device const bfloat *bi1,
                   constant Q4Params &p, uint2 tg, uint tid, uint sg, uint lane,
                   threadgroup uint *arrival) {
  constexpr bool gateUp = E == Epilogue::GateUp;
  constexpr uint tileN = gateUp ? 32 : 64;
  const uint N = p.output_size, K = p.input_size, groups = K / 64;
  const uint splits = p.persistent_groups;
  const uint first = tg.y * (groups / splits);
  const uint end = tg.y + 1 == splits ? groups : first + groups / splits;
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * tileN + sg * (gateUp ? 8 : 16);
  const uint tile = base / 256;
  // Fragment f: the gate/up streams share one column; plain fragments are
  // eight columns apart in one stream.
  auto stream = [&](uint f) { return gateUp ? f : 0u; };
  auto column = [&](uint f) { return base + fm + (gateUp ? 0u : f * 8); };
  device const uchar *tile0 = w0 + ulong(tile) * groups * 8192;
  device const uchar *tile1 = w1 + ulong(tile) * groups * 8192;
  uint2 words[2];
  auto load = [&](uint g) __attribute__((always_inline)) {
#pragma unroll
    for (uint f = 0; f < 2; ++f)
      words[f] = *reinterpret_cast<device const uint2 *>(
          (stream(f) ? tile1 : tile0) + ulong(g) * 8192 + (column(f) % 256) * 32 + c * 8);
  };
  float2 acc[2][L];
#pragma unroll
  for (uint f = 0; f < 2; ++f)
#pragma unroll
    for (uint r = 0; r < L; ++r) acc[f][r] = float2(0);
  load(first);
  for (uint g = first; g < end; ++g) {
    uint2 w[2] = {words[0], words[1]};
    if (g + 1 < end) load(g + 1);
    // Zeroed as a whole, as in decode_linear_q4_sg (GPU validation).
    float2 dot[2][L];
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r) dot[f][r] = float2(0);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
      // k-steps 4h .. 4h + 3 of every lane's table: one 16-byte load each.
      vec<bfloat, 8> x[L];
#pragma unroll
      for (uint r = 0; r < L; ++r)
        x[r] = reinterpret_cast<device const vec<bfloat, 8> *>(
            table + ulong(r) * K * kRows + ulong(g) * kXtPerGroup)[(h * 8 + fm) * 4 + c];
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        float2 b[L];
#pragma unroll
        for (uint r = 0; r < L; ++r) b[r] = float2(reinterpret_cast<thread bfloat2 *>(&x[r])[s]);
#pragma unroll
        for (uint f = 0; f < 2; ++f) {
          const uint nibbles = ((h ? w[f].y : w[f].x) >> (4 * s)) & 0x000F000Fu;
          const half2 a = as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h);
#pragma unroll
          for (uint r = 0; r < L; ++r) mma(dot[f][r], a, b[r]);
        }
      }
    }
#pragma unroll
    for (uint f = 0; f < 2; ++f) {
      const ulong prm = (ulong(tile) * groups + g) * 256 + column(f) % 256;
      const float scale = float((stream(f) ? sc1 : sc0)[prm]);
      const float bias = float((stream(f) ? bi1 : bi0)[prm]);
#pragma unroll
      for (uint r = 0; r < L; ++r) {
        device const float *rowSums = sums + ulong(r) * K / 8 + g * kRows + fn;
        acc[f][r] = fma(dot[f][r], scale, acc[f][r]);
        acc[f][r] = fma(float2(rowSums[0], rowSums[1]), bias, acc[f][r]);
      }
    }
  }
  if (splits > 1) {
    // Each lane's tile owns splits x two streams of eight fp32 rows.
    auto slot = [&](uint r, uint s, uint f) {
      return partials + (ulong(r) * splits * 2 + s * 2 + stream(f)) * 8 * N + column(f);
    };
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r) {
        coherent(device) device float *at = slot(r, tg.y, f);
        at[fn * N] = acc[f][r].x;
        at[(fn + 1) * N] = acc[f][r].y;
      }
    // Same protocol as decode_linear_q4_sg: device-scope fences pair through
    // the counter, and the last partition reduces in split order.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope::thread_scope_device);
      *arrival = atomic_fetch_add_explicit(counters + tg.x, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope::thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (*arrival != splits - 1) return;
    float2 total[2][L];
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r) total[f][r] = float2(0);
    for (uint s = 0; s < splits; ++s)
#pragma unroll
      for (uint f = 0; f < 2; ++f)
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          coherent(device) device const float *at = slot(r, s, f);
          total[f][r] += s == tg.y ? acc[f][r] : float2(at[fn * N], at[(fn + 1) * N]);
        }
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r) acc[f][r] = total[f][r];
    if (tid == 0) atomic_store_explicit(counters + tg.x, 0u, memory_order_relaxed);
  }
#pragma unroll
  for (uint r = 0; r < L; ++r) {
    device bfloat *o = out + ulong(r) * kRows * N;
    if (gateUp) {
      const uint n = column(0);
      const float2 gate = float2(bfloat2(acc[0][r])), up = float2(bfloat2(acc[1][r]));
      const float2 value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
      o[fn * N + n] = bfloat(value.x);
      o[(fn + 1) * N + n] = bfloat(value.y);
    } else {
      device const bfloat *res = residual + ulong(r) * kRows * N;
#pragma unroll
      for (uint f = 0; f < 2; ++f) {
        const uint n = column(f);
        float2 value = float2(bfloat2(acc[f][r]));
        if (E == Epilogue::Residual)
          value += float2(float(res[fn * N + n]), float(res[(fn + 1) * N + n]));
        o[fn * N + n] = bfloat(value.x);
        o[(fn + 1) * N + n] = bfloat(value.y);
      }
    }
  }
}
} // namespace q4sgf

#define Q4_SGF_INPUTS \
    device const bfloat *table [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]], \
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
    coherent(device) device float *partials [[buffer(6)]], device atomic_uint *counters [[buffer(7)]]
#define Q4_SGF_THREADS \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

// One instance per batch width: M16/M24/M32 cover two to four lanes.
#define Q4_SGF_KERNELS(SUFFIX, L) \
kernel void decode_linear_q4_sgf##SUFFIX(Q4_SGF_INPUTS, constant Q4Params &p [[buffer(8)]], \
    Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::Affine, L>(table, weights, scales, biases, output, sums, \
      partials, counters, output, weights, scales, biases, p, tg, tid, sg, lane, &arrival); \
} \
kernel void decode_linear_q4_sgf_residual##SUFFIX(Q4_SGF_INPUTS, \
    device const bfloat *residual [[buffer(8)]], constant Q4Params &p [[buffer(9)]], \
    Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::Residual, L>(table, weights, scales, biases, output, sums, \
      partials, counters, residual, weights, scales, biases, p, tg, tid, sg, lane, &arrival); \
} \
kernel void decode_linear_q4_sgf_gate_up##SUFFIX(Q4_SGF_INPUTS, \
    device const uchar *up [[buffer(8)]], device const bfloat *upScales [[buffer(9)]], \
    device const bfloat *upBiases [[buffer(10)]], constant Q4Params &p [[buffer(11)]], \
    Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::GateUp, L>(table, weights, scales, biases, output, sums, \
      partials, counters, output, up, upScales, upBiases, p, tg, tid, sg, lane, &arrival); \
}
Q4_SGF_KERNELS(, 1)
Q4_SGF_KERNELS(_m16, 2)
Q4_SGF_KERNELS(_m24, 3)
Q4_SGF_KERNELS(_m32, 4)
#undef Q4_SGF_KERNELS
#undef Q4_SGF_INPUTS
#undef Q4_SGF_THREADS
