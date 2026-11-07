// GGUF quantized GEMMs for Apple7/8 (M1/M2) on simdgroup MMAs (kernels/common/gguf_mma_tile.h): the bindings,
// grids and outputs of the staged kernels of gguf_linear.metal, which run MPP matmul2d there. Decode tiles hold 8, 16
// or 32 rows in two simdgroups of 32 columns, each over its own stage; prefill chunks of more than 32 rows run 64-row
// tiles of four simdgroups over a shared stage.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_mma_tile.h"

using namespace gguf_mma;

// The decode kernels: grid (column tiles, K partitions), two simdgroups, as gguf_decode.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_mma(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                            device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                            device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                            device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                            uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint FM = Rows / 8;
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  const uint thread_index = simd_group * 32 + simd_lane;
  quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup uint arrival;
  const uint per = p.input_size / kStep / p.splits, origin = group.x * GGUF_TILE_COLUMNS + simd_group * kColumns,
             column0 = p.out_offset + origin;
  float2 acc[FN][FM];
  zero(acc);
  accumulate<F, FM>(input, w0, w1, meta, p.input_size, origin, stage + simd_group * kSimdgroupStage, tl, simd_lane,
                    group.y * per, (group.y + 1) * per, acc);
  store_sums<Rows>(acc, sgmatrix::lane_map(simd_lane), p.splits, group.y, partials,
                   counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0, thread_index, &arrival,
                   [&](uint row, uint column, float v) {
                     const ulong o = ulong(row) * p.out_stride + column0 + column;
                     if constexpr (Ep == EpResidual) v += float(aux[o]);
                     if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[o]));
                     output[o] = Out(v);
                   });
}
template <class Out>
using GgufDecodeMmaKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                                 device coherent(device) float *, device atomic_uint *, device bfloat *,
                                 constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE_MMA(F, f, R, ep, Ep, Out)                                                                    \
  template [[host_name("gguf_decode_mma_" #f "_m" #R "_" #ep)]] kernel GgufDecodeMmaKernel<Out>                 \
      gguf_decode_mma<F, R, Ep, Out>;
#define GGUF_DECODE_MMA_ROWS(F, f, ep, Ep, Out) \
  GGUF_DECODE_MMA(F, f, 8, ep, Ep, Out) GGUF_DECODE_MMA(F, f, 16, ep, Ep, Out) GGUF_DECODE_MMA(F, f, 32, ep, Ep, Out)
#define GGUF_DECODE_MMA_FORMAT(F, f)                                                                           \
  GGUF_DECODE_MMA_ROWS(F, f, a, EpNone, bfloat) GGUF_DECODE_MMA_ROWS(F, f, a_f32, EpNone, float)               \
  GGUF_DECODE_MMA_ROWS(F, f, r, EpResidual, bfloat) GGUF_DECODE_MMA_ROWS(F, f, g, EpUpWithGate, bfloat)
QUANT_FORMATS(GGUF_DECODE_MMA_FORMAT)
#undef GGUF_DECODE_MMA_FORMAT
#undef GGUF_DECODE_MMA_ROWS
#undef GGUF_DECODE_MMA

// Fused projections: up to three column segments of any formats in one dispatch, as gguf_decode_fused_m*. The
// segments' formats are function constants (ops::Linear names the pipeline with specializedPipelineName), so a
// pipeline compiles only their paths; a slot past the last segment repeats its format.
constant uint kFusedFormat0 [[function_constant(0)]];
constant uint kFusedFormat1 [[function_constant(1)]];
constant uint kFusedFormat2 [[function_constant(2)]];
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_DECODE_MMA_FUSED(R)                                                                                   \
  kernel void gguf_decode_mma_fused_m##R(device bfloat *input [[buffer(0)]], GGUF_SEGMENT(1, w0a, w1a, ma),      \
                                         GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc),            \
                                         device bfloat *output [[buffer(10)]],                                    \
                                         device coherent(device) float *partials [[buffer(11)]],                  \
                                         device atomic_uint *counters [[buffer(12)]],                             \
                                         constant GgufDecodeFusedParams &p [[buffer(13)]],                        \
                                         uint2 group [[threadgroup_position_in_grid]],                            \
                                         uint simd_lane [[thread_index_in_simdgroup]],                            \
                                         uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup half stage[GGUF_TILE_COLUMNS * kStride]; threadgroup half2 tl[kQuantPairTableEntries];            \
    threadgroup uint arrival;                                                                                     \
    const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;                        \
    const uint s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;                                                       \
    device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                         \
    device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                         \
    device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;                                                          \
    const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / kStep / p.splits;          \
    const uint origin = local * GGUF_TILE_COLUMNS + simd_group * kColumns, column0 = p.offset[s] + origin;         \
    const uint thread_index = simd_group * 32 + simd_lane;                                                        \
    float2 acc[FN][R / 8];                                                                                        \
    zero(acc);                                                                                                    \
    const uint3 formats(kFusedFormat0, kFusedFormat1, kFusedFormat2);                                             \
    accumulate_any<R / 8>(formats[s], formats, input, w0, w1, meta, p.input_size, origin,                         \
                          stage + simd_group * kSimdgroupStage, tl, thread_index, GGUF_STAGED_THREADS, simd_lane,  \
                          group.y * per, (group.y + 1) * per, acc);                                               \
    store_sums<R>(acc, sgmatrix::lane_map(simd_lane), p.splits, group.y, partials,                                \
                  counters + p.offset[s] / GGUF_TILE_COLUMNS + local, p.out_stride, column0, thread_index, &arrival, \
                  [&](uint row, uint column, float v) { output[ulong(row) * p.out_stride + column0 + column] = bfloat(v); }); \
  }
GGUF_DECODE_MMA_FUSED(8) GGUF_DECODE_MMA_FUSED(16) GGUF_DECODE_MMA_FUSED(32)
#undef GGUF_DECODE_MMA_FUSED
#undef GGUF_SEGMENT

// The prefill kernels: grid (64-row tiles of the chunk, column tiles of the segment), four simdgroups; the residual
// and gate kernels read aux at buffer 5, the plain one binds none (as gguf_prefill_*).
template <class F, GgufEpilogue Ep>
inline void gguf_prefill_mma(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                             device bfloat *output, device bfloat *aux, constant GgufPrefillParams &p, uint2 group,
                             uint simd_lane, uint simd_group, threadgroup half *stage, threadgroup half2 *tl) {
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, kPrefillThreads);
  const uint stride = p.out_stride;
  const uint first = group.x * kPrefillRows, rows = p.rows > first ? p.rows - first : 0;
  const uint origin = group.y * GGUF_TILE_COLUMNS;
  prefill_tile<F>(input + ulong(first) * p.input_size, w0, w1, meta, p.input_size, origin, rows, stage, tl, simd_lane,
                  simd_group, [&](uint row, uint column, float v) {
                    const ulong o = ulong(first + row) * stride + p.out_offset + origin + column;
                    output[o] = gguf_epilogue<Ep>(v, aux, o);
                  });
}
#define GGUF_PREFILL_MMA_BUFFERS                                                                                   \
  device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],             \
      device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]]
#define GGUF_PREFILL_MMA_THREAD                                                                                    \
  uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],                      \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define GGUF_PREFILL_MMA_TABLES(F)                                                                                 \
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];                                     \
  threadgroup half stage[2 * kPrefillStage]
#define GGUF_PREFILL_MMA(F, f)                                                                                     \
  kernel void gguf_prefill_mma_##f##_a(GGUF_PREFILL_MMA_BUFFERS, constant GgufPrefillParams &p [[buffer(5)]],    \
                                       GGUF_PREFILL_MMA_THREAD) {                                                  \
    GGUF_PREFILL_MMA_TABLES(F);                                                                                    \
    gguf_prefill_mma<F, EpNone>(input, w0, w1, meta, output, nullptr, p, group, simd_lane, simd_group, stage, tl); \
  }
#define GGUF_PREFILL_MMA_EPILOGUE(F, f, ep, Ep)                                                                    \
  kernel void gguf_prefill_mma_##f##_##ep(GGUF_PREFILL_MMA_BUFFERS, device bfloat *aux [[buffer(5)]],            \
                                          constant GgufPrefillParams &p [[buffer(6)]], GGUF_PREFILL_MMA_THREAD) {  \
    GGUF_PREFILL_MMA_TABLES(F);                                                                                    \
    gguf_prefill_mma<F, Ep>(input, w0, w1, meta, output, aux, p, group, simd_lane, simd_group, stage, tl);         \
  }
#define GGUF_PREFILL_MMA_FORMAT(F, f) \
  GGUF_PREFILL_MMA(F, f) GGUF_PREFILL_MMA_EPILOGUE(F, f, r, EpResidual) GGUF_PREFILL_MMA_EPILOGUE(F, f, g, EpUpWithGate)
QUANT_FORMATS(GGUF_PREFILL_MMA_FORMAT)
#undef GGUF_PREFILL_MMA_FORMAT

// The residual kernels over a view of the leading inputs of wider weight rows, for the Neural Engine FFN split's down
// projection, one per format: the planes advance past the groups and meta units the view leaves unread in the rows of
// the plane tiles before the column tile, as gguf_prefill_*_r_leading_inputs does for the MPP tile.
#define GGUF_PREFILL_MMA_LEADING_INPUTS(F, f)                                                                      \
  kernel void gguf_prefill_mma_##f##_r_leading_inputs(GGUF_PREFILL_MMA_BUFFERS, device bfloat *aux [[buffer(5)]],  \
                                                      constant GgufPrefillLeadingParams &leading [[buffer(6)]],    \
                                                      GGUF_PREFILL_MMA_THREAD) {                                   \
    GGUF_PREFILL_MMA_TABLES(F);                                                                                    \
    constant GgufPrefillParams &p = leading.prefill;                                                               \
    const ulong before = ulong(group.y * GGUF_TILE_COLUMNS / QUANT_TILE_ROWS) * QUANT_TILE_ROWS;                   \
    const uint groups = p.input_size / 32, plane_groups = leading.plane_input_size / 32;                           \
    const ulong unread = before * (plane_groups - groups),                                                         \
                unread_units = before * (plane_groups / F::MetaGroups - groups / F::MetaGroups);                   \
    gguf_prefill_mma<F, EpResidual>(input, w0 + unread * F::P0, w1 + unread * F::P1,                               \
                                    meta + unread_units * F::MetaBytes, output, aux, p, group, simd_lane,          \
                                    simd_group, stage, tl);                                                        \
  }
QUANT_FORMATS(GGUF_PREFILL_MMA_LEADING_INPUTS)
#undef GGUF_PREFILL_MMA_LEADING_INPUTS
#undef GGUF_PREFILL_MMA_EPILOGUE
#undef GGUF_PREFILL_MMA
#undef GGUF_PREFILL_MMA_TABLES
#undef GGUF_PREFILL_MMA_THREAD
#undef GGUF_PREFILL_MMA_BUFFERS
