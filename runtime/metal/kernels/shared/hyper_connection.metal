#include "metal/abi/HyperConnection.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/hyper_streams.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// Hyper-connections (metal/abi/HyperConnection.h, Qwen3.8-Flash-Next's:
// model/Qwen4Exp.hpp). The residual streams are fp32 [rows][streams][hidden];
// the projections between these kernels read and write bf16 rows.

// Writes the lane's two values of span `span` of row `row` into the row's
// 8-row tile of the Table16 table the register-tile projection reading the
// rows takes (gguf_sg::Table16, as decode_linear_gguf_prepare writes it).
// Table is gguf_sg::Table16, or Table16F on Apple7/8, whose projections read
// an fp32 table.
template <class Table>
inline void hyper_table16_write(device typename Table::Elem *table, device float *sums, uint width, uint span,
                                uint row, uint lane, bfloat a, bfloat b) {
  Table::write(table + ulong(row / gguf_sg::kRows) * width * gguf_sg::kRows,
               sums + ulong(row / gguf_sg::kRows) * Table::sums_per_tile(width), width, span, row % gguf_sg::kRows,
               lane, a, b);
}

// The first mix step of a row: each stream's RMS norm times the mix's norm
// weights (the GGUF stores 1 + w) into bf16 rows of streams * hidden, and,
// when it injects, the injection weights 2 sigmoid(inject(normalized) /
// streams) of the block's output into each stream. Its kernels are compiled
// for Flash-Next's shape, kStreams streams of kNormHidden values. Each takes
// every load of a pass before the pass's arithmetic: at decode's few rows, a
// threadgroup per row that waits on each load in turn runs latency-bound.
// The two sum alike, so a row's output and weights are bit for bit the same
// whichever of them runs it. They write no table: a register-tile down
// projection prepares its own (decode_linear_gguf_prepare, a simdgroup per
// span and row over the whole GPU). A threadgroup per row writing its row's
// table stores two bytes at a time into a lane's eight-row tiles from one
// core, which cost a 20-core M5 Pro 1.5 to 5.2 us per mix more than the
// plain norm and that preparation, at one to four lanes.
constant constexpr uint kNormHidden = HYPER_CONNECTION_NORM_HIDDEN;
constant constexpr uint kNormValues = kNormHidden / kRowThreads; // per thread and stream
static_assert(kNormHidden % kRowThreads == 0, "a thread's values per stream are whole");

// Prefill's rows, and a mix that injects nothing: a threadgroup of
// kRowThreads threads per row; each thread holds its values of every stream.
inline void hyper_norm_loaded_row(device const float *streams, device const float *norm,
                                  device const float *inject, device bfloat *normalized,
                                  device float *weights, constant HyperConnectionParams &p, uint row, uint tid,
                                  uint lane, uint sg, threadgroup float *scratch) {
  if (row >= p.rows) return;
  const uint width = kStreams * kNormHidden;
  device const float *x = streams + ulong(row) * width;
  float xs[kStreams][kNormValues];
  for (uint s = 0; s < kStreams; ++s)
    for (uint i = 0; i < kNormValues; ++i) xs[s][i] = x[s * kNormHidden + tid + i * kRowThreads];
  float sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < kStreams; ++s)
    for (uint i = 0; i < kNormValues; ++i) sums[s] += xs[s][i] * xs[s][i];
  reduce_streams(sums, kStreams, scratch, lane, sg);
  float scale[kStreams];
  for (uint s = 0; s < kStreams; ++s) scale[s] = rsqrt(sums[s] / kNormHidden + kRmsEpsilon);
  float dots[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  device bfloat *out = normalized + ulong(row) * width;
  for (uint s = 0; s < kStreams; ++s) {
    float ns[kNormValues];
    for (uint i = 0; i < kNormValues; ++i) ns[i] = norm[s * kNormHidden + tid + i * kRowThreads];
    for (uint i = 0; i < kNormValues; ++i) {
      const uint c = s * kNormHidden + tid + i * kRowThreads;
      const float v = xs[s][i] * scale[s] * ns[i];
      out[c] = bfloat(v);
      if (p.inject)
        for (uint j = 0; j < kStreams; ++j) dots[j] += inject[ulong(j) * width + c] * v;
    }
  }
  if (!p.inject) return;
  reduce_streams(dots, kStreams, scratch, lane, sg);
  if (tid < kStreams) weights[row * kStreams + tid] = 2.0f * splash_sigmoid(dots[tid] / kStreams);
}

// Decode's rows of an injecting mix: a threadgroup of
// HYPER_CONNECTION_NORM_QUARTER_THREADS threads per row, whose quarter q stands for the
// loaded row's threadgroup in stream q's sum and in injection weight q's dot:
// each of its threads sums its values of stream q, then runs injection dot q
// over every stream's values in order, and writes stream q's normalized
// values. Staged (Apple7/8's, hyper_connection_norm_staged) works out each
// stream's values once instead of in every quarter: stream s's quarter leaves
// them in threadgroup memory, kStagedStreams streams at a time (all four would
// be 40 KB, over the 32 KB a threadgroup has), where each quarter's thread tid
// reads the values at its own positions. The dots run in the same order, so
// both kernels' outputs are bit for bit the same.
constant constexpr uint kStagedStreams = 2;
template <bool Staged>
inline void hyper_norm_quarter_row(device const float *streams, device const float *norm,
                                   device const float *inject, device bfloat *normalized,
                                   device float *weights, constant HyperConnectionParams &p, uint row,
                                   uint thread_id, uint lane, uint simdgroup, threadgroup float *scratch,
                                   threadgroup float *scales, threadgroup float *staged) {
  if (row >= p.rows) return;
  const uint q = thread_id / kRowThreads, tid = thread_id % kRowThreads;
  const uint width = kStreams * kNormHidden;
  device const float *x = streams + ulong(row) * width;
  // Values are read again where they are used rather than held across the
  // barriers: held, they cap the kernel below its threads on Apple7/8 (896 on
  // an M1 Max). The arithmetic and its order are unchanged.
  float sum = 0.0f;
  for (uint i = 0; i < kNormValues; ++i) {
    const float value = x[q * kNormHidden + tid + i * kRowThreads];
    sum += value * value;
  }
  // The threadgroup's simdgroup `simdgroup` is quarter q's simdgroup
  // simdgroup % kSimdgroups.
  sum = simd_sum(sum);
  if (lane == 0) scratch[simdgroup] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[q * kSimdgroups + g];
    scales[q] = rsqrt(total / kNormHidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat *out = normalized + ulong(row) * width;
  float dot = 0.0f;
  if (!p.inject) {
    for (uint i = 0; i < kNormValues; ++i) {
      const uint c = q * kNormHidden + tid + i * kRowThreads;
      out[c] = bfloat(x[c] * scales[q] * norm[c]);
    }
  } else if constexpr (Staged) {
    device const float *w = inject + ulong(q) * width;
    for (uint first = 0; first < kStreams; first += kStagedStreams) {
      if (q - first < kStagedStreams) {
        threadgroup float *values = staged + (q - first) * kNormHidden;
        for (uint i = 0; i < kNormValues; ++i) {
          const uint c = q * kNormHidden + tid + i * kRowThreads;
          const float v = x[c] * scales[q] * norm[c];
          out[c] = bfloat(v);
          values[tid + i * kRowThreads] = v;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint s = first; s < first + kStagedStreams; ++s) {
        threadgroup const float *values = staged + (s - first) * kNormHidden;
        for (uint i = 0; i < kNormValues; ++i)
          dot += w[s * kNormHidden + tid + i * kRowThreads] * values[tid + i * kRowThreads];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    device const float *w = inject + ulong(q) * width;
    for (uint s = 0; s < kStreams; ++s) {
      const float scale = scales[s];
      for (uint i = 0; i < kNormValues; ++i) {
        const uint c = s * kNormHidden + tid + i * kRowThreads;
        const float v = x[c] * scale * norm[c];
        if (s == q) out[c] = bfloat(v);
        dot += w[c] * v;
      }
    }
  }
  if (!p.inject) return;
  dot = simd_sum(dot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[simdgroup] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float total = 0.0f;
  for (uint g = 0; g < kSimdgroups; ++g) total += scratch[q * kSimdgroups + g];
  weights[row * kStreams + q] = 2.0f * splash_sigmoid(total / kStreams);
}

#define HYPER_NORM_ARGUMENTS                                                                                     \
  device const float *streams [[buffer(0)]], device const float *norm [[buffer(1)]],                            \
      device const float *inject [[buffer(2)]], device bfloat *normalized [[buffer(3)]],                        \
      device float *weights [[buffer(4)]]

kernel void hyper_connection_norm_loaded(HYPER_NORM_ARGUMENTS, constant HyperConnectionParams &p [[buffer(5)]],
                                         uint row [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  hyper_norm_loaded_row(streams, norm, inject, normalized, weights, p, row, tid, lane, sg, scratch);
}

[[max_total_threads_per_threadgroup(HYPER_CONNECTION_NORM_QUARTER_THREADS)]] kernel void hyper_connection_norm_quarters(HYPER_NORM_ARGUMENTS, constant HyperConnectionParams &p [[buffer(5)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint thread_id [[thread_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kStreams * kSimdgroups];
  threadgroup float scales[kStreams];
  hyper_norm_quarter_row<false>(streams, norm, inject, normalized, weights, p, row, thread_id, lane, simdgroup,
                                scratch, scales, nullptr);
}

[[max_total_threads_per_threadgroup(HYPER_CONNECTION_NORM_QUARTER_THREADS)]] kernel void hyper_connection_norm_staged(
    HYPER_NORM_ARGUMENTS, constant HyperConnectionParams &p [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kStreams * kSimdgroups];
  threadgroup float scales[kStreams];
  threadgroup float staged[kStagedStreams * kNormHidden];
  hyper_norm_quarter_row<true>(streams, norm, inject, normalized, weights, p, row, thread_id, lane, simdgroup,
                               scratch, scales, staged);
}

#undef HYPER_NORM_ARGUMENTS

// silu(low / streams) of the mix's low-rank projection, in place.
kernel void hyper_connection_low(device bfloat *low [[buffer(0)]],
                                 constant HyperConnectionParams &p [[buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.rank) return;
  low[index] = bfloat(splash_silu(float(low[index]) / p.streams));
}

// The same values into the up projection's table alone (a decode step's
// rows): a threadgroup per row, a simdgroup per 64-value span.
template <class Table>
inline void hyper_low_table(device const bfloat *low, device typename Table::Elem *table, device float *sums,
                            constant HyperConnectionParams &p, uint row, uint lane, uint sg) {
  if (row >= p.rows) return;
  device const bfloat *x = low + ulong(row) * p.rank;
  for (uint g = sg; g < p.rank / 64; g += kSimdgroups) {
    const uint k = g * 64 + 2 * lane;
    hyper_table16_write<Table>(table, sums, p.rank, g, row, lane, bfloat(splash_silu(float(x[k]) / p.streams)),
                               bfloat(splash_silu(float(x[k + 1]) / p.streams)));
  }
}

#define HYPER_LOW_TABLE(Name, Table)                                                                             \
  kernel void Name(device const bfloat *low [[buffer(0)]], device Table::Elem *table [[buffer(1)]],              \
                   device float *sums [[buffer(2)]], constant HyperConnectionParams &p [[buffer(3)]],             \
                   uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],           \
                   uint sg [[simdgroup_index_in_threadgroup]]) {                                                 \
    hyper_low_table<Table>(low, table, sums, p, row, lane, sg);                                                 \
  }
HYPER_LOW_TABLE(hyper_connection_low_table16, gguf_sg::Table16)
HYPER_LOW_TABLE(hyper_connection_low_table16f, gguf_sg::Table16F)
#undef HYPER_LOW_TABLE

// The block's input: the mean over the streams of the normalized streams
// gated by sigmoid of the mix's up projection.
kernel void hyper_connection_mix(device const bfloat *normalized [[buffer(0)]],
                                 device const bfloat *gate [[buffer(1)]],
                                 device bfloat *mixed [[buffer(2)]],
                                 constant HyperConnectionParams &p [[buffer(3)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const ulong base = ulong(row) * p.streams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s)
    sum += float(normalized[base + s * p.hidden]) * splash_sigmoid(float(gate[base + s * p.hidden]));
  mixed[ulong(row) * p.hidden + d] = bfloat(sum / p.streams);
}

// The mix with the next projection's table (a decode step's rows, hidden a
// multiple of the threadgroup): the threadgroup's 256 values are four
// 64-value spans, which its first four simdgroups write.
template <class Table>
inline void hyper_mix_table(device const bfloat *normalized, device const bfloat *gate, device bfloat *mixed,
                            device typename Table::Elem *table, device float *sums,
                            constant HyperConnectionParams &p, uint2 group, uint tid, uint lane, uint sg,
                            threadgroup bfloat *values) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows) return;
  const ulong base = ulong(row) * p.streams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s)
    sum += float(normalized[base + s * p.hidden]) * splash_sigmoid(float(gate[base + s * p.hidden]));
  const bfloat value = bfloat(sum / p.streams);
  mixed[ulong(row) * p.hidden + d] = value;
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= kRowThreads / 64) return;
  const uint k = sg * 64 + 2 * lane;
  hyper_table16_write<Table>(table, sums, p.hidden, group.y * (kRowThreads / 64) + sg, row, lane, values[k],
                             values[k + 1]);
}

#define HYPER_MIX_TABLE(Name, Table)                                                                             \
  kernel void Name(device const bfloat *normalized [[buffer(0)]], device const bfloat *gate [[buffer(1)]],       \
                   device bfloat *mixed [[buffer(2)]], device Table::Elem *table [[buffer(3)]],                  \
                   device float *sums [[buffer(4)]], constant HyperConnectionParams &p [[buffer(5)]],             \
                   uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],       \
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {        \
    threadgroup bfloat values[kRowThreads];                                                                     \
    hyper_mix_table<Table>(normalized, gate, mixed, table, sums, p, group, tid, lane, sg, values);              \
  }
HYPER_MIX_TABLE(hyper_connection_mix_table16, gguf_sg::Table16)
HYPER_MIX_TABLE(hyper_connection_mix_table16f, gguf_sg::Table16F)
#undef HYPER_MIX_TABLE

// Every stream plus the block's output scaled by the stream's injection weight.
kernel void hyper_connection_combine(device float *streams [[buffer(0)]],
                                     device const bfloat *branch [[buffer(1)]],
                                     device const float *weights [[buffer(2)]],
                                     constant HyperConnectionParams &p [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const float b = float(branch[ulong(row) * p.hidden + d]);
  device float *x = streams + ulong(row) * p.streams * p.hidden + d;
  for (uint s = 0; s < p.streams; ++s) x[s * p.hidden] += b * weights[row * p.streams + s];
}

// The streams' start: every stream a copy of the token's embedding.
kernel void hyper_connection_init(device const bfloat *embedding [[buffer(0)]],
                                  device float *streams [[buffer(1)]],
                                  constant HyperConnectionParams &p [[buffer(2)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const float e = float(embedding[ulong(row) * p.hidden + d]);
  device float *x = streams + ulong(row) * p.streams * p.hidden + d;
  for (uint s = 0; s < p.streams; ++s) x[s * p.hidden] = e;
}

// The mean of each row's streams (bf16), at column p.column of row p.row + r
// of a destination of p.stride values: Qwen3.8-Flash-Next's capture of a
// layer's output, which its DFlash2 draft reads (DFlashDraftLayout's
// target_hidden_reduction).
kernel void hyper_connection_stream_mean(device const float *streams [[buffer(0)]],
                                         device bfloat *destination [[buffer(1)]],
                                         constant HyperStreamMeanParams &p [[buffer(2)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  device const float *x = streams + ulong(row) * p.streams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s) sum += x[s * p.hidden];
  destination[ulong(p.row + row) * p.stride + p.column + d] = bfloat(sum / float(p.streams));
}
