#pragma once
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

#include <metal_stdlib>
using namespace metal;

// GGUF tiles for Apple7/8 (M1/M2), whose MPP matmul2d runs far below the simdgroup MMA rate (the staged tiles of
// kernels/common/gguf_staged_tile.h stall the GPU for minutes on an M1 Max). Weights decode to the staged tiles' half
// values (dequant_chunk); the product is 8x8 simdgroup MMAs with A = those half weights and B = the bf16 activations
// widened to fp32, so every product is exact and only the fp32 accumulation rounds, and activations past half's range
// stay finite. Includers set `#pragma clang fp reassociate(off)` first.
//
// Accumulators are plain float2 registers (sgmatrix.h): acc[i][r] holds a lane's elements of the fragment of columns
// 8i.. and rows 8r..: column 8i + fm, rows 8r + fn and 8r + fn + 1. At MMA step j (0-3) of a 32-input group, logical
// input 2c is chunk c's low element j (physical 4c + j) and 2c + 1 its high one (16 + 4c + j): lane (fm, fn) of A
// takes chunk fn / 2 of column fm whole over the group's four steps, and lane (fm, fn) of B the four adjacent inputs
// 4 (fm / 2) (+ 16 for odd fm) of its two rows. Every output element sums its K inputs in one order whatever the
// tile's rows, so the decode and prefill tiles give equal bits without K splits.
namespace gguf_mma {

constant constexpr uint kStep = 32;           // inputs per step: one group
constant constexpr uint kStride = kStep + 8;  // halves per staged column: a fragment's 8 columns hit distinct banks
constant constexpr uint kColumns = 32;        // columns per simdgroup
constant constexpr uint FN = kColumns / 8;    // column fragments per simdgroup
constant constexpr uint kSimdgroupStage = kColumns * kStride;
// Prefill tile: four simdgroups, two column halves by two 32-row halves, over a double-buffered shared stage.
constant constexpr uint kPrefillRows = 64;
constant constexpr uint kPrefillSimdgroupRows = 32;
constant constexpr uint kPrefillThreads = 128;
constant constexpr uint kPrefillStage = GGUF_TILE_COLUMNS * kStride;
static_assert(kPrefillRows == 2 * kPrefillSimdgroupRows && GGUF_TILE_COLUMNS == 2 * kColumns, "prefill tile");
static_assert(GGUF_STAGED_STEP == kStep, "decode tiles split K like the staged tile");
static_assert(GGUF_PREFILL_ROWS % kPrefillRows == 0, "prefill storage holds whole tiles");

// One group of one column in MMA order: chunk c's element t of lo goes to 8 t + 2c, of hi to 8 t + 2c + 1.
template <class F>
inline void stage_group(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                        threadgroup half *dst) {
  dequant32_chunks<F>(w, meta, j, tl, [&](ushort c, half4 lo, half4 hi) {
#pragma unroll
    for (ushort t = 0; t < 4; ++t) *reinterpret_cast<threadgroup half2 *>(dst + 8 * t + 2 * c) = half2(lo[t], hi[t]);
  });
}

// bf16 element j of four packed in v, as fp32.
inline float bf16_at(uint2 v, uint j) {
  const uint w = j < 2 ? v.x : v.y;
  return as_type<float>((j & 1) ? (w & 0xFFFF0000u) : (w << 16));
}

// A lane's activations of one group: rows 8r + fn (+1) of x (row 0 of the tile), inputs k0 + 4 (fm / 2) + 16 (fm & 1).
template <uint FM>
inline void load_inputs(device const bfloat *x, uint input_size, uint k0, sgmatrix::Lane l, thread uint2 (&v)[FM][2]) {
  const uint k = k0 + 4 * (l.fm >> 1) + 16 * (l.fm & 1);
#pragma unroll
  for (uint r = 0; r < FM; ++r)
#pragma unroll
    for (uint e = 0; e < 2; ++e)
      v[r][e] = *reinterpret_cast<device const uint2 *>(x + ulong(8 * r + l.fn + e) * input_size + k);
}

// acc += one group of FN column fragments, a(i, j) giving the lane's A elements of fragment i at step j, times the
// lane's activations v.
template <uint FM, class A>
inline void mma_group(thread uint2 (&v)[FM][2], A a, thread float2 (&acc)[FN][FM]) {
#pragma unroll
  for (uint j = 0; j < 4; ++j) {
    simdgroup_float8x8 b[FM];
#pragma unroll
    for (uint r = 0; r < FM; ++r) sgmatrix::te(b[r]) = float2(bf16_at(v[r][0], j), bf16_at(v[r][1], j));
#pragma unroll
    for (uint i = 0; i < FN; ++i) {
      simdgroup_half8x8 m;
      sgmatrix::te(m) = a(i, j);
#pragma unroll
      for (uint r = 0; r < FM; ++r) {
        simdgroup_float8x8 c, d;
        sgmatrix::te(c) = acc[i][r];
        simdgroup_multiply_accumulate(d, m, b[r], c);
        acc[i][r] = sgmatrix::te(d);
      }
    }
  }
}

template <uint FM> inline void zero(thread float2 (&acc)[FN][FM]) {
#pragma unroll
  for (uint i = 0; i < FN; ++i)
#pragma unroll
    for (uint r = 0; r < FM; ++r) acc[i][r] = float2(0.0f);
}
// fn(row, column, value) for every element a lane holds, in the simdgroup's rows and columns.
template <uint FM, class Fn> inline void elements(thread float2 (&acc)[FN][FM], sgmatrix::Lane l, Fn fn) {
#pragma unroll
  for (uint i = 0; i < FN; ++i)
#pragma unroll
    for (uint r = 0; r < FM; ++r)
#pragma unroll
      for (uint e = 0; e < 2; ++e) fn(8 * r + l.fn + e, 8 * i + l.fm, acc[i][r][e]);
}

// One simdgroup's 32 columns from `origin` over rows [0, 8 FM) of x, steps [step_begin, step_end) of K: lane c stages
// column c's group of each step into the simdgroup's own stage (kSimdgroupStage halves), loading the next step's
// weights before the multiply. Decoding each column's coefficients once per group this way beat decoding every
// lane's own chunks in registers on an M1 Max (Q4_K 5120 x 17408, ms at one and four lanes: 0.38 and 1.0 against
// 0.46 and 2.2).
template <class F, uint FM>
inline void accumulate(device const bfloat *x, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size,
                       uint origin, threadgroup half *stage, threadgroup half2 *tl, uint lane, uint step_begin,
                       uint step_end, thread float2 (&acc)[FN][FM]) {
  if (step_begin >= step_end) return;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS + lane;
  device uchar *c0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *c1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *cm = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  typename F::Payload packed = F::load(c0 + ulong(step_begin) * QUANT_TILE_ROWS * F::P0,
                                       c1 + ulong(step_begin) * QUANT_TILE_ROWS * F::P1);
  uint unit = step_begin / F::MetaGroups;
  typename F::Meta hdr = F::loadMeta(cm + ulong(unit) * QUANT_TILE_ROWS * F::MetaBytes);
  for (uint step = step_begin; step < step_end; ++step) {
    const uint u = step / F::MetaGroups;
    if (u != unit) {
      hdr = F::loadMeta(cm + ulong(u) * QUANT_TILE_ROWS * F::MetaBytes);
      unit = u;
    }
    stage_group<F>(packed, hdr, ushort(step % F::MetaGroups), tl, stage + lane * kStride);
    if (step + 1 < step_end)
      packed = F::load(c0 + ulong(step + 1) * QUANT_TILE_ROWS * F::P0, c1 + ulong(step + 1) * QUANT_TILE_ROWS * F::P1);
    uint2 v[FM][2];
    load_inputs<FM>(x, input_size, step * kStep, l, v);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    mma_group<FM>(v, [&](uint i, uint j) {
      return *reinterpret_cast<threadgroup const half2 *>(stage + (8 * i + l.fm) * kStride + 8 * j + l.fn);
    }, acc);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// accumulate in run-time format `fmt` (uniform per threadgroup), one of the three of `among`, after every thread of
// the threadgroup fills its pair table. A pipeline holds only the paths of `among` where they are function constants
// (quant_format_switch_among).
template <uint FM>
inline void accumulate_any(uint fmt, uint3 among, device const bfloat *x, device uchar *w0, device uchar *w1,
                           device uchar *meta, uint input_size, uint origin, threadgroup half *stage,
                           threadgroup half2 *tl, uint thread_index, uint threads, uint lane, uint step_begin,
                           uint step_end, thread float2 (&acc)[FN][FM]) {
  quant_format_switch_among(fmt, among, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, thread_index, threads);
    accumulate<F, FM>(x, w0, w1, meta, input_size, origin, stage, tl, lane, step_begin, step_end, acc);
  });
}

// A simdgroup's sums over every K partition, handed to store(row, column, sum), as gguf_store_sums: one partition
// stores its own; more publish fp32 partials [split][Rows][destination column] and the last arriving partition adds
// them in split order.
template <uint Rows, uint FM, class Store>
inline void store_sums(thread float2 (&acc)[FN][FM], sgmatrix::Lane l, uint splits, uint split,
                       device coherent(device) float *partials, device atomic_uint *counter, uint stride,
                       uint column0, uint thread_index, threadgroup uint *arrival, Store store) {
  if (splits == 1) {
    elements(acc, l, store);
    return;
  }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  elements(acc, l, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  elements(acc, l, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

// The prefill tile of chunk rows [first, first + 64) and columns [origin, origin + 64) of one segment: threads 0-63
// stage one column's group per step into a double-buffered stage, and each simdgroup that holds chunk rows runs its
// 32 columns by 32 rows. `rows` counts the chunk's rows from the tile's first: a simdgroup past them only stages.
template <class F, class Store>
inline void prefill_tile(device const bfloat *x, device uchar *w0, device uchar *w1, device uchar *meta,
                         uint input_size, uint origin, uint rows, threadgroup half *stage, threadgroup half2 *tl,
                         uint lane, uint simd_group, Store store) {
  constexpr uint FM = kPrefillSimdgroupRows / 8;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint thread_index = simd_group * 32 + lane, column = thread_index % GGUF_TILE_COLUMNS;
  const uint half_n = simd_group & 1, half_m = simd_group >> 1;
  const bool stages = thread_index < GGUF_TILE_COLUMNS;
  const bool owns = half_m * kPrefillSimdgroupRows < rows;  // uniform per simdgroup
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS + column;
  device uchar *c0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *c1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *cm = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  x += ulong(half_m) * kPrefillSimdgroupRows * input_size;
  typename F::Payload packed;
  typename F::Meta hdr;
  uint unit = 0;
  if (stages) {
    packed = F::load(c0, c1);
    hdr = F::loadMeta(cm);
  }
  float2 acc[FN][FM];
  zero(acc);
  for (uint step = 0; step < groups; ++step) {
    threadgroup half *buffer = stage + (step & 1) * kPrefillStage;
    if (stages) {
      const uint u = step / F::MetaGroups;
      if (u != unit) {
        hdr = F::loadMeta(cm + ulong(u) * QUANT_TILE_ROWS * F::MetaBytes);
        unit = u;
      }
      stage_group<F>(packed, hdr, ushort(step % F::MetaGroups), tl, buffer + column * kStride);
      if (step + 1 < groups)
        packed = F::load(c0 + ulong(step + 1) * QUANT_TILE_ROWS * F::P0, c1 + ulong(step + 1) * QUANT_TILE_ROWS * F::P1);
    }
    // The other buffer's last readers passed this barrier's predecessor, so one barrier per step suffices.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns) {
      uint2 v[FM][2];
      load_inputs<FM>(x, input_size, step * kStep, l, v);
      threadgroup const half *mine = buffer + half_n * kSimdgroupStage;
      mma_group<FM>(v, [&](uint i, uint j) {
        return *reinterpret_cast<threadgroup const half2 *>(mine + (8 * i + l.fm) * kStride + 8 * j + l.fn);
      }, acc);
    }
  }
  if (!owns) return;
  elements(acc, l, [&](uint row, uint col, float v) {
    store(half_m * kPrefillSimdgroupRows + row, half_n * kColumns + col, v);
  });
}

} // namespace gguf_mma
