// GGUF MoE experts for Apple7/8 (M1/M2) on the simdgroup MMA tile (kernels/common/gguf_mma_tile.h): the bindings,
// grids and outputs of moe_expert_gguf_m<rows>_gate_up[_mixed] and _down (kernels/shared/moe_gguf.metal), whose MPP
// matmul2d runs far below the MMA rate there. A tile runs the 8-, 16- or 32-row MMAs that hold its live rows, over the
// tiles of its pass's range (MOE_TILES_*, metal/abi/MoE.h).
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_mma_tile.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

using namespace gguf_mma;

// A pass decodes its tensors' formats alone, as moe_gguf.metal's do: function constants 0 and 1 for the down pass
// (routed, shared) and 0-3 for the gate/up pass (ops/MoE.cpp passFormats: each once in ascending order, the last
// repeated past them). With every format's loops a pipeline holds each format's MMA walk for each live-row height,
// which the Metal compiler takes minutes to build on an M1 or M2, on the thread that first dispatches it.
constant uint kMmaMoeFormat0 [[function_constant(0)]];
constant uint kMmaMoeFormat1 [[function_constant(1)]];
constant uint kMmaMoeFormat2 [[function_constant(2)]];
constant uint kMmaMoeFormat3 [[function_constant(3)]];
template <uint Count> struct moe_mma_formats {
  template <class Body> void operator()(uint format, Body body) const {
    if constexpr (Count > 3)
      if (kMmaMoeFormat3 != kMmaMoeFormat2 && format == kMmaMoeFormat3) return quant_format_switch(kMmaMoeFormat3, body);
    if constexpr (Count > 2)
      if (kMmaMoeFormat2 != kMmaMoeFormat1 && format == kMmaMoeFormat2) return quant_format_switch(kMmaMoeFormat2, body);
    if (kMmaMoeFormat1 != kMmaMoeFormat0 && format == kMmaMoeFormat1) return quant_format_switch(kMmaMoeFormat1, body);
    quant_format_switch(kMmaMoeFormat0, body);
  }
};

// Runs body(integral_constant<uint, FM>) with the fewest 8-row fragments that hold a tile's live rows.
template <ushort Rows, class Body> inline void moe_mma_live_rows(uint rows, Body body) {
  if constexpr (Rows == 8) body(integral_constant<uint, 1>{});
  else if (rows <= 8) body(integral_constant<uint, 1>{});
  else if (rows <= 16) body(integral_constant<uint, 2>{});
  else body(integral_constant<uint, 4>{});
}

// The down pass: projection 2 of the tiles' experts over the intermediate into bf16 outputs. Threadgroup (x, y)
// computes 64 columns of tile y of the pass's range, two simdgroups of 32 columns each.
template <ushort Rows, uint Shared>
kernel void moe_expert_gguf_mma_down(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                     device const uint *tile_count [[buffer(2)]], device uchar *sw0 [[buffer(3)]],
                                     device uchar *sw1 [[buffer(4)]], device uchar *smeta [[buffer(5)]],
                                     device bfloat *output [[buffer(6)]], device const uint *slots [[buffer(7)]],
                                     constant MoeGgufExpertParams &p [[buffer(8)]], constant ulong *chunks [[buffer(9)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint simd_lane [[thread_index_in_simdgroup]],
                                     uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[kQuantPairTableEntries];
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const MoeGgufSegment s =
      moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format,
                               slots, chunks, p.floor_slots, p.chunk_slots, 2, sw0, sw1, smeta);
  device bfloat *x = input + ulong(t) * Rows * p.input_size;
  const ulong out = ulong(t) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * kColumns, thread_index = simd_group * 32 + simd_lane;
  // A tile of at most MOE_GGUF_COLUMN_ROWS live rows runs the column tile (kernels/common/gguf_column_tile.h), as
  // moe_expert_gguf's does: a GEMV in registers without a stage or barrier, where the staged MMA tile's padded 8-row
  // steps and their per-step stage barrier bound a one- or two-row tile, not DRAM. A decode tile of a sparse route
  // (Qwen3.8-Flash-Next: ten of 512 experts a layer) is almost always such a tile. Q2_0 only, which has no pair
  // table: measured on M1 Max, the column tile halves these tiles' time on Q2_0 but runs ~11% slower than the staged
  // MMA on IQ3_XXS, whose heavier weights and dequant outweigh the stage it skips; formats not measured stay on the
  // staged MMA.
  if (moe_column_tile(tiles, t) && s.format == FmtQ20::Id) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    column_dot<FmtQ20, MOE_GGUF_COLUMN_ROWS>(x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane, 0,
                                             p.input_size / 32, tl, sums);
    for (uint row = 0; row < tile.rows; ++row)
      output[out + ulong(row) * p.output_size + origin + simd_lane] = bfloat(sums[row]);
    return;
  }
  moe_mma_live_rows<Rows>(tile.rows, [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    moe_mma_formats<2>()(s.format, [&](auto f) {
      typedef decltype(f) F;
      quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
      accumulate<F, FM>(x, s.w0, s.w1, s.meta, p.input_size, origin, stage + simd_group * kSimdgroupStage, tl,
                        simd_lane, 0, p.input_size / kStep, acc);
    });
    elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
      output[out + ulong(row) * p.output_size + origin + column] = bfloat(v);
    });
  });
}
using MoeExpertGgufMmaDownKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                        device uchar *, device uchar *, device uchar *, device bfloat *,
                                        device const uint *, constant MoeGgufExpertParams &, constant ulong *, uint2,
                                        uint, uint);
template [[host_name("moe_expert_gguf_mma_m8_down")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<8, 1>;
template [[host_name("moe_expert_gguf_mma_m32_down")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<32, 1>;
// Flint's shared expert of two slices (CompiledMoeRouting "_s2").
template [[host_name("moe_expert_gguf_mma_m8_down_s2")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<8, 2>;
template [[host_name("moe_expert_gguf_mma_m32_down_s2")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<32, 2>;

// Gate and up in one pass: the bindings and grid of moe_expert_gguf_m<rows>_gate_up (kernels/shared/moe_gguf.metal),
// threadgroup (x, y) a gate and an up simdgroup on the same 32 columns of tile y of the pass's range. The gate
// simdgroup parks bf16(gate) in its stage, which it no longer reads, and the up simdgroup writes bf16(up) * silu(gate)
// as gguf_epilogue<EpUpWithGate> does, so the intermediate is bitwise that of the gate and up passes apart.
//
// A column tile (moe_column_tile: an expert's only tile, of at most MOE_GGUF_COLUMN_ROWS live rows) runs each
// projection on the tile its own pass would take: the column tile for Q2_0, the 8-row MMA otherwise. Where gate and up
// both take the column tile, a lane runs its column's two chains interleaved (column_dot_pair), whose FMAs overlap and
// whose activation loads serve both; the even threadgroup's two simdgroups take its columns and the odd one's, which
// has none. On an M2 Max (38 cores), against the two passes, a layer of Qwen3.8-Flash-Next's shape
// (gguf-moe-benchmark flash-next) decodes 8 to 32 rows 3-16% faster in Q2_0 and 0.4-6% faster with IQ2_S or IQ3_XXS
// gate and up, and Flash-Next Q2_0's decode cycle is 3.2-4.4% shorter at one to four lanes. Other tiles run the MMAs
// that hold their live rows.
template <ushort Rows, ushort Tables, uint Shared>
kernel void moe_expert_gguf_mma_gate_up(device bfloat *input [[buffer(0)]],
                                        device const MoeTileDescriptor *tiles [[buffer(1)]],
                                        device const uint *tile_count [[buffer(2)]], device uchar *gsw0 [[buffer(3)]],
                                        device uchar *gsw1 [[buffer(4)]], device uchar *gsmeta [[buffer(5)]],
                                        device uchar *usw0 [[buffer(6)]], device uchar *usw1 [[buffer(7)]],
                                        device uchar *usmeta [[buffer(8)]], device bfloat *output [[buffer(9)]],
                                        device const uint *slots [[buffer(10)]],
                                        constant MoeGgufGateUpParams &p [[buffer(11)]],
                                        constant ulong *chunks [[buffer(12)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint simd_lane [[thread_index_in_simdgroup]],
                                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[Tables * kQuantPairTableEntries];
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const MoeGgufSegment gs = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.gate_routed_format, p.gate_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 0, gsw0, gsw1, gsmeta);
  const MoeGgufSegment us = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.up_routed_format, p.up_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 1, usw0, usw1, usmeta);
  const uint K = p.input_size, N = p.output_size;
  device bfloat *x = input + ulong(t) * Rows * K;
  const ulong out = ulong(t) * Rows * N;
  const uint origin = group.x * kColumns, thread_index = simd_group * 32 + simd_lane;
  const bool up = simd_group == 1;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  const bool column = moe_column_tile(tiles, t);
  const bool gate_column = column && gs.format == FmtQ20::Id, up_column = column && us.format == FmtQ20::Id;
  const auto product = [](float v, bfloat g) { return bfloat(float(bfloat(v)) * splash_silu(float(g))); };
  if (gate_column && up_column) {
    // As many simdgroups as the two passes' column tiles run: the even threadgroup's two take the odd one's columns
    // (N is a multiple of the passes' 64 columns, so every odd threadgroup has an even one).
    if (group.x % 2) return;
    // Q2_0 has no pair table, so tl is left unfilled here on purpose.
    const uint pair_column = origin + simd_group * kColumns + simd_lane;
    float g[MOE_GGUF_COLUMN_ROWS], u[MOE_GGUF_COLUMN_ROWS];
    column_dot_pair<FmtQ20, MOE_GGUF_COLUMN_ROWS>(x, tile.rows, K, gs.w0, gs.w1, gs.meta, us.w0, us.w1, us.meta,
                                                  pair_column, tl, g, u);
    for (uint row = 0; row < tile.rows; ++row)
      output[out + ulong(row) * N + pair_column] = product(u[row], bfloat(g[row]));
    return;
  }
  // Every thread fills the pair tables before one barrier, the up's in the second table where the formats differ.
  bool tables = false;
  moe_mma_formats<4>()(gs.format, [&](auto f) {
    tables |= quant_pair_entries<decltype(f)>(tl, thread_index, GGUF_STAGED_THREADS);
  });
  if constexpr (Tables == 2)
    moe_mma_formats<4>()(us.format, [&](auto f) {
      tables |= quant_pair_entries<decltype(f)>(tl + kQuantPairTableEntries, thread_index, GGUF_STAGED_THREADS);
    });
  if (tables) threadgroup_barrier(mem_flags::mem_threadgroup);
  const MoeGgufSegment s = up ? us : gs;
  threadgroup half2 *table = tl + (Tables == 2 && up ? kQuantPairTableEntries : 0);
  threadgroup half *my = stage + simd_group * kSimdgroupStage;
  const sgmatrix::Lane l = sgmatrix::lane_map(simd_lane);
  static_assert(Rows * kColumns * sizeof(bfloat) <= kSimdgroupStage * sizeof(half), "the gate fits a stage");
  if (column) {
    // At most MOE_GGUF_COLUMN_ROWS live rows, one fragment. Each simdgroup sums its projection on its own tile, the
    // gate parks bf16(gate) in its stage, and both meet at the one barrier before the up writes the products: the
    // simdgroups may take different tiles (a shared expert of Q2_0 gate and another up format), so neither tile's
    // branch holds a barrier of its own.
    const bool on_column = up ? up_column : gate_column;
    float sums[MOE_GGUF_COLUMN_ROWS];
    float2 acc[FN][1];
    if (on_column) {
      column_dot<FmtQ20, MOE_GGUF_COLUMN_ROWS>(x, tile.rows, K, s.w0, s.w1, s.meta, origin + simd_lane, 0, K / 32,
                                               table, sums);
    } else {
      zero(acc);
      moe_mma_formats<4>()(s.format, [&](auto f) {
        accumulate<decltype(f), 1>(x, s.w0, s.w1, s.meta, K, origin, my, table, simd_lane, 0, K / kStep, acc);
      });
    }
    if (!up) {
      if (on_column)
        for (uint row = 0; row < tile.rows; ++row) gate[row * kColumns + simd_lane] = bfloat(sums[row]);
      else
        elements(acc, l, [&](uint row, uint column, float v) { gate[row * kColumns + column] = bfloat(v); });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!up) return;
    if (on_column)
      for (uint row = 0; row < tile.rows; ++row)
        output[out + ulong(row) * N + origin + simd_lane] = product(sums[row], gate[row * kColumns + simd_lane]);
    else
      elements(acc, l, [&](uint row, uint column, float v) {
        output[out + ulong(row) * N + origin + column] = product(v, gate[row * kColumns + column]);
      });
    return;
  }
  // Both simdgroups run the MMAs of the tile's live rows, which the tile gives them alike.
  moe_mma_live_rows<Rows>(tile.rows, [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    moe_mma_formats<4>()(s.format, [&](auto f) {
      accumulate<decltype(f), FM>(x, s.w0, s.w1, s.meta, K, origin, my, table, simd_lane, 0, K / kStep, acc);
    });
    if (!up) elements(acc, l, [&](uint row, uint column, float v) { gate[row * kColumns + column] = bfloat(v); });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up)
      elements(acc, l, [&](uint row, uint column, float v) {
        output[out + ulong(row) * N + origin + column] = product(v, gate[row * kColumns + column]);
      });
  });
}
using MoeExpertGgufMmaGateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                          device uchar *, device uchar *, device uchar *, device uchar *,
                                          device uchar *, device uchar *, device bfloat *, device const uint *,
                                          constant MoeGgufGateUpParams &, constant ulong *, uint2, uint, uint);
#define MOE_MMA_GATE_UP(R)                                                                                      \
  template [[host_name("moe_expert_gguf_mma_m" #R "_gate_up")]] kernel MoeExpertGgufMmaGateUpKernel            \
      moe_expert_gguf_mma_gate_up<R, 1, 1>;                                                                     \
  template [[host_name("moe_expert_gguf_mma_m" #R "_gate_up_mixed")]] kernel MoeExpertGgufMmaGateUpKernel      \
      moe_expert_gguf_mma_gate_up<R, 2, 1>;                                                                     \
  template [[host_name("moe_expert_gguf_mma_m" #R "_gate_up_s2")]] kernel MoeExpertGgufMmaGateUpKernel         \
      moe_expert_gguf_mma_gate_up<R, 1, 2>;                                                                     \
  template [[host_name("moe_expert_gguf_mma_m" #R "_gate_up_mixed_s2")]] kernel MoeExpertGgufMmaGateUpKernel   \
      moe_expert_gguf_mma_gate_up<R, 2, 2>;
MOE_MMA_GATE_UP(8)
MOE_MMA_GATE_UP(32)
#undef MOE_MMA_GATE_UP
