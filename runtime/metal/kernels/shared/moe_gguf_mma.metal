// GGUF MoE experts for Apple7/8 (M1/M2) on the simdgroup MMA tile (kernels/common/gguf_mma_tile.h): the bindings,
// grids and outputs of moe_expert_gguf_m<rows>_gate_up[_mixed] and _down (kernels/shared/moe_gguf.metal), whose MPP
// matmul2d runs far below the MMA rate there. A tile runs the 8-, 16-, 24- or 32-row MMAs that hold its live rows, over the
// tiles of its pass's range (MOE_TILES_*, metal/abi/MoE.h).
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_mma_tile.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/softcapped_swiglu.h"

using namespace gguf_mma;

namespace gguf_mma {
struct RoutedInputPair {
  device const bfloat *first;
  device const bfloat *second;
};

template <uint FM>
inline void load_inputs(RoutedInputPair pair, uint, uint k0, sgmatrix::Lane lane, thread uint2 (&v)[FM][2]) {
  static_assert(FM == 1, "the routed pair serves one row fragment");
  const uint column = k0 + 4 * (lane.fm >> 1) + 16 * (lane.fm & 1);
  v[0][0] = *reinterpret_cast<device const uint2 *>(pair.first + column);
  v[0][1] = *reinterpret_cast<device const uint2 *>(pair.second + column);
}
}

template <uint Routes>
inline auto moe_mma_inputs(device bfloat *input, device const uchar *routes, MoeTileDescriptor tile,
                           uint width, sgmatrix::Lane lane) {
  if constexpr (Routes) {
    device const uint *grouped = reinterpret_cast<device const uint *>(routes) + tile.first_row;
    const uint last = tile.rows - 1;
    return RoutedInputPair{input + ulong(grouped[min(uint(lane.fn), last)] / Routes) * width,
                           input + ulong(grouped[min(uint(lane.fn) + 1, last)] / Routes) * width};
  } else {
    return input + ulong(tile.first_row) * width;
  }
}

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

// Writes the elements of `acc` through write(row, column, v), the tile's `live` rows alone: a tile's grouped rows
// start at its first_row and the next tile's follow its live rows (moe_expert_slab.h), so a partial tile's padding
// rows are the next tile's. When FullFastPath is true (the down pass's plain store), a full tile writes every row
// without the guard; gate/up keeps one guarded path so the SiLU epilogue is not duplicated across fragment counts.
template <uint FM, bool FullFastPath = true, class Write>
inline void moe_mma_store(thread float2 (&acc)[FN][FM], sgmatrix::Lane l, uint live, Write write) {
  if constexpr (FullFastPath) {
    if (live == FM * 8) {
      elements(acc, l, write);
      return;
    }
  }
  elements(acc, l, [&](uint row, uint column, float v) {
    if (row < live) write(row, column, v);
  });
}

// The grid formats on the column tile in a pass of Rows-row tiles (moe_mma_column_format but Q2_0, which the passes
// run on their own paths). Their path is in the 8-row passes alone: in the 32-row pass it gained 6-12% on Flash-Next's
// 64- and 128-row prefill chunks but cost its 2048-row chunk 0.6% with IQ2_XXS experts (M2 Max).
template <ushort Rows> constexpr bool moe_mma_grid_column_format(uint format) {
  return format != FmtQ20::Id && moe_mma_column_format(format, Rows);
}
// Whether one of a pass's formats (function constants 0 to Count - 1) is such a grid format: a constant of the
// pipeline, so a pass of none holds none of their column path, and Q2_0's pipelines stay as they were.
template <ushort Rows, uint Count> inline bool moe_mma_grid_column_pass() {
  bool any = moe_mma_grid_column_format<Rows>(kMmaMoeFormat0) || moe_mma_grid_column_format<Rows>(kMmaMoeFormat1);
  if constexpr (Count > 2)
    any = any || moe_mma_grid_column_format<Rows>(kMmaMoeFormat2) || moe_mma_grid_column_format<Rows>(kMmaMoeFormat3);
  return any;
}
// Runs body(f) for `format` when it is such a grid format.
template <ushort Rows, uint Count, class Body> inline void moe_mma_grid_column(uint format, Body body) {
  moe_mma_formats<Count>()(format, [&](auto f) {
    if constexpr (moe_mma_grid_column_format<Rows>(decltype(f)::Id)) body(f);
  });
}

// Whether a pass's formats (function constants 0 to Count - 1) are all a split format (moe_mma_split_format): its
// grid holds MOE_MMA_SPLIT times its column tiles.
template <ushort Rows, uint Count> inline bool moe_mma_split_pass() {
  bool all = moe_mma_split_format(kMmaMoeFormat0, Rows) && moe_mma_split_format(kMmaMoeFormat1, Rows);
  if constexpr (Count > 2)
    all = all && moe_mma_split_format(kMmaMoeFormat2, Rows) && moe_mma_split_format(kMmaMoeFormat3, Rows);
  return all;
}
// A split pass's column tile: lane l sums column `column + l % W` (W = 32 / MOE_MMA_SPLIT) over part l / W of K's
// groups, and the lanes add the other parts' down to the first W, which hold the W columns' sums.
constant constexpr uint kSplitWidth = 32 / MOE_MMA_SPLIT;
template <ushort Rows, uint Count, class RowInput>
inline void moe_mma_split_column(uint format, RowInput row_input, uint rows, uint K, MoeGgufSegment s, uint column,
                                 uint lane, threadgroup half2 *table, thread float (&sums)[GGUF_COLUMN_ROWS]) {
  const uint groups = K / 32, which = lane / kSplitWidth;
  moe_mma_grid_column<Rows, Count>(format, [&](auto f) {
    column_dot_rows<decltype(f), GGUF_COLUMN_ROWS>(row_input, rows, K, s.w0, s.w1, s.meta, column + lane % kSplitWidth,
                                                  which * groups / MOE_MMA_SPLIT, (which + 1) * groups / MOE_MMA_SPLIT,
                                                  table, sums);
  });
  for (uint r = 0; r < GGUF_COLUMN_ROWS; ++r)
    for (uint width = 16; width >= kSplitWidth; width /= 2) sums[r] += simd_shuffle_down(sums[r], width);
}

// Runs body(integral_constant<uint, FM>) with the fewest 8-row fragments that hold a tile's live rows.
template <ushort Rows, class Body> inline void moe_mma_live_rows(uint rows, Body body) {
  if constexpr (Rows == 8) body(integral_constant<uint, 1>{});
  else if (rows <= 8) body(integral_constant<uint, 1>{});
  else if (rows <= 16) body(integral_constant<uint, 2>{});
  else if (rows <= 24) body(integral_constant<uint, 3>{});
  else body(integral_constant<uint, 4>{});
}

// The down pass: projection 2 of the tiles' experts over the intermediate into bf16 outputs. Threadgroup (x, y)
// computes 64 columns of tile y of the pass's range, two simdgroups of 32 columns each.
template <ushort Rows, uint Shared>
kernel void moe_expert_gguf_mma_down(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                     device const uint *tile_count [[buffer(2)]], device uchar *sw0 [[buffer(3)]],
                                     device uchar *sw1 [[buffer(4)]], device uchar *smeta [[buffer(5)]],
                                     device bfloat *output [[buffer(6)]], device const uint *slots [[buffer(7)]],
                                     device const float *expert_scales [[buffer(8)]],
                                     constant MoeGgufExpertParams &p [[buffer(9)]], constant ulong *chunks [[buffer(10)]],
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
  const float scale = moe_expert_scale(expert_scales, tile.expert, p.experts, 2);
  device bfloat *x = input + ulong(tile.first_row) * p.input_size;
  const ulong out = ulong(tile.first_row) * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * kColumns, thread_index = simd_group * 32 + simd_lane;
  // A tile of at most GGUF_COLUMN_ROWS live rows runs the column tile (kernels/common/gguf_column_tile.h), as
  // moe_expert_gguf's does: a GEMV in registers without a stage or barrier, where the staged MMA tile's padded 8-row
  // steps and their per-step stage barrier bound a one- or two-row tile, not DRAM. A decode tile of a sparse route
  // (Qwen3.8-Flash-Next: ten of 512 experts a layer) is almost always such a tile. Only the formats of
  // moe_mma_column_format, which have no pair table: measured on M1 Max, the column tile halves these tiles' time on
  // Q2_0 but runs ~11% slower than the staged MMA on IQ3_XXS, whose heavier weights and dequant outweigh the stage it
  // skips; formats not measured stay on the staged MMA.
  if (moe_column_tile(tiles, t) && s.format == FmtQ20::Id) {
    float sums[GGUF_COLUMN_ROWS];
    column_dot<FmtQ20, GGUF_COLUMN_ROWS>(x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane, 0,
                                             p.input_size / 32, tl, sums);
    for (uint row = 0; row < tile.rows; ++row)
      output[out + ulong(row) * p.output_size + origin + simd_lane] = bfloat(sums[row] * scale);
    return;
  }
  if (moe_mma_grid_column_pass<Rows, 2>() && moe_column_tile(tiles, t) &&
      moe_mma_grid_column_format<Rows>(s.format)) {
    // The column formats but Q2_0 (moe_mma_grid_column_pass): the grids' constant tables, or NVFP4's pair table,
    // which every thread fills first.
    float sums[GGUF_COLUMN_ROWS];
    moe_mma_grid_column<Rows, 2>(s.format, [&](auto f) {
      quant_pair_table<decltype(f)>(tl, thread_index, GGUF_STAGED_THREADS);
      column_dot_rows<decltype(f), GGUF_COLUMN_ROWS>([&](uint r) { return x + ulong(r) * p.input_size; }, tile.rows,
                                                     p.input_size, s.w0, s.w1, s.meta, origin + simd_lane, 0,
                                                     p.input_size / 32, tl, sums);
    });
    for (uint row = 0; row < tile.rows; ++row)
      output[out + ulong(row) * p.output_size + origin + simd_lane] = bfloat(sums[row] * scale);
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
    // Reload the tile descriptor after the K loop so the 64-bit output offset and live row count are not held live
    // across accumulate.
    const MoeTileDescriptor dst = tiles[t];
    const ulong dst_out = ulong(dst.first_row) * p.output_size;
    moe_mma_store(acc, sgmatrix::lane_map(simd_lane), dst.rows, [&](uint row, uint column, float v) {
      output[dst_out + ulong(row) * p.output_size + origin + column] = bfloat(v * scale);
    });
  });
}
using MoeExpertGgufMmaDownKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                        device uchar *, device uchar *, device uchar *, device bfloat *,
                                        device const uint *, device const float *, constant MoeGgufExpertParams &, constant ulong *, uint2,
                                        uint, uint);
template [[host_name("moe_expert_gguf_mma_m8_down")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<8, 1>;
template [[host_name("moe_expert_gguf_mma_m32_down")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<32, 1>;
// Flint's shared expert of two slices (CompiledMoeRouting "_s2").
template [[host_name("moe_expert_gguf_mma_m8_down_s2")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<8, 2>;
template [[host_name("moe_expert_gguf_mma_m32_down_s2")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<32, 2>;

template [[host_name("moe_expert_gguf_mma_m8_down_s0")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<8, 0>;
template [[host_name("moe_expert_gguf_mma_m32_down_s0")]] kernel MoeExpertGgufMmaDownKernel moe_expert_gguf_mma_down<32, 0>;

// Gate and up in one pass: the bindings and grid of moe_expert_gguf_m<rows>_gate_up (kernels/shared/moe_gguf.metal),
// threadgroup (x, y) a gate and an up simdgroup on the same 32 columns of tile y of the pass's range. The gate
// simdgroup parks bf16(gate) in its stage, which it no longer reads, and the up simdgroup writes bf16(up) * silu(gate)
// as gguf_epilogue<EpUpWithGate> does, so the intermediate is bitwise that of the gate and up passes apart.
//
// A column tile (moe_column_tile: an expert's only tile, of at most GGUF_COLUMN_ROWS live rows) runs each
// projection on the tile its own pass would take: the column tile for the formats of moe_mma_column_format, the 8-row
// MMA otherwise. Where Q2_0 gate and up both take the column tile, a lane runs its column's two chains interleaved
// (column_dot_pair), whose FMAs overlap and whose activation loads serve both; the even threadgroup's two simdgroups
// take its columns and the odd one's, which has none. On an M2 Max (38 cores), against the two passes, a layer of
// Qwen3.8-Flash-Next's shape (gguf-moe-benchmark flash-next) decodes 8 to 32 rows 3-16% faster in Q2_0 and 0.4-6%
// faster with IQ2_S or IQ3_XXS gate and up, and Flash-Next Q2_0's decode cycle is 3.2-4.4% shorter at one to four
// lanes. IQ2_XXS and IQ2_S gate and up run column_dot on their own simdgroups, which beat their pair at pooled routing
// (a verify's rows sharing their experts) on an M2 Max; their decode layer runs 8 to 32 rows 16-19% faster at random
// routing and 1-4% at pooled than on the 8-row MMA. Other tiles run the MMAs that hold their live rows.
template <ushort Rows, ushort Tables, uint Shared, bool Softcap = false, uint Routes = 0>
kernel void moe_expert_gguf_mma_gate_up(device bfloat *input [[buffer(0)]],
                                        device const MoeTileDescriptor *tiles [[buffer(1)]],
                                        device const uint *tile_count [[buffer(2)]], device uchar *gsw0 [[buffer(3)]],
                                        device uchar *gsw1 [[buffer(4)]], device uchar *gsmeta [[buffer(5)]],
                                        device uchar *usw0 [[buffer(6)]], device uchar *usw1 [[buffer(7)]],
                                        device uchar *usmeta [[buffer(8)]], device bfloat *output [[buffer(9)]],
                                        device const uint *slots [[buffer(10)]],
                                        device const float *expert_scales [[buffer(11)]],
                                        constant MoeGgufGateUpParams &p [[buffer(12)]],
                                        constant ulong *chunks [[buffer(13)]],
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
  static_assert(!Routes || (Rows == 8 && Shared == 0), "direct input uses one routed fragment");
  device bfloat *x = Routes ? input : input + ulong(tile.first_row) * K;
  const ulong out = ulong(tile.first_row) * N;
  // A split pass's MOE_MMA_SPLIT threadgroups (moe_mma_split_format): the first runs an MMA tile, and of a column
  // tile each takes its part of the 32 columns.
  const bool split = moe_mma_split_pass<Rows, 4>();
  const uint tile_x = split ? group.x / MOE_MMA_SPLIT : group.x, part = split ? group.x % MOE_MMA_SPLIT : 0;
  const uint origin = tile_x * kColumns, thread_index = simd_group * 32 + simd_lane;
  const bool up = simd_group == 1;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  // A column tile reads its rows where they are: ungrouped (Routes) through the tile's grouped routes. Ungrouped
  // tiles take the column tile only where both projections' formats run it on the grid path.
  const auto row_input = [&](uint r) -> device const bfloat * {
    if constexpr (Routes)
      return input + ulong(reinterpret_cast<device const uint *>(gsw0)[tile.first_row + r] / Routes) * K;
    else
      return x + ulong(r) * K;
  };
  const bool column = moe_column_tile(tiles, t) &&
                      (!Routes || (moe_mma_grid_column_format<Rows>(gs.format) &&
                                   moe_mma_grid_column_format<Rows>(us.format)));
  const bool gate_column = column && gs.format == FmtQ20::Id, up_column = column && us.format == FmtQ20::Id;
  const bool grid = moe_mma_grid_column_pass<Rows, 4>() && column;
  const bool gate_grid = grid && moe_mma_grid_column_format<Rows>(gs.format);
  const bool up_grid = grid && moe_mma_grid_column_format<Rows>(us.format);
  const float scale = moe_expert_scale(expert_scales, tile.expert, p.experts, up ? 1 : 0);
  const auto product = [](float v, bfloat g) {
    const float u = float(bfloat(v));
    if constexpr (Softcap) return bfloat(SoftcappedSwiglu<4, 25>::value(u, float(g)));
    return bfloat(u * splash_silu(float(g)));
  };
  if (gate_column && up_column) {
    // As many simdgroups as the two passes' column tiles run: the even threadgroup's two take the odd one's columns
    // (N is a multiple of the passes' 64 columns, so every odd threadgroup has an even one).
    if (group.x % 2) return;
    // Q2_0 has no pair table, so tl is left unfilled here on purpose.
    const uint pair_column = origin + simd_group * kColumns + simd_lane;
    float g[GGUF_COLUMN_ROWS], u[GGUF_COLUMN_ROWS];
    column_dot_pair<FmtQ20, GGUF_COLUMN_ROWS>(x, tile.rows, K, gs.w0, gs.w1, gs.meta, us.w0, us.w1, us.meta,
                                                  pair_column, tl, g, u);
    const float gate_scale = moe_expert_scale(expert_scales, tile.expert, p.experts, 0);
    const float up_scale = moe_expert_scale(expert_scales, tile.expert, p.experts, 1);
    for (uint row = 0; row < tile.rows; ++row)
      output[out + ulong(row) * N + pair_column] = product(u[row] * up_scale, bfloat(g[row] * gate_scale));
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
  if (split && column) {
    // Each simdgroup its projection's columns of the threadgroup's part; the gate parks bf16(gate) for the up.
    float sums[GGUF_COLUMN_ROWS];
    const uint first = origin + part * kSplitWidth;
    moe_mma_split_column<Rows, 4>(s.format, row_input, tile.rows, K, s, first, simd_lane, table, sums);
    if (!up && simd_lane < kSplitWidth)
      for (uint row = 0; row < tile.rows; ++row) gate[row * kColumns + simd_lane] = bfloat(sums[row] * scale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up && simd_lane < kSplitWidth)
      for (uint row = 0; row < tile.rows; ++row)
        output[out + ulong(row) * N + first + simd_lane] = product(sums[row] * scale, gate[row * kColumns + simd_lane]);
    return;
  }
  if (split && part) return;
  if (column) {
    // At most GGUF_COLUMN_ROWS live rows, one fragment. Each simdgroup sums its projection on its own tile, the
    // gate parks bf16(gate) in its stage, and both meet at the one barrier before the up writes the products: the
    // simdgroups may take different tiles (a shared expert of Q2_0 gate and another up format), so neither tile's
    // branch holds a barrier of its own.
    const bool on_column = up ? up_column : gate_column;
    float sums[GGUF_COLUMN_ROWS];
    float2 acc[FN][1];
    const bool on_grid = up ? up_grid : gate_grid, on_dot = on_column || on_grid;
    if (on_column) {
      column_dot<FmtQ20, GGUF_COLUMN_ROWS>(x, tile.rows, K, s.w0, s.w1, s.meta, origin + simd_lane, 0, K / 32,
                                               table, sums);
    } else if (on_grid) {
      moe_mma_grid_column<Rows, 4>(s.format, [&](auto f) {
        column_dot_rows<decltype(f), GGUF_COLUMN_ROWS>(row_input, tile.rows, K, s.w0, s.w1, s.meta,
                                                       origin + simd_lane, 0, K / 32, table, sums);
      });
    } else {
      zero(acc);
      moe_mma_formats<4>()(s.format, [&](auto f) {
        accumulate<decltype(f), 1>(x, s.w0, s.w1, s.meta, K, origin, my, table, simd_lane, 0, K / kStep, acc);
      });
    }
    if (!up) {
      if (on_dot)
        for (uint row = 0; row < tile.rows; ++row) gate[row * kColumns + simd_lane] = bfloat(sums[row] * scale);
      else
        elements(acc, l, [&](uint row, uint column, float v) { gate[row * kColumns + column] = bfloat(v * scale); });
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!up) return;
    if (on_dot)
      for (uint row = 0; row < tile.rows; ++row)
        output[out + ulong(row) * N + origin + simd_lane] = product(sums[row] * scale, gate[row * kColumns + simd_lane]);
    else
      moe_mma_store<1, false>(acc, l, tile.rows, [&](uint row, uint column, float v) {
        output[out + ulong(row) * N + origin + column] = product(v * scale, gate[row * kColumns + column]);
      });
    return;
  }
  // Both simdgroups run the MMAs of the tile's live rows, which the tile gives them alike.
  const auto inputs = moe_mma_inputs<Routes>(input, gsw0, tile, K, l);
  moe_mma_live_rows<Rows>(tile.rows, [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    moe_mma_formats<4>()(s.format, [&](auto f) {
      accumulate<decltype(f), FM>(inputs, s.w0, s.w1, s.meta, K, origin, my, table, simd_lane, 0, K / kStep, acc);
    });
    if (!up) elements(acc, l, [&](uint row, uint column, float v) { gate[row * kColumns + column] = bfloat(v * scale); });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up) {
      // Reload the tile descriptor after the K loop so the 64-bit output offset and live row count are not held live
      // across accumulate; single guarded store path avoids duplicating the SiLU epilogue across fragment counts.
      const MoeTileDescriptor dst = tiles[t];
      const ulong dst_out = ulong(dst.first_row) * N;
      moe_mma_store<FM, false>(acc, l, dst.rows, [&](uint row, uint column, float v) {
        output[dst_out + ulong(row) * N + origin + column] = product(v * scale, gate[row * kColumns + column]);
      });
    }
  });
}
using MoeExpertGgufMmaGateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                          device uchar *, device uchar *, device uchar *, device uchar *,
                                          device uchar *, device uchar *, device bfloat *, device const uint *,
                                          device const float *, constant MoeGgufGateUpParams &, constant ulong *, uint2, uint, uint);
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

template [[host_name("moe_expert_gguf_mma_m8_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufMmaGateUpKernel
    moe_expert_gguf_mma_gate_up<8, 1, 0, true>;
template [[host_name("moe_expert_gguf_mma_m8_gate_up_s0_softcap4_25_routed8")]] kernel MoeExpertGgufMmaGateUpKernel
    moe_expert_gguf_mma_gate_up<8, 1, 0, true, 8>;
template [[host_name("moe_expert_gguf_mma_m32_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufMmaGateUpKernel
    moe_expert_gguf_mma_gate_up<32, 1, 0, true>;
template [[host_name("moe_expert_gguf_mma_m8_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufMmaGateUpKernel
    moe_expert_gguf_mma_gate_up<8, 2, 0, true>;
template [[host_name("moe_expert_gguf_mma_m32_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufMmaGateUpKernel
    moe_expert_gguf_mma_gate_up<32, 2, 0, true>;
