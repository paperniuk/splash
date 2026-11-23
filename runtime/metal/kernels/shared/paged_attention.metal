#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/paged_attention_sgf.h"

// The page loop (splash_paged_attention_tile) over the tiles of a dispatch's
// sequences (SplashAttentionParams), each tile SPLASH_ATTENTION_TILE_ROWS
// query rows of one KV head: a decode step's lanes and a prompt chunk's
// sequences alike, every sequence in one dispatch, its rows those of its
// entry of the command's layout (`layout`). A decode step's split binds its
// tree rows (SplashAttentionParams::trees); a prompt's binds another buffer
// of the dispatch in their place, of which it reads nothing.

// Split: KV heads (and each one's tile parts, splash_attention_tile_part)
// vary first, then the dispatch's tiles, then history splits. A tile's
// sequence names its page table among the bindings.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, typename CacheElement,
          bool Int8Queries>
inline void splash_attention_split_phase(
    device bfloat *queries, device float *partials, device float *statistics,
    device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
    device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
    device const SplashTreeRow *tree, device const SplashLaneRows *layout, constant SplashAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, uint3 group, uint thread_index) {
  constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead);
  const uint split = group.z;
  const uint kv_head = group.x / Parts;
  const uint s = splash_attention_sequence_of_tile(params, group.y);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile =
      splash_attention_tile_part<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, Parts, HeadDim>(
          splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(params, sequence, rows, kv_head,
                                                                      group.y - sequence.first_tile),
          group.x % Parts, sequence.split_count);
  if (!tile.active || split >= tile.splits)
    return;
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS / Parts, HeadDim, Window,
                              CacheElement, Int8Queries>(
      queries + tile.offset, SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3),
      params.kv, kv_head, tile.first_position, tile.active_rows, tile.splits, split, partials, statistics,
      tile.first_slot + split, scores, probabilities, row_max, row_sum, previous_scale, rescale,
      splash_attention_tree(params, sequence, rows, tree, tile.first_position), thread_index);
}

// Reduce: a threadgroup of HeadDim threads per fused row of a tile, group.z
// the dispatch's tile, which writes the row as the out-projection's input
// (splash_attention_output_row): Gated by its query head's gate in its
// packed row, at its output row (splash_attention_output), and with Table
// into the out-projection's table too. A reduce that adds learned sinks
// passes the layer's, one per query head, last.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, bool Gated, uint Table, class... Sinks>
inline void splash_attention_reduce_phase(
    device const float *partials, device const float *statistics, device const bfloat *packed_qkv,
    device bfloat *hidden, device bfloat *table, device float *sums, device const SplashLaneRows *layout,
    constant SplashAttentionParams &params, uint3 group, uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, threadgroup bfloat *pairs, Sinks... sinks) {
  constexpr ushort M = SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  const uint fused_row = group.y;
  const uint s = splash_attention_sequence_of_tile(params, group.z);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, group.x, group.z - sequence.first_tile);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  const bfloat value = splash_attention_reduce_row<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS,
                                                   SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead), HeadDim, Window>(
      partials, statistics, tile, sequence.split_count, params.trees != 0, sequence.committed_tokens, fused_row,
      thread_index, weights, group_values, splash_group_sinks<QueryHeadsPerKVHead>(group.x, sinks)...);
  const SplashAttentionOutput place = splash_attention_output(
      params, s, rows, tile.first_position - sequence.committed_tokens + fused_row / QueryHeadsPerKVHead);
  splash_attention_output_row<KVHeads * QueryHeadsPerKVHead, KVHeads, HeadDim, Gated, Table>(
      value, packed_qkv, hidden, table, sums, place.packed, place.output,
      group.x * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead, thread_index, place.writes, pairs);
}

#define PAGED_ATTENTION_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement, Int8Queries)                   \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device float *statistics [[buffer(2)]],                                                                 \
      device const SplashKvPage *page_table0 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(6)]],                                                   \
      device const SplashTreeRow *tree [[buffer(7)]],                                                         \
      device const SplashLaneRows *layout [[buffer(8)]],                                                      \
      constant SplashAttentionParams &params [[buffer(9)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    constexpr uint M = Group * SPLASH_ATTENTION_TILE_ROWS / SPLASH_ATTENTION_TILE_PARTS(Group);               \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    splash_attention_split_phase<Heads, Group, HeadDim, Window, CacheElement, Int8Queries>(                   \
        queries, partials, statistics, page_table0, page_table1, page_table2, page_table3, tree, layout,     \
        params, scores, probabilities, row_max, row_sum, previous_scale, &rescale, group, thread_index);      \
  }

PAGED_ATTENTION_SPLIT(attention_q8_split, 4, 6, 256, 0, int8_t, false)
PAGED_ATTENTION_SPLIT(attention_q8_split_kv2_g8, 2, 8, 256, 0, int8_t, false)
PAGED_ATTENTION_SPLIT(attention_q8_split_kv2_g16, 2, 16, 256, 0, int8_t, false)
// Four KV heads of 128 dimensions over the whole history, and over a window
// of the 512 keys before each row.
PAGED_ATTENTION_SPLIT(attention_q8_split_kv4_g8_d128, 4, 8, 128, 0, int8_t, false)
PAGED_ATTENTION_SPLIT(attention_q8_split_kv4_g8_d128_w512, 4, 8, 128, 512, int8_t, false)
// INT8 query codes against the INT8 keys, where a split reads enough pages
// (AttentionCalibration::int8QueriesFromPages).
PAGED_ATTENTION_SPLIT(attention_q8q_split, 4, 6, 256, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv2_g8, 2, 8, 256, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv2_g16, 2, 16, 256, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv4_g8_d128, 4, 8, 128, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv4_g8_d128_w512, 4, 8, 128, 512, int8_t, true)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_ATTENTION_SPLIT(attention_bf16_split, 4, 6, 256, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv2_g8, 2, 8, 256, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv2_g16, 2, 16, 256, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv4_g8_d128, 4, 8, 128, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv4_g8_d128_w512, 4, 8, 128, 512, bfloat, false)
#undef PAGED_ATTENTION_SPLIT

// Apple7/8 register tile (paged_attention_sgf.h): the bindings and grid of
// the split above, one simdgroup per eight fused rows of the tile's part
// (32 x Group / Parts threads). Slots, partials and statistics are the MPP
// tile's, so the reduces below combine them unchanged. There are no INT8
// query codes: the register tile reads INT8 keys as exact half already.
#define SGF_ATTENTION_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement)                                 \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device float *statistics [[buffer(2)]],                                                                 \
      device const SplashKvPage *page_table0 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(6)]],                                                   \
      device const SplashTreeRow *tree [[buffer(7)]],                                                         \
      device const SplashLaneRows *layout [[buffer(8)]],                                                      \
      constant SplashAttentionParams &params [[buffer(9)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint sg [[simdgroup_index_in_threadgroup]],                                                             \
      uint lane [[thread_index_in_simdgroup]]) {                                                              \
    constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(Group);                                                \
    constexpr uint Simdgroups = Group / Parts;                                                                \
    threadgroup uint query_stage[Simdgroups * 32 * (HeadDim / 8)];                                            \
    const uint split = group.z, kv_head = group.x / Parts;                                                    \
    const uint s = splash_attention_sequence_of_tile(params, group.y);                                        \
    constant SplashAttentionSequence &sequence = params.sequence[s];                                          \
    const SplashAttentionRows rows = splash_attention_rows(layout, params, s);                                \
    const SplashAttentionTile tile =                                                                          \
        splash_attention_tile_part<Group, SPLASH_ATTENTION_TILE_ROWS, Parts, HeadDim>(                        \
            splash_attention_tile<Heads, Group, HeadDim>(params, sequence, rows, kv_head,                     \
                                                         group.y - sequence.first_tile),                      \
            group.x % Parts, sequence.split_count);                                                           \
    if (!tile.active || split >= tile.splits || sg >= Simdgroups)                                             \
      return;                                                                                                 \
    q8sgf::tile<Heads, Group, CacheElement, SPLASH_ATTENTION_TILE_ROWS / Parts, HeadDim, Window>(             \
        queries + tile.offset, SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3),   \
        params.kv, kv_head, tile.first_position, tile.active_rows, tile.splits, split, partials, statistics,  \
        tile.first_slot + split, splash_attention_tree(params, sequence, rows, tree, tile.first_position),    \
        sg, lane, query_stage);                                                                               \
  }

SGF_ATTENTION_SPLIT(attention_q8_split_sgf, 4, 6, 256, 0, int8_t)
SGF_ATTENTION_SPLIT(attention_q8_split_sgf_kv2_g8, 2, 8, 256, 0, int8_t)
SGF_ATTENTION_SPLIT(attention_q8_split_sgf_kv2_g16, 2, 16, 256, 0, int8_t)
SGF_ATTENTION_SPLIT(attention_q8_split_sgf_kv4_g8_d128, 4, 8, 128, 0, int8_t)
SGF_ATTENTION_SPLIT(attention_q8_split_sgf_kv4_g8_d128_w512, 4, 8, 128, 512, int8_t)
SGF_ATTENTION_SPLIT(attention_bf16_split_sgf, 4, 6, 256, 0, bfloat)
SGF_ATTENTION_SPLIT(attention_bf16_split_sgf_kv2_g8, 2, 8, 256, 0, bfloat)
SGF_ATTENTION_SPLIT(attention_bf16_split_sgf_kv2_g16, 2, 16, 256, 0, bfloat)
SGF_ATTENTION_SPLIT(attention_bf16_split_sgf_kv4_g8_d128, 4, 8, 128, 0, bfloat)
SGF_ATTENTION_SPLIT(attention_bf16_split_sgf_kv4_g8_d128_w512, 4, 8, 128, 512, bfloat)
#undef SGF_ATTENTION_SPLIT

// Bindings: the partials and statistics, the projection's packed rows (a
// gated geometry's gates, another buffer of the dispatch for an ungated one),
// the out-projection's input rows, its table and the table's sums (other
// buffers of the dispatch without a table), the command's layout, then the
// parameters; a reduce that adds learned sinks binds the layer's, an F32 per
// query head, before them. Each is compiled plain and, _table16, writing
// the out-projection's Table16 input too, or _table16f its Table16F input
// (Apple7/8's register tile).
#define PAGED_ATTENTION_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, Table)                       \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const bfloat *packed_qkv [[buffer(2)]],                                                          \
      device bfloat *hidden [[buffer(3)]],                                                                    \
      device bfloat *table [[buffer(4)]],                                                                     \
      device float *sums [[buffer(5)]],                                                                       \
      device const SplashLaneRows *layout [[buffer(6)]],                                                      \
      constant SplashAttentionParams &params [[buffer(7)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float weights[SplashAttentionMaximumSplits];                                                  \
    threadgroup float group_values[HeadDim / 32];                                                             \
    threadgroup bfloat pairs[HeadDim];                                                                        \
    splash_attention_reduce_phase<Heads, Group, HeadDim, Window, Gated, Table>(                               \
        partials, statistics, packed_qkv, hidden, table, sums, layout, params, group, thread_index, weights,   \
        group_values, pairs);                                                                                 \
  }
#define PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, Table)                  \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const bfloat *packed_qkv [[buffer(2)]],                                                          \
      device bfloat *hidden [[buffer(3)]],                                                                    \
      device bfloat *table [[buffer(4)]],                                                                     \
      device float *sums [[buffer(5)]],                                                                       \
      device const SplashLaneRows *layout [[buffer(6)]],                                                      \
      device const float *sinks [[buffer(7)]],                                                                \
      constant SplashAttentionParams &params [[buffer(8)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float weights[SplashAttentionMaximumSplits];                                                  \
    threadgroup float group_values[HeadDim / 32];                                                             \
    threadgroup bfloat pairs[HeadDim];                                                                        \
    splash_attention_reduce_phase<Heads, Group, HeadDim, Window, Gated, Table>(                               \
        partials, statistics, packed_qkv, hidden, table, sums, layout, params, group, thread_index, weights,   \
        group_values, pairs, sinks);                                                                          \
  }
#define PAGED_ATTENTION_REDUCE(Name, Heads, Group, HeadDim, Window, Gated)                                     \
  PAGED_ATTENTION_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, 0)                                 \
  PAGED_ATTENTION_REDUCE_KERNEL(Name##_table16, Heads, Group, HeadDim, Window, Gated, SplashAttentionTable16)   \
  PAGED_ATTENTION_REDUCE_KERNEL(Name##_table16f, Heads, Group, HeadDim, Window, Gated, SplashAttentionTable16F)
#define PAGED_ATTENTION_SINK_REDUCE(Name, Heads, Group, HeadDim, Window, Gated)                                \
  PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, 0)                            \
  PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name##_table16, Heads, Group, HeadDim, Window, Gated, SplashAttentionTable16) \
  PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name##_table16f, Heads, Group, HeadDim, Window, Gated, SplashAttentionTable16F)
PAGED_ATTENTION_REDUCE(attention_reduce, 4, 6, 256, 0, true)
PAGED_ATTENTION_REDUCE(attention_reduce_kv2_g8, 2, 8, 256, 0, true)
PAGED_ATTENTION_REDUCE(attention_reduce_kv2_g16, 2, 16, 256, 0, true)
// Flint's layers have no output gate.
PAGED_ATTENTION_SINK_REDUCE(attention_reduce_kv4_g8_d128_sinks, 4, 8, 128, 0, false)
PAGED_ATTENTION_SINK_REDUCE(attention_reduce_kv4_g8_d128_w512_sinks, 4, 8, 128, 512, false)
#undef PAGED_ATTENTION_SINK_REDUCE
#undef PAGED_ATTENTION_REDUCE
#undef PAGED_ATTENTION_SINK_REDUCE_KERNEL
#undef PAGED_ATTENTION_REDUCE_KERNEL

template <typename CacheElement>
inline void inkling_short_phase(device const bfloat *queries, device const bfloat *relative,
                                device const bfloat *projection, device const SplashKvPage *page_table,
                                device const SplashLaneRows *layout, device const SplashTreeRow *tree,
                                device float *workspace, constant SplashAttentionParams &params,
                                uint3 group, uint lane, uint sg) {
  constant SplashAttentionSequence &sequence = params.sequence[0];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, 0);
  if (group.y >= rows.rows || !splash_attention_rows_valid(params, sequence))
    return;
  const ulong row = ulong(rows.first) + group.y;
  const uint head = group.x * 3 + sg;
  const uint pos = sequence.committed_tokens + (params.trees ? tree[row].depth : group.y);
  const uint anchor = sequence.committed_tokens;
  const uint first = params.last_rows && pos > params.last_rows ? pos - params.last_rows : 0;
  const uint end = anchor + group.y + 1;
  const uint per = (end - first + sequence.split_count - 1) / sequence.split_count;
  const uint begin = min(end, first + group.z * per);
  const uint stop = min(end, begin + per);
  const uint ancestors = params.trees ? tree[row].ancestors : ~0u;
  const SplashKvAddressing<8, 128, CacheElement> addressing(params.kv, group.x);
  float q[4], accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint i = 0; i < 4; ++i)
    q[i] = float(queries[row * 3072 + head * 128 + lane + i * 32]);
  float maximum = -INFINITY, denominator = 0.0f;
  for (uint slot = begin; slot < stop; ++slot) {
    uint key_pos = slot;
    if (params.trees && slot >= anchor) {
      const uint tree_row = slot - anchor;
      if (!((ancestors >> tree_row) & 1u))
        continue;
      key_pos = anchor + tree[ulong(rows.first) + tree_row].depth;
    }
    if (key_pos > pos || (params.last_rows && pos - key_pos > params.last_rows))
      continue;
    const uint distance = pos - key_pos, token = slot % SplashKvPageTokens;
    const SplashKvPageTensors<CacheElement> page = addressing.page(page_table[slot / SplashKvPageTokens]);
    float dot = 0.0f;
    for (uint i = 0; i < 4; ++i)
      dot += q[i] * float(page.keys[token * 128 + lane + i * 32]);
    dot = simd_sum(dot);
    if constexpr (is_same<CacheElement, int8_t>::value)
      dot *= page.key_scales[token];
    const float product = distance < 1024 && lane < 16
        ? float(relative[row * 384 + head * 16 + lane]) * float(projection[lane * 1024 + distance]) : 0.0f;
    const float score = dot * 0.08838834764831845f + float(bfloat(simd_sum(product)));
    const float next = max(maximum, score);
    const float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next);
    const float weight = exp(score - next);
    float value_scale = 1.0f;
    if constexpr (is_same<CacheElement, int8_t>::value)
      value_scale = page.value_scales[token];
    for (uint i = 0; i < 4; ++i) {
      const uint dim = lane + i * 32;
      accum[i] = accum[i] * old_scale + weight * value_scale * float(page.values[dim * SplashKvPageTokens + token]);
    }
    denominator = denominator * old_scale + weight;
    maximum = next;
  }
  device float *partial = workspace + ((row * 24 + head) * sequence.split_count + group.z) * 130;
  for (uint i = 0; i < 4; ++i) partial[lane + i * 32] = accum[i];
  if (lane == 0) {
    partial[128] = maximum;
    partial[129] = denominator;
  }
}

#define INKLING_SHORT(Name, CacheElement) \
kernel void Name(device const bfloat *queries [[buffer(0)]], device const bfloat *relative [[buffer(1)]], \
                 device const bfloat *projection [[buffer(2)]], device const SplashKvPage *pages [[buffer(3)]], \
                 device const SplashLaneRows *layout [[buffer(4)]], device const SplashTreeRow *tree [[buffer(5)]], \
                 device float *workspace [[buffer(6)]], constant SplashAttentionParams &params [[buffer(7)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
  inkling_short_phase<CacheElement>(queries, relative, projection, pages, layout, tree, workspace, params, \
                                    group, lane, sg); \
}
INKLING_SHORT(attention_q8_inkling_short, int8_t)
INKLING_SHORT(attention_bf16_inkling_short, bfloat)
#undef INKLING_SHORT

kernel void attention_inkling_bias(device const bfloat *relative [[buffer(0)]],
                                   device const bfloat *projection [[buffer(1)]],
                                   device float *workspace [[buffer(2)]],
                                   device const SplashLaneRows *layout [[buffer(3)]],
                                   constant SplashAttentionParams &params [[buffer(4)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
  const SplashAttentionRows rows = splash_attention_rows(layout, params, 0);
  if (group.y >= rows.rows)
    return;
  const ulong row = ulong(rows.first) + group.y;
  const uint distance = group.z * 128 + lane;
  float bias = 0.0f;
#pragma unroll
  for (uint k = 0; k < 16; ++k)
    bias += float(relative[row * 384 + group.x * 16 + k]) * float(projection[k * 1024 + distance]);
  device bfloat *biases = reinterpret_cast<device bfloat *>(
      workspace + ulong(params.rows) * 24 * params.tiles * 130);
  biases[(row * 24 + group.x) * 1024 + distance] = bfloat(bias);
}

// The distances an inline bias stage holds a fused query (inkling_split_phase): a page's 32 keys from rows up to 8
// positions apart.
constant constexpr uint kInklingBiasSpan = 40;
template <typename CacheElement, uint TileRows, bool CachedBias>
inline void inkling_split_phase(device const bfloat *queries, device const bfloat *relative,
                                device const bfloat *projection, device const SplashKvPage *page_table,
                                device const SplashLaneRows *layout, device const SplashTreeRow *tree,
                                device float *workspace, constant SplashAttentionParams &params,
                                uint3 group, uint lane, uint sg, threadgroup uint *query_stage,
                                threadgroup float *bias_stage) {
  constant SplashAttentionSequence &sequence = params.sequence[0];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, 0);
  const uint first_row = group.y * TileRows;
  if (first_row >= rows.rows || !splash_attention_rows_valid(params, sequence))
    return;
  const uint active = min(TileRows, rows.rows - first_row);
  if (sg * 8 >= active * 3)
    return;
  const uint anchor = sequence.committed_tokens;
  const uint visible = anchor + first_row + active;
  const uint first_pos = anchor + first_row;
  const uint first_page = params.last_rows && !params.trees && first_pos > params.last_rows
                              ? (first_pos - params.last_rows) / SplashKvPageTokens
                              : params.last_rows && params.trees && anchor > params.last_rows
                                    ? (anchor - params.last_rows) / SplashKvPageTokens : 0;
  const uint pages = splash_attention_pages(visible);
  const uint per = splash_attention_pages_per_split(pages - first_page, sequence.split_count);
  const uint begin = min(pages, first_page + group.z * per);
  const uint end = min(pages, begin + per);
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint fused = sg * 8 + fn;
  const uint r0 = min(fused / 3, active - 1);
  const uint r1 = min((fused + 1) / 3, active - 1);
  const uint h0 = group.x * 3 + fused % 3;
  const uint h1 = group.x * 3 + (fused + 1) % 3;
  const ulong row0 = ulong(rows.first) + first_row + r0;
  const ulong row1 = ulong(rows.first) + first_row + r1;
  const uint pos0 = anchor + (params.trees ? tree[row0].depth : first_row + r0);
  const uint pos1 = anchor + (params.trees ? tree[row1].depth : first_row + r1);
  const uint anc0 = params.trees ? tree[row0].ancestors : ~0u;
  const uint anc1 = params.trees ? tree[row1].ancestors : ~0u;
  threadgroup uint *qstage = query_stage + sg * 512;
  device const ushort *q0 = reinterpret_cast<device const ushort *>(queries + row0 * 3072 + h0 * 128);
  device const ushort *q1 = reinterpret_cast<device const ushort *>(queries + row1 * 3072 + h1 * 128);
#pragma unroll
  for (uint b = 0; b < 2; ++b)
#pragma unroll
    for (uint t = 0; t < 8; ++t) {
      const uint d = 64 * b + q8sgf::block_dimension(fm, t);
      qstage[(b * 8 + t) * 32 + lane] = uint(q0[d]) | (uint(q1[d]) << 16);
    }
  float2 output[16];
#pragma unroll
  for (uint d = 0; d < 16; ++d) output[d] = float2(0.0f);
  float2 maximum(-INFINITY), denominator(0.0f);
  const SplashKvAddressing<8, 128, CacheElement> addressing(params.kv, group.x);
  // The relative bias reaches keys under 1024 positions before a row: a page every key of which lies that far
  // before the tile's first row, as most of a long history's do, skips its sums. Summed for every key, their
  // device loads took 63% of a 512-row prompt chunk's full-layer attention at a 64K history on an M1 Max.
  const uint nearest_row = params.trees ? anchor : first_pos;
  // An inline bias as one matrix product a page: the simdgroup's eight fused queries' relative queries (A, 8 x 16,
  // loaded once) times the projection at the kInklingBiasSpan distances from its lowest row less the page's last
  // key (B, 16 x span) into its stage, from which each score takes its distance's. Summed key by key, the bias was
  // 65% of a 512-row prompt chunk's sliding attention on an M1 Max. A tree's rows (a short history's verify,
  // whose draft keys may sit before their page) sum key by key.
  threadgroup float *bias_rows = bias_stage + sg * 8 * kInklingBiasSpan;
  const uint pos_lo = simd_min(min(pos0, pos1)), pos_hi = simd_max(max(pos0, pos1));
  const bool matrix_bias = !CachedBias && !params.trees && pos_hi - pos_lo + 32 <= kInklingBiasSpan;
  float2 rel_a[2] = {float2(0.0f), float2(0.0f)};
  if (!CachedBias && matrix_bias) {
    const uint fq = sg * 8 + fm;
    const ulong row_q = ulong(rows.first) + first_row + min(fq / 3, active - 1);
    const uint head_q = group.x * 3 + fq % 3;
#pragma unroll
    for (uint step = 0; step < 2; ++step)
      rel_a[step] = q8sgf::bfloat_pair(
          reinterpret_cast<device const uint *>(relative + row_q * 384 + head_q * 16 + step * 8 + fn)[0]);
  }
  for (uint page = begin; page < end; ++page) {
    const SplashKvPageTensors<CacheElement> tensors = addressing.page(page_table[page]);
    const bool near = page * 32 + 31 + 1024 > nearest_row;
    // The span's first distance: the lowest row's from the page's last key (negative distances are masked).
    const int first_distance = int(pos_lo) - int(page * 32 + 31);
    if (!CachedBias && matrix_bias && near) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (uint tile = 0; tile < kInklingBiasSpan / 8; ++tile) {
        float2 c(0.0f);
#pragma unroll
        for (uint step = 0; step < 2; ++step) {
          const int d = first_distance + int(tile * 8 + fn);
          const uint k = step * 8 + fm;
          const float b0 = d >= 0 && d < 1024 ? float(projection[k * 1024 + d]) : 0.0f;
          const float b1 = d + 1 >= 0 && d + 1 < 1024 ? float(projection[k * 1024 + d + 1]) : 0.0f;
          sgmatrix::mma_acc<float>(c, rel_a[step], float2(b0, b1));
        }
        bias_rows[fm * kInklingBiasSpan + tile * 8 + fn] = c.x;
        bias_rows[fm * kInklingBiasSpan + tile * 8 + fn + 1] = c.y;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    float2 scores[4] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
#pragma unroll
    for (uint b = 0; b < 2; ++b)
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        q8sgf::KeyRun<CacheElement> run;
        run.load(tensors.keys + q8sgf::page_token(fm, s) * 128 + 64 * b + 8 * fn);
#pragma unroll
        for (uint t = 0; t < 8; ++t)
          q8sgf::mma(scores[s], run.step(t), q8sgf::bfloat_pair(qstage[(b * 8 + t) * 32 + lane]));
      }
    bool2 seen[4];
    float2 local_max(-INFINITY);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = page * 32 + q8sgf::page_token(fm, s);
      uint key_pos = token;
      if (params.trees && token >= anchor && token < visible)
        key_pos = anchor + tree[ulong(rows.first) + token - anchor].depth;
      const bool on0 = token < anchor || !params.trees || ((anc0 >> min(token - anchor, 31u)) & 1u);
      const bool on1 = token < anchor || !params.trees || ((anc1 >> min(token - anchor, 31u)) & 1u);
      const bool valid0 = token < anchor + first_row + r0 + 1 && key_pos <= pos0 && on0 &&
                          (!params.last_rows || pos0 - key_pos <= params.last_rows);
      const bool valid1 = token < anchor + first_row + r1 + 1 && key_pos <= pos1 && on1 &&
                          (!params.last_rows || pos1 - key_pos <= params.last_rows);
      seen[s] = bool2(valid0, valid1);
      float2 bias(0.0f);
      const uint dist0 = pos0 - key_pos, dist1 = pos1 - key_pos;
      if (!near) {
      } else if constexpr (CachedBias) {
        device const bfloat *biases = reinterpret_cast<device const bfloat *>(
            workspace + ulong(params.rows) * 24 * params.tiles * 130);
        if (valid0 && dist0 < 1024)
          bias.x = float(biases[(row0 * 24 + h0) * 1024 + dist0]);
        if (valid1 && dist1 < 1024)
          bias.y = float(biases[(row1 * 24 + h1) * 1024 + dist1]);
      } else if (matrix_bias) {
        if (valid0 && dist0 < 1024) bias.x = bias_rows[fn * kInklingBiasSpan + uint(int(dist0) - first_distance)];
        if (valid1 && dist1 < 1024)
          bias.y = bias_rows[(fn + 1) * kInklingBiasSpan + uint(int(dist1) - first_distance)];
      } else {
#pragma unroll
        for (uint k = 0; k < 16; ++k) {
          if (valid0 && dist0 < 1024)
            bias.x += float(relative[row0 * 384 + h0 * 16 + k]) * float(projection[k * 1024 + dist0]);
          if (valid1 && dist1 < 1024)
            bias.y += float(relative[row1 * 384 + h1 * 16 + k]) * float(projection[k * 1024 + dist1]);
        }
      }
      if constexpr (is_same<CacheElement, int8_t>::value)
        scores[s] *= tensors.key_scales[token % 32];
      scores[s] = scores[s] * 0.08838834764831845f + float2(bfloat(bias.x), bfloat(bias.y));
      scores[s] = float2(valid0 ? scores[s].x : -INFINITY, valid1 ? scores[s].y : -INFINITY);
      local_max = max(local_max, scores[s]);
    }
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next = max(maximum, local_max);
    float2 weights[4], sum(0.0f);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      float2 p(seen[s].x ? fast::exp(scores[s].x - next.x) : 0.0f,
               seen[s].y ? fast::exp(scores[s].y - next.y) : 0.0f);
      sum += p;
      if constexpr (is_same<CacheElement, int8_t>::value)
        p *= tensors.value_scales[q8sgf::page_token(fm, s)];
      weights[s] = p;
    }
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 16);
    const float2 scale(next.x == -INFINITY || next.x == maximum.x ? 1.0f : fast::exp(maximum.x - next.x),
                       next.y == -INFINITY || next.y == maximum.y ? 1.0f : fast::exp(maximum.y - next.y));
    denominator = denominator * scale + sum;
    maximum = next;
#pragma unroll
    for (uint d = 0; d < 16; ++d) {
      output[d] *= scale;
      q8sgf::ValueRun<CacheElement> run;
      run.load(tensors.values + (fm + d * 8) * 32 + 4 * fn);
      q8sgf::mma(output[d], run.step(0), weights[0]);
      q8sgf::mma(output[d], run.step(1), weights[1]);
      q8sgf::mma(output[d], run.step(2), weights[2]);
      q8sgf::mma(output[d], run.step(3), weights[3]);
    }
  }
  if (fused / 3 < active) {
    device float *part = workspace + ((row0 * 24 + h0) * sequence.split_count + group.z) * 130;
#pragma unroll
    for (uint d = 0; d < 16; ++d) part[8 * d + fm] = output[d].x;
    if (fm == 0) { part[128] = maximum.x; part[129] = denominator.x; }
  }
  if ((fused + 1) / 3 < active) {
    device float *part = workspace + ((row1 * 24 + h1) * sequence.split_count + group.z) * 130;
#pragma unroll
    for (uint d = 0; d < 16; ++d) part[8 * d + fm] = output[d].y;
    if (fm == 0) { part[128] = maximum.y; part[129] = denominator.y; }
  }
}

#define INKLING_SPLIT(Name, CacheElement, TileRows, CachedBias) \
kernel void Name(device const bfloat *queries [[buffer(0)]], device const bfloat *relative [[buffer(1)]], \
                 device const bfloat *projection [[buffer(2)]], device const SplashKvPage *pages [[buffer(3)]], \
                 device const SplashLaneRows *layout [[buffer(4)]], device const SplashTreeRow *tree [[buffer(5)]], \
                 device float *workspace [[buffer(6)]], constant SplashAttentionParams &params [[buffer(7)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
  threadgroup uint query_stage[(TileRows * 3 / 8) * 512]; \
  threadgroup float bias_stage[CachedBias ? 1 : (TileRows * 3 / 8) * 8 * kInklingBiasSpan]; \
  inkling_split_phase<CacheElement, TileRows, CachedBias>(queries, relative, projection, pages, layout, tree, workspace, params, \
                                     group, lane, sg, query_stage, bias_stage); \
}
INKLING_SPLIT(attention_q8_inkling_split, int8_t, 8, false)
INKLING_SPLIT(attention_bf16_inkling_split, bfloat, 8, false)
INKLING_SPLIT(attention_q8_inkling_staged, int8_t, 8, true)
INKLING_SPLIT(attention_bf16_inkling_staged, bfloat, 8, true)
INKLING_SPLIT(attention_q8_inkling_prefill, int8_t, 16, false)
INKLING_SPLIT(attention_bf16_inkling_prefill, bfloat, 16, false)
#undef INKLING_SPLIT

kernel void attention_inkling_reduce(
    device const float *workspace [[buffer(0)]], device const bfloat *sinks [[buffer(1)]],
    device const bfloat *gate [[buffer(2)]], device bfloat *output [[buffer(3)]],
    device const SplashLaneRows *layout [[buffer(4)]], constant SplashAttentionParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constant SplashAttentionSequence &sequence = params.sequence[0];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, 0);
  if (group.y >= rows.rows)
    return;
  const ulong row = ulong(rows.first) + group.y;
  const uint head = group.x * 3 + sg;
  // Lane j holds splits j and j + 32 (at most 64), so their maxima and denominators load at once; then each
  // split's outputs, eight splits' loads in flight: a loop that waits for each split's load in turn took 20 us
  // at 16 splits on an M1 Max.
  device const float *partials = workspace + (row * 24 + head) * sequence.split_count * 130;
  const uint splits = sequence.split_count;
  const float sink = float(sinks[row * 24 + head]);
  float m[2], d[2];
  for (uint h = 0; h < 2; ++h) {
    const uint split = min(lane + h * 32, splits - 1);
    m[h] = partials[split * 130 + 128];
    d[h] = partials[split * 130 + 129];
    if (lane + h * 32 >= splits) m[h] = -INFINITY;
  }
  const float maximum = max(sink, simd_max(max(m[0], m[1])));
  float w[2];
  for (uint h = 0; h < 2; ++h) w[h] = m[h] == -INFINITY ? 0.0f : exp(m[h] - maximum);
  const float denominator = exp(sink - maximum) + simd_sum(w[0] * d[0] + w[1] * d[1]);
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll(8)
  for (uint split = 0; split < splits; ++split) {
    device const float *part = partials + split * 130;
    const float weight = simd_shuffle(split < 32 ? w[0] : w[1], ushort(split & 31));
    for (uint i = 0; i < 4; ++i) accum[i] += weight * part[lane + i * 32];
  }
  for (uint i = 0; i < 4; ++i) {
    const ulong index = row * 3072 + head * 128 + lane + i * 32;
    const float attention = float(bfloat(accum[i] / denominator));
    const float g = float(gate[row * INKLING_PACKED_WIDTH + INKLING_GATE_OFFSET + head * 128 + lane + i * 32]);
    output[index] = bfloat(attention / (1.0f + exp(-g)));
  }
}
