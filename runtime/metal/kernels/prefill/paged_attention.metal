#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (splash_paged_attention_tile) over eight query rows of one KV head.

// Prefill split: KV heads (and each one's tile parts,
// splash_attention_tile_part) vary first, then query tiles, then history
// splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, typename CacheElement>
inline void splash_prefill_attention_split_phase(
    device bfloat *queries, device float *partials,
    device float *statistics, device const SplashKvPage *page_table,
    constant SplashPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group,
    uint thread_index) {
  constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead);
  const uint split = group.z;
  const uint kv_head = group.x / Parts;
  const SplashAttentionTile tile =
      splash_attention_tile_part<QueryHeadsPerKVHead, SPLASH_PREFILL_ATTENTION_TILE_ROWS, Parts, HeadDim>(
          splash_prefill_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(params, kv_head, group.y),
          group.x % Parts, params.split_count);
  if (!tile.active || split >= tile.splits)
    return;
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                              SPLASH_PREFILL_ATTENTION_TILE_ROWS / Parts, HeadDim, Window, CacheElement>(
      queries + tile.offset, page_table, params.kv, kv_head, tile.first_position,
      tile.active_rows, tile.splits, split, partials, statistics,
      tile.first_slot + split, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, thread_index);
}

// Prefill reduce: a threadgroup of HeadDim threads per fused row. A reduce
// that adds learned sinks passes the layer's, one per query head, last.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, class... Sinks>
inline void splash_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashPrefillAttentionParams &params, uint3 group,
    uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, Sinks... sinks) {
  constexpr ushort M = SPLASH_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  const uint fused_row = group.y;
  const SplashAttentionTile tile =
      splash_prefill_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(params, group.x, group.z);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  splash_attention_reduce_row<QueryHeadsPerKVHead,
                              SPLASH_PREFILL_ATTENTION_TILE_ROWS,
                              SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead), HeadDim, Window>(
      partials, statistics, output, tile, params.split_count, fused_row,
      thread_index, weights, group_values,
      splash_group_sinks<QueryHeadsPerKVHead>(group.x, sinks)...);
}

#define PAGED_PREFILL_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement) \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * SPLASH_PREFILL_ATTENTION_TILE_ROWS /            \
                       SPLASH_ATTENTION_TILE_PARTS(Group);                     \
    constexpr uint N = SplashKvPageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    splash_prefill_attention_split_phase<Heads, Group, HeadDim, Window,        \
                                         CacheElement>(                        \
        queries, partials, statistics, page_table, params, scores,             \
        probabilities, row_max, row_sum, previous_scale, &rescale, group,      \
        thread_index);                                                         \
  }

PAGED_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, 256, 0, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, 256, 0, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g16, 2, 16, 256, 0, int8_t)
// Four KV heads of 128 dimensions over the whole history, and over a window
// of the 512 keys before each row.
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g8_d128, 4, 8, 128, 0, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g8_d128_w512, 4, 8, 128, 512, int8_t)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6, 256, 0, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8, 256, 0, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g16, 2, 16, 256, 0, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g8_d128, 4, 8, 128, 0, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g8_d128_w512, 4, 8, 128, 512, bfloat)
#undef PAGED_PREFILL_SPLIT

// Apple7/8 register tile (paged_attention_sgf.h): the bindings and grid of
// the split above, one simdgroup per eight fused rows of the tile's part
// (32 x Group / Parts threads). Slots, partials and statistics are the MPP
// tile's, so the reduce below combines them unchanged. HeadDim 128 is Flint's
// layouts', and Window, as the MPP split's, a sliding-window layer's.
#define SGF_PREFILL_SPLIT(Name, Heads, Group, CacheElement, HeadDim, Window)    \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(Group);                 \
    constexpr uint Simdgroups = Group / Parts;                                 \
    threadgroup uint query_stage[Simdgroups * 32 * (HeadDim / 8)];             \
    const uint split = group.z, kv_head = group.x / Parts;                     \
    const SplashAttentionTile tile =                                           \
        splash_attention_tile_part<Group, SPLASH_PREFILL_ATTENTION_TILE_ROWS,  \
                                   Parts, HeadDim>(                            \
            splash_prefill_attention_tile<Heads, Group, HeadDim>(              \
                params, kv_head, group.y),                                     \
            group.x % Parts, params.split_count);                              \
    if (!tile.active || split >= tile.splits || sg >= Simdgroups)              \
      return;                                                                  \
    q8sgf::tile<Heads, Group, CacheElement,                                    \
                SPLASH_PREFILL_ATTENTION_TILE_ROWS / Parts, HeadDim, Window>(  \
        queries + tile.offset, page_table, params.kv, kv_head,                 \
        tile.first_position, tile.active_rows, tile.splits, split, partials,   \
        statistics, tile.first_slot + split, sg, lane, query_stage);           \
  }

SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf, 4, 6, int8_t, SPLASH_KV_HEAD_DIMENSION, 0)
SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv2_g8, 2, 8, int8_t, SPLASH_KV_HEAD_DIMENSION, 0)
SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv2_g16, 2, 16, int8_t, SPLASH_KV_HEAD_DIMENSION, 0)
SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv4_g8_d128, 4, 8, int8_t, 128, 0)
SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv4_g8_d128_w512, 4, 8, int8_t, 128, 512)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf, 4, 6, bfloat, SPLASH_KV_HEAD_DIMENSION, 0)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf_kv2_g8, 2, 8, bfloat, SPLASH_KV_HEAD_DIMENSION, 0)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf_kv2_g16, 2, 16, bfloat, SPLASH_KV_HEAD_DIMENSION, 0)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf_kv4_g8_d128, 4, 8, bfloat, 128, 0)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf_kv4_g8_d128_w512, 4, 8, bfloat, 128, 512)
#undef SGF_PREFILL_SPLIT

#define PAGED_PREFILL_REDUCE(Name, Heads, Group, HeadDim, Window)              \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant SplashPrefillAttentionParams &params [[buffer(3)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[HeadDim / 32];                              \
    splash_prefill_attention_reduce_phase<Heads, Group, HeadDim, Window>(      \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values);                                                         \
  }

// A reduce with learned sinks binds the layer's, an F32 per query head,
// before its parameters.
#define PAGED_PREFILL_SINK_REDUCE(Name, Heads, Group, HeadDim, Window)         \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device const float *sinks [[buffer(3)]],                                 \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[HeadDim / 32];                              \
    splash_prefill_attention_reduce_phase<Heads, Group, HeadDim, Window>(      \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values, sinks);                                                  \
  }
PAGED_PREFILL_REDUCE(prefill_attention_reduce, 4, 6, 256, 0)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g8, 2, 8, 256, 0)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g16, 2, 16, 256, 0)
PAGED_PREFILL_SINK_REDUCE(prefill_attention_reduce_kv4_g8_d128_sinks, 4, 8, 128, 0)
PAGED_PREFILL_SINK_REDUCE(prefill_attention_reduce_kv4_g8_d128_w512_sinks, 4, 8, 128, 512)
#undef PAGED_PREFILL_SINK_REDUCE
#undef PAGED_PREFILL_REDUCE
