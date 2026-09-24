#include "metal/kernels/common/paged_attention_sgf.h"

// Same bindings and grid as prefill_attention_q8_split (KV heads, query
// tiles, history splits) with 32 x G threads, one simdgroup per eight fused
// rows. Slots, partials and statistics are the MPP tile's, so
// prefill_attention_q8_reduce combines them unchanged.
#define Q8_SGF_PREFILL_SPLIT(Name, Heads, Group)                               \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device int8_t *cache_keys [[buffer(1)]],                                 \
      device const float *key_scales_buffer [[buffer(2)]],                     \
      device int8_t *cache_values [[buffer(3)]],                               \
      device const float *value_scales_buffer [[buffer(4)]],                   \
      device float *partials [[buffer(5)]],                                    \
      device float *statistics [[buffer(6)]],                                  \
      device const uint *page_table [[buffer(7)]],                             \
      constant SplashQ8PrefillAttentionParams &params [[buffer(8)]],           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    constexpr uint D = SplashQ8HeadDimension;                                  \
    const uint kv_head = group.x, tile = group.y, split = group.z;             \
    const uint tile_start = tile * SplashPrefillTileRows;                      \
    if (!splash_q8_prefill_attention_contract_valid(params) ||                 \
        kv_head >= Heads || split >= params.split_count ||                     \
        tile_start >= params.rows || sg >= Group)                              \
      return;                                                                  \
    const uint active_rows =                                                   \
        min(SplashPrefillTileRows, params.rows - tile_start);                  \
    const ulong tile_offset =                                                  \
        (ulong(kv_head) * params.chunk_stride + tile_start) * Group * D;       \
    const ulong slot =                                                         \
        (ulong(tile) * Heads + kv_head) * params.split_count + split;          \
    q8sgf::tile<Heads, Group>(                                                 \
        queries + tile_offset, cache_keys, key_scales_buffer, cache_values,    \
        value_scales_buffer, page_table, kv_head,                              \
        params.committed_tokens + tile_start, active_rows, params.split_count, \
        split, partials, statistics, slot, sg, lane);                          \
  }

Q8_SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf, 4, 6)
Q8_SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv2_g8, 2, 8)
#undef Q8_SGF_PREFILL_SPLIT
