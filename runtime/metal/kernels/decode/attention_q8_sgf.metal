#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/paged_verify_tile.h"

// Same bindings as verify_attention_q8_split; the grid is (KV heads, splits,
// lanes) with 32 x G threads, one simdgroup per eight fused rows.
#define Q8_SGF_VERIFY_SPLIT(Name, Heads, Group)                                \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device int8_t *cache_keys [[buffer(1)]],                                 \
      device const float *key_scales_buffer [[buffer(2)]],                     \
      device int8_t *cache_values [[buffer(3)]],                               \
      device const float *value_scales_buffer [[buffer(4)]],                   \
      device float *partials [[buffer(5)]],                                    \
      device float *statistics [[buffer(6)]],                                  \
      device const uint *page_table0 [[buffer(7)]],                            \
      device const uint *page_table1 [[buffer(8)]],                            \
      device const uint *page_table2 [[buffer(9)]],                            \
      device const uint *page_table3 [[buffer(10)]],                           \
      constant SplashQ8VerifyAttentionParams *params [[buffer(11)]],           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    const SplashQ8VerifyTile tile =                                            \
        splash_q8_verify_attention_tile_at<Heads, Group>(                      \
            queries, page_table0, page_table1, page_table2, page_table3,       \
            params, group);                                                    \
    if (!tile.active || sg >= Group)                                           \
      return;                                                                  \
    q8sgf::tile<Heads, Group>(                                          \
        tile.queries, cache_keys, key_scales_buffer, cache_values,             \
        value_scales_buffer, tile.page_table, tile.kv_head,                    \
        tile.committed_tokens, tile.active_rows, tile.splits, tile.split,      \
        partials, statistics, tile.slot, sg, lane);                            \
  }

Q8_SGF_VERIFY_SPLIT(verify_attention_q8_split_sgf, 4, 6)
Q8_SGF_VERIFY_SPLIT(verify_attention_q8_split_sgf_kv2_g8, 2, 8)
#undef Q8_SGF_VERIFY_SPLIT
