#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

// The q/k norms and RoPE of a layer's packed projection rows, which stage
// its queries and store its keys and values in their pages, over the rows of
// a dispatch's sequences (SplashAttentionParams): a threadgroup per (head,
// row, sequence), each sequence's rows from the command's layout (`layout`,
// a SplashLaneRows a sequence).

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32);
// CacheElement and ValuesTokenMajor: the layer's pages' (int8_t for q8_, and
// token-major values for the selected blocks' readers); HeadDim, RotaryPairs
// and Gated: full_qkv_storage_phase's. A sequence's page table is its binding
// among the eight (SPLASH_LANE_BINDING).
#define ATTENTION_QKV(Name, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, CacheElement, ValuesTokenMajor)     \
  kernel void Name(                                                                                          \
      device const bfloat *qkv [[buffer(0)]],                                                                \
      device const W *q_norm [[buffer(1)]],                                                                  \
      device const W *k_norm [[buffer(2)]],                                                                  \
      device const float *rope_cos [[buffer(3)]],                                                            \
      device const float *rope_sin [[buffer(4)]],                                                            \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device const SplashKvPage *page_table0 [[buffer(6)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(7)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(8)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(9)]], device const SplashKvPage *page_table4 [[buffer(10)]], device const SplashKvPage *page_table5 [[buffer(11)]], device const SplashKvPage *page_table6 [[buffer(12)]], device const SplashKvPage *page_table7 [[buffer(13)]],                                                  \
      device const SplashLaneRows *layout [[buffer(14)]],                                                    \
      constant SplashAttentionParams &params [[buffer(15)]],                                                 \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, CacheElement, ValuesTokenMajor>(  \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries,                                                    \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7), layout, params,     \
        reductions, normalized, maxima, group, thread_index, lane, simd_group);                              \
  }

// A layer without RoPE (NoPE) binds no tables.
#define ATTENTION_QKV_NOPE(Name, QHeads, KHeads, HeadDim, Gated, W, CacheElement, ValuesTokenMajor)            \
  kernel void Name(                                                                                          \
      device const bfloat *qkv [[buffer(0)]],                                                                \
      device const W *q_norm [[buffer(1)]],                                                                  \
      device const W *k_norm [[buffer(2)]],                                                                  \
      device bfloat *queries [[buffer(3)]],                                                                  \
      device const SplashKvPage *page_table0 [[buffer(4)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(5)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(6)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(7)]], device const SplashKvPage *page_table4 [[buffer(8)]], device const SplashKvPage *page_table5 [[buffer(9)]], device const SplashKvPage *page_table6 [[buffer(10)]], device const SplashKvPage *page_table7 [[buffer(11)]],                                                  \
      device const SplashLaneRows *layout [[buffer(12)]],                                                     \
      constant SplashAttentionParams &params [[buffer(13)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, 0, Gated, W, CacheElement, ValuesTokenMajor>(            \
        qkv, q_norm, k_norm, nullptr, nullptr, queries,                                                      \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7), layout, params,     \
        reductions, normalized, maxima, group, thread_index, lane, simd_group);                              \
  }
// Each geometry's instances for its pages' formats and the norm types its
// families load: bf16 (an MLX target's) and F32, or F32 alone for
// Qwen3.8-Flash-Next's (_kv2_g12, whose pages hold token-major values) and
// Flint's (_kv4_g8_d128), whose MLX images widen their norms to the F32 their
// GGUFs hold (model/SafetensorsImage.hpp float32Norms).
#define ATTENTION_QKV_FORMATS(Suffix, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, ValuesTokenMajor)         \
  ATTENTION_QKV(attention_q8_qkv##Suffix, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, int8_t,               \
                ValuesTokenMajor)                                                                            \
  ATTENTION_QKV(attention_bf16_qkv##Suffix, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, bfloat,             \
                ValuesTokenMajor)
ATTENTION_QKV_FORMATS(, 24, 4, 256, 32, true, bfloat, false)
ATTENTION_QKV_FORMATS(_kv2_g8, 16, 2, 256, 32, true, bfloat, false)
ATTENTION_QKV_FORMATS(_f32, 24, 4, 256, 32, true, float, false)
ATTENTION_QKV_FORMATS(_kv2_g8_f32, 16, 2, 256, 32, true, float, false)
ATTENTION_QKV_FORMATS(_kv2_g12_f32, 24, 2, 256, 32, true, float, true)
ATTENTION_QKV_FORMATS(_kv2_g16, 32, 2, 256, 32, true, bfloat, false)
ATTENTION_QKV_FORMATS(_kv2_g16_f32, 32, 2, 256, 32, true, float, false)
// Four KV heads of 128 dimensions without an output gate, their whole heads
// rotated (64 pairs) or unrotated.
ATTENTION_QKV_FORMATS(_kv4_g8_d128_f32, 32, 4, 128, 64, false, float, false)
ATTENTION_QKV_NOPE(attention_q8_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float, int8_t, false)
ATTENTION_QKV_NOPE(attention_bf16_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float, bfloat, false)
#undef ATTENTION_QKV_FORMATS
#undef ATTENTION_QKV_NOPE
#undef ATTENTION_QKV

template <typename CacheElement>
inline void inkling_prepare_phase(
    device const bfloat *q, device const bfloat *k, device const bfloat *v,
    device const bfloat *q_scale, device const bfloat *rel_scale, device bfloat *queries,
    device bfloat *relative, device const SplashKvPage *page_table, device const SplashLaneRows *layout,
    device const SplashTreeRow *tree, constant SplashAttentionParams &params, uint3 group,
    uint dimension, uint lane, uint sg, threadgroup float *sums, threadgroup float *maxima) {
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, group.z);
  if (!splash_attention_rows_valid(params, sequence) || group.y >= rows.rows)
    return;
  const ulong row = ulong(rows.first) + group.y;
  const bool query = group.x < 24;
  const uint head = query ? group.x : group.x - 24;
  float x = query ? float(q[row * INKLING_PACKED_WIDTH + head * 128 + dimension]) *
                        float(q_scale[head * 128 + dimension])
                  : float(k[row * 1024 + head * 128 + dimension]);
  const float y = query ? 0.0f : float(v[row * 1024 + head * 128 + dimension]);
  const float x_sum = simd_sum(x * x);
  const float y_sum = simd_sum(y * y);
  if (lane == 0) {
    sums[sg] = x_sum;
    sums[4 + sg] = y_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float norm = 0.0f, v_norm = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    norm += sums[i];
    v_norm += sums[4 + i];
  }
  x *= min(22.627416997969522f / (sqrt(norm) + 1e-6f), 1.0f);
  const bfloat clipped = bfloat(x);
  if (query) {
    const uint position = sequence.committed_tokens + (params.trees ? tree[row].depth : group.y);
    const float tau = params.last_rows ? 1.0f + 0.1f * log(max(float(position + 1) / 8192.0f, 1.0f)) : 1.0f;
    queries[row * 3072 + head * 128 + dimension] = bfloat(float(clipped) * tau);
    if (sg == 0) {
      const float rv = lane < 16 ? float(q[row * INKLING_PACKED_WIDTH + INKLING_RELATIVE_OFFSET + head * 16 + lane]) *
                                     float(rel_scale[head * 16 + lane]) : 0.0f;
      const float rn = simd_sum(rv * rv);
      if (lane < 16) {
        const bfloat rc = bfloat(rv * min(8.0f / (sqrt(rn) + 1e-6f), 1.0f));
        relative[row * 384 + head * 16 + lane] = bfloat(float(rc) * tau);
      }
    }
    return;
  }
  const bfloat value = bfloat(y * rsqrt(v_norm / 128.0f + 1e-6f));
  splash_store_kv_slot<8, 128, CacheElement, false>(
      splash_store_page(page_table, sequence, group.y), sequence, params.kv, maxima,
      head, group.y, clipped, value, dimension, lane, sg);
}

#define INKLING_PREPARE(Name, CacheElement) \
kernel void Name(device const bfloat *q [[buffer(0)]], \
                 device const bfloat *k [[buffer(1)]], device const bfloat *v [[buffer(2)]], \
                 device const bfloat *q_scale [[buffer(3)]], device const bfloat *rel_scale [[buffer(4)]], \
                 device bfloat *queries [[buffer(5)]], device bfloat *relative [[buffer(6)]], \
                 device const SplashKvPage *p0 [[buffer(7)]], device const SplashKvPage *p1 [[buffer(8)]], \
                 device const SplashKvPage *p2 [[buffer(9)]], device const SplashKvPage *p3 [[buffer(10)]], \
                 device const SplashKvPage *p4 [[buffer(11)]], device const SplashKvPage *p5 [[buffer(12)]], \
                 device const SplashKvPage *p6 [[buffer(13)]], device const SplashKvPage *p7 [[buffer(14)]], \
                 device const SplashLaneRows *layout [[buffer(15)]], device const SplashTreeRow *tree [[buffer(16)]], \
                 constant SplashAttentionParams &params [[buffer(17)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint dim [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float sums[8]; \
  SPLASH_STORE_MAXIMA_##CacheElement(128) \
  inkling_prepare_phase<CacheElement>(q, k, v, q_scale, rel_scale, queries, relative, \
      SPLASH_LANE_BINDING(group.z, p0, p1, p2, p3, p4, p5, p6, p7), layout, tree, params, group, dim, lane, sg, sums, maxima); \
}
INKLING_PREPARE(attention_q8_inkling_prepare, int8_t)
INKLING_PREPARE(attention_bf16_inkling_prepare, bfloat)
#undef INKLING_PREPARE
