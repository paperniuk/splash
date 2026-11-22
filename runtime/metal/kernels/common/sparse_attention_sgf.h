#pragma once

#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/sparse_attention_tile.h"

// Apple7/8 (M1/M2) register-matrix sparse attention (the Register tile of
// sparse_attention_tile.h): the MPP tile's lists, partition, masks, partials
// and statistics, on q8sgf's fragments. Each of the tile's G simdgroups owns
// eight fused rows and keeps their scores, softmax and O^T in registers, as
// q8sgf::tile does over a page. A step takes kSparseSgfBlocks list entries,
// 32 tokens: the entries' pages and places, and the step's values transposed
// to [dimension][token] (the dense page's value layout, which PV reads as
// q8sgf::tile does), go through threadgroup memory; each lane reads its keys
// and scales from the cache itself. Tokens of a slot past the list's end, of
// a block a row does not attend, past a row's causal end or, in a tree, off
// the row's path never contribute.
namespace q8sgf {

constant constexpr uint kSparseSgfBlocks = 32 / SPLASH_ATTENTION_BLOCK_TOKENS;
static_assert(SplashSparsePartitionBlocks % kSparseSgfBlocks == 0,
              "a partition unit is whole register steps");

template <typename CacheElement> struct SparseScratch {
  uint entries[kSparseSgfBlocks];
  SplashKvPage pages[kSparseSgfBlocks];
  uint places[kSparseSgfBlocks];
  alignas(16) CacheElement values[SPLASH_KV_HEAD_DIMENSION * 32];
};

// Copies the step's values, token-major in the cache, into the stage as
// [dimension][token]; tokens of a slot past the list's end are zero.
template <uint KVHeads, typename CacheElement>
inline void sparse_gather_values(threadgroup SparseScratch<CacheElement> &scratch,
                                 const SplashKvAddressing<KVHeads, SPLASH_KV_HEAD_DIMENSION, CacheElement> addressing,
                                 uint thread_index, uint threads) {
  constexpr uint D = SPLASH_KV_HEAD_DIMENSION;
  constexpr uint PerChunk = 16 / sizeof(CacheElement);
  constexpr uint RowChunks = D / PerChunk;
  for (uint chunk = thread_index; chunk < 32 * RowChunks; chunk += threads) {
    const uint token = chunk / RowChunks, first = chunk % RowChunks * PerChunk;
    const uint slot = token / SPLASH_ATTENTION_BLOCK_TOKENS;
    uint4 data = 0;
    if (scratch.entries[slot] != SplashSparseNoEntry) {
      const uint page_token =
          scratch.places[slot] * SPLASH_ATTENTION_BLOCK_TOKENS + token % SPLASH_ATTENTION_BLOCK_TOKENS;
      device const CacheElement *row = addressing.page(scratch.pages[slot]).values + ulong(page_token) * D + first;
      data = *reinterpret_cast<device const uint4 *>(row);
    }
    thread const CacheElement *element = reinterpret_cast<thread const CacheElement *>(&data);
#pragma unroll
    for (uint e = 0; e < PerChunk; ++e)
      scratch.values[(first + e) * 32 + token] = element[e];
  }
}

// One tile of eight query rows x G heads over its split's run of its block
// list (`list`, `count` entries), or with a null list over its history's
// blocks, as splash_sparse_attention_tile, a tree's rows (`tree`) each
// masked to its path's rows of its block and, through a list, to those its
// row mask holds (`attended`). Takes 32 x G threads; every thread reaches
// each barrier.
template <uint KVHeads, uint QueryHeadsPerKVHead, typename CacheElement, bool Located = false>
__attribute__((always_inline)) inline void sparse_tile(
    device const bfloat *tile_queries, device const SplashKvPage *page_table, SplashKvLayer kv, uint kv_head,
    uint first_position, uint active_rows, device const uint *list, uint count, device const uint *locations,
    device const SplashKvPage *location_pages, uint splits, uint split, device float *partials,
    device float *statistics, ulong slot, threadgroup SparseScratch<CacheElement> &scratch,
    SplashAttentionTree tree, device const uint *attended, uint sg, uint lane) {
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr uint G = QueryHeadsPerKVHead;
  constexpr uint M = 8 * G;
  constexpr uint N = 32;
  constexpr uint D = SPLASH_KV_HEAD_DIMENSION;
  constexpr uint BlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  constexpr uint PageBlocks = SplashKvPageTokens / BlockTokens;
  constexpr uint AllRows = (1u << 8) - 1u;
  static_assert(D == 256, "fragment maps assume D = 256");
  static_assert(SPLASH_ATTENTION_TILE_ROWS == 8 && SPLASH_ATTENTION_BLOCK_ROW_BITS == 8,
                "one simdgroup per G fused rows of an eight-row tile");
  const uint per_split = splash_sparse_split_entries(count, splits);
  const uint begin = split * per_split;
  if (begin >= count)
    return;
  const uint end = min(count, begin + per_split);
  const uint history = splash_sparse_history_blocks(first_position, active_rows);
  const uint thread_index = sg * 32 + lane;
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  // This lane's fused rows are row and row + 1, the columns of its fragments;
  // a tile row past the active ones mirrors the last active row.
  const uint row = sg * 8 + fn;
  const uint2 tile_row(min(row / G, active_rows - 1), min((row + 1) / G, active_rows - 1));
  const uint2 causal_end = first_position + tile_row + 1;
  // Of a tree's block, the row's path, through a list only its rows in
  // blocks the row attends: every other token sees all of it.
  const uint2 anchor = tree.rows ? uint2(tree.anchor) : causal_end;
  uint2 ancestors(0);
  if (tree.rows)
    ancestors = uint2(tree.rows[tile_row.x].ancestors & (attended ? attended[tile_row.x] : ~0u),
                      tree.rows[tile_row.y].ancestors & (attended ? attended[tile_row.y] : ~0u));

  uint queries[4][8];
  {
    device const ushort *q0 = reinterpret_cast<device const ushort *>(tile_queries) + row * D;
    device const ushort *q1 = q0 + D;
#pragma unroll
    for (uint b = 0; b < 4; ++b)
#pragma unroll
      for (uint t = 0; t < 8; ++t) {
        const uint d = 64 * b + block_dimension(fm, t);
        queries[b][t] = uint(q0[d]) | (uint(q1[d]) << 16);
      }
  }
  float2 output[32];
#pragma unroll
  for (uint d = 0; d < 32; ++d)
    output[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);
  const SplashKvAddressing<KVHeads, D, CacheElement> addressing(kv, kv_head);

  for (uint step = begin; step < end; step += kSparseSgfBlocks) {
    if (thread_index < kSparseSgfBlocks) {
      const uint index = step + thread_index;
      uint entry = SplashSparseNoEntry;
      if (index < end)
        entry = list ? list[index] : index << SPLASH_ATTENTION_BLOCK_ROW_BITS | AllRows;
      if ((entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS) >= history)
        entry = SplashSparseNoEntry;
      const uint block = entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS;
      scratch.entries[thread_index] = entry;
      if constexpr (Located) {
        const uint location = entry == SplashSparseNoEntry ? SPLASH_ATTENTION_PAGED : locations[index];
        const bool paged = location == SPLASH_ATTENTION_PAGED;
        scratch.pages[thread_index] = entry == SplashSparseNoEntry ? SplashKvPage(0)
                                      : paged ? page_table[block / PageBlocks]
                                              : location_pages[location >> SPLASH_ATTENTION_LOCATION_BLOCK_BITS];
        scratch.places[thread_index] = (paged ? block : location) % PageBlocks;
      } else {
        scratch.pages[thread_index] = entry == SplashSparseNoEntry ? SplashKvPage(0) : page_table[block / PageBlocks];
        scratch.places[thread_index] = block % PageBlocks;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sparse_gather_values<KVHeads, CacheElement>(scratch, addressing, thread_index, 32 * G);

    // This lane's four score tokens page_token(fm, s): which rows attend
    // each, its key rows, and for INT8 its scales.
    bool2 allowed[4];
    device const CacheElement *key_rows[4];
    float key_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float value_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      const uint token = page_token(fm, s), slot_index = token / BlockTokens;
      const uint entry = scratch.entries[slot_index];
      const bool valid = entry != SplashSparseNoEntry;
      const uint position = (entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS) * BlockTokens + token % BlockTokens;
      allowed[s] = bool2(valid && ((entry >> tile_row.x) & 1u) && position < causal_end.x &&
                             (position < anchor.x || ((ancestors.x >> min(position - anchor.x, 31u)) & 1u)),
                         valid && ((entry >> tile_row.y) & 1u) && position < causal_end.y &&
                             (position < anchor.y || ((ancestors.y >> min(position - anchor.y, 31u)) & 1u)));
      key_rows[s] = nullptr;
      if (valid) {
        const SplashKvPageTensors<CacheElement> tensors = addressing.page(scratch.pages[slot_index]);
        const uint page_token_index = scratch.places[slot_index] * BlockTokens + token % BlockTokens;
        key_rows[s] = tensors.keys + ulong(page_token_index) * D;
        if constexpr (Quantized) {
          key_scale[s] = tensors.key_scales[page_token_index];
          value_scale[s] = tensors.value_scales[page_token_index];
        }
      }
    }

    float2 scores[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s)
      scores[s] = float2(0.0f);
#pragma unroll
    for (uint b = 0; b < 4; ++b)
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        KeyRun<CacheElement> run{};
        if (key_rows[s]) run.load(key_rows[s] + 64 * b + 8 * fn);
#pragma unroll
        for (uint t = 0; t < 8; ++t)
          mma(scores[s], run.step(t), bfloat_pair(queries[b][t]));
      }

    float2 local_max(-INFINITY);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      float2 score = scores[s];
      if constexpr (Quantized)
        score *= key_scale[s];
      score *= 0.0625f;
      scores[s] = float2(allowed[s].x ? score.x : -INFINITY, allowed[s].y ? score.y : -INFINITY);
      local_max = max(local_max, scores[s]);
    }
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next_max = max(row_max, local_max);
    float2 probability[4];
    float2 local_sum(0.0f);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      probability[s] = float2(allowed[s].x ? fast::exp(scores[s].x - next_max.x) : 0.0f,
                              allowed[s].y ? fast::exp(scores[s].y - next_max.y) : 0.0f);
      local_sum += probability[s];
    }
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 16);
    const float2 scale(next_max.x == -INFINITY || next_max.x == row_max.x ? 1.0f : fast::exp(row_max.x - next_max.x),
                       next_max.y == -INFINITY || next_max.y == row_max.y ? 1.0f : fast::exp(row_max.y - next_max.y));
    row_sum = row_sum * scale + local_sum;
    row_max = next_max;
    if (simd_any(scale.x != 1.0f || scale.y != 1.0f)) {
#pragma unroll
      for (uint d = 0; d < 32; ++d)
        output[d] *= scale;
    }
    float2 weights[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s)
      weights[s] = float2(allowed[s].x ? float(bfloat(probability[s].x * value_scale[s])) : 0.0f,
                          allowed[s].y ? float(bfloat(probability[s].y * value_scale[s])) : 0.0f);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const CacheElement *value_row = scratch.values + fm * N + 4 * fn;
#pragma unroll
    for (uint d = 0; d < 32; ++d) {
      ValueRun<CacheElement> run;
      run.load(value_row + d * 8 * N);
      mma(output[d], run.step(0), weights[0]);
      mma(output[d], run.step(1), weights[1]);
      mma(output[d], run.step(2), weights[2]);
      mma(output[d], run.step(3), weights[3]);
    }
    // PV's reads of the stage and every lane's reads of the entries end
    // before the next step replaces them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device float *target = partials + slot * M * D;
#pragma unroll
  for (uint d = 0; d < 32; ++d) {
    target[row * D + 8 * d + fm] = output[d].x;
    target[(row + 1) * D + 8 * d + fm] = output[d].y;
  }
  if (fm == 0) {
    device float *stat = statistics + (slot * M + row) * 2;
    stat[0] = row_max.x;
    stat[1] = row_sum.x;
    stat[2] = row_max.y;
    stat[3] = row_sum.y;
  }
}
} // namespace q8sgf
