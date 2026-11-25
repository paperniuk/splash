#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The draft attention the kernels are compiled for, the only one
// ops::DraftAttention accepts: 32 query heads over 8 KV heads of 128
// dimensions, so a q|k|v row holds 6144 values and an attention row 4096.
// The attention core runs a KV head's four query heads over the eight query
// rows together, as 32 attention rows.
#define SPLASH_DRAFT_QUERY_HEADS 32u
#define SPLASH_DRAFT_KV_HEADS 8u
#define SPLASH_DRAFT_HEAD_DIMENSION 128u
#define SPLASH_DRAFT_ATTENTION_WIDTH                                           \
  (SPLASH_DRAFT_QUERY_HEADS * SPLASH_DRAFT_HEAD_DIMENSION)
#define SPLASH_DRAFT_QKV_WIDTH                                                 \
  (SPLASH_DRAFT_ATTENTION_WIDTH +                                              \
   2u * SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_HEAD_DIMENSION)
#define SPLASH_DRAFT_ATTENTION_ROWS                                            \
  (SPLASH_DRAFT_QUERY_HEADS / SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_QUERY_ROWS)

// The dynamic convolutions of a draft layer (draft_conv), each run in two
// stages, before its attention or MLP and after it with the residual: a
// stage weighs a channel's row and the row before it, its two taps, each by
// a base weight plus the dynamic weight of the channel's group of 16
// channels. The draft projects every row's dynamic weights, for each stage,
// tap and group of a `hidden`-wide layer.
#define SPLASH_DRAFT_CONVOLUTION_STAGES 2u
#define SPLASH_DRAFT_CONVOLUTION_TAPS 2u
#define SPLASH_DRAFT_CONVOLUTION_GROUP 16u
inline constexpr uint32_t draft_dynamic_width(uint32_t hidden) {
  return SPLASH_DRAFT_CONVOLUTION_STAGES * SPLASH_DRAFT_CONVOLUTION_TAPS *
         (hidden / SPLASH_DRAFT_CONVOLUTION_GROUP);
}

// The batched draft kernels' grids cover exactly the dispatch's lanes:
// per-lane arrays hold those lanes, and entries past them are zero and unread.

// A convolution stage (finish: the second, with the residual) of a draft
// `hidden` wide, a whole number of convolution groups.
struct DraftConvBatchParams {
  uint32_t finish;
  uint32_t hidden;
};

static_assert(sizeof(DraftConvBatchParams) == 8,
              "Draft convolution parameters are 8 bytes on both sides");

// The attention core's split count (SPLASH_DRAFT_ATTENTION_SPLITS) and its
// rings' slots per KV head (SPLASH_DRAFT_SLIDING_WINDOW) are compiled in.
// value_stride, the values ring's stride between head dimensions, is the
// window too (the op passes nothing else) but stays a run-time value: with it
// compiled into the value tiles, the split kernel returns wrong rows at
// random, and reports no error, on an M5 Max under full
// MTL_SHADER_VALIDATION; without validation its output is bitwise that of
// the run-time stride.
struct DraftAttentionBatchParams {
  uint32_t value_stride;
  uint32_t lanes;
  uint32_t cache_length[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftAttentionBatchParams) == 8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft attention parameters are the same size on both sides");

struct DraftContextParams {
  uint32_t tokens;
  uint32_t start_position;
};

static_assert(sizeof(DraftContextParams) == 8,
              "Draft context prefill parameters are 8 bytes on both sides");

// The context commit of each lane's retained rows (the mask acceptance
// writes, AcceptBatchParams) of its lane rows (SplashLaneRows): the context
// rows among the block's target rows, the path's i-th at position
// start_position[l] + i, rotated by the lane's RoPE row i (a lane's RoPE rows
// are its verify block's positions, SPLASH_TARGET_VERIFY_ROWS a lane).
struct DraftContextBatchParams {
  uint32_t start_position[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextBatchParams) == 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Draft context commit parameters are the same size on both sides");

struct DSparkFusionParams {
  uint32_t width;
  uint32_t consumer;
};

static_assert(sizeof(DSparkFusionParams) == 8, "DSpark fusion parameters are 8 bytes on both sides");

struct DSparkHeadParams {
  uint32_t heads;
  uint32_t start_position;
};

struct DSparkStoreParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t base_position;
  uint32_t capacity;
};

static_assert(sizeof(DSparkHeadParams) == 8 && sizeof(DSparkStoreParams) == 16,
              "DSpark attention parameters match on both sides");

// DSpark's global layer keeps its context K/V (rows of 8 KV heads x 128
// values, [row][head][dimension] for both) in fixed-size segments, never a
// ring: the row of position p lives at row (p - base) % segment_tokens of
// segment (p - base) / segment_tokens, each segment a buffer of its own so
// that no one buffer holds the whole context. The kernels reach the segments
// through a table of their GPU addresses (metal::AddressTable), in the
// constant address space: the `segments` key segments' addresses first, then
// the `segments` value segments'. setBytes passes at most 4 KiB, so a table
// holds at most SPLASH_DRAFT_MAX_GLOBAL_SEGMENTS segments of each. A device
// whose maxBufferLength is below a segment's bytes takes smaller segments;
// the last segment may hold fewer rows than segment_tokens.
#define SPLASH_DRAFT_GLOBAL_SEGMENT_TOKENS 65536u
#define SPLASH_DRAFT_MAX_GLOBAL_SEGMENTS 256u

// A block's attention over its context rows [context_first, context_rows) and its own 8 rows, split over `splits`
// ranges of those keys, each a threadgroup per KV head (dspark_block_attention_split*), whose partials
// (DSPARK_ATTENTION_PARTIAL floats: 128 outputs, maximum, denominator) of the head's 24 queries (8 rows x 3 heads)
// dspark_block_attention_reduce joins. The context is a ring of context_capacity rows (0: rows where they are) or
// the global layer's segments (segment_tokens rows each, `segments` of each plane).
#define DSPARK_ATTENTION_MAX_SPLITS 64u
#define DSPARK_ATTENTION_PARTIAL 130u
struct DSparkSplitAttentionParams {
  uint32_t context_rows;
  uint32_t context_first;
  uint32_t context_capacity;
  uint32_t segment_tokens;
  uint32_t segments;
  uint32_t splits;
};

// A store of `rows` rows to the rows of positions start_position + row of the
// segments, counted from base_position; the commit store writes only as many
// rows as its retained mask counts.
struct DSparkSegmentedStoreParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t base_position;
  uint32_t segment_tokens;
  uint32_t segments;
};

static_assert(sizeof(DSparkSplitAttentionParams) == 24 && sizeof(DSparkSegmentedStoreParams) == 20,
              "DSpark segmented parameters match on both sides");

// A context window holds, per ring slot, the draft's context row, from which
// every layer's ring keys and values are computed: 4-bit codes over groups of
// SPLASH_DRAFT_CONTEXT_GROUP values, all slots' codes first, then each
// group's fp16 scale and minimum. Its kernels run one simdgroup per row and
// group, SPLASH_DRAFT_CONTEXT_SIMDGROUPS to a threadgroup.
#define SPLASH_DRAFT_CONTEXT_GROUP 64u
#define SPLASH_DRAFT_CONTEXT_SIMDGROUPS 8u
inline constexpr uint64_t draft_context_codes_bytes(uint32_t width) {
  return uint64_t(SPLASH_DRAFT_SLIDING_WINDOW) * (width / 2);
}
inline constexpr uint64_t draft_context_window_bytes(uint32_t width) {
  return draft_context_codes_bytes(width) +
         uint64_t(SPLASH_DRAFT_SLIDING_WINDOW) * (width / SPLASH_DRAFT_CONTEXT_GROUP) * 4;
}

struct DraftContextWindowParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t width;
};

static_assert(sizeof(DraftContextWindowParams) == 12,
              "Draft context window parameters are 12 bytes on both sides");
