#pragma once

#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// The CPU's share of a prompt chunk's relative attention: the first rows of a
// full layer, over its whole history, on the matrix units (AMX) of Apple7/8
// CPUs, beside the GPU that runs the rest of the chunk's rows
// (RelativeAttentionPlan skipRows). It computes what the direct kernel
// computes for a row (kernels/shared/relative_attention.metal): the scores of
// every key up to the row, each key's INT8 code times the query and its
// scale, then the row's bias at the key's backward distance below
// SPLASH_RELATIVE_ATTENTION_EXTENT; the softmax with the row's sink; the
// values' INT8 codes weighted by the probabilities rounded to bf16 after
// their scale; the quotient and the gate rounded to bf16, where the GPU
// rounds. The matrix units take fp16 and add in f32: the queries are bf16, so
// their codes and the keys' and values' are exact in fp16, and a dot product
// is exact up to the order of its f32 sum. A row's bits depend on its KV head
// and its chunk alone: the task grid, the order of the key blocks and of
// every sum are fixed, whatever the thread count (the same bits at 1 thread
// and at 10).
class CpuRelativeAttention final {
public:
  static constexpr uint32_t kQueryHeads = 24;
  static constexpr uint32_t kKvHeads = 8;
  static constexpr uint32_t kHeadDimension = 128;
  // A task is a KV head's block of this many rows (three fused rows each): a
  // block's packed keys and values serve its six matrix tiles, and 64 rows
  // keep the tasks of a chunk of 800 rows to about a tenth of a thread's work.
  static constexpr uint32_t kTaskRows = 64;

  // A chunk's first `rows` rows, as the prepare staged them
  // (RelativeAttentionStaging) and the GPU's pages hold the keys and values
  // they attend: rows of one sequence whose history `committed` tokens
  // precede, over the layer's pages as `pageTable`'s entries reach them.
  struct Rows final {
    // Staged queries (bf16), relative states and sinks (f32), KV head h's row
    // r's fused row (3 r + g of query head 3 h + g) at (h * stride + r) * 3 + g.
    const uint16_t *queries = nullptr;
    const float *states = nullptr;
    const float *sinks = nullptr;
    uint32_t stride = 0;
    // The chunk's projection rows (bf16), whose gate block each output row's
    // gates multiply: `projectionStride` values a row, the gates from
    // `gateColumn`.
    const uint16_t *projection = nullptr;
    uint64_t projectionStride = 0;
    uint32_t gateColumn = 0;
    // The layer's bias table, f32 [distance][state].
    const float *bias = nullptr;
    // The layer's pages: the table's entries (kv::PageStorage::writeEntries),
    // its placement in every extent, the extents' host addresses and the
    // layout's bytes of a page's keys and of their scales.
    const SplashKvPage *pageTable = nullptr;
    uint32_t pageTableEntries = 0;
    SplashKvLayer layer{};
    std::span<const kv::HostExtent> extents;
    uint64_t dataBytes = 0;
    uint64_t scaleBytes = 0;
    uint32_t committed = 0;
    uint32_t rows = 0;
    // The output: `rows` rows of the 24 heads' bf16 values, gated.
    uint16_t *hidden = nullptr;
  };

  // Whether this CPU has the matrix units the kernel drives: Apple M1 and M2
  // (M3 has them but no GPU path that benefits; M4 and later replace them by
  // SME, which these instructions are not).
  [[nodiscard]] static bool available() noexcept;

  // The switches of this build's measurement stand, read once from the
  // environment: SPLASH_CPU_ATTN_SHARE (the share of a chunk's work, 0 to 0.9,
  // 0 or unset off), SPLASH_CPU_ATTN_THREADS (default 8), SPLASH_CPU_ATTN_SHADOW=1
  // (the GPU computes every row, the CPU's rows are compared with its and
  // logged, the model goes on with the GPU's) and SPLASH_CPU_ATTN_TIMING=1
  // (logs each host step's pickup, work and resume).
  struct Settings final {
    double share = 0.0;
    uint32_t threads = 8;
    bool shadow = false;
    bool timing = false;
    [[nodiscard]] bool enabled() const noexcept { return share > 0.0 || shadow; }
  };
  [[nodiscard]] static const Settings &settings();

  // The first rows of a chunk of `rows` rows after `committed` tokens that the
  // CPU takes at `share`: the count whose attention pairs (keys the rows see,
  // R (c + R / 2)) are that share of the chunk's, rounded down to the GPU's
  // direct tile (16 rows) so that its rows keep their bits. None (0) before
  // kMinimumHistory tokens, where the rows' attention is a few percent of a
  // chunk, for a chunk of under kMinimumRows rows, and when it rounds to none.
  static constexpr uint32_t kMinimumHistory = 8192;
  static constexpr uint32_t kMinimumRows = 512;
  [[nodiscard]] static uint32_t rowsFor(double share, uint32_t committed, uint32_t rows) noexcept;

  // How far two chunks of bf16 outputs lie apart: the largest difference in
  // bf16 steps, the relative L2 norm of it, and the share of elements two or
  // more steps apart.
  struct Difference final {
    uint32_t largestSteps = 0;
    double relative = 0.0;
    double farShare = 0.0;
  };
  [[nodiscard]] static Difference compare(const uint16_t *expected, const uint16_t *actual, uint64_t count) noexcept;

  // Computes `rows` on `threads` threads (at least one) and returns when all
  // are written. std::invalid_argument for inputs that do not hold the rows'
  // pages, with nothing written; the output is otherwise complete.
  static void run(const Rows &rows, uint32_t threads);
};

} // namespace splash::ops
