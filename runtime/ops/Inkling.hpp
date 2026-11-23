#pragma once

// Inkling 2's own blocks (kernels/shared/inkling.metal): the depth attention
// over a residual's blocks, a loop pass's message gates, the norms that feed
// its sublayers, its short causal convolutions and the head's softcap.
// model/InklingTarget.cpp composes them with ops::Linear, ops::MoE and
// ops::PagedAttention.

#include "metal/CommandGraph.hpp"
#include "metal/abi/VerifyTree.h"
#include "ops/LaneRows.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A view of `rows` rows of `width` values in a buffer whose rows are `stride`
// values apart (stride >= width), from value `offset` of the buffer.
struct InklingRows final {
  metal::MetalBuffer buffer;
  uint32_t width = 0;
  uint32_t stride = 0;
  uint32_t offset = 0;
};

// A message gate's weights (controllers.{attn,k,v,mlp}), bf16 as stored:
// down [rank][2 * hidden], forget and gain [hidden][rank].
struct InklingGateWeights final {
  metal::MetalBuffer down;
  metal::MetalBuffer forget;
  metal::MetalBuffer gain;
};

// A short convolution of `channels` channels over `taps` taps (4). Static:
// `taps` bf16 [channels][taps], tap taps-1 the current input's. Dynamic:
// `staticTaps` bf16 [taps][channels], lag 0 the current input's, plus each
// row's f32 [heads][taps] from its weight projection, heads of channels /
// heads channels each.
struct InklingConvolution final {
  uint32_t channels = 0;
  uint32_t taps = 4;
  metal::MetalBuffer weights;
  bool dynamic = false;
  uint32_t heads = 0;
};

// A convolution's residual add (Inkling::addConvolution): bf16 or f32 addend
// rows, none for the convolution alone, and whether the output rows are f32.
struct InklingResidual final {
  InklingRows addend;
  bool addendFloat = false;
  bool outputFloat = false;
};

// Batched parts' steps (Inkling::addMessageGates, addNorms), in values.
struct InklingPartSteps final {
  uint32_t x = 0;
  uint32_t message = 0;
  uint32_t low = 0;
  uint32_t output = 0;
};

struct Inkling final {
  // Adds f32(a) and f32(b) over the rows, or converts a when b is empty.
  // Each flag selects f32 rather than bf16; output may alias a only when
  // their element types match.
  static void addSum(metal::CommandGraph &graph, InklingRows a, bool aFloat, InklingRows b, bool bFloat,
                     InklingRows output, bool outputFloat, uint32_t width, uint32_t rows);
  // Depth attention of each of `rows` rows over its states: `states` bf16
  // states (each an InklingRows of hidden values, possibly none), then the
  // partial `partial` (bf16, or f32 when partialFloat). Keys are each state's
  // RMS norm without weight (eps 1e-6); each of `queries` (f32
  // [queryCount][hidden]) scores them by key . query * hidden^-0.5; the output
  // is the softmax-weighted sum of the keys, f32, row r's query q at row r of
  // `output` (an InklingRows of queryCount * hidden values a row). Two
  // dispatches over 256-channel chunks of the rows, so that one row spreads
  // over cores: each chunk's squares and dots into `sums`
  // (depthSumsBytes), then each chunk's weighted sum from all of them.
  static void addDepthMix(metal::CommandGraph &graph, std::span<const InklingRows> states, InklingRows partial,
                          bool partialFloat, metal::MetalBuffer queries, uint32_t queryCount, InklingRows output,
                          metal::MetalBuffer sums, uint32_t hidden, uint32_t rows);
  [[nodiscard]] static uint64_t depthSumsBytes(uint32_t hidden, uint32_t rows);
  // A message gate of f32 rows `x` and their f32 messages `message` into f32
  // `output` (may alias x): pair = bf16([rms(x), rms(m)]),
  // h = f32(bf16(pair . down^T)), out = (1 - tanh(h . forget^T)) * x + tanh(h . gain^T) * m.
  // `low` holds h, f32 rows of `rank` values: two dispatches, the first
  // spreading a row's rank outputs over threadgroups so that one row's gate
  // reads its weights in parallel.
  static void addMessageGate(metal::CommandGraph &graph, InklingRows x, InklingRows message,
                             const InklingGateWeights &weights, uint32_t rank, InklingRows low, InklingRows output,
                             uint32_t hidden, uint32_t rows);
  // Up to three gates of their own weights in one pair of dispatches: gate p
  // reads x, message, low and output each `steps` values past gate p - 1's
  // (InklingPartSteps), as a layer's q, k and v inputs sit in its rows.
  static void addMessageGates(metal::CommandGraph &graph, InklingRows x, InklingRows message,
                              std::span<const InklingGateWeights> weights, uint32_t rank, InklingRows low,
                              InklingRows output, const InklingPartSteps &steps, uint32_t hidden, uint32_t rows);
  // bf16(RMSNorm(f32 rows `input`, `norm`)) into bf16 rows `output`, eps 1e-6.
  static void addNorm(metal::CommandGraph &graph, InklingRows input, const NormWeights &norm, InklingRows output,
                      uint32_t hidden, uint32_t rows);
  // Up to three norms of their own weights (of one type) in one dispatch, norm
  // p's input and output `steps.x` and `steps.output` values past norm p - 1's.
  static void addNorms(metal::CommandGraph &graph, InklingRows input, std::span<const NormWeights> norms,
                       InklingRows output, const InklingPartSteps &steps, uint32_t hidden, uint32_t rows);
  // A short convolution of a command's sequences' bf16 input rows `input`
  // (one sequence a SplashLaneRows of `layout`, as `rows` lays them out) into
  // bf16 rows `output`: out = bf16((conv + x) / sqrt(taps + 1)) static,
  // / sqrt(2 * taps + 1) dynamic, conv in f32; `dynamicTaps` the rows' f32
  // [heads][taps] for a dynamic one. Sequence s's taps before its first row
  // read history historyIn[s] (bf16 [taps - 1][channels] rows `historyStride`
  // values apart from value `historyOffset`, oldest first); with historyOut,
  // its history after its rows goes to historyOut[s] in the same form. With
  // `tree` (a SplashTreeRow a row, blocks of at most 32 rows), a row's earlier
  // inputs are its ancestors' along its path, then the history; its history
  // is then written by addConvolutionCommit, from each row's input, which
  // `tape` (bf16 rows, when given) takes as the convolution reads it. With
  // `residual`, output is f32(bf16(conv)) + f32(residual.addend), in f32 or
  // bf16 rows: a layer's residual add in the convolution's dispatch.
  static void addConvolution(metal::CommandGraph &graph, InklingRows input,
                             std::span<const metal::MetalBuffer> historyIn,
                             std::span<const metal::MetalBuffer> historyOut, uint32_t historyOffset,
                             uint32_t historyStride, const InklingConvolution &convolution,
                             InklingRows dynamicTaps, InklingRows output, metal::MetalBuffer layout,
                             const LayoutRows &rows, metal::MetalBuffer tree = {}, InklingRows tape = {},
                             const InklingResidual &residual = {});
  // The history after the first `rows` rows of one sequence's prefill (a
  // state captured inside its command), of `channels` channels of rows
  // `input`, from historyIn into historyOut (offset and stride as above).
  static void addConvolutionHistory(metal::CommandGraph &graph, InklingRows input, metal::MetalBuffer historyIn,
                                    metal::MetalBuffer historyOut, uint32_t historyOffset, uint32_t historyStride,
                                    uint32_t channels, uint32_t taps, uint32_t rows);
  // Each lane's history after its accepted path (`paths`, a row mask a lane,
  // metal/abi/VerifyTree.h) of its verify rows' inputs: `inputs` holds
  // `blocks` blocks of the step's rows, block b at value b * blockStride of
  // every row (so one dispatch commits every visit's convolutions), each block
  // `channels` channels whose history sits at value b * historyBlockStride of
  // a lane's history, rows `historyStride` values apart. With `pathRows`,
  // the history after only the path's first `pathRows` rows (or all of a
  // shorter path): a state captured at a page boundary inside the path.
  static void addConvolutionCommit(metal::CommandGraph &graph, InklingRows inputs, uint32_t blocks,
                                   uint32_t blockStride, uint32_t channels,
                                   std::span<const metal::MetalBuffer> historyIn,
                                   std::span<const metal::MetalBuffer> historyOut, uint32_t historyBlockStride,
                                   uint32_t historyStride, uint32_t taps, metal::MetalBuffer paths,
                                   metal::MetalBuffer layout, const LayoutRows &rows, uint32_t pathRows = 0);
  // The head's logits in place: tanh(x / softcap) * softcap for the first
  // `vocabulary` of each row's `width` f32 logits, -1e30 for the rest.
  static void addLogitCap(metal::CommandGraph &graph, metal::MetalBuffer logits, float softcap, uint32_t vocabulary,
                          uint32_t width, uint32_t rows);
};

} // namespace splash::ops
