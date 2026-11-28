#pragma once

// Parameters of the message gate kernels (kernels/shared/message_gate.metal,
// ops/MessageGate.cpp): a row x takes a message m by gates of a low-rank
// projection of both, g = down(norm(x), norm(m)), as
// (1 - tanh(forget g)) x + tanh(gain g) m, in fp32; a weighted RMS norm then
// writes the projections' bf16 input. The kernels below read the gates'
// weights themselves; the reference path they are checked against, which
// rounds at the same points, runs in the tests
// (dev/tests/engine/message_gate_reference.h).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define MESSAGE_GATE_THREADS 256u

// The fused gates (ops::MessageGate, a decode step's rows): three kernels over
// up to MESSAGE_GATE_MAXIMUM_GATES gates of the same rows (a layer's q, k and
// v inputs: grid z), the last two over groups of MESSAGE_GATE_GROUP_ROWS rows,
// which read a gate's weights, F32 or BF16 rows as stored, once a group.
// message_gate_normalize normalizes and concatenates each row of a gate once,
// a threadgroup a row, into the bf16 rows that each of the down projection's
// threadgroups of the row's group reads, where each had normalized its
// group's rows itself: an eight-row decode step's gates took 2.13 ms where
// they took 2.64 (Inkling-2-Mini on a 16-core M5 Pro, separate dispatches),
// the step 0.47 ms less. message_gate_down writes MESSAGE_GATE_DOWN_OUTPUTS
// of the group's rank values a threadgroup (32 threadgroups a gate for a rank
// of 64; a simdgroup a value, 8 threadgroups, took 2.0 ms of a decode step's
// 96 gates against 1.25 ms on an M5 Pro, F32 weights a gate a dispatch);
// message_gate_apply takes MESSAGE_GATE_THREADS of a row's values a
// threadgroup (a slice), a thread each: forget, gain and the gated row, whose
// squares the slices of a group add, the last to arrive in slice order, which
// then writes the group's normalized rows (slices of 128 and 64 values, two
// and four threads a value, ran no faster). A rank is at most
// MESSAGE_GATE_MAXIMUM_RANK, a multiple of four.
#define MESSAGE_GATE_GROUP_ROWS 8u
#define MESSAGE_GATE_DOWN_OUTPUTS 2u
#define MESSAGE_GATE_MAXIMUM_RANK 64u
#define MESSAGE_GATE_MAXIMUM_GATES 3u
// The narrow gates (message_gate.metal), Apple7/8's: a call of at most MESSAGE_GATE_NARROW_ROWS rows runs
// message_gate_narrow_down (each threadgroup normalizing its group's rows itself, MESSAGE_GATE_NARROW_OUTPUTS rank
// values a threadgroup), message_gate_narrow_up (MESSAGE_GATE_NARROW_UP_THREADS of a row's values a threadgroup) and
// message_gate_norm, where the three kernels above took a one-row Inkling-2-Mini decode step 0.69 ms more on an M1
// Max (35.46 against 34.77 ms) and a four-row one 0.24 ms more; at eight rows the narrow gates ran slower. Apple9+
// keeps the three kernels (ops::MessageGate::add's `narrow`), which ran a four-row call there faster (2.41 against
// 2.68 ms a step on an M3 Ultra). A thread of MESSAGE_GATE_THREADS holds MESSAGE_GATE_ROW_SLOTS of a row's values in
// registers (Inkling-2-Mini's 2,304).
#define MESSAGE_GATE_NARROW_ROWS 4u
#define MESSAGE_GATE_NARROW_OUTPUTS 4u
#define MESSAGE_GATE_NARROW_UP_THREADS 64u
#define MESSAGE_GATE_ROW_SLOTS 9u

// The fused gates of more rows (ops::MessageGate, a prompt chunk's): one
// kernel, message_gate_rows, of a gate's MESSAGE_GATE_TILE_ROWS rows a
// threadgroup of MESSAGE_GATE_TILE_THREADS threads (grid z the gates, as
// above), which reads the tile's rows and the gate's weights itself and keeps
// its rank values in threadgroup memory: the tile's inverse RMS, its rank
// values, its gated rows, which it stores to the gate's scratch and reads
// back, and their norm, the projections on simdgroup float MMA, each
// simdgroup 8 columns of a 64-column block for the tile's rows, so that the
// tile reads each weight once, the down projection's normalized inputs staged
// in threadgroup memory. Its rows of whole 64-value steps, its rank a
// multiple of 8. A 2,583-token Inkling-2-Mini prefill's gates took 102 ms on
// it on a 16-core M5 Pro, against 141 ms as the float projections' four
// dispatches a gate and 128 ms with each simdgroup reading its own operands
// from device memory (2 x 4 tiles of 16 x 16 a threadgroup).
#define MESSAGE_GATE_TILE_ROWS 32u
#define MESSAGE_GATE_TILE_THREADS 256u

// The fused gates of a dispatch: `gates` gates of `rows` rows of `hidden`
// values, gate g's row x at column x_columns[g] of rows of x_stride values and
// its message at message_columns[g] of rows of message_stride values, both
// fp32, in whole float4s; `rank` values a row between each gate's projections.
struct MessageGateFusedParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t x_stride;
  uint32_t message_stride;
  uint32_t rank;
  uint32_t gates;
  uint32_t x_columns[MESSAGE_GATE_MAXIMUM_GATES];
  uint32_t message_columns[MESSAGE_GATE_MAXIMUM_GATES];
};

static_assert(sizeof(MessageGateFusedParams) == 48, "fused message gate parameters are 48 bytes on both sides");
