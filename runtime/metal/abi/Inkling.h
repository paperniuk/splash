#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The channels a row kernel holds in registers, 256 threads at a time, past which it reads them one at a time
// (kernels/shared/inkling.metal).
#define INKLING_ROW_SLOTS 9u
struct InklingRowParams { uint32_t stride, offset; };
struct InklingParams {
  InklingRowParams source[6], output, dynamicRows;
  uint32_t hidden, rows, count, queries, rank, partialFloat;
  uint32_t taps, heads, dynamic, sequences, tree, writeHistory;
  uint32_t historyOffset, historyStride, blocks, blockStride, historyBlockStride;
  // A convolution's residual: bf16(conv) plus addend rows (source[2], f32
  // when addendFloat), into f32 output rows when outputFloat.
  uint32_t addend, addendFloat, outputFloat;
  // Batched parts (grid z): part z's rows of x/input, message, low and output
  // at their offsets plus z times these steps.
  uint32_t partSteps[4];
  float softcap;
};
