#include "metal/abi/Inkling.h"
#include "metal/abi/VerifyTree.h"
#include "metal/kernels/common/lane_bindings.h"
#include <metal_stdlib>
using namespace metal;
#include "metal/kernels/common/verify_tree.h"

constant uint inkThreads = 256;
// The most channels of a row a threadgroup holds in registers, inkThreads at a time: Inkling's 2304 take 9.
constant uint inkRowSlots = INKLING_ROW_SLOTS;
inline float ink_reduce(float v, threadgroup float *scratch, uint tid) {
  float s = simd_sum(v);
  if ((tid & 31) == 0) scratch[tid / 32] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = 0;
  for (uint i = 0; i < 8; ++i) s += scratch[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return s;
}
kernel void inkling_sum(device const uchar *a [[buffer(0)]], device const uchar *b [[buffer(1)]],
                        device uchar *output [[buffer(2)]], constant InklingParams &p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint c = group.x * inkThreads + tid, row = group.y;
  if (c >= p.hidden) return;
  ulong ai = ulong(p.source[0].offset) + ulong(row) * p.source[0].stride + c;
  ulong bi = ulong(p.source[1].offset) + ulong(row) * p.source[1].stride + c;
  ulong oi = ulong(p.output.offset) + ulong(row) * p.output.stride + c;
  float value = p.partialFloat ? ((device const float *)a)[ai] : float(((device const bfloat *)a)[ai]);
  if (p.count) value += p.dynamic ? ((device const float *)b)[bi] : float(((device const bfloat *)b)[bi]);
  if (p.writeHistory) ((device float *)output)[oi] = value;
  else ((device bfloat *)output)[oi] = bfloat(value);
}
inline float ink_read(device const bfloat *p, InklingRowParams r, uint row, uint c) {
  return float(p[ulong(r.offset) + ulong(row) * r.stride + c]);
}
// Depth attention in two dispatches over 256-channel chunks of each row: the
// chunk's squares of each state (sum i) and dots of each query with it (sum
// 6 + 6q + i), then each chunk's weighted sum of the normed states from all
// chunks' sums. One thread a channel.
constant uint inkDepthChunk = 256, inkDepthSums = 32;
inline float ink_state(device const bfloat *s, constant InklingParams &p, uint i, uint row, uint c) {
  const ulong at = ulong(p.source[i].offset) + ulong(row) * p.source[i].stride + c;
  return (i + 1 == p.count && p.partialFloat) ? ((device const float *)s)[at] : float(s[at]);
}
kernel void inkling_depth_sums(device const bfloat *s0 [[buffer(0)]], device const bfloat *s1 [[buffer(1)]],
 device const bfloat *s2 [[buffer(2)]], device const bfloat *s3 [[buffer(3)]], device const bfloat *s4 [[buffer(4)]], device const bfloat *s5 [[buffer(5)]],
 device const float *query [[buffer(6)]], device float *sums [[buffer(8)]], constant InklingParams &p [[buffer(9)]],
 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[inkDepthChunk / 32][inkDepthSums];
  device const bfloat *states[6] = {s0, s1, s2, s3, s4, s5};
  const uint c = group.x * inkDepthChunk + tid, row = group.y;
  const bool live = c < p.hidden;
  // Every load first, at clamped places (inkling_norm_impl): a load inside the loop of reductions waits in turn.
  const uint cc = min(c, p.hidden - 1);
  float qv[4], vs[6];
#pragma unroll
  for (uint q = 0; q < 4; ++q) qv[q] = query[ulong(min(q, p.queries - 1)) * p.hidden + cc];
#pragma unroll
  for (uint i = 0; i < 6; ++i) vs[i] = ink_state(states[min(i, p.count - 1)], p, min(i, p.count - 1), row, cc);
#pragma unroll
  for (uint q = 0; q < 4; ++q) if (!live || q >= p.queries) qv[q] = 0.0f;
#pragma unroll
  for (uint i = 0; i < 6; ++i) {
    const float v = live && i < p.count ? vs[i] : 0.0f;
    float s = simd_sum(v * v);
    if (lane == 0) partial[simd][i] = s;
    for (uint q = 0; q < 4; ++q) {
      s = simd_sum(v * qv[q]);
      if (lane == 0) partial[simd][6 + 6 * q + i] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < inkDepthSums) {
    float s = 0;
    for (uint g = 0; g < inkDepthChunk / 32; ++g) s += partial[g][tid];
    sums[(ulong(row) * ((p.hidden + inkDepthChunk - 1) / inkDepthChunk) + group.x) * inkDepthSums + tid] = s;
  }
}
kernel void inkling_depth_mix(device const bfloat *s0 [[buffer(0)]], device const bfloat *s1 [[buffer(1)]],
 device const bfloat *s2 [[buffer(2)]], device const bfloat *s3 [[buffer(3)]], device const bfloat *s4 [[buffer(4)]], device const bfloat *s5 [[buffer(5)]],
 device float *out [[buffer(7)]], device const float *sums [[buffer(8)]], constant InklingParams &p [[buffer(9)]],
 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float totals[inkDepthSums];
  device const bfloat *states[6] = {s0, s1, s2, s3, s4, s5};
  const uint chunks = (p.hidden + inkDepthChunk - 1) / inkDepthChunk, row = group.y;
  // Every load first, at clamped places (inkling_norm_impl): the chunks' sums, inkRowSlots of them at most in
  // registers, and the states, whose loads then overlap the sums' reduction.
  const uint c = group.x * inkDepthChunk + tid, cc = min(c, p.hidden - 1);
  float v[6];
#pragma unroll
  for (uint i = 0; i < 6; ++i) v[i] = ink_state(states[min(i, p.count - 1)], p, min(i, p.count - 1), row, cc);
  if (tid < inkDepthSums) {
    float part[inkRowSlots], s = 0;
#pragma unroll
    for (uint k = 0; k < inkRowSlots; ++k) part[k] = sums[(ulong(row) * chunks + min(k, chunks - 1)) * inkDepthSums + tid];
#pragma unroll
    for (uint k = 0; k < inkRowSlots; ++k) if (k < chunks) s += part[k];
    for (uint k = inkRowSlots; k < chunks; ++k) s += sums[(ulong(row) * chunks + k) * inkDepthSums + tid];
    totals[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c >= p.hidden) return;
  float inv[6];
  for (uint i = 0; i < p.count; ++i) {
    inv[i] = rsqrt(totals[i] / float(p.hidden) + 1e-6f);
    v[i] *= inv[i];
  }
  for (uint q = 0; q < p.queries; ++q) {
    float scores[6], maximum = -INFINITY, denom = 0, value = 0;
    for (uint i = 0; i < p.count; ++i) {
      scores[i] = totals[6 + 6 * q + i] * inv[i] * rsqrt(float(p.hidden));
      maximum = max(maximum, scores[i]);
    }
    for (uint i = 0; i < p.count; ++i) {
      const float w = exp(scores[i] - maximum);
      denom += w;
      value += w * v[i];
    }
    out[ulong(p.output.offset) + ulong(row) * p.output.stride + q * p.hidden + c] = value / denom;
  }
}
// A prompt chunk's depth mix: a threadgroup of inkThreads threads a row,
// thread t holding channels t + inkThreads i in registers (inkRowSlots of
// them), which sums each state's squares and each query's dots with each
// state over its channels, then reduces each sum once for the row (a
// simdgroup's simd_sum, then the simdgroups' partials in order), and writes
// the queries' outputs from the states it holds. The chunked pair reduce
// every 32 channels' sums and read each state twice: an 8,000-token
// prompt's mixes took 765 ms that way on an M1 Max, 401 this way.
kernel void inkling_depth_mix_rows(device const bfloat *s0 [[buffer(0)]], device const bfloat *s1 [[buffer(1)]],
 device const bfloat *s2 [[buffer(2)]], device const bfloat *s3 [[buffer(3)]], device const bfloat *s4 [[buffer(4)]], device const bfloat *s5 [[buffer(5)]],
 device const float *query [[buffer(6)]], device float *out [[buffer(7)]], constant InklingParams &p [[buffer(9)]],
 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[inkThreads / 32][inkDepthSums];
  threadgroup float totals[inkDepthSums];
  device const bfloat *states[6] = {s0, s1, s2, s3, s4, s5};
  const uint row = group.y;
  float v[inkRowSlots][6], sums[inkDepthSums];
  for (uint i = 0; i < inkDepthSums; ++i) sums[i] = 0.0f;
  // Every load first, at clamped places (inkling_norm_impl).
#pragma unroll
  for (uint k = 0; k < inkRowSlots; ++k) {
    const uint cc = min(tid + k * inkThreads, p.hidden - 1);
#pragma unroll
    for (uint i = 0; i < 6; ++i) v[k][i] = ink_state(states[min(i, p.count - 1)], p, min(i, p.count - 1), row, cc);
  }
#pragma unroll
  for (uint k = 0; k < inkRowSlots; ++k) {
    const uint c = tid + k * inkThreads;
    if (c >= p.hidden) break;
    float qv[4];
#pragma unroll
    for (uint q = 0; q < 4; ++q) qv[q] = q < p.queries ? query[ulong(q) * p.hidden + c] : 0.0f;
#pragma unroll
    for (uint i = 0; i < 6; ++i) {
      const float x = i < p.count ? v[k][i] : 0.0f;
      sums[i] += x * x;
#pragma unroll
      for (uint q = 0; q < 4; ++q) sums[6 + 6 * q + i] += x * qv[q];
    }
  }
#pragma unroll
  for (uint i = 0; i < inkDepthSums; ++i) {
    const float s = simd_sum(sums[i]);
    if (lane == 0) partial[simd][i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < inkDepthSums) {
    float s = 0;
    for (uint g = 0; g < inkThreads / 32; ++g) s += partial[g][tid];
    totals[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[6], scale[4][6];
  for (uint i = 0; i < p.count; ++i) inv[i] = rsqrt(totals[i] / float(p.hidden) + 1e-6f);
  for (uint q = 0; q < p.queries; ++q) {
    float scores[6], maximum = -INFINITY, denom = 0;
    for (uint i = 0; i < p.count; ++i) {
      scores[i] = totals[6 + 6 * q + i] * inv[i] * rsqrt(float(p.hidden));
      maximum = max(maximum, scores[i]);
    }
    for (uint i = 0; i < p.count; ++i) {
      scale[q][i] = exp(scores[i] - maximum);
      denom += scale[q][i];
    }
    for (uint i = 0; i < p.count; ++i) scale[q][i] *= inv[i] / denom;
  }
#pragma unroll
  for (uint k = 0; k < inkRowSlots; ++k) {
    const uint c = tid + k * inkThreads;
    if (c >= p.hidden) break;
    for (uint q = 0; q < p.queries; ++q) {
      float value = 0;
      for (uint i = 0; i < p.count; ++i) value += scale[q][i] * v[k][i];
      out[ulong(p.output.offset) + ulong(row) * p.output.stride + q * p.hidden + c] = value;
    }
  }
}
// The bf16 pair (low half first) of a word as floats.
inline float2 ink_bf16x2(uint word) { return float2(as_type<float>(word << 16), as_type<float>(word & 0xFFFF0000u)); }
// A message gate's low-rank projection of a tile of 8 rows: h[row][k] =
// bf16(pair . down[k]) with pair = bf16([rms(x), rms(m)]), a threadgroup
// inkGateOutputs outputs k, its 256 threads splitting the rows' norms and the
// 2 * hidden inputs, so one row's 64 outputs run on 16 threadgroups; a thread
// reads eight weights a load. A threadgroup an output read every row's inputs
// and norms 64 times: 46 us a call at four rows on an M1 Max.
constant uint inkGateRows = 8, inkGateOutputs = 4;
kernel void inkling_gate_down(device const float *x0 [[buffer(0)]], device const float *m0 [[buffer(1)]],
 device const bfloat *down0 [[buffer(2)]], device const bfloat *down1 [[buffer(3)]], device const bfloat *down2 [[buffer(4)]],
 device float *low0 [[buffer(5)]], constant InklingParams &p [[buffer(6)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[inkThreads / 32][inkGateRows][inkGateOutputs];
  threadgroup float inverse[inkGateRows][2];
  const uint first = group.x * inkGateRows, rows = min(inkGateRows, p.rows - first), k0 = group.y * inkGateOutputs,
             part = group.z;
  device const float *x = x0 + part * p.partSteps[0], *m = m0 + part * p.partSteps[1];
  device float *low = low0 + part * p.partSteps[2];
  device const bfloat *down = part == 0 ? down0 : part == 1 ? down1 : down2;
  // A row's channels load together at clamped places, then sum in channel order (inkling_norm_impl): a loop of
  // loads bounded at run time waited for each in turn: 1.54 ms against 1.11 a 3-row cycle on an M1 Max.
  for (uint r = 0; r < rows; ++r) {
    device const float *xr = x + ulong(p.source[0].offset) + ulong(first + r) * p.source[0].stride;
    device const float *mr = m + ulong(p.source[1].offset) + ulong(first + r) * p.source[1].stride;
    float a[inkRowSlots], b[inkRowSlots], sx = 0, sm = 0;
#pragma unroll
    for (uint i = 0; i < inkRowSlots; ++i) {
      const uint c = min(tid + i * inkThreads, p.hidden - 1);
      a[i] = xr[c];
      b[i] = mr[c];
    }
#pragma unroll
    for (uint i = 0; i < inkRowSlots; ++i)
      if (tid + i * inkThreads < p.hidden) { sx += a[i] * a[i]; sm += b[i] * b[i]; }
    for (uint c = tid + inkRowSlots * inkThreads; c < p.hidden; c += inkThreads) {
      sx += xr[c] * xr[c]; sm += mr[c] * mr[c];
    }
    sx = simd_sum(sx); sm = simd_sum(sm);
    if (lane == 0) { partial[simd][r][0] = sx; partial[simd][r][1] = sm; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2 * rows) {
    float s = 0;
    for (uint g = 0; g < inkThreads / 32; ++g) s += partial[g][tid / 2][tid % 2];
    inverse[tid / 2][tid % 2] = rsqrt(s / float(p.hidden) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[inkGateRows][inkGateOutputs] = {};
  for (uint v = tid; v < 2 * p.hidden / 8; v += inkThreads) {
    const uint c = v * 8;
    const bool fromX = c < p.hidden;
    const uint column = fromX ? c : c - p.hidden;
    device const float *base = fromX ? x : m;
    const InklingRowParams rp = p.source[fromX ? 0 : 1];
    float w[inkGateOutputs][8];
    for (uint o = 0; o < inkGateOutputs; ++o) {
      const uint4 packed = ((device const uint4 *)(down + ulong(k0 + o) * 2 * p.hidden))[v];
      const float2 w01 = ink_bf16x2(packed.x), w23 = ink_bf16x2(packed.y), w45 = ink_bf16x2(packed.z),
                   w67 = ink_bf16x2(packed.w);
      w[o][0] = w01.x; w[o][1] = w01.y; w[o][2] = w23.x; w[o][3] = w23.y;
      w[o][4] = w45.x; w[o][5] = w45.y; w[o][6] = w67.x; w[o][7] = w67.y;
    }
    for (uint r = 0; r < rows; ++r) {
      device const float4 *values = (device const float4 *)(base + ulong(rp.offset) + ulong(first + r) * rp.stride + column);
      const float4 lo = values[0], hi = values[1];
      const float scale = inverse[r][fromX ? 0 : 1];
      const float v8[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float pair[8];
      for (uint j = 0; j < 8; ++j) pair[j] = float(bfloat(v8[j] * scale));
      for (uint o = 0; o < inkGateOutputs; ++o) {
        float sum = 0;
        for (uint j = 0; j < 8; ++j) sum += pair[j] * w[o][j];
        acc[r][o] += sum;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0; r < rows; ++r)
    for (uint o = 0; o < inkGateOutputs; ++o) {
      const float s = simd_sum(acc[r][o]);
      if (lane == 0) partial[simd][r][o] = s;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < rows * inkGateOutputs) {
    const uint r = tid / inkGateOutputs, o = tid % inkGateOutputs;
    float s = 0;
    for (uint g = 0; g < inkThreads / 32; ++g) s += partial[g][r][o];
    low[ulong(p.output.offset) + ulong(first + r) * p.output.stride + k0 + o] = float(bfloat(s));
  }
}
// inkling_gate_down for steps of more than one 8-row tile (prefill): the
// same h[row][k] as 8x8 matrix multiplies over tiles of inkGateMmaRows rows,
// simdgroup s computing outputs 8s..8s+7 of every 8 rows of the tile over
// 64-input chunks of the pair and of the weights staged as fp32, so a
// chunk's weights serve 32 rows and its inputs all 64 outputs. The
// per-output form took 1.1 s of an 8,000-token prefill on an M1 Max.
constant uint inkGateMmaRows = 32;
kernel void inkling_gate_down_mma(device const float *x0 [[buffer(0)]], device const float *m0 [[buffer(1)]],
 device const bfloat *down0 [[buffer(2)]], device const bfloat *down1 [[buffer(3)]], device const bfloat *down2 [[buffer(4)]],
 device float *low0 [[buffer(5)]], constant InklingParams &p [[buffer(6)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partial[inkThreads / 32][inkGateMmaRows][2];
  threadgroup float inverse[inkGateMmaRows][2];
  threadgroup float a[inkGateMmaRows][64];
  threadgroup float b[64][64 + 4];
  const uint first = group.x * inkGateMmaRows, rows = min(inkGateMmaRows, p.rows - first), part = group.z;
  device const float *x = x0 + part * p.partSteps[0], *m = m0 + part * p.partSteps[1];
  device float *low = low0 + part * p.partSteps[2];
  device const bfloat *down = part == 0 ? down0 : part == 1 ? down1 : down2;
  for (uint r = 0; r < rows; ++r) {
    float sx = 0, sm = 0;
    for (uint col = tid; col < p.hidden; col += inkThreads) {
      const float u = x[ulong(p.source[0].offset) + ulong(first + r) * p.source[0].stride + col];
      const float v = m[ulong(p.source[1].offset) + ulong(first + r) * p.source[1].stride + col];
      sx += u * u; sm += v * v;
    }
    sx = simd_sum(sx); sm = simd_sum(sm);
    if (lane == 0) { partial[simd][r][0] = sx; partial[simd][r][1] = sm; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2 * rows) {
    float s = 0;
    for (uint g = 0; g < inkThreads / 32; ++g) s += partial[g][tid / 2][tid % 2];
    inverse[tid / 2][tid % 2] = rsqrt(s / float(p.hidden) + 1e-6f);
  }
  simdgroup_float8x8 acc[inkGateMmaRows / 8];
  for (uint t = 0; t < inkGateMmaRows / 8; ++t) acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint chunk = 0; chunk < 2 * p.hidden; chunk += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool fromX = chunk < p.hidden;
    const uint column = fromX ? chunk : chunk - p.hidden;
    const InklingRowParams rp = p.source[fromX ? 0 : 1];
    device const float *base = fromX ? x : m;
    for (uint i = tid; i < inkGateMmaRows * 64; i += inkThreads) {
      const uint r = i / 64, j = i % 64;
      a[r][j] = r < rows ? float(bfloat(base[ulong(rp.offset) + ulong(first + r) * rp.stride + column + j] *
                                        inverse[r][fromX ? 0 : 1]))
                         : 0.0f;
    }
    for (uint i = tid; i < p.rank * 64; i += inkThreads) {
      const uint k = i / 64, j = i % 64;
      b[j][k] = float(down[ulong(k) * 2 * p.hidden + chunk + j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd * 8 < p.rank)
      for (uint kk = 0; kk < 64; kk += 8) {
        simdgroup_float8x8 mb;
        simdgroup_load(mb, &b[kk][simd * 8], 64 + 4);
        for (uint t = 0; t < inkGateMmaRows / 8; ++t) {
          simdgroup_float8x8 ma;
          simdgroup_load(ma, &a[t * 8][kk], 64);
          simdgroup_multiply_accumulate(acc[t], ma, mb, acc[t]);
        }
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd * 8 >= p.rank) return;
  for (uint t = 0; t < inkGateMmaRows / 8; ++t) {
    simdgroup_store(acc[t], &b[simd * 8][0] + t * 8 * 0, 64 + 4);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 64; i += 32) {
      const uint r = t * 8 + i / 8, o = i % 8;
      if (r < rows)
        low[ulong(p.output.offset) + ulong(first + r) * p.output.stride + simd * 8 + o] =
            float(bfloat(b[simd * 8 + i / 8][o]));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
// The gate's output of a tile of 8 rows and 64 channels: out = (1 -
// tanh(h . forget[c])) * x + tanh(h . gain[c]) * m, one thread a channel,
// whose 64 forget and gain weights it reads once for the tile, eight a load;
// 64 threads a threadgroup, so one row's channels spread over 36 cores.
constant uint inkGateUpThreads = 64;
kernel void inkling_gate_up(device const float *x0 [[buffer(0)]], device const float *m0 [[buffer(1)]],
 device const float *low0 [[buffer(2)]], device const bfloat *forget0 [[buffer(3)]], device const bfloat *forget1 [[buffer(4)]],
 device const bfloat *forget2 [[buffer(5)]], device const bfloat *gain0 [[buffer(6)]], device const bfloat *gain1 [[buffer(7)]],
 device const bfloat *gain2 [[buffer(8)]], device float *out0 [[buffer(9)]], constant InklingParams &p [[buffer(10)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint part = group.z;
  device const float *x = x0 + part * p.partSteps[0], *m = m0 + part * p.partSteps[1], *low = low0 + part * p.partSteps[2];
  device float *out = out0 + part * p.partSteps[3];
  device const bfloat *forget = part == 0 ? forget0 : part == 1 ? forget1 : forget2;
  device const bfloat *gain = part == 0 ? gain0 : part == 1 ? gain1 : gain2;
  threadgroup float h[inkGateRows][64];
  const uint first = group.x * inkGateRows, rows = min(inkGateRows, p.rows - first);
  for (uint i = tid; i < rows * p.rank; i += inkGateUpThreads)
    h[i / p.rank][i % p.rank] = low[ulong(p.dynamicRows.offset) + ulong(first + i / p.rank) * p.dynamicRows.stride + i % p.rank];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = group.y * inkGateUpThreads + tid;
  if (c >= p.hidden) return;
  device const uint4 *fw = (device const uint4 *)(forget + ulong(c) * p.rank);
  device const uint4 *gw = (device const uint4 *)(gain + ulong(c) * p.rank);
  float f[inkGateRows] = {}, g[inkGateRows] = {};
  for (uint v = 0; v < p.rank / 8; ++v) {
    const uint4 fp = fw[v], gp = gw[v];
    const float2 f01 = ink_bf16x2(fp.x), f23 = ink_bf16x2(fp.y), f45 = ink_bf16x2(fp.z), f67 = ink_bf16x2(fp.w);
    const float2 g01 = ink_bf16x2(gp.x), g23 = ink_bf16x2(gp.y), g45 = ink_bf16x2(gp.z), g67 = ink_bf16x2(gp.w);
    const float fv[8] = {f01.x, f01.y, f23.x, f23.y, f45.x, f45.y, f67.x, f67.y};
    const float gv[8] = {g01.x, g01.y, g23.x, g23.y, g45.x, g45.y, g67.x, g67.y};
    for (uint r = 0; r < rows; ++r)
      for (uint j = 0; j < 8; ++j) {
        const float hv = h[r][v * 8 + j];
        f[r] += hv * fv[j];
        g[r] += hv * gv[j];
      }
  }
  for (uint r = 0; r < rows; ++r) {
    const uint row = first + r;
    const float a = x[ulong(p.source[0].offset) + ulong(row) * p.source[0].stride + c];
    const float b = m[ulong(p.source[1].offset) + ulong(row) * p.source[1].stride + c];
    out[ulong(p.output.offset) + ulong(row) * p.output.stride + c] =
        (1 - tanh(clamp(f[r], -20.0f, 20.0f))) * a + tanh(clamp(g[r], -20.0f, 20.0f)) * b;
  }
}
template <typename W> kernel void inkling_norm_impl(device const float *input0 [[buffer(0)]], device const W *weight0 [[buffer(1)]],
 device const W *weight1 [[buffer(2)]], device const W *weight2 [[buffer(3)]],
 device bfloat *out0 [[buffer(4)]], constant InklingParams &p [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, part = group.y;
  device const float *input = input0 + part * p.partSteps[0];
  device bfloat *out = out0 + part * p.partSteps[3];
  device const W *weight = part == 0 ? weight0 : part == 1 ? weight1 : weight2;
  // A thread's channels and weights in registers, every load issued at once (clamped, not predicated): a loop
  // of loads bounded at run time waits for each in turn, 10 us a norm of 2304 channels on an M1 Max against 4.
  threadgroup float scratch[8]; float sum=0, v[inkRowSlots], w[inkRowSlots];
  device const float *x=input+ulong(p.source[0].offset)+ulong(row)*p.source[0].stride;
#pragma unroll
  for(uint i=0;i<inkRowSlots;++i) { const uint c=min(tid+i*inkThreads,p.hidden-1); v[i]=x[c]; w[i]=float(weight[c]); }
#pragma unroll
  for(uint i=0;i<inkRowSlots;++i) if(tid+i*inkThreads<p.hidden) sum+=v[i]*v[i];
  // Channels past the registers' (none of Inkling's) one at a time.
  for(uint c=tid+inkRowSlots*inkThreads;c<p.hidden;c+=inkThreads) sum+=x[c]*x[c];
  float inv=rsqrt(ink_reduce(sum,scratch,tid)/float(p.hidden)+1e-6f);
  device bfloat *o=out+ulong(p.output.offset)+ulong(row)*p.output.stride;
#pragma unroll
  for(uint i=0;i<inkRowSlots;++i) if(tid+i*inkThreads<p.hidden) o[tid+i*inkThreads]=bfloat(v[i]*inv*w[i]);
  for(uint c=tid+inkRowSlots*inkThreads;c<p.hidden;c+=inkThreads) o[c]=bfloat(x[c]*inv*float(weight[c]));
}
template [[host_name("inkling_norm")]] kernel void inkling_norm_impl<bfloat>(device const float *, device const bfloat *, device const bfloat *, device const bfloat *, device bfloat *, constant InklingParams &, uint2, uint);
template [[host_name("inkling_norm_f32")]] kernel void inkling_norm_impl<float>(device const float *, device const float *, device const float *, device const float *, device bfloat *, constant InklingParams &, uint2, uint);

kernel void inkling_conv(device const bfloat *input [[buffer(0)]],
 device const bfloat *h0 [[buffer(1)]], device const bfloat *h1 [[buffer(2)]], device const bfloat *h2 [[buffer(3)]], device const bfloat *h3 [[buffer(4)]],
 device bfloat *o0 [[buffer(5)]], device bfloat *o1 [[buffer(6)]], device bfloat *o2 [[buffer(7)]], device bfloat *o3 [[buffer(8)]],
 device const bfloat *weights [[buffer(9)]], device const float *dt [[buffer(10)]], device uchar *out [[buffer(11)]],
 device const SplashLaneRows *layout [[buffer(12)]], device const SplashTreeRow *tree [[buffer(13)]], device bfloat *tape [[buffer(14)]],
 device const uchar *addend [[buffer(15)]], constant InklingParams &p [[buffer(16)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint c=group.y*inkThreads+tid, r=group.x, s=group.z; if(c>=p.hidden || s>=p.sequences) return;
  SplashLaneRows lr=layout[s]; if(ulong(lr.first)+lr.rows>p.rows) return;
  device const bfloat *hist=SPLASH_LANE_BINDING(s,h0,h1,h2,h3);
  if(r<lr.rows) {
    uint packed=lr.first+r, path=p.tree?tree[packed].ancestors:0; int depth=p.tree?int(tree[packed].depth):int(r);
    float sum=0;
    for(uint lag=0;lag<p.taps;++lag) {
      int pos=depth-int(lag); float v;
      if(pos<0) v=float(hist[ulong(p.historyOffset)+ulong(int(p.taps-1)+pos)*p.historyStride+c]);
      else { uint source=p.tree?splash_path_row(path,uint(pos)):uint(pos); v=ink_read(input,p.source[0],lr.first+source,c); }
      float w=p.dynamic?float(weights[ulong(lag)*p.hidden+c])+dt[ulong(p.dynamicRows.offset)+ulong(packed)*p.dynamicRows.stride+(c/(p.hidden/p.heads))*p.taps+lag]:float(weights[ulong(c)*p.taps+p.taps-1-lag]);
      sum+=w*v;
    }
    float x=ink_read(input,p.source[0],packed,c);
    const bfloat y=bfloat((sum+x)*rsqrt(float(p.dynamic?2*p.taps+1:p.taps+1)));
    const ulong o=ulong(p.output.offset)+ulong(packed)*p.output.stride+c;
    float value=float(y);
    if(p.addend) {
      const ulong a=ulong(p.source[2].offset)+ulong(packed)*p.source[2].stride+c;
      value+=p.addendFloat?((device const float *)addend)[a]:float(((device const bfloat *)addend)[a]);
    }
    if(p.outputFloat) ((device float *)out)[o]=value; else ((device bfloat *)out)[o]=bfloat(value);
    // A verify's inputs, which its commit folds into the lanes' histories.
    if(p.count) tape[ulong(p.source[1].offset)+ulong(packed)*p.source[1].stride+c]=bfloat(x);
  } else if(p.writeHistory && r-lr.rows<p.taps-1) {
    uint j=r-lr.rows; int pos=int(lr.rows)-int(p.taps-1)+int(j);
    bfloat v=pos<0?hist[ulong(p.historyOffset)+ulong(int(p.taps-1)+pos)*p.historyStride+c]:input[ulong(p.source[0].offset)+ulong(lr.first+uint(pos))*p.source[0].stride+c];
    device bfloat *dst=SPLASH_LANE_BINDING(s,o0,o1,o2,o3); dst[ulong(p.historyOffset)+ulong(j)*p.historyStride+c]=v;
  }
}
kernel void inkling_history(device const bfloat *input [[buffer(0)]], device const bfloat *history [[buffer(1)]], device bfloat *out [[buffer(2)]], constant InklingParams &p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
 uint j=group.x,c=group.y*inkThreads+tid; if(c>=p.hidden)return; int pos=int(p.rows)-int(p.taps-1)+int(j);
 out[ulong(p.historyOffset)+ulong(j)*p.historyStride+c]=pos<0?history[ulong(p.historyOffset)+ulong(int(p.taps-1)+pos)*p.historyStride+c]:input[ulong(p.source[0].offset)+ulong(pos)*p.source[0].stride+c];
}
kernel void inkling_commit(device const bfloat *input [[buffer(0)]], device const bfloat *h0 [[buffer(1)]], device const bfloat *h1 [[buffer(2)]], device const bfloat *h2 [[buffer(3)]], device const bfloat *h3 [[buffer(4)]], device bfloat *o0 [[buffer(5)]], device bfloat *o1 [[buffer(6)]], device bfloat *o2 [[buffer(7)]], device bfloat *o3 [[buffer(8)]], device const uint *paths [[buffer(9)]], device const SplashLaneRows *layout [[buffer(10)]], constant InklingParams &p [[buffer(11)]], uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
 uint j=group.x,c=group.y*inkThreads+tid,s=group.z; if(c>=p.hidden*p.blocks||s>=p.sequences)return;
 SplashLaneRows lr=layout[s]; if(ulong(lr.first)+lr.rows>p.rows)return;
 uint b=c/p.hidden, ch=c%p.hidden, path=paths[s], taken=popcount(path); if(p.count) taken=min(taken,p.count);
 int pos=int(taken)-int(p.taps-1)+int(j);
 device const bfloat *hist=SPLASH_LANE_BINDING(s,h0,h1,h2,h3); device bfloat *out=SPLASH_LANE_BINDING(s,o0,o1,o2,o3);
 bfloat v=pos<0?hist[ulong(b)*p.historyBlockStride+ulong(int(p.taps-1)+pos)*p.historyStride+ch]:input[ulong(p.source[0].offset)+ulong(lr.first+splash_path_row(path,uint(pos)))*p.source[0].stride+ulong(b)*p.blockStride+ch];
 out[ulong(b)*p.historyBlockStride+ulong(j)*p.historyStride+ch]=v;
}
kernel void inkling_cap(device float *logits [[buffer(0)]], constant InklingParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 // Padding ids take -1e30, the reference's mask value: finite, as every
 // consumer of logits requires, and exp() of it is 0.
 if(ulong(i)>=ulong(p.rows)*p.hidden)return; uint c=i%p.hidden; logits[i]=c<p.count?tanh(clamp(logits[i]/p.softcap,-20.0f,20.0f))*p.softcap:-1e30f;
}
