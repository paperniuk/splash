#pragma once

#include <metal_stdlib>

using namespace metal;

// Pair `pair` of the target's RoPE row at the (t, h, w) position `position`:
// its cosine and sine, Qwen3.5's interleaved M-RoPE taking frequency `pair` on
// axis pair % 3, both scaled by `scale`, YaRN's attention factor (1 without
// YaRN). Every kernel that rotates by the target's RoPE computes its rows
// with it, so they all rotate alike (ops::RoPE). Precise: Apple7's fast
// cos/sin are off by ~5e-5 at the angles of long contexts, where these rows
// must stay within 1e-6.
inline float2 target_rope_pair(uint3 position, device const float *inverse_frequencies, uint pair,
                               float scale) {
  const float angle = float(position[pair % 3]) * inverse_frequencies[pair];
  float c;
  const float s = precise::sincos(angle, c);
  return float2(c, s) * scale;
}
