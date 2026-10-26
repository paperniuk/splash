// GGUF decode on Apple9's register simdgroup_matrix path. The 8x8x8 MMA runs
// on the FP32 pipe there, so the kernel keeps every other FP32 operation to
// the minimum a group scale needs, and does the rest on the integer pipe:
// - weights enter the MMA as exact bf16: 128 + code for linear codes with a
//   min, 160 + code - zero for seeded linear codes (one add), the code itself
//   (converted) for 8-bit linear codes, or the codebook, int8 or grid value;
// - one MMA chain per 32 inputs (per 16 for 16-input scales), closed by the
//   fp32 epilogue s * chain; a format with a min adds b * sum once per
//   coefficient group, the F::Group inputs one scale and min cover (16, 32,
//   or 64 to 256 where the groups of a meta unit share them, as MLX affine's
//   groups of 64 and 128 do), but at least once per 64-input span of the
//   table, with the last chain of either (b = m - 128 s, or b = m for 8-bit
//   codes; sum those inputs, from the span's sums of 32), so that it reads
//   the current span alone; with a zero point, that chain starts from
//   -160 * sum instead (the operand being 160 + code - zero), the table's
//   seed of 16 inputs for 16-input scales, and Q2_K's min per 16 inputs adds
//   b * seed with b = m / -160;
// - coefficients decoded once per simdgroup and coefficient group into
//   threadgroup memory;
// - L request lanes per threadgroup share the weight operands and coefficients;
//   each lane's chains and epilogue run as with L = 1, so its result does not
//   depend on the batch width.
// Weights: the GGUF image (metal/abi/QuantFormat.h), read by chunk: lane c of
// a fragment reads chunk c of each group. Activations: the Table16 table and
// row sums (kernels/common/gguf_sgmatrix.h).
//
// Apple7/8 (M1/M2) run the same tile with the operands widened (the F32
// instances, gguf_decode_sgf_*): they have no bf16 arithmetic and issue the MMA
// on the ordinary ALU pipes, so the weights enter as exact half (fp32 for bf16
// values, which half does not hold) and the activations as exact fp32. Every
// product stays exact, and the chains, coefficients and epilogue are the Apple9
// tile's before 1.5.0 (Shape's F32). On an M1 Max this replaces the staged MMA
// tile (kernels/shared/gguf_linear_mma.metal), whose per-element dequantization
// into threadgroup memory bounds MLX 4-bit decode. Apple7/8 run a fused
// projection's segments as dispatches of their own: an F32 instance of the
// run-time-format fused kernel below, every format inlined, returned wrong
// values at four lanes under shader validation and then hung an M1 Max's GPU.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

namespace gguf_sg {

// Whether a format's coefficients cover whole 64-input spans: MLX affine in groups of 64 or 128.
template <class F> struct SpanCoefficients { static constexpr constant bool value = false; };
template <uint Bits, uint Groups> struct SpanCoefficients<FmtAffine<Bits, Groups>> {
  static constexpr constant bool value = Groups >= 2;
};

// F32 selects the F32 instances' shape (Apple7/8): every chain closes against a coefficient of its own, one per 32
// inputs, or per 16 from the table's seeds where codes have a zero point, as before this tile seeded zero points'
// chains in the kernel and closed a min once per coefficient group.
template <class F, bool F32 = false> struct Shape {
  enum : uint {
    Linear = F::Kind == QuantLinear,
    HasMin = Linear && F::Zero == 0,                          // Q4_K, Q5_K, Q4_1, Q5_1, Q2_K, MLX affine: s code + m
    // Linear codes past 127 (MLX affine 8-bit), which enter the MMA as the code itself: no bf16 128 + code holds them
    Wide = QuantWideCodes<F>::value,
    // Linear codes whose chains start from -160 times the sum of their inputs, so that 160 + code - zero enters the
    // MMA: s (code - zero) with a zero point, once per coefficient group and span from the table's sums of 32 where a
    // scale covers 32 inputs or more (Q4_0, Q5_0, IQ1_S, PQ2_0, Q2_0) and per 16-input chain from the table's seeds
    // (Q6_K, Q3_K, IQ1_M); and Q2_K's min per 16 inputs
    Seeded = Linear && (!HasMin || F::Group == 16),
    // bf16 bits of the operand of code 0: 128, or 160 - zero when seeded
    Operand = 0x4300 + (Seeded ? kZeroPointOffset - 128 - F::Zero : 0),
    CG = F::Group == 16 || (F32 && Seeded) ? 2 : 1,           // MMA chains per 32 inputs
    CPG = F32 ? 1 : CG * F::Group / 32,                       // chains per coefficient group
    MinChains = CPG < 2 * CG ? CPG : 2 * CG,                  // chains per term: the group's, at most a span's
    // Spans decoded per coefficient unit: every span of eight groups (one meta unit of eight, or two of four), or
    // half of them where a span's coefficients take twice Q4_K's storage (Q2_K's (s, b) per 16 inputs)
    UnitSpans = F::MetaGroups == 1 ? 1 : CG * (HasMin ? 2 : 1) <= 2 ? 4 : 2,
    J = 2 * UnitSpans * CG / CPG,                             // coefficients per column and unit
    // One coefficient over each 64-input span (MLX affine in groups of 64 or 128), whose chain the F32 instances
    // run over the whole span, closing it once
    SpanChain = SpanCoefficients<F>::value,
  };
  // Each coefficient reads its own group's meta unit (coefficient_source), so a unit of two or four groups (MLX
  // affine's 64 or 128 elements, PQ2_0's 128, Q2_0's 64) splits a coefficient unit of four spans: such a format takes
  // rows of whole 256 inputs here (quant_register_tensor).
  static_assert(F::MetaGroups == 8 || F::MetaGroups == 4 || F::MetaGroups == 2 || F::MetaGroups == 1,
                "a meta unit is one, two, four or eight groups");
  static_assert(UnitSpans * GGUF_TABLE16_SPAN_INPUTS == quant_register_unit(F::Id),
                "the host splits K in the coefficient units of quant_register_unit");
  static_assert(!Wide || (HasMin && !Seeded), "wide codes have a min per group");
  static_assert(F::Group == 16 || (F::Group % 32 == 0 && F::Group <= 32 * F::MetaGroups),
                "a coefficient covers 16 inputs, or whole groups of one meta unit");
  static_assert(CPG >= 1 && 2 * UnitSpans * CG % CPG == 0, "a coefficient unit holds whole coefficient groups");
  // decode adds a min term with its last chain, or starts that chain from a zero point's, from the span's 32-input
  // sums of the term's chains: a chain covers one 32-input group, and a term's chains lie in one span, so it reads
  // no sum before the span.
  static_assert(!(HasMin && !Seeded) || CG == 1, "a min term's chains are 32-input groups");
  static_assert(2 * CG % MinChains == 0, "a min term's chains lie in one span");
};
template <class F> using Coef = metal::conditional_t<Shape<F>::HasMin != 0, float2, float>;
// The coefficients of a threadgroup's simdgroups for one unit; the run-time-format kernels hold every format's in
// storage of the largest size.
template <class F, bool F32 = false> constant constexpr uint kCoefs = GGUF_TILE_COLUMNS * Shape<F, F32>::J;
static_assert(GGUF_REGISTER_COLUMNS == 2 * 8, "a simdgroup's columns are its two 8-column MMA fragments");
constant constexpr uint kCoefFloat2s = kCoefs<FmtQ4K>;
#define GGUF_SG_COEF_BYTES(F, f) static_assert(kCoefs<F> * sizeof(Coef<F>) <= kCoefFloat2s * sizeof(float2), #f " coefficients fit");
QUANT_REGISTER_FORMATS(GGUF_SG_COEF_BYTES)
#undef GGUF_SG_COEF_BYTES

// The type of the weight operand: bf16 on Apple9; on Apple7/8 half, which holds every other format's operands
// exactly, or fp32 for bf16 values.
template <class F, bool F32>
using OperandType = metal::conditional_t<!F32, bfloat, metal::conditional_t<F::Kind == QuantFloat16, float, half>>;
// The pair table's entries: as the operand, half for fp32 operands (no codebook format has them).
template <class F, bool F32> using LutPair = vec<metal::conditional_t<F32, half, bfloat>, 2>;
// The grids whose F32 instances read their magnitudes as half pairs from the pair table: entry e's four at lut[2e]
// and lut[2e + 1], so the operand is a load instead of four byte-to-half conversions, and index(q) the entries of
// pairs 0, 1 (x) and 2, 3 (y). On an M2 Max this takes 12-14% off the 27B's IQ3_S and IQ3_XXS projections at one
// lane and 2-9% at two to four.
template <uint Id> struct HalfGridOf { static constexpr constant uint Entries = 0; };
template <> struct HalfGridOf<GGUF_FMT_IQ3S> {
  static constexpr constant uint Entries = 512;
  static uint word(uint e) { return kIQ3SGrid[e]; }
  static uint2 index(uint q) { return uint2((q & 0xFF) | ((q >> 16) & 0x100), ((q >> 8) & 0xFF) | ((q >> 17) & 0x100)); }
};
template <> struct HalfGridOf<GGUF_FMT_IQ3XXS> {
  static constexpr constant uint Entries = 256;
  static uint word(uint e) { return kIQ3XXSGrid[e]; }
  static uint2 index(uint q) { return uint2(q & 0xFF, (q >> 8) & 0xFF); }
};
template <class F> using HalfGrid = HalfGridOf<F::Id>;
// The pair table's entries: a codebook's pairs, an F32 instance's half grid, or none.
template <class F, bool F32>
constant constexpr uint kLutEntries = F::Kind == QuantCodebook ? kQuantPairTableEntries
                                      : F32 && HalfGrid<F>::Entries ? 2 * HalfGrid<F>::Entries
                                                                    : 1;
// Fills the pair table (quant_pair_table, or the half grid); called by all threads of the threadgroup.
template <class F, bool F32, class T> inline void lut_table(threadgroup T *lut, uint thread_index, uint threads) {
  if constexpr (F32 && HalfGrid<F>::Entries) {
    for (uint e = thread_index; e < HalfGrid<F>::Entries; e += threads) {
      const uchar4 m = as_type<uchar4>(HalfGrid<F>::word(e));
      lut[2 * e] = T(half(m[0]), half(m[1]));
      lut[2 * e + 1] = T(half(m[2]), half(m[3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    quant_pair_table<F>(lut, thread_index, threads);
  }
}
// The activation operand: the table's bf16, or on Apple7/8 its fp32 (Table16F), which enters the MMA as it is.
template <bool F32> using InputType = metal::conditional_t<F32, float, bfloat>;
template <bool F32> using TableElem = InputType<F32>;

// The exact half operand of pair f of a chunk, as operand<F> but for Apple7/8: linear codes or-ed into half 1024
// (whose ulp is 1) less 1024 minus the bf16 tile's offset, so the chains are that tile's.
template <class F>
inline vec<OperandType<F, true>, 2> operand_f32(typename F::Chunk ch, uint f, threadgroup const half2 *lut) {
  typedef Shape<F> S;
  if constexpr (F::Kind == QuantLinear) {
    constexpr uint offset = S::Wide ? 0 : S::Seeded ? kZeroPointOffset - F::Zero : 128;
    return as_type<half2>(F::codes(ch)[f] | 0x64006400u) - half2(half(1024 - offset));
  } else if constexpr (F::Kind == QuantCodebook) {
    return lut[(F::indices(ch) >> (8 * f)) & 0xFFu];
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 v = F::values(ch);
    return half2(as_type<char2>(ushort((f < 2 ? v.x : v.y) >> (16 * (f & 1)))));
  } else if constexpr (F::Kind == QuantFloat8) {
    const uint2 v = F::values(ch);
    return quant_e4m3_pair(quant_byte_pairs(((f < 2 ? v.x : v.y) >> (16 * (f & 1))) & 0xFFFFu).x);
  } else if constexpr (F::Kind == QuantFloat16) {
    static_assert(is_same_v<typename F::Value, bfloat>, "the register tile takes bf16 values only");
    return float2(as_type<bfloat2>(F::values(ch)[f]));
  } else if constexpr (HalfGrid<F>::Entries != 0) {
    const uint2 e = HalfGrid<F>::index(ch);
    const uint s = F::signs(ch) >> (2 * f);
    const half2 v = lut[2 * (f < 2 ? e.x : e.y) + (f & 1)];
    return select(v, -v, bool2(s & 1u, s & 2u));
  } else {
    const uint2 g = F::grid(ch);
    const uchar4 m = as_type<uchar4>(f < 2 ? g.x : g.y);
    const uint s = F::signs(ch) >> (2 * f);
    const half2 v = half2(m[2 * (f & 1)], m[2 * (f & 1) + 1]);
    return select(v, -v, bool2(s & 1u, s & 2u));
  }
}

// MLX affine 4-bit over whole spans, whose F32 chains take nibble_operand.
template <class F> struct NibbleSpan { static constexpr constant bool value = false; };
template <uint Groups> struct NibbleSpan<FmtAffine<4, Groups>> { static constexpr constant bool value = Groups >= 2; };
// The exact half operand 1 + code / 64 of pair f of a 4-bit chunk word: the codes at bits 4-7 of each halfword
// below half 1's bits, where pair 1's already sit, which takes a shift, a mask and an or per pair (operand_f32
// adds a subtraction), and a span chain whose min part (64 s against codes to 15) is half operand_f32's.
inline half2 nibble_operand(uint word, uint f) {
  const uint at = f == 0 ? word << 4 : f == 1 ? word : word >> (4 * (f - 1));
  return as_type<half2>((at & 0x00F000F0u) | 0x3C003C00u);
}
// The same of a 5-bit chunk in Q5_K's planes, its fifth bits (pair p's at bits 2p and 16 + 2p) at bit 8 of each
// halfword: 1 + code / 64, exact for codes to 31, where operand_f32 shifts them below the nibble and subtracts. On
// an M2 Max this takes 4-5% off the 27B's Q5_K projections at one lane (1-2.5% at two to four), and the 27B in
// UD-Q4_K_XL and UD-Q5_K_M, 45% Q5_K, decodes 1.8-2.2% faster at one lane.
inline half2 nibble_operand(uint2 chunk, uint f) {
  const uint at = f == 0 ? chunk.x << 4 : f == 1 ? chunk.x : chunk.x >> (4 * (f - 1));
  return as_type<half2>((at & 0x00F000F0u) | ((chunk.y << (8 - 2 * f)) & 0x01000100u) | 0x3C003C00u);
}

// The exact half operand (code - 32) / 64 of pair f of a Q6_K chunk (uint2: the 4-bit word, then the high bits as
// quant_spread2 places them, pair p's at bits 4p and 16 + 4p): the high bits at bits 8-9 of each halfword above the
// nibble, 1 + code / 64, less 1.5, so a chain of Q6_K's zero point 32 needs no seed. On an M2 Max this takes 3-4%
// off the 27B's Q6_K projections at one lane and 6-18% at two to four, and the 27B in UD-Q6_K_M, 60% Q6_K, decodes
// 1.4%, 5.3% and 5.7% faster at one, two and four lanes.
inline half2 nibble6_operand(uint2 chunk, uint f) {
  const uint at = f == 0 ? chunk.x << 4 : f == 1 ? chunk.x : chunk.x >> (4 * (f - 1));
  const uint high = f == 3 ? chunk.y >> 4 : chunk.y << (8 - 4 * f);
  return as_type<half2>((at & 0x00F000F0u) | (high & 0x03000300u) | 0x3C003C00u) - half2(1.5h);
}
// The exact half operand (code - 4) / 8 of pair f of a Q3_K chunk (uint2: the low bits as quant_spread2 places them,
// pair p's at bits 4p and 16 + 4p, then the hmask bits as quant_spread1 does, at 2p and 16 + 2p): the low bits at bits
// 7-8 and the hmask bit at 9 of each halfword, 1 + code / 8, less 1.5, so a chain of Q3_K's zero point 4 needs no
// seed. On an M2 Max this takes 2% off the 27B's Q3_K projections at one and two lanes and 6-10% at three and four.
inline half2 q3_operand(uint2 chunk, uint f) {
  const uint low = f == 0 ? chunk.x << 7 : f == 1 ? chunk.x << 3 : chunk.x >> (4 * f - 7);
  return as_type<half2>((low & 0x01800180u) | ((chunk.y << (9 - 2 * f)) & 0x02000200u) | 0x3C003C00u) - half2(1.5h);
}

// Apple7/8's (F32) view of Q4_0, Q5_0, PQ2_0 and Q2_0: the zero point taken as a min, m = -zero d, so their chains close once per
// 32 inputs against the table's sums, as Q4_1's and Q5_1's do, instead of once per 16 against the seeds. On an M2
// Max this takes 8-14% off Q4_0's 17408 x 5120 gate/up at one to four lanes (Q5_0's 2-8%), and the 27B in Q4_0,
// MLX 4-bit's size, decodes 3-7% below MLX 4-bit instead of 10-14%. Q2_0 shares FmtTwoBit's algebra with PQ2_0
// (value = d * (q - 1), Zero 1), so it takes the same view.
template <class F> struct ZeroAsMin {
  enum : uint { Id = F::Id, P0 = F::P0, P1 = F::P1, MetaBytes = F::MetaBytes };
  enum : ushort { MetaGroups = F::MetaGroups, Zero = 0, Group = F::Group };
  static constexpr constant QuantKind Kind = F::Kind;
  static constexpr constant bool ScaleInChunk = F::ScaleInChunk;
  typedef typename F::Payload Payload; typedef typename F::Chunk Chunk; typedef typename F::Meta Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return F::load(p0, p1); }
  static Meta loadMeta(device uchar *m) { return F::loadMeta(m); }
  static Chunk chunk(Payload w, ushort c) { return F::chunk(w, c); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return F::loadChunk(p0, p1, c); }
  static uint4 codes(Chunk q) { return F::codes(q); }
  static QuantCoef coef(Meta mt, ushort j) {
    QuantCoef k = F::coef(mt, j);
    k.m = -float(F::Zero) * k.s;
    return k;
  }
};
// The format an F32 instance decodes F as.
template <class F> struct F32Format { typedef F type; };
template <> struct F32Format<FmtQ40> { typedef ZeroAsMin<FmtQ40> type; };
template <> struct F32Format<FmtQ50> { typedef ZeroAsMin<FmtQ50> type; };
template <> struct F32Format<FmtPQ20> { typedef ZeroAsMin<FmtPQ20> type; };
template <> struct F32Format<FmtQ20> { typedef ZeroAsMin<FmtQ20> type; };
// Formats with a min per 32 inputs whose chunk is one word of 4-bit codes in Q4_K's order (Q4_K, Q4_1, Q4_0 as
// ZeroAsMin, MLX affine 4-bit in groups of 32), or 5-bit codes in Q5_K's planes (Q5_K, Q5_1, Q5_0 as ZeroAsMin):
// their F32 chains per 32 inputs take nibble_operand too.
template <class F> struct NibbleChunk { static constexpr constant bool value = false; };
template <> struct NibbleChunk<FmtQ4K> { static constexpr constant bool value = true; };
template <> struct NibbleChunk<FmtQ41> { static constexpr constant bool value = true; };
template <> struct NibbleChunk<ZeroAsMin<FmtQ40>> { static constexpr constant bool value = true; };
template <> struct NibbleChunk<FmtAffine<4, 1>> { static constexpr constant bool value = true; };
template <> struct NibbleChunk<FmtQ5K> { static constexpr constant bool value = true; };
template <> struct NibbleChunk<FmtQ51> { static constexpr constant bool value = true; };
template <> struct NibbleChunk<ZeroAsMin<FmtQ50>> { static constexpr constant bool value = true; };
// Q6_K and Q3_K, whose F32 chains per 16 inputs take nibble6_operand and q3_operand, (code - zero) / Scale,
// unseeded.
template <class F> struct CenteredChunk { static constexpr constant bool value = false; };
template <> struct CenteredChunk<FmtQ6K> {
  static constexpr constant bool value = true;
  static constexpr constant float Scale = 64;
  static half2 operand(uint2 chunk, uint f) { return nibble6_operand(chunk, f); }
};
template <> struct CenteredChunk<FmtQ3K> {
  static constexpr constant bool value = true;
  static constexpr constant float Scale = 8;
  static half2 operand(uint2 chunk, uint f) { return q3_operand(chunk, f); }
};

// The exact bf16 operand of pair f of a chunk.
template <class F>
inline bfloat2 operand(typename F::Chunk ch, uint f, threadgroup const bfloat2 *lut) {
  typedef Shape<F> S;
  if constexpr (F::Kind == QuantLinear && S::Wide) {
    const uint pair = F::codes(ch)[f];
    return bfloat2(float2(pair & 0xFFFFu, pair >> 16));   // exact: bf16 holds every integer to 256
  } else if constexpr (F::Kind == QuantLinear) {
    return as_type<bfloat2>(F::codes(ch)[f] + S::Operand * 0x00010001u);   // Operand in both halves
  } else if constexpr (F::Kind == QuantCodebook) {
    return lut[(F::indices(ch) >> (8 * f)) & 0xFFu];
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 v = F::values(ch);
    return bfloat2(float2(as_type<char2>(ushort((f < 2 ? v.x : v.y) >> (16 * (f & 1))))));
  } else if constexpr (F::Kind == QuantFloat8) {   // exact: an E4M3 value has four significant bits
    const uint2 v = F::values(ch);
    const uint pair = ((f < 2 ? v.x : v.y) >> (16 * (f & 1))) & 0xFFFFu;
    return bfloat2(float2(quant_e4m3_pair(quant_byte_pairs(pair).x)));
  } else if constexpr (F::Kind == QuantFloat16) {   // the bf16 values themselves (quant_register_format)
    static_assert(is_same_v<typename F::Value, bfloat>, "the register tile takes bf16 values only");
    return as_type<bfloat2>(F::values(ch)[f]);
  } else {
    const uint2 g = F::grid(ch);
    const uchar4 m = as_type<uchar4>(f < 2 ? g.x : g.y);
    const uint s = F::signs(ch) >> (2 * f);
    const float2 v = float2(m[2 * (f & 1)], m[2 * (f & 1) + 1]);
    return bfloat2(select(v, -v, bool2(s & 1u, s & 2u)));
  }
}

// What coefficient j of a column in coefficient unit u is decoded from: its
// meta unit and, with ScaleInChunk, chunk 0 of its first group (which holds
// the scale).
template <class F> struct CoefSource {
  typename F::Meta meta;
  typename F::Chunk chunk;
};
template <class F, bool F32 = false>
inline CoefSource<F> coefficient_source(device uchar *w0, device uchar *w1, device uchar *meta, uint plane_tile,
                                        uint groups, uint column, uint u, uint j) {
  typedef Shape<F, F32> S;
  const uint g = u * 2 * S::UnitSpans + j * S::CPG / S::CG;   // its first chain's 32-input group
  const uint units = groups / F::MetaGroups;
  CoefSource<F> src;
  src.meta = F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS + column) * F::MetaBytes);
  if constexpr (F::ScaleInChunk) {
    const ulong at = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + column;
    src.chunk = F::loadChunk(w0 + at * F::P0, w1 + at * F::P1, 0);
  }
  return src;
}
// Coefficient j of a column in coefficient unit u, that of its first chain c =
// j CPG: 32-input group c / CG of the unit, and of 16-input coefficients half
// h = c % CG. Formats with a min return (s, m - 128 s), (s, m) with wide
// codes, or with a seeded min (s, m / -160), whose product with the seed is m
// times the sum of the inputs.
template <class F, bool F32 = false> inline Coef<F> coefficient(CoefSource<F> src, uint u, uint j) {
  typedef Shape<F, F32> S;
  const uint g = u * 2 * S::UnitSpans + j * S::CPG / S::CG, h = j * S::CPG % S::CG;
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(src.meta, src.chunk);
  else k = F::coef(src.meta, ushort(g % F::MetaGroups));
  const float s = h ? k.s.y : k.s.x, m = h ? k.m.y : k.m.x;
  if constexpr (S::HasMin && S::Seeded) return float2(s, m * (-1.0f / kZeroPointOffset));
  else if constexpr (S::HasMin && S::Wide) return float2(s, m);
  else if constexpr (S::HasMin) return float2(s, fma(-128.0f, s, m));
  else return s;
}

// One threadgroup: 4 simdgroups x GGUF_REGISTER_COLUMNS = GGUF_TILE_COLUMNS columns of one segment,
// L request lanes of eight rows, one K partition (tg.y) of `splits`, into a destination of type Out; F32 widens the
// operands for Apple7/8.
template <class F, uint L, GgufEpilogue Ep, bool F32 = false, class Out>
inline void decode(device const TableElem<F32> *table, device const float *sums, device uchar *w0,
                   device uchar *w1, device uchar *meta, device Out *out,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *aux, const GgufDecodeParams p, uint2 tg, uint tid, uint sg,
                   uint lane, threadgroup const LutPair<F, F32> *lut, threadgroup Coef<F> *coefs,
                   threadgroup uint *arrival) {
  typedef Shape<F, F32> S;
  typedef Coef<F> C;
  constexpr uint FG = 4 / S::CG;  // fragments per chain
  const uint K = p.input_size, spans = K / GGUF_TABLE16_SPAN_INPUTS, groups = K / 32, splits = p.splits;
  const ulong tileSums = table16_sums_per_tile(K);
  const uint units = spans / S::UnitSpans;
  const uint u0 = tg.y * units / splits, u1 = (tg.y + 1) * units / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * GGUF_TILE_COLUMNS + sg * GGUF_REGISTER_COLUMNS, plane_tile = base / QUANT_TILE_ROWS,
             plane_row = base % QUANT_TILE_ROWS;
  constexpr uint NC = GGUF_REGISTER_COLUMNS;
  threadgroup C *cu = coefs + sg * NC * S::J;

  // Weight streams: lane c reads chunk c of each group; a span's two groups
  // of one column are a plane tile of payloads apart.
  const uint first = u0 * S::UnitSpans;
  device uchar *p0[2], *p1[2];
#pragma unroll
  for (uint nf = 0; nf < 2; ++nf) {
    const ulong at = (ulong(plane_tile) * groups + 2 * first) * QUANT_TILE_ROWS + plane_row + nf * 8 + fm;
    p0[nf] = w0 + at * F::P0;
    p1[nf] = w1 + at * F::P1;
  }
  // Apple7/8 load each span's chunks as it starts: without the next span's in
  // flight more simdgroups fit a core, 3-5% faster at one to four lanes on
  // an M1 Max.
  typename F::Chunk cur[2][2], nxt[2][2];
  const auto load = [&](thread typename F::Chunk (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      ch[nf][0] = F::loadChunk(p0[nf], p1[nf], c);
      ch[nf][1] = F::loadChunk(p0[nf] + QUANT_TILE_ROWS * F::P0, p1[nf] + QUANT_TILE_ROWS * F::P1, c);
    }
  };
  typedef TableElem<F32> T;
  device const vec<T, 8> *xt =
      reinterpret_cast<device const vec<T, 8> *>(table + ulong(first) * GGUF_TABLE16_SPAN_VALUES);
  device const float *seeds = sums + ulong(first) * GGUF_TABLE16_SPAN_SEEDS;
  device const float *s32 = sums + table16_sums32_offset(K) + ulong(first) * GGUF_TABLE16_SPAN_SUMS;

  float2 acc[L][2];
#pragma unroll
  for (uint r = 0; r < L; ++r) acc[r][0] = acc[r][1] = float2(0);
  load(cur);
  for (uint u = u0; u < u1; ++u) {
    // This unit's coefficients for the simdgroup's NC columns, each decoded
    // once. Every source load is issued before the first decode waits for one,
    // which takes 1-8% off the zero-point formats of 16-input scales (16
    // coefficients per column and unit) at one to four lanes on the 40-core
    // M3 Max.
    // The F32 span chains read one coefficient per span, every other one of the unit's, and decode those alone: on an
    // M2 Max, 2.4-2.8% off MLX 4-bit's 27B projections at one lane and 0.7-1.6% at two to four.
    constexpr uint JS = F32 && S::SpanChain ? 2 : 1;
    constexpr uint I = (NC * S::J / JS + 31) / 32;
    CoefSource<F> src[I];
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J / JS)
        src[i] = coefficient_source<F, F32>(w0, w1, meta, plane_tile, groups, plane_row + e % NC, u, JS * (e / NC));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J / JS) {
        C k = coefficient<F, F32>(src[i], u, JS * (e / NC));
        // The F32 per-32 nibble chains' epilogue factors, 64 s and m - 64 s (code = 64 (operand - 1)), once per
        // coefficient rather than per chain: on an M2 Max, 1-2% off Q4_K's, Q4_0's and Q5_K's 27B projections at one to
        // three lanes (Q4_K and Q4_0 0.3% more at four).
        if constexpr (F32 && NibbleChunk<F>::value) k = float2(64 * k.x, fma(64.0f, k.x, k.y));
        // The centered chains' Scale s.
        if constexpr (F32 && CenteredChunk<F>::value) k = CenteredChunk<F>::Scale * k;
        if constexpr (F32) cu[JS * (e / NC) * NC + e % NC] = k;
        else cu[e] = k;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint us = 0; us < S::UnitSpans; ++us) {
      const bool more = us + 1 < S::UnitSpans || u + 1 < u1;
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        p0[nf] += 2 * QUANT_TILE_ROWS * F::P0;
        p1[nf] += 2 * QUANT_TILE_ROWS * F::P1;
      }
      if constexpr (!F32) {
        if (more) load(nxt);
      }
      if constexpr (F32 && S::SpanChain) {
        float2 dot[L][2];
#pragma unroll
        for (uint r = 0; r < L; ++r) dot[r][0] = dot[r][1] = float2(0);
#pragma unroll
        for (uint q = 0; q < 2; ++q) {
          vec<OperandType<F, F32>, 2> a[2][4];
#pragma unroll
          for (uint nf = 0; nf < 2; ++nf)
#pragma unroll
            for (uint f = 0; f < 4; ++f) {
              if constexpr (NibbleSpan<F>::value) a[nf][f] = nibble_operand(cur[nf][q], f);
              else a[nf][f] = operand_f32<F>(cur[nf][q], f, lut);
            }
#pragma unroll
          for (uint r = 0; r < L; ++r) {
            const vec<T, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
#pragma unroll
            for (uint f = 0; f < 4; ++f) {
              const float2 b = reinterpret_cast<thread const float2 *>(&bq)[f];
#pragma unroll
              for (uint nf = 0; nf < 2; ++nf) sgmatrix::mma_mixed(dot[r][nf], a[nf][f], b);
            }
          }
        }
        C cs[2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cs[nf] = cu[us * 2 * NC + nf * 8 + fm];
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          device const float *rs = s32 + ulong(r) * tileSums + fn;
          const float2 sum = *(device const float2 *)rs + *(device const float2 *)(rs + 8);
#pragma unroll
          for (uint nf = 0; nf < 2; ++nf) {
            if constexpr (NibbleSpan<F>::value) {
              // code = 64 (operand - 1): s code + m = 64 s operand + m - 64 s.
              acc[r][nf] = fma(dot[r][nf], 64 * cs[nf].x, acc[r][nf]);
              acc[r][nf] = fma(sum, fma(64.0f, cs[nf].x, cs[nf].y), acc[r][nf]);
            } else {
              acc[r][nf] = fma(dot[r][nf], cs[nf].x, acc[r][nf]);
              acc[r][nf] = fma(sum, cs[nf].y, acc[r][nf]);
            }
          }
        }
      } else
#pragma unroll
      for (uint q = 0; q < 2; ++q) {
        constexpr bool nibbles = F32 && NibbleChunk<F>::value;
        static_assert(!nibbles || (S::HasMin && !S::Seeded && !S::Wide), "nibble chains close against a min");
        constexpr bool centered = F32 && CenteredChunk<F>::value;
        static_assert(!centered || (!S::HasMin && S::Seeded), "centered chains take a zero point");
        vec<OperandType<F, F32>, 2> a[2][4];
        C cs[S::CG][2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) {
#pragma unroll
          for (uint f = 0; f < 4; ++f) {
            if constexpr (nibbles) a[nf][f] = nibble_operand(cur[nf][q], f);
            else if constexpr (centered) a[nf][f] = CenteredChunk<F>::operand(cur[nf][q], f);
            else if constexpr (F32) a[nf][f] = operand_f32<F>(cur[nf][q], f, lut);
            else a[nf][f] = operand<F>(cur[nf][q], f, lut);
          }
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) cs[h][nf] = cu[((us * 2 + q) * S::CG + h) / S::CPG * NC + nf * 8 + fm];
        }
        // Whether this 32-input group of the table closes a term's inputs, a coefficient group or its span.
        const bool closes = ((us * 2 + q) * S::CG + S::CG) % S::MinChains == 0;
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          const vec<T, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
          vec<InputType<F32>, 2> b[4];   // F32: the activation pairs as the F32 MMA takes them
          if constexpr (F32) {
#pragma unroll
            for (uint f = 0; f < 4; ++f) b[f] = reinterpret_cast<thread const vec<T, 2> *>(&bq)[f];
          }
          float2 sum = float2(0), seed[S::CG];
          if ((S::HasMin || S::Seeded) && S::CG == 1 && closes) {
            // The sum of the term's inputs: this 32-input sum and, closing a term of 64 inputs, the span's other.
            sum = *(device const float2 *)(s32 + ulong(r) * tileSums + q * 8 + fn);
#pragma unroll
            for (uint t = 1; t < S::MinChains; ++t) {
              const int earlier = int(q) - int(t);   // an earlier 32-input group of the span
              sum += *(device const float2 *)(s32 + ulong(r) * tileSums + fn + earlier * 8);
            }
          }
#pragma unroll
          for (uint h = 0; h < S::CG; ++h)   // a 16-input chain's seed from the table, a term's once from its sum
            seed[h] = !S::Seeded || centered ? float2(0)
                      : S::CG == 2 ? *(device const float2 *)(seeds + ulong(r) * tileSums + (2 * q + h) * 8 + fn)
                                   : -float(kZeroPointOffset) * sum;
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) {
#pragma unroll
            for (uint nf = 0; nf < 2; ++nf) {
              float2 dot = seed[h];
#pragma unroll
              for (uint f = h * FG; f < (h + 1) * FG; ++f)
                if constexpr (F32) sgmatrix::mma_mixed(dot, a[nf][f], b[f]);
                else sgmatrix::mma_acc<bfloat>(dot, a[nf][f], reinterpret_cast<thread const bfloat2 *>(&bq)[f]);
              if constexpr (S::HasMin) {
                acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
                if (S::Seeded || closes) acc[r][nf] = fma(S::Seeded ? seed[h] : sum, cs[h][nf].y, acc[r][nf]);
              } else {
                acc[r][nf] = fma(dot, float(cs[h][nf]), acc[r][nf]);
              }
            }
          }
        }
      }
      xt += GGUF_TABLE16_SPAN_VALUES / 8;
      seeds += GGUF_TABLE16_SPAN_SEEDS;
      s32 += GGUF_TABLE16_SPAN_SUMS;
      if constexpr (F32) {
        if (more) load(cur);
      } else if (more) {
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cur[nf][0] = nxt[nf][0], cur[nf][1] = nxt[nf][1];
      }
    }
  }
  if (splits > 1) {
    // Partials [lane][split][8 rows][destination columns], one counter per
    // tile of destination columns: the segments of a projection never share them.
    const uint stride = p.out_stride;
    device atomic_uint *counter = counters + (p.out_offset + base) / GGUF_TILE_COLUMNS;
#pragma unroll
    for (uint r = 0; r < L; ++r)
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        device coherent(device) float *slot =
            partials + ((ulong(r) * splits + tg.y) * 8 + fn) * stride + p.out_offset + base + nf * 8 + fm;
        slot[0] = acc[r][nf].x;
        slot[stride] = acc[r][nf].y;
      }
    if (!split_arrive_last(counter, splits, tid, arrival)) return;
#pragma unroll
    for (uint r = 0; r < L; ++r)
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf)
        acc[r][nf] = split_sum(acc[r][nf], tg.y, splits, [&](uint s) {
          device coherent(device) float *slot =
              partials + ((ulong(r) * splits + s) * 8 + fn) * stride + p.out_offset + base + nf * 8 + fm;
          return float2(slot[0], slot[stride]);
        });
    split_release(counter, tid);
  }
#pragma unroll
  for (uint r = 0; r < L; ++r)
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint column = p.out_offset + base + nf * 8 + fm;
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        const ulong at = ulong(r * 8 + fn + i) * p.out_stride + column;
        out[at] = gguf_epilogue<Ep, Out>(acc[r][nf][i], aux, at);
      }
    }
}

// Gate and up in one pass: grid {columns / 32, splits}, each simdgroup's fragment nf=0 is 8 gate columns,
// nf=1 the same 8 up columns. All 4 simdgroups compute SwiGLU in registers and store out with no threadgroup barrier.
template <class F, uint L, bool F32, class Out>
inline void decode_fragment_paired(device const TableElem<F32> *table, device const float *sums,
                                   device uchar *g0, device uchar *g1, device uchar *gm,
                                   device uchar *u0, device uchar *u1, device uchar *um,
                                   device Out *out, device coherent(device) float *partials,
                                   device atomic_uint *counters, const GgufDecodeParams p,
                                   uint2 tg, uint tid, uint sg, uint lane,
                                   threadgroup const LutPair<F, F32> *lut,
                                   threadgroup Coef<F> *coefs, threadgroup uint *arrival) {
  typedef Shape<F, F32> S;
  typedef Coef<F> C;
  constexpr uint FG = 4 / S::CG;  // fragments per coefficient group
  const uint K = p.input_size, spans = K / GGUF_TABLE16_SPAN_INPUTS, groups = K / 32, splits = p.splits;
  const ulong tileSums = table16_sums_per_tile(K);
  const uint units = spans / S::UnitSpans;
  const uint u0_split = tg.y * units / splits, u1_split = (tg.y + 1) * units / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint col_base = tg.x * 32 + sg * 8;
  const uint plane_tile = col_base / QUANT_TILE_ROWS, plane_row = col_base % QUANT_TILE_ROWS;
  constexpr uint NC = GGUF_REGISTER_COLUMNS;
  threadgroup C *cu = coefs + sg * NC * S::J;

  // Weight streams: fragment 0 reads gate, fragment 1 reads up.
  const uint first = u0_split * S::UnitSpans;
  const ulong at = (ulong(plane_tile) * groups + 2 * first) * QUANT_TILE_ROWS + plane_row + fm;
  device uchar *p0[2] = {g0 + at * F::P0, u0 + at * F::P0};
  device uchar *p1[2] = {g1 + at * F::P1, u1 + at * F::P1};

  typename F::Chunk cur[2][2], nxt[2][2];
  const auto load = [&](thread typename F::Chunk (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      ch[nf][0] = F::loadChunk(p0[nf], p1[nf], c);
      ch[nf][1] = F::loadChunk(p0[nf] + QUANT_TILE_ROWS * F::P0, p1[nf] + QUANT_TILE_ROWS * F::P1, c);
    }
  };
  typedef TableElem<F32> T;
  device const vec<T, 8> *xt = reinterpret_cast<device const vec<T, 8> *>(table + ulong(first) * GGUF_TABLE16_SPAN_VALUES);
  device const float *seeds = sums + ulong(first) * GGUF_TABLE16_SPAN_SEEDS;
  device const float *s32 = sums + table16_sums32_offset(K) + ulong(first) * GGUF_TABLE16_SPAN_SUMS;

  float2 acc[L][2];
#pragma unroll
  for (uint r = 0; r < L; ++r) acc[r][0] = acc[r][1] = float2(0);
  load(cur);
  for (uint u = u0_split; u < u1_split; ++u) {
    constexpr uint JS = F32 && S::SpanChain ? 2 : 1;
    constexpr uint I = (NC * S::J / JS + 31) / 32;
    CoefSource<F> src[I];
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J / JS) {
        const bool is_up = (e % NC) >= 8;
        device uchar *mw0 = is_up ? u0 : g0;
        device uchar *mw1 = is_up ? u1 : g1;
        device uchar *mm = is_up ? um : gm;
        const uint col = plane_row + (is_up ? (e % NC - 8) : (e % NC));
        src[i] = coefficient_source<F, F32>(mw0, mw1, mm, plane_tile, groups, col, u, JS * (e / NC));
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J / JS) {
        C k = coefficient<F, F32>(src[i], u, JS * (e / NC));
        if constexpr (F32 && NibbleChunk<F>::value) k = float2(64 * k.x, fma(64.0f, k.x, k.y));
        if constexpr (F32 && CenteredChunk<F>::value) k = CenteredChunk<F>::Scale * k;
        cu[JS * (e / NC) * NC + e % NC] = k;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint us = 0; us < S::UnitSpans; ++us) {
      const bool more = us + 1 < S::UnitSpans || u + 1 < u1_split;
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        p0[nf] += 2 * QUANT_TILE_ROWS * F::P0;
        p1[nf] += 2 * QUANT_TILE_ROWS * F::P1;
      }
      if constexpr (!F32) {
        if (more) load(nxt);
      }
      if constexpr (F32 && S::SpanChain) {
        float2 dot[L][2];
#pragma unroll
        for (uint r = 0; r < L; ++r) dot[r][0] = dot[r][1] = float2(0);
#pragma unroll
        for (uint q = 0; q < 2; ++q) {
          vec<OperandType<F, F32>, 2> a[2][4];
#pragma unroll
          for (uint nf = 0; nf < 2; ++nf)
#pragma unroll
            for (uint f = 0; f < 4; ++f) {
              if constexpr (NibbleSpan<F>::value) a[nf][f] = nibble_operand(cur[nf][q], f);
              else a[nf][f] = operand_f32<F>(cur[nf][q], f, lut);
            }
#pragma unroll
          for (uint r = 0; r < L; ++r) {
            const vec<T, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
#pragma unroll
            for (uint f = 0; f < 4; ++f) {
              const float2 b = reinterpret_cast<thread const float2 *>(&bq)[f];
#pragma unroll
              for (uint nf = 0; nf < 2; ++nf) sgmatrix::mma_mixed(dot[r][nf], a[nf][f], b);
            }
          }
        }
        C cs[2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cs[nf] = cu[us * 2 * NC + nf * 8 + fm];
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          device const float *rs = s32 + ulong(r) * tileSums + fn;
          const float2 sum = *(device const float2 *)rs + *(device const float2 *)(rs + 8);
#pragma unroll
          for (uint nf = 0; nf < 2; ++nf) {
            if constexpr (NibbleSpan<F>::value) {
              acc[r][nf] = fma(dot[r][nf], 64 * cs[nf].x, acc[r][nf]);
              acc[r][nf] = fma(sum, fma(64.0f, cs[nf].x, cs[nf].y), acc[r][nf]);
            } else {
              acc[r][nf] = fma(dot[r][nf], cs[nf].x, acc[r][nf]);
              acc[r][nf] = fma(sum, cs[nf].y, acc[r][nf]);
            }
          }
        }
      } else
#pragma unroll
      for (uint q = 0; q < 2; ++q) {
        constexpr bool nibbles = F32 && NibbleChunk<F>::value;
        static_assert(!nibbles || (S::HasMin && !S::Seeded && !S::Wide), "nibble chains close against a min");
        constexpr bool centered = F32 && CenteredChunk<F>::value;
        static_assert(!centered || (!S::HasMin && S::Seeded), "centered chains take a zero point");
        vec<OperandType<F, F32>, 2> a[2][4];
        C cs[S::CG][2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) {
#pragma unroll
          for (uint f = 0; f < 4; ++f) {
            if constexpr (nibbles) a[nf][f] = nibble_operand(cur[nf][q], f);
            else if constexpr (centered) a[nf][f] = CenteredChunk<F>::operand(cur[nf][q], f);
            else if constexpr (F32) a[nf][f] = operand_f32<F>(cur[nf][q], f, lut);
            else a[nf][f] = operand<F>(cur[nf][q], f, lut);
          }
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) cs[h][nf] = cu[((us * 2 + q) * S::CG + h) * NC + nf * 8 + fm];
        }
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          const vec<T, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
          vec<InputType<F32>, 2> b[4];
#pragma unroll
          for (uint f = 0; f < 4; ++f) b[f] = reinterpret_cast<thread const vec<T, 2> *>(&bq)[f];
          float2 sum = float2(0), seed[S::CG];
          if (S::HasMin && !S::Seeded) sum = *(device const float2 *)(s32 + ulong(r) * tileSums + q * 8 + fn);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h)
            seed[h] = S::Seeded && !centered ? *(device const float2 *)(seeds + ulong(r) * tileSums + (2 * q + h) * 8 + fn)
                                : float2(0);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) {
#pragma unroll
            for (uint nf = 0; nf < 2; ++nf) {
              float2 dot = seed[h];
#pragma unroll
              for (uint f = h * FG; f < (h + 1) * FG; ++f)
                sgmatrix::mma_mixed(dot, a[nf][f], b[f]);
              if constexpr (nibbles) {
                acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
                acc[r][nf] = fma(sum, cs[h][nf].y, acc[r][nf]);
              } else if constexpr (S::HasMin) {
                acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
                acc[r][nf] = fma(S::Seeded ? seed[h] : sum, cs[h][nf].y, acc[r][nf]);
              } else if constexpr (centered) {
                acc[r][nf] = fma(dot, float(cs[h][nf]), acc[r][nf]);
              } else {
                acc[r][nf] = fma(dot, float(cs[h][nf]), acc[r][nf]);
              }
            }
          }
        }
      }
      xt += GGUF_TABLE16_SPAN_VALUES / 8;
      seeds += GGUF_TABLE16_SPAN_SEEDS;
      s32 += GGUF_TABLE16_SPAN_SUMS;
      if constexpr (F32) {
        if (more) load(cur);
      } else if (more) {
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cur[nf][0] = nxt[nf][0], cur[nf][1] = nxt[nf][1];
      }
    }
  }

  // Cross-split reduction and SwiGLU in registers:
  const uint stride = p.out_stride, slots = 2 * splits;
  const uint column = p.out_offset + col_base + fm;
  device atomic_uint *counter = counters + (p.out_offset + tg.x * 32) / 32;
  const auto slot = [&](uint r, uint s) {
    return partials + ((ulong(r) * slots + s) * 8 + fn) * stride + column;
  };
  if (splits > 1) {
#pragma unroll
    for (uint r = 0; r < L; ++r) {
      device coherent(device) float *at_g = slot(r, tg.y);
      at_g[0] = acc[r][0].x;
      at_g[stride] = acc[r][0].y;
      device coherent(device) float *at_u = slot(r, splits + tg.y);
      at_u[0] = acc[r][1].x;
      at_u[stride] = acc[r][1].y;
    }
    if (!split_arrive_last(counter, splits, tid, arrival)) return;
#pragma unroll
    for (uint r = 0; r < L; ++r) {
      acc[r][0] = split_sum(acc[r][0], tg.y, splits, [&](uint s) {
        device coherent(device) float *at = slot(r, s);
        return float2(at[0], at[stride]);
      });
      acc[r][1] = split_sum(acc[r][1], tg.y, splits, [&](uint s) {
        device coherent(device) float *at = slot(r, splits + s);
        return float2(at[0], at[stride]);
      });
    }
    split_release(counter, tid);
  }
#pragma unroll
  for (uint r = 0; r < L; ++r) {
    const float2 gate = float2(float(bfloat(acc[r][0].x)), float(bfloat(acc[r][0].y)));
    const float2 up = float2(float(bfloat(acc[r][1].x)), float(bfloat(acc[r][1].y)));
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const ulong at_out = ulong(r * 8 + fn + i) * p.out_stride + column;
      out[at_out] = Out(up[i] * splash_silu(gate[i]));
    }
  }
}

} // namespace gguf_sg

// Table16 for `lanes` eight-row tiles of `width` inputs: one simdgroup per (span, row).
kernel void decode_linear_gguf_prepare(device const bfloat *input [[buffer(0)]],
                                       device bfloat *table [[buffer(1)]], device float *sums [[buffer(2)]],
                                       constant uint &width [[buffer(3)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  const uint span = (tg.x * 4 + sg) / 8, row = (tg.x * 4 + sg) % 8;
  input += ulong(tg.y) * width * 8;
  const uint k = span * GGUF_TABLE16_SPAN_INPUTS + 2 * lane;
  gguf_sg::write_input(table + ulong(tg.y) * width * 8, sums + ulong(tg.y) * table16_sums_per_tile(width), width,
                     span, row, lane, input[row * width + k], input[row * width + k + 1]);
}
// Table16F, the fp32 table of Apple7/8, as above.
inline void prepare_table_f(device const bfloat *input, device float *table, device float *sums, uint width, uint2 tg,
                            uint sg, uint lane) {
  const uint span = (tg.x * 4 + sg) / 8, row = (tg.x * 4 + sg) % 8;
  input += ulong(tg.y) * width * 8;
  const uint k = span * GGUF_TABLE16_SPAN_INPUTS + 2 * lane;
  gguf_sg::write_input(table + ulong(tg.y) * width * 8, sums + ulong(tg.y) * table16_sums_per_tile(width), width,
                       span, row, lane, input[row * width + k], input[row * width + k + 1]);
}
kernel void decode_linear_gguf_prepare_f(device const bfloat *input [[buffer(0)]], device float *table [[buffer(1)]],
                                         device float *sums [[buffer(2)]], constant uint &width [[buffer(3)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
  prepare_table_f(input, table, sums, width, tg, sg, lane);
}

#define GGUF_SG_KERNEL(Name, F, L, EP, Out)                                                                   \
  kernel void Name(device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],       \
                   device uchar *w0 [[buffer(2)]], device uchar *w1 [[buffer(3)]],                          \
                   device uchar *meta [[buffer(4)]], device Out *out [[buffer(5)]],                         \
                   device coherent(device) float *partials [[buffer(6)]],                                   \
                   device atomic_uint *counters [[buffer(7)]], device const bfloat *aux [[buffer(8)]],      \
                   constant GgufDecodeParams &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],       \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
                   uint lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup bfloat2 lut[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];                         \
    threadgroup gguf_sg::Coef<F> coefs[gguf_sg::kCoefs<F>];                                                 \
    threadgroup uint arrival;                                                                               \
    quant_pair_table<F>(lut, tid, GGUF_REGISTER_THREADS);                                                                              \
    gguf_sg::decode<F, L, EP>(table, sums, w0, w1, meta, out, partials, counters, aux, p, tg, tid, sg, lane, \
                              lut, coefs, &arrival);                                                        \
  }
// Each epilogue into bf16, and the plain one also into fp32 (_a_f32: the logits, ops::Projection::destination).
#define GGUF_SG_EPILOGUES(F, f, L)                                                                            \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_a, F, L, EpNone, bfloat)                                \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_a_f32, F, L, EpNone, float)                             \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_r, F, L, EpResidual, bfloat)                            \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_g, F, L, EpUpWithGate, bfloat)
#define GGUF_SG_FORMAT(F, f) \
  GGUF_SG_EPILOGUES(F, f, 1) GGUF_SG_EPILOGUES(F, f, 2) GGUF_SG_EPILOGUES(F, f, 3) GGUF_SG_EPILOGUES(F, f, 4)
QUANT_REGISTER_FORMATS(GGUF_SG_FORMAT)
#undef GGUF_SG_FORMAT
#undef GGUF_SG_EPILOGUES
#undef GGUF_SG_KERNEL

// Apple7/8's F32 instances (gguf_decode_sgf_*) of each format in F32Format's view, with each epilogue as above, and
// their one-pass gate/up (_gu): grid {columns / 32, splits}, each simdgroup's fragment nf=0 is 8 gate columns, nf=1
// the same 8 up columns.
#define GGUF_SGF_KERNEL(Name, F, L, EP, Out)                                                                    \
  kernel void Name(device const float *table [[buffer(0)]], device const float *sums [[buffer(1)]],             \
                   device uchar *w0 [[buffer(2)]], device uchar *w1 [[buffer(3)]],                              \
                   device uchar *meta [[buffer(4)]], device Out *out [[buffer(5)]],                             \
                   device coherent(device) float *partials [[buffer(6)]],                                       \
                   device atomic_uint *counters [[buffer(7)]], device const bfloat *aux [[buffer(8)]],          \
                   constant GgufDecodeParams &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],       \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],        \
                   uint lane [[thread_index_in_simdgroup]]) {                                                   \
    threadgroup gguf_sg::LutPair<F, true> lut[gguf_sg::kLutEntries<F, true>];                                   \
    threadgroup gguf_sg::Coef<F> coefs[gguf_sg::kCoefs<F, true>];                                               \
    threadgroup uint arrival;                                                                                   \
    gguf_sg::lut_table<F, true>(lut, tid, GGUF_REGISTER_THREADS);                                               \
    gguf_sg::decode<F, L, EP, true>(table, sums, w0, w1, meta, out, partials, counters, aux, p, tg, tid, sg,    \
                                    lane, lut, coefs, &arrival);                                                \
  }
#define GGUF_SGF_GATE_UP_KERNEL(Name, F, L)                                                                     \
  kernel void Name(device const float *table [[buffer(0)]], device const float *sums [[buffer(1)]],             \
                   device uchar *g0 [[buffer(2)]], device uchar *g1 [[buffer(3)]], device uchar *gm [[buffer(4)]], \
                   device uchar *u0 [[buffer(5)]], device uchar *u1 [[buffer(6)]], device uchar *um [[buffer(7)]], \
                   device bfloat *out [[buffer(8)]], device coherent(device) float *partials [[buffer(9)]],     \
                   device atomic_uint *counters [[buffer(10)]], constant GgufDecodeParams &p [[buffer(11)]],    \
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],         \
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {       \
    threadgroup gguf_sg::LutPair<F, true> lut[gguf_sg::kLutEntries<F, true>];                                   \
    threadgroup gguf_sg::Coef<F> coefs[gguf_sg::kCoefs<F, true>];                                               \
    threadgroup uint arrival;                                                                                   \
    gguf_sg::lut_table<F, true>(lut, tid, GGUF_REGISTER_THREADS);                                               \
    gguf_sg::decode_fragment_paired<F, L, true, bfloat>(table, sums, g0, g1, gm, u0, u1, um, out, partials,     \
                                                        counters, p, tg, tid, sg, lane, lut, coefs, &arrival);  \
  }
#define GGUF_SGF_EPILOGUES(F, f, L)                                                                             \
  GGUF_SGF_KERNEL(gguf_decode_sgf_##f##_l##L##_a, F, L, EpNone, bfloat)                                         \
  GGUF_SGF_KERNEL(gguf_decode_sgf_##f##_l##L##_a_f32, F, L, EpNone, float)                                      \
  GGUF_SGF_KERNEL(gguf_decode_sgf_##f##_l##L##_r, F, L, EpResidual, bfloat)                                     \
  GGUF_SGF_KERNEL(gguf_decode_sgf_##f##_l##L##_g, F, L, EpUpWithGate, bfloat)
#define GGUF_SGF_LANES(F, f)                                                                                    \
  GGUF_SGF_EPILOGUES(F, f, 1) GGUF_SGF_EPILOGUES(F, f, 2) GGUF_SGF_EPILOGUES(F, f, 3) GGUF_SGF_EPILOGUES(F, f, 4) \
  GGUF_SGF_GATE_UP_KERNEL(gguf_decode_sgf_##f##_l1_gu, F, 1) GGUF_SGF_GATE_UP_KERNEL(gguf_decode_sgf_##f##_l2_gu, F, 2) \
  GGUF_SGF_GATE_UP_KERNEL(gguf_decode_sgf_##f##_l3_gu, F, 3) GGUF_SGF_GATE_UP_KERNEL(gguf_decode_sgf_##f##_l4_gu, F, 4)
#define GGUF_SGF_FORMAT(F, f) GGUF_SGF_LANES(gguf_sg::F32Format<F>::type, f)
QUANT_REGISTER_FORMATS(GGUF_SGF_FORMAT)
#undef GGUF_SGF_FORMAT
#undef GGUF_SGF_LANES
#undef GGUF_SGF_EPILOGUES
#undef GGUF_SGF_GATE_UP_KERNEL
#undef GGUF_SGF_KERNEL

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any
// formats in one dispatch, so the small segments do not run as dispatches of
// their own. The threadgroup's tile picks its segment, and the segment's
// format picks the decode; every segment takes the same K splits.
#define GGUF_SG_SEGMENT(i, w0, w1, m) \
  device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
template <uint L>
inline void gguf_sg_fused(device const bfloat *table, device const float *sums, device uchar *w0a, device uchar *w1a,
                          device uchar *ma, device uchar *w0b, device uchar *w1b, device uchar *mb, device uchar *w0c,
                          device uchar *w1c, device uchar *mc, device bfloat *out, device coherent(device) float *partials,
                          device atomic_uint *counters, constant GgufDecodeFusedParams &p, uint2 tg, uint tid, uint sg,
                          uint lane, threadgroup bfloat2 *lut, threadgroup float2 *coefs, threadgroup uint &arrival) {
  const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;
  const uint s = tg.x < t0 ? 0 : tg.x < t1 ? 1 : 2;
  device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;
  device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;
  device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;
  const GgufDecodeParams q{p.input_size, p.splits, p.out_stride, p.offset[s], 0};
  const uint2 local(tg.x - (s == 0 ? 0 : s == 1 ? t0 : t1), tg.y);
  quant_register_format_switch(p.fmt[s], [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(lut, tid, GGUF_REGISTER_THREADS);
    gguf_sg::decode<F, L, EpNone>(table, sums, w0, w1, meta, out, partials, counters, out, q, local,
                                              tid, sg, lane, lut,
                                              reinterpret_cast<threadgroup gguf_sg::Coef<F> *>(coefs), &arrival);
  });
}
#define GGUF_SG_FUSED(L)                                                                                          \
  kernel void gguf_decode_sg_fused_l##L(                                                                   \
      device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],                          \
      GGUF_SG_SEGMENT(2, w0a, w1a, ma), GGUF_SG_SEGMENT(5, w0b, w1b, mb), GGUF_SG_SEGMENT(8, w0c, w1c, mc),       \
      device bfloat *out [[buffer(11)]], device coherent(device) float *partials [[buffer(12)]],                  \
      device atomic_uint *counters [[buffer(13)]], constant GgufDecodeFusedParams &p [[buffer(14)]],                  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                       \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                     \
    threadgroup bfloat2 lut[kQuantPairTableEntries];                                                              \
    threadgroup float2 coefs[gguf_sg::kCoefFloat2s];                                                              \
    threadgroup uint arrival;                                                                                     \
    gguf_sg_fused<L>(table, sums, w0a, w1a, ma, w0b, w1b, mb, w0c, w1c, mc, out, partials, counters, p, tg, tid, sg, \
                     lane, lut, coefs, arrival);                                                                  \
  }
GGUF_SG_FUSED(1)
GGUF_SG_FUSED(2)
GGUF_SG_FUSED(3)
GGUF_SG_FUSED(4)
#undef GGUF_SG_FUSED
#undef GGUF_SG_SEGMENT

// MoE experts (ops/MoE.cpp): threadgroup (x, y) computes 64 columns of grouped 8-row tile y from its Table16 tile
// (kernels/shared/moe.metal) with the weights of the tile's expert (moe_gguf_segment) of a block whose shared expert
// is Shared slices (each instance's slices suffix, ops::kCompiledMoeRoutings), in the format the tile picks
// at run time: on the 40-core M3 Max one run-time-format kernel is within +1.6% of the per-format kernels
// (23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). No K splits, so no partials or counters.
// aux is the gate of the up pass. Grid (column tiles, expert tiles), GGUF_REGISTER_THREADS threads.
template <GgufEpilogue Ep, uint Shared>
inline void gguf_sg_expert(device const bfloat *table, device const float *sums, device const MoeTileDescriptor *tiles,
                           device const uint *tile_count, device uchar *w0, device uchar *w1, device uchar *meta,
                           device uchar *sw0, device uchar *sw1, device uchar *smeta, device bfloat *out,
                           device const bfloat *aux, device const uint *slots, constant MoeGgufExpertParams &p,
                           uint2 tg, uint tid, uint sg, uint lane, threadgroup bfloat2 *lut, threadgroup float2 *coefs,
                           threadgroup uint *arrival) {
  if (tg.y >= *tile_count) return;
  const MoeGgufSegment s = moe_gguf_segment<Shared>(tiles[tg.y].expert, p, slots, w0, w1, meta, sw0, sw1, smeta);
  const uint K = p.input_size, N = p.output_size;
  const ulong rows = ulong(tg.y) * 8;
  const GgufDecodeParams q{K, 1, N, 0, 0};
  quant_register_format_switch(s.format, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(lut, tid, GGUF_REGISTER_THREADS);
    gguf_sg::decode<F, 1, Ep>(
        table + rows * K, sums + tg.y * table16_sums_per_tile(K), s.w0, s.w1, s.meta, out + rows * N, nullptr, nullptr,
        aux + rows * N, q, uint2(tg.x, 0), tid, sg, lane, lut, reinterpret_cast<threadgroup gguf_sg::Coef<F> *>(coefs),
        arrival);
  });
}
#define GGUF_SG_EXPERT(Name, EP, Shared)                                                                            \
  kernel void Name(device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],              \
                   device const MoeTileDescriptor *tiles [[buffer(2)]], device const uint *tile_count [[buffer(3)]], \
                   device uchar *w0 [[buffer(4)]], device uchar *w1 [[buffer(5)]], device uchar *meta [[buffer(6)]], \
                   device uchar *sw0 [[buffer(7)]], device uchar *sw1 [[buffer(8)]], device uchar *smeta [[buffer(9)]], \
                   device bfloat *out [[buffer(10)]], device const bfloat *aux [[buffer(11)]],                      \
                   device const uint *slots [[buffer(12)]], constant MoeGgufExpertParams &p [[buffer(13)]],         \
                   uint2 tg [[threadgroup_position_in_grid]],                                                       \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],            \
                   uint lane [[thread_index_in_simdgroup]]) {                                                       \
    threadgroup bfloat2 lut[kQuantPairTableEntries];                                                                \
    threadgroup float2 coefs[gguf_sg::kCoefFloat2s];                                                                \
    threadgroup uint arrival;                                                                                       \
    gguf_sg_expert<EP, Shared>(table, sums, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, out, aux, slots, p, tg, \
                               tid, sg, lane, lut, coefs, &arrival);                                                \
  }
GGUF_SG_EXPERT(moe_expert_gguf_sg_a, EpNone, 1)
GGUF_SG_EXPERT(moe_expert_gguf_sg_g, EpUpWithGate, 1)
GGUF_SG_EXPERT(moe_expert_gguf_sg_a_s2, EpNone, 2)
GGUF_SG_EXPERT(moe_expert_gguf_sg_g_s2, EpUpWithGate, 2)
#undef GGUF_SG_EXPERT
