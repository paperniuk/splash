// Inkling 2's own blocks (ops/Inkling.hpp) against fp64 CPU references written
// from the Splash port's spec and the MLX reference: the depth attention over
// a residual's blocks (and, with a loop pass's two bank states and four
// queries, its messages), the message gate, the RMS norm that feeds a
// sublayer, the static and dynamic short convolutions of a prompt chunk, of a
// decode step and of a verify tree, the history a prefill hands to decode,
// the history a verify step commits along its accepted path, and the head's
// softcap with its masked padding rows.
//
// Every operator reads and writes views of rows `stride` values apart from a
// value `offset`: the tests run each with and without that padding and
// require the padding, and every buffer past what an operator may write, to
// keep its poison. Reference values are rounded where the model rounds (the
// bf16 pair and bf16 hidden of the gate, the bf16 convolution output), and
// checks that rest on a rounding are made on inputs the rounding decides
// exactly.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/VerifyTree.h"
#include "ops/Inkling.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Normalization.hpp"

#include "LinearNumerics.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::Inkling;
using splash::ops::InklingConvolution;
using splash::ops::InklingGateWeights;
using splash::ops::InklingRows;
using splash::ops::LayoutRows;
using splash::ops::NormWeights;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::makeNormWeights;
using splash::test::normWeight;
using splash::test::roundedOnceToBf16;
using splash::test::sharedBuffer;
using splash::test::ulpBf16;

constexpr uint32_t kTaps = 4;
constexpr uint32_t kRank = 64;
constexpr uint32_t kHidden = 2304;
constexpr double kEpsilon = SPLASH_RMS_EPSILON;
// The most rows of a verify block a convolution tree holds (one bit of the
// 32-bit ancestor mask a row).
constexpr uint32_t kTreeRows = 32;
// The values a view's padding holds, and a buffer's rows past the last one.
constexpr uint16_t kPoison16 = 0xA5A5;
constexpr uint32_t kPoison32 = 0xA5A5A5A5u;
// Values of padding between a view's rows, and before its first row.
constexpr uint32_t kPad = 8;
constexpr uint32_t kGuard = 16;

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

std::string text(double value) {
  char buffer[48];
  std::snprintf(buffer, sizeof buffer, "%.9g", value);
  return buffer;
}

void near(const std::string &what, uint64_t row, uint64_t column, double got, double expected, double tolerance) {
  if (std::isfinite(got) && std::fabs(got - expected) <= tolerance)
    return;
  fail(what + " row " + std::to_string(row) + " column " + std::to_string(column) + ": got " + text(got) +
       ", expected " + text(expected) + " within " + text(tolerance));
}

// Whether bf16 `got` is `exact` rounded once to bf16, up to the fp32
// arithmetic's noise on terms of total magnitude `magnitude`.
void nearBf16(const std::string &what, uint64_t row, uint64_t column, uint16_t got, double exact,
              double magnitude) {
  const double value = bf16ToFloat(got);
  const double tolerance = 0.5 * ulpBf16(float(exact)) * (1 + 1.0 / 256) + 1e-6 * magnitude;
  near(what, row, column, value, exact, tolerance);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }
  uint32_t below(uint32_t count) { return next() % count; }
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }
  float gauss() {
    float sum = 0.0F;
    for (int i = 0; i < 4; ++i)
      sum += unit();
    return sum * 0.8660254F;
  }

private:
  uint64_t state_;
};

template <class T> constexpr T poison() {
  if constexpr (sizeof(T) == 2)
    return static_cast<T>(kPoison16);
  else
    return std::bit_cast<T>(kPoison32);
}
template <class T> T encode(float value) {
  if constexpr (std::is_same_v<T, uint16_t>)
    return floatToBf16(value);
  else
    return value;
}
template <class T> double decode(T value) {
  if constexpr (std::is_same_v<T, uint16_t>)
    return bf16ToFloat(value);
  else
    return value;
}
template <class T> bool isPoison(const T &value) {
  const T mark = poison<T>();
  return !std::memcmp(&value, &mark, sizeof(T));
}

// `rows` rows of `width` values `stride` apart from value `offset` of a
// poisoned buffer that also ends in guard values; padded views leave kPad
// values before their first row and between rows.
template <class T> class Matrix final {
public:
  Matrix(MetalBackend &backend, uint32_t rows, uint32_t width, bool padded)
      : rows_(rows), width_(width), stride_(width + (padded ? kPad : 0)), offset_(padded ? kPad : 0) {
    count_ = uint64_t{offset_} + uint64_t{rows_} * stride_ + kGuard;
    buffer_ = sharedBuffer(backend, count_ * sizeof(T));
    std::fill_n(data(), count_, poison<T>());
  }
  [[nodiscard]] T *data() const { return static_cast<T *>(buffer_.contents()); }
  [[nodiscard]] T &at(uint32_t row, uint32_t column) const {
    return data()[offset_ + uint64_t{row} * stride_ + column];
  }
  [[nodiscard]] InklingRows view() const { return {buffer_, width_, stride_, offset_}; }
  [[nodiscard]] const MetalBuffer &buffer() const { return buffer_; }
  [[nodiscard]] uint32_t rows() const { return rows_; }

  // Everything outside the rows' windows still holds the poison.
  void requireGuarded(const std::string &what) const {
    for (uint64_t index = 0; index < count_; ++index) {
      bool inside = false;
      if (index >= offset_) {
        const uint64_t relative = index - offset_;
        inside = relative / stride_ < rows_ && relative % stride_ < width_;
      }
      if (!inside && !isPoison(data()[index]))
        fail(what + ": a value outside the view's rows, at index " + std::to_string(index) + ", was written");
    }
  }

private:
  uint32_t rows_, width_, stride_, offset_;
  uint64_t count_ = 0;
  MetalBuffer buffer_;
};

// A bf16 buffer and the content it must hold once an operator ran: inputs
// keep what they were given, outputs hold the poison but for the values the
// reference expects.
class Words final {
public:
  Words(MetalBackend &backend, uint64_t count)
      : buffer_(sharedBuffer(backend, count * 2)), expected_(count, kPoison16) {
    std::fill_n(data(), count, kPoison16);
  }
  [[nodiscard]] uint16_t *data() const { return static_cast<uint16_t *>(buffer_.contents()); }
  [[nodiscard]] const MetalBuffer &buffer() const { return buffer_; }
  void put(uint64_t index, uint16_t value) {
    data()[index] = value;
    expected_[index] = value;
  }
  void expect(uint64_t index, uint16_t value) { expected_[index] = value; }
  void requireMatches(const std::string &what) const {
    for (uint64_t index = 0; index < expected_.size(); ++index)
      if (data()[index] != expected_[index])
        fail(what + ": value " + std::to_string(index) + " is " + std::to_string(data()[index]) + ", expected " +
             std::to_string(expected_[index]));
  }

private:
  MetalBuffer buffer_;
  std::vector<uint16_t> expected_;
};

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}
void run(MetalBackend &backend, CommandGraph &graph, const std::string &what) {
  require(!graph.empty(), what + " encoded nothing");
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
}


MetalBuffer uploaded(MetalBackend &backend, const void *bytes, uint64_t size) {
  MetalBuffer buffer = sharedBuffer(backend, std::max<uint64_t>(size, 8));
  std::memcpy(buffer.contents(), bytes, size);
  return buffer;
}
template <class A, class B, class O>
void sumCase(MetalBackend &backend, Random &random, uint32_t rows, bool padded, bool hasB, bool alias) {
  constexpr uint32_t width = 2304;
  Matrix<A> a(backend, rows, width, padded);
  Matrix<B> b(backend, rows, width, padded);
  Matrix<O> output(backend, rows, width, padded);
  std::vector<O> expected(uint64_t{rows} * width);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < width; ++c) {
      a.at(r, c) = encode<A>(random.gauss());
      b.at(r, c) = encode<B>(random.gauss());
      float value = float(decode(a.at(r, c)));
      if (hasB) value += float(decode(b.at(r, c)));
      expected[uint64_t{r} * width + c] = encode<O>(value);
    }
  }
  InklingRows result = alias ? a.view() : output.view();
  CommandGraph graph;
  Inkling::addSum(graph, a.view(), std::is_same_v<A, float>, hasB ? b.view() : InklingRows{},
                  std::is_same_v<B, float>, result, std::is_same_v<O, float>, width, rows);
  run(backend, graph, "sum");
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < width; ++c) {
      O got;
      if constexpr (std::is_same_v<A, O>) got = alias ? a.at(r, c) : output.at(r, c);
      else got = output.at(r, c);
      require(got == expected[uint64_t{r} * width + c], "sum value mismatch");
    }
  }
  a.requireGuarded("sum a");
  b.requireGuarded("sum b");
  output.requireGuarded("sum output");
}

template <class A, class B, class O>
uint32_t sumTypes(MetalBackend &backend, Random &random) {
  uint32_t count = 0;
  for (uint32_t rows : {1u, 17u, 2048u})
    for (bool padded : {false, true})
      for (bool hasB : {false, true}) {
        sumCase<A, B, O>(backend, random, rows, padded, hasB, false);
        ++count;
        if constexpr (std::is_same_v<A, O>) {
          sumCase<A, B, O>(backend, random, rows, padded, hasB, true);
          ++count;
        }
      }
  return count;
}

uint32_t sums(MetalBackend &backend) {
  Random random(0x51A);
  return sumTypes<uint16_t, uint16_t, uint16_t>(backend, random) +
         sumTypes<uint16_t, uint16_t, float>(backend, random) +
         sumTypes<uint16_t, float, uint16_t>(backend, random) +
         sumTypes<uint16_t, float, float>(backend, random) +
         sumTypes<float, uint16_t, uint16_t>(backend, random) +
         sumTypes<float, uint16_t, float>(backend, random) +
         sumTypes<float, float, uint16_t>(backend, random) +
         sumTypes<float, float, float>(backend, random);
}


// The depth attention ----------------------------------------------------

struct DepthCase final {
  uint32_t hidden, rows, states, queries;
  bool partialFloat, padded;
  double magnitude;
  bool zeroState;
};

void depthMix(MetalBackend &backend, Random &random, const DepthCase &c) {
  const uint32_t hidden = c.hidden, count = c.states + 1;
  const std::string what = "depth mix hidden " + std::to_string(hidden) + " rows " + std::to_string(c.rows) +
                           " states " + std::to_string(c.states) + " queries " + std::to_string(c.queries) +
                           (c.partialFloat ? " f32" : " bf16") + " partial" + (c.padded ? " padded" : "") +
                           " x" + text(c.magnitude) + (c.zeroState ? " zero state" : "");
  std::vector<float> queries(uint64_t{c.queries} * hidden);
  for (float &query : queries)
    query = 2.0F * random.gauss();
  const MetalBuffer queryBuffer = uploaded(backend, queries.data(), queries.size() * 4);

  // A state: noise, and for every other one a part along one of the queries,
  // so that the softmax is far from uniform.
  auto draw = [&](uint32_t state, uint32_t channel) -> float {
    if (c.zeroState && state == 0)
      return 0.0F;
    float value = random.gauss();
    if (state % 2 == 1)
      value += 2.0F * queries[uint64_t{state % c.queries} * hidden + channel];
    return static_cast<float>(c.magnitude * value);
  };
  std::vector<std::vector<double>> values(count, std::vector<double>(uint64_t{c.rows} * hidden));
  std::vector<Matrix<uint16_t>> states;
  std::vector<InklingRows> views;
  for (uint32_t state = 0; state < c.states; ++state) {
    states.emplace_back(backend, c.rows, hidden, c.padded);
    for (uint32_t row = 0; row < c.rows; ++row)
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        states[state].at(row, channel) = floatToBf16(draw(state, channel));
        values[state][uint64_t{row} * hidden + channel] = bf16ToFloat(states[state].at(row, channel));
      }
    views.push_back(states[state].view());
  }
  std::optional<Matrix<uint16_t>> partial16;
  std::optional<Matrix<float>> partial32;
  InklingRows partial;
  if (c.partialFloat) {
    partial32.emplace(backend, c.rows, hidden, c.padded);
    for (uint32_t row = 0; row < c.rows; ++row)
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        partial32->at(row, channel) = draw(c.states, channel);
        values[c.states][uint64_t{row} * hidden + channel] = partial32->at(row, channel);
      }
    partial = partial32->view();
  } else {
    partial16.emplace(backend, c.rows, hidden, c.padded);
    for (uint32_t row = 0; row < c.rows; ++row)
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        partial16->at(row, channel) = floatToBf16(draw(c.states, channel));
        values[c.states][uint64_t{row} * hidden + channel] = bf16ToFloat(partial16->at(row, channel));
      }
    partial = partial16->view();
  }
  Matrix<float> output(backend, c.rows, c.queries * hidden, c.padded);

  CommandGraph graph;
  const MetalBuffer sums =
      backend.allocateBuffer(Inkling::depthSumsBytes(hidden, c.rows), splash::metal::BufferStorage::Shared, "inkling-depth-sums");
  Inkling::addDepthMix(graph, views, partial, c.partialFloat, queryBuffer, c.queries, output.view(), sums, hidden,
                       c.rows);
  run(backend, graph, what);

  const double scale = 1.0 / std::sqrt(double(hidden));
  std::vector<std::vector<double>> keys(count, std::vector<double>(hidden));
  for (uint32_t row = 0; row < c.rows; ++row) {
    for (uint32_t state = 0; state < count; ++state) {
      double squares = 0;
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        const double value = values[state][uint64_t{row} * hidden + channel];
        squares += value * value;
      }
      const double inverse = 1.0 / std::sqrt(squares / hidden + kEpsilon);
      for (uint32_t channel = 0; channel < hidden; ++channel)
        keys[state][channel] = values[state][uint64_t{row} * hidden + channel] * inverse;
    }
    for (uint32_t query = 0; query < c.queries; ++query) {
      std::vector<double> weight(count);
      double top = -std::numeric_limits<double>::infinity();
      for (uint32_t state = 0; state < count; ++state) {
        double score = 0;
        for (uint32_t channel = 0; channel < hidden; ++channel)
          score += keys[state][channel] * queries[uint64_t{query} * hidden + channel];
        weight[state] = score * scale;
        top = std::max(top, weight[state]);
      }
      double total = 0;
      for (double &w : weight) {
        w = std::exp(w - top);
        total += w;
      }
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        double expected = 0;
        for (uint32_t state = 0; state < count; ++state)
          expected += weight[state] / total * keys[state][channel];
        near(what, row, uint64_t{query} * hidden + channel, output.at(row, query * hidden + channel), expected,
             1e-4 + 1e-4 * std::fabs(expected));
      }
    }
  }
  output.requireGuarded(what + " output");
  for (const Matrix<uint16_t> &state : states)
    state.requireGuarded(what + " state");
  if (partial16)
    partial16->requireGuarded(what + " partial");
  if (partial32)
    partial32->requireGuarded(what + " partial");
}

uint32_t depthMixes(MetalBackend &backend) {
  Random random(0x1D3A7);
  static constexpr double magnitudes[] = {1.0, 1e-3, 300.0};
  uint32_t cases = 0;
  for (const uint32_t hidden : {kHidden, 512u})
    for (const uint32_t states : {0u, 1u, 2u, 4u})
      for (const uint32_t queries : {1u, 3u, 4u})
        for (const bool partialFloat : {false, true})
          for (const uint32_t rows : {1u, 3u})
            for (const bool padded : {false, true}) {
              depthMix(backend, random,
                       {hidden, rows, states, queries, partialFloat, padded, magnitudes[cases % 3], cases % 7 == 3});
              ++cases;
            }
  // A loop pass's messages: the two bank states (one bf16 state and the f32
  // partial) under four queries, for the rows a decode step or a prompt
  // chunk gives a layer.
  for (const uint32_t rows : {1u, 4u, 17u})
    for (const bool padded : {false, true}) {
      depthMix(backend, random, {kHidden, rows, 1, 4, true, padded, 1.0, false});
      ++cases;
    }
  // Many rows of the head's four blocks and the partial.
  depthMix(backend, random, {kHidden, 33, 4, 1, false, false, 1.0, false});
  return cases + 1;
}

// The message gate -------------------------------------------------------

struct GateWeights final {
  MetalBuffer down, forget, gain;
  std::vector<double> d, f, g;
};

// down [rank][2 * hidden] and forget, gain [hidden][rank] as bf16. `grid`
// weights make the gate's hidden state exact: down in multiples of 2^-11 up
// to 31 of them, whose sums over +-1 inputs are exact in fp32 and fall
// between bf16 values, so that only a hidden state rounded to bf16 matches.
GateWeights makeGateWeights(MetalBackend &backend, Random &random, uint32_t hidden, bool grid) {
  GateWeights weights;
  auto fill = [&](MetalBuffer &buffer, std::vector<double> &values, uint64_t count, auto draw) {
    buffer = sharedBuffer(backend, count * 2);
    values.resize(count);
    for (uint64_t index = 0; index < count; ++index) {
      const uint16_t value = floatToBf16(draw());
      static_cast<uint16_t *>(buffer.contents())[index] = value;
      values[index] = bf16ToFloat(value);
    }
  };
  fill(weights.down, weights.d, uint64_t{kRank} * 2 * hidden, [&] {
    return grid ? float(int(random.below(63)) - 31) / 2048.0F : 0.015F * random.gauss();
  });
  fill(weights.forget, weights.f, uint64_t{hidden} * kRank, [&] { return (grid ? 0.5F : 0.05F) * random.gauss(); });
  fill(weights.gain, weights.g, uint64_t{hidden} * kRank, [&] { return (grid ? 0.5F : 0.05F) * random.gauss(); });
  return weights;
}

void gate(MetalBackend &backend, Random &random, uint32_t hidden, uint32_t rows, bool alias, bool padded,
          bool exact) {
  const std::string what = std::string("message gate hidden ") + std::to_string(hidden) + " rows " +
                           std::to_string(rows) + (alias ? " in place" : "") + (padded ? " padded" : "") +
                           (exact ? " exact hidden state" : "");
  const GateWeights weights = makeGateWeights(backend, random, hidden, exact);
  Matrix<float> x(backend, rows, hidden, padded), message(backend, rows, hidden, padded);
  std::optional<Matrix<float>> separate;
  if (!alias)
    separate.emplace(backend, rows, hidden, padded);
  static constexpr float exactScales[] = {0.25F, 1.0F, 3.0F, 40.0F};
  static constexpr float randomX[] = {1.0F, 40.0F, 0.05F}, randomM[] = {1.0F, 0.3F, 20.0F};
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t channel = 0; channel < hidden; ++channel) {
      if (exact) {
        // Every value +-a (and +-b): the gate's RMS norms give +-1 exactly
        // once rounded to bf16.
        x.at(row, channel) = (random.below(2) ? 1.0F : -1.0F) * exactScales[row % 4];
        message.at(row, channel) = (random.below(2) ? 1.0F : -1.0F) * exactScales[(row / 2 + 1) % 4];
      } else {
        x.at(row, channel) = randomX[row % 3] * random.gauss();
        message.at(row, channel) = randomM[(row / 2) % 3] * random.gauss();
      }
    }
  const Matrix<float> &result = alias ? x : *separate;
  const std::vector<float> before(x.data(), x.data() + uint64_t{rows} * x.view().stride + x.view().offset);

  Matrix<float> low(backend, rows, kRank, padded);
  CommandGraph graph;
  Inkling::addMessageGate(graph, x.view(), message.view(), {weights.down, weights.forget, weights.gain}, kRank,
                          low.view(), result.view(), hidden, rows);
  run(backend, graph, what);

  std::vector<double> pair(2 * size_t{hidden}), h(kRank);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t half = 0; half < 2; ++half) {
      double squares = 0;
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        const double value = half ? message.at(row, channel) : before[x.view().offset + uint64_t{row} * x.view().stride + channel];
        squares += value * value;
      }
      const double inverse = 1.0 / std::sqrt(squares / hidden + kEpsilon);
      for (uint32_t channel = 0; channel < hidden; ++channel) {
        const double value = half ? message.at(row, channel) : before[x.view().offset + uint64_t{row} * x.view().stride + channel];
        pair[half * size_t{hidden} + channel] = bf16ToFloat(floatToBf16(float(value * inverse)));
      }
    }
    // The model's hidden state is the bf16 matmul's output (MLX's `pair @
    // down.T` is bf16), widened.
    for (uint32_t k = 0; k < kRank; ++k) {
      double sum = 0;
      for (uint32_t j = 0; j < 2 * hidden; ++j)
        sum += pair[j] * weights.d[uint64_t{k} * 2 * hidden + j];
      h[k] = bf16ToFloat(floatToBf16(float(sum)));
    }
    for (uint32_t channel = 0; channel < hidden; ++channel) {
      double forgetting = 0, gaining = 0;
      for (uint32_t k = 0; k < kRank; ++k) {
        forgetting += h[k] * weights.f[uint64_t{channel} * kRank + k];
        gaining += h[k] * weights.g[uint64_t{channel} * kRank + k];
      }
      const double a = before[x.view().offset + uint64_t{row} * x.view().stride + channel];
      const double m = message.at(row, channel);
      const double expected = (1.0 - std::tanh(forgetting)) * a + std::tanh(gaining) * m;
      // Random inputs may land a pair or hidden value on the other side of a
      // bf16 rounding than the fp32 kernel: one flipped hidden value moves
      // the gates by a weight times a bf16 ulp. The exact case has none.
      const double tolerance = exact ? 2e-5 * (std::fabs(a) + std::fabs(m)) + 1e-6
                                     : 4e-3 * (std::fabs(a) + std::fabs(m)) + 1e-4;
      near(what, row, channel, result.at(row, channel), expected, tolerance);
    }
  }
  result.requireGuarded(what + " output");
  message.requireGuarded(what + " message");
  if (!alias) {
    // Inputs were not written.
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t channel = 0; channel < hidden; ++channel)
        require(x.at(row, channel) == before[x.view().offset + uint64_t{row} * x.view().stride + channel],
                what + ": the input rows were written");
    x.requireGuarded(what + " input");
  }
}

uint32_t gates(MetalBackend &backend) {
  Random random(0x6A7E);
  uint32_t cases = 0;
  for (const uint32_t hidden : {kHidden, 1024u})
    // 600 rows run down as a matrix multiply (Inkling::addMessageGates), the others per output.
    for (const uint32_t rows : {1u, 3u, 17u, 600u})
      for (const bool alias : {false, true})
        for (const bool padded : {false, true})
          for (const bool exact : {false, true}) {
            gate(backend, random, hidden, rows, alias, padded, exact);
            ++cases;
          }
  return cases;
}

// The sublayer norm ------------------------------------------------------

void norm(MetalBackend &backend, Random &random, uint32_t width, uint32_t rows, bool float32, bool padded,
          float magnitude) {
  const std::string what = "norm width " + std::to_string(width) + " rows " + std::to_string(rows) +
                           (float32 ? " F32" : " bf16") + " weights" + (padded ? " padded" : "") + " x" +
                           text(magnitude);
  const NormWeights weights = makeNormWeights(backend, width, float32, [&](uint32_t) {
    return 1.0F + 0.3F * random.gauss();
  });
  Matrix<float> input(backend, rows, width, padded);
  Matrix<uint16_t> output(backend, rows, width, padded);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t channel = 0; channel < width; ++channel)
      input.at(row, channel) = row == 1 ? 0.0F : magnitude * random.gauss() * std::exp(random.unit());
  CommandGraph graph;
  Inkling::addNorm(graph, input.view(), weights, output.view(), width, rows);
  run(backend, graph, what);
  for (uint32_t row = 0; row < rows; ++row) {
    double squares = 0;
    for (uint32_t channel = 0; channel < width; ++channel)
      squares += double(input.at(row, channel)) * input.at(row, channel);
    const double inverse = 1.0 / std::sqrt(squares / width + kEpsilon);
    for (uint32_t channel = 0; channel < width; ++channel) {
      const double expected = input.at(row, channel) * inverse * normWeight(weights, channel);
      if (!roundedOnceToBf16(output.at(row, channel), expected))
        near(what, row, channel, bf16ToFloat(output.at(row, channel)), expected,
             0.5 * ulpBf16(float(expected)) * (1 + 1.0 / 256));
    }
  }
  output.requireGuarded(what + " output");
  input.requireGuarded(what + " input");
}

uint32_t norms(MetalBackend &backend) {
  Random random(0x4E04);
  uint32_t cases = 0;
  for (const uint32_t width : {kHidden, 1024u, 4096u})
    for (const uint32_t rows : {1u, 3u, 17u})
      for (const bool float32 : {false, true})
        for (const bool padded : {false, true})
          for (const float magnitude : {1.0F, 1e-3F, 500.0F}) {
            norm(backend, random, width, rows, float32, padded, magnitude);
            ++cases;
          }
  return cases;
}

// Verify trees -----------------------------------------------------------

// A verify block's rows as a tree (metal/abi/VerifyTree.h): each row's
// parent before it, the anchor's none.
struct Tree final {
  std::vector<uint32_t> parent;

  [[nodiscard]] uint32_t rows() const { return uint32_t(parent.size()); }
  [[nodiscard]] std::vector<uint32_t> path(uint32_t row) const {
    std::vector<uint32_t> rows{row};
    while (rows.front())
      rows.insert(rows.begin(), parent[rows.front()]);
    return rows;
  }
  [[nodiscard]] SplashTreeRow place(uint32_t row) const {
    uint32_t ancestors = 0;
    for (const uint32_t on : path(row))
      ancestors |= 1U << on;
    return {ancestors, static_cast<uint16_t>(row ? parent[row] : SPLASH_TREE_ROOT),
            static_cast<uint16_t>(path(row).size() - 1)};
  }
};

enum class Shape { Chain, Star, Random, Comb };

Tree makeTree(Shape shape, uint32_t rows, Random &random) {
  Tree tree;
  tree.parent.assign(rows, 0);
  for (uint32_t row = 1; row < rows; ++row)
    switch (shape) {
    case Shape::Chain:
      tree.parent[row] = row - 1;
      break;
    case Shape::Star:
      tree.parent[row] = 0;
      break;
    case Shape::Random:
      tree.parent[row] = random.below(row);
      break;
    case Shape::Comb:
      tree.parent[row] = row >= 2 ? row - 2 : 0;
      break;
    }
  return tree;
}

// The lanes' trees, in the forms the tests run.
std::vector<std::vector<Tree>> treeSets(Random &random) {
  std::vector<std::vector<Tree>> sets;
  sets.push_back({makeTree(Shape::Chain, 1, random)});
  sets.push_back({makeTree(Shape::Chain, 5, random)});
  sets.push_back({makeTree(Shape::Star, 7, random)});
  sets.push_back({makeTree(Shape::Comb, 12, random)});
  sets.push_back({makeTree(Shape::Random, 16, random)});
  sets.push_back({makeTree(Shape::Chain, 16, random)});
  sets.push_back({makeTree(Shape::Chain, kTreeRows, random)});
  sets.push_back({makeTree(Shape::Random, kTreeRows, random)});
  sets.push_back({makeTree(Shape::Random, 16, random), makeTree(Shape::Chain, 1, random),
                  makeTree(Shape::Star, 9, random), makeTree(Shape::Random, 5, random)});
  sets.push_back({makeTree(Shape::Chain, 2, random), makeTree(Shape::Random, 3, random)});
  return sets;
}

// The short convolutions -------------------------------------------------

struct ConvCase final {
  uint32_t channels = 0;
  bool dynamic = false;
  uint32_t heads = 0;
  std::vector<uint32_t> rows; // each sequence's rows
  std::vector<Tree> trees;    // each sequence's block, when rows are tree rows
  bool padded = false;
  bool writeHistory = false;
};

void convolution(MetalBackend &backend, Random &random, const ConvCase &c) {
  const uint32_t channels = c.channels, sequences = uint32_t(c.rows.size());
  const bool tree = !c.trees.empty();
  uint32_t total = 0, bound = 0;
  std::vector<uint32_t> first(sequences);
  for (uint32_t s = 0; s < sequences; ++s) {
    first[s] = total;
    total += c.rows[s];
    bound = std::max(bound, c.rows[s]);
  }
  const uint32_t heads = c.dynamic ? c.heads : 0, perHead = c.dynamic ? channels / heads : channels;
  std::string what = std::string(c.dynamic ? "dynamic" : "static") + " convolution channels " +
                     std::to_string(channels) + (tree ? " tree" : " chain") + " rows";
  for (const uint32_t rows : c.rows)
    what += " " + std::to_string(rows);
  what += std::string(c.padded ? " padded" : "") + (c.writeHistory ? " with history out" : "");

  // static [channels][taps], or dynamic [taps][channels]
  std::vector<double> weights(uint64_t{channels} * kTaps);
  const MetalBuffer weightBuffer = sharedBuffer(backend, weights.size() * 2);
  for (uint64_t index = 0; index < weights.size(); ++index) {
    const uint16_t value = floatToBf16(0.35F * random.gauss());
    static_cast<uint16_t *>(weightBuffer.contents())[index] = value;
    weights[index] = bf16ToFloat(value);
  }
  InklingConvolution convolution;
  convolution.channels = channels;
  convolution.taps = kTaps;
  convolution.weights = weightBuffer;
  convolution.dynamic = c.dynamic;
  convolution.heads = heads;

  Matrix<uint16_t> input(backend, total, channels, c.padded);
  Matrix<uint16_t> output(backend, total, channels, c.padded);
  for (uint32_t row = 0; row < total; ++row)
    for (uint32_t channel = 0; channel < channels; ++channel)
      input.at(row, channel) = floatToBf16(2.0F * random.gauss());
  std::optional<Matrix<float>> dynamicTaps;
  InklingRows dynamicView;
  if (c.dynamic) {
    dynamicTaps.emplace(backend, total, heads * kTaps, c.padded);
    for (uint32_t row = 0; row < total; ++row)
      for (uint32_t k = 0; k < heads * kTaps; ++k)
        dynamicTaps->at(row, k) = 0.3F * random.gauss();
    dynamicView = dynamicTaps->view();
  }

  // Each sequence's history: [taps - 1][channels], rows `stride` apart from
  // value `offset`, in a buffer of its own.
  const uint32_t historyStride = channels + (c.padded ? kPad : 0), historyOffset = c.padded ? kPad : 0;
  const uint64_t historyCount = uint64_t{historyOffset} + uint64_t{kTaps - 1} * historyStride + kGuard;
  auto historyIndex = [&](uint32_t row, uint32_t channel) {
    return uint64_t{historyOffset} + uint64_t{row} * historyStride + channel;
  };
  std::vector<Words> historyIn, historyOut;
  std::vector<MetalBuffer> historyInBuffers, historyOutBuffers;
  for (uint32_t s = 0; s < sequences; ++s) {
    historyIn.emplace_back(backend, historyCount);
    for (uint32_t row = 0; row < kTaps - 1; ++row)
      for (uint32_t channel = 0; channel < channels; ++channel)
        historyIn[s].put(historyIndex(row, channel), floatToBf16(2.0F * random.gauss()));
    historyOut.emplace_back(backend, historyCount);
  }
  for (uint32_t s = 0; s < sequences; ++s) {
    historyInBuffers.push_back(historyIn[s].buffer());
    if (c.writeHistory)
      historyOutBuffers.push_back(historyOut[s].buffer());
  }

  std::vector<SplashLaneRows> layout(sequences);
  for (uint32_t s = 0; s < sequences; ++s)
    layout[s] = {first[s], c.rows[s]};
  const MetalBuffer layoutBuffer = uploaded(backend, layout.data(), layout.size() * sizeof(SplashLaneRows));
  MetalBuffer treeBuffer;
  if (tree) {
    std::vector<SplashTreeRow> placed(total);
    for (uint32_t s = 0; s < sequences; ++s)
      for (uint32_t row = 0; row < c.rows[s]; ++row)
        placed[first[s] + row] = c.trees[s].place(row);
    treeBuffer = uploaded(backend, placed.data(), placed.size() * sizeof(SplashTreeRow));
  }

  // A verify's tree convolution writes each row's input into its tape.
  std::optional<Matrix<uint16_t>> tape;
  if (tree) tape.emplace(backend, total, channels, c.padded);
  CommandGraph graph;
  Inkling::addConvolution(graph, input.view(), historyInBuffers, historyOutBuffers, historyOffset, historyStride,
                          convolution, dynamicView, output.view(), layoutBuffer, LayoutRows{sequences, total, bound},
                          treeBuffer, tape ? tape->view() : InklingRows{});
  run(backend, graph, what);
  if (tape) {
    for (uint32_t s = 0; s < sequences; ++s)
      for (uint32_t row = 0; row < c.rows[s]; ++row)
        for (uint32_t channel = 0; channel < channels; ++channel)
          if (tape->at(first[s] + row, channel) != input.at(first[s] + row, channel))
            fail(what + " tape row " + std::to_string(first[s] + row) + " channel " + std::to_string(channel) +
                 ": not the row's input");
    tape->requireGuarded(what + " tape");
  }

  const double divisor = std::sqrt(c.dynamic ? 2.0 * kTaps + 1.0 : kTaps + 1.0);
  for (uint32_t s = 0; s < sequences; ++s) {
    const uint32_t rows = c.rows[s];
    auto previous = [&](const std::vector<uint32_t> &path, uint32_t lag, uint32_t channel) -> double {
      if (lag < path.size())
        return bf16ToFloat(input.at(first[s] + path[path.size() - 1 - lag], channel));
      return bf16ToFloat(historyIn[s].data()[historyIndex((kTaps - 2) - (lag - uint32_t(path.size())), channel)]);
    };
    std::vector<uint32_t> chain;
    for (uint32_t row = 0; row < rows; ++row) {
      std::vector<uint32_t> path;
      if (tree) {
        path = c.trees[s].path(row);
      } else {
        chain.push_back(row);
        path = chain;
      }
      for (uint32_t channel = 0; channel < channels; ++channel) {
        double sum = 0, magnitude = 0;
        const double current = previous(path, 0, channel);
        for (uint32_t lag = 0; lag < kTaps; ++lag) {
          double tap;
          if (c.dynamic)
            tap = weights[uint64_t{lag} * channels + channel] +
                  double(dynamicTaps->at(first[s] + row, (channel / perHead) * kTaps + lag));
          else
            tap = weights[uint64_t{channel} * kTaps + (kTaps - 1 - lag)];
          const double term = tap * previous(path, lag, channel);
          sum += term;
          magnitude += std::fabs(term);
        }
        nearBf16(what, first[s] + row, channel, output.at(first[s] + row, channel), (sum + current) / divisor,
                 (magnitude + std::fabs(current)) / divisor);
      }
    }
    if (c.writeHistory) {
      // The last taps - 1 of the history followed by the sequence's inputs.
      for (uint32_t row = 0; row < kTaps - 1; ++row)
        for (uint32_t channel = 0; channel < channels; ++channel) {
          const uint32_t at = rows + row;
          historyOut[s].expect(historyIndex(row, channel),
                               at < kTaps - 1 ? historyIn[s].data()[historyIndex(at, channel)]
                                              : input.at(first[s] + at - (kTaps - 1), channel));
        }
    }
  }
  output.requireGuarded(what + " output");
  input.requireGuarded(what + " input");
  if (dynamicTaps)
    dynamicTaps->requireGuarded(what + " dynamic taps");
  for (uint32_t s = 0; s < sequences; ++s) {
    historyIn[s].requireMatches(what + " history in");
    historyOut[s].requireMatches(what + " history out");
  }
}

uint32_t convolutions(MetalBackend &backend) {
  Random random(0xC0DE);
  uint32_t cases = 0;
  // Prompt chunks and decode steps: sequences of 1 to 130 rows after their
  // histories, over 2304 and 1024 channels, static, and dynamic over 8 heads.
  const std::vector<std::vector<uint32_t>> shapes = {{1}, {2}, {3}, {4}, {5}, {1, 2, 3, 4}, {37, 5}, {70, 1, 130, 2}};
  for (const bool padded : {false, true})
    for (const bool writeHistory : {false, true}) {
      for (const uint32_t channels : {2304u, 1024u})
        for (const auto &rows : shapes) {
          ConvCase c;
          c.channels = channels;
          c.rows = rows;
          c.padded = padded;
          c.writeHistory = writeHistory;
          convolution(backend, random, c);
          ++cases;
        }
      for (const auto &rows : std::vector<std::vector<uint32_t>>{{1}, {3}, {40, 2, 7}, {1, 1, 1, 1}}) {
        ConvCase c;
        c.channels = 1024;
        c.dynamic = true;
        c.heads = 8;
        c.rows = rows;
        c.padded = padded;
        c.writeHistory = writeHistory;
        convolution(backend, random, c);
        ++cases;
      }
    }
  // Verify trees: a row's earlier inputs are its ancestors', then the lane's
  // history. A tree's history is written by the commit, not the convolution.
  for (const std::vector<Tree> &trees : treeSets(random))
    for (const bool padded : {false, true})
      for (const bool dynamic : {false, true})
        for (const uint32_t channels : dynamic ? std::vector<uint32_t>{1024} : std::vector<uint32_t>{2304, 1024}) {
          ConvCase c;
          c.channels = channels;
          c.dynamic = dynamic;
          c.heads = dynamic ? 8 : 0;
          for (const Tree &tree : trees)
            c.rows.push_back(tree.rows());
          c.trees = trees;
          c.padded = padded;
          convolution(backend, random, c);
          ++cases;
        }
  return cases;
}

// The history a prefill captures -------------------------------------------

void historyAfter(MetalBackend &backend, Random &random, uint32_t channels, uint32_t rows, bool padded) {
  const std::string what = "convolution history channels " + std::to_string(channels) + " rows " +
                           std::to_string(rows) + (padded ? " padded" : "");
  // The sequence's rows past the first `rows` hold values the history must
  // not take.
  Matrix<uint16_t> input(backend, rows + 2, channels, padded);
  for (uint32_t row = 0; row < rows + 2; ++row)
    for (uint32_t channel = 0; channel < channels; ++channel)
      input.at(row, channel) = floatToBf16(2.0F * random.gauss());
  const uint32_t stride = channels + (padded ? kPad : 0), offset = padded ? kPad : 0;
  const uint64_t count = uint64_t{offset} + uint64_t{kTaps - 1} * stride + kGuard;
  Words historyIn(backend, count), historyOut(backend, count);
  auto index = [&](uint32_t row, uint32_t channel) { return uint64_t{offset} + uint64_t{row} * stride + channel; };
  for (uint32_t row = 0; row < kTaps - 1; ++row)
    for (uint32_t channel = 0; channel < channels; ++channel)
      historyIn.put(index(row, channel), floatToBf16(2.0F * random.gauss()));
  CommandGraph graph;
  Inkling::addConvolutionHistory(graph, input.view(), historyIn.buffer(), historyOut.buffer(), offset, stride,
                                 channels, kTaps, rows);
  run(backend, graph, what);
  for (uint32_t row = 0; row < kTaps - 1; ++row)
    for (uint32_t channel = 0; channel < channels; ++channel) {
      const uint32_t at = rows + row;
      historyOut.expect(index(row, channel), at < kTaps - 1 ? historyIn.data()[index(at, channel)]
                                                             : input.at(at - (kTaps - 1), channel));
    }
  historyIn.requireMatches(what + " history in");
  historyOut.requireMatches(what + " history out");
  input.requireGuarded(what + " input");
}

uint32_t histories(MetalBackend &backend) {
  Random random(0x815);
  uint32_t cases = 0;
  for (const uint32_t channels : {2304u, 1024u})
    for (const uint32_t rows : {1u, 2u, 3u, 4u, 5u, 17u, 100u})
      for (const bool padded : {false, true}) {
        historyAfter(backend, random, channels, rows, padded);
        ++cases;
      }
  return cases;
}

// The history a verify step commits ---------------------------------------

// What a lane accepted: its anchor alone, the path to a random row, or to
// its last row.
uint32_t acceptedPath(const Tree &tree, Random &random, uint32_t mode) {
  const uint32_t row = mode == 0 ? 0 : mode == 1 ? random.below(tree.rows()) : tree.rows() - 1;
  return tree.place(row).ancestors;
}

void commit(MetalBackend &backend, Random &random, uint32_t channels, uint32_t blocks, bool interleaved, bool padded,
            const std::vector<Tree> &trees, uint32_t modeOffset, uint32_t pathRows) {
  const uint32_t lanes = uint32_t(trees.size());
  const std::string what = "convolution commit channels " + std::to_string(channels) + " blocks " +
                           std::to_string(blocks) + (interleaved ? " interleaved" : " block-major") +
                           (padded ? " padded" : "") + " lanes " + std::to_string(lanes) + " first rows " +
                           std::to_string(trees[0].rows()) + " path rows " + std::to_string(pathRows);
  uint32_t total = 0, bound = 0;
  std::vector<uint32_t> first(lanes);
  std::vector<SplashLaneRows> layout(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    first[lane] = total;
    layout[lane] = {total, trees[lane].rows()};
    total += trees[lane].rows();
    bound = std::max(bound, trees[lane].rows());
  }
  // Block b of a row's inputs at value b * blockWidth of the row.
  const uint32_t blockWidth = channels + (padded ? kPad : 0);
  Matrix<uint16_t> inputs(backend, total, blocks * blockWidth, padded);
  for (uint32_t row = 0; row < total; ++row)
    for (uint32_t block = 0; block < blocks; ++block)
      for (uint32_t channel = 0; channel < channels; ++channel)
        inputs.at(row, block * blockWidth + channel) = floatToBf16(2.0F * random.gauss());

  // A lane's history of block b, row i, at b * blockStride + i * rowStride:
  // block after block, or rows of every block interleaved.
  const uint32_t rowStride = interleaved ? blocks * blockWidth : blockWidth;
  const uint32_t blockStride = interleaved ? blockWidth : (kTaps - 1) * blockWidth;
  const uint64_t count = uint64_t{kTaps - 1} * blocks * blockWidth + kGuard;
  auto index = [&](uint32_t block, uint32_t row, uint32_t channel) {
    return uint64_t{block} * blockStride + uint64_t{row} * rowStride + channel;
  };
  std::vector<Words> historyIn, historyOut;
  std::vector<MetalBuffer> historyInBuffers, historyOutBuffers;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    historyIn.emplace_back(backend, count);
    historyOut.emplace_back(backend, count);
    for (uint32_t block = 0; block < blocks; ++block)
      for (uint32_t row = 0; row < kTaps - 1; ++row)
        for (uint32_t channel = 0; channel < channels; ++channel)
          historyIn[lane].put(index(block, row, channel), floatToBf16(2.0F * random.gauss()));
  }
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    historyInBuffers.push_back(historyIn[lane].buffer());
    historyOutBuffers.push_back(historyOut[lane].buffer());
  }
  std::vector<uint32_t> paths(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    paths[lane] = acceptedPath(trees[lane], random, (lane + modeOffset) % 3);
  const MetalBuffer pathBuffer = uploaded(backend, paths.data(), paths.size() * 4);
  const MetalBuffer layoutBuffer = uploaded(backend, layout.data(), layout.size() * sizeof(SplashLaneRows));

  CommandGraph graph;
  Inkling::addConvolutionCommit(graph, inputs.view(), blocks, blockWidth, channels, historyInBuffers,
                                historyOutBuffers, blockStride, rowStride, kTaps, pathBuffer, layoutBuffer,
                                LayoutRows{lanes, total, bound}, pathRows);
  run(backend, graph, what);

  for (uint32_t lane = 0; lane < lanes; ++lane) {
    std::vector<uint32_t> accepted;
    for (uint32_t row = 0; row < trees[lane].rows(); ++row)
      if (paths[lane] >> row & 1)
        accepted.push_back(row);
    require(!accepted.empty(), what + ": the test's accepted path is empty");
    // A capture's history follows only the path's first pathRows rows.
    const uint32_t taken = pathRows ? std::min(uint32_t(accepted.size()), pathRows) : uint32_t(accepted.size());
    for (uint32_t block = 0; block < blocks; ++block)
      for (uint32_t row = 0; row < kTaps - 1; ++row)
        for (uint32_t channel = 0; channel < channels; ++channel) {
          // The history followed by the accepted rows' inputs.
          const uint32_t at = taken + row;
          historyOut[lane].expect(index(block, row, channel),
                                  at < kTaps - 1
                                      ? historyIn[lane].data()[index(block, at, channel)]
                                      : inputs.at(first[lane] + accepted[at - (kTaps - 1)],
                                                  block * blockWidth + channel));
        }
  }
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    historyIn[lane].requireMatches(what + " history in");
    historyOut[lane].requireMatches(what + " history out");
  }
  inputs.requireGuarded(what + " inputs");
}

uint32_t commits(MetalBackend &backend) {
  Random random(0xC033);
  uint32_t cases = 0;
  const std::vector<std::vector<Tree>> sets = treeSets(random);
  for (const uint32_t channels : {2304u, 1024u})
    for (const uint32_t blocks : {1u, 3u})
      for (const bool interleaved : {false, true})
        for (const bool padded : {false, true})
          for (uint32_t set = 0; set < sets.size(); ++set) {
            commit(backend, random, channels, blocks, interleaved, padded, sets[set], cases, cases % 4);
            ++cases;
          }
  // Every visit's convolution at once: 48 blocks of one width.
  commit(backend, random, 1024, 48, false, false, sets[8], 1, 0);
  commit(backend, random, 1024, 48, true, true, sets[8], 2, 3);
  return cases + 2;
}

// The head's softcap -------------------------------------------------------

void logitCap(MetalBackend &backend, Random &random, uint32_t rows, uint32_t width, uint32_t vocabulary,
              float softcap) {
  const std::string what = "logit cap rows " + std::to_string(rows) + " width " + std::to_string(width) +
                           " vocabulary " + std::to_string(vocabulary) + " softcap " + text(softcap);
  const uint64_t count = uint64_t{rows} * width;
  const MetalBuffer buffer = sharedBuffer(backend, (count + kGuard) * 4);
  auto *logits = static_cast<float *>(buffer.contents());
  std::fill_n(logits + count, kGuard, std::bit_cast<float>(kPoison32));
  // Mostly the head's range; the ends of the valid ids and the first padding
  // ones, and values that saturate the cap.
  static constexpr float special[] = {0.0F, -0.0F, 1e-8F, -3e4F, 3e4F, 1e6F, -1e6F, 30.0F, -88.0F};
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t id = 0; id < width; ++id)
      logits[uint64_t{row} * width + id] = 25.0F * random.gauss();
    for (uint32_t k = 0; k < std::size(special) && k < width; ++k)
      logits[uint64_t{row} * width + (k * 7919 + row) % width] = special[k];
    logits[uint64_t{row} * width + vocabulary - 1] = 61.0F;
    if (vocabulary < width)
      logits[uint64_t{row} * width + vocabulary] = 61.0F;
    logits[uint64_t{row} * width + width - 1] = -61.0F;
  }
  std::vector<float> before(logits, logits + count);
  CommandGraph graph;
  Inkling::addLogitCap(graph, buffer, softcap, vocabulary, width, rows);
  run(backend, graph, what);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t id = 0; id < width; ++id) {
      const float got = logits[uint64_t{row} * width + id];
      if (id >= vocabulary) {
        if (got != -1e30F)
          fail(what + " row " + std::to_string(row) + " id " + std::to_string(id) + ": got " + text(got) +
               ", expected -1e30");
        continue;
      }
      const double expected = std::tanh(double(before[uint64_t{row} * width + id]) / softcap) * softcap;
      near(what, row, id, got, expected, 3e-5 * std::fabs(expected) + 1e-5);
    }
  for (uint32_t i = 0; i < kGuard; ++i)
    require(std::bit_cast<uint32_t>(logits[count + i]) == kPoison32, what + ": a value past the rows was written");
}

uint32_t logitCaps(MetalBackend &backend) {
  Random random(0xCA9);
  struct Head final {
    uint32_t rows, width, vocabulary;
    float softcap;
  };
  // The head's: 201,024 rows of lm_head, ids from 200,058 on masked.
  const Head shapes[] = {{1, 201024, 200058, 30.0F}, {3, 201024, 200058, 30.0F}, {2, 1030, 1001, 30.0F},
                          {4, 4096, 4096, 30.0F},     {1, 37, 20, 30.0F},         {5, 257, 1, 30.0F},
                          {2, 2048, 1500, 5.0F}};
  for (const Head &shape : shapes)
    logitCap(backend, random, shape.rows, shape.width, shape.vocabulary, shape.softcap);
  return uint32_t(std::size(shapes));
}

void runAll(const std::string &metallib) {
  MetalBackend backend(metallib);
  const uint32_t summing = sums(backend);
  const uint32_t depth = depthMixes(backend);
  const uint32_t gating = gates(backend);
  const uint32_t norming = norms(backend);
  const uint32_t convolving = convolutions(backend);
  const uint32_t capturing = histories(backend);
  const uint32_t committing = commits(backend);
  const uint32_t capping = logitCaps(backend);
  std::cout << "inkling_ops_test: PASS (" << summing << " sum, " << depth << " depth mix, " << gating << " gate, " << norming << " norm, "
            << convolving << " convolution, " << capturing << " history, " << committing << " commit, " << capping
            << " logit cap cases)\n";
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::invalid_argument("usage: inkling-ops METALLIB");
      runAll(argv[1]);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "inkling_ops_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
