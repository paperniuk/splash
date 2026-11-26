#include "TestChecks.hpp"
#include "metal/abi/KvExtent.h"
#include "metal/abi/RelativeAttention.h"
#include "ops/CpuRelativeAttention.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// The CPU's share of the relative attention (ops::CpuRelativeAttention), with
// no GPU: a chunk's first rows over pages of random INT8 keys and values at
// the model's shapes (24 query heads over 8 of 128, a bias at distances below
// 1024, a sink a head, a gate), against the fp64 attention of what the rows
// stage and the pages hold. Each output is within the bound of
// relative_attention_metal_test.mm: the bf16 rounding of the probabilities
// (2^-9 of the mass, twice over) and of the row, both times the gate, and of the
// gated row. The same bits come at 1, 4, 8 and 10 threads and on a repeat.

namespace {

using namespace splash;
using ops::CpuRelativeAttention;
using test::require;

constexpr uint32_t kQueryHeads = 24, kKvHeads = 8, kGroup = 3, kD = 128, kStates = SPLASH_RELATIVE_ATTENTION_STATES;
constexpr uint32_t kExtent = SPLASH_RELATIVE_ATTENTION_EXTENT, kPage = kv::kPageTokens;
constexpr uint32_t kProjectionStride = 6592, kGateColumn = kQueryHeads * kD;
constexpr std::array kSinks{-9.0f, 0.0f, 12.4f, 60.0f};

float bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  std::memcpy(&x, &u, 4);
  return x;
}
uint16_t bf16Bits(float x) {
  const float r = bf16(x);
  uint32_t u;
  std::memcpy(&u, &r, 4);
  return static_cast<uint16_t>(u >> 16);
}
float fromBits(uint16_t bits) {
  const uint32_t u = uint32_t{bits} << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
double ulp(double x) {
  if (x == 0) return 0;
  int exponent;
  std::frexp(std::fabs(x), &exponent);
  return std::ldexp(1.0, exponent - 8);
}

struct Random final {
  std::mt19937_64 engine{12345};
  float unit() { return std::uniform_real_distribution<float>(-1.0f, 1.0f)(engine); }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>(engine() % n); }
};

struct Case final {
  uint32_t committed = 0, rows = 0, stride = 0;
  std::vector<uint16_t> queries, projection;
  std::vector<float> states, sinks, bias;
  // The pages in extents of `extentPages`, in a shuffled order.
  uint32_t extentPages = 0;
  std::vector<std::vector<std::byte>> extents;
  std::vector<kv::HostExtent> hosts;
  std::vector<SplashKvPage> table;
  SplashKvLayer layer{};
  uint64_t dataBytes = 0, scaleBytes = 0;

  uint32_t pageOf(uint32_t token) const { return slots[token / kPage]; }
  std::vector<uint32_t> slots;
  // Tensor `tensor` (SPLASH_KV_KEYS to SPLASH_KV_VALUE_SCALES) of a token's page.
  [[nodiscard]] std::byte *tensor(uint32_t token, uint32_t which) {
    const uint32_t page = pageOf(token);
    const uint64_t offset = splash_kv_offset(extentPages, static_cast<uint32_t>(dataBytes),
                                             static_cast<uint32_t>(scaleBytes), 0, which, page % extentPages);
    return extents[page / extentPages].data() + offset;
  }
};

Case makeCase(Random &random, uint32_t committed, uint32_t rows) {
  Case data;
  data.committed = committed;
  data.rows = rows;
  data.stride = (rows + kPage - 1) / kPage * kPage;
  const uint32_t pages = (committed + rows + kPage - 1) / kPage;
  kv::Layout layout{.attentionLayers = 1, .kvHeads = kKvHeads, .headDimension = kD, .format = kv::Format::Int8};
  data.dataBytes = layout.dataBytesPerLayerPage();
  data.scaleBytes = layout.scaleBytesPerLayerPage();
  data.extentPages = std::max(3U, (pages + 3) / 3) | 1U;
  const uint32_t extents = (pages + data.extentPages - 1) / data.extentPages;
  data.layer = splash_kv_layer(data.extentPages, static_cast<uint32_t>(data.dataBytes),
                               static_cast<uint32_t>(data.scaleBytes), 0);
  for (uint32_t e = 0; e < extents; ++e) {
    data.extents.emplace_back(data.extentPages * layout.bytesPerLayerPage(), std::byte{0});
    data.hosts.push_back({0x100000000ull + uint64_t{e} * 0x4000000ull, data.extents.back().data()});
  }
  data.slots.resize(pages);
  std::iota(data.slots.begin(), data.slots.end(), 0U);
  std::shuffle(data.slots.begin(), data.slots.end(), random.engine);
  for (const uint32_t slot : data.slots)
    data.table.push_back(splash_kv_page_entry(data.hosts[slot / data.extentPages].gpuAddress, slot % data.extentPages));
  // Keys and values of the tokens the rows attend; the rest of the last page
  // holds NaN scales, which no row reads.
  for (uint32_t token = 0; token < pages * kPage; ++token) {
    const bool written = token < committed + rows;
    auto *keys = reinterpret_cast<int8_t *>(data.tensor(token, SPLASH_KV_KEYS));
    auto *values = reinterpret_cast<int8_t *>(data.tensor(token, SPLASH_KV_VALUES));
    auto *keyScales = reinterpret_cast<float *>(data.tensor(token, SPLASH_KV_KEY_SCALES));
    auto *valueScales = reinterpret_cast<float *>(data.tensor(token, SPLASH_KV_VALUE_SCALES));
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      const uint32_t place = token % kPage;
      const uint64_t scale = splash_kv_scale_element(head, place);
      keyScales[scale] = written ? 0.02f * (1 + float((token + head) % 9) / 16) : NAN;
      valueScales[scale] = written ? 0.09f * (1 + float((token * 5 + head) % 11) / 16) / 2 : NAN;
      for (uint32_t d = 0; d < kD; ++d) {
        keys[splash_kv_key_element(head, place, d, kD)] = static_cast<int8_t>(std::lround(127 * random.unit()));
        values[splash_kv_value_element(head, place, d, kD)] = static_cast<int8_t>(std::lround(127 * random.unit()));
      }
    }
  }
  const uint64_t fused = uint64_t{kKvHeads} * data.stride * kGroup;
  data.queries.resize(fused * kD);
  data.states.resize(fused * kStates);
  data.sinks.resize(fused);
  for (uint64_t f = 0; f < fused; ++f) {
    const float amplitude = 1.0f + float(f % 3);
    for (uint32_t d = 0; d < kD; ++d) data.queries[f * kD + d] = bf16Bits(amplitude * random.unit());
    for (uint32_t k = 0; k < kStates; ++k) data.states[f * kStates + k] = bf16(2.0f * random.unit());
    data.sinks[f] = bf16(kSinks[(f + f / 7) % kSinks.size()] + 0.25f * float(f % 5));
  }
  data.projection.assign(uint64_t{rows} * kProjectionStride, 0);
  for (uint64_t at = 0; at < data.projection.size(); ++at) data.projection[at] = bf16Bits(2.0f * random.unit());
  data.bias.resize(size_t{kExtent} * kStates);
  for (float &b : data.bias) b = 0.6f * random.unit();
  return data;
}

std::vector<uint16_t> compute(Case &data, uint32_t threads) {
  std::vector<uint16_t> hidden(uint64_t{data.rows} * kQueryHeads * kD, 0xffff);
  CpuRelativeAttention::Rows rows;
  rows.queries = data.queries.data();
  rows.states = data.states.data();
  rows.sinks = data.sinks.data();
  rows.stride = data.stride;
  rows.projection = data.projection.data();
  rows.projectionStride = kProjectionStride;
  rows.gateColumn = kGateColumn;
  rows.bias = data.bias.data();
  rows.pageTable = data.table.data();
  rows.pageTableEntries = static_cast<uint32_t>(data.table.size());
  rows.layer = data.layer;
  rows.extents = data.hosts;
  rows.dataBytes = data.dataBytes;
  rows.scaleBytes = data.scaleBytes;
  rows.committed = data.committed;
  rows.rows = data.rows;
  rows.hidden = hidden.data();
  CpuRelativeAttention::run(rows, threads);
  return hidden;
}

// The fp64 attention of one row and query head.
struct Expected final {
  std::array<double, kD> output{}, mass{};
};
Expected reference(Case &data, uint32_t row, uint32_t head) {
  const uint32_t kvHead = head / kGroup, position = data.committed + row;
  const uint64_t fused = (uint64_t{kvHead} * data.stride + row) * kGroup + head % kGroup;
  std::array<double, kD> q{};
  for (uint32_t d = 0; d < kD; ++d) q[d] = fromBits(data.queries[fused * kD + d]);
  const double sink = data.sinks[fused];
  std::vector<double> scores(position + 1);
  double maximum = sink;
  for (uint32_t token = 0; token <= position; ++token) {
    const auto *keys = reinterpret_cast<const int8_t *>(data.tensor(token, SPLASH_KV_KEYS));
    const auto *scales = reinterpret_cast<const float *>(data.tensor(token, SPLASH_KV_KEY_SCALES));
    double score = 0;
    for (uint32_t d = 0; d < kD; ++d)
      score += q[d] * double(keys[splash_kv_key_element(kvHead, token % kPage, d, kD)]);
    score *= double(scales[splash_kv_scale_element(kvHead, token % kPage)]);
    score /= std::sqrt(double(kD));
    const uint32_t distance = position - token;
    if (distance < kExtent)
      for (uint32_t k = 0; k < kStates; ++k) score += double(data.states[fused * kStates + k]) * data.bias[distance * kStates + k];
    scores[token] = score;
    maximum = std::max(maximum, score);
  }
  double denominator = std::exp(sink - maximum);
  for (double &s : scores) {
    s = std::exp(s - maximum);
    denominator += s;
  }
  Expected result;
  for (uint32_t token = 0; token <= position; ++token) {
    const auto *values = reinterpret_cast<const int8_t *>(data.tensor(token, SPLASH_KV_VALUES));
    const auto *scales = reinterpret_cast<const float *>(data.tensor(token, SPLASH_KV_VALUE_SCALES));
    const double weight = scores[token] / denominator * double(scales[splash_kv_scale_element(kvHead, token % kPage)]);
    for (uint32_t d = 0; d < kD; ++d) {
      const double value = double(values[splash_kv_value_element(kvHead, token % kPage, d, kD)]);
      result.output[d] += weight * value;
      result.mass[d] += std::fabs(weight * value);
    }
  }
  return result;
}

// The rows checked: the first and last tiles' and every task's edge, and a
// spread of the others.
std::vector<uint32_t> checkedRows(uint32_t rows) {
  std::vector<uint32_t> checked;
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t inTask = row % CpuRelativeAttention::kTaskRows;
    if (rows <= 80 || row < 3 || row + 3 >= rows || inTask < 2 || inTask + 2 >= CpuRelativeAttention::kTaskRows ||
        row % 37 == 0)
      checked.push_back(row);
  }
  return checked;
}

double check(Case &data, const std::vector<uint16_t> &hidden, const std::string &what) {
  for (const uint16_t bits : hidden)
    require(bits != 0xffff && std::isfinite(fromBits(bits)), what + ": an output element is not written or not finite");
  double largest = 0;
  const std::vector<uint32_t> rows = checkedRows(data.rows);
  // A few query heads of each KV head's three, over every checked row.
  for (const uint32_t row : rows)
    for (uint32_t head = 0; head < kQueryHeads; head += (row % 2 ? 1 : 4)) {
      const Expected expected = reference(data, row, head);
      for (uint32_t d = 0; d < kD; ++d) {
        const double gateInput =
            fromBits(data.projection[uint64_t{row} * kProjectionStride + kGateColumn + head * kD + d]);
        const double gate = 1 / (1 + std::exp(-gateInput));
        const double exact = expected.output[d] * gate;
        const double actual = fromBits(hidden[(uint64_t{row} * kQueryHeads + head) * kD + d]);
        const double bound = gate * (expected.mass[d] / 256 + ulp(expected.output[d])) + ulp(exact);
        const double error = std::fabs(actual - exact);
        if (!(error <= bound))
          throw std::runtime_error(what + ": row " + std::to_string(row) + " head " + std::to_string(head) +
                                   " dimension " + std::to_string(d) + " is " + std::to_string(actual) +
                                   ", the reference " + std::to_string(exact) + " (bound " + std::to_string(bound) +
                                   ")");
        largest = std::max(largest, error / bound);
      }
    }
  return largest;
}

void runCase(Random &random, uint32_t committed, uint32_t rows, bool timed = false) {
  Case data = makeCase(random, committed, rows);
  const std::string what = "history " + std::to_string(committed) + " rows " + std::to_string(rows);
  const auto begin = std::chrono::steady_clock::now();
  const std::vector<uint16_t> eight = compute(data, 8);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  const double largest = check(data, eight, what);
  for (const uint32_t threads : {1U, 4U, 10U})
    require(compute(data, threads) == eight, what + ": the bits depend on " + std::to_string(threads) + " threads");
  require(compute(data, 8) == eight, what + ": the bits change on a repeat");
  std::cout << what << ": largest error " << largest << " of its bound, same bits at 1, 4, 8, 10 threads PASS";
  if (timed) {
    double pairs = 0;
    for (uint32_t row = 0; row < rows; ++row) pairs += double(committed + row + 1);
    std::cout << " (" << seconds << " s at 8 threads, " << pairs * 12288 / seconds / 1e12 << " TFLOP/s)";
  }
  std::cout << '\n';
}

// The rows the CPU takes: whole direct tiles that carry the share of the chunk's
// attention pairs, none for a short history or chunk or a share of none.
void checkShare() {
  using ops::CpuRelativeAttention;
  require(CpuRelativeAttention::rowsFor(0.3, 4096, 2048) == 0 && CpuRelativeAttention::rowsFor(0.3, 65536, 256) == 0 &&
              CpuRelativeAttention::rowsFor(0.0, 65536, 2048) == 0,
          "the CPU takes rows of a short history, a short chunk or no share");
  for (const double share : {0.1, 0.3, 0.5, 0.9})
    for (const uint32_t history : {8192U, 65536U, 204800U})
      for (const uint32_t rows : {512U, 1000U, 2048U}) {
        const uint32_t taken = CpuRelativeAttention::rowsFor(share, history, rows);
        const double pairs = double(taken) * (history + taken / 2.0), whole = double(rows) * (history + rows / 2.0);
        require(taken % 16 == 0 && taken < rows && pairs <= share * whole &&
                    pairs >= share * whole - 16.0 * (history + rows),
                "the CPU's rows are not a tile-aligned share of the chunk's pairs");
      }
  const std::array<uint16_t, 4> a{0x3f80, 0x3f80, 0xbf80, 0x0000}, b{0x3f80, 0x3f81, 0xbf7f, 0x8000};
  const CpuRelativeAttention::Difference difference = CpuRelativeAttention::compare(a.data(), b.data(), a.size());
  require(difference.largestSteps == 1 && difference.farShare == 0.0 && difference.relative > 0.0,
          "bf16 steps between outputs are miscounted");
}

} // namespace

int main() {
  if (!CpuRelativeAttention::available()) {
    std::cout << "cpu attention: SKIP (no AMX on this CPU)\n";
    return 0;
  }
  try {
    checkShare();
    Random random;
    // Histories at the bias extent's and the pages' edges, rows that do and do
    // not fill a task, a tile or a page.
    for (const uint32_t history : {0U, 1U, 31U, 1023U, 1024U, 1025U})
      for (const uint32_t rows : {1U, 16U, 48U, 67U, 130U}) runCase(random, history, rows);
    runCase(random, 8192, 96);
    runCase(random, 40000, 144, true);
    std::cout << "cpu attention: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "cpu attention: FAIL: " << error.what() << '\n';
    return 1;
  }
}
