// The decode norms that also write the register tile's input table
// (norm_rms_table16_decode and, for Apple7/8, norm_rms_table16f_decode, with
// bf16 or F32 weights): their rows match the plain norm's bit for bit, and
// their table and sums match the register tile's own preparation of those
// rows byte for byte, for the 27B's and smaller hidden sizes (the streaming
// path at 2048 and below, the chunked one above), one to four lanes and
// inputs of very different magnitudes.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using namespace splash::test;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer poisoned(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer = sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0xA5, bytes);
  return buffer;
}

void tableMatchesPreparation(MetalBackend &backend, LinearInput layout, uint32_t width, uint32_t lanes,
                             bool float32, float scale) {
  const uint32_t rows = lanes * kRows;
  const std::string what = std::string(layout == LinearInput::Table16F ? "Table16F" : "Table16") + " norm width " +
                           std::to_string(width) + " lanes " + std::to_string(lanes) +
                           (float32 ? " F32" : " bf16") + " weights, inputs x" + std::to_string(scale);
  Random random(0x5EED0000ULL + width * 8 + lanes * 2 + float32);
  const uint64_t rowBytes = uint64_t{rows} * width * 2;
  MetalBuffer input = sharedBuffer(backend, rowBytes);
  auto *values = static_cast<uint16_t *>(input.contents());
  for (uint64_t index = 0; index < uint64_t{rows} * width; ++index)
    values[index] = floatToBf16(random.unit() * scale * std::exp(2.0F * random.unit()));
  const NormWeights norm = makeNormWeights(backend, width, float32, [&](uint32_t) { return 1.0F + random.unit(); });
  const uint64_t tableSize = tableBytes(width, rows, layout), sumsSize = tableSumsBytes(layout, width, rows);
  MetalBuffer output = poisoned(backend, rowBytes), plain = poisoned(backend, rowBytes),
              table = poisoned(backend, tableSize), sums = poisoned(backend, sumsSize),
              referenceTable = poisoned(backend, tableSize), referenceSums = poisoned(backend, sumsSize);
  CommandGraph graph;
  const PreparedInput prepared =
      Normalization::addRms(graph, input, norm, output, width, rows, {table, sums, {}, {}}, layout);
  (void)Normalization::addRms(graph, input, norm, plain, width, rows);
  addReferencePreparation(graph, layout, output, referenceTable, referenceSums, width, rows / GGUF_TABLE16_ROWS);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  require(prepared.layout == layout && prepared.source.sameView(output), what + " did not report its table");
  require(!std::memcmp(output.contents(), plain.contents(), rowBytes), what + ": rows differ from the plain norm's");
  require(!std::memcmp(table.contents(), referenceTable.contents(), tableSize), what + ": table mismatch");
  require(!std::memcmp(sums.contents(), referenceSums.contents(), sumsSize), what + ": sums mismatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: norm-table METALLIB");
    MetalBackend backend(argv[1]);
    for (const LinearInput layout : {LinearInput::Table16, LinearInput::Table16F})
      for (const uint32_t width : {1024U, 2048U, 2560U, 5120U, 6144U, 17408U})
        for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
          for (const bool float32 : {false, true})
            for (const float scale : {1.0F, 300.0F, 1e-3F})
              tableMatchesPreparation(backend, layout, width, lanes, float32, scale);
    std::cout << "norm_table_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "norm_table_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
