#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <iostream>

int main() {
  using namespace splash;
  try {
    test::TemporaryDirectory dir("inkling-affine");
    const std::array<uint32_t, 2> scales{0x3f808000, 0x3f818000};
    const std::array<uint32_t, 2> biases{0xbf808000, 0xbf818000};
    std::vector<uint8_t> s(sizeof scales), b(sizeof biases);
    std::memcpy(s.data(), scales.data(), s.size());
    std::memcpy(b.data(), biases.data(), b.size());
    test::writeSyntheticShard(dir.path() / "model.safetensors", {
        {"x.weight", "U32", {1, 16}}, {"x.scales", "F32", {1, 2}, s}, {"x.biases", "F32", {1, 2}, b}});
    model::SafetensorsCheckpoint checkpoint(dir.path());
    const auto &codes = checkpoint.require("x.weight"), &scale = checkpoint.require("x.scales"),
               &bias = checkpoint.require("x.biases");
    const auto &format = kQuantFormats[GGUF_FMT_AF4G64];
    model::gguf::TensorRows rows{"x", format.ggml_type, 0, 1, 2 * format.block_bytes, {}, codes.file,
                                  {&codes, &scale, &bias}};
    model::gguf::Repack repack{.format = GGUF_FMT_AF4G64, .rows = 1, .columns = 128, .sources = {rows}};
    std::array<uint8_t, 72> result{};
    model::readRepackRows(repack, 0, 1, result.data());
    const auto half = [&](size_t offset) {
      uint16_t value;
      std::memcpy(&value, result.data() + offset, 2);
      return value;
    };
    test::require(half(0) == 0x3f80 && half(36) == 0x3f82 && half(2) == 0xbf80 && half(38) == 0xbf82,
                  "F32 affine scales and biases do not round ties to even");
    std::cout << "Inkling affine metadata rounding PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
