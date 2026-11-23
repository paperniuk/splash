#include "model/InklingImage.hpp"

#include "model/Inkling.hpp"
#include "model/WeightStore.hpp"

#include <string>
#include <vector>

namespace splash::model {
namespace {

class Builder final {
public:
  Builder(const SafetensorsCheckpoint &checkpoint, std::string name, uint32_t layer, uint32_t type)
      : checkpoint_(checkpoint), image_(std::move(name), layer, type) {}

  void stored(const std::string &name, const char *dtype, std::vector<uint64_t> shape) {
    const auto &tensor = require(name, dtype, shape);
    image_.copy({name, dtype == std::string_view("F32") ? ggml::kF32 : ggml::kBF16,
                 tensor.offset, 1, tensor.bytes, {}, tensor.file});
  }

  void floatRows(const std::string &name, uint64_t rows, uint64_t columns) {
    const auto &tensor = require(name, "BF16", {rows, columns});
    image_.descriptor(ggml::kBF16, rows, columns, {}, {tensor.bytes, 0, 0}, name);
    image_.copy({name, ggml::kBF16, tensor.offset, rows, tensor.bytes / rows, {}, tensor.file});
  }

  void projection(const std::string &name, uint64_t rows, uint64_t columns, uint64_t experts = 1) {
    auto source = quantized(name, rows, columns, experts);
    auto repack = experts == 1 ? image_.planes(GGUF_FMT_AF4G64, rows, columns, name)
                               : image_.expertPlanes(GGUF_FMT_AF4G64, experts * rows, columns, name);
    repack.sources.push_back(std::move(source));
    if (experts == 1) image_.repack(std::move(repack));
    else image_.expertRepack(std::move(repack));
  }

  // Tensors of `columns` columns repacked as one, their rows in order.
  void fusedProjection(const std::string &name, std::initializer_list<std::pair<std::string, uint64_t>> tensors,
                       uint64_t columns) {
    uint64_t rows = 0;
    for (const auto &[tensor, count] : tensors) rows += count;
    auto repack = image_.planes(GGUF_FMT_AF4G64, rows, columns, name);
    for (const auto &[tensor, count] : tensors) repack.sources.push_back(quantized(tensor, count, columns, 1));
    image_.repack(std::move(repack));
  }

  void embedding(const std::string &name, uint64_t rows, uint64_t columns) {
    auto source = quantized(name, rows, columns, 1);
    image_.descriptor(source.type, rows, columns, {}, {rows * source.rowBytes, 0, 0}, name);
    image_.copy(std::move(source));
  }

  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const SourceTensor &require(const std::string &name, std::string_view dtype,
                              const std::vector<uint64_t> &shape) const {
    const auto &tensor = checkpoint_.require(name);
    if (tensor.dtype != dtype || tensor.shape != shape)
      throw WeightStoreError("source tensor type or shape does not match: " + name);
    return tensor;
  }

  gguf::TensorRows quantized(const std::string &name, uint64_t rows, uint64_t columns, uint64_t experts) const {
    std::vector<uint64_t> codeShape{rows, columns / 8}, metaShape{rows, columns / 64};
    if (experts != 1) {
      codeShape.insert(codeShape.begin(), experts);
      metaShape.insert(metaShape.begin(), experts);
    }
    const auto &codes = require(name + ".weight", "U32", codeShape);
    const auto &scales = checkpoint_.require(name + ".scales");
    if ((scales.dtype != "BF16" && scales.dtype != "F32") || scales.shape != metaShape)
      throw WeightStoreError("source tensor type or shape does not match: " + name + ".scales");
    const auto &biases = require(name + ".biases", scales.dtype, metaShape);
    const auto &format = kQuantFormats[GGUF_FMT_AF4G64];
    return {name, format.ggml_type, 0, experts * rows, ggufRowBytes(format, columns), {}, codes.file,
            {&codes, &scales, &biases}};
  }

  const SafetensorsCheckpoint &checkpoint_;
  gguf::ImageBuilder image_;
};

} // namespace

gguf::Image planInklingLayer(const SafetensorsCheckpoint &checkpoint, uint32_t layer) {
  if (layer >= InklingLayout::storedLayers) throw WeightStoreError("Inkling layer is out of range");
  const bool sliding = InklingLayout::isSlidingLayer(layer);
  const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
  const std::string a = p + "self_attn.", m = p + "mlp.";
  Builder b(checkpoint, "layer-" + std::to_string(layer) + ".bin", layer, 1);
  if (sliding) b.stored(p + "input_layernorm.weight", "BF16", {2304});
  else for (const char *n : {"q", "k", "v"}) b.stored(p + n + "_input_layernorm.weight", "BF16", {2304});
  // q, r and gate read the same rows, and a sliding layer's k and v too: one
  // tensor of their rows (Inkling.cpp).
  if (sliding) {
    b.fusedProjection(a + "q_r_gate_k_v_proj", {{a + "q_proj", 3072}, {a + "r_proj", 384}, {a + "gate_proj", 3072},
                                                {a + "k_proj", 1024}, {a + "v_proj", 1024}}, 2304);
  } else {
    b.fusedProjection(a + "q_r_gate_proj", {{a + "q_proj", 3072}, {a + "r_proj", 384}, {a + "gate_proj", 3072}},
                      2304);
    for (const auto &[name, width] : {std::pair{"k_proj", 1024}, {"v_proj", 1024}})
      b.projection(a + name, width, 2304);
  }
  b.projection(a + "o_proj", 2304, 3072);
  b.floatRows(a + "sink_proj.weight", 24, 2304);
  b.stored(a + "q_scale", "BF16", {24, 128});
  b.stored(a + "rel_scale", "BF16", {24, 16});
  b.stored(a + "rel_proj", "BF16", {16, 1024});
  for (const char *name : {"k_sconv", "v_sconv"}) {
    if (sliding) b.stored(a + name + ".conv.weight", "BF16", {1024, 4, 1});
    else {
      b.stored(a + name + ".static_weight", "BF16", {4, 1024});
      b.floatRows(a + name + ".weight_proj.weight", 32, 2304);
    }
  }
  b.stored(p + "attn_sconv.conv.weight", "BF16", {2304, 4, 1});
  b.stored(p + "mlp_sconv.conv.weight", "BF16", {2304, 4, 1});
  if (layer) b.stored(p + "attention_input.query", "F32", sliding ? std::vector<uint64_t>{2304} : std::vector<uint64_t>{3, 2304});
  b.stored(p + "mlp_input.query", "F32", {2304});
  if (layer >= InklingLayout::coreBegin && layer < InklingLayout::coreEnd)
    for (const char *name : {"attn", "k", "v", "mlp"}) {
      const std::string c = p + "controllers." + name + ".";
      b.stored(c + "down_weight", "BF16", {64, 4608});
      b.stored(c + "forget_weight", "BF16", {2304, 64});
      b.stored(c + "gain_weight", "BF16", {2304, 64});
    }
  b.stored(p + "post_attention_layernorm.weight", "BF16", {2304});
  b.floatRows(m + "gate_weight", 128, 2304);
  b.stored(m + "e_score_correction_bias", "F32", {128});
  b.projection(m + "switch_mlp.gate_proj", 1152, 2304, 128);
  b.projection(m + "switch_mlp.up_proj", 1152, 2304, 128);
  b.projection(m + "switch_mlp.down_proj", 2304, 1152, 128);
  return b.finish();
}

gguf::Image planInklingHead(const SafetensorsCheckpoint &checkpoint) {
  Builder b(checkpoint, "head.bin", 24, 2);
  b.stored("language_model.model.norm.weight", "BF16", {2304});
  b.stored("language_model.model.final_input.query", "F32", {2304});
  b.stored("language_model.model.message_queries", "F32", {12, 4, 2304});
  b.projection("language_model.lm_head", 201024, 2304);
  return b.finish();
}

gguf::Image planInklingEmbedding(const SafetensorsCheckpoint &checkpoint) {
  Builder b(checkpoint, "embedding.bin", 201024, 2304);
  b.stored("language_model.model.embed_norm.weight", "BF16", {2304});
  b.embedding("language_model.model.embed_tokens", 201024, 2304);
  return b.finish();
}

} // namespace splash::model
