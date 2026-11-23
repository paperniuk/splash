#pragma once

#include "model/GgufImage.hpp"
#include "model/SafetensorsCheckpoint.hpp"

namespace splash::model {

[[nodiscard]] gguf::Image planInklingLayer(const SafetensorsCheckpoint &checkpoint, uint32_t layer);
[[nodiscard]] gguf::Image planInklingHead(const SafetensorsCheckpoint &checkpoint);
[[nodiscard]] gguf::Image planInklingEmbedding(const SafetensorsCheckpoint &checkpoint);

} // namespace splash::model
