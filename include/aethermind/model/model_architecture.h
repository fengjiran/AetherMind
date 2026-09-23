#ifndef AETHERMIND_MODEL_MODEL_ARCHITECTURE_H
#define AETHERMIND_MODEL_MODEL_ARCHITECTURE_H

/// @file model_architecture.h
/// @brief Model family identification for HF → semantic graph conversion.

#include "aethermind/base/macros.h"
#include "aethermind/model/formats/hf/hf_model_config.h"

#include <cstdint>
#include <string_view>

namespace aethermind {

/// @brief Model family variants supported by graph construction.
///
/// The underlying type is fixed to uint8_t to keep the enum compact and any
/// future serialization stable. P1 recognizes only dense Llama models;
/// remaining enumerators are reserved for Qwen dense (P2) and MoE families
/// (P3).
enum class ModelArchitecture : uint8_t {
    kLlamaDense = 0,
    kQwenDense = 1,  // Reserved (P2).
    kQwenMoe = 2,    // Reserved (P3).
    kMixtralMoe = 3, // Reserved (P3).
    kUnknown = 255,
};

/// @brief Maps a parsed HF config to its model family.
///
/// Single authority for family identification: the per-family graph builders
/// and their dispatch entry point consume this result instead of re-reading HF
/// strings. P1 recognizes only `model_type == "llama"`; every other spelling
/// maps to kUnknown and is rejected by the dispatch layer.
AM_NODISCARD ModelArchitecture ParseModelArchitecture(
        const HfModelConfig& config) noexcept;

/// @brief Returns a human-readable spelling of a ModelArchitecture.
constexpr std::string_view ToString(ModelArchitecture architecture) noexcept {
    switch (architecture) {
        case ModelArchitecture::kLlamaDense:
            return "llama_dense";
        case ModelArchitecture::kQwenDense:
            return "qwen_dense";
        case ModelArchitecture::kQwenMoe:
            return "qwen_moe";
        case ModelArchitecture::kMixtralMoe:
            return "mixtral_moe";
        case ModelArchitecture::kUnknown:
            return "unknown";
    }
    return "unknown";
}

} // namespace aethermind

#endif // AETHERMIND_MODEL_MODEL_ARCHITECTURE_H