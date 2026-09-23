#ifndef AETHERMIND_TEST_LLAMA_CHECKPOINT_HELPERS_H
#define AETHERMIND_TEST_LLAMA_CHECKPOINT_HELPERS_H

#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/model/resolved_model_weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace aethermind::test {

/// @brief Byte-backed storage for one fabricated weight.
///
/// Weights must own real bytes. Shape-only placeholders (null data, zero bytes)
/// pass graph construction but fail ValidateRawWeightView, which the production
/// preparation path enforces before binding a weight.
struct LlamaWeightStorage : RawStorage {
    std::vector<std::byte> data{};
};

/// @brief Tiny GQA Llama config: 4 attention heads over 2 KV heads, head_dim 2.
inline HfModelConfig MakeTinyLlamaConfig(int64_t num_layers, bool tie_word_embeddings) {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 8,
            .intermediate_size = 16,
            .num_hidden_layers = num_layers,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .vocab_size = 32,
            .max_position_embeddings = 128,
            .head_dim = 2,
            .rms_norm_eps = 1.0e-5,
            .hidden_act = "silu",
            .tie_word_embeddings = tie_word_embeddings,
            .weight_dtype_hint = DataType::Float32(),
    };
}

/// @brief Carves zeroed, data-backed weight views with distinct backing each.
///
/// One allocation per weight keeps every data pointer unique, so tests can tell
/// two weights apart by address, and keeps the storage alive through the view's
/// own shared_ptr. Each view's data starts 64-byte aligned.
class RawWeightCarver {
public:
    RawWeightView Carve(std::vector<int64_t> shape) {
        constexpr size_t kAlignment = 64;
        int64_t count = 1;
        for (const int64_t dim: shape) {
            count *= dim;
        }
        const size_t bytes = static_cast<size_t>(count) * sizeof(float);
        auto storage = std::make_shared<LlamaWeightStorage>();
        storage->data.resize(bytes + kAlignment, std::byte{0});
        const auto base = reinterpret_cast<uintptr_t>(storage->data.data());
        const size_t padding = static_cast<size_t>((kAlignment - (base % kAlignment)) % kAlignment);
        return RawWeightView{
                .data = storage->data.data() + padding,
                .bytes = bytes,
                .dtype = DataType::Float32(),
                .shape = std::move(shape),
                .storage = storage,
                .is_contiguous = true,
        };
    }
};

/// @brief A fabricated checkpoint: resolved weights, each owning its bytes.
struct TinyLlamaCheckpoint {
    ResolvedModelWeights weights{};
};

/// @brief Builds every Llama weight for `config`.
///
/// A tied config leaves `lm_head` empty, mirroring the HF resolver: resolution
/// must then fall back to embed_tokens.
inline TinyLlamaCheckpoint MakeTinyLlamaCheckpoint(const HfModelConfig& config) {
    const int64_t head_dim = config.head_dim != 0
                                     ? config.head_dim
                                     : config.hidden_size / config.num_attention_heads;
    const int64_t q_hidden = config.num_attention_heads * head_dim;
    const int64_t kv_hidden = config.num_key_value_heads * head_dim;

    RawWeightCarver carver;
    TinyLlamaCheckpoint checkpoint;
    ResolvedModelWeights& weights = checkpoint.weights;
    weights.embed_tokens = carver.Carve({config.vocab_size, config.hidden_size});
    weights.final_norm = carver.Carve({config.hidden_size});
    if (!config.tie_word_embeddings) {
        weights.lm_head = carver.Carve({config.vocab_size, config.hidden_size});
    }

    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        DecoderLayerRawWeights layer;
        layer.norm.input_rmsnorm = carver.Carve({config.hidden_size});
        layer.norm.post_attn_rmsnorm = carver.Carve({config.hidden_size});
        layer.attn.q_proj = carver.Carve({q_hidden, config.hidden_size});
        layer.attn.k_proj = carver.Carve({kv_hidden, config.hidden_size});
        layer.attn.v_proj = carver.Carve({kv_hidden, config.hidden_size});
        layer.attn.o_proj = carver.Carve({config.hidden_size, q_hidden});
        layer.mlp.gate_proj = carver.Carve({config.intermediate_size, config.hidden_size});
        layer.mlp.up_proj = carver.Carve({config.intermediate_size, config.hidden_size});
        layer.mlp.down_proj = carver.Carve({config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(layer));
    }
    return checkpoint;
}

} // namespace aethermind::test

#endif
