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

/// @brief Byte-backed storage for a fabricated tiny Llama checkpoint.
///
/// Weights must own real bytes. Shape-only placeholders (null data, zero bytes)
/// pass graph construction but fail ValidateRawWeightView, which the production
/// preparation path enforces before binding a weight.
struct LlamaCheckpointStorage : RawStorage {
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

/// @brief A fabricated checkpoint: owned byte storage plus the weights carved
///        out of it.
struct TinyLlamaCheckpoint {
    std::shared_ptr<LlamaCheckpointStorage> storage{};
    ResolvedModelWeights weights{};
};

/// @brief Carves every Llama weight out of one zeroed, 64-byte aligned buffer.
///
/// A tied config leaves `lm_head` empty, mirroring the HF resolver: resolution
/// must then fall back to embed_tokens.
inline TinyLlamaCheckpoint MakeTinyLlamaCheckpoint(const HfModelConfig& config) {
    const int64_t head_dim = config.head_dim != 0
                                     ? config.head_dim
                                     : config.hidden_size / config.num_attention_heads;
    const int64_t q_hidden = config.num_attention_heads * head_dim;
    const int64_t kv_hidden = config.num_key_value_heads * head_dim;

    TinyLlamaCheckpoint checkpoint;
    checkpoint.storage = std::make_shared<LlamaCheckpointStorage>();
    ResolvedModelWeights& weights = checkpoint.weights;
    weights.layers.resize(static_cast<size_t>(config.num_hidden_layers));
    RawWeightView lm_head_slot{};

    // Addresses stay valid: `weights.layers` is sized once, before being handed
    // out, and the buffer is sized before any data pointer is recorded.
    std::vector<std::pair<RawWeightView*, std::vector<int64_t>>> pending;
    pending.push_back({&weights.embed_tokens, {config.vocab_size, config.hidden_size}});
    pending.push_back({&weights.final_norm, {config.hidden_size}});
    if (!config.tie_word_embeddings) {
        pending.push_back({&lm_head_slot, {config.vocab_size, config.hidden_size}});
    }
    for (auto& layer: weights.layers) {
        pending.push_back({&layer.norm.input_rmsnorm, {config.hidden_size}});
        pending.push_back({&layer.norm.post_attn_rmsnorm, {config.hidden_size}});
        pending.push_back({&layer.attn.q_proj, {q_hidden, config.hidden_size}});
        pending.push_back({&layer.attn.k_proj, {kv_hidden, config.hidden_size}});
        pending.push_back({&layer.attn.v_proj, {kv_hidden, config.hidden_size}});
        pending.push_back({&layer.attn.o_proj, {config.hidden_size, q_hidden}});
        pending.push_back({&layer.mlp.gate_proj, {config.intermediate_size, config.hidden_size}});
        pending.push_back({&layer.mlp.up_proj, {config.intermediate_size, config.hidden_size}});
        pending.push_back({&layer.mlp.down_proj, {config.hidden_size, config.intermediate_size}});
    }

    constexpr size_t kAlignment = 64;
    const auto align_up = [](size_t offset) {
        return (offset + kAlignment - 1) & ~(kAlignment - 1);
    };
    const auto byte_size = [](const std::vector<int64_t>& shape) {
        int64_t count = 1;
        for (const int64_t dim: shape) {
            count *= dim;
        }
        return static_cast<size_t>(count) * sizeof(float);
    };

    size_t total = 0;
    for (const auto& entry: pending) {
        total = align_up(total) + byte_size(entry.second);
    }
    checkpoint.storage->data.resize(total, std::byte{0});

    size_t cursor = 0;
    for (const auto& [target, shape]: pending) {
        cursor = align_up(cursor);
        *target = RawWeightView{
                .data = checkpoint.storage->data.data() + cursor,
                .bytes = byte_size(shape),
                .dtype = DataType::Float32(),
                .shape = shape,
                .storage = checkpoint.storage,
                .is_contiguous = true,
        };
        cursor += byte_size(shape);
    }

    if (!config.tie_word_embeddings) {
        weights.lm_head = lm_head_slot;
    }
    return checkpoint;
}

} // namespace aethermind::test

#endif
