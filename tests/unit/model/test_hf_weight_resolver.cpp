#include "aethermind/model/formats/hf/hf_weight_resolver.h"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>

namespace aethermind {
namespace {

constexpr size_t kMaxTestTensors = 64;
const std::array<std::byte, kMaxTestTensors> kTensorMarkers{};

HfModelConfig MakeLlamaConfig(int64_t num_layers) {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 64,
            .intermediate_size = 256,
            .num_hidden_layers = num_layers,
            .num_attention_heads = 8,
            .num_key_value_heads = 4,
            .vocab_size = 1000,
            .rms_norm_eps = 1e-6,
            .tie_word_embeddings = false,
    };
}

void AddTensor(RawWeightTable* tensors, std::string name, size_t marker_index) {
    AM_CHECK(marker_index < kTensorMarkers.size());
    tensors->emplace(std::move(name), RawWeightView{
                                              .data = &kTensorMarkers[marker_index],
                                              .bytes = marker_index + 1,
                                      });
}

RawWeightTable MakeCompleteTensorSet(int64_t num_layers, bool include_lm_head = false) {
    RawWeightTable tensors;
    size_t marker_index = 0;
    AddTensor(&tensors, "model.embed_tokens.weight", marker_index++);
    AddTensor(&tensors, "model.norm.weight", marker_index++);
    if (include_lm_head) {
        AddTensor(&tensors, "lm_head.weight", marker_index++);
    }

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        AddTensor(&tensors, prefix + ".self_attn.q_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".self_attn.k_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".self_attn.v_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".self_attn.o_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".mlp.gate_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".mlp.up_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".mlp.down_proj.weight", marker_index++);
        AddTensor(&tensors, prefix + ".input_layernorm.weight", marker_index++);
        AddTensor(&tensors, prefix + ".post_attention_layernorm.weight", marker_index++);
    }
    return tensors;
}

void ExpectSameView(const RawWeightView& actual, const RawWeightView& expected) {
    EXPECT_EQ(actual.data, expected.data);
    EXPECT_EQ(actual.bytes, expected.bytes);
}

TEST(ModelLoader_HfWeightResolverTest, ResolvesSingleLayerDenseLlamaWeights) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const RawWeightTable tensors = MakeCompleteTensorSet(config.num_hidden_layers);

    const auto resolved = hf::ResolveWeights(config, tensors);

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_EQ(resolved->NumLayers(), 1);
    EXPECT_FALSE(resolved->lm_head.has_value());
    ExpectSameView(resolved->embed_tokens, tensors.at("model.embed_tokens.weight"));
    ExpectSameView(resolved->final_norm, tensors.at("model.norm.weight"));
    ExpectSameView(resolved->layers[0].attn.q_proj, tensors.at("model.layers.0.self_attn.q_proj.weight"));
    ExpectSameView(resolved->layers[0].attn.k_proj, tensors.at("model.layers.0.self_attn.k_proj.weight"));
    ExpectSameView(resolved->layers[0].attn.v_proj, tensors.at("model.layers.0.self_attn.v_proj.weight"));
    ExpectSameView(resolved->layers[0].attn.o_proj, tensors.at("model.layers.0.self_attn.o_proj.weight"));
    ExpectSameView(resolved->layers[0].ffn.gate_proj, tensors.at("model.layers.0.mlp.gate_proj.weight"));
    ExpectSameView(resolved->layers[0].ffn.up_proj, tensors.at("model.layers.0.mlp.up_proj.weight"));
    ExpectSameView(resolved->layers[0].ffn.down_proj, tensors.at("model.layers.0.mlp.down_proj.weight"));
    ExpectSameView(resolved->layers[0].norm.input_rmsnorm, tensors.at("model.layers.0.input_layernorm.weight"));
    ExpectSameView(resolved->layers[0].norm.post_attn_rmsnorm, tensors.at("model.layers.0.post_attention_layernorm.weight"));
}

TEST(ModelLoader_HfWeightResolverTest, ResolvesTwoLayerDenseLlamaWeights) {
    const HfModelConfig config = MakeLlamaConfig(2);
    const RawWeightTable tensors = MakeCompleteTensorSet(config.num_hidden_layers);

    const auto resolved = hf::ResolveWeights(config, tensors);

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_EQ(resolved->NumLayers(), 2);
    ExpectSameView(resolved->layers[0].attn.q_proj, tensors.at("model.layers.0.self_attn.q_proj.weight"));
    ExpectSameView(resolved->layers[1].attn.q_proj, tensors.at("model.layers.1.self_attn.q_proj.weight"));
    ExpectSameView(resolved->layers[1].ffn.down_proj, tensors.at("model.layers.1.mlp.down_proj.weight"));
    ExpectSameView(resolved->layers[1].norm.post_attn_rmsnorm, tensors.at("model.layers.1.post_attention_layernorm.weight"));
}

TEST(ModelLoader_HfWeightResolverTest, RejectsMissingLayerAttentionWeight) {
    const HfModelConfig config = MakeLlamaConfig(2);
    RawWeightTable tensors = MakeCompleteTensorSet(config.num_hidden_layers);
    tensors.erase("model.layers.1.self_attn.q_proj.weight");

    const auto resolved = hf::ResolveWeights(config, tensors);

    ASSERT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(resolved.status().message().find("model.layers.1.self_attn.q_proj.weight"), std::string::npos);
}

TEST(ModelLoader_HfWeightResolverTest, RejectsIncompleteLayerCount) {
    const HfModelConfig config = MakeLlamaConfig(2);
    const RawWeightTable tensors = MakeCompleteTensorSet(1);

    const auto resolved = hf::ResolveWeights(config, tensors);

    ASSERT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(resolved.status().message().find("model.layers.1"), std::string::npos);
}

TEST(ModelLoader_HfWeightResolverTest, TreatsLmHeadAsOptional) {
    const HfModelConfig config = MakeLlamaConfig(1);
    RawWeightTable tensors = MakeCompleteTensorSet(config.num_hidden_layers, true);

    auto resolved = hf::ResolveWeights(config, tensors);

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    ASSERT_TRUE(resolved->lm_head.has_value());
    ExpectSameView(*resolved->lm_head, tensors.at("lm_head.weight"));

    tensors.erase("lm_head.weight");
    resolved = hf::ResolveWeights(config, tensors);

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_FALSE(resolved->lm_head.has_value());
}

}// namespace
}// namespace aethermind
