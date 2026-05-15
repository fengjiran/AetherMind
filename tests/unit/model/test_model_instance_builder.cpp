#include "aethermind/model/formats/hf/hf_weight_resolver.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/model/model_instance_builder.h"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace aethermind {
namespace {

struct TestStorage : RawStorage {
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
    std::vector<std::byte> data;
};

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

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage,
                             size_t offset,
                             size_t nbytes,
                             DataType dtype,
                             const std::vector<int64_t>& shape) {
    return RawWeightView{
            .data = storage->data.data() + offset,
            .bytes = nbytes,
            .dtype = dtype,
            .shape = shape,
            .storage = storage,
    };
}

DecoderLayerRawWeights MakeTestLayer(const std::shared_ptr<TestStorage>& storage,
                                     size_t base_offset) {
    DecoderLayerRawWeights layer;
    layer.attn.q_proj = MakeWeightView(storage, base_offset + 0, 4, DataType::Float32(), {1});
    layer.attn.k_proj = MakeWeightView(storage, base_offset + 4, 4, DataType::Float32(), {1});
    layer.attn.v_proj = MakeWeightView(storage, base_offset + 8, 4, DataType::Float32(), {1});
    layer.attn.o_proj = MakeWeightView(storage, base_offset + 12, 4, DataType::Float32(), {1});
    layer.ffn.gate_proj = MakeWeightView(storage, base_offset + 16, 4, DataType::Float32(), {1});
    layer.ffn.up_proj = MakeWeightView(storage, base_offset + 20, 4, DataType::Float32(), {1});
    layer.ffn.down_proj = MakeWeightView(storage, base_offset + 24, 4, DataType::Float32(), {1});
    layer.norm.input_rmsnorm = MakeWeightView(storage, base_offset + 28, 4, DataType::Float32(), {1});
    layer.norm.post_attn_rmsnorm = MakeWeightView(storage, base_offset + 32, 4, DataType::Float32(), {1});
    return layer;
}

TEST(ModelInstanceBuilderTest, CreatesModelInstanceWithConfigAndWeights) {
    auto storage = std::make_shared<TestStorage>(128);
    for (size_t i = 0; i < storage->data.size(); ++i) {
        storage->data[i] = static_cast<std::byte>(i);
    }

    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 4, DataType::Float32(), {1});
    index.final_norm = MakeWeightView(storage, 4, 4, DataType::Float32(), {1});
    index.layers.push_back(MakeTestLayer(storage, 8));

    auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(1), std::move(index));
    ASSERT_TRUE(model.ok());
    ASSERT_NE(*model, nullptr);

    const auto& config = (*model)->GetConfig();
    EXPECT_EQ(config.hidden_size, 64);
    EXPECT_EQ(config.num_hidden_layers, 1);

    const auto& weight_index = (*model)->GetRawWeightIndex();
    ASSERT_EQ(weight_index.layers.size(), 1);
    EXPECT_TRUE(weight_index.embed_tokens.IsValid());
    EXPECT_TRUE(weight_index.final_norm.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].ffn.gate_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].norm.input_rmsnorm.IsValid());
}

TEST(ModelInstanceBuilderTest, ViewsPointToCorrectData) {
    auto storage = std::make_shared<TestStorage>(64);
    for (size_t i = 0; i < storage->data.size(); ++i) {
        storage->data[i] = static_cast<std::byte>(i + 1);
    }

    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 10, 4, DataType::Float32(), {1});
    index.final_norm = MakeWeightView(storage, 20, 4, DataType::Float32(), {1});
    index.layers.push_back(MakeTestLayer(storage, 30));

    auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(1), std::move(index));
    ASSERT_TRUE(model.ok());

    const auto& weight_index = (*model)->GetRawWeightIndex();
    EXPECT_EQ(weight_index.embed_tokens.data, storage->data.data() + 10);
    EXPECT_EQ(weight_index.embed_tokens.bytes, 4);
    EXPECT_EQ(weight_index.final_norm.data, storage->data.data() + 20);
    EXPECT_EQ(weight_index.layers[0].attn.q_proj.data, storage->data.data() + 30);
    EXPECT_EQ(weight_index.layers[0].norm.post_attn_rmsnorm.data, storage->data.data() + 62);
}

TEST(ModelInstanceBuilderTest, BackingStorageLifetime) {
    bool backing_destroyed = false;

    {
        auto storage = std::make_shared<TestStorage>(128);

        // Store a weak_ptr to observe when the last shared_ptr is released.
        std::weak_ptr<const RawStorage> weak_storage = storage;

        ModelWeightIndex index;
        index.embed_tokens = MakeWeightView(storage, 0, 4, DataType::Float32(), {1});
        index.final_norm = MakeWeightView(storage, 4, 4, DataType::Float32(), {1});
        index.layers.push_back(MakeTestLayer(storage, 8));

        auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(1), std::move(index));
        ASSERT_TRUE(model.ok());

        // Release the local shared_ptr.  ModelInstance must hold its own copy.
        storage.reset();
        ASSERT_FALSE(weak_storage.expired())
                << "Backing storage was destroyed while ModelInstance still references it";

        // Views inside ModelInstance must still be valid and point to live memory.
        const auto& weight_index = (*model)->GetRawWeightIndex();
        EXPECT_TRUE(weight_index.embed_tokens.IsValid());
        EXPECT_EQ(weight_index.embed_tokens.data[0], std::byte{0});
        EXPECT_TRUE(weight_index.layers[0].attn.q_proj.IsValid());

        // Destroy the ModelInstance; this should release the last shared_ptr.
        model.value().reset();
        backing_destroyed = weak_storage.expired();
    }

    EXPECT_TRUE(backing_destroyed)
            << "Backing storage was not destroyed after ModelInstance went out of scope";
}

TEST(ModelInstanceBuilderTest, ResolvedWeightsIntegratedCorrectly) {
    auto storage = std::make_shared<TestStorage>(256);

    RawWeightTable raw_table;
    size_t offset = 0;
    const auto add = [&](std::string name) {
        raw_table.emplace(std::move(name),
                          MakeWeightView(storage, offset, 4, DataType::Float32(), {1}));
        offset += 4;
    };

    add("model.embed_tokens.weight");
    add("model.norm.weight");
    add("model.layers.0.self_attn.q_proj.weight");
    add("model.layers.0.self_attn.k_proj.weight");
    add("model.layers.0.self_attn.v_proj.weight");
    add("model.layers.0.self_attn.o_proj.weight");
    add("model.layers.0.mlp.gate_proj.weight");
    add("model.layers.0.mlp.up_proj.weight");
    add("model.layers.0.mlp.down_proj.weight");
    add("model.layers.0.input_layernorm.weight");
    add("model.layers.0.post_attention_layernorm.weight");

    auto resolved = hf::ResolveWeights(MakeLlamaConfig(1), raw_table);
    ASSERT_TRUE(resolved.ok());

    auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(1), std::move(*resolved));
    ASSERT_TRUE(model.ok());

    const auto& weight_index = (*model)->GetRawWeightIndex();
    ASSERT_EQ(weight_index.layers.size(), 1);
    EXPECT_TRUE(weight_index.embed_tokens.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].ffn.down_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].norm.input_rmsnorm.IsValid());
}

}// namespace
}// namespace aethermind
