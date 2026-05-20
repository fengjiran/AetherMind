#include "aethermind/model/model_loader.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/model/model_instance.h"
#include "test_utils.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {
namespace {

std::string MakeMinimalLlamaConfigJson() {
    return R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false
    })";
}

std::vector<std::string> MakeCompleteTensorNames(int64_t num_layers) {
    std::vector<std::string> names{
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
    };
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        const std::string suffixes[] = {
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
        };
        for (const auto& suffix: suffixes) {
            names.push_back(prefix + suffix);
        }
    }
    return names;
}

bool IsNormTensorName(std::string_view name) {
    return name == std::string_view("model.norm.weight") ||
           name.find("layernorm.weight") != std::string_view::npos;
}

std::string MakeTensorHeaderForNames(std::span<const std::string> names) {
    std::string header = "{";
    size_t offset = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            header += ",";
        }
        header += "\"" + names[i] + "\":";
        header += R"({"dtype":"F32","shape":)";
        header += IsNormTensorName(names[i]) ? "[1]" : "[1,1]";
        header += ",\"data_offsets\":[" +
                  std::to_string(offset) + "," + std::to_string(offset + 4) + "]}";
        offset += 4;
    }
    header += "}";
    return header;
}

std::string MakeCompleteTensorHeader(int64_t num_layers) {
    const auto names = MakeCompleteTensorNames(num_layers);
    return MakeTensorHeaderForNames(names);
}

struct ShardTensorNames {
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::span<const std::string> names;
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::string_view filename;
};

std::string MakeShardedWeightMapJson(const ShardTensorNames& first_shard,
                                     const ShardTensorNames& second_shard) {
    std::string json = R"({"metadata":{"total_size":48},"weight_map":{)";
    bool first = true;
    const auto append = [&](const ShardTensorNames& shard) {
        for (const auto& name: shard.names) {
            if (!first) {
                json += ",";
            }
            first = false;
            json += "\"" + name + "\":\"" + std::string(shard.filename) + "\"";
        }
    };
    append(first_shard);
    append(second_shard);
    json += "}}";
    return json;
}

std::vector<std::byte> ZeroFloatBytes(size_t count) {
    std::vector<float> values(count, 0.0f);
    return FloatArrayToBytes(values);
}

TEST(ModelLoader_PipelineTest, ValidSingleFileDirectoryReachesModelInstanceBoundary) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", MakeMinimalLlamaConfigJson());
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 12>{});
    WriteSafetensorsFile(temp_dir.path() / "model.safetensors",
                         MakeCompleteTensorHeader(1),
                         raw_bytes);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.path()}, backend, registry);

    ASSERT_TRUE(model.ok()) << model.status().message();
    ASSERT_NE(*model, nullptr);

    const auto& config = (*model)->GetConfig();
    EXPECT_EQ(config.hidden_size, 64);
    EXPECT_EQ(config.num_hidden_layers, 1);
    EXPECT_EQ(config.num_attention_heads, 8);
    EXPECT_EQ(config.num_key_value_heads, 4);
    EXPECT_EQ(config.vocab_size, 1000);

    const auto& resolved_weights = (*model)->GetResolvedWeights();
    ASSERT_EQ(resolved_weights.layers.size(), 1);
    EXPECT_TRUE(resolved_weights.embed_tokens.IsValid());
    EXPECT_TRUE(resolved_weights.final_norm.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].attn.k_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].attn.v_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].attn.o_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].mlp.gate_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].mlp.up_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].mlp.down_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].norm.input_rmsnorm.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].norm.post_attn_rmsnorm.IsValid());

    const KernelSelector expected_selector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
    const PackedWeights* packed = (*model)->FindPackedWeights(
            OpType::kLinear, expected_selector);
    ASSERT_NE(packed, nullptr);
    EXPECT_EQ(packed->op_type(), OpType::kLinear);
    EXPECT_EQ(packed->selector(), expected_selector);
    EXPECT_TRUE(packed->storage().is_initialized());
    EXPECT_GT(packed->storage().nbytes(), 0U);
}

TEST(ModelLoader_PipelineTest, ValidShardedDirectoryReachesModelInstanceBoundary) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", MakeMinimalLlamaConfigJson());

    const auto names = MakeCompleteTensorNames(1);
    const auto split = names.begin() + static_cast<std::ptrdiff_t>(names.size() / 2);
    const std::vector<std::string> first_shard_names(names.begin(), split);
    const std::vector<std::string> second_shard_names(split, names.end());

    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  MakeShardedWeightMapJson(
                          ShardTensorNames{.names = first_shard_names,
                                           .filename = "model-00001-of-00002.safetensors"},
                          ShardTensorNames{.names = second_shard_names,
                                           .filename = "model-00002-of-00002.safetensors"}));
    WriteSafetensorsFile(temp_dir.path() / "model-00001-of-00002.safetensors",
                         MakeTensorHeaderForNames(first_shard_names),
                         ZeroFloatBytes(first_shard_names.size()));
    WriteSafetensorsFile(temp_dir.path() / "model-00002-of-00002.safetensors",
                         MakeTensorHeaderForNames(second_shard_names),
                         ZeroFloatBytes(second_shard_names.size()));

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.path()}, backend, registry);

    ASSERT_TRUE(model.ok()) << model.status().message();
    ASSERT_NE(*model, nullptr);

    const auto& resolved_weights = (*model)->GetResolvedWeights();
    ASSERT_EQ(resolved_weights.layers.size(), 1);
    EXPECT_TRUE(resolved_weights.embed_tokens.IsValid());
    EXPECT_TRUE(resolved_weights.final_norm.IsValid());
    ASSERT_TRUE(resolved_weights.lm_head.has_value());
    EXPECT_TRUE(resolved_weights.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(resolved_weights.layers[0].mlp.down_proj.IsValid());

    const KernelSelector expected_selector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
    const PackedWeights* packed = (*model)->FindPackedWeights(
            OpType::kLinear, expected_selector);
    ASSERT_NE(packed, nullptr);
    EXPECT_TRUE(packed->storage().is_initialized());
    EXPECT_GT(packed->storage().nbytes(), 0U);
}

TEST(ModelLoader_PipelineTest, RejectsUnsupportedModelFamily) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", R"({
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6
    })");
    WriteTextFile(temp_dir.path() / "model.safetensors", "dummy");

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_PipelineTest, PropagatesSafetensorsArtifactError) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", MakeMinimalLlamaConfigJson());
    const auto prefix = EncodeLittleEndianU64(1024);
    WriteRawFile(temp_dir.path() / "model.safetensors", prefix);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_PipelineTest, RejectsIncompleteWeightSet) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", MakeMinimalLlamaConfigJson());
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{});
    WriteSafetensorsFile(temp_dir.path() / "model.safetensors",
                         R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                         raw_bytes);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
