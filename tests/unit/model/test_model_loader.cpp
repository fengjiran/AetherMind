#include "aethermind/model/model_loader.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/model/model_instance.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {
namespace {

class TempDirectory {
public:
    TempDirectory() {
        static size_t counter = 0;
        const auto unique_id = std::to_string(
                                       std::chrono::steady_clock::now().time_since_epoch().count()) +
                               "_" + std::to_string(counter++);
        path_ = std::filesystem::temp_directory_path() /
                ("aethermind_model_loader_" + unique_id);
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    TempDirectory(TempDirectory&&) = delete;
    TempDirectory& operator=(TempDirectory&&) = delete;

    AM_NODISCARD const std::filesystem::path& Path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_{};
};

std::array<std::byte, sizeof(uint64_t)> EncodeLittleEndianU64(uint64_t value) {
    std::array<std::byte, sizeof(uint64_t)> bytes{};
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
    }
    return bytes;
}

std::vector<std::byte> ToBytes(std::string_view text) {
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

std::vector<std::byte> FloatArrayToBytes(std::span<const float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

void WriteTextFile(const std::filesystem::path& path, std::string_view content) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    stream << content;
}

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

std::string MakeCompleteTensorHeader(int64_t num_layers) {
    std::string header = "{";
    header += R"("model.embed_tokens.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)";
    header += R"("model.norm.weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]})";

    size_t offset = 8;
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
            header += ",";
            header += "\"" + prefix + suffix + "\":";
            header += "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
                      std::to_string(offset) + "," + std::to_string(offset + 4) + "]}";
            offset += 4;
        }
    }
    header += "}";
    return header;
}

void WriteRawFile(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
}

void WriteSafetensorsFile(
        const std::filesystem::path& path,
        std::string_view header_json,
        std::span<const std::byte> raw_bytes) {
    const auto prefix = EncodeLittleEndianU64(header_json.size());
    const auto header_bytes = ToBytes(header_json);

    std::vector<std::byte> file_bytes;
    file_bytes.reserve(prefix.size() + header_bytes.size() + raw_bytes.size());
    file_bytes.insert(file_bytes.end(), prefix.begin(), prefix.end());
    file_bytes.insert(file_bytes.end(), header_bytes.begin(), header_bytes.end());
    file_bytes.insert(file_bytes.end(), raw_bytes.begin(), raw_bytes.end());
    WriteRawFile(path, file_bytes);
}

TEST(ModelLoaderTest, ValidSingleFileDirectoryReachesModelInstanceBoundary) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.Path() / "config.json", MakeMinimalLlamaConfigJson());
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 11>{});
    WriteSafetensorsFile(
            temp_dir.Path() / "model.safetensors",
            MakeCompleteTensorHeader(1),
            raw_bytes);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.Path()}, backend, registry);

    ASSERT_TRUE(model.ok()) << model.status().message();
    ASSERT_NE(*model, nullptr);

    const auto& config = (*model)->GetConfig();
    EXPECT_EQ(config.hidden_size, 64);
    EXPECT_EQ(config.num_hidden_layers, 1);
    EXPECT_EQ(config.num_attention_heads, 8);
    EXPECT_EQ(config.num_key_value_heads, 4);
    EXPECT_EQ(config.vocab_size, 1000);

    const auto& weight_index = (*model)->GetRawWeightIndex();
    ASSERT_EQ(weight_index.layers.size(), 1);
    EXPECT_TRUE(weight_index.embed_tokens.IsValid());
    EXPECT_TRUE(weight_index.final_norm.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.k_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.v_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.o_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].ffn.gate_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].ffn.up_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].ffn.down_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].norm.input_rmsnorm.IsValid());
    EXPECT_TRUE(weight_index.layers[0].norm.post_attn_rmsnorm.IsValid());

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

TEST(ModelLoaderTest, RejectsUnsupportedModelFamily) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.Path() / "config.json", R"({
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
    WriteTextFile(temp_dir.Path() / "model.safetensors", "dummy");

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.Path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoaderTest, PropagatesSafetensorsArtifactError) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.Path() / "config.json", MakeMinimalLlamaConfigJson());
    const auto prefix = EncodeLittleEndianU64(1024);
    WriteRawFile(temp_dir.Path() / "model.safetensors", prefix);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.Path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoaderTest, RejectsIncompleteWeightSet) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.Path() / "config.json", MakeMinimalLlamaConfigJson());
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{});
    WriteSafetensorsFile(
            temp_dir.Path() / "model.safetensors",
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.Path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
