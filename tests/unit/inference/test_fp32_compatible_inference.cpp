#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/inference/model_execution_policy.h"
#include "inference/test_hf_fixed_trajectory_helpers.h"
#include "model/test_utils.h"
#include "test_utils/test_resource_allocator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
namespace numerical = aethermind::test::numerical;

struct CompatibilityCase {
    bool bfloat16;
    bool tied;
    int64_t kv_heads;
    const char* name;
};

struct CheckpointTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint16_t> bits;
};

struct CheckpointPair {
    std::filesystem::path source;
    std::filesystem::path reference;
    size_t source_payload_bytes = 0;
    size_t fp32_payload_bytes = 0;
};

// This oracle evaluates the format's value as a double, independently of the
// production integer-bit widening and Half/BFloat16 conversion helpers.
float DecodeCheckpointValue(uint16_t bits, bool bfloat16) {
    const int fraction_bits = bfloat16 ? 7 : 10;
    const int bias = bfloat16 ? 127 : 15;
    const unsigned exponent_mask = bfloat16 ? 0xFFU : 0x1FU;
    const unsigned fraction_mask = (1U << fraction_bits) - 1U;
    const unsigned exponent = (bits >> fraction_bits) & exponent_mask;
    const unsigned fraction = bits & fraction_mask;
    const double magnitude = exponent == 0
                                     ? std::ldexp(static_cast<double>(fraction), 1 - bias - fraction_bits)
                                     : std::ldexp(static_cast<double>((1U << fraction_bits) + fraction),
                                                  static_cast<int>(exponent) - bias - fraction_bits);
    return static_cast<float>((bits & 0x8000U) != 0 ? -magnitude : magnitude);
}

std::vector<CheckpointTensor> MakeCheckpointTensors(const CompatibilityCase& test_case) {
    constexpr int64_t kHidden = 8;
    constexpr int64_t kIntermediate = 12;
    constexpr int64_t kVocabulary = 32;
    constexpr int64_t kHeadDimension = 2;
    const int64_t kv_hidden = test_case.kv_heads * kHeadDimension;
    std::vector<CheckpointTensor> tensors;
    const auto append = [&](std::string name, std::vector<int64_t> shape, bool norm) {
        size_t count = 1;
        for (const int64_t dimension: shape) {
            count *= static_cast<size_t>(dimension);
        }
        CheckpointTensor tensor{.name = std::move(name), .shape = std::move(shape), .bits = std::vector<uint16_t>(count)};
        constexpr std::array<uint16_t, 11> kHalfValues{
                0x2800, 0x2C00, 0x3000, 0x3200, 0xA800, 0xAC00, 0xB000, 0xB200, 0x0000, 0x0001, 0x2E00};
        constexpr std::array<uint16_t, 11> kBfloatValues{
                0x3D00, 0x3D80, 0x3E00, 0x3E40, 0xBD00, 0xBD80, 0xBE00, 0xBE40, 0x0000, 0x0001, 0x3DC0};
        constexpr std::array<uint16_t, 3> kHalfNorm{0x3B00, 0x3C00, 0x3C80};
        constexpr std::array<uint16_t, 3> kBfloatNorm{0x3F60, 0x3F80, 0x3F90};
        const auto& values = test_case.bfloat16 ? kBfloatValues : kHalfValues;
        const auto& norms = test_case.bfloat16 ? kBfloatNorm : kHalfNorm;
        for (size_t index = 0; index < count; ++index) {
            tensor.bits[index] = norm ? norms[(index + tensors.size()) % norms.size()]
                                      : values[(index * 7 + index / 3 + tensors.size() * 5) % values.size()];
        }
        tensors.push_back(std::move(tensor));
    };
    append("model.embed_tokens.weight", {kVocabulary, kHidden}, false);
    append("model.norm.weight", {kHidden}, true);
    if (!test_case.tied) {
        append("lm_head.weight", {kVocabulary, kHidden}, false);
    }
    for (int64_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        append(prefix + ".input_layernorm.weight", {kHidden}, true);
        append(prefix + ".post_attention_layernorm.weight", {kHidden}, true);
        append(prefix + ".self_attn.q_proj.weight", {kHidden, kHidden}, false);
        append(prefix + ".self_attn.k_proj.weight", {kv_hidden, kHidden}, false);
        append(prefix + ".self_attn.v_proj.weight", {kv_hidden, kHidden}, false);
        append(prefix + ".self_attn.o_proj.weight", {kHidden, kHidden}, false);
        append(prefix + ".mlp.gate_proj.weight", {kIntermediate, kHidden}, false);
        append(prefix + ".mlp.up_proj.weight", {kIntermediate, kHidden}, false);
        append(prefix + ".mlp.down_proj.weight", {kHidden, kIntermediate}, false);
    }
    return tensors;
}

void WriteCheckpoint(const std::filesystem::path& directory,
                     const CompatibilityCase& test_case,
                     const std::vector<CheckpointTensor>& tensors, bool reference) {
    std::filesystem::create_directories(directory);
    const std::string dtype = reference ? "float32" : (test_case.bfloat16 ? "bfloat16" : "float16");
    WriteTextFile(directory / "config.json",
                  "{\"model_type\":\"llama\",\"architectures\":[\"LlamaForCausalLM\"],"
                  "\"hidden_size\":8,\"intermediate_size\":12,\"num_hidden_layers\":2,"
                  "\"num_attention_heads\":4,\"num_key_value_heads\":" +
                          std::to_string(test_case.kv_heads) +
                          ",\"vocab_size\":32,\"max_position_embeddings\":16,\"head_dim\":2,"
                          "\"rms_norm_eps\":0.00001,\"hidden_act\":\"silu\",\"tie_word_embeddings\":" +
                          (test_case.tied ? "true" : "false") + ",\"torch_dtype\":\"" + dtype + "\"}");
    std::string header = "{";
    std::vector<std::byte> payload;
    for (const auto& tensor: tensors) {
        if (header.size() != 1) {
            header += ',';
        }
        const size_t begin = payload.size();
        const size_t element_bytes = reference ? sizeof(float) : sizeof(uint16_t);
        payload.resize(begin + tensor.bits.size() * element_bytes);
        for (size_t index = 0; index < tensor.bits.size(); ++index) {
            if (reference) {
                const float value = DecodeCheckpointValue(tensor.bits[index], test_case.bfloat16);
                const uint32_t bits = std::bit_cast<uint32_t>(value);
                for (size_t byte = 0; byte < sizeof(bits); ++byte) {
                    payload[begin + index * element_bytes + byte] = static_cast<std::byte>((bits >> (byte * 8U)) & 0xFFU);
                }
            } else {
                const uint16_t bits = tensor.bits[index];
                payload[begin + index * element_bytes] = static_cast<std::byte>(bits & 0xFFU);
                payload[begin + index * element_bytes + 1] = static_cast<std::byte>(bits >> 8U);
            }
        }
        header += '"' + tensor.name + "\":{\"dtype\":\"" +
                  (reference ? "F32" : (test_case.bfloat16 ? "BF16" : "F16")) + "\",\"shape\":[";
        for (size_t index = 0; index < tensor.shape.size(); ++index) {
            if (index != 0) {
                header += ',';
            }
            header += std::to_string(tensor.shape[index]);
        }
        header += "],\"data_offsets\":[" + std::to_string(begin) + ',' + std::to_string(payload.size()) + "]}";
    }
    header += '}';
    while ((sizeof(uint64_t) + header.size()) % 64 != 0) {
        header += ' ';
    }
    WriteSafetensorsFile(directory / "model.safetensors", header, payload);
}

CheckpointPair WriteCheckpointPair(const std::filesystem::path& root, const CompatibilityCase& test_case) {
    CheckpointPair pair{.source = root / "source", .reference = root / "reference"};
    const auto tensors = MakeCheckpointTensors(test_case);
    for (const auto& tensor: tensors) {
        pair.source_payload_bytes += tensor.bits.size() * sizeof(uint16_t);
        pair.fp32_payload_bytes += tensor.bits.size() * sizeof(float);
    }
    WriteCheckpoint(pair.source, test_case, tensors, false);
    WriteCheckpoint(pair.reference, test_case, tensors, true);
    return pair;
}

Runtime MakeCompatibilityRuntime(int64_t kv_heads,
                                 const std::shared_ptr<test::ResourceAllocationStats>& stats = {}) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = static_cast<size_t>(kv_heads),
                        .max_tokens = 8,
                        .head_dim = 2,
                        .kv_dtype = DataType::Float32(),
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    if (stats != nullptr) {
        builder.RegisterCustomAllocatorProvider(DeviceType::kCPU, std::make_unique<test::ResourceRecordingProvider>(stats));
    }
    return builder.Build();
}

CpuModelPrepareOptions CompatibilityOptions() {
    return {.cpu_preset = CpuInferencePreset::kO1Plain,
            .precision = InferencePrecisionPreset::kFp32Compatible};
}

StatusOr<numerical::CapturedTrajectory> ObserveCompatibilityTrajectory(Runtime& runtime,
                                                                       const ExecutableModel& model) {
    constexpr std::array<int64_t, 3> kPrompt{1, 7, 3};
    constexpr std::array<int64_t, 2> kDecodeInputs{17, 23};
    constexpr std::array<int64_t, 3> kPrefillRows{0, 1, 2};
    constexpr std::array<int64_t, 1> kDecodeRows{0};
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("Compatibility trajectory requires a KV manager");
    }
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(kPrompt.size(), kDecodeInputs.size()));
    numerical::SessionReservation reservation(*manager, view);
    const size_t workspace_bytes = std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes());
    const size_t alignment = std::max(prefill->workspace_alignment(), decode->workspace_alignment());
    Buffer workspace_storage;
    if (workspace_bytes != 0) {
        AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
        AM_ASSIGN_OR_RETURN(workspace_storage, allocator->TryAllocate({.nbytes = workspace_bytes, .alignment = alignment}));
    }
    CpuWorkspaceArena workspace(workspace_bytes == 0 ? nullptr : workspace_storage.mutable_data(), workspace_bytes);
    WorkspaceArena* const workspace_ptr = workspace_bytes == 0 ? nullptr : &workspace;
    numerical::CapturedTrajectory snapshots;
    test::TestBuffer prompt(DataType::Int(64), {static_cast<int64_t>(kPrompt.size())});
    test::TestBuffer positions(DataType::Int(64), {static_cast<int64_t>(kPrompt.size())});
    std::memcpy(prompt.mutable_data(), kPrompt.data(), sizeof(kPrompt));
    for (size_t index = 0; index < kPrompt.size(); ++index) {
        static_cast<int64_t*>(positions.mutable_data())[index] = static_cast<int64_t>(index);
    }
    {
        AM_ASSIGN_OR_RETURN(auto context, numerical::PrepareContext(runtime, *prefill, *prefill_weights, prompt, positions, workspace_ptr, view));
        AM_RETURN_IF_ERROR(Executor::Execute(*prefill, context));
        AM_ASSIGN_OR_RETURN(snapshots[0], numerical::CaptureStage(context, *prefill, kPrefillRows, kPrompt.size(), view));
    }
    test::TestBuffer token(DataType::Int(64), {1});
    test::TestBuffer position(DataType::Int(64), {1});
    static_cast<int64_t*>(token.mutable_data())[0] = kDecodeInputs[0];
    static_cast<int64_t*>(position.mutable_data())[0] = static_cast<int64_t>(kPrompt.size());
    {
        AM_ASSIGN_OR_RETURN(auto context, numerical::PrepareContext(runtime, *decode, *decode_weights, token, position, workspace_ptr, view));
        for (size_t index = 0; index < kDecodeInputs.size(); ++index) {
            static_cast<int64_t*>(token.mutable_data())[0] = kDecodeInputs[index];
            static_cast<int64_t*>(position.mutable_data())[0] = static_cast<int64_t>(kPrompt.size() + index);
            workspace.Reset();
            AM_RETURN_IF_ERROR(Executor::Execute(*decode, context));
            AM_ASSIGN_OR_RETURN(snapshots[index + 1], numerical::CaptureStage(context, *decode, kDecodeRows, 1, view));
        }
    }
    AM_RETURN_IF_ERROR(reservation.Release());
    return snapshots;
}

class Fp32CompatibleInference : public ::testing::TestWithParam<CompatibilityCase> {};

TEST_P(Fp32CompatibleInference, MatchesIndependentFp32TrajectoryAndGenerate) {
    const auto test_case = GetParam();
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    Runtime reference_runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    auto result = LoadAndPrepareCpuExecutableModel(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    auto reference_result = LoadAndPrepareCpuExecutableModel(reference_runtime, pair.reference, CpuModelPrepareOptions{});
    ASSERT_TRUE(reference_result.ok()) << reference_result.status().ToString();
    const auto& usage = result->resource_usage();
    ASSERT_TRUE(usage.weight_normalization.has_value());
    const auto& normalization = *usage.weight_normalization;
    EXPECT_EQ(normalization.converted_source_bytes, pair.source_payload_bytes);
    EXPECT_EQ(normalization.converted_weight_bytes, pair.fp32_payload_bytes);
    EXPECT_EQ(normalization.source_weights.mapped_bytes, std::filesystem::file_size(pair.source / "model.safetensors"));
    EXPECT_EQ(normalization.source_weights.owned_bytes, 0U);
    EXPECT_TRUE(normalization.source_weights.backing_complete);
    EXPECT_EQ(normalization.normalized_weights.mapped_bytes, 0U);
    EXPECT_EQ(normalization.normalized_weights.owned_bytes, pair.fp32_payload_bytes);
    EXPECT_TRUE(normalization.normalized_weights.backing_complete);
    EXPECT_EQ(normalization.peak_owned_weight_bytes, pair.fp32_payload_bytes);
    EXPECT_TRUE(normalization.peak_owned_weight_bytes_complete);
    EXPECT_EQ(usage.weights.owned_bytes, pair.fp32_payload_bytes);
    EXPECT_EQ(usage.weights.mapped_bytes, 0U);
    EXPECT_FALSE(reference_result->resource_usage().weight_normalization.has_value());

    // The convenience entry has already destroyed its loader and conversion
    // locals; only ExecutableModel's normal ownership chain supplies these bytes.
    const auto actual = ObserveCompatibilityTrajectory(runtime, *result);
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    const auto expected = ObserveCompatibilityTrajectory(reference_runtime, *reference_result);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    for (size_t stage = 0; stage < actual->size(); ++stage) {
        EXPECT_TRUE(numerical::CompareStage((*actual)[stage], (*expected)[stage], numerical::kStageNames[stage]));
        EXPECT_EQ((*actual)[stage].position, 3U + stage);
    }

    auto model = std::make_shared<ExecutableModel>(std::move(*result));
    auto reference_model = std::make_shared<ExecutableModel>(std::move(*reference_result));
    auto session = InferenceSession::Create(runtime, model);
    ASSERT_TRUE(session.ok()) << session.status().ToString();
    auto reference_session = InferenceSession::Create(reference_runtime, reference_model);
    ASSERT_TRUE(reference_session.ok()) << reference_session.status().ToString();
    constexpr std::array<uint32_t, 3> kPrompt{1, 7, 3};
    const auto generated = session->Generate(kPrompt, {.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    const auto reference_generated = reference_session->Generate(kPrompt, {.max_new_tokens = 3});
    ASSERT_TRUE(reference_generated.ok()) << reference_generated.status().ToString();
    ASSERT_EQ(generated->size(), 3U);
    EXPECT_EQ(*generated, *reference_generated);
    const auto repeated = session->Generate(kPrompt, {.max_new_tokens = 3});
    ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
    EXPECT_EQ(*repeated, *generated);
}

INSTANTIATE_TEST_SUITE_P(
        CompatibleCheckpoints, Fp32CompatibleInference,
        ::testing::Values(CompatibilityCase{false, false, 4, "Float16MhaUntied"},
                          CompatibilityCase{false, true, 2, "Float16GqaTied"},
                          CompatibilityCase{true, false, 4, "BFloat16MhaUntied"},
                          CompatibilityCase{true, true, 2, "BFloat16GqaTied"}),
        [](const ::testing::TestParamInfo<CompatibilityCase>& info) { return info.param.name; });

TEST(Fp32CompatibilityAdmission, StrictModeStillRejectsReducedCheckpointBeforePayloadAllocation) {
    for (const bool bfloat16: {false, true}) {
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), {.bfloat16 = bfloat16, .tied = true, .kv_heads = 2, .name = "Admission"});
        auto stats = std::make_shared<test::ResourceAllocationStats>();
        Runtime runtime = MakeCompatibilityRuntime(2, stats);
        stats->requests.clear();
        const auto result = LoadAndPrepareCpuExecutableModel(runtime, pair.source, CpuModelPrepareOptions{});
        EXPECT_EQ(result.status().code(), StatusCode::kUnimplemented);
        EXPECT_TRUE(stats->requests.empty());
    }
}

TEST(Fp32CompatibilityAdmission, ModelBudgetRejectsBeforeConversionAndAcceptsExactPreparedPayload) {
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), {.bfloat16 = false, .tied = true, .kv_heads = 2, .name = "Budget"});
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeCompatibilityRuntime(2, stats);
    auto options = CompatibilityOptions();
    options.resources.max_model_payload_bytes = pair.fp32_payload_bytes - 1;
    stats->requests.clear();
    const auto rejected = LoadAndPrepareCpuExecutableModel(runtime, pair.source, options);
    EXPECT_EQ(rejected.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(stats->requests.empty());
    options.resources.max_model_payload_bytes = 0;
    const auto baseline = LoadAndPrepareCpuExecutableModel(runtime, pair.source, options);
    ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
    options.resources.max_model_payload_bytes = baseline->resource_usage().managed_payload_bytes;
    const auto exact = LoadAndPrepareCpuExecutableModel(runtime, pair.source, options);
    ASSERT_TRUE(exact.ok()) << exact.status().ToString();
    EXPECT_EQ(exact->resource_usage().managed_payload_bytes, options.resources.max_model_payload_bytes);
}

TEST(Fp32CompatibilityAdmission, ConversionAllocationFailureAllowsFreshRetryAndGenerate) {
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), {.bfloat16 = true, .tied = false, .kv_heads = 4, .name = "Retry"});
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeCompatibilityRuntime(4, stats);
    stats->requests.clear();
    stats->fail_on_call = 1;
    const auto failed = LoadAndPrepareCpuExecutableModel(runtime, pair.source, CompatibilityOptions());
    EXPECT_EQ(failed.status().code(), StatusCode::kResourceExhausted);
    ASSERT_EQ(stats->requests.size(), 2U);
    stats->fail_on_call = std::numeric_limits<size_t>::max();
    auto retried = LoadAndPrepareCpuExecutableModel(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(retried.ok()) << retried.status().ToString();
    auto session = InferenceSession::Create(runtime, std::make_shared<ExecutableModel>(std::move(*retried)));
    ASSERT_TRUE(session.ok()) << session.status().ToString();
    constexpr std::array<uint32_t, 1> kPrompt{7};
    const auto generated = session->Generate(kPrompt, {.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    EXPECT_EQ(generated->size(), 3U);
}

TEST(Fp32CompatibilityAdmission, UnsupportedAndUnknownPresetsRejectBeforePayloadAllocation) {
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), {.bfloat16 = false, .tied = false, .kv_heads = 4, .name = "Preset"});
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeCompatibilityRuntime(4, stats);
    stats->requests.clear();
    auto options = CompatibilityOptions();
    options.cpu_preset = CpuInferencePreset::kO2Packed;
    EXPECT_FALSE(LoadAndPrepareCpuExecutableModel(runtime, pair.source, options).ok());
    EXPECT_TRUE(stats->requests.empty());
    options = CompatibilityOptions();
    options.precision = static_cast<InferencePrecisionPreset>(255);
    EXPECT_EQ(LoadAndPrepareCpuExecutableModel(runtime, pair.source, options).status().code(), StatusCode::kInvalidArgument);
    EXPECT_TRUE(stats->requests.empty());
    options = CompatibilityOptions();
    options.cpu_preset = static_cast<CpuInferencePreset>(255);
    EXPECT_EQ(LoadAndPrepareCpuExecutableModel(runtime, pair.source, options).status().code(), StatusCode::kInvalidArgument);
    EXPECT_TRUE(stats->requests.empty());
}

} // namespace
