#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/inference/model_execution_policy.h"
#include "inference/test_reduced_weight_inference_helpers.h"
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
using namespace aethermind::test::reduced;

ModelLoadOptions CompatibilityOptions() {
    return {.preset = CompilePreset::kO1Plain,
            .weights = WeightPolicy::kConvertToFp32};
}

struct BatchConversionCalls {
    static inline size_t calls = 0;
    static inline size_t elements = 0;
    static inline size_t bfloat_calls = 0;
    static inline size_t bfloat_elements = 0;
    static void Widen(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++calls;
        elements += count;
        ConvertFp16ToFp32Scalar(source, destination, count);
    }
    static void WidenBfloat(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++bfloat_calls;
        bfloat_elements += count;
        ConvertBf16ToFp32Scalar(source, destination, count);
    }
};

class ConversionRecordingBackend final : public Backend {
public:
    explicit ConversionRecordingBackend(bool record) : record_(record) {}
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType op, const KernelSelector& selector, const OpParams& params) const override {
        return cpu_.PrepareKernel(op, selector, params);
    }
    Fp16ConversionFuncs GetHostFp16Conversions() const noexcept override {
        return record_ ? Fp16ConversionFuncs{BatchConversionCalls::Widen, ConvertFp32ToFp16Scalar}
                       : Backend::GetHostFp16Conversions();
    }
    Bf16ConversionFuncs GetHostBf16Conversions() const noexcept override {
        return record_ ? Bf16ConversionFuncs{BatchConversionCalls::WidenBfloat, ConvertFp32ToBf16Scalar}
                       : Backend::GetHostBf16Conversions();
    }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return cpu_.TryGetKernelRegistryForDebug();
    }

private:
    CpuBackend cpu_;
    bool record_;
};

class ConversionRecordingFactory final : public BackendFactory {
public:
    explicit ConversionRecordingFactory(bool record) : record_(record) {}
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<std::unique_ptr<Backend>> Create() const override {
        return std::unique_ptr<Backend>(std::make_unique<ConversionRecordingBackend>(record_));
    }

private:
    bool record_;
};

class Fp32CompatibilityBatch : public ::testing::Test {
protected:
    void SetUp() override { Reset(); }
    void TearDown() override { Reset(); }

private:
    static void Reset() {
        BatchConversionCalls::calls = 0;
        BatchConversionCalls::elements = 0;
        BatchConversionCalls::bfloat_calls = 0;
        BatchConversionCalls::bfloat_elements = 0;
    }
};

TEST_F(Fp32CompatibilityBatch, LoadUsesBackendCallbackOncePerDistinctFp16Tensor) {
    TempDirectory directory;
    const CompatibilityCase test_case{false, false, 2, "fp16"};
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(2, {}, std::make_unique<ConversionRecordingFactory>(true));
    auto result = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(BatchConversionCalls::calls, MakeCheckpointTensors(test_case).size());
    EXPECT_EQ(BatchConversionCalls::elements * sizeof(uint16_t), pair.source_payload_bytes);
    EXPECT_EQ(BatchConversionCalls::bfloat_calls, 0U);
}

TEST_F(Fp32CompatibilityBatch, LoadUsesBackendCallbackOncePerDistinctBfloat16Tensor) {
    TempDirectory directory;
    const CompatibilityCase test_case{true, true, 2, "bf16"};
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(2, {}, std::make_unique<ConversionRecordingFactory>(true));
    auto result = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(BatchConversionCalls::bfloat_calls, MakeCheckpointTensors(test_case).size());
    EXPECT_EQ(BatchConversionCalls::bfloat_elements * sizeof(uint16_t), pair.source_payload_bytes);
    EXPECT_EQ(BatchConversionCalls::calls, 0U);
}

TEST_F(Fp32CompatibilityBatch, CustomBackendKeepsDefaultScalarConversion) {
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), {false, true, 2, "fp16"});
    Runtime runtime = MakeCompatibilityRuntime(2, {}, std::make_unique<ConversionRecordingFactory>(false));
    auto result = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(BatchConversionCalls::calls, 0U);
    ASSERT_TRUE(runtime.GetBackend(DeviceType::kCPU).ok());
    EXPECT_EQ((*runtime.GetBackend(DeviceType::kCPU))->GetHostFp16Conversions().to_fp32, ConvertFp16ToFp32Scalar);
}

TEST_F(Fp32CompatibilityBatch, CustomBackendKeepsDefaultBfloatScalarConversions) {
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), {true, true, 2, "bf16"});
    Runtime runtime = MakeCompatibilityRuntime(2, {}, std::make_unique<ConversionRecordingFactory>(false));
    auto result = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(BatchConversionCalls::bfloat_calls, 0U);
    ASSERT_TRUE(runtime.GetBackend(DeviceType::kCPU).ok());
    const auto conversions = (*runtime.GetBackend(DeviceType::kCPU))->GetHostBf16Conversions();
    EXPECT_EQ(conversions.to_fp32, ConvertBf16ToFp32Scalar);
    EXPECT_EQ(conversions.from_fp32, ConvertFp32ToBf16Scalar);
}

class Fp32CompatibleInference : public ::testing::TestWithParam<CompatibilityCase> {};

TEST_P(Fp32CompatibleInference, MatchesIndependentFp32TrajectoryAndGenerate) {
    const auto test_case = GetParam();
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    Runtime reference_runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    auto result = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    auto reference_result = ExecutableModel::Load(reference_runtime, pair.reference, ModelLoadOptions{});
    ASSERT_TRUE(reference_result.ok()) << reference_result.status().ToString();
    const auto& usage = result->resource_usage();
    ASSERT_TRUE(usage.weight_fp32_conversion.has_value());
    const auto& conversion = *usage.weight_fp32_conversion;
    EXPECT_EQ(conversion.converted_source_bytes, pair.source_payload_bytes);
    EXPECT_EQ(conversion.converted_weight_bytes, pair.fp32_payload_bytes);
    EXPECT_EQ(conversion.source_weights.mapped_bytes, std::filesystem::file_size(pair.source / "model.safetensors"));
    EXPECT_EQ(conversion.source_weights.owned_bytes, 0U);
    EXPECT_TRUE(conversion.source_weights.backing_usage_complete);
    EXPECT_EQ(conversion.converted_weights.mapped_bytes, 0U);
    EXPECT_EQ(conversion.converted_weights.owned_bytes, pair.fp32_payload_bytes);
    EXPECT_TRUE(conversion.converted_weights.backing_usage_complete);
    EXPECT_EQ(conversion.peak_owned_weight_bytes, pair.fp32_payload_bytes);
    EXPECT_TRUE(conversion.peak_owned_weight_bytes_complete);
    EXPECT_EQ(usage.weights.owned_bytes, pair.fp32_payload_bytes);
    EXPECT_EQ(usage.weights.mapped_bytes, 0U);
    EXPECT_FALSE(reference_result->resource_usage().weight_fp32_conversion.has_value());

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
        const auto result = ExecutableModel::Load(runtime, pair.source, ModelLoadOptions{});
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
    options.limits.max_model_payload_bytes = pair.fp32_payload_bytes - 1;
    stats->requests.clear();
    const auto rejected = ExecutableModel::Load(runtime, pair.source, options);
    EXPECT_EQ(rejected.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(stats->requests.empty());
    options.limits.max_model_payload_bytes = 0;
    const auto baseline = ExecutableModel::Load(runtime, pair.source, options);
    ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
    options.limits.max_model_payload_bytes = baseline->resource_usage().managed_payload_bytes;
    const auto exact = ExecutableModel::Load(runtime, pair.source, options);
    ASSERT_TRUE(exact.ok()) << exact.status().ToString();
    EXPECT_EQ(exact->resource_usage().managed_payload_bytes, options.limits.max_model_payload_bytes);
}

TEST(Fp32CompatibilityAdmission, ConversionAllocationFailureAllowsFreshRetryAndGenerate) {
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), {.bfloat16 = true, .tied = false, .kv_heads = 4, .name = "Retry"});
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeCompatibilityRuntime(4, stats);
    stats->requests.clear();
    stats->fail_on_call = 1;
    const auto failed = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
    EXPECT_EQ(failed.status().code(), StatusCode::kResourceExhausted);
    ASSERT_EQ(stats->requests.size(), 2U);
    stats->fail_on_call = std::numeric_limits<size_t>::max();
    auto retried = ExecutableModel::Load(runtime, pair.source, CompatibilityOptions());
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
    options.preset = CompilePreset::kO2Packed;
    EXPECT_FALSE(ExecutableModel::Load(runtime, pair.source, options).ok());
    EXPECT_TRUE(stats->requests.empty());
    for (const auto target: {DeviceType::kCUDA, DeviceType::kCANN}) {
        options = CompatibilityOptions();
        options.target = target;
        EXPECT_EQ(ExecutableModel::Load(runtime, pair.source, options).status().code(),
                  StatusCode::kUnimplemented);
        EXPECT_TRUE(stats->requests.empty());
    }
    options = CompatibilityOptions();
    options.target = static_cast<DeviceType>(255);
    EXPECT_EQ(ExecutableModel::Load(runtime, pair.source, options).status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_TRUE(stats->requests.empty());
    options = CompatibilityOptions();
    options.weights = static_cast<WeightPolicy>(255);
    EXPECT_EQ(ExecutableModel::Load(runtime, pair.source, options).status().code(), StatusCode::kInvalidArgument);
    EXPECT_TRUE(stats->requests.empty());
    options = CompatibilityOptions();
    options.preset = static_cast<CompilePreset>(255);
    EXPECT_EQ(ExecutableModel::Load(runtime, pair.source, options).status().code(), StatusCode::kInvalidArgument);
    EXPECT_TRUE(stats->requests.empty());
}

} // namespace
