#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "inference/test_reduced_weight_inference_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::test::reduced;
namespace numerical = aethermind::test::numerical;

constexpr float kAbsoluteTolerance = 1.0e-5F;
constexpr float kRelativeTolerance = 1.0e-4F;

ModelLoadOptions W16Options(bool bfloat16, CompilePreset preset = CompilePreset::kO1Plain) {
    return {.preset = preset,
            .weights = bfloat16 ? WeightPolicy::kKeepBf16WeightsFp32Activation
                                : WeightPolicy::kKeepFp16WeightsFp32Activation};
}

ModelLoadOptions Fp32Options() {
    return {.preset = CompilePreset::kO1Plain, .weights = WeightPolicy::kConvertToFp32};
}

const TensorView* FindImmutableWeight(const ExternalTensorBindings& bindings, ExecutionValueId value) {
    for (const auto& binding: bindings.read_only) {
        if (binding.value == value) {
            return &binding.tensor;
        }
    }
    return nullptr;
}

void ExpectW16Bindings(const ExecutableModel& model, const CompatibilityCase& test_case,
                       size_t expected_matrix_bytes) {
    const DataType weight_dtype = test_case.bfloat16 ? DataType::BFloat(16) : DataType::Float(16);
    for (const ExecPhase phase: {ExecPhase::kPrefill, ExecPhase::kDecode}) {
        const auto plan = model.plan(phase);
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        const auto bindings = model.immutable_weight_bindings(phase);
        ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
        size_t matrices = 0;
        size_t norms = 0;
        for (const auto& step: (*plan)->steps()) {
            if (step.kernel.op_type != OpType::kEmbedding && step.kernel.op_type != OpType::kLinear &&
                step.kernel.op_type != OpType::kRmsNorm) {
                continue;
            }
            ASSERT_EQ(step.inputs.size(), 2U);
            const TensorView* const weight = FindImmutableWeight(**bindings, step.inputs[1]);
            ASSERT_NE(weight, nullptr);
            if (step.kernel.op_type == OpType::kRmsNorm) {
                EXPECT_EQ(weight->dtype(), DataType::Float32());
                ++norms;
            } else {
                EXPECT_EQ(weight->dtype(), weight_dtype);
                EXPECT_EQ(weight->dtype().nbytes(), 2);
                ++matrices;
            }
            EXPECT_EQ((*plan)->values()[step.inputs[1].index].spec.dtype, weight->dtype());
        }
        EXPECT_GT(matrices, 0U);
        EXPECT_GT(norms, 0U);

        std::map<uintptr_t, size_t> matrix_views;
        for (const auto& binding: (*bindings)->read_only) {
            if (binding.tensor.dtype() == weight_dtype) {
                matrix_views.try_emplace(reinterpret_cast<uintptr_t>(binding.tensor.data()),
                                         static_cast<size_t>(binding.tensor.numel()) * 2);
            }
        }
        size_t actual_matrix_bytes = 0;
        for (const auto& [address, bytes]: matrix_views) {
            (void) address;
            actual_matrix_bytes += bytes;
        }
        EXPECT_EQ(actual_matrix_bytes, expected_matrix_bytes);
    }
}

void ExpectTensorClose(const numerical::TensorData<float>& actual,
                       const numerical::TensorData<float>& expected) {
    ASSERT_EQ(actual.shape, expected.shape);
    ASSERT_EQ(actual.values.size(), expected.values.size());
    for (size_t index = 0; index < actual.values.size(); ++index) {
        const float value = actual.values[index];
        const float reference = expected.values[index];
        EXPECT_TRUE(std::isfinite(value)) << index;
        EXPECT_TRUE(std::isfinite(reference)) << index;
        const double allowed = kAbsoluteTolerance + kRelativeTolerance * std::abs(static_cast<double>(reference));
        EXPECT_LE(std::abs(static_cast<double>(value) - reference), allowed) << index;
    }
}

std::vector<size_t> RankedIndices(const float* values, size_t count) {
    std::vector<size_t> ranked(count);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&](size_t lhs, size_t rhs) {
        return values[lhs] > values[rhs] || (values[lhs] == values[rhs] && lhs < rhs);
    });
    return ranked;
}

// The F32 accumulation path can choose a different token at a numerical tie.
// A top-1/top-k boundary wider than twice the measured error must be preserved.
bool ExpectStageCloseAndRanking(const numerical::StageSnapshot& actual,
                                const numerical::StageSnapshot& expected) {
    EXPECT_EQ(actual.position, expected.position);
    EXPECT_EQ(actual.logit_rows, expected.logit_rows);
    ExpectTensorClose(actual.logits, expected.logits);
    ExpectTensorClose(actual.keys, expected.keys);
    ExpectTensorClose(actual.values, expected.values);
    if (actual.logits.shape != expected.logits.shape || actual.logits.shape.size() != 2 ||
        actual.logits.values.size() != expected.logits.values.size() || actual.tokens.size() != expected.tokens.size()) {
        ADD_FAILURE() << "Cannot compare ranking of incompatible stage snapshots";
        return false;
    }
    const size_t vocabulary = static_cast<size_t>(actual.logits.shape[1]);
    if (vocabulary < 5) {
        ADD_FAILURE() << "The fixture must provide a top-4 boundary";
        return false;
    }
    bool last_row_separated = false;
    for (size_t row = 0; row < actual.logit_rows.size(); ++row) {
        const float* const values = actual.logits.values.data() + row * vocabulary;
        const float* const reference = expected.logits.values.data() + row * vocabulary;
        double maximum_error = 0.0;
        for (size_t token = 0; token < vocabulary; ++token) {
            maximum_error = std::max(maximum_error, std::abs(static_cast<double>(values[token]) - reference[token]));
        }
        const auto actual_order = RankedIndices(values, vocabulary);
        const auto reference_order = RankedIndices(reference, vocabulary);
        const double margin = static_cast<double>(reference[reference_order[0]]) - reference[reference_order[1]];
        const size_t token_row = static_cast<size_t>(actual.logit_rows[row]);
        if (token_row >= actual.tokens.size()) {
            ADD_FAILURE() << "Logit row exceeds the Argmax output";
            return false;
        }
        const bool top1_separated = margin > 2.0 * maximum_error;
        if (top1_separated) {
            EXPECT_EQ(actual_order[0], reference_order[0]);
            EXPECT_EQ(actual.tokens[token_row], expected.tokens[token_row]);
        } else {
            EXPECT_GE(static_cast<double>(reference[actual_order[0]]),
                      static_cast<double>(reference[reference_order[0]]) - 2.0 * maximum_error);
        }
        if (token_row + 1 == actual.tokens.size()) {
            last_row_separated = top1_separated;
        }
        const double top4_gap = static_cast<double>(reference[reference_order[3]]) - reference[reference_order[4]];
        if (top4_gap > 2.0 * maximum_error) {
            std::vector<size_t> actual_top4(actual_order.begin(), actual_order.begin() + 4);
            std::vector<size_t> reference_top4(reference_order.begin(), reference_order.begin() + 4);
            std::sort(actual_top4.begin(), actual_top4.end());
            std::sort(reference_top4.begin(), reference_top4.end());
            EXPECT_EQ(actual_top4, reference_top4);
        } else {
            for (size_t rank = 0; rank < 4; ++rank) {
                EXPECT_GE(static_cast<double>(reference[actual_order[rank]]),
                          static_cast<double>(reference[reference_order[3]]) - 2.0 * maximum_error);
            }
        }
    }
    return last_row_separated;
}

class W16Inference : public ::testing::TestWithParam<CompatibilityCase> {};

TEST_P(W16Inference, MatchesFp32CompatibilityTrajectoryAndGenerate) {
    const auto test_case = GetParam();
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    Runtime reference_runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    auto actual = ExecutableModel::Load(runtime, pair.source, W16Options(test_case.bfloat16));
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    auto reference = ExecutableModel::Load(reference_runtime, pair.source, Fp32Options());
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();

    const auto observed = ObserveCompatibilityTrajectory(runtime, *actual);
    ASSERT_TRUE(observed.ok()) << observed.status().ToString();
    const auto expected = ObserveCompatibilityTrajectory(reference_runtime, *reference);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    for (size_t stage = 0; stage < observed->size(); ++stage) {
        SCOPED_TRACE(numerical::kStageNames[stage]);
        (void) ExpectStageCloseAndRanking((*observed)[stage], (*expected)[stage]);
    }

    const auto greedy_reference = ObserveCompatibilityTrajectory(
            reference_runtime, *reference, {.greedy_decode_inputs = true});
    ASSERT_TRUE(greedy_reference.ok()) << greedy_reference.status().ToString();
    const std::array<int64_t, 2> decode_inputs{(*greedy_reference)[0].tokens.back(), (*greedy_reference)[1].tokens.back()};
    const auto teacher_forced = ObserveCompatibilityTrajectory(runtime, *actual, {.decode_inputs = decode_inputs});
    ASSERT_TRUE(teacher_forced.ok()) << teacher_forced.status().ToString();
    std::array<bool, 3> separated{};
    for (size_t stage = 0; stage < teacher_forced->size(); ++stage) {
        SCOPED_TRACE(numerical::kStageNames[stage]);
        separated[stage] = ExpectStageCloseAndRanking((*teacher_forced)[stage], (*greedy_reference)[stage]);
    }

    auto session = InferenceSession::Create(runtime, std::make_shared<ExecutableModel>(std::move(*actual)));
    ASSERT_TRUE(session.ok()) << session.status().ToString();
    auto reference_session = InferenceSession::Create(reference_runtime, std::make_shared<ExecutableModel>(std::move(*reference)));
    ASSERT_TRUE(reference_session.ok()) << reference_session.status().ToString();
    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    const auto reference_generated = reference_session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(reference_generated.ok()) << reference_generated.status().ToString();
    ASSERT_EQ(generated->size(), 3U);
    ASSERT_EQ(reference_generated->size(), 3U);
    bool stable_prefix = true;
    for (size_t stage = 0; stage < generated->size(); ++stage) {
        EXPECT_EQ((*reference_generated)[stage], (*greedy_reference)[stage].tokens.back());
        stable_prefix = stable_prefix && separated[stage];
        if (stable_prefix) {
            EXPECT_EQ((*generated)[stage], (*reference_generated)[stage]);
        }
    }
    const auto repeated = session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
    EXPECT_EQ(*repeated, *generated);
}

TEST_P(W16Inference, HandlesEosStoppingAndExactKvCapacity) {
    const auto test_case = GetParam();
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    auto model = ExecutableModel::Load(runtime, pair.source, W16Options(test_case.bfloat16));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    auto session = InferenceSession::Create(runtime, std::make_shared<ExecutableModel>(std::move(*model)));
    ASSERT_TRUE(session.ok()) << session.status().ToString();

    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    // The compatibility runtime reserves 8 KV tokens. Generation needs
    // prompt + max_new_tokens - 1 slots (the final token has no KV append),
    // so 6 new tokens fill capacity exactly while 7 must fail before execution.
    const auto full = session->Generate(prompt, {.max_new_tokens = 6});
    ASSERT_TRUE(full.ok()) << full.status().ToString();
    ASSERT_EQ(full->size(), 6U);

    const auto over = session->Generate(prompt, {.max_new_tokens = 7});
    ASSERT_FALSE(over.ok());
    EXPECT_EQ(over.status(), StatusCode::kOutOfRange);

    const auto retry = session->Generate(prompt, {.max_new_tokens = 6});
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_EQ(*retry, *full);

    const auto zero = session->Generate(prompt, {.max_new_tokens = 0});
    ASSERT_TRUE(zero.ok()) << zero.status().ToString();
    EXPECT_TRUE(zero->empty());

    const auto first_eos = session->Generate(
            prompt, {.max_new_tokens = 6, .eos_token_id = (*full)[0]});
    ASSERT_TRUE(first_eos.ok()) << first_eos.status().ToString();
    EXPECT_EQ(*first_eos, std::vector<uint32_t>{(*full)[0]});

    // Stopping happens at the first occurrence of the EOS token, so repeated
    // earlier tokens do not change the expected prefix length.
    const uint32_t mid_token = (*full)[2];
    const auto mid_it = std::find(full->begin(), full->end(), mid_token);
    ASSERT_NE(mid_it, full->end());
    const auto mid_eos = session->Generate(
            prompt, {.max_new_tokens = 6, .eos_token_id = mid_token});
    ASSERT_TRUE(mid_eos.ok()) << mid_eos.status().ToString();
    EXPECT_EQ(*mid_eos,
              std::vector<uint32_t>(full->begin(), mid_it + 1));
}

TEST_P(W16Inference, RetainsTwoByteMatrixBindingsAndConvertsOnlyNormPayload) {
    const auto test_case = GetParam();
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    auto actual = ExecutableModel::Load(runtime, pair.source, W16Options(test_case.bfloat16));
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    ExpectW16Bindings(*actual, test_case, pair.matrix_payload_bytes);
    const auto& usage = actual->resource_usage();
    ASSERT_TRUE(usage.weight_fp32_conversion.has_value());
    const auto& conversion = *usage.weight_fp32_conversion;
    EXPECT_EQ(conversion.converted_source_bytes, pair.reduced_norm_source_bytes);
    EXPECT_EQ(conversion.converted_weight_bytes, pair.converted_norm_bytes);
    EXPECT_EQ(conversion.converted_weights.owned_bytes, pair.converted_norm_bytes);
    EXPECT_EQ(usage.weights.owned_bytes, pair.converted_norm_bytes);
    EXPECT_EQ(usage.weights.mapped_bytes, std::filesystem::file_size(pair.source / "model.safetensors"));
    EXPECT_TRUE(usage.weights.backing_usage_complete);
    EXPECT_EQ(usage.packed_weight_bytes, 0U);
    EXPECT_EQ(conversion.peak_owned_weight_bytes, pair.converted_norm_bytes);
}

TEST_P(W16Inference, O2PackedMatchesO1NumericsAndRetainsReducedArtifacts) {
    const auto test_case = GetParam();
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    Runtime plain_runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    auto packed = ExecutableModel::Load(
            runtime, pair.source, W16Options(test_case.bfloat16, CompilePreset::kO2Packed));
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    auto plain = ExecutableModel::Load(plain_runtime, pair.source, W16Options(test_case.bfloat16));
    ASSERT_TRUE(plain.ok()) << plain.status().ToString();

    const DataType weight_dtype = test_case.bfloat16 ? DataType::BFloat(16) : DataType::Float(16);
    size_t fused_consumers = 0;
    size_t packed_artifacts = 0;
    for (const ExecPhase phase: {ExecPhase::kPrefill, ExecPhase::kDecode}) {
        const auto plan = packed->plan(phase);
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        for (const auto& step: (*plan)->steps()) {
            if (step.kernel.op_type == OpType::kQkvLinear ||
                step.kernel.op_type == OpType::kGateUpLinear) {
                ++fused_consumers;
            }
            if (step.packed_weights == nullptr) {
                continue;
            }
            ++packed_artifacts;
            // Norm weights are widened to F32 before compilation; only matrix
            // consumers retain the reduced packed payload.
            const bool norm_consumer = step.kernel.op_type == OpType::kRmsNorm ||
                                       step.kernel.op_type == OpType::kAddRmsNorm;
            const DataType expected_dtype = norm_consumer ? DataType::Float32() : weight_dtype;
            const size_t element_bytes = norm_consumer ? 4U : 2U;
            EXPECT_EQ(step.packed_weights->logical_dtype(), expected_dtype);
            int64_t elements = 1;
            for (const int64_t dim: step.packed_weights->logical_shape()) {
                elements *= dim;
            }
            EXPECT_EQ(step.packed_weights->storage().nbytes(),
                      static_cast<size_t>(elements) * element_bytes);
            EXPECT_EQ(step.packed_weights->recipe().layout, PackingLayout::kCpuIdentity);
        }
    }
    EXPECT_GT(fused_consumers, 0U);
    EXPECT_GT(packed_artifacts, 0U);
    EXPECT_GT(packed->resource_usage().packed_weight_bytes, 0U);

    const auto observed = ObserveCompatibilityTrajectory(runtime, *packed);
    ASSERT_TRUE(observed.ok()) << observed.status().ToString();
    const auto expected = ObserveCompatibilityTrajectory(plain_runtime, *plain);
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();
    for (size_t stage = 0; stage < observed->size(); ++stage) {
        SCOPED_TRACE(numerical::kStageNames[stage]);
        (void) ExpectStageCloseAndRanking((*observed)[stage], (*expected)[stage]);
    }

    auto session = InferenceSession::Create(
            runtime, std::make_shared<ExecutableModel>(std::move(*packed)));
    ASSERT_TRUE(session.ok()) << session.status().ToString();
    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    EXPECT_EQ(generated->size(), 3U);
    const auto repeated = session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
    EXPECT_EQ(*repeated, *generated);
}

TEST_P(W16Inference, DecodeExecutionHasNoMallocFamilyCallsAfterBinding) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "The malloc-family gate requires the glibc interposer without sanitizer interception";
    }
    const auto test_case = GetParam();
    for (const auto preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), test_case);
        Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
        auto actual = ExecutableModel::Load(runtime, pair.source, W16Options(test_case.bfloat16, preset));
        ASSERT_TRUE(actual.ok()) << actual.status().ToString();
        std::array<test::MallocCallCounts, 2> counts{};
        const auto trajectory = ObserveCompatibilityTrajectory(runtime, *actual, {.decode_allocations = &counts});
        ASSERT_TRUE(trajectory.ok()) << trajectory.status().ToString();
        for (const auto& count: counts) {
            EXPECT_EQ(count.malloc_calls, 0U);
            EXPECT_EQ(count.calloc_calls, 0U);
            EXPECT_EQ(count.realloc_calls, 0U);
            EXPECT_EQ(count.free_calls, 0U);
            EXPECT_EQ(count.aligned_alloc_calls, 0U);
            EXPECT_EQ(count.posix_memalign_calls, 0U);
            EXPECT_EQ(count.memalign_calls, 0U);
        }
    }
}

TEST_P(W16Inference, SessionRetainsWeightsAfterSourceScopeAndModelHandleRelease) {
    const auto test_case = GetParam();
    Runtime runtime = MakeCompatibilityRuntime(test_case.kv_heads);
    std::optional<InferenceSession> session;
    std::weak_ptr<const ExecutableModel> weak_model;
    {
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), test_case);
        auto actual = ExecutableModel::Load(runtime, pair.source, W16Options(test_case.bfloat16));
        ASSERT_TRUE(actual.ok()) << actual.status().ToString();
        auto model = std::make_shared<ExecutableModel>(std::move(*actual));
        weak_model = model;
        auto created = InferenceSession::Create(runtime, model);
        ASSERT_TRUE(created.ok()) << created.status().ToString();
        session.emplace(std::move(*created));
    }
    EXPECT_FALSE(weak_model.expired());
    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    EXPECT_EQ(generated->size(), 3U);
    session.reset();
    EXPECT_TRUE(weak_model.expired());
}

INSTANTIATE_TEST_SUITE_P(
        ReducedCheckpoints, W16Inference,
        ::testing::Values(
                CompatibilityCase{false, false, 4, "Float16MhaUntiedF32Norm", "float32", DataType::Float32()},
                CompatibilityCase{false, true, 2, "Float16GqaTiedBfloatNorm", "auto", DataType::BFloat(16)},
                CompatibilityCase{false, false, 2, "Float16GqaUntiedHalfNorm", "bfloat16", DataType::Float(16)},
                CompatibilityCase{true, false, 4, "Bfloat16MhaUntiedHalfNorm", "float16", DataType::Float(16)},
                CompatibilityCase{true, false, 2, "Bfloat16GqaUntiedF32Norm", "auto", DataType::Float32()},
                CompatibilityCase{true, true, 2, "Bfloat16GqaTiedDefaultNorm", ""}),
        [](const ::testing::TestParamInfo<CompatibilityCase>& info) { return info.param.name; });

struct ConversionCalls {
    static inline size_t fp16_calls = 0;
    static inline size_t bf16_calls = 0;
    static inline size_t elements = 0;
    static void Fp16(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++fp16_calls;
        elements += count;
        ConvertFp16ToFp32Scalar(source, destination, count);
    }
    static void Bf16(const std::byte* source, std::byte* destination, size_t count) noexcept {
        ++bf16_calls;
        elements += count;
        ConvertBf16ToFp32Scalar(source, destination, count);
    }
    static void Reset() {
        fp16_calls = 0;
        bf16_calls = 0;
        elements = 0;
    }
};

class ConversionBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType op, const KernelSelector& selector, const OpParams& params) const override {
        return cpu_.PrepareKernel(op, selector, params);
    }
    Fp16ConversionFuncs GetHostFp16Conversions() const noexcept override {
        return {ConversionCalls::Fp16, ConvertFp32ToFp16Scalar};
    }
    Bf16ConversionFuncs GetHostBf16Conversions() const noexcept override {
        return {ConversionCalls::Bf16, ConvertFp32ToBf16Scalar};
    }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return cpu_.TryGetKernelRegistryForDebug();
    }

private:
    CpuBackend cpu_;
};

class ConversionFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<std::unique_ptr<Backend>> Create() const override {
        return std::unique_ptr<Backend>(std::make_unique<ConversionBackend>());
    }
};

class W16Preparation : public ::testing::Test {
protected:
    void SetUp() override { ConversionCalls::Reset(); }
    void TearDown() override { ConversionCalls::Reset(); }
};

TEST_F(W16Preparation, BackendCallbacksWidenOnlyNorms) {
    for (const bool bfloat16: {false, true}) {
        ConversionCalls::Reset();
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), {bfloat16, true, 2, "Callback"});
        Runtime runtime = MakeCompatibilityRuntime(2, {}, std::make_unique<ConversionFactory>());
        auto model = ExecutableModel::Load(runtime, pair.source, W16Options(bfloat16));
        ASSERT_TRUE(model.ok()) << model.status().ToString();
        EXPECT_EQ(ConversionCalls::fp16_calls, bfloat16 ? 0U : 5U);
        EXPECT_EQ(ConversionCalls::bf16_calls, bfloat16 ? 5U : 0U);
        EXPECT_EQ(ConversionCalls::elements * sizeof(uint16_t), pair.reduced_norm_source_bytes);
    }
}

TEST_F(W16Preparation, BudgetRejectsBeforeNormCallbackAndAcceptsExactManagedPayload) {
    for (const bool bfloat16: {false, true}) {
        ConversionCalls::Reset();
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), {bfloat16, true, 2, "Budget"});
        auto stats = std::make_shared<test::ResourceAllocationStats>();
        Runtime runtime = MakeCompatibilityRuntime(2, stats, std::make_unique<ConversionFactory>());
        stats->requests.clear();
        auto options = W16Options(bfloat16);
        options.limits.max_model_payload_bytes = pair.converted_norm_bytes - 1;
        const auto rejected = ExecutableModel::Load(runtime, pair.source, options);
        ASSERT_FALSE(rejected.ok());
        EXPECT_EQ(rejected.status(), StatusCode::kResourceExhausted);
        EXPECT_TRUE(stats->requests.empty());
        EXPECT_EQ(ConversionCalls::fp16_calls + ConversionCalls::bf16_calls, 0U);
        options.limits.max_model_payload_bytes = 0;
        const auto baseline = ExecutableModel::Load(runtime, pair.source, options);
        ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
        options.limits.max_model_payload_bytes = baseline->resource_usage().managed_payload_bytes;
        const auto exact = ExecutableModel::Load(runtime, pair.source, options);
        ASSERT_TRUE(exact.ok()) << exact.status().ToString();
        EXPECT_EQ(exact->resource_usage().managed_payload_bytes, options.limits.max_model_payload_bytes);
    }
}

TEST_F(W16Preparation, NormAllocationFailureAllowsLoadRetryAndGenerate) {
    for (const bool bfloat16: {false, true}) {
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), {bfloat16, false, 4, "Retry"});
        auto stats = std::make_shared<test::ResourceAllocationStats>();
        Runtime runtime = MakeCompatibilityRuntime(4, stats);
        stats->requests.clear();
        stats->fail_on_call = 1;
        const auto failed = ExecutableModel::Load(runtime, pair.source, W16Options(bfloat16));
        ASSERT_FALSE(failed.ok());
        EXPECT_EQ(failed.status(), StatusCode::kResourceExhausted);
        EXPECT_EQ(stats->requests.size(), 2U);
        stats->fail_on_call = std::numeric_limits<size_t>::max();
        auto retried = ExecutableModel::Load(runtime, pair.source, W16Options(bfloat16));
        ASSERT_TRUE(retried.ok()) << retried.status().ToString();
        auto session = InferenceSession::Create(runtime, std::make_shared<ExecutableModel>(std::move(*retried)));
        ASSERT_TRUE(session.ok()) << session.status().ToString();
        constexpr std::array<uint32_t, 1> prompt{7};
        const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
        ASSERT_TRUE(generated.ok()) << generated.status().ToString();
        EXPECT_EQ(generated->size(), 3U);
    }
}

TEST_F(W16Preparation, KnownHintsPreserveActualDtypeAdmission) {
    for (const bool bfloat16: {false, true}) {
        for (const char* hint: {"float32", "float16", "bfloat16", "auto", ""}) {
            SCOPED_TRACE(hint);
            TempDirectory directory;
            const auto pair = WriteCheckpointPair(directory.path(), {bfloat16, true, 2, "Hint", hint});
            auto stats = std::make_shared<test::ResourceAllocationStats>();
            Runtime runtime = MakeCompatibilityRuntime(2, stats);
            auto accepted = ExecutableModel::Load(runtime, pair.source, W16Options(bfloat16));
            ASSERT_TRUE(accepted.ok()) << accepted.status().ToString();
            stats->requests.clear();
            const auto wrong_matrix = ExecutableModel::Load(runtime, pair.source, W16Options(!bfloat16));
            ASSERT_FALSE(wrong_matrix.ok());
            EXPECT_EQ(wrong_matrix.status(), StatusCode::kUnimplemented);
            EXPECT_TRUE(stats->requests.empty());
        }
    }
}

} // namespace
