#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/test_reduced_weight_inference_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::test::reduced;
namespace numerical = aethermind::test::numerical;

// Reduced-fixture quality checks compare the production A16 path with the same
// W16 checkpoint executed as W16/A32/KV32. They are NOT the §4.5 frozen
// budgets: those were calibrated for the tiny-random-Llama numerical fixtures.
// This reduced checkpoint (hidden 8, head_dim 2) has a larger V-projection
// scale, so its tolerance is measured for this fixture. Recorded max normalized
// errors with the budgets below: standard prefill logits 0.30 / keys 0.22 /
// values 0.38 (decode 0.41 / 0.22 / 0.38), near-capacity 0.57 / 0.41 / 0.89,
// GQA+tied 0.17 / 0.27 / 0.05. The §4.5 budgets remain the acceptance authority
// for the calibrated numerical fixtures.
struct QualityBudget {
    float logits_atol;
    float logits_rtol;
    float kv_atol;
    float kv_rtol;
};

QualityBudget BudgetFor(DataType act_dtype) {
    if (act_dtype == DataType::Float(16)) {
        return {2e-3F, 5e-3F, 1e-3F, 1e-2F};
    }
    return {1e-2F, 3e-2F, 5e-3F, 4e-2F};
}

bool ScalarWithin(const std::vector<float>& actual, const std::vector<float>& ref,
                  float atol, float rtol, float* max_normalized = nullptr) {
    if (actual.size() != ref.size()) {
        return false;
    }
    float worst = 0.0F;
    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(ref[i])) {
            return false;
        }
        const float bound = atol + rtol * std::abs(ref[i]);
        const float normalized = std::abs(actual[i] - ref[i]) / bound;
        worst = std::max(worst, normalized);
        if (normalized > 1.0F) {
            ok = false;
        }
    }
    if (max_normalized != nullptr) {
        *max_normalized = worst;
    }
    return ok;
}

// Fixed ranking rule (§4.5): classification derives from the reference logits
// and budget only; top-1 / top-4 sets must match when their margin exceeds
// 2*epsilon, otherwise the row is a near-tie (scalar gates still apply).
bool LogitsRankingWithin(const numerical::TensorData<float>& actual,
                         const numerical::TensorData<float>& ref, float atol, float rtol) {
    if (actual.shape != ref.shape || actual.shape.size() != 2 ||
        actual.shape[0] <= 0 || actual.shape[1] <= 0 ||
        actual.values.size() != ref.values.size()) {
        return false;
    }
    const size_t rows = static_cast<size_t>(actual.shape[0]);
    const size_t vocab = static_cast<size_t>(actual.shape[1]);
    const auto finite = [](float value) { return std::isfinite(value); };
    if (actual.values.size() % vocab != 0 || actual.values.size() / vocab != rows ||
        !std::all_of(actual.values.begin(), actual.values.end(), finite) ||
        !std::all_of(ref.values.begin(), ref.values.end(), finite)) {
        return false;
    }
    if (vocab < 2) {
        return true;
    }
    for (size_t row = 0; row < rows; ++row) {
        const float* a = actual.values.data() + row * vocab;
        const float* r = ref.values.data() + row * vocab;
        float epsilon = 0.0F;
        for (size_t i = 0; i < vocab; ++i) {
            epsilon = std::max(epsilon, atol + rtol * std::abs(r[i]));
        }
        std::vector<size_t> ref_order(vocab);
        std::vector<size_t> act_order(vocab);
        for (size_t i = 0; i < vocab; ++i) {
            ref_order[i] = i;
            act_order[i] = i;
        }
        std::stable_sort(ref_order.begin(), ref_order.end(),
                         [&](size_t x, size_t y) { return r[x] > r[y]; });
        std::stable_sort(act_order.begin(), act_order.end(),
                         [&](size_t x, size_t y) { return a[x] > a[y]; });
        const float top1_margin = r[ref_order[0]] - r[ref_order[1]];
        if (top1_margin > 2 * epsilon && act_order[0] != ref_order[0]) {
            return false;
        }
        if (vocab >= 5) {
            const float top4_margin = r[ref_order[3]] - r[ref_order[4]];
            if (top4_margin > 2 * epsilon) {
                std::array<size_t, 4> ref_top4{};
                std::array<size_t, 4> act_top4{};
                for (size_t i = 0; i < 4; ++i) {
                    ref_top4[i] = ref_order[i];
                    act_top4[i] = act_order[i];
                }
                std::sort(ref_top4.begin(), ref_top4.end());
                std::sort(act_top4.begin(), act_top4.end());
                if (ref_top4 != act_top4) {
                    return false;
                }
            }
        }
    }
    return true;
}

void ExpectStageQuality(const numerical::StageSnapshot& actual,
                        const numerical::StageSnapshot& ref,
                        DataType act_dtype,
                        const std::string& label) {
    const QualityBudget budget = BudgetFor(act_dtype);
    EXPECT_EQ(actual.position, ref.position);
    EXPECT_EQ(actual.logit_rows, ref.logit_rows);
    float logits_norm = 0.0F;
    float keys_norm = 0.0F;
    float values_norm = 0.0F;
    EXPECT_TRUE(ScalarWithin(actual.logits.values, ref.logits.values,
                             budget.logits_atol, budget.logits_rtol, &logits_norm));
    EXPECT_TRUE(LogitsRankingWithin(actual.logits, ref.logits,
                                    budget.logits_atol, budget.logits_rtol));
    EXPECT_EQ(actual.keys.shape, ref.keys.shape);
    EXPECT_EQ(actual.values.shape, ref.values.shape);
    EXPECT_TRUE(ScalarWithin(actual.keys.values, ref.keys.values,
                             budget.kv_atol, budget.kv_rtol, &keys_norm));
    EXPECT_TRUE(ScalarWithin(actual.values.values, ref.values.values,
                             budget.kv_atol, budget.kv_rtol, &values_norm));
    const auto record = [&](std::string_view metric, float value) {
        ::testing::Test::RecordProperty(label + "_" + std::string(metric),
                                        static_cast<double>(value));
    };
    record("logits_max_normalized", logits_norm);
    record("keys_max_normalized", keys_norm);
    record("values_max_normalized", values_norm);
}

Runtime MakeA16Runtime(int64_t kv_heads, DataType act_dtype, size_t max_tokens = 8) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = static_cast<size_t>(kv_heads),
                        .max_tokens = max_tokens,
                        .head_dim = 2,
                        .kv_dtype = act_dtype,
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

ModelLoadOptions A16Options(bool bfloat16, CompilePreset preset) {
    const DataType activation = bfloat16 ? DataType::BFloat(16) : DataType::Float(16);
    return {.preset = preset,
            .weights = bfloat16 ? WeightPolicy::kKeepBf16WeightsFp32Activation
                                : WeightPolicy::kKeepFp16WeightsFp32Activation,
            .kv_dtype = activation,
            .activation_dtype = activation};
}

ModelLoadOptions F32Options(bool bfloat16, CompilePreset preset) {
    return {.preset = preset,
            .weights = bfloat16 ? WeightPolicy::kKeepBf16WeightsFp32Activation
                                : WeightPolicy::kKeepFp16WeightsFp32Activation,
            .kv_dtype = DataType::Float32()};
}

TEST(A16ModelAcceptance, SameDtypeCombinationsMatchKv32WithinCalibratedBudgetsO1ThenO2) {
    // P4-A executes the full-same-type BF16 preset (A=W=KV=residual). The FP16
    // preset keeps an F32 residual stream and needs the P4-B Cast path, so it is
    // rejected at Load admission rather than exercised here.
    size_t verified_combinations = 0;
    for (const CompilePreset preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        {
            constexpr bool bfloat16 = true;
            const DataType act_dtype = DataType::BFloat(16);
            SCOPED_TRACE(std::string(preset == CompilePreset::kO1Plain ? "O1/" : "O2/") +
                         (bfloat16 ? "Bf16" : "F16"));
            const CompatibilityCase test_case{bfloat16, false, 4, "A16Acceptance",
                                              "float32", DataType::Float32()};
            TempDirectory directory;
            const auto pair = WriteCheckpointPair(directory.path(), test_case);

            Runtime runtime16 = MakeA16Runtime(test_case.kv_heads, act_dtype);
            Runtime runtime32 = MakeA16Runtime(test_case.kv_heads, DataType::Float32());
            auto model16 = ExecutableModel::Load(runtime16, pair.source, A16Options(bfloat16, preset));
            ASSERT_TRUE(model16.ok()) << model16.status().ToString();
            auto model32 = ExecutableModel::Load(runtime32, pair.source, F32Options(bfloat16, preset));
            ASSERT_TRUE(model32.ok()) << model32.status().ToString();

            const auto traj16 = ObserveCompatibilityTrajectory(runtime16, *model16);
            ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
            const auto traj32 = ObserveCompatibilityTrajectory(runtime32, *model32);
            ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();
            size_t differing = 0;
            for (size_t stage = 0; stage < traj16->size(); ++stage) {
                SCOPED_TRACE(numerical::kStageNames[stage]);
                ExpectStageQuality((*traj16)[stage], (*traj32)[stage], act_dtype,
                                   std::string("o1") + "_" + (bfloat16 ? "bf16" : "f16") + "_" +
                                           std::string(numerical::kStageNames[stage]));
                for (size_t index = 0; index < (*traj16)[stage].logits.values.size(); ++index) {
                    if ((*traj16)[stage].logits.values[index] !=
                        (*traj32)[stage].logits.values[index]) {
                        ++differing;
                    }
                }
            }
            // True A16 evidence: silently executing the F32 kernels would make
            // the two trajectories bit-identical.
            EXPECT_GT(differing, 0U) << "A16 execution is indistinguishable from F32/A32";

            auto session = InferenceSession::Create(
                    runtime16, std::make_shared<ExecutableModel>(std::move(*model16)));
            ASSERT_TRUE(session.ok()) << session.status().ToString();
            constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
            const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
            ASSERT_TRUE(generated.ok()) << generated.status().ToString();
            EXPECT_EQ(generated->size(), 3U);
            const auto repeated = session->Generate(prompt, {.max_new_tokens = 3});
            ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
            EXPECT_EQ(*repeated, *generated);
            ++verified_combinations;
        }
    }
    EXPECT_EQ(verified_combinations, 2U);
}

StatusOr<numerical::CapturedTrajectory> RunLongTrajectory(Runtime& runtime,
                                                          const ExecutableModel& model,
                                                          size_t prompt_len) {
    constexpr std::array<int64_t, 2> decode_inputs{17, 23};
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("Long A16 trajectory requires a KV manager");
    }
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(prompt_len, decode_inputs.size()));
    numerical::SessionReservation reservation(*manager, view);
    const size_t workspace_bytes = std::max(prefill->total_workspace_bytes(),
                                            decode->total_workspace_bytes());
    const size_t alignment = std::max(prefill->workspace_alignment(),
                                      decode->workspace_alignment());
    Buffer workspace_storage;
    if (workspace_bytes != 0) {
        AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
        AM_ASSIGN_OR_RETURN(workspace_storage,
                            allocator->TryAllocate({.nbytes = workspace_bytes,
                                                    .alignment = alignment}));
    }
    CpuWorkspaceArena workspace(workspace_bytes == 0 ? nullptr : workspace_storage.mutable_data(),
                                workspace_bytes);
    WorkspaceArena* const workspace_ptr = workspace_bytes == 0 ? nullptr : &workspace;
    numerical::CapturedTrajectory snapshots;

    constexpr std::array<int64_t, 3> pattern{1, 7, 3};
    test::TestBuffer prompt(DataType::Int(64), {static_cast<int64_t>(prompt_len)});
    test::TestBuffer positions(DataType::Int(64), {static_cast<int64_t>(prompt_len)});
    std::vector<int64_t> prefill_rows(prompt_len);
    for (size_t i = 0; i < prompt_len; ++i) {
        static_cast<int64_t*>(prompt.mutable_data())[i] = pattern[i % pattern.size()];
        static_cast<int64_t*>(positions.mutable_data())[i] = static_cast<int64_t>(i);
        prefill_rows[i] = static_cast<int64_t>(i);
    }
    {
        AM_ASSIGN_OR_RETURN(auto context, numerical::PrepareContext(
                                                  runtime, *prefill, *prefill_weights, prompt,
                                                  positions, workspace_ptr, view));
        AM_RETURN_IF_ERROR(Executor::Execute(*prefill, context));
        AM_ASSIGN_OR_RETURN(snapshots[0], numerical::CaptureStage(context, *prefill, prefill_rows,
                                                                  prompt_len, view));
    }
    test::TestBuffer token(DataType::Int(64), {1});
    test::TestBuffer position(DataType::Int(64), {1});
    const std::array<int64_t, 1> decode_rows{0};
    {
        AM_ASSIGN_OR_RETURN(auto context, numerical::PrepareContext(
                                                  runtime, *decode, *decode_weights, token,
                                                  position, workspace_ptr, view));
        for (size_t i = 0; i < decode_inputs.size(); ++i) {
            static_cast<int64_t*>(token.mutable_data())[0] = decode_inputs[i];
            static_cast<int64_t*>(position.mutable_data())[0] = static_cast<int64_t>(prompt_len + i);
            workspace.Reset();
            AM_RETURN_IF_ERROR(Executor::Execute(*decode, context));
            AM_ASSIGN_OR_RETURN(snapshots[i + 1],
                                numerical::CaptureStage(context, *decode, decode_rows, 1, view));
        }
    }
    AM_RETURN_IF_ERROR(reservation.Release());
    return snapshots;
}

TEST(A16ModelAcceptance, NearCapacityLongContextMatchesKv32WithinBudgets) {
    {
        constexpr bool bfloat16 = true;
        const DataType act_dtype = DataType::BFloat(16);
        SCOPED_TRACE("Bf16");
        const CompatibilityCase test_case{bfloat16, false, 4, "A16NearCap", "float32",
                                          DataType::Float32()};
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), test_case);
        constexpr size_t prompt_len = 14;
        Runtime runtime16 = MakeA16Runtime(test_case.kv_heads, act_dtype, 16);
        Runtime runtime32 = MakeA16Runtime(test_case.kv_heads, DataType::Float32(), 16);
        auto model16 = ExecutableModel::Load(runtime16, pair.source,
                                             A16Options(bfloat16, CompilePreset::kO1Plain));
        ASSERT_TRUE(model16.ok()) << model16.status().ToString();
        auto model32 = ExecutableModel::Load(runtime32, pair.source,
                                             F32Options(bfloat16, CompilePreset::kO1Plain));
        ASSERT_TRUE(model32.ok()) << model32.status().ToString();

        const auto traj16 = RunLongTrajectory(runtime16, *model16, prompt_len);
        ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
        const auto traj32 = RunLongTrajectory(runtime32, *model32, prompt_len);
        ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();
        for (size_t stage = 0; stage < traj16->size(); ++stage) {
            SCOPED_TRACE(numerical::kStageNames[stage]);
            ExpectStageQuality((*traj16)[stage], (*traj32)[stage], act_dtype,
                               std::string(bfloat16 ? "bf16" : "f16") + "_nearcap_" +
                                       std::string(numerical::kStageNames[stage]));
        }
        EXPECT_EQ((*traj16)[2].position, 16U);
    }
}

TEST(A16ModelAcceptance, GqaTiedGeometryMatchesKv32WithinBudgets) {
    constexpr bool bfloat16 = true;
    const DataType act_dtype = DataType::BFloat(16);
    const CompatibilityCase test_case{true, true, 2, "A16GqaTied", "auto", DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime16 = MakeA16Runtime(test_case.kv_heads, act_dtype);
    Runtime runtime32 = MakeA16Runtime(test_case.kv_heads, DataType::Float32());
    auto model16 = ExecutableModel::Load(runtime16, pair.source,
                                         A16Options(bfloat16, CompilePreset::kO1Plain));
    ASSERT_TRUE(model16.ok()) << model16.status().ToString();
    auto model32 = ExecutableModel::Load(runtime32, pair.source,
                                         F32Options(bfloat16, CompilePreset::kO1Plain));
    ASSERT_TRUE(model32.ok()) << model32.status().ToString();
    const auto traj16 = ObserveCompatibilityTrajectory(runtime16, *model16);
    ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
    const auto traj32 = ObserveCompatibilityTrajectory(runtime32, *model32);
    ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();
    for (size_t stage = 0; stage < traj16->size(); ++stage) {
        SCOPED_TRACE(numerical::kStageNames[stage]);
        ExpectStageQuality((*traj16)[stage], (*traj32)[stage], act_dtype,
                           std::string("gqa_tied_") + std::string(numerical::kStageNames[stage]));
    }
}

TEST(A16ModelAcceptance, DecodeExecutionHasNoMallocFamilyCallsAfterBinding) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "The malloc-family gate requires the glibc interposer without sanitizer interception";
    }
    size_t verified_combinations = 0;
    for (const CompilePreset preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        {
            constexpr bool bfloat16 = true;
            const DataType act_dtype = DataType::BFloat(16);
            SCOPED_TRACE(preset == CompilePreset::kO1Plain ? "O1/Bf16" : "O2/Bf16");
            const CompatibilityCase test_case{bfloat16, false, 4, "A16ZeroAlloc", "float32",
                                              DataType::Float32()};
            TempDirectory directory;
            const auto pair = WriteCheckpointPair(directory.path(), test_case);
            Runtime runtime = MakeA16Runtime(test_case.kv_heads, act_dtype);
            auto model = ExecutableModel::Load(runtime, pair.source, A16Options(bfloat16, preset));
            ASSERT_TRUE(model.ok()) << model.status().ToString();
            std::array<test::MallocCallCounts, 2> counts{};
            const auto trajectory = ObserveCompatibilityTrajectory(
                    runtime, *model, {.decode_allocations = &counts});
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
            ++verified_combinations;
        }
    }
    EXPECT_EQ(verified_combinations, 2U);
}

TEST(A16ModelAcceptance, RejectsRuntimeKvDtypeThatDiffersFromActivation) {
    constexpr bool bfloat16 = true;
    const CompatibilityCase test_case{bfloat16, false, 4, "A16Mismatch", "float32",
                                      DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    // Activation BF16 but the runtime owns an F16 KV cache.
    Runtime runtime = MakeA16Runtime(test_case.kv_heads, DataType::Float(16));
    const auto model = ExecutableModel::Load(runtime, pair.source,
                                             A16Options(bfloat16, CompilePreset::kO1Plain));
    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status(), StatusCode::kInvalidArgument);
}

} // namespace
