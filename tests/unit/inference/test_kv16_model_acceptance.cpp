#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_options.h"
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

// Reduced-fixture quality checks compare the production KV16 path with the
// same W16 checkpoint using KV32. The calibrated-acceptance suite separately
// compares production execution with the independent quantized oracle.

// Reduced-fixture quality tolerances for the KV16-vs-KV32 production comparison.
// These are NOT the §2.8 frozen budgets: those were calibrated for the
// tiny-random-Llama numerical fixtures. The reduced checkpoint (hidden 8,
// head_dim 2) has a larger V-projection scale, so its K/V tolerance is measured
// for this fixture (max normalized error: logits <=0.37, keys <=0.24,
// values <=0.73 with the budgets below). The §2.8 budgets remain the acceptance
// authority for the calibrated numerical fixtures.
struct QualityBudget {
    float logits_atol;
    float logits_rtol;
    float kv_atol;
    float kv_rtol;
};

QualityBudget BudgetFor(DataType kv_dtype) {
    if (kv_dtype == DataType::Float(16)) {
        return {1e-3F, 2e-3F, 1e-3F, 1e-2F};
    }
    return {5e-3F, 1e-2F, 4e-3F, 4e-2F};
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

// Fixed ranking rule (§2.8): classification derives from the reference logits
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
                        const numerical::StageSnapshot& ref, DataType kv_dtype) {
    const QualityBudget budget = BudgetFor(kv_dtype);
    EXPECT_EQ(actual.position, ref.position);
    EXPECT_EQ(actual.logit_rows, ref.logit_rows);
    EXPECT_TRUE(ScalarWithin(actual.logits.values, ref.logits.values,
                             budget.logits_atol, budget.logits_rtol));
    EXPECT_TRUE(LogitsRankingWithin(actual.logits, ref.logits,
                                    budget.logits_atol, budget.logits_rtol));
    EXPECT_EQ(actual.keys.shape, ref.keys.shape);
    EXPECT_EQ(actual.values.shape, ref.values.shape);
    EXPECT_TRUE(ScalarWithin(actual.keys.values, ref.keys.values, budget.kv_atol, budget.kv_rtol));
    EXPECT_TRUE(ScalarWithin(actual.values.values, ref.values.values, budget.kv_atol, budget.kv_rtol));
}

Runtime MakeKvRuntime(int64_t kv_heads, DataType kv_dtype, size_t max_tokens = 8) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = static_cast<size_t>(kv_heads),
                        .max_tokens = max_tokens,
                        .head_dim = 2,
                        .kv_dtype = kv_dtype,
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

ModelLoadOptions W16KvOptions(bool bfloat16, DataType kv_dtype, CompilePreset preset) {
    return {.preset = preset,
            .weights = bfloat16 ? WeightPolicy::kKeepBf16WeightsFp32Activation
                                : WeightPolicy::kKeepFp16WeightsFp32Activation,
            .kv_dtype = kv_dtype};
}

TEST(KV16ModelAcceptance, FourCombinationsMatchKv32WithinCalibratedBudgetsO1ThenO2) {
    for (const CompilePreset preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        for (const bool weight_bf16: {false, true}) {
            for (const DataType kv_dtype: {DataType::Float(16), DataType::BFloat(16)}) {
                SCOPED_TRACE(std::string(preset == CompilePreset::kO1Plain ? "O1/" : "O2/") +
                             (weight_bf16 ? "Wbf16/" : "Wf16/") + ToString(kv_dtype));
                const CompatibilityCase test_case{weight_bf16, false, 4, "KV16Acceptance",
                                                  "float32", DataType::Float32()};
                TempDirectory directory;
                const auto pair = WriteCheckpointPair(directory.path(), test_case);

                Runtime runtime16 = MakeKvRuntime(test_case.kv_heads, kv_dtype);
                Runtime runtime32 = MakeKvRuntime(test_case.kv_heads, DataType::Float32());
                auto model16 = ExecutableModel::Load(runtime16, pair.source,
                                                     W16KvOptions(weight_bf16, kv_dtype, preset));
                ASSERT_TRUE(model16.ok()) << model16.status().ToString();
                auto model32 = ExecutableModel::Load(runtime32, pair.source,
                                                     W16KvOptions(weight_bf16, DataType::Float32(), preset));
                ASSERT_TRUE(model32.ok()) << model32.status().ToString();

                const auto traj16 = ObserveCompatibilityTrajectory(runtime16, *model16);
                ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
                const auto traj32 = ObserveCompatibilityTrajectory(runtime32, *model32);
                ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();
                for (size_t stage = 0; stage < traj16->size(); ++stage) {
                    SCOPED_TRACE(numerical::kStageNames[stage]);
                    ExpectStageQuality((*traj16)[stage], (*traj32)[stage], kv_dtype);
                }

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
            }
        }
    }
}

StatusOr<numerical::CapturedTrajectory> RunLongTrajectory(Runtime& runtime,
                                                          const ExecutableModel& model,
                                                          size_t prompt_len,
                                                          std::array<int64_t, 2> decode_inputs) {
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("Long trajectory requires a KV manager");
    }
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(prompt_len, decode_inputs.size()));
    numerical::SessionReservation reservation(*manager, view);
    const size_t workspace_bytes = std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes());
    const size_t alignment = std::max(prefill->workspace_alignment(), decode->workspace_alignment());
    Buffer workspace_storage;
    if (workspace_bytes != 0) {
        AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
        AM_ASSIGN_OR_RETURN(workspace_storage,
                            allocator->TryAllocate({.nbytes = workspace_bytes, .alignment = alignment}));
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
    return snapshots;
}

TEST(KV16ModelAcceptance, NearCapacityLongContextMatchesKv32WithinBudgets) {
    // The reduced fixture caps context at 16; 14 Prefill + 2 Decode reaches the
    // capacity boundary, exercising long-context KV16 accumulation end to end.
    for (const DataType kv_dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        SCOPED_TRACE(ToString(kv_dtype));
        const CompatibilityCase test_case{false, false, 4, "KV16NearCap", "float32",
                                          DataType::Float32()};
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), test_case);
        constexpr size_t prompt_len = 14;
        Runtime runtime16 = MakeKvRuntime(test_case.kv_heads, kv_dtype, 16);
        Runtime runtime32 = MakeKvRuntime(test_case.kv_heads, DataType::Float32(), 16);
        auto model16 = ExecutableModel::Load(runtime16, pair.source,
                                             W16KvOptions(false, kv_dtype, CompilePreset::kO1Plain));
        ASSERT_TRUE(model16.ok()) << model16.status().ToString();
        auto model32 = ExecutableModel::Load(runtime32, pair.source,
                                             W16KvOptions(false, DataType::Float32(), CompilePreset::kO1Plain));
        ASSERT_TRUE(model32.ok()) << model32.status().ToString();
        const auto traj16 = RunLongTrajectory(runtime16, *model16, prompt_len, {17, 23});
        ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
        const auto traj32 = RunLongTrajectory(runtime32, *model32, prompt_len, {17, 23});
        ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();
        for (size_t stage = 0; stage < traj16->size(); ++stage) {
            SCOPED_TRACE(numerical::kStageNames[stage]);
            ExpectStageQuality((*traj16)[stage], (*traj32)[stage], kv_dtype);
        }
        EXPECT_EQ((*traj16)[2].position, 16U);
    }
}

TEST(KV16ModelAcceptance, GqaTiedGeometryMatchesKv32WithinBudgets) {
    const DataType kv_dtype = DataType::BFloat(16);
    const CompatibilityCase test_case{true, true, 2, "KV16GqaTied", "auto", DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime16 = MakeKvRuntime(test_case.kv_heads, kv_dtype);
    Runtime runtime32 = MakeKvRuntime(test_case.kv_heads, DataType::Float32());
    auto model16 = ExecutableModel::Load(runtime16, pair.source,
                                         W16KvOptions(true, kv_dtype, CompilePreset::kO1Plain));
    ASSERT_TRUE(model16.ok()) << model16.status().ToString();
    auto model32 = ExecutableModel::Load(runtime32, pair.source,
                                         W16KvOptions(true, DataType::Float32(), CompilePreset::kO1Plain));
    ASSERT_TRUE(model32.ok()) << model32.status().ToString();
    const auto traj16 = ObserveCompatibilityTrajectory(runtime16, *model16);
    ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
    const auto traj32 = ObserveCompatibilityTrajectory(runtime32, *model32);
    ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();
    for (size_t stage = 0; stage < traj16->size(); ++stage) {
        SCOPED_TRACE(numerical::kStageNames[stage]);
        ExpectStageQuality((*traj16)[stage], (*traj32)[stage], kv_dtype);
    }
}

TEST(KV16ModelAcceptance, DecodeExecutionHasNoMallocFamilyCallsAfterBinding) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "The malloc-family gate requires the glibc interposer without sanitizer interception";
    }
    size_t verified_combinations = 0;
    for (const CompilePreset preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        for (const bool weight_bf16: {false, true}) {
            for (const DataType kv_dtype: {DataType::Float(16), DataType::BFloat(16)}) {
                SCOPED_TRACE(std::string(preset == CompilePreset::kO1Plain ? "O1/" : "O2/") +
                             (weight_bf16 ? "Wbf16/" : "Wf16/") + ToString(kv_dtype));
                const CompatibilityCase test_case{weight_bf16, false, 4, "KV16ZeroAlloc", "float32",
                                                  DataType::Float32()};
                TempDirectory directory;
                const auto pair = WriteCheckpointPair(directory.path(), test_case);
                Runtime runtime = MakeKvRuntime(test_case.kv_heads, kv_dtype);
                auto model = ExecutableModel::Load(runtime, pair.source,
                                                   W16KvOptions(weight_bf16, kv_dtype, preset));
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
    }
    RecordProperty("verified_w_kv_preset_combinations", static_cast<int>(verified_combinations));
}

} // namespace
