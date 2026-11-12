#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_options.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/fixtures/kv16_reference_identities.h"
#include "inference/test_hf_fixed_trajectory_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;
namespace numerical = aethermind::test::numerical;
namespace fs = std::filesystem;

// Separate implementation conformance (independent quantized FP64 oracle) from
// quality loss (same production W16 model with KV32). Both use frozen budgets.

#ifndef AETHERMIND_TEST_MODELS_DIR
#error "Calibrated KV16 acceptance requires AETHERMIND_TEST_MODELS_DIR"
#endif

struct FrozenBudget {
    float logits_atol;
    float logits_rtol;
    float kv_atol;
    float kv_rtol;
};

FrozenBudget FrozenBudgetFor(DataType kv_dtype) {
    if (kv_dtype == DataType::Float(16)) {
        return {1e-3F, 2e-3F, 4e-4F, 4e-3F};
    }
    return {5e-3F, 1e-2F, 2e-3F, 2e-2F};
}

FrozenBudget ImplementationBudgetFor(DataType kv_dtype) {
    return kv_dtype == DataType::Float(16)
                   ? FrozenBudget{1e-6F, 2e-5F, 2e-4F, 2e-3F}
                   : FrozenBudget{1e-6F, 2e-5F, 1.5e-3F, 2e-3F};
}

Status VerifyReferenceBundle(const fs::path& root) {
    for (const auto& expected: test::kv16::kFileIdentities) {
        const fs::path path = root / expected.path;
        std::error_code error;
        const auto size = fs::file_size(path, error);
        if (error || size != expected.bytes) {
            return Status::FailedPrecondition("Missing or resized required KV16 asset: " + std::string(expected.path));
        }
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::FailedPrecondition("Missing required KV16 asset: " + std::string(expected.path));
        }
        uint64_t bytes = 0;
        uint64_t fingerprint = 14695981039346656037ULL;
        std::array<char, 8192> buffer{};
        while (input.read(buffer.data(), buffer.size()) || input.gcount() != 0) {
            const auto count = input.gcount();
            bytes += static_cast<uint64_t>(count);
            for (std::streamsize index = 0; index < count; ++index) {
                fingerprint = (fingerprint ^ static_cast<unsigned char>(buffer[index])) * 1099511628211ULL;
            }
        }
        if (!input.eof() || bytes != expected.bytes || fingerprint != expected.fnv1a64) {
            return Status::InvalidArgument("Changed required KV16 asset: " + std::string(expected.path));
        }
    }
    return Status::Ok();
}

StatusOr<numerical::ReferenceTrajectory> LoadIndependentReference(const fs::path& path) {
    AM_ASSIGN_OR_RETURN(auto file, HfSafetensorsFile::Open(path));
    if (file.Entries().size() != 22) {
        return Status::InvalidArgument("Independent KV16 reference requires exactly 22 tensors");
    }
    numerical::ReferenceTrajectory reference;
    AM_ASSIGN_OR_RETURN(reference.prompt, numerical::ReadVector<int64_t>(file, "prompt"));
    AM_ASSIGN_OR_RETURN(reference.decode_inputs, numerical::ReadVector<int64_t>(file, "decode_inputs"));
    AM_ASSIGN_OR_RETURN(reference.greedy_tokens, numerical::ReadVector<int64_t>(file, "greedy_tokens"));
    AM_ASSIGN_OR_RETURN(reference.greedy_margins, numerical::ReadVector<float>(file, "greedy_margins"));
    for (size_t index = 0; index < numerical::kStageNames.size(); ++index) {
        const std::string prefix(numerical::kStageNames[index]);
        auto& stage = reference.stages[index];
        AM_ASSIGN_OR_RETURN(stage.logit_rows, numerical::ReadVector<int64_t>(file, prefix + ".logit_rows"));
        AM_ASSIGN_OR_RETURN(stage.logits, numerical::ReadTensor<float>(file, prefix + ".logits"));
        AM_ASSIGN_OR_RETURN(stage.tokens, numerical::ReadVector<int64_t>(file, prefix + ".tokens"));
        AM_ASSIGN_OR_RETURN(stage.keys, numerical::ReadTensor<float>(file, prefix + ".keys"));
        AM_ASSIGN_OR_RETURN(stage.values, numerical::ReadTensor<float>(file, prefix + ".values"));
        AM_ASSIGN_OR_RETURN(auto position, numerical::ReadVector<int64_t>(file, prefix + ".position"));
        if (position.size() != 1 || position[0] <= 0 || !std::in_range<size_t>(position[0])) {
            return Status::InvalidArgument("Independent KV16 reference has an invalid watermark");
        }
        stage.position = static_cast<size_t>(position[0]);
    }
    return reference;
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

Runtime MakeKvRuntime(DataType kv_dtype) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = 4,
                        .max_tokens = 2048,
                        .head_dim = 4,
                        .kv_dtype = kv_dtype,
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

ModelLoadOptions W16KvOptions(bool weight_bf16, DataType kv_dtype, CompilePreset preset) {
    return {.preset = preset,
            .weights = weight_bf16 ? WeightPolicy::kKeepBf16WeightsFp32Activation
                                   : WeightPolicy::kKeepFp16WeightsFp32Activation,
            .kv_dtype = kv_dtype};
}

StatusOr<numerical::CapturedTrajectory> RunTrajectory(Runtime& runtime,
                                                      const ExecutableModel& model,
                                                      std::span<const int64_t> prompt_ids,
                                                      std::span<const int64_t> decode_inputs,
                                                      std::span<const int64_t> prefill_rows) {
    const size_t prompt_len = prompt_ids.size();
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("near-limit trajectory requires a KV manager");
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

    test::TestBuffer prompt(DataType::Int(64), {static_cast<int64_t>(prompt_len)});
    test::TestBuffer positions(DataType::Int(64), {static_cast<int64_t>(prompt_len)});
    for (size_t i = 0; i < prompt_len; ++i) {
        static_cast<int64_t*>(prompt.mutable_data())[i] = prompt_ids[i];
        static_cast<int64_t*>(positions.mutable_data())[i] = static_cast<int64_t>(i);
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

StatusOr<numerical::CapturedTrajectory> RunNearLimitTrajectory(Runtime& runtime,
                                                               const ExecutableModel& model) {
    std::vector<int64_t> prompt(2046);
    constexpr std::array<int64_t, 3> pattern{1, 7, 3};
    for (size_t index = 0; index < prompt.size(); ++index) {
        prompt[index] = pattern[index % pattern.size()];
    }
    constexpr std::array<int64_t, 2> decode_inputs{17, 23};
    constexpr std::array<int64_t, 1> last_row{2045};
    return RunTrajectory(runtime, model, prompt, decode_inputs, last_row);
}

TEST(KV16CalibratedAcceptance, NearLimitFourCombinationsMatchKv32WithinFrozenBudgets) {
    const fs::path models_dir = fs::path(AETHERMIND_TEST_MODELS_DIR) / "kv16-reference";
    const auto verified = VerifyReferenceBundle(models_dir);
    ASSERT_TRUE(verified.ok()) << verified.ToString();
    for (const CompilePreset preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        for (const bool weight_bf16: {false, true}) {
            const fs::path model_dir = models_dir / "mha-near-limit" / (weight_bf16 ? "wbf16" : "wf16");
            ASSERT_TRUE(fs::is_regular_file(model_dir / "model.safetensors"))
                    << "Missing required calibrated W16 checkpoint: " << model_dir;
            for (const DataType kv_dtype: {DataType::Float(16), DataType::BFloat(16)}) {
                SCOPED_TRACE(std::string(preset == CompilePreset::kO1Plain ? "O1/" : "O2/") +
                             (weight_bf16 ? "Wbf16/" : "Wf16/") + ToString(kv_dtype));
                Runtime runtime16 = MakeKvRuntime(kv_dtype);
                Runtime runtime32 = MakeKvRuntime(DataType::Float32());
                auto model16 = ExecutableModel::Load(runtime16, model_dir,
                                                     W16KvOptions(weight_bf16, kv_dtype, preset));
                ASSERT_TRUE(model16.ok()) << model16.status().ToString();
                auto model32 = ExecutableModel::Load(runtime32, model_dir,
                                                     W16KvOptions(weight_bf16, DataType::Float32(), preset));
                ASSERT_TRUE(model32.ok()) << model32.status().ToString();
                const auto traj16 = RunNearLimitTrajectory(runtime16, *model16);
                ASSERT_TRUE(traj16.ok()) << traj16.status().ToString();
                const auto traj32 = RunNearLimitTrajectory(runtime32, *model32);
                ASSERT_TRUE(traj32.ok()) << traj32.status().ToString();

                const FrozenBudget budget = FrozenBudgetFor(kv_dtype);
                for (size_t stage = 0; stage < traj16->size(); ++stage) {
                    SCOPED_TRACE(numerical::kStageNames[stage]);
                    const auto& a = (*traj16)[stage];
                    const auto& r = (*traj32)[stage];
                    EXPECT_EQ(a.position, r.position);
                    EXPECT_EQ(a.logit_rows, r.logit_rows);
                    float norm = 0;
                    EXPECT_TRUE(ScalarWithin(a.logits.values, r.logits.values,
                                             budget.logits_atol, budget.logits_rtol, &norm));
                    EXPECT_TRUE(LogitsRankingWithin(a.logits, r.logits,
                                                    budget.logits_atol, budget.logits_rtol));
                    EXPECT_EQ(a.keys.shape, r.keys.shape);
                    EXPECT_EQ(a.values.shape, r.values.shape);
                    EXPECT_TRUE(ScalarWithin(a.keys.values, r.keys.values,
                                             budget.kv_atol, budget.kv_rtol, &norm));
                    EXPECT_TRUE(ScalarWithin(a.values.values, r.values.values,
                                             budget.kv_atol, budget.kv_rtol, &norm));
                }
                EXPECT_EQ((*traj16)[2].position, 2048U);
            }
        }
    }
}

::testing::AssertionResult CompareIndependentTensor(const numerical::TensorData<float>& actual,
                                                    const numerical::TensorData<float>& expected,
                                                    double atol, double rtol,
                                                    const std::string& property) {
    if (actual.shape != expected.shape || actual.values.size() != expected.values.size()) {
        return ::testing::AssertionFailure() << property << ": shape/element count mismatch";
    }
    double max_abs = 0;
    double max_rel = 0;
    double max_normalized = 0;
    size_t worst_index = 0;
    for (size_t index = 0; index < actual.values.size(); ++index) {
        const double value = actual.values[index];
        const double reference = expected.values[index];
        if (!std::isfinite(value) || !std::isfinite(reference)) {
            return ::testing::AssertionFailure() << property << ": non-finite element " << index;
        }
        const double error = std::abs(value - reference);
        const double normalized = error / (atol + rtol * std::abs(reference));
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::abs(reference), 1e-30));
        if (normalized > max_normalized) {
            max_normalized = normalized;
            worst_index = index;
        }
    }
    const auto record = [&](std::string_view metric, double value) {
        std::ostringstream encoded;
        encoded << std::setprecision(17) << value;
        ::testing::Test::RecordProperty(property + "_" + std::string(metric), encoded.str());
    };
    record("max_abs", max_abs);
    record("max_rel", max_rel);
    record("max_normalized", max_normalized);
    if (max_normalized > 1) {
        return ::testing::AssertionFailure() << property << ": budget exceeded at element "
                                             << worst_index << ", normalized error=" << max_normalized
                                             << ", actual=" << actual.values[worst_index]
                                             << ", independent=" << expected.values[worst_index];
    }
    return ::testing::AssertionSuccess();
}

void CheckIndependentCases(std::span<const std::string_view> cases) {
    const fs::path root = fs::path(AETHERMIND_TEST_MODELS_DIR) / "kv16-reference";
    const auto verified = VerifyReferenceBundle(root);
    ASSERT_TRUE(verified.ok()) << verified.ToString()
                               << "; prepare assets with tools/prepare_kv16_reference.py";
    for (const auto case_id: cases) {
        for (const bool weight_bf16: {false, true}) {
            const std::string weight_name = weight_bf16 ? "wbf16" : "wf16";
            const fs::path model_dir = root / case_id / weight_name;
            auto loaded = ModelLoader::Load(model_dir);
            ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
            const HfModelConfig config = (*loaded)->GetConfig();
            for (const auto kv_dtype: {DataType::Float(16), DataType::BFloat(16)}) {
                const std::string kv_name = kv_dtype == DataType::Float(16) ? "kvf16" : "kvbf16";
                const auto independent = LoadIndependentReference(model_dir / ("oracle-" + kv_name + ".safetensors"));
                ASSERT_TRUE(independent.ok()) << independent.status().ToString();
                const auto valid = numerical::ValidateReference(*independent, config, case_id == "mha-near-limit");
                ASSERT_TRUE(valid.ok()) << valid.ToString();
                for (const auto preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
                    const std::string preset_name = preset == CompilePreset::kO1Plain ? "o1" : "o2";
                    SCOPED_TRACE(std::string(case_id) + "/" + weight_name + "/" + kv_name + "/" + preset_name);
                    auto options = MakeCpuRuntimeOptions(config, {.context_tokens = independent->prompt.size() + 2,
                                                                  .kv_dtype = kv_dtype});
                    ASSERT_TRUE(options.ok()) << options.status().ToString();
                    RuntimeBuilder builder;
                    builder.WithOptions(*options);
                    auto runtime_result = builder.TryBuild();
                    ASSERT_TRUE(runtime_result.ok()) << runtime_result.status().ToString();
                    Runtime runtime = std::move(*runtime_result);
                    auto model = ExecutableModel::Load(runtime, model_dir, W16KvOptions(weight_bf16, kv_dtype, preset));
                    ASSERT_TRUE(model.ok()) << model.status().ToString();
                    const auto actual = RunTrajectory(runtime, *model, independent->prompt,
                                                      independent->decode_inputs,
                                                      independent->stages[0].logit_rows);
                    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
                    const auto budget = ImplementationBudgetFor(kv_dtype);
                    std::string prefix = std::string(case_id) + "_" + weight_name + "_" + kv_name + "_" + preset_name;
                    std::replace(prefix.begin(), prefix.end(), '-', '_');
                    for (size_t stage = 0; stage < numerical::kStageNames.size(); ++stage) {
                        const auto& a = (*actual)[stage];
                        const auto& expected = independent->stages[stage];
                        const std::string property = prefix + "_" + std::string(numerical::kStageNames[stage]);
                        EXPECT_EQ(a.position, expected.position);
                        EXPECT_EQ(a.logit_rows, expected.logit_rows);
                        EXPECT_TRUE(CompareIndependentTensor(a.logits, expected.logits, budget.logits_atol,
                                                             budget.logits_rtol, property + "_logits"));
                        EXPECT_TRUE(CompareIndependentTensor(a.keys, expected.keys, budget.kv_atol,
                                                             budget.kv_rtol, property + "_keys"));
                        EXPECT_TRUE(CompareIndependentTensor(a.values, expected.values, budget.kv_atol,
                                                             budget.kv_rtol, property + "_values"));
                        EXPECT_TRUE(LogitsRankingWithin(a.logits, expected.logits,
                                                        budget.logits_atol, budget.logits_rtol));
                    }
                }
            }
        }
    }
}

TEST(KV16CalibratedAcceptance, ShortFixturesMatchIndependentQuantizedOracle) {
    constexpr std::array<std::string_view, 4> cases{"mha-untied", "mha-tied", "gqa-tail", "mqa-tail"};
    CheckIndependentCases(cases);
}

TEST(KV16CalibratedAcceptance, RankingRejectsMalformedOrNonFiniteLogits) {
    const numerical::TensorData<float> reference{.shape = {1, 5}, .values = {5, 4, 3, 2, 1}};
    auto malformed = reference;
    malformed.values.pop_back();
    EXPECT_FALSE(LogitsRankingWithin(malformed, reference, 1e-6F, 2e-5F));
    for (const float invalid: {std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()}) {
        auto actual = reference;
        actual.values[2] = invalid;
        EXPECT_FALSE(LogitsRankingWithin(actual, reference, 1e-6F, 2e-5F));
        EXPECT_FALSE(LogitsRankingWithin(reference, actual, 1e-6F, 2e-5F));
    }
}

TEST(KV16CalibratedAcceptance, NearLimitMatchesIndependentQuantizedOracle) {
    constexpr std::array<std::string_view, 1> cases{"mha-near-limit"};
    CheckIndependentCases(cases);
}

} // namespace
