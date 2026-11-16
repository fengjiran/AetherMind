#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_options.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/fixtures/a16_reference_identities.h"
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

// Implementation conformance compares the real CPU A16 path with the
// independent FP64-reduction oracle of the same quantized data flow; the quality
// layer compares the same W16 checkpoint executed as A32/KV32. Both use the
// budgets frozen in proposal 11 section 4.5.

#ifndef AETHERMIND_TEST_MODELS_DIR
#error "Calibrated A16 acceptance requires AETHERMIND_TEST_MODELS_DIR"
#endif

struct FrozenBudget {
    float logits_atol;
    float logits_rtol;
    float kv_atol;
    float kv_rtol;
};

FrozenBudget ImplementationBudget() {
    // BF16 same-type triplet; the FP16 preset belongs to P4-B.
    return {2e-4F, 2e-3F, 2e-3F, 2e-2F};
}

FrozenBudget QualityBudget() {
    return {1e-2F, 3e-2F, 5e-3F, 3e-2F};
}

Status VerifyReferenceBundle(const fs::path& root) {
    for (const auto& expected: test::a16::kFileIdentities) {
        const fs::path path = root / expected.path;
        std::error_code error;
        const auto size = fs::file_size(path, error);
        if (error || size != expected.bytes) {
            return Status::FailedPrecondition("Missing or resized required A16 asset: " +
                                              std::string(expected.path));
        }
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::FailedPrecondition("Missing required A16 asset: " +
                                              std::string(expected.path));
        }
        uint64_t bytes = 0;
        uint64_t fingerprint = 14695981039346656037ULL;
        std::array<char, 8192> buffer{};
        while (input.read(buffer.data(), buffer.size()) || input.gcount() != 0) {
            const auto count = input.gcount();
            bytes += static_cast<uint64_t>(count);
            for (std::streamsize index = 0; index < count; ++index) {
                fingerprint =
                        (fingerprint ^ static_cast<unsigned char>(buffer[index])) * 1099511628211ULL;
            }
        }
        if (!input.eof() || bytes != expected.bytes || fingerprint != expected.fnv1a64) {
            return Status::InvalidArgument("Changed required A16 asset: " +
                                           std::string(expected.path));
        }
    }
    return Status::Ok();
}

StatusOr<numerical::ReferenceTrajectory> LoadIndependentReference(const fs::path& path) {
    AM_ASSIGN_OR_RETURN(auto file, HfSafetensorsFile::Open(path));
    if (file.Entries().size() != 22) {
        return Status::InvalidArgument("Independent A16 reference requires exactly 22 tensors");
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
            return Status::InvalidArgument("Independent A16 reference has an invalid watermark");
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

// Fixed ranking rule (proposal 11 section 4.5): classification derives from the
// reference logits and budget only; top-1 / top-4 sets must match when their
// margin exceeds 2*epsilon, otherwise the row is a near-tie.
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

ModelLoadOptions A16Options(CompilePreset preset) {
    return {.preset = preset,
            .weights = WeightPolicy::kKeepBf16WeightsFp32Activation,
            .kv_dtype = DataType::BFloat(16),
            .activation_dtype = DataType::BFloat(16)};
}

ModelLoadOptions A32Options(CompilePreset preset) {
    return {.preset = preset,
            .weights = WeightPolicy::kKeepBf16WeightsFp32Activation,
            .kv_dtype = DataType::Float32()};
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
        return Status::Internal("A16 calibrated trajectory requires a KV manager");
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
    AM_RETURN_IF_ERROR(reservation.Release());
    return snapshots;
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

void CheckCaseAgainstIndependentOracle(std::string_view case_id, bool run_quality_layer) {
    const fs::path root = fs::path(AETHERMIND_TEST_MODELS_DIR) / "a16-reference";
    const auto verified = VerifyReferenceBundle(root);
    ASSERT_TRUE(verified.ok()) << verified.ToString()
                               << "; prepare assets with tools/prepare_a16_reference.py";
    const fs::path model_dir = root / case_id / "wbf16";
    auto loaded = ModelLoader::Load(model_dir);
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    const HfModelConfig config = (*loaded)->GetConfig();
    const fs::path oracle_path = model_dir / "oracle-af16.safetensors";
    const auto independent = LoadIndependentReference(oracle_path);
    ASSERT_TRUE(independent.ok()) << independent.status().ToString();
    const auto valid = numerical::ValidateReference(*independent, config,
                                                    case_id == "mha-near-limit");
    ASSERT_TRUE(valid.ok()) << valid.ToString();

    for (const auto preset: {CompilePreset::kO1Plain, CompilePreset::kO2Packed}) {
        const std::string preset_name = preset == CompilePreset::kO1Plain ? "o1" : "o2";
        SCOPED_TRACE(std::string(case_id) + "/abf16/" + preset_name);
        auto options = MakeCpuRuntimeOptions(config,
                                             {.context_tokens = independent->prompt.size() + 2,
                                              .kv_dtype = DataType::BFloat(16)});
        ASSERT_TRUE(options.ok()) << options.status().ToString();
        RuntimeBuilder builder;
        builder.WithOptions(*options);
        auto runtime_result = builder.TryBuild();
        ASSERT_TRUE(runtime_result.ok()) << runtime_result.status().ToString();
        Runtime runtime = std::move(*runtime_result);
        auto model = ExecutableModel::Load(runtime, model_dir, A16Options(preset));
        ASSERT_TRUE(model.ok()) << model.status().ToString();
        const auto actual = RunTrajectory(runtime, *model, independent->prompt,
                                          independent->decode_inputs,
                                          independent->stages[0].logit_rows);
        ASSERT_TRUE(actual.ok()) << actual.status().ToString();
        const FrozenBudget budget = ImplementationBudget();
        std::string prefix = std::string(case_id) + "_abf16_" + preset_name;
        std::replace(prefix.begin(), prefix.end(), '-', '_');
        for (size_t stage = 0; stage < numerical::kStageNames.size(); ++stage) {
            const auto& a = (*actual)[stage];
            const auto& expected = independent->stages[stage];
            const std::string property =
                    prefix + "_" + std::string(numerical::kStageNames[stage]);
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

        if (!run_quality_layer) {
            continue;
        }

        // Quality layer: the same W16 checkpoint executed as A32/KV32.
        auto baseline_options = MakeCpuRuntimeOptions(
                config, {.context_tokens = independent->prompt.size() + 2,
                         .kv_dtype = DataType::Float32()});
        ASSERT_TRUE(baseline_options.ok()) << baseline_options.status().ToString();
        RuntimeBuilder baseline_builder;
        baseline_builder.WithOptions(*baseline_options);
        auto baseline_runtime_result = baseline_builder.TryBuild();
        ASSERT_TRUE(baseline_runtime_result.ok()) << baseline_runtime_result.status().ToString();
        Runtime baseline_runtime = std::move(*baseline_runtime_result);
        auto baseline = ExecutableModel::Load(baseline_runtime, model_dir, A32Options(preset));
        ASSERT_TRUE(baseline.ok()) << baseline.status().ToString();
        const auto baseline_trajectory = RunTrajectory(
                baseline_runtime, *baseline, independent->prompt,
                independent->decode_inputs, independent->stages[0].logit_rows);
        ASSERT_TRUE(baseline_trajectory.ok()) << baseline_trajectory.status().ToString();
        const FrozenBudget quality = QualityBudget();
        for (size_t stage = 0; stage < numerical::kStageNames.size(); ++stage) {
            SCOPED_TRACE(numerical::kStageNames[stage]);
            const auto& a = (*actual)[stage];
            const auto& r = (*baseline_trajectory)[stage];
            EXPECT_EQ(a.position, r.position);
            EXPECT_EQ(a.logit_rows, r.logit_rows);
            float norm = 0;
            EXPECT_TRUE(ScalarWithin(a.logits.values, r.logits.values,
                                     quality.logits_atol, quality.logits_rtol, &norm));
            EXPECT_TRUE(LogitsRankingWithin(a.logits, r.logits,
                                            quality.logits_atol, quality.logits_rtol));
            EXPECT_EQ(a.keys.shape, r.keys.shape);
            EXPECT_EQ(a.values.shape, r.values.shape);
            EXPECT_TRUE(ScalarWithin(a.keys.values, r.keys.values,
                                     quality.kv_atol, quality.kv_rtol, &norm));
            EXPECT_TRUE(ScalarWithin(a.values.values, r.values.values,
                                     quality.kv_atol, quality.kv_rtol, &norm));
        }
        EXPECT_EQ((*actual)[2].position, independent->prompt.size() + 2);
    }
}

TEST(A16CalibratedAcceptance, ShortFixturesMatchIndependentQuantizedOracle) {
    for (const std::string_view case_id: {"mha-untied", "mha-tied", "gqa-tail", "mqa-tail"}) {
        CheckCaseAgainstIndependentOracle(case_id, false);
    }
}

TEST(A16CalibratedAcceptance, NearLimitMatchesIndependentQuantizedOracleAndKv32Quality) {
    CheckCaseAgainstIndependentOracle("mha-near-limit", true);
}

TEST(A16CalibratedAcceptance, RankingRejectsMalformedOrNonFiniteLogits) {
    const numerical::TensorData<float> reference{.shape = {1, 5}, .values = {5, 4, 3, 2, 1}};
    auto malformed = reference;
    malformed.values.pop_back();
    EXPECT_FALSE(LogitsRankingWithin(malformed, reference, 2e-4F, 2e-3F));
    for (const float invalid: {std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()}) {
        auto actual = reference;
        actual.values[2] = invalid;
        EXPECT_FALSE(LogitsRankingWithin(actual, reference, 2e-4F, 2e-3F));
        EXPECT_FALSE(LogitsRankingWithin(reference, actual, 2e-4F, 2e-3F));
    }
}

} // namespace
