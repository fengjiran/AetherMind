#include "test_hf_fixed_trajectory_helpers.h"

#include "aethermind/inference/inference_session.h"
#include "model/test_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#ifndef AETHERMIND_TEST_NUMERICAL_DIR
#error "Numerical tests require the configured AETHERMIND_TEST_NUMERICAL_DIR"
#endif

namespace {

using namespace aethermind;
using namespace aethermind::test;
using namespace aethermind::test::numerical;
namespace fs = std::filesystem;

struct FixtureCase {
    const char* directory;
    const char* name;
};

fs::path NumericalDirectory(const char* fixture) {
    return fs::path(AETHERMIND_TEST_NUMERICAL_DIR) / fixture;
}

struct RepeatedGeneration {
    std::vector<uint32_t> first;
    std::vector<uint32_t> repeated;
};

StatusOr<RepeatedGeneration> RunPublicGenerate(const fs::path& model_dir,
                                               const ReferenceTrajectory& reference,
                                               CpuInferencePreset preset) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_dir));
    const HfModelConfig config = loaded->GetConfig();
    AM_RETURN_IF_ERROR(ValidateReference(reference, config, false));
    AM_ASSIGN_OR_RETURN(auto compile_options, MakeCpuCompileOptions(preset));
    AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), compile_options));
    RuntimeOptions options;
    AM_ASSIGN_OR_RETURN(options.kv_cache, MakeFp32KVCacheOptions(config, reference.prompt.size() + 2));
    RuntimeBuilder builder;
    builder.WithOptions(options);
    AM_ASSIGN_OR_RETURN(auto runtime, builder.TryBuild());
    AM_ASSIGN_OR_RETURN(auto executable, PrepareExecutableModel(runtime, std::move(artifact)));
    auto model = std::make_shared<ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(runtime, model));
    std::vector<uint32_t> prompt(reference.prompt.begin(), reference.prompt.end());
    RepeatedGeneration result;
    AM_ASSIGN_OR_RETURN(result.first, session.Generate(prompt, GenerationConfig{.max_new_tokens = 3}));
    AM_ASSIGN_OR_RETURN(result.repeated, session.Generate(prompt, GenerationConfig{.max_new_tokens = 3}));
    return result;
}

class HfFixedTrajectory : public ::testing::TestWithParam<std::tuple<FixtureCase, CpuInferencePreset>> {};

TEST_P(HfFixedTrajectory, FixedTrajectoryMatchesExternalReference) {
    const FixtureCase fixture = std::get<0>(GetParam());
    const auto preset = std::get<1>(GetParam());
    const auto reference = LoadReference(NumericalDirectory(fixture.directory) / "reference.safetensors");
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();

    const auto captured = RunFixedTrajectory(NumericalDirectory(fixture.directory) / "model", *reference, preset);

    ASSERT_TRUE(captured.ok()) << captured.status().ToString();
    for (size_t index = 0; index < captured->size(); ++index) {
        const auto compared = CompareStage((*captured)[index], reference->stages[index], kStageNames[index]);
        EXPECT_TRUE(compared) << compared.message();
    }
}

TEST_P(HfFixedTrajectory, GenerateMatchesExternalGreedyAcrossFreshRequests) {
    const FixtureCase fixture = std::get<0>(GetParam());
    const auto preset = std::get<1>(GetParam());
    const auto reference = LoadReference(NumericalDirectory(fixture.directory) / "reference.safetensors");
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();

    const auto generated = RunPublicGenerate(NumericalDirectory(fixture.directory) / "model", *reference, preset);

    ASSERT_TRUE(generated.ok()) << generated.status().ToString();
    const std::vector<uint32_t> expected(reference->greedy_tokens.begin(), reference->greedy_tokens.end());
    EXPECT_EQ(generated->first, expected) << "reference first-token margin=" << reference->greedy_margins.front();
    EXPECT_EQ(generated->repeated, expected);
}

INSTANTIATE_TEST_SUITE_P(
        NumericalPresets, HfFixedTrajectory,
        ::testing::Combine(
                ::testing::Values(FixtureCase{"mha-untied", "MhaUntied"}, FixtureCase{"mha-tied", "MhaTied"},
                                  FixtureCase{"gqa-tail", "GqaTail"}, FixtureCase{"mqa-tail", "MqaTail"}),
                ::testing::Values(CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed)),
        [](const ::testing::TestParamInfo<HfFixedTrajectory::ParamType>& info) {
            return std::string(std::get<0>(info.param).name) +
                   (std::get<1>(info.param) == CpuInferencePreset::kO1Plain ? "O1Plain" : "O2Packed");
        });

class HfNearContextLimit : public ::testing::TestWithParam<CpuInferencePreset> {};

TEST_P(HfNearContextLimit, MatchesExternalFixedTrajectoryAtContextLimit) {
    const auto directory = NumericalDirectory("mha-near-limit");
    const auto reference = LoadReference(directory / "reference.safetensors");
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();

    const auto captured = RunFixedTrajectory(directory / "model", *reference, GetParam(), true);

    ASSERT_TRUE(captured.ok()) << captured.status().ToString();
    for (size_t index = 0; index < captured->size(); ++index) {
        const auto compared = CompareStage((*captured)[index], reference->stages[index], kStageNames[index]);
        EXPECT_TRUE(compared) << compared.message();
    }
}

INSTANTIATE_TEST_SUITE_P(
        NumericalPresets, HfNearContextLimit,
        ::testing::Values(CpuInferencePreset::kO1Plain, CpuInferencePreset::kO2Packed),
        [](const ::testing::TestParamInfo<CpuInferencePreset>& info) {
            return info.param == CpuInferencePreset::kO1Plain ? "O1Plain" : "O2Packed";
        });

StageSnapshot ComparisonFixture() {
    StageSnapshot result;
    result.position = 2;
    result.logit_rows = {0, 1};
    result.logits = {.shape = {2, 3}, .values = std::vector<float>(6, 0.125F)};
    result.tokens = {1, 2};
    result.keys = {.shape = {2, 2, 1, 2}, .values = std::vector<float>(8, 0.25F)};
    result.values = {.shape = {2, 2, 1, 2}, .values = std::vector<float>(8, -0.375F)};
    return result;
}

TEST(HfNumericalComparison, IdenticalCompleteStageMatches) {
    const auto reference = ComparisonFixture();
    EXPECT_TRUE(CompareStage(reference, reference, "prefill"));
}

TEST(HfNumericalComparison, PerturbedLogitReportsFirstPosition) {
    const auto reference = ComparisonFixture();
    auto actual = reference;
    actual.logits.values[4] += 0.01F;

    const auto compared = CompareStage(actual, reference, "prefill");

    EXPECT_FALSE(compared);
    EXPECT_NE(std::string(compared.message()).find("prefill logits [row=1,column=1]"), std::string::npos);
}

TEST(HfNumericalComparison, PerturbedKeyReportsAllKvCoordinates) {
    const auto reference = ComparisonFixture();
    auto actual = reference;
    actual.keys.values[7] += 0.01F;

    const auto compared = CompareStage(actual, reference, "decode_0");

    EXPECT_FALSE(compared);
    EXPECT_NE(std::string(compared.message()).find("decode_0 keys [layer=1,position=1,head=0,dim=1]"), std::string::npos);
}

TEST(HfNumericalComparison, NonFiniteValueCannotPassTolerance) {
    const auto reference = ComparisonFixture();
    auto actual = reference;
    actual.values.values[0] = std::numeric_limits<float>::quiet_NaN();

    const auto compared = CompareStage(actual, reference, "decode_1");

    EXPECT_FALSE(compared);
    EXPECT_NE(std::string(compared.message()).find("decode_1 values"), std::string::npos);
}

TEST(HfNumericalComparison, MissingLogitElementFailsCompleteness) {
    const auto reference = ComparisonFixture();
    auto actual = reference;
    actual.logits.values.pop_back();
    EXPECT_FALSE(CompareStage(actual, reference, "prefill"));
}

TEST(HfNumericalComparison, ChangedLogitRowsFailIdentity) {
    const auto reference = ComparisonFixture();
    auto actual = reference;
    actual.logit_rows = {1, 0};
    EXPECT_FALSE(CompareStage(actual, reference, "prefill"));
}

TEST(HfNumericalComparison, ChangedWatermarkFailsBeforeTensorComparison) {
    const auto reference = ComparisonFixture();
    auto actual = reference;
    ++actual.position;
    const auto compared = CompareStage(actual, reference, "prefill");
    EXPECT_FALSE(compared);
    EXPECT_NE(std::string(compared.message()).find("watermark mismatch"), std::string::npos);
}

TEST(HfNumericalComparison, MissingReferenceStagesAreRejected) {
    TempDirectory directory;
    const std::array<int64_t, 1> prompt{1};
    const auto path = directory.path() / "incomplete.safetensors";
    WriteSafetensorsFile(path, R"({"prompt":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}})",
                         std::as_bytes(std::span(prompt)));

    const auto reference = LoadReference(path);

    ASSERT_FALSE(reference.ok());
    EXPECT_EQ(reference.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(reference.status().message().find("incomplete"), std::string::npos);
}

TEST(HfNumericalFaultInjection, ChangedNormWeightIsDetected) {
    const auto source = NumericalDirectory("mha-untied");
    const auto reference = LoadReference(source / "reference.safetensors");
    ASSERT_TRUE(reference.ok()) << reference.status().ToString();
    uint64_t offset = 0;
    {
        const auto weights = HfSafetensorsFile::Open(source / "model/model.safetensors");
        ASSERT_TRUE(weights.ok()) << weights.status().ToString();
        const auto* const norm = weights->Find("model.norm.weight");
        ASSERT_NE(norm, nullptr);
        ASSERT_EQ(norm->dtype, DataType::Float32());
        ASSERT_GE(norm->ByteSize(), sizeof(float));
        offset = norm->data_offset_begin;
    }
    TempDirectory directory;
    std::error_code error;
    fs::copy(source / "model", directory.path() / "model", fs::copy_options::recursive, error);
    ASSERT_FALSE(error) << error.message();
    {
        std::fstream stream(directory.path() / "model/model.safetensors", std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(stream.is_open());
        std::array<unsigned char, 8> header{};
        stream.read(reinterpret_cast<char*>(header.data()), header.size());
        ASSERT_TRUE(stream.good());
        uint64_t header_size = 0;
        for (size_t index = 0; index < header.size(); ++index) {
            header_size |= static_cast<uint64_t>(header[index]) << (index * 8U);
        }
        const auto file_offset = static_cast<std::streamoff>(header.size() + header_size + offset);
        stream.seekg(file_offset);
        float weight = 0;
        stream.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        ASSERT_TRUE(stream.good());
        weight += 0.5F;
        stream.seekp(file_offset);
        stream.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        ASSERT_TRUE(stream.good());
    }

    // The cloned HF file remains structurally valid: loading, compilation and
    // execution must succeed before the independent numerical comparison fails.
    const auto captured = RunFixedTrajectory(directory.path() / "model", *reference, CpuInferencePreset::kO1Plain);

    ASSERT_TRUE(captured.ok()) << captured.status().ToString();
    const auto compared = CompareStage((*captured)[0], reference->stages[0], "prefill");
    EXPECT_FALSE(compared);
    EXPECT_NE(std::string(compared.message()).find("prefill logits"), std::string::npos);
}

} // namespace
