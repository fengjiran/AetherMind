#include "aethermind/model/model_compiler.h"

#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <variant>

namespace {

using namespace aethermind;
namespace fs = std::filesystem;

fs::path TestModelDir() {
    return fs::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM";
}

bool HasWeightRecipe(const LoweredGraph& graph, bool (*predicate)(const WeightBindingSpec&)) {
    return std::ranges::any_of(graph.values, [predicate](const LoweredValueDesc& value) {
        const auto* weight = std::get_if<WeightValue>(&value.payload);
        return weight != nullptr && predicate(weight->binding.spec);
    });
}

bool IsQkvRecipe(const WeightBindingSpec& binding) {
    return std::holds_alternative<QkvWeightBinding>(binding);
}

bool IsGateUpRecipe(const WeightBindingSpec& binding) {
    return std::holds_alternative<GateUpWeightBinding>(binding);
}

TEST(ModelCompiler, LoadAndLowerTinyLlamaAtO2) {
    const auto lowered = ModelCompiler::LoadAndLowerModel(
            ModelLoadOptions{.model_dir = TestModelDir()});

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_NE(lowered->loaded_model, nullptr);
    EXPECT_EQ(lowered->loaded_model->GetConfig().num_hidden_layers, 2);
    EXPECT_FALSE(lowered->graph.steps.empty());
    EXPECT_EQ(lowered->graph.steps.size(), lowered->graph.step_bindings.size());
    EXPECT_EQ(lowered->graph.steps.front().op_type, OpType::kEmbedding);
    EXPECT_EQ(lowered->graph.steps.back().op_type, OpType::kArgmax);
    EXPECT_TRUE(HasWeightRecipe(lowered->graph, &IsQkvRecipe));
    EXPECT_TRUE(HasWeightRecipe(lowered->graph, &IsGateUpRecipe));

    for (const LoweredStepBinding& binding: lowered->graph.step_bindings) {
        for (const GraphValueId value: binding.input_values) {
            EXPECT_LT(value.index, lowered->graph.values.size());
        }
        for (const GraphValueId value: binding.output_values) {
            EXPECT_LT(value.index, lowered->graph.values.size());
        }
    }
}

TEST(ModelCompiler, SupportsO0AndO2) {
    ModelLoweringOptions o0;
    o0.optimization.opt_level = 0;

    const auto unoptimized = ModelCompiler::LoadAndLowerModel(
            ModelLoadOptions{.model_dir = TestModelDir()}, o0);
    ASSERT_TRUE(unoptimized.ok()) << unoptimized.status().ToString();
    EXPECT_FALSE(HasWeightRecipe(unoptimized->graph, &IsQkvRecipe));
    EXPECT_FALSE(HasWeightRecipe(unoptimized->graph, &IsGateUpRecipe));
    EXPECT_EQ(unoptimized->graph.steps.size(), 34U);

    const auto optimized = ModelCompiler::LoadAndLowerModel(
            ModelLoadOptions{.model_dir = TestModelDir()});
    ASSERT_TRUE(optimized.ok()) << optimized.status().ToString();
    EXPECT_LT(optimized->graph.steps.size(), unoptimized->graph.steps.size());
}

TEST(ModelCompiler, PropagatesLoweringTargetConfiguration) {
    ModelLoweringOptions options;
    options.lowering.device_type = DeviceType::kCUDA;
    options.lowering.isa = IsaLevel::kAVX2;
    options.lowering.weight_format = WeightFormat::kPacked;
    options.lowering.phase = ExecPhase::kPrefill;

    const auto lowered = ModelCompiler::LoadAndLowerModel(
            ModelLoadOptions{.model_dir = TestModelDir()}, options);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    for (const ExecutionPlanNodeSpec& step: lowered->graph.steps) {
        EXPECT_EQ(step.device_type, DeviceType::kCUDA);
        EXPECT_EQ(step.isa, IsaLevel::kAVX2);
        EXPECT_EQ(step.weight_format, WeightFormat::kPacked);
        EXPECT_EQ(step.phase, ExecPhase::kPrefill);
    }
}

TEST(ModelCompiler, SupportsLinearRopeScalingAndRejectsUnsupportedSemanticVariants) {
    const auto loaded = ModelLoader::Load(ModelLoadOptions{.model_dir = TestModelDir()});
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();

    HfModelConfig linear_config = (*loaded)->GetConfig();
    linear_config.rope.scaling_type = HfRopeScalingType::kLinear;
    linear_config.rope.scaling_factor = 2.0;
    auto linear_model = std::make_unique<LoadedModel>(
            std::move(linear_config), (*loaded)->GetResolvedWeights());

    ModelLoweringOptions o0;
    o0.optimization.opt_level = 0;
    const auto linear = ModelCompiler::BuildLoweredModel(std::move(linear_model), o0);
    ASSERT_TRUE(linear.ok()) << linear.status().ToString();

    HfModelConfig unsupported_config = (*loaded)->GetConfig();
    unsupported_config.rope.scaling_type = HfRopeScalingType::kDynamicNtk;
    unsupported_config.rope.scaling_factor = 2.0;
    auto unsupported_model = std::make_unique<LoadedModel>(
            std::move(unsupported_config), (*loaded)->GetResolvedWeights());

    const auto unsupported = ModelCompiler::BuildLoweredModel(std::move(unsupported_model), o0);
    ASSERT_FALSE(unsupported.ok());
    EXPECT_EQ(unsupported.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(unsupported.status().message().find("Model graph construction failed"),
              std::string::npos);
    EXPECT_NE(unsupported.status().message().find("not representable"), std::string::npos);
}

TEST(ModelCompiler, RejectsNullLoadedModel) {
    const auto lowered = ModelCompiler::BuildLoweredModel(nullptr);

    ASSERT_FALSE(lowered.ok());
    EXPECT_EQ(lowered.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
