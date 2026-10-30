#include "aethermind/compiler/model_compiler.h"
#include "aethermind/operators/operator_schema.h"

#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "model/test_model_graph_helpers.h"

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
    return std::ranges::any_of(graph.values(), [predicate](const LoweredValueDesc& value) {
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

TEST(ModelCompiler, LoadAndCompileTinyLlamaAtO2) {
    const auto lowered = ModelCompiler::LoadAndCompile(
            TestModelDir());

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_NE(lowered->loaded_model, nullptr);
    EXPECT_EQ(lowered->loaded_model->GetConfig().num_hidden_layers, 2);
    EXPECT_FALSE(lowered->graph.steps().empty());
    EXPECT_EQ(lowered->graph.steps().front().spec.op_type, OpType::kEmbedding);
    EXPECT_EQ(lowered->graph.steps().back().spec.op_type, OpType::kArgmax);
    EXPECT_TRUE(HasWeightRecipe(lowered->graph, &IsQkvRecipe));
    EXPECT_TRUE(HasWeightRecipe(lowered->graph, &IsGateUpRecipe));

    for (const LoweredStep& step: lowered->graph.steps()) {
        const LoweredStepBinding& binding = step.binding;
        for (const GraphValueId value: binding.input_values) {
            EXPECT_LT(value.index, lowered->graph.values().size());
        }
        for (const GraphValueId value: binding.output_values) {
            EXPECT_LT(value.index, lowered->graph.values().size());
        }
    }
}

TEST(ModelCompiler, SupportsO0AndO2) {
    ModelCompileOptions o0;
    o0.optimization.opt_level = 0;

    const auto unoptimized = ModelCompiler::LoadAndCompile(
            TestModelDir(), o0);
    ASSERT_TRUE(unoptimized.ok()) << unoptimized.status().ToString();
    EXPECT_FALSE(HasWeightRecipe(unoptimized->graph, &IsQkvRecipe));
    EXPECT_FALSE(HasWeightRecipe(unoptimized->graph, &IsGateUpRecipe));
    EXPECT_EQ(unoptimized->graph.steps().size(), 34U);

    const auto optimized = ModelCompiler::LoadAndCompile(
            TestModelDir());
    ASSERT_TRUE(optimized.ok()) << optimized.status().ToString();
    EXPECT_LT(optimized->graph.steps().size(), unoptimized->graph.steps().size());
}

TEST(ModelCompiler, PropagatesLoweringTargetConfiguration) {
    ModelCompileOptions options;
    options.lowering.selector.device_type = DeviceType::kCUDA;
    options.lowering.selector.weight_format = WeightFormat::kPacked;
    options.lowering.selector.phase = ExecPhase::kPrefill;
    options.lowering.enable_packed_weights = true;

    const auto lowered = ModelCompiler::LoadAndCompile(
            TestModelDir(), options);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    for (const LoweredStep& step: lowered->graph.steps()) {
        EXPECT_EQ(step.spec.selector.device_type, DeviceType::kCUDA);
        EXPECT_EQ(step.spec.selector.phase, ExecPhase::kPrefill);
        // Packed is scoped to weight-consuming steps; every other step stays
        // kPlain even though the config selector requested kPacked.
        const auto schema = GetOperatorSchema(step.spec.op_type);
        ASSERT_TRUE(schema.ok()) << schema.status().ToString();
        const bool consumes_weight = std::ranges::any_of(
                schema->input_ports,
                [](const OperatorInputPort& port) {
                    return port.kind == OperatorPortKind::kWeight;
                });
        EXPECT_EQ(step.spec.selector.weight_format,
                  consumes_weight ? WeightFormat::kPacked
                                  : WeightFormat::kPlain);
    }
}

TEST(ModelCompiler, SupportsLinearAndDynamicNtkRopeScaling) {
    const auto loaded = ModelLoader::Load(TestModelDir());
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();

    HfModelConfig linear_config = (*loaded)->GetConfig();
    linear_config.rope.algorithm = HfRoPEAlgorithm::kLinear;
    linear_config.rope.factor = 2.0;
    auto linear_model = std::make_unique<LoadedModel>(
            std::move(linear_config), (*loaded)->GetResolvedWeights());

    ModelCompileOptions o0;
    o0.optimization.opt_level = 0;
    const auto linear = ModelCompiler::Compile(std::move(linear_model), o0);
    ASSERT_TRUE(linear.ok()) << linear.status().ToString();

    HfModelConfig unsupported_config = (*loaded)->GetConfig();
    unsupported_config.rope.algorithm = HfRoPEAlgorithm::kDynamicNtk;
    unsupported_config.rope.factor = 2.0;
    auto unsupported_model = std::make_unique<LoadedModel>(
            std::move(unsupported_config), (*loaded)->GetResolvedWeights());

    const auto dynamic = ModelCompiler::Compile(std::move(unsupported_model), o0);
    ASSERT_TRUE(dynamic.ok()) << dynamic.status().ToString();
}

TEST(ModelCompiler, RejectsNullLoadedModel) {
    const auto lowered = ModelCompiler::Compile(nullptr);

    ASSERT_FALSE(lowered.ok());
    EXPECT_EQ(lowered.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelCompilerGraphBuildOptions, PropagatesExplicitFp32DTypesAndPreservesSourceHint) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.weight_dtype_hint = DataType::BFloat(16);
    config.weight_dtype_hint_name = "bfloat16";
    auto model = std::make_unique<LoadedModel>(config, MakeWeights(config));
    ModelCompileOptions options;
    options.optimization.opt_level = 1;
    options.graph_build = {.activation_dtype = DataType::Float32(),
                           .kv_dtype = DataType::Float32()};

    const auto artifact = ModelCompiler::Compile(std::move(model), options);

    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
    ASSERT_NE(artifact->loaded_model, nullptr);
    EXPECT_EQ(artifact->loaded_model->GetConfig().weight_dtype_hint, DataType::BFloat(16));
    EXPECT_EQ(artifact->loaded_model->GetConfig().weight_dtype_hint_name, "bfloat16");
    for (const auto& value: artifact->graph.values()) {
        if (value.spec.dtype.IsFloat()) {
            EXPECT_EQ(value.spec.dtype, DataType::Float32()) << value.name;
        }
    }
    EXPECT_TRUE(ValidateLoweredGraph(artifact->graph).ok());
}

TEST(ModelCompilerGraphBuildOptions, ReportsGraphConstructionDTypeError) {
    const HfModelConfig config = MakeLlamaConfig(1);
    auto model = std::make_unique<LoadedModel>(config, MakeWeights(config));
    ModelCompileOptions options;
    options.graph_build.activation_dtype = DataType::Int(32);

    const auto artifact = ModelCompiler::Compile(std::move(model), options);

    ASSERT_FALSE(artifact.ok());
    EXPECT_EQ(artifact.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(artifact.status().message().find("Model graph construction failed"),
              std::string::npos);
    EXPECT_NE(artifact.status().message().find("activation_dtype"), std::string::npos);
}

} // namespace
