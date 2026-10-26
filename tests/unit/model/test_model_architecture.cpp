#include "aethermind/model/build_model_graph.h"
#include "aethermind/model/llama_dense_graph_builder.h"
#include "aethermind/model/model_architecture.h"
#include "test_model_graph_helpers.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

namespace {

using namespace aethermind;

TEST(ModelArchitecture, ParsesLlamaModelTypeAsLlamaDense) {
    const HfModelConfig config = MakeLlamaConfig(1);

    EXPECT_EQ(ParseModelArchitecture(config), ModelArchitecture::kLlamaDense);
}

TEST(ModelArchitecture, MapsUnknownModelTypesToUnknown) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.model_type = "gpt_neox";
    config.architectures = {"GPTNeoXForCausalLM"};

    EXPECT_EQ(ParseModelArchitecture(config), ModelArchitecture::kUnknown);
}

TEST(ModelArchitecture, MapsEmptyModelTypeToUnknown) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.model_type.clear();

    EXPECT_EQ(ParseModelArchitecture(config), ModelArchitecture::kUnknown);
}

TEST(ModelArchitecture, ToStringCoversAllEnumerators) {
    EXPECT_EQ(ToString(ModelArchitecture::kLlamaDense), "llama_dense");
    EXPECT_EQ(ToString(ModelArchitecture::kQwenDense), "qwen_dense");
    EXPECT_EQ(ToString(ModelArchitecture::kQwenMoe), "qwen_moe");
    EXPECT_EQ(ToString(ModelArchitecture::kMixtralMoe), "mixtral_moe");
    EXPECT_EQ(ToString(ModelArchitecture::kUnknown), "unknown");
}

TEST(BuildModelGraph, DispatchesLlamaDenseFamily) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = BuildModelGraph(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    EXPECT_TRUE(graph->Validate().ok());
}

TEST(BuildModelGraph, RejectsUnsupportedFamilyBeforeWeightInspection) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.model_type = "qwen2";
    const ResolvedModelWeights empty_weights{};

    const StatusOr<ModelGraph> graph = BuildModelGraph(config, empty_weights);

    EXPECT_FALSE(graph.ok());
    EXPECT_EQ(graph.status().code(), StatusCode::kInvalidArgument);
}

class GraphSlidingWindow : public ::testing::TestWithParam<int64_t> {};

TEST_P(GraphSlidingWindow, DispatchRejectsBeforeWeightInspection) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.sliding_window = GetParam();

    const auto graph = BuildModelGraph(config, ResolvedModelWeights{});

    ASSERT_FALSE(graph.ok());
    EXPECT_EQ(graph.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(graph.status().message().find("sliding_window"), std::string::npos);
}

TEST_P(GraphSlidingWindow, LlamaBuilderRejectsBeforeWeightInspection) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.sliding_window = GetParam();

    const auto graph = BuildLlamaDense(config, ResolvedModelWeights{});

    ASSERT_FALSE(graph.ok());
    EXPECT_EQ(graph.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(graph.status().message().find("sliding_window"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(WindowValues, GraphSlidingWindow,
                         ::testing::Values(int64_t{0}, int64_t{-1}, int64_t{128}));

} // namespace
