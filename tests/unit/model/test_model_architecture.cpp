#include "aethermind/model/build_model_graph.h"
#include "aethermind/model/model_architecture.h"
#include "test_model_graph_helpers.h"

#include <gtest/gtest.h>

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

} // namespace