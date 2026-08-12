#ifndef AETHERMIND_TESTS_UNIT_GRAPH_OPTIMIZATION_TEST_OPTIMIZATION_HELPERS_H_
#define AETHERMIND_TESTS_UNIT_GRAPH_OPTIMIZATION_TEST_OPTIMIZATION_HELPERS_H_

#include "../test_graph_helpers.h"
#include "aethermind/graph/graph.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/operators/op_params.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aethermind::test_utils {

// Builds a two-embedding graph sharing one weight table:
//   v0=tokens_a, v1=tokens_b, v2=weight
//   n0=Embedding(v0,v2) -> v3=hidden_a (graph output)
//   n1=Embedding(v1,v2) -> v4=hidden_b (unreferenced)
// Shared by GraphRewriteSession and GraphPassManager tests; the value/node
// index layout above is asserted by both suites.
inline ModelGraph BuildTwoEmbeddingGraph() {
    ModelGraph graph;
    const GraphValueId tokens_a = graph.AddInput(
            Spec(DataType::Int(32), {1}), "tokens_a");
    const GraphValueId tokens_b = graph.AddInput(
            Spec(DataType::Int(32), {1}), "tokens_b");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding),
            "embed.weight");
    auto embed_a_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_a, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden_a"}},
            EmbeddingParams{});
    AM_CHECK(embed_a_or.ok(), "BuildTwoEmbeddingGraph embed_a AddNode failed");
    const AddedNode& embed_a = *embed_a_or;
    auto embed_b_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_b, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden_b"}},
            EmbeddingParams{});
    AM_CHECK(embed_b_or.ok(), "BuildTwoEmbeddingGraph embed_b AddNode failed");
    (void) *embed_b_or;
    graph.MarkOutput(embed_a.outputs[0]);
    return graph;
}

// Appends an Embedding node (Int32[2] tokens -> Float32[16,4] weight lookup)
// and returns its single activation output. The node itself is named
// `debug_name` to satisfy DeadCodeEliminationPass assertions
// (GetNodes()[0].name == debug_name); the produced value is also named
// `debug_name` for SiluMulFusionPass assertions (GetValue(inputs[i]).name).
inline GraphValueId AddActivation(ModelGraph& graph, const char* debug_name) {
    const GraphValueId tokens = graph.AddInput(
            Spec(DataType::Int(32), {2}), std::string(debug_name) + ".tokens");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding),
            std::string(debug_name) + ".weight");
    auto node_or = graph.AddNode(OpType::kEmbedding,
                                 std::nullopt,
                                 {tokens, weight},
                                 {NodeOutputDesc{.payload = ActivationValue{},
                                                 .name = debug_name}},
                                 EmbeddingParams{},
                                 {},
                                 debug_name);
    AM_CHECK(node_or.ok(), "AddActivation AddNode failed");
    const AddedNode& node = *node_or;
    return node.outputs[0];
}

// Builds a fully-static float constant value carrying the given bytes.
inline std::shared_ptr<const std::vector<std::byte>> InlineFloats(std::vector<float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_shared<const std::vector<std::byte>>(std::move(bytes));
}

template<typename T>
std::shared_ptr<const std::vector<std::byte>> InlineValues(std::vector<T> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_shared<const std::vector<std::byte>>(std::move(bytes));
}

// Adds a Float32 constant value with the given shape and inline data.
inline GraphValueId AddFloatConstant(ModelGraph& graph,
                                     std::vector<float> values,
                                     std::vector<int64_t> shape,
                                     const std::string& name) {
    return graph.AddConstant(
            Spec(DataType::Float32(), std::move(shape)),
            ConstantBinding{.inline_data = InlineFloats(std::move(values)), .name = name},
            name);
}

// Reads back the Float32 inline data of a constant GraphValue.
inline std::vector<float> ReadFloatConstant(const GraphValue& value) {
    const auto* constant = std::get_if<ConstantValue>(&value.payload);
    AM_CHECK(constant != nullptr, "expected constant value");
    AM_CHECK(constant->binding.inline_data != nullptr, "expected inline data");
    std::vector<float> result(constant->binding.inline_data->size() / sizeof(float));
    std::memcpy(result.data(), constant->binding.inline_data->data(),
                constant->binding.inline_data->size());
    return result;
}

template<typename T>
std::vector<T> ReadTypedConstant(const GraphValue& value) {
    const auto* constant = std::get_if<ConstantValue>(&value.payload);
    AM_CHECK(constant != nullptr, "expected constant value");
    AM_CHECK(constant->binding.inline_data != nullptr, "expected inline data");
    std::vector<T> result(constant->binding.inline_data->size() / sizeof(T));
    std::memcpy(result.data(), constant->binding.inline_data->data(),
                constant->binding.inline_data->size());
    return result;
}

// Builds a RmsNorm skeleton with the given input/output TensorSpecs. Uses the
// test-only ModelGraph constructor to bypass AddNode validation, so callers
// can inject forged/invalid specs that AddNode would normally reject.
// RmsNorm input[0] = kActivation (accepts ConstantValue), input[1] = kWeight
// (requires kScale slot), output[0] = kActivation.
struct RmsNormGraphSpecs {
    TensorSpec activation_input;
    TensorSpec weight;
    TensorSpec output;
};

inline ModelGraph BuildRmsNormGraphWithSpecs(const RmsNormGraphSpecs& specs) {
    std::vector<GraphValue> values;
    values.push_back(GraphValue{
            .payload = ConstantValue{},
            .spec = specs.activation_input,
            .producer = std::nullopt,
            .name = "act_in",
    });
    values.push_back(GraphValue{
            .payload = WeightValue{.binding = MakeDirectWeightBinding(ParameterSlot::kScale)},
            .spec = specs.weight,
            .producer = std::nullopt,
            .name = "weight_in",
    });
    values.push_back(GraphValue{
            .payload = ActivationValue{},
            .spec = specs.output,
            .producer = GraphNodeId{.index = 0},
            .name = "act_out",
    });

    GraphNode node;
    node.op_type = OpType::kRmsNorm;
    node.inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 1}};
    node.outputs = {GraphValueId{.index = 2}};
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};

    return ModelGraph({node}, values);
}

// Structurally-valid but semantically-invalid graph: a RmsNorm node whose
// activation input carries Int32 (InferRmsNorm only accepts floating-point).
inline ModelGraph BuildGraphWithWrongInputDtype() {
    return BuildRmsNormGraphWithSpecs(
            {.activation_input = Spec(DataType::Int(32), {4, 8}),
             .weight = Spec(DataType::Float32(), {8}),
             .output = Spec(DataType::Float32(), {4, 8})});
}

// Graph with a forged output spec: InferRmsNorm derives a Float32 [4, 8]
// output, but the stored GraphValue carries Float16 to simulate stale/forged
// metadata. ValidateAndTopologicalOrder must catch this.
inline ModelGraph BuildGraphWithForgedOutputSpec() {
    return BuildRmsNormGraphWithSpecs(
            {.activation_input = Spec(DataType::Float32(), {4, 8}),
             .weight = Spec(DataType::Float32(), {8}),
             .output = Spec(DataType::Float(16), {4, 8})});
}

}// namespace aethermind::test_utils

#endif// AETHERMIND_TESTS_UNIT_GRAPH_OPTIMIZATION_TEST_OPTIMIZATION_HELPERS_H_
