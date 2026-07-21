#include "../test_graph_helpers.h"
#include "aethermind/model/graph/optimization/constant_folding_pass.h"

#include "aethermind/model/graph/graph_op_builder.h"
#include "aethermind/model/graph/optimization/dead_code_elimination_pass.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace aethermind {
namespace {

StatusOr<ModelGraph> RunConstantFolding(const ModelGraph& graph, PassContext ctx = {});

std::shared_ptr<const std::vector<std::byte>> InlineFloats(std::vector<float> values) {
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

template<typename T>
GraphValueId AddTypedConstant(ModelGraph& graph,
                              DataType dtype,
                              std::vector<T> values,
                              std::vector<int64_t> shape,
                              const std::string& name) {
    return graph.AddConstant(
            Spec(dtype, std::move(shape)),
            ConstantBinding{.inline_data = InlineValues(std::move(values)), .name = name},
            name);
}

GraphValueId AddFloatConstant(ModelGraph& graph,
                              std::vector<float> values,
                              std::vector<int64_t> shape,
                              const std::string& name) {
    return graph.AddConstant(
            Spec(DataType::Float32(), std::move(shape)),
            ConstantBinding{.inline_data = InlineFloats(std::move(values)), .name = name},
            name);
}

GraphValueId AddFloatAddWithOutputShape(ModelGraph& graph,
                                        GraphValueId lhs,
                                        GraphValueId rhs,
                                        std::vector<int64_t> output_shape,
                                        std::string name) {
    auto node_or = graph.AddNode(
            OpType::kAdd,
            std::nullopt,
            {lhs, rhs},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            AddParams{},
            {},
            std::move(name));
    AM_CHECK(node_or.ok(), "AddFloatAddWithOutputShape AddNode failed");
    const auto& node = *node_or;
    AM_CHECK(node.outputs.size() == 1U, "expected test Add node to have one output");
    return node.outputs.front();
}

std::vector<float> ReadFloatConstant(const GraphValue& value) {
    const auto* constant = std::get_if<ConstantValue>(&value.payload);
    AM_CHECK(constant != nullptr, "expected constant value");
    AM_CHECK(constant->binding.inline_data != nullptr, "expected inline data");
    std::vector<float> result(constant->binding.inline_data->size() / sizeof(float));
    std::memcpy(result.data(), constant->binding.inline_data->data(), constant->binding.inline_data->size());
    return result;
}

template<typename T>
std::vector<T> ReadTypedConstant(const GraphValue& value) {
    const auto* constant = std::get_if<ConstantValue>(&value.payload);
    AM_CHECK(constant != nullptr, "expected constant value");
    AM_CHECK(constant->binding.inline_data != nullptr, "expected inline data");
    std::vector<T> result(constant->binding.inline_data->size() / sizeof(T));
    std::memcpy(result.data(), constant->binding.inline_data->data(), constant->binding.inline_data->size());
    return result;
}

std::vector<BFloat16> BFloat16Values(std::vector<uint16_t> bits) {
    std::vector<BFloat16> values;
    values.reserve(bits.size());
    for (const uint16_t value: bits) {
        values.emplace_back(value, BFloat16::from_bits());
    }
    return values;
}

std::vector<uint16_t> BFloat16Bits(const std::vector<BFloat16>& values) {
    std::vector<uint16_t> bits;
    bits.reserve(values.size());
    for (const BFloat16 value: values) {
        bits.push_back(value.x);
    }
    return bits;
}

template<typename T>
struct FoldedAddCase {
    DataType dtype;
    std::array<std::vector<T>, 3> values;
};

template<typename T>
void ExpectFoldedAdd(FoldedAddCase<T> test_case) {
    ModelGraph graph;
    constexpr size_t kLhs = 0;
    constexpr size_t kRhs = 1;
    constexpr size_t kExpected = 2;
    const std::vector<int64_t> shape{static_cast<int64_t>(test_case.values[kExpected].size())};
    const GraphValueId lhs = AddTypedConstant(graph, test_case.dtype, std::move(test_case.values[kLhs]), shape, "lhs");
    const GraphValueId rhs = AddTypedConstant(graph, test_case.dtype, std::move(test_case.values[kRhs]), shape, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<T> values = ReadTypedConstant<T>(output);
    ASSERT_EQ(values.size(), test_case.values[kExpected].size());
    for (size_t i = 0; i < test_case.values[kExpected].size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_DOUBLE_EQ(static_cast<double>(values[i]), static_cast<double>(test_case.values[kExpected][i]));
        } else {
            EXPECT_EQ(values[i], test_case.values[kExpected][i]);
        }
    }
}

StatusOr<ModelGraph> RunConstantFolding(const ModelGraph& graph, PassContext ctx) {
    GraphPassManager pipeline(ctx);
    pipeline.Add(std::make_unique<ConstantFoldingPass>());
    return pipeline.Run(graph);
}

StatusOr<ModelGraph> RunConstantFoldingThenDce(const ModelGraph& graph) {
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<ConstantFoldingPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    return pipeline.Run(graph);
}

TEST(ConstantFoldingPass, FoldsAddOfTwoConstants) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.0F, 2.0F}, {2}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {3.0F, 4.0F}, {2}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 2U);
    EXPECT_FLOAT_EQ(values[0], 4.0F);
    EXPECT_FLOAT_EQ(values[1], 6.0F);
}

TEST(ConstantFoldingPass, FoldsAddBroadcastRowVectorConstants) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph,
                                              {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                              {2, 3},
                                              "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {10.0F, 20.0F, 30.0F}, {3}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 6U);
    EXPECT_FLOAT_EQ(values[0], 11.0F);
    EXPECT_FLOAT_EQ(values[1], 22.0F);
    EXPECT_FLOAT_EQ(values[2], 33.0F);
    EXPECT_FLOAT_EQ(values[3], 14.0F);
    EXPECT_FLOAT_EQ(values[4], 25.0F);
    EXPECT_FLOAT_EQ(values[5], 36.0F);
}

TEST(ConstantFoldingPass, FoldsAddBroadcastReversedInputConstants) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {10.0F, 20.0F, 30.0F}, {3}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph,
                                              {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                              {2, 3},
                                              "rhs");
    const GraphValueId sum = AddFloatAddWithOutputShape(graph, lhs, rhs, {2, 3}, "sum");
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 6U);
    EXPECT_FLOAT_EQ(values[0], 11.0F);
    EXPECT_FLOAT_EQ(values[1], 22.0F);
    EXPECT_FLOAT_EQ(values[2], 33.0F);
    EXPECT_FLOAT_EQ(values[3], 14.0F);
    EXPECT_FLOAT_EQ(values[4], 25.0F);
    EXPECT_FLOAT_EQ(values[5], 36.0F);
}

TEST(ConstantFoldingPass, FoldsAddFloat64Constants) {
    ExpectFoldedAdd<double>({.dtype = DataType::Double(), .values = {{{1.25, 2.5}, {3.5, 4.75}, {4.75, 7.25}}}});
}

TEST(ConstantFoldingPass, FoldsAddInt32Constants) {
    ExpectFoldedAdd<int32_t>({.dtype = DataType::Int(32), .values = {{{-3, 2, 7}, {5, -4, 8}, {2, -2, 15}}}});
}

TEST(ConstantFoldingPass, FoldsAddInt64Constants) {
    ExpectFoldedAdd<int64_t>({.dtype = DataType::Int(64), .values = {{{-3, 2, 7}, {5, -4, 8}, {2, -2, 15}}}});
}

TEST(ConstantFoldingPass, FoldsAddBFloat16Constants) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant(graph,
                                              DataType::BFloat(16),
                                              BFloat16Values({0x3F80U, 0x4000U}),
                                              {2},
                                              "lhs");
    const GraphValueId rhs = AddTypedConstant(graph,
                                              DataType::BFloat(16),
                                              BFloat16Values({0x4040U, 0x4080U}),
                                              {2},
                                              "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_EQ(BFloat16Bits(ReadTypedConstant<BFloat16>(output)),
              (std::vector<uint16_t>{0x4080U, 0x40C0U}));
}

TEST(ConstantFoldingPass, SkipsAddInt32Overflow) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<int32_t>(
            graph, DataType::Int(32), {std::numeric_limits<int32_t>::max()}, {1}, "lhs");
    const GraphValueId rhs = AddTypedConstant<int32_t>(graph, DataType::Int(32), {1}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, SkipsAddInt64Overflow) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<int64_t>(
            graph, DataType::Int(64), {std::numeric_limits<int64_t>::max()}, {1}, "lhs");
    const GraphValueId rhs = AddTypedConstant<int64_t>(graph, DataType::Int(64), {1}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, SkipsWhenDisabled) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.0F}, {1}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {2.0F}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);
    PassContext ctx;
    ctx.enable_constant_folding = false;

    const StatusOr<ModelGraph> result = RunConstantFolding(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, SkipsMissingInlineData) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(
            Spec(DataType::Float32(), {1}), ConstantBinding{.name = "lhs"}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {2.0F}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, SkipsInlineDataByteMismatch) {
    ModelGraph graph;
    const auto bad_bytes = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x01}});
    const GraphValueId lhs = graph.AddConstant(
            Spec(DataType::Float32(), {1}),
            ConstantBinding{.inline_data = bad_bytes, .name = "lhs"},
            "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {2.0F}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, SkipsWeightInputInSession) {
    // Construct graph through escape hatch: AddNode validates inputs,
    // so a non-foldable Add with a non-activation input must be built raw.
    auto inline_data = InlineFloats({1.0F});
    ModelGraph graph(
            HfModelConfig{},
            {GraphNode{.op_type = OpType::kAdd,
                       .inputs = {GraphValueId{0}, GraphValueId{1}},
                       .outputs = {GraphValueId{2}},
                       .op_params = AddParams{},
                       .name = "sum"}},
            {GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.inline_data = std::move(inline_data), .name = "lhs"}},
                        .spec = Spec(DataType::Float32(), {1}),
                        .name = "lhs"},
             GraphValue{.payload = ModelInputValue{},
                        .spec = Spec(DataType::Float32(), {1}),
                        .name = "input"},
             GraphValue{.payload = ActivationValue{},
                        .spec = Spec(DataType::Float32(), {1}),
                        .producer = GraphNodeId{0},
                        .name = "sum"}});
    const GraphValueId sum{2};
    GraphRewriteSession session(graph);
    ConstantFoldingPass pass;

    const Status status = pass.Run(session, PassContext{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(session.GetResolvedValue(sum), sum);
}

TEST(ConstantFoldingPass, DceRemovesFoldedAdd) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.0F}, {1}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {2.0F}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

// ── Chain folding: A = B + C, D = A + E -> D should fold to B + C + E ──

TEST(ConstantFoldingPass, FoldsChainedAddOfConstants) {
    ModelGraph graph;
    const GraphValueId b = AddFloatConstant(graph, {1.0F, 2.0F}, {2}, "b");
    const GraphValueId c = AddFloatConstant(graph, {3.0F, 4.0F}, {2}, "c");
    const GraphValueId e = AddFloatConstant(graph, {5.0F, 6.0F}, {2}, "e");
    auto a_or = AddElementwiseAdd(graph, 0U, b, c, "a");
    ASSERT_TRUE(a_or.ok()) << a_or.status().ToString();
    const GraphValueId a = *a_or;
    auto d_or = AddElementwiseAdd(graph, 0U, a, e, "d");
    ASSERT_TRUE(d_or.ok()) << d_or.status().ToString();
    const GraphValueId d = *d_or;
    graph.MarkOutput(d);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    // Both Add nodes should be eliminated after folding + DCE.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
    const std::vector<float> values = ReadFloatConstant(result->GetValue(result->GetOutputs()[0].value));
    ASSERT_EQ(values.size(), 2U);
    // Expected: (1+3+5, 2+4+6) = (9, 12)
    EXPECT_FLOAT_EQ(values[0], 9.0F);
    EXPECT_FLOAT_EQ(values[1], 12.0F);
}

// ── Multi-node graph: foldable Add + non-foldable Add (non-constant input).
// The non-foldable node must not abort the pass; the foldable one must be folded.
// Runs directly on session (like SkipsWeightInputInSession) to avoid schema
// validation of weight inputs to Add.

TEST(ConstantFoldingPass, SkipsNonFoldableNodeAndContinuesToFoldable) {
    // Build graph via escape hatch: the non-foldable Add has a
    // model-input operand which AddNode would reject in the
    // validated path, so the full graph must be constructed raw.
    auto const_a_data = InlineFloats({1.0F, 2.0F});
    auto const_b_data = InlineFloats({3.0F, 4.0F});
    auto const_c_data = InlineFloats({5.0F, 6.0F});
    ModelGraph graph(
            HfModelConfig{},
            {GraphNode{.op_type = OpType::kAdd,
                       .inputs = {GraphValueId{0}, GraphValueId{1}},
                       .outputs = {GraphValueId{2}},
                       .op_params = AddParams{},
                       .name = "foldable_sum"},
             GraphNode{.op_type = OpType::kAdd,
                       .inputs = {GraphValueId{3}, GraphValueId{4}},
                       .outputs = {GraphValueId{5}},
                       .op_params = AddParams{},
                       .name = "non_foldable_sum"}},
            {GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.inline_data = std::move(const_a_data), .name = "const_a"}},
                        .spec = Spec(DataType::Float32(), {2}),
                        .name = "const_a"},
             GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.inline_data = std::move(const_b_data), .name = "const_b"}},
                        .spec = Spec(DataType::Float32(), {2}),
                        .name = "const_b"},
             GraphValue{.payload = ActivationValue{},
                        .spec = Spec(DataType::Float32(), {2}),
                        .producer = GraphNodeId{0},
                        .name = "foldable_sum"},
             GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.inline_data = std::move(const_c_data), .name = "const_c"}},
                        .spec = Spec(DataType::Float32(), {2}),
                        .name = "const_c"},
             GraphValue{.payload = ModelInputValue{},
                        .spec = Spec(DataType::Float32(), {2}),
                        .name = "input"},
             GraphValue{.payload = ActivationValue{},
                        .spec = Spec(DataType::Float32(), {2}),
                        .producer = GraphNodeId{1},
                        .name = "non_foldable_sum"}});

    graph.MarkOutput(GraphValueId{2});
    graph.MarkOutput(GraphValueId{5});

    GraphRewriteSession session(graph);
    ConstantFoldingPass pass;
    const Status status = pass.Run(session, PassContext{});
    ASSERT_TRUE(status.ok()) << status.ToString();

    // Foldable output should be redirected to a folded constant.
    EXPECT_NE(session.GetResolvedValue(GraphValueId{2}), GraphValueId{2});

    // Non-foldable output should remain unchanged.
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{5}), GraphValueId{5});
}

// ── Empty tensor (numel == 0) folding ──

TEST(ConstantFoldingPass, FoldsAddOfEmptyTensors) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {}, {0}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {}, {0}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    ASSERT_TRUE(constant->binding.inline_data != nullptr);
    EXPECT_EQ(constant->binding.inline_data->size(), 0U);
}

// ── Zero-element tensor with extreme shape causing stride overflow ──
// Regression: shape {0, 1LL<<62, 1LL<<62} has numel==0 (so CountBytes==0 and
// all budget checks pass) but its contiguous stride computation overflows
// int64_t. The pass must detect the overflow and skip folding without UB.

TEST(ConstantFoldingPass, SkipsAddWhenContiguousStridesOverflow) {
    constexpr int64_t kHuge = 1LL << 62;
    const std::vector<int64_t> shape{0, kHuge, kHuge};

    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {}, shape, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {}, shape, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    // The Add node must NOT be folded because stride computation overflows.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(
            result->GetValue(result->GetOutputs()[0].value).payload));
}

// ── kNotFound semantics: the pass must skip unknown op types without error ──

TEST(ConstantFoldingPass, SkipsUnknownOpTypeNode) {
    // Build via escape hatch: kUnknown has no registered schema,
    // so AddNode would reject it. The pass must still tolerate it.
    ModelGraph graph(
            HfModelConfig{},
            {GraphNode{.op_type = OpType::kUnknown,
                       .inputs = {},
                       .outputs = {GraphValueId{0}},
                       .op_params = std::monostate{},
                       .name = "unknown_op"}},
            {GraphValue{.payload = ActivationValue{},
                        .spec = Spec(DataType::Float32(), {1}),
                        .producer = GraphNodeId{0},
                        .name = "unknown_op_out"}});

    GraphRewriteSession session(graph);
    ConstantFoldingPass pass;

    const Status status = pass.Run(session, PassContext{});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{0}), GraphValueId{0});
}

// ── Single-element tensor (shape {1}) folding ──

TEST(ConstantFoldingPass, FoldsAddOfSingleElementConstants) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.5F}, {1}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {2.5F}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 1U);
    EXPECT_FLOAT_EQ(values[0], 4.0F);
}

// ── ElementwiseMul constant folding tests ──

TEST(ConstantFoldingPass, FoldsMulOfTwoConstants) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {2.0F, 3.0F}, {2}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {4.0F, 5.0F}, {2}, "rhs");
    auto product_or = AddElementwiseMul(graph, 0U, lhs, rhs, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 2U);
    EXPECT_FLOAT_EQ(values[0], 8.0F);
    EXPECT_FLOAT_EQ(values[1], 15.0F);
}

TEST(ConstantFoldingPass, FoldsMulBFloat16Constants) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant(graph,
                                              DataType::BFloat(16),
                                              BFloat16Values({0x4000U, 0x4040U}),
                                              {2},
                                              "lhs");
    const GraphValueId rhs = AddTypedConstant(graph,
                                              DataType::BFloat(16),
                                              BFloat16Values({0x4000U, 0x4000U}),
                                              {2},
                                              "rhs");
    auto product_or = AddElementwiseMul(graph, 0U, lhs, rhs, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_EQ(BFloat16Bits(ReadTypedConstant<BFloat16>(output)),
              (std::vector<uint16_t>{0x4080U, 0x40C0U}));
}

TEST(ConstantFoldingPass, SkipsAddInt32OverflowWithLargerRhs) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<int32_t>(
            graph, DataType::Int(32), {std::numeric_limits<int32_t>::max()}, {1}, "lhs");
    const GraphValueId rhs = AddTypedConstant<int32_t>(graph, DataType::Int(32), {2}, {1}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, FoldsAddThenMulChainedConstants) {
    ModelGraph graph;
    const GraphValueId a = AddFloatConstant(graph, {1.0F, 2.0F}, {2}, "a");
    const GraphValueId b = AddFloatConstant(graph, {3.0F, 4.0F}, {2}, "b");
    const GraphValueId c = AddFloatConstant(graph, {2.0F, 3.0F}, {2}, "c");
    auto sum_or = AddElementwiseAdd(graph, 0U, a, b, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    auto product_or = AddElementwiseMul(graph, 0U, sum, c, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
    const std::vector<float> values = ReadFloatConstant(result->GetValue(result->GetOutputs()[0].value));
    ASSERT_EQ(values.size(), 2U);
    // Expected: (1+3)*2=8, (2+4)*3=18
    EXPECT_FLOAT_EQ(values[0], 8.0F);
    EXPECT_FLOAT_EQ(values[1], 18.0F);
}

TEST(ConstantFoldingPass, DceRemovesFoldedMul) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {2.0F}, {1}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {3.0F}, {1}, "rhs");
    auto product_or = AddElementwiseMul(graph, 0U, lhs, rhs, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, FoldsSiluOfConstant) {
    ModelGraph graph;
    const GraphValueId input = AddFloatConstant(graph, {0.0F, 1.0F, 2.0F}, {3}, "input");
    auto act_or = AddSilu(graph, 0U, input, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 3U);
    EXPECT_NEAR(values[0], 0.0F, 1e-5F);
    EXPECT_NEAR(values[1], 0.7310586F, 1e-5F);
    EXPECT_NEAR(values[2], 1.7615942F, 1e-5F);
}

TEST(ConstantFoldingPass, FoldsSiluBFloat16Constants) {
    ModelGraph graph;
    const GraphValueId input = AddTypedConstant(graph,
                                                DataType::BFloat(16),
                                                BFloat16Values({0x3F80U, 0x4000U, 0x4040U}),
                                                {3},
                                                "input");
    auto act_or = AddSilu(graph, 0U, input, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<BFloat16> values = ReadTypedConstant<BFloat16>(output);
    ASSERT_EQ(values.size(), 3U);
    const std::vector<float> input_f = {1.0F, 2.0F, 3.0F};
    std::vector<BFloat16> expected;
    expected.reserve(3);
    for (float x: input_f) {
        float r;
        if (x >= 0.0F) {
            r = x / (1.0F + std::exp(-x));
        } else {
            r = x * std::exp(x) / (1.0F + std::exp(x));
        }
        expected.emplace_back(r);
    }
    EXPECT_EQ(BFloat16Bits(values), BFloat16Bits(expected));
}

TEST(ConstantFoldingPass, FoldsAddThenSiluChainedConstants) {
    ModelGraph graph;
    const GraphValueId a = AddFloatConstant(graph, {1.0F, 2.0F}, {2}, "a");
    const GraphValueId b = AddFloatConstant(graph, {3.0F, 4.0F}, {2}, "b");
    auto sum_or = AddElementwiseAdd(graph, 0U, a, b, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    auto act_or = AddSilu(graph, 0U, sum, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
    const std::vector<float> values = ReadFloatConstant(result->GetValue(result->GetOutputs()[0].value));
    ASSERT_EQ(values.size(), 2U);
    EXPECT_NEAR(values[0], 3.928055F, 1e-5F);
    EXPECT_NEAR(values[1], 5.985164F, 1e-5F);
}

TEST(ConstantFoldingPass, DceRemovesFoldedSilu) {
    ModelGraph graph;
    const GraphValueId input = AddFloatConstant(graph, {1.0F}, {1}, "input");
    auto act_or = AddSilu(graph, 0U, input, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

// ── SiluMul constant folding tests ──

TEST(ConstantFoldingPass, FoldsSiluMulOfTwoConstants) {
    ModelGraph graph;
    const GraphValueId gate = AddFloatConstant(graph, {0.0F, 1.0F, 2.0F}, {3}, "gate");
    const GraphValueId up = AddFloatConstant(graph, {1.0F, 2.0F, 3.0F}, {3}, "up");
    auto act_or = AddSiluMul(graph, 0U, gate, up, "silu_mul");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 3U);
    // silu(0)*1 = 0, silu(1)*2 ≈ 1.4621172, silu(2)*3 ≈ 5.2847826
    EXPECT_NEAR(values[0], 0.0F, 1e-5F);
    EXPECT_NEAR(values[1], 1.4621172F, 1e-5F);
    EXPECT_NEAR(values[2], 5.2847826F, 1e-5F);
}

TEST(ConstantFoldingPass, FoldsSiluMulBFloat16Constants) {
    ModelGraph graph;
    const GraphValueId gate = AddTypedConstant(graph,
                                               DataType::BFloat(16),
                                               BFloat16Values({0x3F80U, 0x4000U, 0x4040U}),
                                               {3},
                                               "gate");
    const GraphValueId up = AddTypedConstant(graph,
                                             DataType::BFloat(16),
                                             BFloat16Values({0x4000U, 0x4040U, 0x4080U}),
                                             {3},
                                             "up");
    auto act_or = AddSiluMul(graph, 0U, gate, up, "silu_mul");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    const std::vector<BFloat16> values = ReadTypedConstant<BFloat16>(output);
    ASSERT_EQ(values.size(), 3U);

    const std::vector<float> gate_f = {1.0F, 2.0F, 3.0F};
    const std::vector<float> up_f = {2.0F, 3.0F, 4.0F};
    std::vector<BFloat16> expected;
    expected.reserve(3);
    for (size_t i = 0; i < gate_f.size(); ++i) {
        const float x = gate_f[i];
        float silu;
        if (x >= 0.0F) {
            silu = x / (1.0F + std::exp(-x));
        } else {
            silu = x * std::exp(x) / (1.0F + std::exp(x));
        }
        expected.emplace_back(silu * up_f[i]);
    }
    EXPECT_EQ(BFloat16Bits(values), BFloat16Bits(expected));
}

TEST(ConstantFoldingPass, RejectsSiluMulInt32Inputs) {
    // Build via raw constructor since SiluMul no longer accepts Int32
    // through AddNode. Constant folding must still handle such pre-existing
    // graphs gracefully.
    std::vector<GraphValue> values = {
            GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.inline_data = InlineValues<int32_t>({1, 2})}}, .spec = {DataType::Int(32), SymbolicShape({ShapeSymbol::CreateFromValue(2)})}, .name = "gate"},
            GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.inline_data = InlineValues<int32_t>({3, 4})}}, .spec = {DataType::Int(32), SymbolicShape({ShapeSymbol::CreateFromValue(2)})}, .name = "up"},
            GraphValue{.payload = ActivationValue{}, .spec = {DataType::Int(32), SymbolicShape({ShapeSymbol::CreateFromValue(2)})}, .producer = GraphNodeId{0}, .name = "silu_mul"},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kSiluMul, .inputs = {GraphValueId{0}, GraphValueId{1}}, .outputs = {GraphValueId{2}}, .op_params = SiluMulParams{}, .name = "silu_mul"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    graph.MarkOutput(GraphValueId{2});

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    // SiluMul only supports float dtypes. The semantic layer rejects
    // Int32 inputs during validation, so the folding pipeline cannot
    // produce a valid output graph for invalid inputs.
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ConstantFoldingPass, FoldsSiluMulThenAddChainedConstants) {
    ModelGraph graph;
    const GraphValueId gate = AddFloatConstant(graph, {1.0F, 2.0F}, {2}, "gate");
    const GraphValueId up = AddFloatConstant(graph, {3.0F, 4.0F}, {2}, "up");
    const GraphValueId bias = AddFloatConstant(graph, {0.5F, 0.5F}, {2}, "bias");
    auto silu_mul_or = AddSiluMul(graph, 0U, gate, up, "silu_mul");
    ASSERT_TRUE(silu_mul_or.ok()) << silu_mul_or.status().ToString();
    const GraphValueId silu_mul = *silu_mul_or;
    auto sum_or = AddElementwiseAdd(graph, 0U, silu_mul, bias, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
    const std::vector<float> values = ReadFloatConstant(result->GetValue(result->GetOutputs()[0].value));
    ASSERT_EQ(values.size(), 2U);
    // silu(1)*3 + 0.5 ≈ 2.6931758, silu(2)*4 + 0.5 ≈ 7.5463768
    EXPECT_NEAR(values[0], 2.6931758F, 1e-4F);
    EXPECT_NEAR(values[1], 7.5463768F, 1e-4F);
}

TEST(ConstantFoldingPass, DceRemovesFoldedSiluMul) {
    ModelGraph graph;
    const GraphValueId gate = AddFloatConstant(graph, {1.0F}, {1}, "gate");
    const GraphValueId up = AddFloatConstant(graph, {1.0F}, {1}, "up");
    auto act_or = AddSiluMul(graph, 0U, gate, up, "silu_mul");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

// ── Rank-zero constant folding ──
// These tests trace rank-0 inline bytes through HasInlineConstantBytes,
// BuildInputViews, AllocateOutputViews, Evaluate, node replacement, and DCE.

TEST(ConstantFoldingPass, FoldsRankZeroAdd) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<float>(graph, DataType::Float32(), {3.0F}, {}, "lhs");
    const GraphValueId rhs = AddTypedConstant<float>(graph, DataType::Float32(), {5.0F}, {}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    // Verify rank-zero TensorSpec is preserved.
    EXPECT_EQ(output.spec.shape.rank(), std::optional<size_t>{0});
    EXPECT_TRUE(output.spec.IsRankZero());
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    ASSERT_TRUE(constant->binding.inline_data != nullptr);
    EXPECT_EQ(constant->binding.inline_data->size(), static_cast<size_t>(output.spec.dtype.nbytes()));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 1U);
    EXPECT_FLOAT_EQ(values[0], 8.0F);
}

TEST(ConstantFoldingPass, FoldsRankZeroAddInt32) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<int32_t>(graph, DataType::Int(32), {7}, {}, "lhs");
    const GraphValueId rhs = AddTypedConstant<int32_t>(graph, DataType::Int(32), {3}, {}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    EXPECT_EQ(constant->binding.inline_data->size(), static_cast<size_t>(DataType::Int(32).nbytes()));
    const std::vector<int32_t> values = ReadTypedConstant<int32_t>(output);
    ASSERT_EQ(values.size(), 1U);
    EXPECT_EQ(values[0], 10);
}

TEST(ConstantFoldingPass, FoldsRankZeroAddBFloat16) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant(graph,
                                              DataType::BFloat(16),
                                              BFloat16Values({0x4000U}),// 2.0
                                              {},
                                              "lhs");
    const GraphValueId rhs = AddTypedConstant(graph,
                                              DataType::BFloat(16),
                                              BFloat16Values({0x4080U}),// 4.0
                                              {},
                                              "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
    EXPECT_EQ(BFloat16Bits(ReadTypedConstant<BFloat16>(output)),
              (std::vector<uint16_t>{0x40C0U}));// 6.0
}

TEST(ConstantFoldingPass, FoldsRankZeroElementwiseMul) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<float>(graph, DataType::Float32(), {4.0F}, {}, "lhs");
    const GraphValueId rhs = AddTypedConstant<float>(graph, DataType::Float32(), {2.5F}, {}, "rhs");
    auto product_or = AddElementwiseMul(graph, 0U, lhs, rhs, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    EXPECT_EQ(constant->binding.inline_data->size(), static_cast<size_t>(DataType::Float32().nbytes()));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 1U);
    EXPECT_FLOAT_EQ(values[0], 10.0F);
}

TEST(ConstantFoldingPass, FoldsRankZeroSilu) {
    ModelGraph graph;
    const GraphValueId input = AddTypedConstant<float>(graph, DataType::Float32(), {1.0F}, {}, "input");
    auto act_or = AddSilu(graph, 0U, input, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    EXPECT_EQ(constant->binding.inline_data->size(), static_cast<size_t>(DataType::Float32().nbytes()));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 1U);
    // silu(1.0) ≈ 0.7310586
    EXPECT_NEAR(values[0], 0.7310586F, 1e-5F);
}

TEST(ConstantFoldingPass, FoldsRankZeroSiluMul) {
    ModelGraph graph;
    const GraphValueId gate = AddTypedConstant<float>(graph, DataType::Float32(), {2.0F}, {}, "gate");
    const GraphValueId up = AddTypedConstant<float>(graph, DataType::Float32(), {3.0F}, {}, "up");
    auto act_or = AddSiluMul(graph, 0U, gate, up, "silu_mul");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    EXPECT_EQ(constant->binding.inline_data->size(), static_cast<size_t>(DataType::Float32().nbytes()));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 1U);
    // silu(2.0) * 3.0 ≈ 1.7615942 * 3.0 ≈ 5.2847826
    EXPECT_NEAR(values[0], 5.2847826F, 1e-5F);
}

// ── Rank-zero chain folding ──

TEST(ConstantFoldingPass, FoldsRankZeroAddThenSiluThenMulChained) {
    ModelGraph graph;
    const GraphValueId a = AddTypedConstant<float>(graph, DataType::Float32(), {1.0F}, {}, "a");
    const GraphValueId b = AddTypedConstant<float>(graph, DataType::Float32(), {2.0F}, {}, "b");
    const GraphValueId c = AddTypedConstant<float>(graph, DataType::Float32(), {3.0F}, {}, "c");
    auto sum_or = AddElementwiseAdd(graph, 0U, a, b, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    auto act_or = AddSilu(graph, 0U, sum, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    auto product_or = AddElementwiseMul(graph, 0U, act, c, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
    const auto* constant = std::get_if<ConstantValue>(&output.payload);
    EXPECT_EQ(constant->binding.inline_data->size(), static_cast<size_t>(DataType::Float32().nbytes()));
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 1U);
    // sum = 1 + 2 = 3, silu(3) ≈ 2.8577224, * 3 ≈ 8.5731672
    EXPECT_NEAR(values[0], 8.5731672F, 1e-5F);
}

// ── Rank-zero scalar-tensor broadcast folding ──

TEST(ConstantFoldingPass, FoldsRankZeroScalarTensorAddBroadcast) {
    ModelGraph graph;
    // rank-0 scalar + rank-2 tensor → rank-2 output
    const GraphValueId scalar = AddFloatConstant(graph, {2.0F}, {}, "scalar");
    const GraphValueId tensor = AddFloatConstant(graph, {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2}, "tensor");
    auto sum_or = AddElementwiseAdd(graph, 0U, scalar, tensor, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    // Output must be rank-2, not rank-0.
    EXPECT_EQ(output.spec.shape.rank(), std::optional<size_t>{2});
    EXPECT_FALSE(output.spec.IsRankZero());
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 4U);
    EXPECT_FLOAT_EQ(values[0], 3.0F);
    EXPECT_FLOAT_EQ(values[1], 4.0F);
    EXPECT_FLOAT_EQ(values[2], 5.0F);
    EXPECT_FLOAT_EQ(values[3], 6.0F);
}

TEST(ConstantFoldingPass, FoldsRankZeroTensorScalarAddBroadcast) {
    ModelGraph graph;
    // rank-2 tensor + rank-0 scalar → rank-2 output
    const GraphValueId tensor = AddFloatConstant(graph, {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2}, "tensor");
    const GraphValueId scalar = AddFloatConstant(graph, {10.0F}, {}, "scalar");
    auto sum_or = AddElementwiseAdd(graph, 0U, tensor, scalar, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_EQ(output.spec.shape.rank(), std::optional<size_t>{2});
    EXPECT_FALSE(output.spec.IsRankZero());
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 4U);
    EXPECT_FLOAT_EQ(values[0], 11.0F);
    EXPECT_FLOAT_EQ(values[1], 12.0F);
    EXPECT_FLOAT_EQ(values[2], 13.0F);
    EXPECT_FLOAT_EQ(values[3], 14.0F);
}

// ── Rank-zero malformed inline data ──

TEST(ConstantFoldingPass, SkipsRankZeroMalformedInlineBytes) {
    ModelGraph graph;
    // Rank-zero float spec expects 4 bytes, but we provide only 1 byte.
    const auto bad_bytes = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x01}});
    const GraphValueId lhs = graph.AddConstant(
            Spec(DataType::Float32(), {}),
            ConstantBinding{.inline_data = bad_bytes, .name = "bad_lhs"},
            "bad_lhs");
    const GraphValueId rhs = AddTypedConstant<float>(graph, DataType::Float32(), {2.0F}, {}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    // Malformed bytes cause HasInlineConstantBytes to reject → skip fold, keep Add.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(
            result->GetValue(result->GetOutputs()[0].value).payload));
}

// ── Rank-zero folding + DCE ──

TEST(ConstantFoldingPass, DceRemovesFoldedRankZeroAdd) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<float>(graph, DataType::Float32(), {3.0F}, {}, "lhs");
    const GraphValueId rhs = AddTypedConstant<float>(graph, DataType::Float32(), {7.0F}, {}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    const GraphValue& output = result->GetValue(result->GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));
    EXPECT_TRUE(output.spec.IsRankZero());
}

TEST(ConstantFoldingPass, DceRemovesFoldedRankZeroMul) {
    ModelGraph graph;
    const GraphValueId lhs = AddTypedConstant<float>(graph, DataType::Float32(), {2.0F}, {}, "lhs");
    const GraphValueId rhs = AddTypedConstant<float>(graph, DataType::Float32(), {3.0F}, {}, "rhs");
    auto product_or = AddElementwiseMul(graph, 0U, lhs, rhs, "product");
    ASSERT_TRUE(product_or.ok()) << product_or.status().ToString();
    const GraphValueId product = *product_or;
    graph.MarkOutput(product);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, DceRemovesFoldedRankZeroSilu) {
    ModelGraph graph;
    const GraphValueId input = AddTypedConstant<float>(graph, DataType::Float32(), {0.5F}, {}, "input");
    auto act_or = AddSilu(graph, 0U, input, "act");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, DceRemovesFoldedRankZeroSiluMul) {
    ModelGraph graph;
    const GraphValueId gate = AddTypedConstant<float>(graph, DataType::Float32(), {1.0F}, {}, "gate");
    const GraphValueId up = AddTypedConstant<float>(graph, DataType::Float32(), {2.0F}, {}, "up");
    auto act_or = AddSiluMul(graph, 0U, gate, up, "silu_mul");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    graph.MarkOutput(act);

    const StatusOr<ModelGraph> result = RunConstantFoldingThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

}// namespace
}// namespace aethermind
