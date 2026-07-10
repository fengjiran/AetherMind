#include "aethermind/model/graph/constant_folding_pass.h"

#include "aethermind/model/graph/dead_code_elimination_pass.h"
#include "aethermind/model/graph/graph_op_builder.h"

#include <gtest/gtest.h>

#include <array>
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

TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return {.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

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
    const auto node = graph.AddNode(
            OpType::kAdd,
            std::nullopt,
            {lhs, rhs},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), std::move(output_shape)),
                            .payload = ActivationValue{}}},
            AddParams{},
            {},
            std::move(name));
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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");
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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

    const StatusOr<ModelGraph> result = RunConstantFolding(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(result->GetValue(result->GetOutputs()[0].value).payload));
}

TEST(ConstantFoldingPass, SkipsWeightInputInSession) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.0F}, {1}, "lhs");
    const GraphValueId rhs = graph.AddWeight(
            Spec(DataType::Float32(), {1}),
            WeightBinding{.slot = ParameterSlot::kKernel,
                          .semantic_role = TransformerWeightRole::kAttentionQ},
            "rhs_weight");
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId a = AddElementwiseAdd(graph, 0U, b, c, "a");
    const GraphValueId d = AddElementwiseAdd(graph, 0U, a, e, "d");
    graph.MarkOutput(d, "output");

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
    ModelGraph graph;
    // Foldable: const_a + const_b -> foldable_sum
    const GraphValueId const_a = AddFloatConstant(graph, {1.0F, 2.0F}, {2}, "const_a");
    const GraphValueId const_b = AddFloatConstant(graph, {3.0F, 4.0F}, {2}, "const_b");
    const GraphValueId foldable_sum = AddElementwiseAdd(graph, 0U, const_a, const_b, "foldable_sum");

    // Non-foldable: const_c + input -> non_foldable_sum
    // This Add has a non-constant input, so AllInputsAreInlineConstantValues returns false.
    const GraphValueId const_c = AddFloatConstant(graph, {5.0F, 6.0F}, {2}, "const_c");
    const GraphValueId input = graph.AddInput(
            Spec(DataType::Float32(), {2}), "input");
    const GraphValueId non_foldable_sum = AddElementwiseAdd(graph, 0U, const_c, input, "non_foldable_sum");

    graph.MarkOutput(foldable_sum, "output1");
    graph.MarkOutput(non_foldable_sum, "output2");

    GraphRewriteSession session(graph);
    ConstantFoldingPass pass;
    const Status status = pass.Run(session, PassContext{});
    ASSERT_TRUE(status.ok()) << status.ToString();

    // Foldable output should be redirected to a folded constant.
    EXPECT_NE(session.GetResolvedValue(foldable_sum), foldable_sum);

    // Non-foldable output should remain unchanged.
    EXPECT_EQ(session.GetResolvedValue(non_foldable_sum), non_foldable_sum);
}

// ── Empty tensor (numel == 0) folding ──

TEST(ConstantFoldingPass, FoldsAddOfEmptyTensors) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {}, {0}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {}, {0}, "rhs");
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    ModelGraph graph;
    const AddedNode node = graph.AddNode(
            OpType::kUnknown,
            std::nullopt,
            {},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), {1}),
                            .payload = ActivationValue{}}},
            std::monostate{},
            {},
            "unknown_op");
    ASSERT_EQ(node.outputs.size(), 1U);

    GraphRewriteSession session(graph);
    ConstantFoldingPass pass;

    const Status status = pass.Run(session, PassContext{});
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(session.GetResolvedValue(node.outputs[0]), node.outputs[0]);
}

// ── Single-element tensor (shape {1}) folding ──

TEST(ConstantFoldingPass, FoldsAddOfSingleElementConstants) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.5F}, {1}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {2.5F}, {1}, "rhs");
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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

}// namespace
}// namespace aethermind
