#include "aethermind/operators/embedding_op.h"

#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(EmbeddingOp, ValidatesInputContract) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({2})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeFromTokenIdsAndWeight) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({5})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, AcceptsUint32TokenIds) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::UInt(32), .shape = StaticShape({2})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeWithUint32Tokens) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::UInt(32), .shape = StaticShape({5})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, PreservesSymbolicTokenAndHiddenDims) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const ShapeSymbol token_count = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = SymbolicShape(std::vector<ShapeSymbol>{token_count})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), hidden_size})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0], token_count);
    EXPECT_EQ(inference->outputs[0].shape[1], hidden_size);
}

TEST(EmbeddingOp, RegistryCreatesDefaultEmbeddingOperator) {
    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(
            OpType::kEmbedding,
            EmbeddingOp::Params{});

    ASSERT_TRUE(op.ok()) << op.status().ToString();
    ASSERT_NE(op.value(), nullptr);
    EXPECT_EQ(op.value()->Type(), OpType::kEmbedding);
    EXPECT_STREQ(op.value()->Name(), "Embedding");
}

}// namespace
}// namespace aethermind
