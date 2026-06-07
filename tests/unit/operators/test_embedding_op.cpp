#include "aethermind/operators/embedding_op.h"

#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(EmbeddingOp, ValidatesInputContract) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = {2}},
            TensorSpec{.dtype = DataType::Float32(), .shape = {3, 2}},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeFromTokenIdsAndWeight) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = {5}},
            TensorSpec{.dtype = DataType::Float32(), .shape = {32000, 4096}},
    };

    const StatusOr<std::vector<TensorSpec>> shapes = op.InferOutputShapes(inputs);

    ASSERT_TRUE(shapes.ok()) << shapes.status().ToString();
    ASSERT_EQ(shapes->size(), 1U);
    EXPECT_EQ((*shapes)[0].dtype, DataType::Float32());
    ASSERT_EQ((*shapes)[0].shape.size(), 2U);
    EXPECT_EQ((*shapes)[0].shape[0], 5);
    EXPECT_EQ((*shapes)[0].shape[1], 4096);
}

TEST(EmbeddingOp, AcceptsUint32TokenIds) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::UInt(32), .shape = {2}},
            TensorSpec{.dtype = DataType::Float32(), .shape = {3, 2}},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeWithUint32Tokens) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::UInt(32), .shape = {5}},
            TensorSpec{.dtype = DataType::Float32(), .shape = {32000, 4096}},
    };

    const StatusOr<std::vector<TensorSpec>> shapes = op.InferOutputShapes(inputs);

    ASSERT_TRUE(shapes.ok()) << shapes.status().ToString();
    ASSERT_EQ(shapes->size(), 1U);
    EXPECT_EQ((*shapes)[0].dtype, DataType::Float32());
    ASSERT_EQ((*shapes)[0].shape.size(), 2U);
    EXPECT_EQ((*shapes)[0].shape[0], 5);
    EXPECT_EQ((*shapes)[0].shape[1], 4096);
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
