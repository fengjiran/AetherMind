#include "aethermind/operators/embedding_op.h"

#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

TEST(EmbeddingOp, ValidatesInputContract) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const ShapeInfo inputs[2] = {
            ShapeInfo{.dtype_ = DataType::Int(64), .shape_ = {2}},
            ShapeInfo{.dtype_ = DataType::Float32(), .shape_ = {3, 2}},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckShapes(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeFromTokenIdsAndWeight) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const ShapeInfo inputs[2] = {
            ShapeInfo{.dtype_ = DataType::Int(64), .shape_ = {5}},
            ShapeInfo{.dtype_ = DataType::Float32(), .shape_ = {32000, 4096}},
    };

    const StatusOr<std::vector<ShapeInfo>> shapes = op.InferOutputShapes(inputs);

    ASSERT_TRUE(shapes.ok()) << shapes.status().ToString();
    ASSERT_EQ(shapes->size(), 1U);
    EXPECT_EQ((*shapes)[0].dtype_, DataType::Float32());
    ASSERT_EQ((*shapes)[0].shape_.size(), 2U);
    EXPECT_EQ((*shapes)[0].shape_[0], 5);
    EXPECT_EQ((*shapes)[0].shape_[1], 4096);
}

TEST(EmbeddingOp, AcceptsUint32TokenIds) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const ShapeInfo inputs[2] = {
            ShapeInfo{.dtype_ = DataType::UInt(32), .shape_ = {2}},
            ShapeInfo{.dtype_ = DataType::Float32(), .shape_ = {3, 2}},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckShapes(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeWithUint32Tokens) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const ShapeInfo inputs[2] = {
            ShapeInfo{.dtype_ = DataType::UInt(32), .shape_ = {5}},
            ShapeInfo{.dtype_ = DataType::Float32(), .shape_ = {32000, 4096}},
    };

    const StatusOr<std::vector<ShapeInfo>> shapes = op.InferOutputShapes(inputs);

    ASSERT_TRUE(shapes.ok()) << shapes.status().ToString();
    ASSERT_EQ(shapes->size(), 1U);
    EXPECT_EQ((*shapes)[0].dtype_, DataType::Float32());
    ASSERT_EQ((*shapes)[0].shape_.size(), 2U);
    EXPECT_EQ((*shapes)[0].shape_[0], 5);
    EXPECT_EQ((*shapes)[0].shape_[1], 4096);
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
