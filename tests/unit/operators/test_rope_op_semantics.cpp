#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, RoPEValidParams) {
    RoPEParams p{64, 32, 8, 2048};
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    std::vector<TensorSpec> inputs = {q, k, pos};
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(OperatorSemanticsValidate, RoPEInvalidParams) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    std::vector<TensorSpec> inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE,
                               RoPEParams{0,
                                          32,
                                          8,
                                          2048},
                               inputs)
                         .ok());
    EXPECT_FALSE(InferOperator(OpType::kRoPE,
                               RoPEParams{64,
                                          0,
                                          8,
                                          2048},
                               inputs)
                         .ok());
    EXPECT_FALSE(InferOperator(OpType::kRoPE,
                               RoPEParams{64,
                                          32,
                                          0,
                                          2048},
                               inputs)
                         .ok());
    EXPECT_FALSE(InferOperator(OpType::kRoPE,
                               RoPEParams{64,
                                          32,
                                          8,
                                          0},
                               inputs)
                         .ok());
}

TEST(OperatorSemanticsValidate, RoPEInvalidTheta) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    std::vector<TensorSpec> inputs = {q, k, pos};
    RoPEParams p{64, 32, 8, 2048};
    p.theta = 0.0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
    p.theta = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEFloat32Ok) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, rope_inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 2);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, RoPEGQACompatible) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 2048});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 512});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, rope_inputs);
    ASSERT_TRUE(result.ok());
}

TEST(OperatorSemanticsInfer, RoPEQLastDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 100});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEKLastDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 100});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPERankMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEBatchDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {2, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEWrongPositionIdsDtype) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Float32(), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

}// namespace
