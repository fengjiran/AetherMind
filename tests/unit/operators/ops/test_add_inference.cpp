#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/ops/add_op.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(AddInference, AcceptsEverySupportedDType) {
    for (const DataType& dtype: kAddSupportedDTypes) {
        SCOPED_TRACE(ToString(dtype));
        const TensorSpec inputs[2] = {
                MakeSpec(dtype, {2, 3}),
                MakeSpec(dtype, {2, 3}),
        };
        EXPECT_TRUE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
    }
}

TEST(AddInference, RejectsInvalidInputs) {
    const TensorSpec valid = MakeSpec(DataType::Float32(), {2, 3});
    EXPECT_EQ(InferOperator(OpType::kAdd, AddParams{}, std::span{&valid, size_t{1}})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);

    const TensorSpec mismatched_dtype[2] = {
            MakeSpec(DataType::Int(32), {2, 3}),
            valid,
    };
    EXPECT_EQ(InferOperator(OpType::kAdd, AddParams{}, mismatched_dtype)
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);

    const TensorSpec unsupported_dtype[2] = {
            MakeSpec(DataType::Float(16), {2, 3}),
            MakeSpec(DataType::Float(16), {2, 3}),
    };
    EXPECT_EQ(InferOperator(OpType::kAdd, AddParams{}, unsupported_dtype)
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);

    const TensorSpec incompatible_shape[2] = {
            valid,
            MakeSpec(DataType::Float32(), {4, 3}),
    };
    EXPECT_EQ(InferOperator(OpType::kAdd, AddParams{}, incompatible_shape)
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);

    const TensorSpec unranked[2] = {
            MakeSpec(DataType::Float32()),
            valid,
    };
    EXPECT_EQ(InferOperator(OpType::kAdd, AddParams{}, unranked).status().code(),
              StatusCode::kInvalidArgument);
}

TEST(AddInference, RejectsWrongParamsVariant) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3}),
    };
    EXPECT_EQ(InferOperator(OpType::kAdd, RmsNormParams{}, inputs).status().code(),
              StatusCode::kInvalidArgument);
}

TEST(AddInference, InfersBroadcastOutput) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {1, 3}),
            MakeSpec(DataType::Float32(), {2, 1}),
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kAdd, AddParams{}, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
    EXPECT_TRUE(inference->runtime_checks.empty());
}

TEST(AddInference, InfersDifferentRankAndRankZeroBroadcast) {
    const TensorSpec vector_inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3}),
    };
    const StatusOr<InferenceResult> vector_result =
            InferOperator(OpType::kAdd, AddParams{}, vector_inputs);
    ASSERT_TRUE(vector_result.ok()) << vector_result.status().ToString();
    EXPECT_EQ(vector_result->outputs[0].shape, vector_inputs[0].shape);

    const TensorSpec scalar_inputs[2] = {
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {}),
    };
    const StatusOr<InferenceResult> scalar_result =
            InferOperator(OpType::kAdd, AddParams{}, scalar_inputs);
    ASSERT_TRUE(scalar_result.ok()) << scalar_result.status().ToString();
    EXPECT_TRUE(scalar_result->outputs[0].shape.IsRankZero());
}

TEST(AddInference, EmitsDeferredBroadcastConstraint) {
    const ShapeSymbol lhs_dim0 = ShapeSymbol::Create();
    const ShapeSymbol lhs_dim1 = ShapeSymbol::Create();
    const ShapeSymbol rhs_dim0 = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{lhs_dim0, lhs_dim1})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_dim0})},
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kAdd, AddParams{}, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const auto* constraint =
            std::get_if<DimBroadcastableConstraint>(&inference->runtime_checks[0].condition);
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(constraint->lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(constraint->lhs.dim_index, 1U);
    EXPECT_EQ(constraint->rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(constraint->rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(constraint->rhs.dim_index, 0U);
}

}// namespace
