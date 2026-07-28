#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/add_op.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_semantics_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Validation ---

TEST(AddInference, InferOperatorAcceptsValidParams) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3})};
    EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
}

TEST(AddInference, RejectsWrongArity) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[3] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3}),
    };
    const Status status = InferOperator(op.Type(), op_params, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddInference, AcceptsSupportedDTypes) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    for (const DataType& dtype: kAddSupportedDTypes) {
        SCOPED_TRACE(ToString(dtype));
        const TensorSpec inputs[2] = {
                MakeSpec(DataType::Float32(), {2, 3}),
                MakeSpec(DataType::Float32(), {2, 3}),
        };
        EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
    }
}

TEST(AddInference, RejectsMismatchedInputDTypes) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Int(32), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3})};
    const Status status = InferOperator(op.Type(), op_params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddInference, RejectsUnsupportedInputDType) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float(16), {2, 3}),
            MakeSpec(DataType::Float(16), {2, 3})};
    const Status status = InferOperator(op.Type(), op_params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddInference, RejectsStaticIncompatibleBroadcast) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {4, 3}),
    };
    const Status status = InferOperator(op.Type(), op_params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddInference, RejectsUnrankedInput) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32()),
            MakeSpec(DataType::Float32(), {2, 3}),
    };
    const Status status = InferOperator(op.Type(), op_params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddInference, AcceptsSameShape) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3}),
    };
    EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
}

TEST(AddInference, AcceptsBroadcastShape) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {1, 3}),
            MakeSpec(DataType::Float32(), {2, 1}),
    };
    EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
}

TEST(AddInference, AcceptsDifferentRankBroadcast) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3}),
    };
    EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
}

TEST(AddInference, AcceptsRankZeroInput) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {}),
    };
    EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
}

TEST(AddInference, AcceptsZeroDimension) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {0, 3}),
            MakeSpec(DataType::Float32(), {0, 3}),
    };
    EXPECT_TRUE(InferOperator(op.Type(), op_params, inputs).status().ok());
}

// --- Inference ---

TEST(AddInference, InfersSupportedOutputDTypes) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    for (const DataType& dtype: kAddSupportedDTypes) {
        SCOPED_TRACE(ToString(dtype));
        const TensorSpec inputs[2] = {
                MakeSpec(dtype, {2, 3}),
                MakeSpec(dtype, {2, 3}),
        };
        const StatusOr<InferenceResult> inference = InferOperator(op.Type(), op_params, inputs);
        ASSERT_TRUE(inference.ok()) << inference.status().ToString();
        ASSERT_EQ(inference->outputs.size(), 1U);
        EXPECT_EQ(inference->outputs[0].dtype, dtype);
    }
}

TEST(AddInference, InfersSameShapeOutput) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {4, 8}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), op_params, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(AddInference, InfersBroadcastOutputShape) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {1, 3}),
            MakeSpec(DataType::Float32(), {2, 1}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), op_params, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(AddInference, InfersRankZeroOutput) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), op_params, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(AddInference, InfersDifferentRankBroadcast) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), op_params, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(AddInference, EmitsDeferredDimBroadcastableConstraints) {
    AddOp::Params op_params{};
    const AddOp op{op_params};
    const ShapeSymbol lhs_dim0 = ShapeSymbol::Create();
    const ShapeSymbol lhs_dim1 = ShapeSymbol::Create();
    const ShapeSymbol rhs_dim0 = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{lhs_dim0, lhs_dim1})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_dim0})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), op_params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimBroadcastableConstraint>(constraint.condition));
    const auto& bc = std::get<DimBroadcastableConstraint>(constraint.condition);
    EXPECT_EQ(bc.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(bc.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(bc.lhs.dim_index, 1U);
    EXPECT_EQ(bc.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(bc.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(bc.rhs.dim_index, 0U);
}

// --- Semantics ---

TEST(AddInference, AcceptsFloat32ViaMakeSpec) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_TRUE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

TEST(AddInference, InfersFloat32OutputWithRank) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kAdd, AddParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(AddInference, InfersBroadcastOutput) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kAdd, AddParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(AddInference, RejectsDtypeMismatch) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Double(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

TEST(AddInference, RejectsWrongParamsVariant) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, RmsNormParams{}, inputs).ok());
}

TEST(AddInference, RejectsWrongInputCount) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

}// namespace
