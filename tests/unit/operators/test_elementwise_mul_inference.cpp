#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Validation ---

TEST(ElementwiseMulInference, InferOperatorAcceptsValidParams) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status().ok());
}

TEST(ElementwiseMulInference, RejectsWrongArity) {
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulInference, RejectsNonFloat32Input) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulInference, RejectsStaticIncompatibleBroadcast) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 3}).shape},
    };
    const Status status = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulInference, RejectsUnrankedInput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape{}},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulInference, AcceptsSameShape) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status().ok());
}

TEST(ElementwiseMulInference, AcceptsBroadcastShape) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {1, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 1}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status().ok());
}

TEST(ElementwiseMulInference, AcceptsDifferentRankBroadcast) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status().ok());
}

TEST(ElementwiseMulInference, AcceptsRankZeroInput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status().ok());
}

TEST(ElementwiseMulInference, AcceptsZeroDimension) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {0, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {0, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs).status().ok());
}

// --- Inference ---

TEST(ElementwiseMulInference, InferOperatorRejectsNonFloat32) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulInference, InfersSameShapeOutput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 8}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 8}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(ElementwiseMulInference, InfersBroadcastOutputShape) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {1, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 1}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(ElementwiseMulInference, InfersRankZeroOutput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(ElementwiseMulInference, InfersDifferentRankBroadcast) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(ElementwiseMulInference, EmitsDeferredDimBroadcastableConstraints) {
    const ShapeSymbol lhs_dim0 = ShapeSymbol::Create();
    const ShapeSymbol lhs_dim1 = ShapeSymbol::Create();
    const ShapeSymbol rhs_dim0 = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{lhs_dim0, lhs_dim1})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_dim0})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kElementwiseMul, OpParams{ElementwiseMulParams{}}, inputs);

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

TEST(ElementwiseMulInference, AcceptsFloat32ViaMakeSpec) {
    auto lhs = MakeSpec(DataType::Float32(), {4, 256});
    auto rhs = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_TRUE(InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs).ok());
}

TEST(ElementwiseMulInference, InfersFloat32OutputDtype) {
    auto lhs = MakeSpec(DataType::Float32(), {4, 256});
    auto rhs = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(ElementwiseMulInference, AcceptsBFloat16) {
    auto lhs = MakeSpec(DataType::BFloat(16), {4, 256});
    auto rhs = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(ElementwiseMulInference, AcceptsFloat16) {
    auto lhs = MakeSpec(DataType::Float(16), {4, 256});
    auto rhs = MakeSpec(DataType::Float(16), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(ElementwiseMulInference, RejectsInt32Input) {
    auto lhs = MakeSpec(DataType::Int(32), {4, 256});
    auto rhs = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs).ok());
}

TEST(ElementwiseMulInference, RejectsFloat8E4M3FN) {
    // FP8 is intentionally not supported by ElementwiseMul (precision concerns
    // for the multiplication result).
    auto lhs = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    auto rhs = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs).ok());
}

}// namespace
