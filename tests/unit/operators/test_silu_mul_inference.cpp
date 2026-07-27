#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <variant>
#include <vector>

namespace {
using namespace aethermind;

// --- Validation ---

TEST(SiluMulInference, InferOperatorAcceptsValidParams) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status().ok());
}

TEST(SiluMulInference, RejectsWrongArity) {
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulInference, RejectsNonFloat32Input) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulInference, RejectsStaticIncompatibleBroadcast) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 3}).shape},
    };
    const Status status = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulInference, RejectsUnrankedInput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape{}},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulInference, AcceptsSameShape) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status().ok());
}

TEST(SiluMulInference, AcceptsBroadcastShape) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {1, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 1}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status().ok());
}

TEST(SiluMulInference, AcceptsDifferentRankBroadcast) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status().ok());
}

TEST(SiluMulInference, AcceptsRankZeroInput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status().ok());
}

TEST(SiluMulInference, AcceptsZeroDimension) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {0, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {0, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs).status().ok());
}

// --- Inference ---

TEST(SiluMulInference, InferOperatorRejectsNonFloat32) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulInference, InfersSameShapeOutput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 8}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 8}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(SiluMulInference, InfersBroadcastOutputShape) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {1, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 1}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(SiluMulInference, InfersRankZeroOutput) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(SiluMulInference, InfersDifferentRankBroadcast) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(SiluMulInference, EmitsDeferredDimBroadcastableConstraints) {
    const ShapeSymbol gate_dim0 = ShapeSymbol::Create();
    const ShapeSymbol gate_dim1 = ShapeSymbol::Create();
    const ShapeSymbol up_dim0 = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{gate_dim0, gate_dim1})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{up_dim0})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSiluMul, OpParams{SiluMulParams{}}, inputs);

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

TEST(SiluMulInference, AcceptsFloat32ViaMakeSpec) {
    auto gate = MakeSpec(DataType::Float32(), {4, 256});
    auto up = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs).ok());
}

TEST(SiluMulInference, InfersFloat32OutputDtype) {
    auto gate = MakeSpec(DataType::Float32(), {4, 256});
    auto up = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    auto result = InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(SiluMulInference, AcceptsBFloat16) {
    auto gate = MakeSpec(DataType::BFloat(16), {4, 256});
    auto up = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    auto result = InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

}// namespace
