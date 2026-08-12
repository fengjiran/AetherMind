#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Validation ---

TEST(MatMulInference, InferOperatorAcceptsValidParams) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, RejectsWrongArity) {
    constexpr MatMulParams params;
    const TensorSpec inputs[3] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3, 4}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsNonFloat32Input) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Int(32), {2, 3}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsUnrankedLhs) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32()),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsUnrankedRhs) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32()),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsRank1Lhs) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {3}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsRank1Rhs) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsStaticInnerMismatch) {
    constexpr MatMulParams params;// transpose_rhs=false
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {4, 5}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsStaticInnerMismatchWithTransposeRhs) {
    const MatMulParams params{.transpose_rhs = true};// rhs layout [..., N, K]
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),// K=3
            MakeSpec(DataType::Float32(), {4, 5}),// K=5 -> mismatch
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, RejectsStaticIncompatibleBatch) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 2, 3}),
            MakeSpec(DataType::Float32(), {4, 3, 5}),
    };
    const Status status = InferOperator(OpType::kMatMul, params, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, AcceptsRank2Inputs) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsBatchedMatMul) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {5, 2, 3}),
            MakeSpec(DataType::Float32(), {5, 3, 4}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsBroadcastBatch) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {1, 2, 3}),
            MakeSpec(DataType::Float32(), {4, 3, 5}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsTransposeRhs) {
    const MatMulParams params{.transpose_rhs = true};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {4, 3}),// [N, K] = [4, 3]
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsZeroMDim) {
    // M=0 yields an empty output [0, N]; valid NumPy-style MatMul.
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {0, 3}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsZeroNDim) {
    // N=0 yields an empty output [M, 0]; valid NumPy-style MatMul.
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3, 0}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsZeroKDim) {
    // K=0 yields a zero-valued output [M, N]; valid NumPy-style MatMul.
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 0}),
            MakeSpec(DataType::Float32(), {0, 4}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

TEST(MatMulInference, AcceptsZeroBatchDim) {
    // batch=0 yields an empty output [0, M, N]; valid NumPy-style MatMul.
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {0, 2, 3}),
            MakeSpec(DataType::Float32(), {0, 3, 4}),
    };
    EXPECT_TRUE(InferOperator(OpType::kMatMul, params, inputs).status().ok());
}

// --- Inference ---

TEST(MatMulInference, InferOperatorRejectsNonFloat32) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Int(32), {2, 3}),
            MakeSpec(DataType::Int(32), {3, 4}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(MatMulInference, InfersRank2Output) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {3, 4}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4);
}

TEST(MatMulInference, InfersRank2OutputWithTransposeRhs) {
    const MatMulParams params{.transpose_rhs = true};
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3}),// [M, K]
            MakeSpec(DataType::Float32(), {4, 3}),// [N, K]
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);// M
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4);// N
}

TEST(MatMulInference, InfersBatchedOutput) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {5, 2, 3}),
            MakeSpec(DataType::Float32(), {5, 3, 4}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 4);
}

TEST(MatMulInference, InfersBroadcastBatchOutput) {
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {1, 2, 3}),
            MakeSpec(DataType::Float32(), {4, 3, 5}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);// broadcast batch
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 5);
}

TEST(MatMulInference, InfersDifferentRankBatchBroadcast) {
    // lhs batch [2], rhs batch [] -> output batch [2]
    constexpr MatMulParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 3, 4}),
            MakeSpec(DataType::Float32(), {4, 5}),
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 5);
}

TEST(MatMulInference, EmitsInnerDimEqualConstraintForSymbolicInner) {
    constexpr MatMulParams params;
    const ShapeSymbol lhs_inner = ShapeSymbol::Create();
    const ShapeSymbol rhs_inner = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), lhs_inner})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_inner, ShapeSymbol::CreateFromValue(4)})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& eq = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(eq.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(eq.lhs.dim_index, 1U);// lhs_inner at axis 1
    EXPECT_EQ(eq.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(eq.rhs.dim_index, 0U);// rhs_inner (K) at axis 0 (transpose_rhs=false)
}

TEST(MatMulInference, EmitsInnerDimEqualConstraintForTransposeRhs) {
    const MatMulParams params{.transpose_rhs = true};
    const ShapeSymbol lhs_inner = ShapeSymbol::Create();
    const ShapeSymbol rhs_inner = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), lhs_inner})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(4), rhs_inner})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& eq = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(eq.lhs.dim_index, 1U);
    // transpose_rhs=true: rhs layout [N, K], K at axis 1
    EXPECT_EQ(eq.rhs.dim_index, 1U);
}

TEST(MatMulInference, EmitsBatchBroadcastableConstraintForSymbolicBatch) {
    constexpr MatMulParams params;
    const ShapeSymbol lhs_batch = ShapeSymbol::Create();
    const ShapeSymbol rhs_batch = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{
                     lhs_batch, ShapeSymbol::CreateFromValue(2), ShapeSymbol::CreateFromValue(3)})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{
                     rhs_batch, ShapeSymbol::CreateFromValue(3), ShapeSymbol::CreateFromValue(4)})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kMatMul, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // Inner dims are static 3 == 3, so only the batch dim emits a constraint.
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimBroadcastableConstraint>(constraint.condition));
    const auto& bc = std::get<DimBroadcastableConstraint>(constraint.condition);
    EXPECT_EQ(bc.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(bc.lhs.dim_index, 0U);// batch axis 0 on lhs
    EXPECT_EQ(bc.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(bc.rhs.dim_index, 0U);// batch axis 0 on rhs
}

// --- Semantics ---

TEST(MatMulInference, AcceptsFloat32ViaMakeSpec) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_TRUE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(MatMulInference, InfersRank2OutputViaMakeSpec) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(MatMulInference, RejectsInnerMismatch) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {5, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(MatMulInference, AcceptsTransposeRhsViaMakeSpec) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {4, 3});
    MatMulParams p{true};
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, p, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(MatMulInference, AcceptsFloat16) {
    auto lhs = MakeSpec(DataType::Float(16), {2, 3});
    auto rhs = MakeSpec(DataType::Float(16), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(MatMulInference, AcceptsBFloat16) {
    auto lhs = MakeSpec(DataType::BFloat(16), {2, 3});
    auto rhs = MakeSpec(DataType::BFloat(16), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(MatMulInference, AcceptsFloat8E4M3) {
    auto lhs = MakeSpec(DataType::Float8E4M3FN(), {2, 3});
    auto rhs = MakeSpec(DataType::Float8E4M3FN(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(MatMulInference, AcceptsFloat8E5M2) {
    auto lhs = MakeSpec(DataType::Float8E5M2(), {2, 3});
    auto rhs = MakeSpec(DataType::Float8E5M2(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E5M2());
}

TEST(MatMulInference, AcceptsInt8) {
    auto lhs = MakeSpec(DataType::Int(8), {2, 3});
    auto rhs = MakeSpec(DataType::Int(8), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(8));
}

TEST(MatMulInference, AcceptsMixedDType) {
    auto lhs = MakeSpec(DataType::Float(16), {2, 3});
    auto rhs = MakeSpec(DataType::Int(8), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(MatMulInference, RejectsWrongLhsDtype) {
    auto lhs = MakeSpec(DataType::Int(32), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(MatMulInference, RejectsWrongRhsDtype) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Double(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

}// namespace
