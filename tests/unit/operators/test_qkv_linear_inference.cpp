#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

QkvLinearParams MakeParams(int64_t q = 16, int64_t k = 8, int64_t v = 8) {
    return QkvLinearParams{.q_out_features = q,
                           .k_out_features = k,
                           .v_out_features = v,
                           .has_bias = false};
}

// --- Validation ---

TEST(QkvLinearInference, ValidatesStaticInputContract) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };

    EXPECT_TRUE(InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status().ok());
}

TEST(QkvLinearInference, AcceptsRank1Input) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };

    EXPECT_TRUE(InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status().ok());
}

TEST(QkvLinearInference, RejectsRankZeroInput) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };

    const Status status = InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsRankZeroWeight) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {}),
    };

    const Status status = InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsRank1Weight) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {4096}),
    };

    const Status status = InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsPackedRowsMismatch) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            // 31 rows != q(16) + k(8) + v(8) = 32.
            MakeSpec(DataType::Float32(), {31, 4096}),
    };

    const Status status = InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsNegativeOutFeatures) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };

    const Status status = InferOperator(
                                  OpType::kQkvLinear, MakeParams(-1, 8, 8), inputs)
                                  .status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsHasBias) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };
    const QkvLinearParams params{.q_out_features = 16,
                                 .k_out_features = 8,
                                 .v_out_features = 8,
                                 .has_bias = true};

    const Status status = InferOperator(OpType::kQkvLinear, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsWrongInputCount) {
    const TensorSpec inputs[1] = {MakeSpec(DataType::Float32(), {4, 4096})};

    const Status status = InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearInference, RejectsStaticKMismatch) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {32, 16}),
    };

    const Status status = InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

// --- Inference ---

TEST(QkvLinearInference, InfersQKVSplitShapes) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kQkvLinear, MakeParams(), inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 3U);
    for (const TensorSpec& output: inference->outputs) {
        EXPECT_EQ(output.dtype, DataType::Float32());
        ASSERT_EQ(output.shape.rank(), 2U);
        EXPECT_EQ(output.shape[0].GetStaticValue(), 4);
    }
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 16);
    EXPECT_EQ(inference->outputs[1].shape[1].GetStaticValue(), 8);
    EXPECT_EQ(inference->outputs[2].shape[1].GetStaticValue(), 8);
}

TEST(QkvLinearInference, InfersRank1Outputs) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kQkvLinear, MakeParams(), inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->outputs.size(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 16);
    EXPECT_EQ(inference->outputs[1].shape[0].GetStaticValue(), 8);
    EXPECT_EQ(inference->outputs[2].shape[0].GetStaticValue(), 8);
}

TEST(QkvLinearInference, AcceptsZeroOutFeatures) {
    // Zero q_out yields an empty q output; valid NumPy/PyTorch MatMul semantics.
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {16, 4096}),
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kQkvLinear, MakeParams(0, 8, 8), inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->outputs.size(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 0);
    EXPECT_EQ(inference->outputs[1].shape[1].GetStaticValue(), 8);
    EXPECT_EQ(inference->outputs[2].shape[1].GetStaticValue(), 8);
}

TEST(QkvLinearInference, AcceptsZeroBatchDim) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {0, 4096}),
            MakeSpec(DataType::Float32(), {32, 4096}),
    };
    EXPECT_TRUE(InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status().ok());
}

TEST(QkvLinearInference, EmitsRuntimeCheckForDistinctSymbolicK) {
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol input_k = ShapeSymbol::Create();
    const ShapeSymbol weight_k = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, input_k})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(
                     std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(32), weight_k})},
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kQkvLinear, MakeParams(), inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& equal = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(equal.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(equal.lhs.dim_index, 1U);
    EXPECT_EQ(equal.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(equal.rhs.dim_index, 1U);
}

TEST(QkvLinearInference, AcceptsSharedSymbolicK) {
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol k = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, k})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(
                     std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(32), k})},
    };

    const StatusOr<InferenceResult> inference =
            InferOperator(OpType::kQkvLinear, MakeParams(), inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
}

TEST(QkvLinearInference, DefersSymbolicPackedRows) {
    // Params are constants but the shape-constraint vocabulary only compares
    // tensor dimensions; a symbolic packed row count is deferred.
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol k = ShapeSymbol::Create();
    const ShapeSymbol packed_rows = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, k})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{packed_rows, k})},
    };

    EXPECT_TRUE(InferOperator(OpType::kQkvLinear, MakeParams(), inputs).status().ok());
}

// --- Dtypes ---

TEST(QkvLinearInference, AcceptsFloat16Activation) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float(16), {4, 256}),
            MakeSpec(DataType::Float(16), {32, 256}),
    };
    const auto result = InferOperator(OpType::kQkvLinear, MakeParams(), inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(QkvLinearInference, AcceptsBFloat16Activation) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::BFloat(16), {4, 256}),
            MakeSpec(DataType::BFloat(16), {32, 256}),
    };
    const auto result = InferOperator(OpType::kQkvLinear, MakeParams(), inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(QkvLinearInference, AcceptsInt8Weight) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 256}),
            MakeSpec(DataType::Int(8), {32, 256}),
    };
    const auto result = InferOperator(OpType::kQkvLinear, MakeParams(), inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(QkvLinearInference, AcceptsInt4Weight) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 256}),
            MakeSpec(DataType::Int(4), {32, 256}),
    };
    const auto result = InferOperator(OpType::kQkvLinear, MakeParams(), inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(QkvLinearInference, AcceptsMixedDType) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float(16), {4, 256}),
            MakeSpec(DataType::Int(8), {32, 256}),
    };
    const auto result = InferOperator(OpType::kQkvLinear, MakeParams(), inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(QkvLinearInference, RejectsWrongActivationDtype) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Int(32), {4, 256}),
            MakeSpec(DataType::Float32(), {32, 256}),
    };
    EXPECT_FALSE(InferOperator(OpType::kQkvLinear, MakeParams(), inputs).ok());
}

TEST(QkvLinearInference, RejectsWrongWeightDtype) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 256}),
            MakeSpec(DataType::Double(), {32, 256}),
    };
    EXPECT_FALSE(InferOperator(OpType::kQkvLinear, MakeParams(), inputs).ok());
}

}// namespace
