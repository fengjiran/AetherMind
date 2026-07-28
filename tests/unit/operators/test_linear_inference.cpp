#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_semantics_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// --- Validation ---

TEST(LinearInference, ValidatesStaticInputContract) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {16, 8}),
    };

    EXPECT_TRUE(InferOperator(OpType::kLinear, params, inputs).status().ok());
}

TEST(LinearInference, AcceptsRank1Input) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {8}),
            MakeSpec(DataType::Float32(), {16, 8}),
    };

    EXPECT_TRUE(InferOperator(OpType::kLinear, params, inputs).status().ok());
}

TEST(LinearInference, RejectsRankZeroInput) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {}),
            MakeSpec(DataType::Float32(), {16, 8}),
    };

    const Status status = InferOperator(OpType::kLinear, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearInference, RejectsRankZeroWeight) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {}),
    };

    const Status status = InferOperator(OpType::kLinear, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearInference, RejectsRank3Input) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {2, 4, 3}),
            MakeSpec(DataType::Float32(), {16, 8}),
    };

    const Status status = InferOperator(OpType::kLinear, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearInference, RejectsWeightRank1) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {8}),
    };

    const Status status = InferOperator(OpType::kLinear, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearInference, RejectsStaticKMismatch) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {16, 16}),
    };

    const Status status = InferOperator(OpType::kLinear, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

// --- Inference ---

TEST(LinearInference, InfersOutputShapeFromWeight) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {11008, 4096}),
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kLinear, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 11008);
}

TEST(LinearInference, InfersRank1OutputShape) {
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4096}),
            MakeSpec(DataType::Float32(), {11008, 4096}),
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kLinear, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 1U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 11008);
}

TEST(LinearInference, EmitsRuntimeCheckForDistinctSymbolicK) {
    constexpr LinearParams params;
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol input_k = ShapeSymbol::Create();
    const ShapeSymbol out_features = ShapeSymbol::Create();
    const ShapeSymbol weight_k = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, input_k})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{out_features, weight_k})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kLinear, params, inputs);

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

TEST(LinearInference, AcceptsSharedSymbolicK) {
    constexpr LinearParams params;
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol k = ShapeSymbol::Create();
    const ShapeSymbol out_features = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, k})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{out_features, k})},
    };

    EXPECT_TRUE(InferOperator(OpType::kLinear, params, inputs).status().ok());
}

TEST(LinearInference, AcceptsZeroBatchDim) {
    // Zero batch yields empty output [0, out]; valid NumPy/PyTorch MatMul semantics.
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {0, 8}),
            MakeSpec(DataType::Float32(), {16, 8}),
    };
    EXPECT_TRUE(InferOperator(OpType::kLinear, params, inputs).status().ok());
}

TEST(LinearInference, AcceptsZeroInFeatures) {
    // Zero in_features (K=0) yields zero-valued output; valid NumPy/PyTorch MatMul.
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 0}),
            MakeSpec(DataType::Float32(), {16, 0}),
    };
    EXPECT_TRUE(InferOperator(OpType::kLinear, params, inputs).status().ok());
}

TEST(LinearInference, AcceptsZeroOutFeatures) {
    // Zero out_features yields empty output; valid NumPy/PyTorch MatMul.
    constexpr LinearParams params;
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {0, 8}),
    };
    EXPECT_TRUE(InferOperator(OpType::kLinear, params, inputs).status().ok());
}

// --- Semantics ---

TEST(LinearInference, AcceptsFloat32ViaMakeSpec) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_TRUE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(LinearInference, InfersRank2Output) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(LinearInference, InfersRank1Output) {
    auto input = MakeSpec(DataType::Float32(), {256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 1);
}

TEST(LinearInference, RejectsInFeaturesMismatch) {
    auto input = MakeSpec(DataType::Float32(), {4, 128});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(LinearInference, AcceptsFloat16Activation) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Float(16), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(LinearInference, AcceptsBFloat16Activation) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(LinearInference, AcceptsFloat8E4M3Activation) {
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E4M3FN(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(LinearInference, AcceptsFloat8E5M2Activation) {
    auto input = MakeSpec(DataType::Float8E5M2(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E5M2(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E5M2());
}

TEST(LinearInference, AcceptsInt8Weight) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Int(8), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(LinearInference, AcceptsInt4Weight) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Int(4), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(LinearInference, AcceptsMixedDType) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Int(8), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(LinearInference, RejectsWrongActivationDtype) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(LinearInference, RejectsWrongWeightDtype) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Double(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

}// namespace
