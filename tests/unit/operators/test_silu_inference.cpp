#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

// --- Validation ---

TEST(SiluInference, InferOperatorAcceptsValidParams) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs).status().ok());
}

TEST(SiluInference, RejectsWrongArity) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluInference, RejectsNonFloat32Input) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
    };
    const Status status = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluInference, AcceptsArbitraryShape) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 8, 16}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs).status().ok());
}

TEST(SiluInference, AcceptsRankZeroInput) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    EXPECT_TRUE(InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs).status().ok());
}

TEST(SiluInference, AcceptsZeroDimension) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {0, 3}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs).status().ok());
}

// --- Inference ---

TEST(SiluInference, InferOperatorRejectsWrongArity) {
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {2, 3}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluInference, InferOperatorRejectsNonFloat32) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = MakeSpec(DataType::Int(32), {2, 3}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluInference, InfersIdenticalOutputShape) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {4, 8}).shape},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(SiluInference, InfersRankZeroOutput) {
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(SiluInference, InfersSymbolicOutputShape) {
    const ShapeSymbol dim0 = ShapeSymbol::Create();
    const ShapeSymbol dim1 = ShapeSymbol::Create();
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{dim0, dim1})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(OpType::kSilu, OpParams{SiluParams{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    // Symbolic dimensions are preserved as-is (element-wise op).
    EXPECT_EQ(inference->outputs[0].shape[0], dim0);
    EXPECT_EQ(inference->outputs[0].shape[1], dim1);
}

// --- Semantics ---

TEST(SiluInference, AcceptsFloat32ViaMakeSpec) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSilu, SiluParams{}, inputs).ok());
}

TEST(SiluInference, InfersFloat32OutputDtype) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSilu, SiluParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(SiluInference, AcceptsBFloat16) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSilu, SiluParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(SiluInference, RejectsInt32Input) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kSilu, SiluParams{}, inputs).ok());
}

}// namespace
