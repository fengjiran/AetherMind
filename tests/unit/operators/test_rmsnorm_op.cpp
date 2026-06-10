#include "aethermind/operators/rmsnorm_op.h"

#include <gtest/gtest.h>

#include <variant>

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(RmsNormOp, ValidatesStaticInputContract) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(RmsNormOp, PreservesInputShapeAsOutputShape) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(RmsNormOp, EmitsRuntimeCheckForDistinctSymbolicHiddenDimension) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{seq_len, input_hidden})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{weight_hidden})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

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
    EXPECT_EQ(equal.rhs.dim_index, 0U);
}

TEST(RmsNormOp, AcceptsSharedSymbolicHiddenDimension) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{seq_len, hidden_size})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{hidden_size})},
    };

    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(RmsNormOp, RejectsStaticHiddenMismatch) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
