#include "aethermind/base/shape_constraint.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(ShapeConstraint, TensorPortDefaultsToInputZero) {
    TensorPort port;

    EXPECT_EQ(port.direction, TensorPortType::kInput);
    EXPECT_EQ(port.tensor_idx, 0U);
}

TEST(ShapeConstraint, DimLocatorIdentifiesTensorDimension) {
    DimLocator locator{
            .tensor_port = {.direction = TensorPortType::kOutput, .tensor_idx = 1},
            .dim_index = 2,
    };

    EXPECT_EQ(locator.tensor_port.direction, TensorPortType::kOutput);
    EXPECT_EQ(locator.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(locator.dim_index, 2U);
}

TEST(ShapeConstraint, VariantStoresDimensionEquality) {
    DimEqualConstraint equal{
            .lhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 0}, .dim_index = 1},
            .rhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1}, .dim_index = 0},
    };
    ShapeConstraint constraint{
            .condition = equal,
            .error_context = "matmul inner dimension",
    };

    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& stored = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(stored, equal);
    EXPECT_EQ(constraint.error_context, "matmul inner dimension");
}

TEST(ShapeConstraint, BroadcastableConstraintStoresBothDims) {
    DimBroadcastableConstraint broadcast{
            .lhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 0}, .dim_index = 3},
            .rhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1}, .dim_index = 1},
    };

    ConstraintVariant condition = broadcast;

    ASSERT_TRUE(std::holds_alternative<DimBroadcastableConstraint>(condition));
    EXPECT_EQ(std::get<DimBroadcastableConstraint>(condition), broadcast);
}

TEST(ShapeConstraint, VolumeEqualAllowsScalarVolumeSide) {
    VolumeEqualConstraint scalar_to_tensor{
            .lhs_dims = {},
            .rhs_dims = {{.tensor_port = {.direction = TensorPortType::kOutput, .tensor_idx = 0}, .dim_index = 0}},
    };

    EXPECT_TRUE(scalar_to_tensor.lhs_dims.empty());
    ASSERT_EQ(scalar_to_tensor.rhs_dims.size(), 1U);
    EXPECT_EQ(scalar_to_tensor.rhs_dims[0].tensor_port.direction, TensorPortType::kOutput);
}

TEST(ShapeConstraint, RankConstraintsStorePortAndRank) {
    TensorPort output_port{.direction = TensorPortType::kOutput, .tensor_idx = 0};
    RankEqualConstraint rank_equal{.port = output_port, .target_rank = 2};
    RankAtLeastConstraint rank_at_least{.port = output_port, .min_rank = 1};

    EXPECT_EQ(rank_equal.port, output_port);
    EXPECT_EQ(rank_equal.target_rank, 2U);
    EXPECT_EQ(rank_at_least.port, output_port);
    EXPECT_EQ(rank_at_least.min_rank, 1U);
}

TEST(ShapeConstraint, EvaluationResultSupportsDeferredState) {
    EXPECT_NE(ShapeConstraintEvaluationResult::kDeferred, ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_NE(ShapeConstraintEvaluationResult::kDeferred, ShapeConstraintEvaluationResult::kViolated);
}

}// namespace
