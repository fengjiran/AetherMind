#include "aethermind/base/shape_constraint.h"

#include "aethermind/base/shape_constraint_evaluator.h"

#include <gtest/gtest.h>

#include <vector>

using namespace aethermind;

namespace {

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

struct RuntimeTensorStorage {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    explicit RuntimeTensorStorage(std::vector<int64_t> input_shape)
        : shape(std::move(input_shape)),
          strides(MakeStrides(shape)) {}

    AM_NODISCARD TensorView View() const {
        return {nullptr, DataType::Float32(), shape, strides};
    }

    AM_NODISCARD MutableTensorView MutableView() {
        return {nullptr, DataType::Float32(), shape, strides};
    }
};

DimLocator InputDim(size_t tensor_idx, size_t dim_index) {
    return DimLocator{.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = tensor_idx},
                      .dim_index = dim_index};
}

TensorPort InputPort(size_t tensor_idx) {
    return TensorPort{.direction = TensorPortType::kInput, .tensor_idx = tensor_idx};
}

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

TEST(ShapeConstraintEvaluator, EvaluatesRuntimeDimensionEquality) {
    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage matching_weight{std::vector<int64_t>{8}};
    RuntimeTensorStorage mismatching_weight{std::vector<int64_t>{16}};
    const ShapeConstraint constraint{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "hidden size mismatch",
    };

    std::vector<TensorView> matching_inputs{input.View(), matching_weight.View()};
    const StatusOr<ShapeConstraintEvaluationResult> satisfied = EvaluateShapeConstraint(
            constraint,
            std::span<const TensorView>(matching_inputs),
            std::span<const MutableTensorView>());

    ASSERT_TRUE(satisfied.ok()) << satisfied.status().ToString();
    EXPECT_EQ(*satisfied, ShapeConstraintEvaluationResult::kSatisfied);

    std::vector<TensorView> mismatching_inputs{input.View(), mismatching_weight.View()};
    const StatusOr<ShapeConstraintEvaluationResult> violated = EvaluateShapeConstraint(
            constraint,
            std::span<const TensorView>(mismatching_inputs),
            std::span<const MutableTensorView>());

    ASSERT_TRUE(violated.ok()) << violated.status().ToString();
    EXPECT_EQ(*violated, ShapeConstraintEvaluationResult::kViolated);
}

TEST(ShapeConstraintEvaluator, EvaluatesRuntimeBroadcastVolumeAndRankConstraints) {
    RuntimeTensorStorage lhs{std::vector<int64_t>{2, 1, 4}};
    RuntimeTensorStorage rhs{std::vector<int64_t>{8, 7, 4}};
    std::vector<TensorView> inputs{lhs.View(), rhs.View()};

    const ShapeConstraint broadcastable{
            .condition = DimBroadcastableConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 1)},
            .error_context = "broadcast mismatch",
    };
    const ShapeConstraint volume_equal{
            .condition = VolumeEqualConstraint{.lhs_dims = {InputDim(0, 0), InputDim(0, 2)},
                                               .rhs_dims = {InputDim(1, 0)}},
            .error_context = "volume mismatch",
    };
    const ShapeConstraint rank_at_least{
            .condition = RankAtLeastConstraint{.port = InputPort(0), .min_rank = 2},
            .error_context = "rank too small",
    };
    const ShapeConstraint rank_equal{
            .condition = RankEqualConstraint{.port = InputPort(1), .target_rank = 2},
            .error_context = "rank mismatch",
    };

    EXPECT_EQ(*EvaluateShapeConstraint(broadcastable, std::span<const TensorView>(inputs), std::span<const MutableTensorView>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(*EvaluateShapeConstraint(volume_equal, std::span<const TensorView>(inputs), std::span<const MutableTensorView>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(*EvaluateShapeConstraint(rank_at_least, std::span<const TensorView>(inputs), std::span<const MutableTensorView>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(*EvaluateShapeConstraint(rank_equal, std::span<const TensorView>(inputs), std::span<const MutableTensorView>()),
              ShapeConstraintEvaluationResult::kViolated);
}

TEST(ShapeConstraintEvaluator, EvaluatesSymbolicDimensionEqualityStates) {
    const ShapeSymbol shared = ShapeSymbol::Create();
    std::vector<SymbolicShape> shared_inputs{
            SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), shared}),
            SymbolicShape(std::vector<ShapeSymbol>{shared}),
    };
    std::vector<SymbolicShape> conflicting_inputs{
            SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), ShapeSymbol::CreateFromValue(8)}),
            SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(16)}),
    };
    std::vector<SymbolicShape> deferred_inputs{
            SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), ShapeSymbol::Create()}),
            SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create()}),
    };
    const ShapeConstraint constraint{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "hidden size mismatch",
    };

    EXPECT_EQ(*EvaluateShapeConstraint(constraint, std::span<const SymbolicShape>(shared_inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(*EvaluateShapeConstraint(constraint, std::span<const SymbolicShape>(conflicting_inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kViolated);
    EXPECT_EQ(*EvaluateShapeConstraint(constraint, std::span<const SymbolicShape>(deferred_inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kDeferred);
}

TEST(ShapeConstraintEvaluator, SatisfiesIdenticalSymbolicVolumeDimensions) {
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    std::vector<SymbolicShape> inputs{
            SymbolicShape(std::vector<ShapeSymbol>{batch, ShapeSymbol::CreateFromValue(4), hidden}),
            SymbolicShape(std::vector<ShapeSymbol>{batch, ShapeSymbol::CreateFromValue(4), hidden}),
    };
    const ShapeConstraint identical_volume{
            .condition = VolumeEqualConstraint{.lhs_dims = {InputDim(0, 0), InputDim(0, 1), InputDim(0, 2)},
                                               .rhs_dims = {InputDim(1, 0), InputDim(1, 1), InputDim(1, 2)}},
            .error_context = "volume mismatch",
    };
    const ShapeConstraint permuted_volume{
            .condition = VolumeEqualConstraint{.lhs_dims = {InputDim(0, 0), InputDim(0, 1), InputDim(0, 2)},
                                               .rhs_dims = {InputDim(1, 2), InputDim(1, 1), InputDim(1, 0)}},
            .error_context = "volume mismatch",
    };

    EXPECT_EQ(*EvaluateShapeConstraint(identical_volume, std::span<const SymbolicShape>(inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(*EvaluateShapeConstraint(permuted_volume, std::span<const SymbolicShape>(inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kDeferred);
}

TEST(ShapeConstraintEvaluator, ValidateShapeConstraintsReturnsConstraintContext) {
    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage weight{std::vector<int64_t>{16}};
    std::vector<TensorView> inputs{input.View(), weight.View()};
    const ShapeConstraint constraint{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "hidden size mismatch",
    };

    const Status status = ValidateShapeConstraints(
            std::span<const ShapeConstraint>(&constraint, 1),
            std::span<const TensorView>(inputs),
            std::span<const MutableTensorView>());

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "hidden size mismatch");
}

}// namespace
