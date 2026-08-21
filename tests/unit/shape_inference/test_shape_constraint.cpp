#include "aethermind/shape_inference/shape_constraint.h"

#include "aethermind/shape_inference/shape_constraint_evaluator.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

namespace {
using namespace aethermind;

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
        static const int dummy = 0;
        return {&dummy, DataType::Float32(), shape, strides};
    }

    AM_NODISCARD MutableTensorView MutableView() {
        static int dummy = 0;
        return {&dummy, DataType::Float32(), shape, strides};
    }
};

DimLocator InputDim(size_t tensor_idx, size_t dim_index) {
    return DimLocator{.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = tensor_idx},
                      .dim_index = dim_index};
}

DimLocator OutputDim(size_t tensor_idx, size_t dim_index) {
    return DimLocator{.tensor_port = {.direction = TensorPortType::kOutput, .tensor_idx = tensor_idx},
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

TEST(ShapeConstraint, DimPositiveConstraintStoresLocator) {
    DimPositiveConstraint constraint{.dim = InputDim(1, 0)};
    EXPECT_EQ(constraint.dim.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(constraint.dim.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(constraint.dim.dim_index, 0U);
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
    const auto matching_result = EvaluateShapeConstraint(
            constraint,
            std::span<const TensorView>(matching_inputs),
            std::span<const MutableTensorView>());
    ASSERT_TRUE(matching_result.ok()) << matching_result.status().ToString();
    EXPECT_EQ(*matching_result, ShapeConstraintEvaluationResult::kSatisfied);

    std::vector<TensorView> mismatching_inputs{input.View(), mismatching_weight.View()};
    const auto mismatching_result = EvaluateShapeConstraint(
            constraint,
            std::span<const TensorView>(mismatching_inputs),
            std::span<const MutableTensorView>());
    ASSERT_TRUE(mismatching_result.ok()) << mismatching_result.status().ToString();
    EXPECT_EQ(*mismatching_result, ShapeConstraintEvaluationResult::kViolated);
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

    const auto broadcastable_result = EvaluateShapeConstraint(
            broadcastable, std::span<const TensorView>(inputs), std::span<const MutableTensorView>());
    ASSERT_TRUE(broadcastable_result.ok()) << broadcastable_result.status().ToString();
    EXPECT_EQ(*broadcastable_result, ShapeConstraintEvaluationResult::kSatisfied);
    const auto volume_equal_result = EvaluateShapeConstraint(
            volume_equal, std::span<const TensorView>(inputs), std::span<const MutableTensorView>());
    ASSERT_TRUE(volume_equal_result.ok()) << volume_equal_result.status().ToString();
    EXPECT_EQ(*volume_equal_result, ShapeConstraintEvaluationResult::kSatisfied);
    const auto rank_at_least_result = EvaluateShapeConstraint(
            rank_at_least, std::span<const TensorView>(inputs), std::span<const MutableTensorView>());
    ASSERT_TRUE(rank_at_least_result.ok()) << rank_at_least_result.status().ToString();
    EXPECT_EQ(*rank_at_least_result, ShapeConstraintEvaluationResult::kSatisfied);
    const auto rank_equal_result = EvaluateShapeConstraint(
            rank_equal, std::span<const TensorView>(inputs), std::span<const MutableTensorView>());
    ASSERT_TRUE(rank_equal_result.ok()) << rank_equal_result.status().ToString();
    EXPECT_EQ(*rank_equal_result, ShapeConstraintEvaluationResult::kViolated);
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

    EXPECT_EQ(EvaluateShapeConstraint(constraint, std::span<const SymbolicShape>(shared_inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(EvaluateShapeConstraint(constraint, std::span<const SymbolicShape>(conflicting_inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kViolated);
    EXPECT_EQ(EvaluateShapeConstraint(constraint, std::span<const SymbolicShape>(deferred_inputs), std::span<const SymbolicShape>()),
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

    EXPECT_EQ(EvaluateShapeConstraint(identical_volume, std::span<const SymbolicShape>(inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kSatisfied);
    EXPECT_EQ(EvaluateShapeConstraint(permuted_volume, std::span<const SymbolicShape>(inputs), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kDeferred);
}

TEST(ShapeConstraintEvaluator, EvaluatesSymbolicAndRuntimeDimPositive) {
    // Symbolic evaluation: static positive → satisfied, static zero → violated, symbolic → deferred.
    const ShapeConstraint positive_constraint{
            .condition = DimPositiveConstraint{.dim = InputDim(0, 0)},
            .error_context = "must be positive",
    };
    std::vector<SymbolicShape> symbolic_positive{SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(8)})};
    EXPECT_EQ(EvaluateShapeConstraint(positive_constraint, std::span<const SymbolicShape>(symbolic_positive), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kSatisfied);

    std::vector<SymbolicShape> symbolic_zero{SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(0)})};
    EXPECT_EQ(EvaluateShapeConstraint(positive_constraint, std::span<const SymbolicShape>(symbolic_zero), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kViolated);

    std::vector<SymbolicShape> symbolic_deferred{SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create()})};
    EXPECT_EQ(EvaluateShapeConstraint(positive_constraint, std::span<const SymbolicShape>(symbolic_deferred), std::span<const SymbolicShape>()),
              ShapeConstraintEvaluationResult::kDeferred);

    // Runtime evaluation: positive → satisfied, zero → violated.
    RuntimeTensorStorage runtime_positive{std::vector<int64_t>{8}};
    std::vector<TensorView> positive_inputs{runtime_positive.View()};
    const auto runtime_positive_result = EvaluateShapeConstraint(
            positive_constraint,
            std::span<const TensorView>(positive_inputs),
            std::span<const MutableTensorView>());
    ASSERT_TRUE(runtime_positive_result.ok()) << runtime_positive_result.status().ToString();
    EXPECT_EQ(*runtime_positive_result, ShapeConstraintEvaluationResult::kSatisfied);

    RuntimeTensorStorage runtime_zero{std::vector<int64_t>{0}};
    std::vector<TensorView> zero_inputs{runtime_zero.View()};
    const auto runtime_zero_result = EvaluateShapeConstraint(
            positive_constraint,
            std::span<const TensorView>(zero_inputs),
            std::span<const MutableTensorView>());
    ASSERT_TRUE(runtime_zero_result.ok()) << runtime_zero_result.status().ToString();
    EXPECT_EQ(*runtime_zero_result, ShapeConstraintEvaluationResult::kViolated);
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

TEST(ShapeConstraintEvaluator, RuntimeEvaluationRejectsOutOfRangeReferences) {
    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    std::vector<TensorView> inputs{input.View()};

    // Input tensor index beyond the bound spans.
    const ShapeConstraint missing_input{
            .condition = DimEqualConstraint{.lhs = InputDim(1, 0), .rhs = InputDim(0, 0)},
            .error_context = "unused",
    };
    const auto missing_input_result = EvaluateShapeConstraint(
            missing_input, std::span<const TensorView>(inputs),
            std::span<const MutableTensorView>());
    ASSERT_FALSE(missing_input_result.ok());
    EXPECT_EQ(missing_input_result.status().code(), StatusCode::kInvalidArgument);

    // Output tensor index beyond the bound spans (no outputs bound).
    const ShapeConstraint missing_output{
            .condition = DimEqualConstraint{.lhs = OutputDim(0, 0), .rhs = InputDim(0, 0)},
            .error_context = "unused",
    };
    const auto missing_output_result = EvaluateShapeConstraint(
            missing_output, std::span<const TensorView>(inputs),
            std::span<const MutableTensorView>());
    ASSERT_FALSE(missing_output_result.ok());
    EXPECT_EQ(missing_output_result.status().code(), StatusCode::kInvalidArgument);

    // Dimension index beyond the referenced tensor's rank.
    const ShapeConstraint missing_dim{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 3), .rhs = InputDim(0, 0)},
            .error_context = "unused",
    };
    const auto missing_dim_result = EvaluateShapeConstraint(
            missing_dim, std::span<const TensorView>(inputs),
            std::span<const MutableTensorView>());
    ASSERT_FALSE(missing_dim_result.ok());
    EXPECT_EQ(missing_dim_result.status().code(), StatusCode::kInvalidArgument);

    // Rank constraint referencing a missing tensor.
    const ShapeConstraint missing_rank{
            .condition = RankEqualConstraint{.port = InputPort(2), .target_rank = 2},
            .error_context = "unused",
    };
    const auto missing_rank_result = EvaluateShapeConstraint(
            missing_rank, std::span<const TensorView>(inputs),
            std::span<const MutableTensorView>());
    ASSERT_FALSE(missing_rank_result.ok());
    EXPECT_EQ(missing_rank_result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ShapeConstraintEvaluator, ValidateShapeConstraintsPropagatesReferenceErrors) {
    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    std::vector<TensorView> inputs{input.View()};
    const ShapeConstraint bad_reference{
            .condition = RankEqualConstraint{.port = InputPort(1), .target_rank = 2},
            .error_context = "unused",
    };

    const Status status = ValidateShapeConstraints(
            std::span<const ShapeConstraint>(&bad_reference, 1),
            std::span<const TensorView>(inputs),
            std::span<const MutableTensorView>());

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("missing input tensor"), std::string::npos);
}

TEST(ShapeConstraint, EqualityIgnoresErrorContext) {
    ShapeConstraint a{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "hidden size mismatch",
    };
    ShapeConstraint b{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "different message",
    };

    EXPECT_EQ(a, b);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

TEST(ShapeConstraint, OrderingIgnoresErrorContext) {
    ShapeConstraint a{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "hidden size mismatch",
    };
    ShapeConstraint b{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "different message",
    };

    EXPECT_FALSE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
}

TEST(ShapeConstraint, DifferentConditionsAreNotEqual) {
    ShapeConstraint dim_equal{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "dim equal",
    };
    ShapeConstraint rank_equal{
            .condition = RankEqualConstraint{.port = InputPort(0), .target_rank = 2},
            .error_context = "rank equal",
    };

    EXPECT_NE(dim_equal, rank_equal);
    EXPECT_FALSE(dim_equal == rank_equal);
}

TEST(ShapeConstraint, SetDeduplicatesByConditionOnly) {
    ShapeConstraint a{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "msg a",
    };
    ShapeConstraint b{
            .condition = DimEqualConstraint{.lhs = InputDim(0, 1), .rhs = InputDim(1, 0)},
            .error_context = "msg b",
    };
    ShapeConstraint c{
            .condition = DimBroadcastableConstraint{.lhs = InputDim(0, 0), .rhs = InputDim(1, 0)},
            .error_context = "msg c",
    };

    std::set<ShapeConstraint> s;
    s.insert(a);
    s.insert(b);
    s.insert(c);

    EXPECT_EQ(s.size(), 2U);
}

}// namespace
