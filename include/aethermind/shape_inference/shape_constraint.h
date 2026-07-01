#ifndef AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_H
#define AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace aethermind {

/// Identifies whether a port is an input or an output of an operator.
///
/// 'Port' is the standard graph theory term for node interfaces.
enum class TensorPortType : std::uint8_t {
    kInput,
    kOutput
};

/// Result of evaluating a shape constraint against currently known shape facts.
///
/// Evaluators should return kDeferred when a constraint is neither proven nor
/// disproven because rank or dimension values are still symbolic or runtime-only.
enum class ShapeConstraintEvaluationResult : std::uint8_t {
    kSatisfied,
    kViolated,
    kDeferred
};

/// Uniquely identifies a specific input or output port of an operator.
struct TensorPort {
    TensorPortType direction{TensorPortType::kInput};
    size_t tensor_idx{0};

    auto operator<=>(const TensorPort&) const noexcept = default;
};

/// Locates a specific dimension within a specific tensor port.
struct DimLocator {
    TensorPort tensor_port{};
    size_t dim_index{0};

    auto operator<=>(const DimLocator&) const noexcept = default;
};

/// Constrains two specific dimensions to be exactly equal.
struct DimEqualConstraint {
    DimLocator lhs;
    DimLocator rhs;

    auto operator<=>(const DimEqualConstraint&) const noexcept = default;
};

/// Constrains two dimensions to be broadcastable (a == b || a == 1 || b == 1).
struct DimBroadcastableConstraint {
    DimLocator lhs;
    DimLocator rhs;

    auto operator<=>(const DimBroadcastableConstraint&) const noexcept = default;
};

/// Constrains the product of two sets of dimensions to be equal (e.g., Reshape).
/// An empty dimension list represents scalar volume 1.
struct VolumeEqualConstraint {
    std::vector<DimLocator> lhs_dims;
    std::vector<DimLocator> rhs_dims;

    auto operator<=>(const VolumeEqualConstraint&) const = default;
};

/// Constrains the rank of a tensor to an exact value.
struct RankEqualConstraint {
    TensorPort port;
    size_t target_rank;

    auto operator<=>(const RankEqualConstraint&) const noexcept = default;
};

/// Constrains the rank of a tensor to a minimum value.
struct RankAtLeastConstraint {
    TensorPort port;
    size_t min_rank;

    auto operator<=>(const RankAtLeastConstraint&) const noexcept = default;
};

using ConstraintVariant = std::variant<
        DimEqualConstraint,
        DimBroadcastableConstraint,
        VolumeEqualConstraint,
        RankEqualConstraint,
        RankAtLeastConstraint>;

/// The unified shape constraint object emitted by AOT inference.
///
/// ShapeConstraint is a data-only contract. Evaluation is performed by a
/// separate checker that returns ShapeConstraintEvaluationResult: satisfied,
/// violated, or deferred until concrete runtime shapes are available.
struct ShapeConstraint {
    ConstraintVariant condition;
    std::string error_context;

    friend bool operator==(const ShapeConstraint& lhs, const ShapeConstraint& rhs) noexcept(
            noexcept(lhs.condition == rhs.condition)) {
        return lhs.condition == rhs.condition;
    }

    friend auto operator<=>(const ShapeConstraint& lhs, const ShapeConstraint& rhs) noexcept(
            noexcept(lhs.condition <=> rhs.condition)) {
        return lhs.condition <=> rhs.condition;
    }
};

}// namespace aethermind

#endif// AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_H
