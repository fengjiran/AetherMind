#ifndef AETHERMIND_SHAPE_CONSTRAINT_H
#define AETHERMIND_SHAPE_CONSTRAINT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace aethermind {

///  Identifies whether a port is an input or an output of an operator.
///
/// 'Port' is the standard graph theory term for node interfaces.
enum class PortType : std::uint8_t {
    kInput,
    kOutput
};

/// Uniquely identifies a specific input or output port of an operator.
struct TensorPort {
    PortType direction{PortType::kInput};
    size_t tensor_idx{0};

    bool operator==(const TensorPort&) const noexcept = default;
};

/// Locates a specific dimension within a specific tensor port.
struct DimLocator {
    TensorPort tensor_port{};
    size_t dim_index{0};

    bool operator==(const DimLocator&) const noexcept = default;
};

/// Constrains two specific dimensions to be exactly equal.
struct DimEqualConstraint {
    DimLocator lhs;
    DimLocator rhs;

    bool operator==(const DimEqualConstraint&) const noexcept = default;
};

/// Constrains two dimensions to be broadcastable (a == b || a == 1 || b == 1).
struct DimBroadcastableConstraint {
    DimLocator lhs;
    DimLocator rhs;

    bool operator==(const DimBroadcastableConstraint&) const noexcept = default;
};

/// Constrains the product of two sets of dimensions to be equal (e.g., Reshape).
/// An empty dimension list represents scalar volume 1.
struct VolumeEqualConstraint {
    std::vector<DimLocator> lhs_dims;
    std::vector<DimLocator> rhs_dims;

    bool operator==(const VolumeEqualConstraint&) const noexcept = default;
};

/// Constrains the rank of a tensor to an exact value.
struct RankEqualConstraint {
    TensorPort port;
    size_t target_rank;

    bool operator==(const RankEqualConstraint&) const noexcept = default;
};

/// Constrains the rank of a tensor to a minimum value.
struct RankAtLeastConstraint {
    TensorPort port;
    size_t min_rank;

    bool operator==(const RankAtLeastConstraint&) const noexcept = default;
};

using ConstraintVariant = std::variant<
        DimEqualConstraint,
        DimBroadcastableConstraint,
        VolumeEqualConstraint,
        RankEqualConstraint,
        RankAtLeastConstraint>;

/// The unified shape constraint object emitted by AOT inference.
struct ShapeConstraint {
    ConstraintVariant condition;
    std::string error_context;

    bool operator==(const ShapeConstraint&) const noexcept = default;
};

using ShapeConstraintList = std::vector<ShapeConstraint>;

}// namespace aethermind

#endif// AETHERMIND_SHAPE_CONSTRAINT_H
