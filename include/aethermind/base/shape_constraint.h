#ifndef AETHERMIND_SHAPE_CONSTRAINT_H
#define AETHERMIND_SHAPE_CONSTRAINT_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace aethermind {

/// @brief Identifies whether a port is an input or an output of an operator.
/// @note  'Port' is the standard graph theory term for node interfaces.
enum class PortType : std::uint8_t {
    kInput,
    kOutput,
};

/// @brief Uniquely identifies a specific input or output port of an operator.
struct TensorPort {
    PortType direction_{PortType::kInput};
    size_t tensor_idx_{0};

    bool operator==(const TensorPort&) const noexcept = default;
};

/// @brief Locates a specific dimension within a specific tensor port.
struct DimLocator {
    TensorPort tensor_{};
    size_t dim_index_{0};

    bool operator==(const DimLocator&) const noexcept = default;
};

/// @brief Runtime-verifiable relation emitted when symbolic inference is unresolved.
enum class ShapeConstraintKind : std::uint8_t {
    kDimEqual,        // lhs_dim == rhs_dim
    kDimBroadcastable,// lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1
    kVolumeEqual,     // product(lhs_dims) == product(rhs_dims) (e.g., for Reshape)
    kRankEqual,       // rank(tensor) == target_value
    kRankAtLeast      // rank(tensor) >= target_value
};

/// Shape relation produced by operator inference and checked after concrete runtime shapes are known.
struct ShapeConstraint {
    ShapeConstraintKind kind_{ShapeConstraintKind::kDimEqual};
    std::vector<DimLocator> lhs_dims_{};
    std::vector<DimLocator> rhs_dims_{};
    std::optional<TensorPort> tensor_{};
    std::optional<int64_t> value_{};
    std::string debug_context_{};
};

using ShapeConstraints = std::vector<ShapeConstraint>;

}// namespace aethermind

#endif// AETHERMIND_SHAPE_CONSTRAINT_H
