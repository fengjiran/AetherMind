#ifndef AETHERMIND_OPERATORS_OP_TYPE_H
#define AETHERMIND_OPERATORS_OP_TYPE_H

/// @file op_type.h
/// @brief Stable type tags used by operator semantic dispatch.

#include "aethermind/base/macros.h"

#include <cstdint>
#include <functional>

namespace aethermind {

/// @brief Type tag for operator dispatch and kernel resolution.
///
/// Each operator in the execution graph is identified by its OpType.
/// The type is used by OperatorRegistry for factory lookup, by
/// KernelRegistry for kernel selection, and by ExecutionPlanBuilder
/// for constructing execution steps.
///
/// `kUnknown` is reserved as an invalid sentinel.
enum class OpType : uint16_t {
    kUnknown = 0,
    kEmbedding,
    kRmsNorm,
    kLinear,
    kQkvLinear,
    kMatMul,
    kRoPE,
    kAttention,
    kSilu,
    kSiluMul,
    kElementwiseMul,
    kKVCacheUpdate,
    kAdd,
    kAddRmsNorm,
    kSoftmax,
    kArgmax,
    kReshape,
    kPermute,
    kReorder,
};

/// @brief Returns a human-readable name for an operator type.
///
/// Returns "Unknown" for `kUnknown` and for any out-of-range value
/// (the function is safe to call with arbitrary uint16_t casts).
///
/// @param op_type Operator type to format.
/// @return Null-terminated static string naming the operator type.
AM_NODISCARD const char* ToString(OpType op_type) noexcept;

}// namespace aethermind

template<>
struct std::hash<aethermind::OpType> {
    size_t operator()(aethermind::OpType op) const noexcept {
        return std::hash<uint16_t>{}(static_cast<uint16_t>(op));
    }
};

#endif
