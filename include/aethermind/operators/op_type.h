#ifndef AETHERMIND_OPERATORS_OP_TYPE_H
#define AETHERMIND_OPERATORS_OP_TYPE_H

#include "macros.h"

#include <cstdint>
#include <functional>

namespace aethermind {

/// Type tag for operator dispatch and kernel resolution.
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
    kMatMul,
    kRoPE,
    kAttention,
    kSilu,
    kSiluMul,
    kElementwiseMul,
    kAdd,
    kSoftmax,
    kArgmax,
};

/// Returns a human-readable name for the given operator type.
///
/// Returns "Unknown" for `kUnknown` and for any out-of-range value
/// (the function is safe to call with arbitrary uint16_t casts).
AM_NODISCARD const char* ToString(OpType op_type) noexcept;

}// namespace aethermind

template<>
struct std::hash<aethermind::OpType> {
    size_t operator()(aethermind::OpType op) const noexcept {
        return std::hash<uint16_t>{}(static_cast<uint16_t>(op));
    }
};

#endif
