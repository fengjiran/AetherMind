#ifndef AETHERMIND_OPERATORS_OP_TYPE_H
#define AETHERMIND_OPERATORS_OP_TYPE_H

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind {

class OperatorName;

enum class OpType : uint16_t {
    kUnknown = 0,
    kEmbedding,
    kLinear,
    kMatMul,
    kRMSNorm,
    kRoPE,
    kAttentionPrefill,
    kAttentionDecode,
    kSiluMul,
    kSoftmax,
    kArgMax,
};

// OpType captures operator semantics only. Backend dispatch consumes this
// semantic contract when mapping an op to a concrete kernel implementation.
// Transition hook for moving from legacy OperatorName-based dispatch to the
// OpType-centered dispatch mainline. Batch 1 only freezes the contract.
AM_NODISCARD StatusOr<OpType> ToOpType(const OperatorName& op_name) noexcept;

AM_NODISCARD const char* ToString(OpType op_type) noexcept;

}// namespace aethermind

#endif
