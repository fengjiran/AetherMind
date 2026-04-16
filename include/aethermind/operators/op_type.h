#ifndef AETHERMIND_OPERATORS_OP_TYPE_H
#define AETHERMIND_OPERATORS_OP_TYPE_H

#include "macros.h"

#include <cstdint>

namespace aethermind {

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

AM_NODISCARD const char* ToString(OpType op_type) noexcept;

}// namespace aethermind

#endif
