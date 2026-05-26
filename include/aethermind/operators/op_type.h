#ifndef AETHERMIND_OPERATORS_OP_TYPE_H
#define AETHERMIND_OPERATORS_OP_TYPE_H

#include "macros.h"

#include <cstdint>

namespace aethermind {

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

AM_NODISCARD const char* ToString(OpType op_type) noexcept;

}// namespace aethermind

#endif
