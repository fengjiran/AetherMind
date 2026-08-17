#ifndef AETHERMIND_OPERATORS_OPS_SILU_MUL_OP_H
#define AETHERMIND_OPERATORS_OPS_SILU_MUL_OP_H

/// @file silu_mul_op.h
/// @brief Fused SiLU-multiply semantic contract.

#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief Dtypes accepted by both SiluMul inputs.
///
/// Both inputs must use the same dtype from this set; no implicit conversion
/// policy is declared.
inline const std::array<DataType, 5> kSiluMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// @brief Checks whether SiluMul semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kSiluMulSupportedDTypes`.
inline bool IsSiluMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSiluMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for SiluMul.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeSiluMulUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

}// namespace aethermind

#endif
