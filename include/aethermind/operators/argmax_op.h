#ifndef AETHERMIND_OPERATORS_ARGMAX_OP_H
#define AETHERMIND_OPERATORS_ARGMAX_OP_H

/// @file argmax_op.h
/// @brief Shared dtype contract for Argmax semantic inference.

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

#include "aethermind/dtypes/data_type.h"

namespace aethermind {

/// @brief Dtypes accepted by Argmax semantic inference.
///
/// All Argmax validation sites must reference this set instead of maintaining
/// private copies.
inline const std::array<DataType, 3> kArgmaxSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether Argmax semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kArgmaxSupportedDTypes`.
inline bool IsArgmaxSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kArgmaxSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for Argmax.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeArgmaxUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
