#ifndef AETHERMIND_OPERATORS_OPS_SOFTMAX_OP_H
#define AETHERMIND_OPERATORS_OPS_SOFTMAX_OP_H

/// @file softmax_op.h
/// @brief Shared dtype contract for Softmax semantic inference.

#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief Dtypes accepted by Softmax semantic inference.
///
/// All Softmax validation sites must reference this set instead of maintaining
/// private copies.
inline const std::array<DataType, 3> kSoftmaxSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether Softmax semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kSoftmaxSupportedDTypes`.
inline bool IsSoftmaxSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSoftmaxSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for Softmax.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeSoftmaxUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
