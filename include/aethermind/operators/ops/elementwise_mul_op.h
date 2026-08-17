#ifndef AETHERMIND_OPERATORS_OPS_ELEMENTWISE_MUL_OP_H
#define AETHERMIND_OPERATORS_OPS_ELEMENTWISE_MUL_OP_H

/// @file elementwise_mul_op.h
/// @brief Elementwise multiplication semantic contract.

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"

namespace aethermind {

/// @brief Dtypes accepted by ElementwiseMul semantic inference.
///
/// All ElementwiseMul validation sites must reference this set instead of
/// maintaining private copies.
inline const std::array<DataType, 3> kElementwiseMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether ElementwiseMul semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kElementwiseMulSupportedDTypes`.
inline bool IsElementwiseMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kElementwiseMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for ElementwiseMul.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeElementwiseMulUnsupportedDTypeMessage(
        std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
