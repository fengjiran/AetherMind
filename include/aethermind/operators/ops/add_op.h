#ifndef AETHERMIND_OPERATORS_OPS_ADD_OP_H
#define AETHERMIND_OPERATORS_OPS_ADD_OP_H

/// @file add_op.h
/// @brief Elementwise addition semantic contract.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace aethermind {

/// @brief Dtypes accepted by Add semantic inference.
///
/// All Add-related validation (op params, CPU kernel dispatch, constant
/// folding) must reference these definitions instead of maintaining private
/// copies. The CPU Add registrations in add_entry.cpp must cover exactly
/// these dtypes; test_cpu_add_kernel.cpp asserts the count at compile time.
inline const std::array<DataType, 5> kAddSupportedDTypes = {
        DataType::Float32(),
        DataType::Double(),
        DataType::BFloat(16),
        DataType::Int(32),
        DataType::Int(64),
};

/// @brief Checks whether Add semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kAddSupportedDTypes`.
inline bool IsAddSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kAddSupportedDTypes, [&](const DataType& supported) {
        return dtype == supported;
    });
}

/// @brief Builds a consistent unsupported-dtype message for Add.
///
/// `context` is the caller name (e.g. "Add", "AddKernel", "Add constant
/// evaluator") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeAddUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float64, bfloat16, int32, and int64 tensors";
    return msg;
}

} // namespace aethermind

#endif
