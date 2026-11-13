#ifndef AETHERMIND_OPERATORS_OPS_CAST_OP_H
#define AETHERMIND_OPERATORS_OPS_CAST_OP_H

/// @file cast_op.h
/// @brief Storage-format Cast semantics.
///
/// Cast performs a typed graph transition between FP32 and the two 16-bit
/// floating formats. It is a storage-format operation, not a general numeric
/// converter: the supported set is fixed and every narrowing uses the frozen
/// RNE/gradual-underflow/quiet-NaN conversion contract.

#include "aethermind/base/macros.h"
#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <array>
#include <string>

namespace aethermind {

/// @brief Dtypes accepted as Cast source and target.
inline const std::array kCastSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether Cast accepts `dtype`.
AM_NODISCARD inline bool IsCastSupportedDType(const DataType& dtype) noexcept {
    return dtype.IsScalar() &&
           std::ranges::any_of(kCastSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds the consistent unsupported-dtype diagnostic for Cast.
AM_NODISCARD inline std::string MakeCastUnsupportedDTypeMessage(std::string_view context) {
    return std::string(context) + " supports only scalar Float32, Float16, or BFloat16";
}

} // namespace aethermind

#endif // AETHERMIND_OPERATORS_OPS_CAST_OP_H
