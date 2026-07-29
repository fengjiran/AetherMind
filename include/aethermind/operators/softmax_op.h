#ifndef AETHERMIND_OPERATORS_SOFTMAX_OP_H
#define AETHERMIND_OPERATORS_SOFTMAX_OP_H

#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {

/// Single source of truth for the dtype set supported by the Softmax operator.
/// All Softmax-related validation (semantic analysis in InferSoftmax, future
/// CPU kernel dispatch) must reference these definitions instead of maintaining
/// private copies. The Phase 1 CPU kernel currently implements only Float32;
/// the semantic layer accepts the full set below.
inline const std::array<DataType, 3> kSoftmaxSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// Returns true if `dtype` is in `kSoftmaxSupportedDTypes`. Used by operator-
/// level validation to keep the dtype check in one place. Backend kernel
/// dispatch must reference this same set when adding new dtype paths.
inline bool IsSoftmaxSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSoftmaxSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported dtype" error message for Softmax-related
/// validation points. `context` is the caller name (e.g. "Softmax",
/// "CpuSoftmaxKernel") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
inline std::string MakeSoftmaxUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
