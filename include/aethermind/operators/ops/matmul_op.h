#ifndef AETHERMIND_OPERATORS_OPS_MATMUL_OP_H
#define AETHERMIND_OPERATORS_OPS_MATMUL_OP_H

/// @file matmul_op.h
/// @brief Batched matrix-multiplication semantic contract.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief Dtypes accepted independently for MatMul inputs.
///
/// All MatMul validation sites must reference this set instead of maintaining
/// private copies. Output dtype follows the lhs dtype.
inline const std::array<DataType, 6> kMatMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
        DataType::Int(8),
};

/// @brief Checks whether MatMul semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kMatMulSupportedDTypes`.
inline bool IsMatMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kMatMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for MatMul.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeMatMulUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, "
           "float8_e4m3fn, float8_e5m2, and int8 dtypes";
    return msg;
}

/// @brief Semantic operator for batched matrix multiplication.
///
/// Computes `output = lhs @ effective_rhs` where:
/// - `effective_rhs = rhs` when `transpose_rhs == false` (rhs shape [..., K, N])
/// - `effective_rhs = rhs^T` when `transpose_rhs == true`  (rhs shape [..., N, K])
///
/// Both inputs must have rank >= 2. Trailing two axes are the matrix axes;
/// leading axes are batch axes broadcast according to NumPy semantics.
/// Output shape: `broadcast(lhs_batch, rhs_batch) + [M, N]` where
/// `M = lhs.shape[-2]` and `N = effective_rhs.shape[-1]`.
///
} // namespace aethermind

#endif // AETHERMIND_OPERATORS_OPS_MATMUL_OP_H
