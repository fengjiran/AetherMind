#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_PACKED_WEIGHT_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_PACKED_WEIGHT_UTILS_H

/// @file packed_weight_utils.h
/// @brief Shared packed-weight validation for CPU kernels that consume
///        identity-packed artifacts.
///
/// Hosts the binding-time validation core shared by the packed-only weight
/// kernels (QkvLinear, GateUpLinear, ...): checks that a PackedWeightView
/// carries the expected logical Float32 weight shape and the canonical
/// "cpu_identity" packing recipe with enough storage for the logical weight.
/// The implementation lives in packed_weight_utils.cpp.

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/status.h"

#include <cstdint>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Validates a packed weight artifact against the identity-packing
///        contract.
///
/// Requires a Float32 logical weight of shape `[total_out_features,
/// in_features]`, the canonical "cpu_identity" recipe with 64-byte
/// alignment, matching storage alignment, and storage whose byte size covers
/// the logical weight. Used by the binding-time KernelParamsBuilders of the
/// packed-only weight kernels; callers derive per-split weight pointers from
/// the artifact's logical row ranges after this check passes.
///
/// @param packed Packed weight artifact exposed to the params builder.
/// @param total_out_features Expected logical row count (sum of the fused
///        output splits, e.g. Q+K+V for QkvLinear).
/// @param in_features Expected logical column count.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return Ok when the artifact satisfies the identity-packing contract,
///         InvalidArgument/Overflow otherwise.
Status ValidateIdentityPackedWeight(const PackedWeightView& packed,
                                    int64_t total_out_features,
                                    int64_t in_features,
                                    std::string_view kernel_name) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_PACKED_WEIGHT_UTILS_H
