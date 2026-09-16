#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_PACKED_WEIGHT_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_PACKED_WEIGHT_UTILS_H

/// @file packed_weight_utils.h
/// @brief Shared packed-weight validation for CPU kernels that consume
///        identity-packed artifacts.
///
/// Hosts the binding-time validation core shared by the packed-only weight
/// kernels (QkvLinear, GateUpLinear, ...): checks that a PackedWeightView
/// carries the expected logical Float32 weight shape and the canonical CPU
/// identity-packing recipe with enough storage for the logical weight.
/// The implementation lives in packed_weight_utils.cpp.

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/status.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Validates a packed weight artifact against the identity-packing
///        contract for the expected logical Float32 shape.
///
/// Requires exact logical metadata (rank and every dimension), the canonical
/// CPU identity-packing recipe with 64-byte alignment, matching storage
/// alignment, and enough storage for every Float32 logical element. Shape
/// products are checked before converting them to bytes. Used by the
/// binding-time KernelParamsBuilders of the packed-only weight kernels;
/// callers derive operand pointers from the artifact's logical ranges after
/// this check passes.
///
/// @param packed Packed weight artifact exposed to the params builder.
/// @param expected_shape Expected logical shape; the rank and every dimension
///        are compared exactly and the element/byte products are computed
///        with overflow checks.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return Ok when the artifact satisfies the identity-packing contract,
///         InvalidArgument/Overflow otherwise.
Status ValidateIdentityPackedWeight(const PackedWeightView& packed,
                                    std::span<const int64_t> expected_shape,
                                    std::string_view kernel_name) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_PACKED_WEIGHT_UTILS_H
