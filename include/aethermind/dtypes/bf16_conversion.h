#ifndef AETHERMIND_DTYPES_BF16_CONVERSION_H
#define AETHERMIND_DTYPES_BF16_CONVERSION_H

/// @file
/// Host-memory bfloat16/binary32 batch conversion contracts.
#include "aethermind/base/status.h"

#include <cstddef>
#include <span>

namespace aethermind {

/// @brief Converts a prevalidated batch without allocation or CPU detection.
/// @param src Readable bytes containing count source elements.
/// @param dst Writable bytes for count destination elements.
/// @param count Number of elements; zero permits null pointers.
/// @note Ranges must be disjoint and large enough, with representable byte
///       counts. Arbitrary byte alignment is supported. Both directions preserve
///       signed zero and NaN sign/representable payload, forcing the quiet bit.
///       Narrowing uses fixed round-to-nearest-even with gradual underflow.
///       Floating-point control and exception flags remain unchanged.
using Bf16ConversionFunc = void (*)(const std::byte* src,
                                    std::byte* dst,
                                    size_t count) noexcept;

/// @brief Widens bfloat16 bytes exactly, quieting NaNs in the integer domain.
/// @param src Source bfloat16 bytes; see Bf16ConversionFunc preconditions.
/// @param dst Destination binary32 bytes.
/// @param count Number of elements.
/// @note Matches BFloat16::operator float(), including signaling-NaN quieting.
void ConvertBf16ToFp32Scalar(const std::byte* src,
                             std::byte* dst,
                             size_t count) noexcept;

/// @brief Narrows binary32 bytes using fixed RNE and gradual underflow.
/// @param src Source binary32 bytes; see Bf16ConversionFunc preconditions.
/// @param dst Destination bfloat16 bytes.
/// @param count Number of elements.
void ConvertFp32ToBf16Scalar(const std::byte* src,
                             std::byte* dst,
                             size_t count) noexcept;

/// @brief Prepared host conversion functions, independent of backend lifetime.
/// @note Functions must be non-null and obey Bf16ConversionFunc's contract.
///       Selection belongs to the backend; this value carries no ISA policy.
struct Bf16ConversionFuncs {
    Bf16ConversionFunc to_fp32 = ConvertBf16ToFp32Scalar;
    Bf16ConversionFunc from_fp32 = ConvertFp32ToBf16Scalar;
};

/// @brief Validates byte ranges and widens a contiguous bfloat16 batch.
/// @param src Complete bfloat16 elements, with arbitrary byte alignment.
/// @param dst Exactly twice source.size() bytes, disjoint from source.
/// @param functions Prepared conversion functions; defaults to portable scalar.
/// @return Ok, InvalidArgument for malformed ranges/functions, or Overflow for
///         an unrepresentable destination byte count. Errors precede writes.
/// @note A successful call allocates no storage and leaves the floating-point
///       environment unchanged. Empty ranges are accepted.
Status ConvertBf16ToFp32(std::span<const std::byte> src,
                         std::span<std::byte> dst,
                         const Bf16ConversionFuncs& functions = {});

/// @brief Validates byte ranges and narrows with fixed RNE and gradual underflow.
/// @param src Complete binary32 elements, with arbitrary byte alignment.
/// @param dst Exactly half source.size() bytes, disjoint from source.
/// @param functions Prepared conversion functions; defaults to portable scalar.
/// @return Ok, or InvalidArgument before any writes for malformed ranges/functions.
/// @note A successful call allocates no storage and leaves the floating-point
///       environment unchanged. Empty ranges are accepted.
Status ConvertFp32ToBf16(std::span<const std::byte> src,
                         std::span<std::byte> dst,
                         const Bf16ConversionFuncs& functions = {});

} // namespace aethermind
#endif
