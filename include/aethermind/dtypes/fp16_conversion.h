#ifndef AETHERMIND_DTYPES_FP16_CONVERSION_H
#define AETHERMIND_DTYPES_FP16_CONVERSION_H

/// @file
/// Host-memory binary16/binary32 batch conversion contracts.
#include "aethermind/base/status.h"

#include <cstddef>
#include <span>

namespace aethermind {

/// @brief Converts a prevalidated batch without allocation or CPU detection.
/// @param src Readable bytes containing count source elements.
/// @param dst Writable bytes for count destination elements.
/// @param count Number of elements; zero permits null pointers.
/// @note Ranges must be disjoint and large enough, with representable byte
///       counts. Arbitrary byte alignment is supported. Implementations must
///       match Half conversions bitwise, quiet NaNs, preserve signed zero,
///       and leave floating-point control and exception flags unchanged.
using Fp16ConversionFunc = void (*)(const std::byte* src,
                                    std::byte* dst,
                                    size_t count) noexcept;

/// @brief Widens binary16 bytes using portable integer conversion.
/// @param src Source binary16 bytes; see Fp16ConversionFunc preconditions.
/// @param dst Destination binary32 bytes.
/// @param count Number of elements.
void ConvertFp16ToFp32Scalar(const std::byte* src,
                             std::byte* dst,
                             size_t count) noexcept;

/// @brief Narrows binary32 bytes using fixed RNE and gradual underflow.
/// @param src Source binary32 bytes; see Fp16ConversionFunc preconditions.
/// @param dst Destination binary16 bytes.
/// @param count Number of elements.
void ConvertFp32ToFp16Scalar(const std::byte* src,
                             std::byte* dst,
                             size_t count) noexcept;

/// @brief Prepared host conversion functions, independent of backend lifetime.
/// @note Functions must be non-null and obey Fp16ConversionFunc's contract.
///       Selection belongs to the backend; this value carries no ISA policy.
struct Fp16ConversionFuncs {
    Fp16ConversionFunc to_fp32 = ConvertFp16ToFp32Scalar;
    Fp16ConversionFunc from_fp32 = ConvertFp32ToFp16Scalar;
};

/// @brief Validates byte ranges and widens a contiguous binary16 batch.
/// @param src Complete binary16 elements, with arbitrary byte alignment.
/// @param dst Exactly twice source.size() bytes, disjoint from source.
/// @param functions Prepared conversion functions; defaults to portable scalar.
/// @return Ok, InvalidArgument for malformed ranges/functions, or Overflow for
///         an unrepresentable destination byte count. Errors precede writes.
/// @note A successful call allocates no storage. Empty ranges are accepted.
Status ConvertFp16ToFp32(std::span<const std::byte> src,
                         std::span<std::byte> dst,
                         const Fp16ConversionFuncs& functions = {});

/// @brief Validates byte ranges and narrows with fixed RNE and gradual underflow.
/// @param src Complete binary32 elements, with arbitrary byte alignment.
/// @param dst Exactly half source.size() bytes, disjoint from source.
/// @param functions Prepared conversion functions; defaults to portable scalar.
/// @return Ok, or InvalidArgument before any writes for malformed ranges/functions.
/// @note A successful call allocates no storage and leaves the floating-point
///       environment unchanged. Empty ranges are accepted.
Status ConvertFp32ToFp16(std::span<const std::byte> src,
                         std::span<std::byte> dst,
                         const Fp16ConversionFuncs& functions = {});

} // namespace aethermind
#endif
