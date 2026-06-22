/// \file
/// Bit-preserving casts between `float` and `uint32_t`.
///
/// Used by the half-precision and bfloat16 code paths to manipulate IEEE 754
/// binary32 bit patterns without invoking floating-point arithmetic. On GPU
/// device compilers (OpenCL, CUDA/HIP) we use the vendor intrinsics; on host
/// compilers we fall back to `std::bit_cast`.

#ifndef AETHERMIND_DTYPES_DETAIL_FLOATING_POINT_BITS_H
#define AETHERMIND_DTYPES_DETAIL_FLOATING_POINT_BITS_H

#include <bit>
#include <cstdint>

namespace aethermind {
namespace detail {

/// Reinterprets a 32-bit unsigned integer as an IEEE 754 binary32 `float`.
/// The bit pattern is preserved verbatim; no rounding or normalization occurs.
inline float fp32_from_bits(uint32_t value) {
#if defined(__OPENCL_VERSION__)
    return as_float(value);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float((unsigned int) value);
#else
    return std::bit_cast<float>(value);
#endif
}

/// Reinterprets an IEEE 754 binary32 `float` as a 32-bit unsigned integer.
/// The bit pattern is preserved verbatim; NaN payloads and signaling bits are
/// not canonicalized.
inline uint32_t fp32_to_bits(float value) {
#if defined(__OPENCL_VERSION__)
    return as_uint(value);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return (uint32_t) __float_as_uint(value);
#else
    return std::bit_cast<uint32_t>(value);
#endif
}

}// namespace detail
}// namespace aethermind

#endif// AETHERMIND_DTYPES_DETAIL_FLOATING_POINT_BITS_H
