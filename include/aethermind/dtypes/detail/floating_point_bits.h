/// \file
/// Bit-preserving casts between `float` and `uint32_t`.
///
/// Used by the half-precision and bfloat16 code paths to manipulate IEEE 754
/// binary32 bit patterns without invoking floating-point arithmetic. Each
/// helper dispatches to the vendor-preferred intrinsic on OpenCL, CUDA/HIP,
/// and Intel ICC, and falls back to `std::bit_cast` elsewhere.

#ifndef AETHERMIND_DTYPES_DETAIL_FLOATING_POINT_BITS_H
#define AETHERMIND_DTYPES_DETAIL_FLOATING_POINT_BITS_H

#include <bit>
#include <cstdint>

namespace aethermind {
namespace details {

/// Reinterprets a 32-bit unsigned integer as an IEEE 754 binary32 `float`.
/// The bit pattern is preserved verbatim; no rounding or normalization occurs.
inline float fp32_from_bits(uint32_t value) {
#if defined(__OPENCL_VERSION__)
    return as_float(value);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float((unsigned int) value);
#elif defined(__INTEL_COMPILER)
    return _castu32_f32(value);
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
#elif defined(__INTEL_COMPILER)
    return _castf32_u32(value);
#else
    return std::bit_cast<uint32_t>(value);
#endif
}

}// namespace details
}// namespace aethermind

#endif// AETHERMIND_DTYPES_DETAIL_FLOATING_POINT_BITS_H
