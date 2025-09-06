//
// Created by richard on 9/6/25.
//

#ifndef AETHERMIND_FLOATING_POINT_UTILS_H
#define AETHERMIND_FLOATING_POINT_UTILS_H

#include <bit>
#include <cstdint>

namespace aethermind {
namespace details {

inline float fp32_from_bits(uint32_t value) {
#if defined(__OPENCL_VERSION__)
    return as_float(value);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float((unsigned int)value);
#elif defined(__INTEL_COMPILER)
    return _castu32_f32(value);
#else
    return std::bit_cast<float>(value);
#endif
}

inline uint32_t fp32_to_bits(float value) {
#if defined(__OPENCL_VERSION__)
    return as_uint(value);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return (uint32_t)__float_as_uint(value);
#elif defined(__INTEL_COMPILER)
    return _castf32_u32(value);
#else
    return std::bit_cast<uint32_t>(value);
#endif
}

}
}

#endif//AETHERMIND_FLOATING_POINT_UTILS_H
