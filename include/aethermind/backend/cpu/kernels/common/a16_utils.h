#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_A16_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_A16_UTILS_H

/// @file a16_utils.h
/// @brief Byte-safe scalar widening/narrowing helpers shared by A16 kernels.
///
/// A16 kernels read 16-bit activations that may be arbitrarily aligned (raw
/// model storage, packed artifacts, KV planes) and write 16-bit outputs. Every
/// access therefore copies through `std::memcpy`; no live `Half`/`BFloat16`
/// object is required at the storage address. Narrowing follows the frozen
/// conversion contract (RNE, gradual underflow, signed zero, overflow to
/// infinity) because it reuses the `Half`/`BFloat16` constructors.
///
/// Internal linkage keeps these helpers local to each translation unit.
/// In non-inlined builds, weak symbol coalescing could otherwise route a
/// baseline caller to a body compiled for an optional ISA.

#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/dtypes/half.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace aethermind::cpu::detail {

/// @brief True for the scalar 16-bit activation dtypes executed by A16 kernels.
static inline bool IsA16DType(const DataType& dtype) noexcept {
    return dtype == DataType::Float(16) || dtype == DataType::BFloat(16);
}

/// @brief Byte width of one scalar element of `dtype`.
static inline size_t A16ElementBytes(const DataType& dtype) noexcept {
    return static_cast<size_t>(dtype.nbytes());
}

/// @brief Widens one stored scalar to F32 without alignment assumptions.
///
/// Precondition: `dtype` is Float32, Float(16), or BFloat(16).
static inline float LoadA16Element(const std::byte* data, const DataType& dtype) noexcept {
    if (dtype == DataType::Float32()) {
        float value = 0.0F;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    uint16_t bits = 0;
    std::memcpy(&bits, data, sizeof(bits));
    if (dtype == DataType::Float(16)) {
        return Half(bits, Half::from_bits());
    }
    return BFloat16(bits, BFloat16::from_bits());
}

/// @brief Narrows one F32 value into a stored scalar, or stores it as F32.
///
/// Precondition: `dtype` is Float32, Float(16), or BFloat(16).
static inline void StoreA16Element(std::byte* data, float value, const DataType& dtype) noexcept {
    if (dtype == DataType::Float32()) {
        std::memcpy(data, &value, sizeof(value));
        return;
    }

    const uint16_t bits = dtype == DataType::Float(16)
                                  ? Half(value).bits()
                                  : BFloat16(value).x;
    std::memcpy(data, &bits, sizeof(bits));
}

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_A16_UTILS_H
