#ifndef AETHERMIND_UTILS_BITS_H
#define AETHERMIND_UTILS_BITS_H

/// @file
/// @brief Uninterpreted fixed-width bit-storage types.
///
/// These types preserve raw storage bits for tensor data types whose logical
/// width is smaller than, or equal to, their byte-aligned representation.

#include <cstdint>

namespace aethermind {

/// @brief Represents an uninterpreted 1-bit tensor type in byte-aligned storage.
///
/// The type does not define a bit-level interpretation. Its raw byte storage is
/// preserved without masking or conversion.
struct alignas(1) bits1x8 {
    using underlying = uint8_t;
    /// Raw byte storage for the value.
    underlying value_;

    bits1x8() = default;
    /// @brief Constructs the value from raw byte storage.
    /// @param value Raw byte value; no masking is applied.
    explicit bits1x8(underlying value) : value_(value) {}
};

/// @brief Represents an uninterpreted 2-bit tensor type in byte-aligned storage.
///
/// The type does not define a bit-level interpretation. Its raw byte storage is
/// preserved without masking or conversion.
struct alignas(1) bits2x4 {
    using underlying = uint8_t;
    /// Raw byte storage for the value.
    underlying value_;

    bits2x4() = default;
    /// @brief Constructs the value from raw byte storage.
    /// @param value Raw byte value; no masking is applied.
    explicit bits2x4(underlying value) : value_(value) {}
};

/// @brief Represents an uninterpreted 4-bit tensor type in byte-aligned storage.
///
/// The type does not define a bit-level interpretation. Its raw byte storage is
/// preserved without masking or conversion.
struct alignas(1) bits4x2 {
    using underlying = uint8_t;
    /// Raw byte storage for the value.
    underlying value_;

    bits4x2() = default;
    /// @brief Constructs the value from raw byte storage.
    /// @param value Raw byte value; no masking is applied.
    explicit bits4x2(underlying value) : value_(value) {}
};

/// @brief Represents an uninterpreted 8-bit tensor type.
///
/// The type does not define a semantic interpretation for its raw byte value.
struct alignas(1) bits8 {
    using underlying = uint8_t;
    /// Raw byte storage for the value.
    underlying value_;

    bits8() = default;
    /// @brief Constructs the value from raw byte storage.
    /// @param value Raw byte value.
    explicit bits8(underlying value) : value_(value) {}
};

/// @brief Represents an uninterpreted 16-bit tensor type.
///
/// The type does not define a semantic interpretation for its raw 16-bit value.
struct alignas(2) bits16 {
    using underlying = uint16_t;
    /// Raw 16-bit storage for the value.
    underlying value_;

    bits16() = default;
    /// @brief Constructs the value from raw storage.
    /// @param value Raw 16-bit value.
    explicit bits16(underlying value) : value_(value) {}
};

}// namespace aethermind

#endif// AETHERMIND_UTILS_BITS_H
