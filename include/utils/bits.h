//
// Created by 赵丹 on 2025/8/31.
//

#ifndef AETHERMIND_UTILS_BITS_H
#define AETHERMIND_UTILS_BITS_H

#include <cstdint>

namespace aethermind {

/**
 * bits1x8 is an uninterpreted dtype of a tensor with 1 bit (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits1x8 {
    using underlying = uint8_t;
    underlying value_;

    bits1x8() = default;
    explicit bits1x8(underlying value) : value_(value) {}
};

/**
 * bits2x4 is an uninterpreted dtype of a tensor with 2 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits2x4 {
    using underlying = uint8_t;
    underlying value_;

    bits2x4() = default;
    explicit bits2x4(underlying value) : value_(value) {}
};

/**
 * bits4x2 is an uninterpreted dtype of a tensor with 4 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits4x2 {
    using underlying = uint8_t;
    underlying value_;

    bits4x2() = default;
    explicit bits4x2(underlying value) : value_(value) {}
};

/**
 * bits8 is an uninterpreted dtype of a tensor with 8 bits, without any
 * semantics defined.
 */
struct alignas(1) bits8 {
    using underlying = uint8_t;
    underlying value_;

    bits8() = default;
    explicit bits8(underlying value) : value_(value) {}
};

/**
 * bits16 is an uninterpreted dtype of a tensor with 16 bits, without any
 * semantics defined.
 */
struct alignas(2) bits16 {
    using underlying = uint16_t;
    underlying value_;

    bits16() = default;
    explicit bits16(underlying value) : value_(value) {}
};

}// namespace aethermind

#endif//AETHERMIND_UTILS_BITS_H
