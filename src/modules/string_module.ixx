//
// Created by richard on 1/23/26.
//
module;

#include <bit>

export module String;

namespace aethermind {

// constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
// constexpr auto kIsBigEndian = __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;

constexpr auto kIsLittleEndian = std::endian::native == std::endian::little;

}

