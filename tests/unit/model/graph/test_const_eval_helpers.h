#pragma once

#include "aethermind/model/graph/const_evaluator.h"

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace aethermind {
namespace test_utils {

// Unwraps MakeContiguousStrides or fatally aborts the test.
// Test shapes are crafted to avoid overflow; a failure here indicates
// a bug in the test fixture, not the code under test.
inline std::vector<int64_t> MakeContiguousStridesOrDie(std::span<const int64_t> shape) {
    auto result = MakeContiguousStrides(shape);
    AM_CHECK(result.ok(), "Test helper: MakeContiguousStrides failed: {}", result.status().ToString());
    return std::move(*result);
}

template<typename T>
inline std::vector<std::byte> BytesFromValues(std::vector<T> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template<typename T>
inline std::vector<T> ValuesFromBytes(const std::vector<std::byte>& bytes) {
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

inline std::vector<uint16_t> BFloat16Bits(const std::vector<BFloat16>& values) {
    std::vector<uint16_t> bits;
    bits.reserve(values.size());
    for (const BFloat16 value: values) {
        bits.push_back(value.x);
    }
    return bits;
}

inline std::vector<BFloat16> BFloat16Values(std::vector<uint16_t> bits) {
    std::vector<BFloat16> values;
    values.reserve(bits.size());
    for (const uint16_t value: bits) {
        values.emplace_back(value, BFloat16::from_bits());
    }
    return values;
}

}// namespace test_utils
}// namespace aethermind
