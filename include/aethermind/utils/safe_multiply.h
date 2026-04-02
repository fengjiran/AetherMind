//
// Created by richard on 4/2/26.
//

#ifndef AETHERMIND_UTILS_SAFE_MULTIPLY_H
#define AETHERMIND_UTILS_SAFE_MULTIPLY_H

#include "utils/logging.h"

#include <cstdint>
#include <type_traits>

namespace aethermind {

inline bool mul_overflow(int64_t a, int64_t b, int64_t* out) noexcept {
    return __builtin_mul_overflow(a, b, out);
}

inline bool mul_overflow(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    return __builtin_mul_overflow(a, b, out);
}

/// Multiplies a range of non-negative integer values into `*out`.
///
/// Requirements:
/// - `out` must not be null.
/// - every element in [first, last) must be non-negative and convertible to uint64_t.
///
/// Returns:
/// - true if any intermediate multiplication overflowed uint64_t.
/// - false otherwise.
///
/// Note:
/// - when true is returned, `*out` contains the last intermediate value and must
///   not be treated as a valid final product.
template<typename Iter>
bool safe_multiply_u64(Iter first, Iter last, uint64_t* out) noexcept {
    AM_DCHECK(out != nullptr);

    uint64_t prod = 1;
    bool overflowed = false;
    while (first != last) {
        using value_type = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
        if constexpr (std::is_signed_v<value_type>) {
            AM_DCHECK(*first >= 0);
        }
        overflowed |= mul_overflow(prod, static_cast<uint64_t>(*first), &prod);
        ++first;
    }
    *out = prod;
    return overflowed;
}

template<typename Container>
bool safe_multiply_u64(const Container& c, uint64_t* out) noexcept {
    return safe_multiply_u64(c.begin(), c.end(), out);
}

}// namespace aethermind


#endif// AETHERMIND_UTILS_SAFE_MULTIPLY_H
