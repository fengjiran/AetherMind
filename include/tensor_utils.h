//
// Created by 赵丹 on 25-6-24.
//

#ifndef AETHERMIND_TENSOR_UTILS_H
#define AETHERMIND_TENSOR_UTILS_H

namespace aethermind {

inline bool mul_overflow(int64_t a, int64_t b, int64_t* out) {
    return __builtin_mul_overflow(a, b, out);
}

inline bool mul_overflow(uint64_t a, uint64_t b, uint64_t* out) {
    return __builtin_mul_overflow(a, b, out);
}

template<typename Iter>
bool safe_multiply_u64(Iter first, Iter last, uint64_t* out) {
    uint64_t prod = 1;
    bool overflowed = false;
    while (first != last) {
        overflowed |= mul_overflow(prod, *first, &prod);
        ++first;
    }
    *out = prod;
    return overflowed;
}

template<typename Container>
bool safe_multiply_u64(const Container& c, uint64_t* out) {
    return safe_multiply_u64(c.begin(), c.end(), out);
}

}// namespace aethermind

#endif//AETHERMIND_TENSOR_UTILS_H
