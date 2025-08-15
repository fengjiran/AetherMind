//
// Created by 赵丹 on 25-6-24.
//

#ifndef AETHERMIND_TENSOR_UTILS_H
#define AETHERMIND_TENSOR_UTILS_H

#include <cxxabi.h>
#include <functional>
#include <memory>
#include <string>

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

// This function will demangle the mangled function name into a more human
// readable format, e.g. _Z1gv -> g().
// More information:
// https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/cxxabi.h
inline std::string demangle(const char* name) {
    int status = -1;
    std::unique_ptr<char, std::function<void(char*)>> demangled(
            abi::__cxa_demangle(
                    name,
                    /*__output_buffer=*/nullptr,
                    /*__length=*/nullptr,
                    &status),
            /*deleter=*/free);

    // Demangling may fail, for example when the name does not follow the
    // standard C++ (Itanium ABI) mangling scheme. This is the case for `main`
    // or `clone` for example, so the mangled name is a fine default.
    if (status == 0) {
        return demangled.get();
    }

    return name;
}

}// namespace aethermind

#endif//AETHERMIND_TENSOR_UTILS_H
