//
// Created by richard on 10/30/25.
//

#ifndef AETHERMIND_UTILS_HASH_H
#define AETHERMIND_UTILS_HASH_H

#include "container/string.h"
#include "utils/xxh3.h"

#include <complex>
#include <cstddef>
#include <functional>
#include <glog/logging.h>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace aethermind {

inline uint32_t twang_mix32(uint32_t key) noexcept {
    key = ~key + (key << 15);// key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;// key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return key;
}

inline uint64_t twang_mix64(uint64_t key) noexcept {
    key = (~key) + (key << 21);// key *= (1 << 21) - 1; key -= 1;
    key = key ^ (key >> 24);
    key = key + (key << 3) + (key << 8);// key *= 1 + (1 << 3) + (1 << 8)
    key = key ^ (key >> 14);
    key = key + (key << 2) + (key << 4);// key *= 1 + (1 << 2) + (1 << 4)
    key = key ^ (key >> 28);
    key = key + (key << 31);// key *= 1 + (1 << 31)
    return key;
}

#ifdef CPP20
template<typename T>
    requires std::convertible_to<T, size_t>
#else
template<typename T, std::enable_if_t<std::is_convertible_v<T, size_t>>* = nullptr>
#endif
size_t hash_combine(size_t seed, const T& hash_value) {
    return seed ^ (hash_value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}

namespace details {

#ifdef CPP20

template<typename T>
concept has_std_hash = requires(T v) {
    typename std::hash<T>;
    { std::hash<T>()(v) } -> std::same_as<size_t>;
};

template<typename T>
    requires requires(T v) {
        T::hash;
        { T::hash(v) } -> std::same_as<size_t>;
    }
size_t dispatch_hash(const T& o) {
    return T::hash(o);
}

template<typename T>
    requires has_std_hash<T> && std::is_enum_v<T>
size_t dispatch_hash(const T& v) {
    using R = std::underlying_type_t<T>;
    return std::hash<R>()(static_cast<R>(v));
}

template<typename T>
    requires has_std_hash<T>
size_t dispatch_hash(const T& v) {
    return std::hash<T>()(v);
}

template<typename T>
size_t simple_get_hash(const T& v);

#else

template<typename T, typename V>
using type_if_not_enum = std::enable_if_t<!std::is_enum_v<T>, V>;

// Use SFINAE to dispatch to std::hash if possible, cast enum types to int
// automatically, and fall back to T::hash otherwise. NOTE: C++14 added support
// for hashing enum types to the standard, and some compilers implement it even
// when C++14 flags aren't specified. This is why we have to disable this
// overload if T is an enum type (and use the one below in this case).
template<typename T>
auto dispatch_hash(const T& o) -> decltype(std::hash<T>()(o), type_if_not_enum<T, size_t>()) {
    return std::hash<T>()(o);
}

template<typename T>
std::enable_if_t<std::is_enum_v<T>, size_t> dispatch_hash(const T& o) {
    using R = std::underlying_type_t<T>;
    return std::hash<R>()(static_cast<R>(o));
}

template<typename T>
auto dispatch_hash(const T& o) -> decltype(T::hash(o), size_t()) {
    return T::hash(o);
}

#endif

}// namespace details

// Hasher struct
template<typename T>
struct hash {
    size_t operator()(const T& v) const {
        return details::dispatch_hash(v);
    }
};

template<>
struct hash<uint32_t> {
    size_t operator()(uint32_t x) const noexcept {
        return twang_mix32(x);
    }
};

template<>
struct hash<int> {
    size_t operator()(int x) const noexcept {
        return twang_mix32(static_cast<uint32_t>(x));
    }
};

template<>
struct hash<uint64_t> {
    size_t operator()(uint64_t x) const noexcept {
        return twang_mix64(x);
    }
};

template<>
struct hash<long long> {
    size_t operator()(long long x) const noexcept {
        return twang_mix64(static_cast<uint64_t>(x));
    }
};

// Specialization for std::tuple
template<typename... Types>
struct hash<std::tuple<Types...>> {
    size_t operator()(const std::tuple<Types...>& t) const {
        size_t seed = 0;
        auto func = [&seed](auto&&... x) {
            ((seed = hash_combine(seed, details::simple_get_hash(x))), ...);
        };
        std::apply(func, t);
        return seed;
    }
};

template<typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
    size_t operator()(const std::pair<T1, T2>& pair) const {
        std::tuple<T1, T2> tuple = std::make_tuple(pair.first, pair.second);
        return details::simple_get_hash(tuple);
    }
};

// Specialization for std::vector
template<typename T>
struct hash<std::vector<T>> {
    size_t operator()(const std::vector<T>& v) const {
        size_t seed = 0;
        for (const auto& elem: v) {
            seed = hash_combine(seed, details::simple_get_hash(elem));
        }
        return seed;
    }
};

namespace details {

template<typename T>
size_t simple_get_hash(const T& v) {
    return hash<T>()(v);
}

inline size_t FibonacciHash(size_t hash_value, uint32_t fib_shift) {
    constexpr size_t coeff = 11400714819323198485ull;
    return (coeff * hash_value) >> fib_shift;
}

}// namespace details

// Use this function to actually hash multiple things in one line.
// Dispatches to aethermind::hash, so it can hash containers.
// Example:
//
// static size_t hash(const MyStruct& s) {
//   return get_hash(s.member1, s.member2, s.member3);
// }
template<typename... Types>
size_t get_hash(const Types&... args) {
    return hash<decltype(std::tie(args...))>()(std::tie(args...));
}

// Specialization for aethermind::complex
template<typename T>
struct hash<std::complex<T>> {
    size_t operator()(const std::complex<T>& c) const {
        return get_hash(c.real(), c.imag());
    }
};
}// namespace aethermind

namespace std {

template<typename... Types>
struct hash<std::tuple<Types...>> {
    size_t operator()(const std::tuple<Types...>& t) const {
        return aethermind::hash<std::tuple<Types...>>()(t);
    }
};

template<typename T>
struct hash<std::vector<T>> {
    size_t operator()(const std::vector<T>& v) const {
        return aethermind::hash<std::vector<T>>()(v);
    }
};

template<typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
    size_t operator()(const std::pair<T1, T2>& pair) const {
        return aethermind::hash<std::pair<T1, T2>>()(pair);
    }
};

template<typename T>
struct hash<std::complex<T>> {
    size_t operator()(const std::complex<T>& c) const {
        return aethermind::hash<std::complex<T>>()(c);
    }
};

}// namespace std

#endif//AETHERMIND_UTILS_HASH_H
