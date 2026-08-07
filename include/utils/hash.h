#ifndef AETHERMIND_UTILS_HASH_H
#define AETHERMIND_UTILS_HASH_H

/// @file
/// @brief Hash mixing and composite hashing utilities.
///
/// The AetherMind hash functor dispatches to user-provided `T::hash`, enum
/// underlying values, or `std::hash`, and provides consistent support for
/// common aggregate types.

#include "container/string.h"
#include "utils/xxh3.h"

#include <complex>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace aethermind {

/// @brief Applies a 32-bit integer mixing function.
/// @param key Value to mix.
/// @return The mixed 32-bit value.
inline uint32_t twang_mix32(uint32_t key) noexcept {
    key = ~key + (key << 15);// key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;// key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return key;
}

/// @brief Applies a 64-bit integer mixing function.
/// @param key Value to mix.
/// @return The mixed 64-bit value.
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

/// @brief Combines one hash value into an existing seed.
/// @tparam T Type convertible to `size_t`.
/// @param seed Existing combined hash seed.
/// @param hash_value Hash value to incorporate.
/// @return The updated combined hash value.
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

// Use SFINAE to dispatch to std::hash when possible, cast enum types to their
// underlying type, and fall back to T::hash otherwise. Enum dispatch remains
// separate because some compilers provide enum hashing as an extension.
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

/// @brief Computes hashes using AetherMind's type-specific dispatch rules.
/// @tparam T Type to hash.
template<typename T>
struct hash {
    /// @brief Hashes one value.
    /// @param v Value to hash.
    /// @return The computed hash value.
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

/// @brief Hashes a `std::tuple` by combining each element in order.
/// @tparam Types Tuple element types.
template<typename... Types>
struct hash<std::tuple<Types...>> {
    /// @brief Hashes the tuple elements in order.
    /// @param t Tuple to hash.
    /// @return The combined tuple hash.
    size_t operator()(const std::tuple<Types...>& t) const {
        size_t seed = 0;
        auto func = [&seed](auto&&... x) {
            ((seed = hash_combine(seed, details::simple_get_hash(x))), ...);
        };
        std::apply(func, t);
        return seed;
    }
};

/// @brief Hashes a pair by combining its first and second elements.
/// @tparam T1 First element type.
/// @tparam T2 Second element type.
template<typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
    /// @brief Hashes both pair elements in order.
    /// @param pair Pair to hash.
    /// @return The combined pair hash.
    size_t operator()(const std::pair<T1, T2>& pair) const {
        std::tuple<T1, T2> tuple = std::make_tuple(pair.first, pair.second);
        return details::simple_get_hash(tuple);
    }
};

/// @brief Hashes a vector by combining its elements in order.
/// @tparam T Vector element type.
template<typename T>
struct hash<std::vector<T>> {
    /// @brief Hashes all vector elements in sequence.
    /// @param v Vector to hash.
    /// @return The combined vector hash.
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

/// @brief Hashes multiple values as one composite key.
///
/// Each argument is dispatched through `aethermind::hash`, so supported
/// aggregate and container types can be combined without custom boilerplate.
///
/// @tparam Types Types of the values to combine.
/// @param args Values to hash in order.
/// @return The combined hash value.
template<typename... Types>
size_t get_hash(const Types&... args) {
    return hash<decltype(std::tie(args...))>()(std::tie(args...));
}

/// @brief Hashes a standard complex number from its real and imaginary parts.
/// @tparam T Complex component type.
template<typename T>
struct hash<std::complex<T>> {
    /// @brief Hashes the real and imaginary components in order.
    /// @param c Complex value to hash.
    /// @return The combined complex-value hash.
    size_t operator()(const std::complex<T>& c) const {
        return get_hash(c.real(), c.imag());
    }
};
}// namespace aethermind

/// @brief Standard-library hash specializations backed by `aethermind::hash`.
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

#endif// AETHERMIND_UTILS_HASH_H
