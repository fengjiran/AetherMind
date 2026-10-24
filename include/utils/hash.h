#ifndef AETHERMIND_UTILS_HASH_H
#define AETHERMIND_UTILS_HASH_H

/// @file
/// @brief Hash mixing and composite hashing utilities.
///
/// The AetherMind hash functor dispatches to user-provided `T::hash`, enum
/// underlying values, or `std::hash`, and provides consistent support for
/// common aggregate types.
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
    key = ~key + (key << 15); // key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057; // key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return key;
}

/// @brief Applies a 64-bit integer mixing function.
/// @param key Value to mix.
/// @return The mixed 64-bit value.
inline uint64_t twang_mix64(uint64_t key) noexcept {
    key = (~key) + (key << 21); // key *= (1 << 21) - 1; key -= 1;
    key = key ^ (key >> 24);
    key = key + (key << 3) + (key << 8); // key *= 1 + (1 << 3) + (1 << 8)
    key = key ^ (key >> 14);
    key = key + (key << 2) + (key << 4); // key *= 1 + (1 << 2) + (1 << 4)
    key = key ^ (key >> 28);
    key = key + (key << 31); // key *= 1 + (1 << 31)
    return key;
}

/// @brief Combines one hash value into an existing seed.
/// @tparam T Type convertible to `size_t`.
/// @param seed Existing combined hash seed.
/// @param hash_value Hash value to incorporate.
/// @return The updated combined hash value.
template<typename T>
    requires std::convertible_to<T, size_t>
constexpr size_t hash_combine(size_t seed, const T& hash_value) noexcept {
    const size_t value = static_cast<size_t>(hash_value);
    return seed ^ (value + size_t{0x9e3779b9U} + (seed << 6u) + (seed >> 2u));
}

namespace details {

template<typename T>
concept has_static_hash = requires(const T& value) {
    { T::hash(value) } -> std::same_as<size_t>;
};

template<typename T>
concept has_std_hash = requires(const T& value) {
    { std::hash<T>{}(value) } -> std::same_as<size_t>;
};

template<typename T>
    requires(has_static_hash<T> || std::is_enum_v<T> || has_std_hash<T>)
size_t dispatch_hash(const T& value) {
    // A type's explicit customization wins even if std::hash is also provided.
    if constexpr (has_static_hash<T>) {
        return T::hash(value);
    } else if constexpr (std::is_enum_v<T>) {
        using R = std::underlying_type_t<T>;
        return std::hash<R>{}(static_cast<R>(value));
    } else {
        return std::hash<T>{}(value);
    }
}

template<typename T>
size_t simple_get_hash(const T& value);

} // namespace details

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
        size_t seed = hash_combine(size_t{0}, details::simple_get_hash(pair.first));
        return hash_combine(seed, details::simple_get_hash(pair.second));
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

} // namespace details

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
} // namespace aethermind

#endif // AETHERMIND_UTILS_HASH_H
