//
// Created by richard on 10/23/25.
//

#ifndef AETHERMIND_ANY_UTILS_H
#define AETHERMIND_ANY_UTILS_H

#include "object.h"

#include <map>
#include <unordered_map>

#ifdef CPP20
#include <concepts>
#endif

namespace aethermind {

struct Half;
struct BFloat16;
struct Float8_e4m3fn;
struct Float8_e5m2;
class String;

namespace details {

#ifdef CPP20
template<typename T>
concept is_integral = std::integral<T> && !std::same_as<T, bool>;

template<typename T>
concept is_floating_point = std::floating_point<T> ||
                            std::same_as<T, Half> ||
                            std::same_as<T, BFloat16> ||
                            std::same_as<T, Float8_e4m3fn> ||
                            std::same_as<T, Float8_e5m2>;

template<typename T>
concept is_string = std::same_as<T, std::string> ||
                    std::same_as<T, std::string_view> ||
                    std::same_as<T, const char*> ||
                    std::same_as<T, String>;

template<typename T>
concept is_plain_type = is_integral<T> ||
                        is_floating_point<T> ||
                        is_string<T>;

template<typename T>
concept has_use_count_method_v = requires(const T& t) {
    t.use_count();
};

#else
template<typename T>
constexpr bool is_integral = std::is_integral_v<T> && !std::is_same_v<T, bool>;

template<typename T>
constexpr bool is_floating_point = std::is_floating_point_v<T> ||
                                   std::is_same_v<T, Half> ||
                                   std::is_same_v<T, BFloat16> ||
                                   std::is_same_v<T, Float8_e4m3fn> ||
                                   std::is_same_v<T, Float8_e5m2>;

template<typename T>
constexpr bool is_string = std::is_same_v<T, std::string> ||
                           std::is_same_v<T, std::string_view> ||
                           std::is_same_v<T, const char*> ||
                           std::is_same_v<T, String>;

template<typename T>
constexpr bool is_plain_type = is_integral<T> ||
                               is_floating_point<T> ||
                               is_string<T>;

template<typename T, typename = void>
struct has_use_count_method : std::false_type {};

template<typename T>
struct has_use_count_method<T, decltype((void) std::declval<T>().use_count())> : std::true_type {};

template<typename T>
constexpr bool has_use_count_method_v = has_use_count_method<T>::value;

#endif

template<typename T>
struct is_map : std::false_type {};

template<typename K, typename V>
struct is_map<std::unordered_map<K, V>> : std::true_type {};

template<typename K, typename V>
struct is_map<std::map<K, V>> : std::true_type {};

template<typename T>
constexpr bool is_map_v = is_map<T>::value;


}// namespace details

}// namespace aethermind

#endif//AETHERMIND_ANY_UTILS_H
