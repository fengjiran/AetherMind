//
// Created by richard on 10/23/25.
//

#ifndef AETHERMIND_ANY_UTILS_H
#define AETHERMIND_ANY_UTILS_H

#include "object.h"

namespace aethermind {

struct Half;
struct BFloat16;
struct Float8_e4m3fn;
struct Float8_e5m2;
class String;

// template<typename T>
// struct complex;

namespace details {

template<typename T>
constexpr bool is_integral_v = std::is_integral_v<T> && !std::is_same_v<T, bool>;

template<typename T>
constexpr bool is_floating_point_v = std::is_floating_point_v<T> ||
                                     std::is_same_v<T, Half> ||
                                     std::is_same_v<T, BFloat16> ||
                                     std::is_same_v<T, Float8_e4m3fn> ||
                                     std::is_same_v<T, Float8_e5m2>;

template<typename T>
constexpr bool is_string_v = std::is_same_v<T, std::string> ||
                             std::is_same_v<T, std::string_view> ||
                             std::is_same_v<T, const char*> ||
                             std::is_same_v<T, String>;

template<typename T>
constexpr bool is_plain_v = is_integral_v<T> ||
                            is_floating_point_v<T> ||
                            is_string_v<T>;

template<typename T, typename = void>
struct has_use_count_method : std::false_type {};

template<typename T>
struct has_use_count_method<T, decltype((void) std::declval<T>().use_count())> : std::true_type {};

template<typename T>
constexpr bool has_use_count_method_v = has_use_count_method<T>::value;

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_ANY_UTILS_H
