//
// Created by richard on 10/23/25.
//

#ifndef AETHERMIND_ANY_UTILS_H
#define AETHERMIND_ANY_UTILS_H

#include "data_type.h"
#include "device.h"
#include "object.h"

#include <map>
#include <unordered_map>

#ifdef CPP20
#include <concepts>
#endif

namespace aethermind {

class Tensor;
class Function;
template<typename FType>
class TypedFunction;
class Any;

namespace details {

#ifdef CPP20
template<typename T>
concept is_integral = std::integral<T> && !std::same_as<T, bool>;

template<typename T>
concept is_boolean = std::is_same_v<T, bool>;

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
                        is_boolean<T> ||
                        is_floating_point<T> ||
                        is_string<T>;

template<typename T>
concept has_use_count_method_v = requires(T t) {
    t.use_count();
};

template<typename T, typename IndexType>
concept has_operator_subscript = requires(T t, IndexType i) {
    t[i];
};

template<typename T>
concept is_map_subscript = requires {
    typename T::key_type;
    typename T::mapped_type;
    requires has_operator_subscript<T, typename T::key_type>;
};

template<typename T>
concept is_array_subscript = has_operator_subscript<T, typename T::size_type>;

template<typename T>
concept is_container = requires(T t) {
    requires !is_string<T>;
    typename T::value_type;
    typename T::iterator;
    typename T::size_type;
    { t.begin() } -> std::convertible_to<typename T::iterator>;
    { t.end() } -> std::convertible_to<typename T::iterator>;
};

template<typename T>
concept is_map = requires(T t) {
    typename T::key_type;
    typename T::mapped_type;
    typename T::iterator;
    typename T::value_type;
    { t.begin() } -> std::convertible_to<typename T::iterator>;
    { t.end() } -> std::convertible_to<typename T::iterator>;
    requires std::is_same_v<typename T::value_type, std::pair<const typename T::key_type, typename T::mapped_type>>;
};

template<typename T>
concept is_unordered_map = is_map<T> && requires(T t) {
    { t.hash_function() };
};

template<typename T>
concept is_unique_key_map = is_map<T> && (std::is_same_v<T, std::unordered_map<typename T::key_type, typename T::mapped_type>> ||
                                           std::is_same_v<T, std::map<typename T::key_type, typename T::mapped_type>>);

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

// template<typename T>
// struct is_map : std::false_type {};
//
// template<typename K, typename V>
// struct is_map<std::unordered_map<K, V>> : std::true_type {};
//
// template<typename K, typename V>
// struct is_map<std::map<K, V>> : std::true_type {};
//
// template<typename T>
// constexpr bool is_map_v = is_map<T>::value;

template<typename T>
struct TypeName {
    static String value() {
        return typeid(T).name();
    }
};

#define DEFINE_TYPE_NAME(code, bits, lanes, T, name) \
    template<>                                       \
    struct TypeName<T> {                             \
        static String value() {                      \
            return #name;                            \
        }                                            \
    };


SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_TYPE_NAME);
DEFINE_TYPE_NAME(_, _, _, Tensor, Tensor);
DEFINE_TYPE_NAME(_, _, _, Device, Device);
DEFINE_TYPE_NAME(_, _, _, Any, Any);
DEFINE_TYPE_NAME(_, _, _, Any*, Any*);
DEFINE_TYPE_NAME(_, _, _, const Any*, const Any*);
DEFINE_TYPE_NAME(_, _, _, const Any&, const Any&);
DEFINE_TYPE_NAME(_, _, _, void, void);
DEFINE_TYPE_NAME(_, _, _, Function, Function);

template<typename FType>
struct TypeName<TypedFunction<FType>> {
    static String value() {
        return "Function";
    }
};

#undef DEFINE_TYPE_NAME

template<typename T>
struct Type2Str {
    using U = std::remove_const_t<std::remove_reference_t<T>>;
    static String value() {
        return TypeName<U>::value();
    }
};

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_ANY_UTILS_H
