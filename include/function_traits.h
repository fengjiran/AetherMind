//
// Created by 赵丹 on 25-7-19.
//

#ifndef AETHERMIND_FUNCTION_TRAITS_H
#define AETHERMIND_FUNCTION_TRAITS_H

#include "any.h"

#include <functional>
#include <tuple>
#include <type_traits>

namespace aethermind {
namespace details {

template<typename T>
struct is_tuple : std::false_type {};

template<typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};

template<typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

/**
 * is_function_type<T> is true_type if T is a plain function type (i.e.
 * "Result(Args...)")
 */
template<typename T>
struct is_plain_function_type : std::false_type {};

template<typename R, typename... Args>
struct is_plain_function_type<R(Args...)> : std::true_type {};

template<typename T>
constexpr bool is_plain_function_type_v = is_plain_function_type<T>::value;

template<typename ArgType, typename = std::enable_if_t<is_tuple_v<ArgType>>>
struct Arg2Str {
    template<size_t i>
    static void f(std::ostream& os) {
        using Arg = std::tuple_element_t<i, ArgType>;
        if constexpr (i != 0) {
            os << ", ";
        }
        os << i << ": " << Type2Str<Arg>::value();
    }

    template<size_t... I>
    static void Run(std::ostream& os, std::index_sequence<I...>) {
        (f<I>(os), ...);
    }
};

template<typename T>
static constexpr bool ArgSupported = std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, Any> ||
                                     TypeTraitsNoCR<T>::convert_enabled;

// NOTE: return type can only support non-reference managed returns
template<typename T>
static constexpr bool RetSupported = std::is_same_v<T, Any> || std::is_void_v<T> || TypeTraits<T>::convert_enabled;


template<typename FuncType>
struct function_traits;

// plain function type.
template<typename R, typename... Args>
struct function_traits<R(Args...)> {
    using func_type = R(Args...);
    using return_type = R;
    using args_type_tuple = std::tuple<Args...>;
    static constexpr auto num_args = sizeof...(Args);

    /*! \brief Whether this function can be converted to Function via FromTyped */
    static constexpr bool unpacked_args_supported = (ArgSupported<Args> && ...) && RetSupported<R>;

    static std::string Schema() {
        using idx_seq = std::make_index_sequence<sizeof...(Args)>;
        std::stringstream ss;
        ss << "(";
        Arg2Str<args_type_tuple>::Run(ss, idx_seq{});
        ss << ") -> " << Type2Str<R>::value();
        return ss.str();
    }
};

/**
 * creates a function_traits type for a simple function (pointer) or functor (lambda/struct).
 * Currently, does not support class methods.
 */
template<typename Functor>
struct infer_function_traits {
    using type = function_traits<std::remove_const_t<decltype(Functor::operator())>>;
};

template<typename R, typename... Args>
struct infer_function_traits<R(Args...)> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
struct infer_function_traits<R (*)(Args...)> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
struct infer_function_traits<std::function<R(Args...)>> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
struct infer_function_traits<std::function<R (*)(Args...)>> {
    using type = function_traits<R(Args...)>;
};

template<typename T>
using infer_function_traits_t = infer_function_traits<T>::type;

template<typename R, typename args_tuple>
struct make_function_traits;

template<typename R, typename... Args>
struct make_function_traits<R, std::tuple<Args...>> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
using make_function_traits_t = typename make_function_traits<R, std::tuple<Args...>>::type;

template<size_t start, size_t N, size_t... Is>
struct make_offset_index_sequence_impl : make_offset_index_sequence_impl<start, N - 1, start + N - 1, Is...> {
    static_assert(static_cast<int>(start) >= 0);
    static_assert(static_cast<int>(N) >= 0);
};

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_TRAITS_H
