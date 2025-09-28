//
// Created by 赵丹 on 25-7-19.
//

#ifndef AETHERMIND_FUNCTION_TRAITS_H
#define AETHERMIND_FUNCTION_TRAITS_H

#include "any.h"
#include "container/string.h"

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
struct FunctionTraits;

// plain function type.
template<typename R, typename... Args>
struct FunctionTraits<R(Args...)> {
    using func_type = R(Args...);
    using return_type = R;
    using args_type_tuple = std::tuple<Args...>;
    static constexpr auto num_args = sizeof...(Args);

    /*! \brief Whether this function can be converted to Function via FromTyped */
    static constexpr bool unpacked_args_supported = (ArgSupported<Args> && ...) && RetSupported<R>;

    static String Schema() {
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
struct FunctionInfoForFunctor;

template<typename Functor, typename R, typename... Args>
struct FunctionInfoForFunctor<R (Functor::*)(Args...)> : FunctionTraits<R(Args...)> {};

template<typename Functor, typename R, typename... Args>
struct FunctionInfoForFunctor<R (Functor::*)(Args...) const> : FunctionTraits<R(Args...)> {};

template<typename Functor>
struct FunctionInfo : FunctionInfoForFunctor<decltype(&Functor::operator())> {};

template<typename R, typename... Args>
struct FunctionInfo<R(Args...)> : FunctionTraits<R(Args...)> {};

template<typename R, typename... Args>
struct FunctionInfo<R (*)(Args...)> : FunctionTraits<R(Args...)> {};

template<typename R, typename... Args>
struct FunctionInfo<std::function<R(Args...)>> : FunctionTraits<R(Args...)> {};

template<typename R, typename... Args>
struct FunctionInfo<std::function<R (*)(Args...)>> : FunctionTraits<R(Args...)> {};

using FGetFunctionSchema = String (*)();
class Any2Arg {
public:
    Any2Arg(const Any* args, int32_t idx, const std::string* opt_name, FGetFunctionSchema f_schema)
        : args_(args), idx_(idx), opt_name_(opt_name), f_schema_(f_schema) {}

    template<typename T>
    operator T() {//NOLINT
        using TypeWithoutCR = std::remove_const_t<std::remove_reference_t<T>>;
        if constexpr (std::is_same_v<TypeWithoutCR, Any>) {
            return args_[idx_];
        } else {
            std::optional<TypeWithoutCR> opt = args_[idx_].try_cast<TypeWithoutCR>();
            if (!opt.has_value()) {
                AETHERMIND_THROW(TypeError) << "Mismatched type on argument #" << idx_
                                            << " when calling: `"
                                            << (opt_name_ == nullptr ? "" : *opt_name_)
                                            << (f_schema_ == nullptr ? "" : (*f_schema_)()) << "`. Expected `"
                                            << Type2Str<TypeWithoutCR>::value();
            }
            return opt.value();
        }
    }

private:
    const Any* args_;
    int32_t idx_;
    const std::string* opt_name_;
    FGetFunctionSchema f_schema_;
};

template<typename R, size_t... Is, typename F>
void unpack_call(const F& callable, std::index_sequence<Is...>, const std::string* opt_name,
                 const Any* args, int32_t num_args, Any* res) {
    using FuncInfo = FunctionInfo<F>;
    const FGetFunctionSchema f_schema = FuncInfo::Schema;
    static_assert(FuncInfo::unpacked_args_supported, "the function signature do not support unpacked.");
    constexpr size_t nargs = sizeof...(Is);
    if (nargs != num_args) {
        AETHERMIND_THROW(TypeError) << "Mismatched number of arguments when calling: `"
                                    << (opt_name == nullptr ? "" : *opt_name)
                                    << (f_schema == nullptr ? "" : (*f_schema)()) << "`. Expected " << nargs
                                    << " but got " << num_args << " arguments";
    }

    if constexpr (std::is_same_v<R, void>) {
        callable(Any2Arg(args, Is, opt_name, f_schema)...);
    } else {
        *res = R(callable(Any2Arg(args, Is, opt_name, f_schema)...));
    }
}

template<typename R, typename args_tuple>
struct make_function_traits;

template<typename R, typename... Args>
struct make_function_traits<R, std::tuple<Args...>> {
    using type = FunctionTraits<R(Args...)>;
};

template<typename R, typename... Args>
using make_function_traits_t = make_function_traits<R, std::tuple<Args...>>::type;

template<size_t start, size_t N, size_t... Is>
struct make_offset_index_sequence_impl : make_offset_index_sequence_impl<start, N - 1, start + N - 1, Is...> {
    static_assert(static_cast<int>(start) >= 0);
    static_assert(static_cast<int>(N) >= 0);
};

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_TRAITS_H
