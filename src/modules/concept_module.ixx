//
// Created by richard on 1/23/26.
//
module;
//Global Module Fragment
// #include <glog/logging.h>
// #include "data_type.h"

// #include "container/string.h"
#include "utils/bfloat16.h"
#include "utils/float8_e4m3fn.h"
#include "utils/float8_e5m2.h"
#include "utils/half.h"

#include <concepts>
export module concept_module;

namespace aethermind {

class Tensor;
class Function;
template<typename FType>
class TypedFunction;
class Any;

namespace details {

export template<typename T>
concept is_integral = std::integral<T> && !std::same_as<T, bool>;

export template<typename T>
concept is_boolean = std::is_same_v<T, bool>;

export template<typename T>
concept is_floating_point = std::floating_point<T> ||
                            std::same_as<T, Half> ||
                            std::same_as<T, BFloat16> ||
                            std::same_as<T, Float8_e4m3fn> ||
                            std::same_as<T, Float8_e5m2>;

//
// export template<typename T>
// concept is_string = std::same_as<T, std::string> ||
//                     std::same_as<T, std::string_view> ||
//                     std::same_as<T, const char*> ||
//                     std::same_as<T, String>;
//
// export template<typename T>
// concept is_plain_type = is_integral<T> ||
//                         is_boolean<T> ||
//                         is_floating_point<T> ||
//                         is_string<T>;

export template<typename T>
concept has_use_count_method_v = requires(T t) {
    t.use_count();
};

}// namespace details

}// namespace aethermind
