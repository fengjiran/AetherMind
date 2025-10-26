//
// Created by richard on 10/23/25.
//

#ifndef AETHERMIND_ANY_UTILS_H
#define AETHERMIND_ANY_UTILS_H

#include <memory>
#include <ostream>
#include <typeindex>

namespace aethermind {

// The following tags are used to tag the type of any value.
// The bool var represents whether ObjectPtr.
#define AETHERMIND_FORALL_ANY_TAGS(_) \
    _(None, false)                    \
    _(OpaquePtr, false)               \
    _(Tensor, true)                   \
    _(Storage, true)                  \
    _(Double, false)                  \
    _(ComplexDouble, true)            \
    _(Int, false)                     \
    _(SymInt, true)                   \
    _(SymFloat, true)                 \
    _(SymBool, true)                  \
    _(Bool, false)                    \
    _(Tuple, true)                    \
    _(String, true)                   \
    _(Blob, true)                     \
    _(GenericList, true)              \
    _(GenericDict, true)              \
    _(Future, true)                   \
    _(Await, true)                    \
    _(Device, true)                   \
    _(Stream, true)                   \
    _(Object, true)                   \
    _(PyObject, true)                 \
    _(Uninitialized, false)           \
    _(Capsule, true)                  \
    _(RRef, true)                     \
    _(Quantizer, true)                \
    _(Generator, true)                \
    _(Enum, true)                     \
    _(Function, true)

enum class AnyTag : uint32_t {
#define DEFINE_TAG(x, _) x,
    AETHERMIND_FORALL_ANY_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

inline bool IsObjectPtr(AnyTag tag) {
#define CASE(T, v)  \
    case AnyTag::T: \
        return v;

    switch (tag) {
        AETHERMIND_FORALL_ANY_TAGS(CASE)
    }
#undef CASE
    return false;
}

inline std::string AnyTagToString(AnyTag t) {
#define CASE(T, _)  \
    case AnyTag::T: \
        return #T;

    switch (t) {
        AETHERMIND_FORALL_ANY_TAGS(CASE)
    }
#undef CASE
    return "";
}

inline std::ostream& operator<<(std::ostream& os, AnyTag t) {
    os << AnyTagToString(t);
    return os;
}

struct AetherMindAny {
    using Payload = std::variant<int64_t,
                                 double,
                                 bool,
                                 void*,
                                 Object*>;
    Payload payload_;
    AnyTag tag_{AnyTag::None};
};

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

template<typename T>
constexpr bool is_pod_v = std::is_standard_layout_v<T> && std::is_trivial_v<T>;

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_ANY_UTILS_H
