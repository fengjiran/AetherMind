//
// Created by richard on 8/16/25.
//

#ifndef AETHERMIND_TYPE_TRAITS_H
#define AETHERMIND_TYPE_TRAITS_H

#include "device.h"
#include "tensor.h"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace aethermind {

#define AETHERMIND_FORALL_TAGS(_) \
    _(None)                       \
    _(OpaquePtr)                  \
    _(Tensor)                     \
    _(Storage)                    \
    _(Double)                     \
    _(ComplexDouble)              \
    _(Int)                        \
    _(SymInt)                     \
    _(SymFloat)                   \
    _(SymBool)                    \
    _(Bool)                       \
    _(Tuple)                      \
    _(String)                     \
    _(Blob)                       \
    _(GenericList)                \
    _(GenericDict)                \
    _(Future)                     \
    _(Await)                      \
    _(Device)                     \
    _(Stream)                     \
    _(Object)                     \
    _(PyObject)                   \
    _(Uninitialized)              \
    _(Capsule)                    \
    _(RRef)                       \
    _(Quantizer)                  \
    _(Generator)                  \
    _(Enum)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
    AETHERMIND_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

std::string TagToString(Tag t);

struct AetherMindAny {
    using Payload = std::variant<int64_t,
                                 double,
                                 bool,
                                 void*,
                                 std::string,
                                 Device,
                                 Tensor>;
    Payload payload_;
    Tag tag_{Tag::None};
};

// struct AetherMindAny {
//     union Payload {
//         union data {
//             int64_t v_int;
//             double v_double;
//             bool v_bool;
//             const char* v_str;
//             void* v_handle;
//             Device v_device;
//
//             data() : v_int(0) {}
//         };
//
//         data u;
//         Tensor v_tensor;
//
//         static_assert(std::is_trivially_copyable_v<data>);
//         Payload() : u() {}
//         Payload(const Payload&) = delete;
//         Payload(Payload&&) = delete;
//         Payload& operator=(const Payload&) = delete;
//         Payload& operator=(Payload&&) = delete;
//
//         ~Payload() {}
//     };
//
//     Payload payload_;
//     Tag tag_{Tag::None};
// };


template<typename, typename = void>
struct TypeTraits;

template<>
struct TypeTraits<bool> {
    static void CopyToAny(const bool& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Bool;
        dst->payload_ = src;
    }

    static void MoveToAny(bool src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static bool CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return std::get<bool>(src->payload_);
    }

    static bool MoveFromAnyAfterCheck(AetherMindAny* src) {
        return std::get<bool>(src->payload_);
    }

    static std::optional<bool> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return std::get<bool>(src->payload_);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Bool;
    }

    static std::string TypeStr() {
        return "bool";
    }
};

// POD Int type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_integral_v<T>>> {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Int;
        dst->payload_ = static_cast<int64_t>(src);
    }

    static void MoveToAny(T src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static T CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return static_cast<T>(std::get<int64_t>(src->payload_));
    }

    static T MoveFromAnyAfterCheck(AetherMindAny* src) {
        return static_cast<T>(std::get<int64_t>(src->payload_));
    }

    static std::optional<T> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return static_cast<T>(std::get<int64_t>(src->payload_));
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Int;
    }

    static std::string TypeStr() {
        return "int";
    }
};

// POD Float type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Double;
        dst->payload_ = static_cast<double>(src);
    }

    static void MoveToAny(T src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static T CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return static_cast<T>(std::get<double>(src->payload_));
    }

    static T MoveFromAnyAfterCheck(AetherMindAny* src) {
        return static_cast<T>(std::get<double>(src->payload_));
    }

    static std::optional<T> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return static_cast<T>(std::get<double>(src->payload_));
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Double;
    }

    static std::string TypeStr() {
        return "double";
    }
};

// string type
template<>
struct TypeTraits<std::string> {
    static void CopyToAny(const std::string& src, AetherMindAny* dst) {
        dst->tag_ = Tag::String;
        dst->payload_ = src;
    }

    static void MoveToAny(std::string src, AetherMindAny* dst) {
        dst->tag_ = Tag::String;
        dst->payload_ = std::move(src);
    }

    static std::string CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return std::get<std::string>(src->payload_);
    }

    static std::string MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto str = std::get<std::string>(std::move(src->payload_));
        src->payload_ = 0;
        src->tag_ = Tag::None;
        return str;
    }

    static std::optional<std::string> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return std::get<std::string>(src->payload_);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::String;
    }

    static std::string TypeStr() {
        return "std::string";
    }
};

template<>
struct TypeTraits<void*> {
    static void CopyToAny(void* src, AetherMindAny* dst) {
        dst->tag_ = Tag::OpaquePtr;
        dst->payload_ = src;
    }

    static void MoveToAny(void* src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static void* CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return std::get<void*>(src->payload_);
    }

    static void* MoveFromAnyAfterCheck(AetherMindAny* src) {
        return std::get<void*>(src->payload_);
    }

    static std::optional<void*> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return std::get<void*>(src->payload_);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::OpaquePtr;
    }

    static std::string TypeStr() {
        return "void*";
    }
};

template<>
struct TypeTraits<Device> {
    static void CopyToAny(const Device& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Device;
        dst->payload_ = src;
    }

    static void MoveToAny(Device src, AetherMindAny* dst) {
        dst->tag_ = Tag::Device;
        dst->payload_ = std::move(src);
    }

    static Device CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return std::get<Device>(src->payload_);
    }

    static Device MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto dev = std::get<Device>(std::move(src->payload_));
        src->payload_ = 0;
        src->tag_ = Tag::None;
        return dev;
    }

    static std::optional<Device> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return std::get<Device>(src->payload_);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Device;
    }

    static std::string TypeStr() {
        return "device";
    }
};

template<>
struct TypeTraits<Tensor> {
    static void CopyToAny(const Tensor& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Tensor;
        dst->payload_ = src;
    }

    static void MoveToAny(Tensor src, AetherMindAny* dst) {
        dst->tag_ = Tag::Tensor;
        dst->payload_ = std::move(src);
    }

    static Tensor CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return std::get<Tensor>(src->payload_);
    }

    static Tensor MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto t = std::get<Tensor>(std::move(src->payload_));
        src->payload_ = 0;
        src->tag_ = Tag::None;
        return t;
    }

    static std::optional<Tensor> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return std::get<Tensor>(src->payload_);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Tensor;
    }

    static std::string TypeStr() {
        return "tensor";
    }
};

}// namespace aethermind

#endif//AETHERMIND_TYPE_TRAITS_H
