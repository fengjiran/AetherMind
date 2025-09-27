//
// Created by richard on 8/16/25.
//

#ifndef AETHERMIND_TYPE_TRAITS_H
#define AETHERMIND_TYPE_TRAITS_H

#include "device.h"
#include "object.h"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

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
    _(Device, false)                  \
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
                                 Object*,
                                 Device>;
    Payload payload_;
    AnyTag tag_{AnyTag::None};
};

template<typename, typename = void>
struct TypeTraits {
    /*! \brief Whether the type can appear as a storage type in Container */
    static constexpr bool storage_enabled = false;

    /*! \brief Whether the type can be converted to Any. */
    static constexpr bool convert_enabled = false;
};

struct TypeTraitsBase {
    static constexpr bool storage_enabled = true;
    static constexpr bool convert_enabled = true;
};

/*!
 * \brief TypeTraits that removes const and reference keywords.
 * \tparam T the original type
 */
template<typename T>
using TypeTraitsNoCR = TypeTraits<std::remove_const_t<std::remove_reference_t<T>>>;

template<>
struct TypeTraits<std::nullptr_t> : TypeTraitsBase {
    static void CopyToAny(const std::nullptr_t&, AetherMindAny* dst) {
        dst->tag_ = AnyTag::None;
        dst->payload_ = 0;
    }

    static void MoveToAny(const std::nullptr_t&, AetherMindAny* dst) {
        dst->tag_ = AnyTag::None;
        dst->payload_ = 0;
    }

    static std::nullptr_t CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return nullptr;
    }

    static std::nullptr_t MoveFromAnyAfterCheck(const AetherMindAny* src) {
        return nullptr;
    }

    static std::optional<std::nullptr_t> TryCastFromAny(const AetherMindAny* src) {
        if (src->tag_ == AnyTag::None) {
            return nullptr;
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::None;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::None);
    }
};

template<>
struct TypeTraits<bool> : TypeTraitsBase {
    static void CopyToAny(const bool& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Bool;
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
        if (src->tag_ == AnyTag::Bool) {
            return std::get<bool>(src->payload_);
        }

        if (src->tag_ == AnyTag::Int) {
            return static_cast<bool>(std::get<int64_t>(src->payload_));
        }

        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Bool;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Bool);
    }
};

// POD Int type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_integral_v<T>>> : TypeTraitsBase {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Int;
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
        if (src->tag_ == AnyTag::Int) {
            return static_cast<T>(std::get<int64_t>(src->payload_));
        }

        if (src->tag_ == AnyTag::Bool) {
            return static_cast<T>(std::get<bool>(src->payload_));
        }

        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Int;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Int);
    }
};

// POD Float type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_floating_point_v<T>>> : TypeTraitsBase {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Double;
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
        if (src->tag_ == AnyTag::Double) {
            return static_cast<T>(std::get<double>(src->payload_));
        }

        if (src->tag_ == AnyTag::Int) {
            return static_cast<T>(std::get<int64_t>(src->payload_));
        }

        if (src->tag_ == AnyTag::Bool) {
            return static_cast<T>(std::get<bool>(src->payload_));
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Double;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Double);
    }
};

template<>
struct TypeTraits<void*> : TypeTraitsBase {
    static void CopyToAny(void* src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::OpaquePtr;
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
        return src->tag_ == AnyTag::OpaquePtr;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::OpaquePtr);
    }
};

// Device type
template<>
struct TypeTraits<Device> : TypeTraitsBase {
    static void CopyToAny(const Device& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Device;
        dst->payload_ = src;
    }

    static void MoveToAny(Device src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Device;
        dst->payload_ = src;
    }

    static Device CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return std::get<Device>(src->payload_);
    }

    static Device MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto dev = std::get<Device>(std::move(src->payload_));
        src->payload_ = Device(kUndefined);
        src->tag_ = AnyTag::None;
        return dev;
    }

    static std::optional<Device> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return std::get<Device>(src->payload_);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Device;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Device);
    }
};

}// namespace aethermind

#endif//AETHERMIND_TYPE_TRAITS_H
