//
// Created by richard on 8/16/25.
//

#ifndef AETHERMIND_TYPE_TRAITS_H
#define AETHERMIND_TYPE_TRAITS_H

#include "any_utils.h"
#include "container/string.h"
#include "device.h"
#include "function.h"
#include "tensor.h"

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

template<>
struct TypeTraits<std::nullptr_t> : TypeTraitsBase {
    static void CopyToAny(const std::nullptr_t&, AetherMindAny* dst) {
        dst->tag_ = tag();
        dst->payload_ = 0;
    }

    static void MoveToAny(const std::nullptr_t&, AetherMindAny* dst) {
        dst->tag_ = tag();
        dst->payload_ = 0;
    }

    static std::nullptr_t CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return nullptr;
    }

    static std::nullptr_t MoveFromAnyAfterCheck(const AetherMindAny* src) {
        return nullptr;
    }

    static std::optional<std::nullptr_t> TryCastFromAny(const AetherMindAny* src) {
        if (src->tag_ == tag()) {
            return nullptr;
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::None;
    }
};

template<>
struct TypeTraits<bool> : TypeTraitsBase {
    static void CopyToAny(const bool& src, AetherMindAny* dst) {
        dst->tag_ = tag();
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
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::Bool;
    }
};

// POD Int type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_integral_v<T>>> : TypeTraitsBase {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = tag();
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
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::Int;
    }
};

// POD Float type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_floating_point_v<T>>> : TypeTraitsBase {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = tag();
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
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::Double;
    }
};

template<>
struct TypeTraits<void*> : TypeTraitsBase {
    static void CopyToAny(void* src, AetherMindAny* dst) {
        dst->tag_ = tag();
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
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::OpaquePtr;
    }
};


// string type
template<>
struct TypeTraits<String> : TypeTraitsBase {
    static void CopyToAny(const String& src, AetherMindAny* dst) {
        dst->tag_ = tag();
        Object* obj = src.GetImplPtrUnsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
    }

    static void MoveToAny(String src, AetherMindAny* dst) {
        dst->tag_ = tag();
        dst->payload_ = static_cast<Object*>(src.release_impl_unsafe());
    }

    static String CopyFromAnyAfterCheck(const AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
        return String(ObjectPtr<StringImpl>::reclaim(static_cast<StringImpl*>(obj)));
    }

    static String MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        src->payload_ = static_cast<Object*>(nullptr);
        src->tag_ = AnyTag::None;
        return String(ObjectPtr<StringImpl>::reclaim(static_cast<StringImpl*>(obj)));
    }

    static std::optional<String> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::String;
    }
};

template<>
struct TypeTraits<const char*> : TypeTraits<String> {
    static void CopyToAny(const char* src, AetherMindAny* dst) {
        TypeTraits<String>::CopyToAny(src, dst);
    }

    static void MoveToAny(const char* src, AetherMindAny* dst) {
        TypeTraits<String>::MoveToAny(src, dst);
    }
};

template<>
struct TypeTraits<std::string> : TypeTraits<String> {
    static void CopyToAny(const std::string& src, AetherMindAny* dst) {
        TypeTraits<String>::CopyToAny(src, dst);
    }

    static void MoveToAny(std::string src, AetherMindAny* dst) {
        TypeTraits<String>::MoveToAny(std::move(src), dst);
    }

    static std::string CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return TypeTraits<String>::CopyFromAnyAfterCheck(src);
    }

    static std::string MoveFromAnyAfterCheck(AetherMindAny* src) {
        return TypeTraits<String>::MoveFromAnyAfterCheck(src);
    }

    static std::optional<std::string> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }
};

// Tensor type
template<>
struct TypeTraits<Tensor> : TypeTraitsBase {
    static void CopyToAny(const Tensor& src, AetherMindAny* dst) {
        dst->tag_ = tag();
        Object* obj = src.get_impl_ptr_unsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
    }

    static void MoveToAny(Tensor src, AetherMindAny* dst) {
        dst->tag_ = tag();
        dst->payload_ = static_cast<Object*>(src.release_impl_unsafe());
    }

    static Tensor CopyFromAnyAfterCheck(const AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }

        return Tensor(ObjectPtr<TensorImpl>::reclaim(static_cast<TensorImpl*>(obj)));
    }

    static Tensor MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        src->payload_ = static_cast<Object*>(nullptr);
        src->tag_ = AnyTag::None;
        return Tensor(ObjectPtr<TensorImpl>::reclaim(static_cast<TensorImpl*>(obj)));
    }

    static std::optional<Tensor> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == tag();
    }

    static std::string TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::Tensor;
    }
};

// Device type
template<>
struct TypeTraits<Device> : TypeTraitsBase {
    static void CopyToAny(const Device& src, AetherMindAny* dst) {
        dst->tag_ = tag();
        Object* obj = src.get_impl_ptr_unsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
    }

    static void MoveToAny(Device src, AetherMindAny* dst) {
        dst->tag_ = tag();
        dst->payload_ = static_cast<Object*>(src.release_impl_unsafe());
    }

    static Device CopyFromAnyAfterCheck(const AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
        return Device(ObjectPtr<DeviceImpl>::reclaim(static_cast<DeviceImpl*>(obj)));
    }

    static Device MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        src->payload_ = static_cast<Object*>(nullptr);
        src->tag_ = AnyTag::None;
        return Device(ObjectPtr<DeviceImpl>::reclaim(static_cast<DeviceImpl*>(obj)));
    }

    static std::optional<Device> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == tag();
    }

    static String TypeStr() {
        return AnyTagToString(tag());
    }

    static AnyTag tag() {
        return AnyTag::Device;
    }
};

// function
template<>
struct TypeTraits<Function> : TypeTraitsBase {
    static void CopyToAny(const Function& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Function;
        FunctionImpl* obj = src.get_impl_ptr_unsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
    }

    static void MoveToAny(Function src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Function;
        dst->payload_ = static_cast<Object*>(src.release_impl_unsafe());
    }

    static Function CopyFromAnyAfterCheck(const AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
        return Function(ObjectPtr<FunctionImpl>::reclaim(static_cast<FunctionImpl*>(obj)));
    }

    static Function MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        src->payload_ = static_cast<Object*>(nullptr);
        src->tag_ = AnyTag::None;
        return Function(ObjectPtr<FunctionImpl>::reclaim(static_cast<FunctionImpl*>(obj)));
    }

    static std::optional<Function> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Function;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Function);
    }
};

template<typename F>
struct TypeTraits<TypedFunction<F>> : TypeTraitsBase {
    static void CopyToAny(const TypedFunction<F>& src, AetherMindAny* dst) {
        TypeTraits<Function>::CopyToAny(src.packed(), dst);
    }

    static void MoveToAny(TypedFunction<F> src, AetherMindAny* dst) {
        TypeTraits<Function>::MoveToAny(std::move(src.packed()), dst);
    }

    static TypedFunction<F> CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return TypeTraits<Function>::CopyFromAnyAfterCheck(src);
    }

    static TypedFunction<F> MoveFromAnyAfterCheck(AetherMindAny* src) {
        return TypeTraits<Function>::MoveFromAnyAfterCheck(src);
    }

    static std::optional<TypedFunction<F>> TryCastFromAny(const AetherMindAny* src) {
        auto opt = TypeTraits<Function>::TryCastFromAny(src);
        if (opt.has_value()) {
            return TypedFunction<F>(*std::move(opt));
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Function;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Function);
    }
};


}// namespace aethermind

#endif//AETHERMIND_TYPE_TRAITS_H
