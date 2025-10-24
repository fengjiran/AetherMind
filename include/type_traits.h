//
// Created by richard on 8/16/25.
//

#ifndef AETHERMIND_TYPE_TRAITS_H
#define AETHERMIND_TYPE_TRAITS_H

#include "any_utils.h"
#include "container/string.h"
#include "device.h"
#include "object.h"
#include "tensor.h"
// #include "function.h"

namespace aethermind {

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
        Object* obj = src.get_impl_ptr_unsafe();
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

}// namespace aethermind

#endif//AETHERMIND_TYPE_TRAITS_H
