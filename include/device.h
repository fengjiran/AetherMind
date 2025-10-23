//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_DEVICE_H
#define AETHERMIND_DEVICE_H

#include "container/string.h"
#include "macros.h"
#include "object.h"
#include "type_traits.h"

#include <cstdint>

namespace aethermind {

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
    kCANN,
    kUndefined
};

constexpr DeviceType kCPU = DeviceType::kCPU;
constexpr DeviceType kCUDA = DeviceType::kCUDA;
constexpr DeviceType kCANN = DeviceType::kCANN;
constexpr DeviceType kUndefined = DeviceType::kUndefined;

class String;

/// Represents a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
class DeviceImpl : public Object {
public:
    DeviceImpl() : DeviceImpl(kCPU) {}

    explicit DeviceImpl(DeviceType type, int8_t index = -1) : type_(type), index_(index) {
        validate();
    }

    NODISCARD DeviceType type() const noexcept {
        return type_;
    }

    NODISCARD int8_t index() const noexcept {
        return index_;
    }

    NODISCARD bool has_index() const noexcept {
        return index_ != -1;
    }

    NODISCARD bool is_cpu() const noexcept {
        return type_ == kCPU;
    }

    NODISCARD bool is_cuda() const noexcept {
        return type_ == kCUDA;
    }

    NODISCARD bool is_cann() const noexcept {
        return type_ == kCANN;
    }

private:
    DeviceType type_;
    int8_t index_ = -1;

    void validate() const {
        CHECK(index() >= -1) << "Device index must be greater than or equal to -1, but got " << index();
        CHECK(!is_cpu() || index() <= 0) << "CPU device index must be -1 or zero, but got " << index();
    }
};

class Device {
public:
    Device() = default;// default cpu

    explicit Device(DeviceType type, int8_t index = -1);

    explicit Device(ObjectPtr<DeviceImpl>);

    NODISCARD bool defined() const noexcept {
        return impl_;
    }

    NODISCARD uint32_t use_count() const noexcept {
        return impl_.use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD DeviceType type() const noexcept {
        return impl_->type();
    }

    NODISCARD int8_t index() const noexcept {
        return impl_->index();
    }

    NODISCARD bool has_index() const noexcept {
        return impl_->has_index();
    }

    NODISCARD DeviceImpl* get_impl_ptr_unsafe() const noexcept;

    NODISCARD DeviceImpl* release_impl_unsafe();

    NODISCARD bool is_cpu() const noexcept {
        return type() == kCPU;
    }

    NODISCARD bool is_cuda() const noexcept {
        return type() == kCUDA;
    }

    NODISCARD bool is_cann() const noexcept {
        return type() == kCANN;
    }

    NODISCARD String str() const;

    static Device CPU();
    static Device CUDA();
    static Device CANN();

    bool operator==(const Device& other) const {
        return type() == other.type() && index() == other.index();
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

private:
    ObjectPtr<DeviceImpl> impl_;
};

String DeviceType2Str(DeviceType device_type, bool lower_case = false);
bool IsValidDeviceType(DeviceType device_type);
std::ostream& operator<<(std::ostream& os, DeviceType device_type);
std::ostream& operator<<(std::ostream& os, const Device& device);

// Device type
template<>
struct TypeTraits<Device> : TypeTraitsBase {
    static void CopyToAny(const Device& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Device;
        Object* obj = src.get_impl_ptr_unsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
    }

    static void MoveToAny(Device src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Device;
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
        return src->tag_ == AnyTag::Device;
    }

    static String TypeStr() {
        return AnyTagToString(AnyTag::Device);
    }
};

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::DeviceType> {
    std::size_t operator()(aethermind::DeviceType k) const noexcept {
        return std::hash<int>()(static_cast<int>(k));
    }
};
}// namespace std

#endif//AETHERMIND_DEVICE_H
