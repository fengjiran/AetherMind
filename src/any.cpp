//
// Created by 赵丹 on 2025/8/24.
//
#include "any.h"
#include "container/string.h"
#include "device.h"
#include "tensor.h"

namespace aethermind {

Any::Any(const Any& other) : data_(other.data_) {
    if (is_object_ptr()) {
        auto* obj = std::get<Object*>(data_.payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRefObjectHandle(obj);
        }
    }
}

Any::Any(Any&& other) noexcept : data_(std::move(other.data_)) {
    other.data_.payload_ = 0;
    other.data_.tag_ = AnyTag::None;
}

void Any::reset() {
    if (is_object_ptr()) {
        if (auto* obj = std::get<Object*>(data_.payload_); !IsNullTypePtr(obj)) {
            details::ObjectUnsafe::DecRefObjectHandle(obj);
        }
    }
    data_.payload_ = 0;
    data_.tag_ = AnyTag::None;
}

uint32_t Any::use_count() const noexcept {
    if (tag() == AnyTag::None) {
        return 0;
    }

    if (is_object_ptr()) {
        auto* obj = std::get<Object*>(data_.payload_);
        return IsNullTypePtr(obj) ? 0 : obj->use_count();
    }
    return 1;
}

int64_t Any::to_int() const {
    CHECK(is_int()) << "Expected Int but got " << AnyTagToString(tag());
    return cast<int64_t>();
}

double Any::to_double() const {
    CHECK(is_double()) << "Expected Double but got " << AnyTagToString(tag());
    return cast<double>();
}

bool Any::to_bool() const {
    CHECK(is_bool()) << "Expected Bool but got " << AnyTagToString(tag());
    return cast<bool>();
}

void* Any::to_void_ptr() const {
    CHECK(is_void_ptr()) << "Expected VoidPtr but got " << AnyTagToString(tag());
    return cast<void*>();
}

Device Any::to_device() const {
    CHECK(is_device()) << "Expected Device but got " << AnyTagToString(tag());
    return cast<Device>();
}

String Any::to_string() const& {
    CHECK(is_string()) << "Expected String but got " << AnyTagToString(tag());
    return cast<String>();
}

String Any::to_string() && {
    CHECK(is_string()) << "Expected String but got " << AnyTagToString(tag());
    return TypeTraits<String>::MoveFromAnyAfterCheck(&data_);
}

Tensor Any::to_tensor() const& {
    CHECK(is_tensor()) << "Expected Tensor but got " << AnyTagToString(tag());
    return cast<Tensor>();
}

Tensor Any::to_tensor() && {
    CHECK(is_tensor()) << "Expected Tensor but got " << AnyTagToString(tag());
    return TypeTraits<Tensor>::MoveFromAnyAfterCheck(&data_);
}


}// namespace aethermind