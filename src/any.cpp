//
// Created by 赵丹 on 2025/8/24.
//
#include "any.h"
#include "container/string.h"
#include "device.h"
#include "tensor.h"

namespace aethermind {

Any::Any(const Any& other) {
    if (other.has_value()) {
        ptr_ = other.ptr_->Clone();
    }
}

Any::Any(Any&& other) noexcept : ptr_(std::move(other.ptr_)) {}

Any& Any::operator=(const Any& other) & {
    Any(other).swap(*this);
    return *this;
}

Any& Any::operator=(Any&& other) & noexcept {
    Any(std::move(other)).swap(*this);
    return *this;
}

const std::type_index& Any::type() const {
    if (has_value()) {
        return ptr_->type();
    }
    AETHERMIND_THROW(BadAnyCast) << "Any has no value.";
    AETHERMIND_UNREACHABLE();
}

void Any::swap(Any& other) noexcept {
    std::swap(ptr_, other.ptr_);
}

void Any::reset() {
    ptr_.reset();
}

bool Any::has_value() const noexcept {
    return ptr_ != nullptr;
}

void* Any::GetUnderlyingPtr() const {

}

bool Any::IsBool() const noexcept {
    return type() == std::type_index(typeid(bool));
}

bool Any::IsInteger() const noexcept {
    return type() == std::type_index(typeid(int64_t));
}

bool Any::IsFloatingPoint() const noexcept {
    return type() == std::type_index(typeid(double));
}

bool Any::IsString() const noexcept {
    return type() == std::type_index(typeid(String));
}

bool Any::IsVoidPtr() const noexcept {
    return type() == std::type_index(typeid(void*));
}

bool Any::IsObjectRef() const noexcept {
    return has_value() ? ptr_->IsObjectRef() : false;
}

bool Any::IsDevice() const noexcept {
    return type() == std::type_index(typeid(Device));
}

bool Any::IsTensor() const noexcept {
    return type() == std::type_index(typeid(Tensor));
}

uint32_t Any::use_count() const noexcept {
    return has_value() ? ptr_->use_count() : 0;
}


double Any::ToDouble() const {
    CHECK(IsFloatingPoint()) << "Expected Double.";
    return cast<double>();
}

bool Any::ToBool() const {
    CHECK(IsBool()) << "Expected Bool.";
    return cast<bool>();
}

void* Any::ToVoidPtr() const {
    CHECK(IsVoidPtr()) << "Expected VoidPtr.";
    return cast<void*>();
}

String Any::ToString() const {
    CHECK(IsString()) << "Expected String.";
    return cast<String>();
}

Device Any::ToDevice() const {
    CHECK(IsDevice()) << "Expected Device.";
    return cast<Device>();
}

Tensor Any::ToTensor() const {
    CHECK(IsTensor()) << "Expected Tensor.";
    return cast<Tensor>();
}

bool Any::unique() const noexcept {
    return use_count() == 1;
}

int64_t Any::ToInt() const {
    CHECK(IsInteger()) << "Expected Int.";
    return cast<int64_t>();
}

bool Any::operator==(std::nullptr_t) const noexcept {
    return has_value() ? dynamic_cast<Holder<std::nullptr_t>*>(ptr_.get()) != nullptr : true;
}

bool Any::operator!=(std::nullptr_t p) const noexcept {
    return !operator==(p);
}

/*
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
*/

}// namespace aethermind