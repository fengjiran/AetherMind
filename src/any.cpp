//
// Created by 赵丹 on 2025/8/24.
//
#include "any.h"
#include "container/string.h"
#include "device.h"
#include "tensor.h"
#include "type_system/tensor_type.h"
#include "type_system/type.h"

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

SingletonOrSharedTypePtr<Type> Any::GetTypePtr() const noexcept {
    if (!has_value()) {
        return NoneType::Global();
    }

    if (IsBool()) {
        return BoolType::Global();
    }

    if (IsInteger()) {
        return IntType::Global();
    }

    if (IsFloatingPoint()) {
        return FloatType::Global();
    }

    if (IsString()) {
        return StringType::Global();
    }

    if (IsDevice()) {
        return DeviceObjType::Global();
    }

    if (IsTensor()) {
        return TensorType::Create(ToTensor());
    }

    CHECK(false) << "The type is unknown!";
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
    return has_value() ? ptr_->GetUnderlyingPtr() : nullptr;
}

bool Any::IsNone() const noexcept {
    return !has_value();
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

bool Any::IsMap() const noexcept {
    return has_value() ? ptr_->IsMap() : false;
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

String Any::ToNone() const noexcept {
    CHECK(IsNone());
    return "None";
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

bool AnyEqual::operator()(const Any& lhs, const Any& rhs) const {
    if (!(lhs.has_value() || rhs.has_value())) {
        return true;
    }

    if (!(lhs.has_value() && rhs.has_value())) {
        return false;
    }

    if (lhs.type() != rhs.type()) {
        return false;
    }

    if (lhs.IsInteger()) {
        return lhs.ToInt() == rhs.ToInt();
    }

    if (lhs.IsFloatingPoint()) {
        return lhs.ToDouble() == rhs.ToDouble();
    }

    if (lhs.IsBool()) {
        return lhs.ToBool() == rhs.ToBool();
    }

    if (lhs.IsString()) {
        return lhs.ToString() == rhs.ToString();
    }

    if (lhs.IsDevice()) {
        return lhs.ToDevice() == rhs.ToDevice();
    }

    return lhs.ptr_ == rhs.ptr_;
}

// TODO: any hash
size_t AnyHash::operator()(const Any& v) const {
    if (!v.has_value()) {
        return 0;
    }

    if (v.IsBool()) {
        return get_hash(v.ToBool());
    }

    if (v.IsInteger()) {
        return get_hash(v.ToInt());
    }

    if (v.IsFloatingPoint()) {
        return get_hash(v.ToDouble());
    }

    if (v.IsString()) {
        return get_hash(v.ToString());
    }

    // if (v.IsDevice()) {
    //     return get_hash(v.ToDevice());
    // }

    CHECK(false) << "Unhashable type: '" << v.GetTypePtr()->ReprStr() << "'";
    AETHERMIND_UNREACHABLE();
}


}// namespace aethermind