//
// Created by 赵丹 on 2025/8/24.
//
#include "any.h"
#include "container/string.h"
#include "tensor.h"
#include "type_system/tensor_type.h"
#include "type_system/type.h"

namespace aethermind {

const HolderBase* Any::GetHolderPtr() const {
    if (IsSmallObject()) {
        return reinterpret_cast<const HolderBase*>(std::get<SmallObject>(data_).local_buffer);
    }

    if (IsLargeObject()) {
        return std::get<std::unique_ptr<HolderBase>>(data_).get();
    }

    return nullptr;
}

void* Any::GetUnderlyingPtr() const {
    if (IsSmallObject()) {
        return std::get<SmallObject>(data_).GetUnderlyingPtr();
    }

    if (IsLargeObject()) {
        return std::get<std::unique_ptr<HolderBase>>(data_)->GetUnderlyingPtr();
    }

    return nullptr;
}

std::type_index Any::type() const {
    if (has_value()) {
        return GetHolderPtr()->type();
    }
    AETHERMIND_THROW(BadAnyCast) << "Any has no value.";
    AETHERMIND_UNREACHABLE();
}


bool Any::IsTensor() const noexcept {
    return type() == std::type_index(typeid(Tensor));
}

Tensor Any::ToTensor() const {
    CHECK(IsTensor()) << "Expected Tensor.";
    return cast<Tensor>();
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

    return lhs.GetUnderlyingPtr() == rhs.GetUnderlyingPtr();
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

    if (v.IsDevice()) {
        return get_hash(v.ToDevice());
    }

    CHECK(false) << "Unhashable type: '" << v.GetTypePtr()->ReprStr() << "'";
    AETHERMIND_UNREACHABLE();
}

}// namespace aethermind