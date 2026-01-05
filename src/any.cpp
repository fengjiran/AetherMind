//
// Created by 赵丹 on 2025/8/24.
//
#include "any.h"
#include "container/string.h"
#include "tensor.h"
#include "type_system/tensor_type.h"
#include "type_system/type.h"

namespace aethermind {

#ifdef USE_VISITOR_PATTERN

Any::Any(const Any& other) : type_info_cache_(other.type_info_cache_) {
    auto visitor = [&]<typename T>(const T& arg) {
        using U = std::decay_t<T>;
        if constexpr (std::is_same_v<U, SmallObject>) {
            data_ = arg;
        } else if constexpr (std::is_same_v<U, std::unique_ptr<HolderBase>>) {
            data_ = arg->Clone();
        }
    };

    std::visit(visitor, other.data_);
}

const HolderBase* Any::GetHolderPtr() const {
    auto visitor = []<typename T>(const T& arg) -> const HolderBase* {
        using U = std::decay_t<T>;
        if constexpr (std::is_same_v<U, SmallObject>) {
            return reinterpret_cast<const HolderBase*>(arg.local_buffer);
        } else if constexpr (std::is_same_v<U, std::unique_ptr<HolderBase>>) {
            return arg.get();
        } else {
            return nullptr;
        }
    };

    return std::visit(visitor, data_);
}

void* Any::GetDataPtr() {
    auto visitor = []<typename T>(T& arg) -> void* {
        using U = std::decay_t<T>;
        if constexpr (std::is_same_v<U, SmallObject>) {
            return arg.GetDataPtr();
        } else if constexpr (std::is_same_v<U, std::unique_ptr<HolderBase>>) {
            return arg->GetDataPtr();
        } else {
            return nullptr;
        }
    };

    return std::visit(visitor, data_);
}

#else

Any::Any(const Any& other) : type_info_cache_(other.type_info_cache_) {
    switch (other.data_.index()) {
        case 0:// std::monostate
            data_ = std::monostate{};
            break;
        case 1:// small object
            data_ = std::get<SmallObject>(other.data_);
            break;
        case 2:// large object
            data_ = std::get<std::unique_ptr<HolderBase>>(other.data_)->Clone();
            break;
        default:// monostate
            break;
    }
}

const HolderBase* Any::GetHolderPtr() const {
    switch (data_.index()) {
        case 0:// std::monostate
            return nullptr;
        case 1:// small object
            return reinterpret_cast<const HolderBase*>(std::get<SmallObject>(data_).local_buffer);
        case 2:// large object
            return std::get<std::unique_ptr<HolderBase>>(data_).get();
        default:// monostate
            return nullptr;
    }
}

void* Any::GetDataPtr() {
    switch (data_.index()) {
        case 0:// std::monostate
            return nullptr;
        case 1:// small object
            return std::get<SmallObject>(data_).GetDataPtr();
        case 2:
            return std::get<std::unique_ptr<HolderBase>>(data_)->GetDataPtr();
        default:
            return nullptr;
    }
}

#endif


const void* Any::GetDataPtr() const {
    return const_cast<Any*>(this)->GetDataPtr();
}

std::type_index Any::type() const {
    if (has_value()) {
        if (type_info_cache_ == std::type_index(typeid(void))) {
            type_info_cache_ = GetHolderPtr()->type();
        }
        return type_info_cache_;
    }
    AETHERMIND_THROW(BadAnyCast) << "Any has no value.";
    AETHERMIND_UNREACHABLE();
}

bool Any::IsTensor() const noexcept {
    return CheckType<Tensor>();
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

void Any::DebugPrint(std::ostream& os) const {
    if (const auto* holder_ptr = GetHolderPtr()) {
        os << "{type: " << details::demangle(holder_ptr->type().name()) << ", value: ";
        holder_ptr->print(os);
        os << "}";
    } else {
        os << "{empty}";
    }
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

    return lhs.GetHolderPtr()->EqualsTo(rhs.GetHolderPtr());

    // if (lhs.IsInteger()) {
    //     return lhs.ToInt() == rhs.ToInt();
    // }
    //
    // if (lhs.IsFloatingPoint()) {
    //     return lhs.ToDouble() == rhs.ToDouble();
    // }
    //
    // if (lhs.IsBool()) {
    //     return lhs.ToBool() == rhs.ToBool();
    // }
    //
    // if (lhs.IsString()) {
    //     return lhs.ToString() == rhs.ToString();
    // }
    //
    // if (lhs.IsDevice()) {
    //     return lhs.ToDevice() == rhs.ToDevice();
    // }
    //
    // return lhs.GetDataPtr() == rhs.GetDataPtr();
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

std::ostream& operator<<(std::ostream& os, const Any& any) {
    if (const auto* holder_ptr = any.GetHolderPtr()) {
        holder_ptr->print(os);
    } else {
        os << "{empty}";
    }
    return os;
}


}// namespace aethermind