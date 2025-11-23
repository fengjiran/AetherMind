//
// Created by richard on 10/1/25.
//

#ifndef AETHERMIND_FUNCTION_SCHEMA_H
#define AETHERMIND_FUNCTION_SCHEMA_H

#include "any.h"
#include "container/string.h"
#include "type_system/tensor_type.h"
#include "type_system/type.h"

#include <optional>

namespace aethermind {

class Argument {
public:
    explicit Argument(String name = "",
                      TypePtr type = nullptr,
                      std::optional<int32_t> N = std::nullopt,
                      std::optional<Any> default_value = std::nullopt,
                      bool kwarg_only = false,
                      bool is_out = false)
        : name_(std::move(name)),
          type_(std::move(type)),
          N_(N),
          default_value_(std::move(default_value)),
          kwarg_only_(kwarg_only),
          is_out_(is_out) {}

    Argument(String name,
             TypePtr fake_type,
             TypePtr real_type,
             std::optional<int32_t> N = std::nullopt,
             std::optional<Any> default_value = std::nullopt,
             bool kwarg_only = false,
             bool is_out = false)
        : name_(std::move(name)),
          type_(fake_type ? std::move(fake_type) : TensorType::Get()),
          real_type_(real_type ? std::move(real_type) : type_),
          N_(N),
          default_value_(std::move(default_value)),
          kwarg_only_(kwarg_only),
          is_out_(is_out) {}

    Argument(const Argument& other) = default;
    Argument(Argument&& other) noexcept = default;

    Argument& operator=(const Argument& other) {
        Argument(other).swap(*this);
        return *this;
    }

    Argument& operator=(Argument&& other) noexcept {
        Argument(std::move(other)).swap(*this);
        return *this;
    }

    NODISCARD const String& name() const {
        return name_;
    }

    NODISCARD const TypePtr& type() const {
        return type_;
    }

    NODISCARD const TypePtr& real_type() const {
        return real_type_;
    }

    NODISCARD const std::optional<int32_t>& N() const {
        return N_;
    }

    NODISCARD const std::optional<Any>& default_value() const {
        return default_value_;
    }

    NODISCARD bool IsKwargOnly() const {
        return kwarg_only_;
    }

    NODISCARD bool IsOut() const {
        return is_out_;
    }

    NODISCARD bool IsInferredType() const {
        bool is_inferred_type = false;
        CHECK(type_ != nullptr);
        if (auto t = type_->Cast<TensorType>()) {
            is_inferred_type = t->IsInferredType();
        }
        return is_inferred_type;
    }

    NODISCARD Argument CloneWithType(const TypePtr& new_type) const {
        return Argument(name_, new_type, N_, default_value_, kwarg_only_);
    }

    NODISCARD String TypeMismatchMsg(const String& actual_type) const {
        String inferred_type_hint;
        if (IsInferredType()) {
            inferred_type_hint = "Inferred type '";
            inferred_type_hint += name();
            inferred_type_hint += "' to be of type 'Tensor' because it was not annotated with an explicit type.\n";
        }
        return "Expected a value of type '" + type()->ReprStr() + "' for argument '" + name() +
               "', but instead found type '" + actual_type + ".\n" + inferred_type_hint;
    }

    void swap(Argument& other) noexcept {
        std::swap(name_, other.name_);
        std::swap(type_, other.type_);
        std::swap(real_type_, other.real_type_);
        std::swap(default_value_, other.default_value_);
        std::swap(N_, other.N_);
        std::swap(kwarg_only_, other.kwarg_only_);
        std::swap(is_out_, other.is_out_);
    }

private:
    String name_;
    TypePtr type_;
    TypePtr real_type_;// this is ScalarType, not int, e.g.
    // for list types, an optional statically known length for the list
    // e.g. for int[3]: type = ListType::ofInts(), N = 3
    // If present, this will allow scalars to be broadcast to this length to
    // become a list.
    std::optional<int32_t> N_;
    std::optional<Any> default_value_;
    // is this only specifiable as a keyword argument?
    bool kwarg_only_;
    // whether the argument is marked as out
    bool is_out_;
};

inline bool operator==(const Argument& lhs, const Argument& rhs) {
    return lhs.name() == rhs.name() &&
           *lhs.type() == *rhs.type() &&
           lhs.N() == rhs.N() &&
           lhs.default_value() == rhs.default_value() &&
           lhs.IsKwargOnly() == rhs.IsKwargOnly();
}

inline bool operator!=(const Argument& lhs, const Argument& rhs) {
    return !(lhs == rhs);
}

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_SCHEMA_H
