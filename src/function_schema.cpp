//
// Created by richard on 11/25/25.
//
#include "function_schema.h"

namespace aethermind {

Argument::Argument(String name,
                   TypePtr fake_type,
                   TypePtr real_type,
                   std::optional<int32_t> N,
                   std::optional<Any> default_value,
                   bool kwarg_only,
                   std::optional<AliasInfo> alias_info)
    : name_(std::move(name)),
      type_(fake_type ? std::move(fake_type) : TensorType::Get()),
      real_type_(real_type ? std::move(real_type) : type_),
      N_(N),
      default_value_(std::move(default_value)),
      kwarg_only_(kwarg_only),
      alias_info_(alias_info ? std::make_unique<AliasInfo>(std::move(*alias_info)) : nullptr) {
    bool is_alias = alias_info_ != nullptr && alias_info_->IsWrite();
    is_out_ = kwarg_only_ && is_alias;
}

Argument::Argument(String name,
                   const TypePtr& type,
                   std::optional<int32_t> N,
                   std::optional<Any> default_value,
                   bool kwarg_only,
                   std::optional<AliasInfo> alias_info)
    : Argument(std::move(name), type, type, N,
               std::move(default_value), kwarg_only, std::move(alias_info)) {}

Argument::Argument(const Argument& other)
    : name_(other.name_), type_(other.type_), real_type_(other.real_type_),
      N_(other.N_), default_value_(other.default_value_),
      kwarg_only_(other.kwarg_only_),
      alias_info_(other.alias_info_ ? std::make_unique<AliasInfo>(*other.alias_info_) : nullptr),
      is_out_(other.is_out_) {}

Argument& Argument::operator=(const Argument& other) {
    if (this != &other) {
        name_ = other.name_;
        type_ = other.type_;
        real_type_ = other.real_type_;
        N_ = other.N_;
        default_value_ = other.default_value_;
        kwarg_only_ = other.kwarg_only_;
        alias_info_ = other.alias_info_ ? std::make_unique<AliasInfo>(*other.alias_info_) : nullptr;
        is_out_ = other.is_out_;
    }
    return *this;
}

Argument& Argument::operator=(Argument&& other) noexcept {
    Argument(std::move(other)).swap(*this);
    return *this;
}

bool Argument::IsInferredType() const {
    bool is_inferred_type = false;
    CHECK(type_ != nullptr);
    if (auto t = type_->Cast<TensorType>()) {
        is_inferred_type = t->IsInferredType();
    }
    return is_inferred_type;
}

Argument Argument::CloneWithType(const TypePtr& new_type) const {
    return Argument(name_, new_type, N_, default_value_, kwarg_only_,
                    alias_info_ ? std::optional(*alias_info_) : std::nullopt);
}


String Argument::TypeMismatchMsg(const String& actual_type) const {
    String inferred_type_hint;
    if (IsInferredType()) {
        inferred_type_hint = "Inferred type '";
        inferred_type_hint += name();
        inferred_type_hint += "' to be of type 'Tensor' because it was not annotated with an explicit type.\n";
    }
    return "Expected a value of type '" + type()->ReprStr() + "' for argument '" + name() +
           "', but instead found type '" + actual_type + ".\n" + inferred_type_hint;
}

void Argument::swap(Argument& other) noexcept {
    std::swap(name_, other.name_);
    std::swap(type_, other.type_);
    std::swap(real_type_, other.real_type_);
    std::swap(default_value_, other.default_value_);
    std::swap(N_, other.N_);
    std::swap(kwarg_only_, other.kwarg_only_);
    std::swap(alias_info_, other.alias_info_);
    std::swap(is_out_, other.is_out_);
}

}// namespace aethermind