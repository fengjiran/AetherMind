//
// Created by richard on 10/1/25.
//

#ifndef AETHERMIND_FUNCTION_SCHEMA_H
#define AETHERMIND_FUNCTION_SCHEMA_H

#include "any.h"
#include "container/string.h"
#include "type.h"

#include <optional>

namespace aethermind {

class Argument {
public:
    Argument() : Argument("", nullptr, std::nullopt, std::nullopt) {}

    Argument(String name, TypePtr type, std::optional<Any> default_value = std::nullopt,
             std::optional<int32_t> N = std::nullopt)
        : name_(std::move(name)), type_(std::move(type)), default_value_(std::move(default_value)),
          N_(N) {}

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

    NODISCARD const std::optional<int32_t>& N() const {
        return N_;
    }

    NODISCARD const std::optional<Any>& default_value() const {
        return default_value_;
    }

    void swap(Argument& other) noexcept {
        std::swap(name_, other.name_);
        std::swap(type_, other.type_);
        std::swap(default_value_, other.default_value_);
        std::swap(N_, other.N_);
    }

private:
    String name_;
    TypePtr type_;
    std::optional<Any> default_value_;
    std::optional<int32_t> N_;
};

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_SCHEMA_H
