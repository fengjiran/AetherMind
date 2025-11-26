//
// Created by richard on 10/1/25.
//

#ifndef AETHERMIND_FUNCTION_SCHEMA_H
#define AETHERMIND_FUNCTION_SCHEMA_H

#include "alias_info.h"
#include "container/string.h"
#include "operator_name.h"
#include "type_system/tensor_type.h"
#include "type_system/type.h"

#include <optional>

namespace aethermind {

class Argument {
public:
    Argument(String name,
             TypePtr fake_type,
             TypePtr real_type,
             std::optional<int32_t> N = std::nullopt,
             std::optional<Any> default_value = std::nullopt,
             bool kwarg_only = false,
             std::optional<AliasInfo> alias_info = std::nullopt);

    explicit Argument(String name = "",
                      const TypePtr& type = nullptr,
                      std::optional<int32_t> N = std::nullopt,
                      std::optional<Any> default_value = std::nullopt,
                      bool kwarg_only = false,
                      std::optional<AliasInfo> alias_info = std::nullopt);

    Argument(const Argument& other);
    Argument(Argument&& other) noexcept = default;

    Argument& operator=(const Argument& other);
    Argument& operator=(Argument&& other) noexcept;

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

    NODISCARD const AliasInfo* alias_info() const {
        return alias_info_.get();
    }

    NODISCARD bool IsKwargOnly() const {
        return kwarg_only_;
    }

    NODISCARD bool IsOut() const {
        return is_out_;
    }

    NODISCARD bool IsInferredType() const;

    NODISCARD Argument CloneWithType(const TypePtr& new_type) const;

    NODISCARD String TypeMismatchMsg(const String& actual_type) const;

    void swap(Argument& other) noexcept;

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
    std::unique_ptr<AliasInfo> alias_info_;
    // whether the argument is marked as out, like int& ref
    bool is_out_;
};

inline bool operator==(const Argument& lhs, const Argument& rhs) {
    return lhs.name() == rhs.name() &&
           *lhs.type() == *rhs.type() &&
           lhs.N() == rhs.N() &&
           lhs.default_value() == rhs.default_value() &&
           lhs.IsKwargOnly() == rhs.IsKwargOnly() &&
           (lhs.alias_info() == rhs.alias_info() ||
            (lhs.alias_info() != nullptr && rhs.alias_info() != nullptr &&
             *lhs.alias_info() == *rhs.alias_info()));
}

inline bool operator!=(const Argument& lhs, const Argument& rhs) {
    return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const Argument& arg);

enum class ArgDirection {
    INPUT,
    OUTPUT
};

struct SchemaArgument {
    ArgDirection direction;
    size_t index;

    SchemaArgument(ArgDirection arg_dir, size_t idx) : direction(arg_dir), index(idx) {}
    bool operator==(const SchemaArgument& other) const {
        return direction == other.direction && index == other.index;
    }
};

class FunctionSchema {
public:
    FunctionSchema(String name, String overload_name, std::vector<Argument> arguments,
                   std::vector<Argument> returns, bool is_var_args, bool is_var_returns);

    FunctionSchema(Symbol name, String overload_name, std::vector<Argument> arguments,
                   std::vector<Argument> returns, bool is_var_args, bool is_var_returns);

    NODISCARD const std::vector<Argument>& arguments() const {
        return arguments_;
    }

private:
    OperatorName name_;
    std::vector<Argument> arguments_;
    std::vector<Argument> returns_;

    // if true then this schema takes an arbitrary number of additional arguments
    // after the argument specified in arguments,
    // currently this is used primarily to represent 'primitive' operators whose
    // arguments are not checked by schema
    bool is_var_args_;
    bool is_var_returns_;

    void Check() const;
};

std::ostream& operator<<(std::ostream& os, const FunctionSchema& schema);

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_SCHEMA_H
