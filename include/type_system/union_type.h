//
// Created by richard on 10/18/25.
//

#ifndef AETHERMIND_TYPE_SYSTEM_UNION_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_UNION_TYPE_H

#include "type_system/type.h"

namespace aethermind {

class UnionType;
using UnionTypePtr = std::shared_ptr<UnionType>;
class UnionType : public SharedType {
public:
    bool IsUnionType() const override;

    bool Equals(const Type& rhs) const override;

    bool IsSubtypeOfImpl(const Type& rhs) const override;

    ArrayView<TypePtr> GetContainedTypes() const override {
        return types_;
    }

    bool HasFreeVars() const override {
        return has_free_variables_;
    }

    // just for test
    ArrayView<TypePtr> getTypes() const {
        return types_;
    }

    AM_NODISCARD String str() const override {
        return union_str(nullptr, false);
    }

    std::optional<TypePtr> to_optional() const;

    bool canHoldType(const Type& type) const;

    static UnionTypePtr Create(const std::vector<TypePtr>& ref);

    TypePtr CreateWithContainedTypes(const std::vector<TypePtr>&) const override;

    static constexpr auto Kind = TypeKind::UnionType;

protected:
    explicit UnionType(const std::vector<TypePtr>& types, TypeKind kind = TypeKind::UnionType);

    String union_str(const TypePrinter& printer = nullptr, bool is_annotation_str = false) const;

    std::vector<TypePtr> types_;
    bool can_hold_none_;
    bool has_free_variables_;
};

class OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
class OptionalType : public UnionType {
public:
    bool IsUnionType() const override {
        return true;
    }

    AM_NODISCARD String str() const override {
        return get_element_type()->str() + "?";
    }

    bool Equals(const Type& rhs) const override;

    bool IsSubtypeOfImpl(const Type& other) const override;

    const TypePtr& get_element_type() const {
        return contained_type_;
    }

    static OptionalTypePtr Create(const TypePtr& contained);

    TypePtr CreateWithContainedTypes(const std::vector<TypePtr>& contained_types) const override;

    static TypePtr Get(const TypePtr& inner);

    static TypePtr OfTensor();

    static constexpr auto Kind = TypeKind::OptionalType;

private:
    explicit OptionalType(const TypePtr& contained);

    TypePtr contained_type_;
};

void StandardizeVectorForUnion(const std::vector<TypePtr>& ref, std::vector<TypePtr>& need_to_fill);
void StandardizeVectorForUnion(std::vector<TypePtr>& to_flatten);

}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_UNION_TYPE_H
