//
// Created by 赵丹 on 2025/8/13.
//

#include "type.h"
#include "type_traits.h"

namespace aethermind {

const char* TypeKindToString(TypeKind kind) {
#define CASE(T)       \
    case TypeKind::T: \
        return #T;

    switch (kind) {
        AETHERMIND_FORALL_TYPES(CASE)
    }
#undef CASE
    return "";
}

bool Type::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
    if (rhs.kind() == TypeKind::AnyType || *this == rhs) {
        return true;
    }

    if (auto opt_rhs = rhs.cast<OptionalType>()) {
        return isSubtypeOfExt(*opt_rhs->get_element_type(), why_not);
    }

    // Check if `this` is a subtype of the types within the Union
    if (auto union_type = rhs.cast<UnionType>()) {
        return std::any_of(union_type->containedTypes().begin(), union_type->containedTypes().end(),
                           [&](const TypePtr& inner) {
                               return isSubtypeOfExt(*inner, why_not);
                           });
    }

    // TODO: handle dynamic type
    // if (auto dyn = rhs.castRaw<DynamicType>()) {
    //     return DynamicType::create(*this)->isSubtypeOf(*dyn);
    // }

    return false;
}


bool NumberType::equals(const Type& other) const {
    if (auto union_type = other.cast<UnionType>()) {
        return union_type->containedTypeSize() == 3 && union_type->canHoldType(*Global());
    }
    return kind() == other.kind();
}

// TODO: unify types impl
static std::optional<TypePtr> unify_types_impl(const TypePtr& t1, const TypePtr& t2,
                                               bool default_to_union = false,
                                               const TypePtr& type_hint = nullptr) {
    return std::nullopt;
}

std::optional<TypePtr> unify_types(const TypePtr& t1, const TypePtr& t2, bool default_to_union, const TypePtr& type_hint) {
    auto unified = unify_types_impl(t1, t2, default_to_union, type_hint);
    if (default_to_union && !unified) {
        return UnionType::create({t1, t2});
    }
    return unified;
}


}// namespace aethermind
