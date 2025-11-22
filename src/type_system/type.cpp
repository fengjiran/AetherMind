//
// Created by 赵丹 on 2025/8/13.
//

#include "type_system/union_type.h"

namespace aethermind {

String TypeKindToString(TypeKind kind) {
#define CASE(T)       \
    case TypeKind::T: \
        return #T;

    switch (kind) {
        AETHERMIND_ALL_TYPES(CASE)
    }
#undef CASE
    return "";
}

TypeKind Type::kind() const {
    return kind_;
}

bool Type::IsSymmetric() const {
    return true;
}

bool Type::IsUnionType() const {
    return false;
}

bool Type::requires_grad() const {
    const auto types = GetContainedTypes();
    return std::any_of(types.begin(), types.end(),
                       [](const TypePtr& t) { return t->requires_grad(); });
}

ArrayView<TypePtr> Type::GetContainedTypes() const {
    return {};
}

TypePtr Type::GetContainedType(size_t i) const {
    return GetContainedTypes().at(i);
}

size_t Type::GetContainedTypeSize() const {
    return GetContainedTypes().size();
}

TypePtr Type::CreateWithContainedTypes(const std::vector<TypePtr>&) const {
    CHECK(false) << "CreateWithContainedTypes() is not implemented: " << str();
    AETHERMIND_UNREACHABLE();
}

TypePtr Type::WithContainedTypes(const std::vector<TypePtr>& contained_types) {
    auto cur_contained_types = GetContainedTypes();
    CHECK(!cur_contained_types.empty() && cur_contained_types.size() == contained_types.size());
    if (cur_contained_types.equals(contained_types)) {
        return std::static_pointer_cast<Type>(static_cast<SharedType*>(this)->shared_from_this());
    }
    return CreateWithContainedTypes(contained_types);
}

bool Type::HasFreeVars() const {
    return false;
}

String Type::Annotation(const TypePrinter& printer) const {
    if (printer) {
        if (auto renamed = printer(*this)) {
            return *renamed;
        }
    }
    return this->AnnotationImpl(printer);
}

String Type::Annotation() const {
    // Overload instead of define a default value for `printer` to help
    // debuggers out.
    return Annotation(nullptr);
}

String Type::ReprStr() const {
    return Annotation();
}

bool Type::IsModule() const {
    return false;
}

bool Type::IsSubtypeOfExt(const Type& other, std::ostream* why_not) const {
    if (other.kind() == TypeKind::AnyType || *this == other) {
        return true;
    }

    if (auto opt_rhs = other.CastTo<OptionalType>()) {
        return IsSubtypeOfExt(*opt_rhs->get_element_type(), why_not);
    }

    // Check if `this` is a subtype of the types within the Union
    if (auto union_type = other.CastTo<UnionType>()) {
        return std::any_of(union_type->GetContainedTypes().begin(), union_type->GetContainedTypes().end(),
                           [&](const TypePtr& inner) {
                               return IsSubtypeOfExt(*inner, why_not);
                           });
    }

    // TODO: handle dynamic type
    // if (auto dyn = rhs.castRaw<DynamicType>()) {
    //     return DynamicType::create(*this)->isSubtypeOf(*dyn);
    // }

    return false;
}


bool NumberType::Equals(const Type& other) const {
    if (auto union_type = other.CastTo<UnionType>()) {
        return union_type->GetContainedTypeSize() == 3 && union_type->canHoldType(*Global());
    }
    return kind() == other.kind();
}

bool NumberType::IsSubtypeOfExt(const Type& other, std::ostream* why_not) const {
    if (auto union_type = other.CastTo<UnionType>()) {
        return union_type->canHoldType(*Global());
    }
    return Type::IsSubtypeOfExt(other, why_not);
}


// TODO: unify types impl
static std::optional<TypePtr> unify_types_impl(const TypePtr& t1, const TypePtr& t2,
                                               bool default_to_union = false,
                                               const TypePtr& type_hint = nullptr) {
    // check direct subtyping relation
    if (t1->IsSubtypeOf(*t2)) {
        return t2;
    }

    if (t2->IsSubtypeOf(*t1)) {
        return t1;
    }

    return std::nullopt;
}

std::optional<TypePtr> unify_types(const TypePtr& t1, const TypePtr& t2, bool default_to_union, const TypePtr& type_hint) {
    auto unified = unify_types_impl(t1, t2, default_to_union, type_hint);
    if (default_to_union && !unified) {
        return UnionType::Create({t1, t2});
    }
    return unified;
}


}// namespace aethermind
