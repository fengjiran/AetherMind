//
// Created by 赵丹 on 2025/8/29.
//
#include "type.h"

namespace aethermind {

// Remove nested Optionals/Unions during the instantiation of a Union or
// an Optional. This populates `types` with all the types found during
// flattening. At the end of `flattenUnion`, `types` may have
// duplicates, but it will not have nested Optionals/Unions
void flattenUnion(const TypePtr& type, std::vector<TypePtr>& need_to_fill) {
    if (auto union_type = type->cast<UnionType>()) {
        for (const auto& inner_type: union_type->containedTypes()) {
            flattenUnion(inner_type, need_to_fill);
        }
    } else if (auto opt_type = type->cast<OptionalType>()) {
        const auto& inner_type = opt_type->get_element_type();
        flattenUnion(inner_type, need_to_fill);
        need_to_fill.emplace_back(NoneType::Global());
    } else if (type->kind() == NumberType::Kind) {
        need_to_fill.emplace_back(IntType::Global());
        need_to_fill.emplace_back(FloatType::Global());
        need_to_fill.emplace_back(ComplexType::Global());
    } else {
        need_to_fill.emplace_back(type);
    }
}

// TODO: UnionType constructor
UnionType::UnionType(std::vector<TypePtr> types, TypeKind kind) : SharedType(kind) {
}

OptionalType::OptionalType(const TypePtr& contained)
    : UnionType({contained, NoneType::Global()}, TypeKind::OptionalType) {
}

OptionalTypePtr OptionalType::create(const TypePtr& contained) {
    return OptionalTypePtr(new OptionalType(contained));
}


}// namespace aethermind