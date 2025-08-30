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

// TODO: not complete
UnionTypePtr UnionType::create(std::vector<TypePtr> ref) {
    UnionTypePtr union_type(new UnionType(std::move(ref)));
}

bool UnionType::equals(const Type& rhs) const {
    if (auto union_rhs = rhs.cast<UnionType>()) {
        if (this->containedTypeSize() != rhs.containedTypeSize()) {
            return false;
        }
        // Check that all the types in `this->types_` are also in
        // `union_rhs->types_`
        return std::all_of(this->containedTypes().begin(), this->containedTypes().end(),
                           [&](TypePtr lhs_type) {
                               return std::any_of(union_rhs->containedTypes().begin(),
                                                  union_rhs->containedTypes().end(),
                                                  [&](const TypePtr& rhs_type) {
                                                      return *lhs_type == *rhs_type;
                                                  });
                           });
    }
}

std::string UnionType::union_str(const TypePrinter& printer, bool is_annotation_str) const {
    std::stringstream ss;

    bool can_hold_numbertype = this->canHoldType(*NumberType::Global());
    std::vector<TypePtr> number_types{IntType::Global(), FloatType::Global(), ComplexType::Global()};
    auto is_numbertype = [&](const TypePtr& lhs) {
        for (const auto& rhs: number_types) {
            if (*lhs == *rhs) {
                return true;
            }
        }
        return false;
    };
    std::string open_delimeter = is_annotation_str ? "[" : "(";
    std::string close_delimeter = is_annotation_str ? "]" : ")";
    ss << "Union" + open_delimeter;

    bool printed = false;
    for (size_t i = 0; i < types_.size(); ++i) {
        if (!can_hold_numbertype || !is_numbertype(types_[i])) {
            if (i > 0) {
                ss << ", ";
                printed = true;
            }
            if (is_annotation_str) {
                ss << this->containedTypes()[i]->annotation_str(printer);
            } else {
                ss << this->containedTypes()[i]->str();
            }
        }
    }

    if (can_hold_numbertype) {
        if (printed) {
            ss << ", ";
        }
        if (is_annotation_str) {
            ss << NumberType::Global()->annotation_str(printer);
        } else {
            ss << NumberType::Global()->str();
        }
    }

    ss << close_delimeter;
    return ss.str();
}

OptionalType::OptionalType(const TypePtr& contained)
    : UnionType({contained, NoneType::Global()}, TypeKind::OptionalType) {
}

OptionalTypePtr OptionalType::create(const TypePtr& contained) {
    return OptionalTypePtr(new OptionalType(contained));
}


}// namespace aethermind