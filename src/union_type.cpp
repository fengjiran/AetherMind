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

// Helper function for `standardizeUnion`
//
// NB: If we have types `T1`, `T2`, `T3`, and `PARENT_T` such that `T1`,
// `T2`, and `T2` are children of `PARENT_T`, then `unifyTypes(T1, T2)`
// will return `PARENT_T`. This could be a problem if we didn't want our
// Union to also be able to take `T3 `. In our current type hierarchy,
// this isn't an issue--most types SHOULD be unified even if the parent
// type wasn't in the original vector. However, later additions to the
// type system might necessitate reworking `get_supertype`
void filterDuplicateSubtypes(std::vector<TypePtr>& types) {
    if (types.empty()) {
        return;
    }

    auto get_supertype = [](const TypePtr& t1, const TypePtr& t2) -> std::optional<TypePtr> {
        if ((t1->is_subtype_of(*NoneType::Global()) && !t2->is_subtype_of(*NoneType::Global())) ||
            (!t1->is_subtype_of(*NoneType::Global()) && t2->is_subtype_of(*NoneType::Global()))) {
            return std::nullopt;
        }
        return unify_types(t1, t2, false);
    };

    // Coalesce types and delete all duplicates. Moving from right to left
    // through the vector, we try to unify the current element (`i`) with
    // each element (`j`) before the "new" end of the vector (`end`).
    // If we're able to unify the types at `types[i]` and `types[j]`, we
    // decrement `end`, swap `types[j]` with the unified type, and
    // break. Otherwise, we keep `end` where it is to signify that the
    // new end of the vector hasn't shifted
    auto end_idx = types.size() - 1;
    for (size_t i = types.size() - 1; i > 0; --i) {
        size_t j = std::min(i - 1, end_idx);
        while (true) {
            if (auto unified = get_supertype(types[i], types[j])) {
                types[j] = unified.value();
                types[i] = types[end_idx];
                --end_idx;
                break;
            }
            if (j == 0) {
                break;
            }
            --j;
        }
    }
    // Cut off the vector's tail so that `end` is the real last element
    types.erase(types.begin() + static_cast<std::ptrdiff_t>(end_idx) + 1, types.end());
}

static void sortUnion(std::vector<TypePtr>& types) {
    // We want the elements to be sorted so we can easily compare two
    // UnionType objects for equality in the future. Note that this order
    // is guaranteed to be stable since we've already coalesced any
    // possible types
    auto cmp = [](const TypePtr& a, const TypePtr& b) {
        return a->kind() != b->kind() ? a->kind() < b->kind() : a->str() < b->str();
    };
    std::sort(types.begin(), types.end(), cmp);
}

void standardizeVectorForUnion(const std::vector<TypePtr>& ref, std::vector<TypePtr>& need_to_fill) {
    for (const auto& type: ref) {
        flattenUnion(type, need_to_fill);
    }
    filterDuplicateSubtypes(need_to_fill);
    sortUnion(need_to_fill);
}

void standardizeVectorForUnion(std::vector<TypePtr>& to_flatten) {
    std::vector<TypePtr> need_to_fill;
    standardizeVectorForUnion(to_flatten, need_to_fill);
    to_flatten = std::move(need_to_fill);
}

UnionType::UnionType(std::vector<TypePtr> types, TypeKind kind) : SharedType(kind) {
    CHECK(!types.empty()) << "Can not create an empty Union type.";
    standardizeVectorForUnion(types, types_);

    if (types_.size() == 1) {
        std::stringstream msg;
        msg << "After type unification was performed, the Union with the "
            << "original types {";
        for (size_t i = 0; i < types.size(); ++i) {
            msg << types[i]->repr_str();
            if (i > 0) {
                msg << ",";
            }
            msg << " ";
        }
        msg << "} has the single type " << types_[0]->repr_str()
            << ". Use the common supertype instead of creating a Union"
            << "type";
        CHECK(false) << msg.str();
    }

    can_hold_none_ = false;
    has_free_variables_ = false;
    for (const auto& t: types_) {
        if (t->kind() == NoneType::Kind) {
            can_hold_none_ = true;
        }
        if (t->hasFreeVars()) {
            has_free_variables_ = true;
        }
    }
}

bool UnionType::canHoldType(const Type& type) const {
    if (type.cast<NumberType>()) {
        return canHoldType(*IntType::Global()) &&
               canHoldType(*FloatType::Global()) &&
               canHoldType(*ComplexType::Global());
    }
    return std::any_of(this->containedTypes().begin(), this->containedTypes().end(),
                       [&](const TypePtr& inner) {
                           return type.is_subtype_of(*inner);
                       });
}


UnionTypePtr UnionType::create(std::vector<TypePtr> ref) {
    UnionTypePtr union_type(new UnionType(std::move(ref)));
    bool int_found = false;
    bool float_found = false;
    bool complex_found = false;
    bool nonetype_found = false;

    auto f = [&](const TypePtr& t) {
        if (t == IntType::Global()) {
            int_found = true;
        } else if (t == FloatType::Global()) {
            float_found = true;
        } else if (t == ComplexType::Global()) {
            complex_found = true;
        } else if (t == NoneType::Global()) {
            nonetype_found = true;
        }
    };

    for (const auto& t: union_type->containedTypes()) {
        f(t);
    }

    bool numbertype_found = int_found && float_found && complex_found;
    if (nonetype_found) {
        if (union_type->containedTypeSize() == 4 && numbertype_found) {
            return OptionalType::create(NumberType::Global());
        }

        if (union_type->containedTypeSize() == 2) {
            auto not_none = union_type->containedTypes()[0] != NoneType::Global()
                                    ? union_type->containedTypes()[0]
                                    : union_type->containedTypes()[1];
            return OptionalType::create(not_none);
        }
    }

    return union_type;
}

std::optional<TypePtr> UnionType::to_optional() const {
    if (!canHoldType(*NoneType::Global())) {
        return std::nullopt;
    }

    std::vector<TypePtr> copied_types = this->containedTypes().vec();
    auto maybe_opt = create(std::move(copied_types));
    if (maybe_opt->kind() == Kind) {
        return std::nullopt;
    }
    return maybe_opt;
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

    if (auto optional_rhs = rhs.cast<OptionalType>()) {
        if (optional_rhs->get_element_type() == NumberType::Global()) {
            return this->containedTypeSize() == 4 && this->can_hold_none_ && this->canHoldType(*NumberType::Global());
        }
        auto optional_lhs = this->to_optional();
        return optional_lhs && *optional_rhs == *optional_lhs.value()->expect<OptionalType>();
    }

    if (rhs.kind() == NumberType::Kind) {
        return this->containedTypeSize() == 3 && canHoldType(*NumberType::Global());
    }

    return false;
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