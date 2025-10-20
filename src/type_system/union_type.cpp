//
// Created by 赵丹 on 2025/8/29.
//
#include "type_system/union_type.h"

namespace aethermind {

std::optional<TypePtr> subtractTypeSetFrom(std::vector<TypePtr>& to_subtract, ArrayView<TypePtr> from) {
    std::vector<TypePtr> types;

    // Given a TypePtr `lhs`, this function says whether or not `lhs` (or
    // one of its parent types) is in the `to_subtract` vector
    auto should_subtract = [&](const TypePtr& lhs) -> bool {
        return std::any_of(to_subtract.begin(), to_subtract.end(),
                           [&](const TypePtr& rhs) {
                               return lhs->IsSubtypeOf(*rhs);
                           });
    };

    // Copy all the elements that should NOT be subtracted to the `types`
    // vector
    std::copy_if(from.begin(), from.end(), std::back_inserter(types),
                 [&](const TypePtr& t) {
                     return !should_subtract(t);
                 });

    if (types.empty()) {
        return std::nullopt;
    }
    if (types.size() == 1) {
        return types[0];
    }
    return UnionType::create(types);
}

// Remove nested Optionals/Unions during the instantiation of a Union or
// an Optional. This populates `types` with all the types found during
// flattening. At the end of this function, `types` may have
// duplicates, but it will not have nested Optionals/Unions
static void FlattenUnionTypes(const TypePtr& type, std::vector<TypePtr>& need_to_fill) {
    if (auto union_type = type->CastTo<UnionType>()) {
        for (const auto& t: union_type->GetContainedTypes()) {
            FlattenUnionTypes(t, need_to_fill);
        }
    } else if (auto opt_type = type->CastTo<OptionalType>()) {
        const auto& t = opt_type->get_element_type();
        FlattenUnionTypes(t, need_to_fill);
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
void RemoveDuplicateSubtypes(std::vector<TypePtr>& types) {
    if (types.empty()) {
        return;
    }

    auto get_supertype = [](const TypePtr& t1, const TypePtr& t2) -> std::optional<TypePtr> {
        if ((t1->IsSubtypeOf(*NoneType::Global()) && !t2->IsSubtypeOf(*NoneType::Global())) ||
            (!t1->IsSubtypeOf(*NoneType::Global()) && t2->IsSubtypeOf(*NoneType::Global()))) {
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

static void SortUnionTypes(std::vector<TypePtr>& types) {
    // We want the elements to be sorted so we can easily compare two
    // UnionType objects for equality in the future. Note that this order
    // is guaranteed to be stable since we've already coalesced any
    // possible types
    auto cmp = [](const TypePtr& a, const TypePtr& b) {
        return a->kind() != b->kind() ? a->kind() < b->kind() : a->str() < b->str();
    };
    std::sort(types.begin(), types.end(), cmp);
}

void StandardizeVectorForUnion(const std::vector<TypePtr>& ref, std::vector<TypePtr>& need_to_fill) {
    for (const auto& type: ref) {
        FlattenUnionTypes(type, need_to_fill);
    }
    RemoveDuplicateSubtypes(need_to_fill);
    SortUnionTypes(need_to_fill);
}

void StandardizeVectorForUnion(std::vector<TypePtr>& to_flatten) {
    std::vector<TypePtr> need_to_fill;
    StandardizeVectorForUnion(to_flatten, need_to_fill);
    to_flatten = std::move(need_to_fill);
}

UnionType::UnionType(const std::vector<TypePtr>& types, TypeKind kind) : SharedType(kind) {
    CHECK(!types.empty()) << "Can not create an empty Union type.";
    StandardizeVectorForUnion(types, types_);

    if (types_.size() == 1) {
        std::stringstream msg;
        msg << "After type unification was performed, the Union with the "
            << "original types {";
        for (size_t i = 0; i < types.size(); ++i) {
            msg << types[i]->ReprStr();
            if (i > 0) {
                msg << ",";
            }
            msg << " ";
        }
        msg << "} has the single type " << types_[0]->ReprStr()
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
        if (t->HasFreeVars()) {
            has_free_variables_ = true;
        }
    }
}

bool UnionType::canHoldType(const Type& type) const {
    if (type.CastTo<NumberType>()) {
        return canHoldType(*IntType::Global()) &&
               canHoldType(*FloatType::Global()) &&
               canHoldType(*ComplexType::Global());
    }
    return std::any_of(this->GetContainedTypes().begin(), this->GetContainedTypes().end(),
                       [&](const TypePtr& inner) {
                           return type.IsSubtypeOf(*inner);
                       });
}

bool UnionType::IsSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
    std::vector<const Type*> rhs_types;
    if (const auto union_rhs = rhs.CastTo<UnionType>()) {
        // Fast path
        if (this->GetContainedTypes() == rhs.GetContainedTypes()) {
            return true;
        }

        for (const auto& t: rhs.GetContainedTypes()) {
            rhs_types.push_back(t.get());
        }
    } else if (const auto optional_rhs = rhs.CastTo<OptionalType>()) {
        rhs_types.push_back(NoneType::Global().get());
        if (optional_rhs->get_element_type() == NumberType::Global()) {
            std::array<const Type*, 3> number_types{IntType::Global().get(), FloatType::Global().get(), ComplexType::Global().get()};
            rhs_types.insert(rhs_types.end(), number_types.begin(), number_types.end());
        } else {
            rhs_types.push_back(optional_rhs->get_element_type().get());
        }
    } else if (const auto number_rhs = rhs.CastTo<NumberType>()) {
        std::array<const Type*, 3> number_types{IntType::Global().get(), FloatType::Global().get(), ComplexType::Global().get()};
        rhs_types.insert(rhs_types.end(), number_types.begin(), number_types.end());
    } else {
        rhs_types.push_back(&rhs);
    }

    return std::all_of(this->GetContainedTypes().begin(), this->GetContainedTypes().end(),
                       [&](const TypePtr& lhs_type) -> bool {
                           return std::any_of(rhs_types.begin(), rhs_types.end(),
                                              [&](const Type* rhs_type) -> bool {
                                                  return lhs_type->IsSubtypeOfExt(*rhs_type, why_not);
                                              });
                       });
}


UnionTypePtr UnionType::create(const std::vector<TypePtr>& ref) {
    UnionTypePtr union_type(new UnionType(ref));
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

    for (const auto& t: union_type->GetContainedTypes()) {
        f(t);
    }

    bool numbertype_found = int_found && float_found && complex_found;
    if (nonetype_found) {
        if (union_type->GetContainedTypeSize() == 4 && numbertype_found) {
            return OptionalType::create(NumberType::Global());
        }

        if (union_type->GetContainedTypeSize() == 2) {
            auto not_none = union_type->GetContainedTypes()[0] != NoneType::Global()
                                    ? union_type->GetContainedTypes()[0]
                                    : union_type->GetContainedTypes()[1];
            return OptionalType::create(not_none);
        }
    }

    return union_type;
}

std::optional<TypePtr> UnionType::to_optional() const {
    if (!canHoldType(*NoneType::Global())) {
        return std::nullopt;
    }

    std::vector<TypePtr> copied_types = this->GetContainedTypes().vec();
    auto maybe_opt = create(copied_types);
    if (maybe_opt->kind() == Kind) {
        return std::nullopt;
    }
    return maybe_opt;
}


bool UnionType::Equals(const Type& rhs) const {
    if (auto union_rhs = rhs.CastTo<UnionType>()) {
        if (this->GetContainedTypeSize() != rhs.GetContainedTypeSize()) {
            return false;
        }
        // Check that all the types in `this->types_` are also in
        // `union_rhs->types_`
        return std::all_of(this->GetContainedTypes().begin(), this->GetContainedTypes().end(),
                           [&](TypePtr lhs_type) {
                               return std::any_of(union_rhs->GetContainedTypes().begin(),
                                                  union_rhs->GetContainedTypes().end(),
                                                  [&](const TypePtr& rhs_type) {
                                                      return *lhs_type == *rhs_type;
                                                  });
                           });
    }

    if (auto optional_rhs = rhs.CastTo<OptionalType>()) {
        if (optional_rhs->get_element_type() == NumberType::Global()) {
            return this->GetContainedTypeSize() == 4 && this->can_hold_none_ && this->canHoldType(*NumberType::Global());
        }
        auto optional_lhs = this->to_optional();
        return optional_lhs && *optional_rhs == *optional_lhs.value()->Expect<OptionalType>();
    }

    if (rhs.kind() == NumberType::Kind) {
        return this->GetContainedTypeSize() == 3 && canHoldType(*NumberType::Global());
    }

    return false;
}

String UnionType::union_str(const TypePrinter& printer, bool is_annotation_str) const {
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
                ss << this->GetContainedTypes()[i]->Annotation(printer);
            } else {
                ss << this->GetContainedTypes()[i]->str();
            }
        }
    }

    if (can_hold_numbertype) {
        if (printed) {
            ss << ", ";
        }
        if (is_annotation_str) {
            ss << NumberType::Global()->Annotation(printer);
        } else {
            ss << NumberType::Global()->str();
        }
    }

    ss << close_delimeter;
    return ss.str();
}

OptionalType::OptionalType(const TypePtr& contained)
    : UnionType({contained, NoneType::Global()}, TypeKind::OptionalType) {
    bool is_numbertype = false;
    if (auto as_union = contained->CastTo<UnionType>()) {
        is_numbertype = as_union->GetContainedTypeSize() == 3 && as_union->canHoldType(*NumberType::Global());
    }

    if (UnionType::GetContainedTypeSize() == 2) {
        contained_type_ = UnionType::GetContainedTypes()[0]->kind() != NoneType::Kind
                                 ? UnionType::GetContainedTypes()[0]
                                 : UnionType::GetContainedTypes()[1];
    } else if (contained == NumberType::Global() || is_numbertype) {
        contained_type_ = NumberType::Global();
        types_.clear();
        types_.emplace_back(NumberType::Global());
        types_.emplace_back(NoneType::Global());
    } else {
        std::vector<TypePtr> to_subtract{NoneType::Global()};
        auto without_none = subtractTypeSetFrom(to_subtract, types_);
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        contained_type_ = UnionType::create({std::move(without_none.value())});
    }
    has_free_variables_ = contained_type_->HasFreeVars();
}

bool OptionalType::Equals(const Type& rhs) const {
    if (auto union_rhs = rhs.CastTo<UnionType>()) {
        auto optional_rhs = union_rhs->to_optional();
        return optional_rhs && *this == **optional_rhs;
    }

    if (auto optional_rhs = rhs.CastTo<OptionalType>()) {
        return *this->get_element_type() == *optional_rhs->get_element_type();
    }
    return false;
}

bool OptionalType::IsSubtypeOfExt(const Type& other, std::ostream* why_not) const {
    if (auto opt_other = other.CastTo<OptionalType>()) {
        return this->get_element_type()->IsSubtypeOfExt(*opt_other->get_element_type(), why_not);
    }

    if (auto union_other = other.CastTo<UnionType>()) {
        if (!union_other->canHoldType(*NoneType::Global())) {
            if (why_not) {
                *why_not << other.ReprStr() << " cannot hole none";
            }
            return false;
        }

        if (!union_other->canHoldType(*this->get_element_type())) {
            if (why_not) {
                *why_not << other.ReprStr() << " cannot hold " << this->get_element_type();
            }
            return false;
        }
        return true;
    }

    return UnionType::IsSubtypeOfExt(other, why_not);
}


OptionalTypePtr OptionalType::create(const TypePtr& contained) {
    return OptionalTypePtr(new OptionalType(contained));
}


}// namespace aethermind