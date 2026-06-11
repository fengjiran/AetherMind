#include "aethermind/base/symbol_constraint_solver.h"

#include <string>

namespace aethermind {
namespace {

AM_NODISCARD Status StaticConflict(int64_t lhs, int64_t rhs) {
    return Status::InvalidArgument(
            "Symbol constraint static dimension conflict: " + std::to_string(lhs) +
            " vs " + std::to_string(rhs));
}

AM_NODISCARD ShapeConstraintEvaluationResult CompareStaticValues(int64_t lhs, int64_t rhs) noexcept {
    return lhs == rhs ? ShapeConstraintEvaluationResult::kSatisfied
                      : ShapeConstraintEvaluationResult::kViolated;
}

}// namespace

Status SymbolConstraintSolver::AddEqual(ShapeSymbol lhs, ShapeSymbol rhs) {
    if (lhs.IsStatic() && rhs.IsStatic()) {
        if (lhs.GetStaticValue() != rhs.GetStaticValue()) {
            return StaticConflict(lhs.GetStaticValue(), rhs.GetStaticValue());
        }
        return Status::Ok();
    }

    if (lhs.IsUnknown() || rhs.IsUnknown()) {
        return Status::Ok();
    }

    if (lhs.IsStatic()) {
        return AddEqual(rhs, lhs.GetStaticValue());
    }
    if (rhs.IsStatic()) {
        return AddEqual(lhs, rhs.GetStaticValue());
    }

    EnsureSymbol(lhs.value());
    EnsureSymbol(rhs.value());

    int64_t lhs_root = FindRoot(lhs.value());
    int64_t rhs_root = FindRoot(rhs.value());
    if (lhs_root == rhs_root) {
        return Status::Ok();
    }

    const std::optional<int64_t> lhs_static = StaticBindingForRoot(lhs_root);
    const std::optional<int64_t> rhs_static = StaticBindingForRoot(rhs_root);
    if (lhs_static.has_value() && rhs_static.has_value() && *lhs_static != *rhs_static) {
        return StaticConflict(*lhs_static, *rhs_static);
    }

    if (ranks_[lhs_root] < ranks_[rhs_root]) {
        std::swap(lhs_root, rhs_root);
    }
    parents_[rhs_root] = lhs_root;
    if (ranks_[lhs_root] == ranks_[rhs_root]) {
        ++ranks_[lhs_root];
    }

    static_bindings_.erase(rhs_root);
    if (rhs_static.has_value()) {
        static_bindings_[lhs_root] = *rhs_static;
    } else if (lhs_static.has_value()) {
        static_bindings_[lhs_root] = *lhs_static;
    }

    return Status::Ok();
}

Status SymbolConstraintSolver::AddEqual(ShapeSymbol symbol, int64_t static_value) {
    if (static_value < 0) {
        return Status::InvalidArgument("Symbol constraint static value must be non-negative");
    }

    if (symbol.IsStatic()) {
        if (symbol.GetStaticValue() != static_value) {
            return StaticConflict(symbol.GetStaticValue(), static_value);
        }
        return Status::Ok();
    }

    if (symbol.IsUnknown()) {
        return Status::Ok();
    }

    EnsureSymbol(symbol.value());
    return BindRoot(FindRoot(symbol.value()), static_value);
}

ShapeConstraintEvaluationResult SymbolConstraintSolver::EvaluateEqual(ShapeSymbol lhs,
                                                                      ShapeSymbol rhs) const {
    if (lhs.IsStatic() && rhs.IsStatic()) {
        return CompareStaticValues(lhs.GetStaticValue(), rhs.GetStaticValue());
    }

    if (lhs.IsUnknown() || rhs.IsUnknown()) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }

    if (lhs.IsSymbolic() && rhs.IsSymbolic() && lhs == rhs) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    if (lhs.IsStatic()) {
        const std::optional<int64_t> rhs_static = GetStaticBinding(rhs);
        return rhs_static.has_value() ? CompareStaticValues(lhs.GetStaticValue(), *rhs_static)
                                      : ShapeConstraintEvaluationResult::kDeferred;
    }
    if (rhs.IsStatic()) {
        const std::optional<int64_t> lhs_static = GetStaticBinding(lhs);
        return lhs_static.has_value() ? CompareStaticValues(*lhs_static, rhs.GetStaticValue())
                                      : ShapeConstraintEvaluationResult::kDeferred;
    }

    const std::optional<int64_t> lhs_root = FindRootIfPresent(lhs.value());
    const std::optional<int64_t> rhs_root = FindRootIfPresent(rhs.value());
    if (lhs_root.has_value() && rhs_root.has_value() && *lhs_root == *rhs_root) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    const std::optional<int64_t> lhs_static = GetStaticBinding(lhs);
    const std::optional<int64_t> rhs_static = GetStaticBinding(rhs);
    if (lhs_static.has_value() && rhs_static.has_value()) {
        return CompareStaticValues(*lhs_static, *rhs_static);
    }

    return ShapeConstraintEvaluationResult::kDeferred;
}

bool SymbolConstraintSolver::AreEqual(ShapeSymbol lhs, ShapeSymbol rhs) const {
    return EvaluateEqual(lhs, rhs) == ShapeConstraintEvaluationResult::kSatisfied;
}

std::optional<int64_t> SymbolConstraintSolver::GetStaticBinding(ShapeSymbol symbol) const {
    if (symbol.IsStatic()) {
        return symbol.GetStaticValue();
    }
    if (!symbol.IsSymbolic()) {
        return std::nullopt;
    }

    const std::optional<int64_t> root = FindRootIfPresent(symbol.value());
    if (!root.has_value()) {
        return std::nullopt;
    }
    return StaticBindingForRoot(*root);
}

void SymbolConstraintSolver::EnsureSymbol(int64_t symbol_value) {
    if (!parents_.contains(symbol_value)) {
        parents_[symbol_value] = symbol_value;
        ranks_[symbol_value] = 0;
    }
}

int64_t SymbolConstraintSolver::FindRoot(int64_t symbol_value) {
    int64_t parent = parents_[symbol_value];
    if (parent == symbol_value) {
        return symbol_value;
    }
    parent = FindRoot(parent);
    parents_[symbol_value] = parent;
    return parent;
}

std::optional<int64_t> SymbolConstraintSolver::FindRootIfPresent(int64_t symbol_value) const {
    const auto it = parents_.find(symbol_value);
    if (it == parents_.end()) {
        return std::nullopt;
    }

    int64_t current = symbol_value;
    int64_t parent = it->second;
    while (parent != current) {
        current = parent;
        const auto parent_it = parents_.find(current);
        if (parent_it == parents_.end()) {
            return std::nullopt;
        }
        parent = parent_it->second;
    }
    return current;
}

std::optional<int64_t> SymbolConstraintSolver::StaticBindingForRoot(int64_t root) const {
    const auto it = static_bindings_.find(root);
    if (it == static_bindings_.end()) {
        return std::nullopt;
    }
    return it->second;
}

Status SymbolConstraintSolver::BindRoot(int64_t root, int64_t static_value) {
    const std::optional<int64_t> existing = StaticBindingForRoot(root);
    if (existing.has_value() && *existing != static_value) {
        return StaticConflict(*existing, static_value);
    }
    static_bindings_[root] = static_value;
    return Status::Ok();
}

}// namespace aethermind
