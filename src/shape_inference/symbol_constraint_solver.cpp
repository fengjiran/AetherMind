#include "aethermind/shape_inference/symbol_constraint_solver.h"

#include <string>

namespace aethermind {
namespace {

AM_NODISCARD Status StaticConflict(int64_t lhs, int64_t rhs) {
    return Status::InvalidArgument("Symbol constraint static dimension conflict: " +
                                   std::to_string(lhs) + " vs " + std::to_string(rhs));
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

    size_t lhs_idx = GetIndex(lhs_root);
    size_t rhs_idx = GetIndex(rhs_root);
    const std::optional<int64_t> lhs_static = static_bindings_[lhs_idx];
    const std::optional<int64_t> rhs_static = static_bindings_[rhs_idx];
    if (lhs_static.has_value() && rhs_static.has_value() && *lhs_static != *rhs_static) {
        return StaticConflict(*lhs_static, *rhs_static);
    }

    // Union by rank: attach the shallower tree under the deeper root.
    if (ranks_[lhs_idx] < ranks_[rhs_idx]) {
        std::swap(lhs_root, rhs_root);
        std::swap(lhs_idx, rhs_idx);
    }

    parents_[rhs_idx] = lhs_root;
    if (ranks_[lhs_idx] == ranks_[rhs_idx]) {
        ++ranks_[lhs_idx];
    }

    // Propagate static binding to the new root. If both classes already
    // have the same static binding, lhs already holds it — no move needed.
    if (static_bindings_[rhs_idx].has_value() && !static_bindings_[lhs_idx].has_value()) {
        static_bindings_[lhs_idx] = static_bindings_[rhs_idx];
        static_bindings_[rhs_idx] = std::nullopt;
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
    if (const size_t idx = GetIndex(symbol_value); idx >= parents_.size()) {
        const size_t old_size = parents_.size();
        parents_.resize(idx + 1);
        ranks_.resize(idx + 1, 0);
        static_bindings_.resize(idx + 1, std::nullopt);

        // Fill the gap with self-roots. Each new entry at index i
        // corresponds to symbol value -(i+2), so its self-root is
        // exactly that value.
        for (size_t i = old_size; i <= idx; ++i) {
            parents_[i] = -static_cast<int64_t>(i) - 2;
        }
    }
}

int64_t SymbolConstraintSolver::FindRoot(int64_t symbol_value) {
    const size_t idx = GetIndex(symbol_value);
    // Path compression: flatten the chain so subsequent lookups are O(1).
    if (symbol_value != parents_[idx]) {
        parents_[idx] = FindRoot(parents_[idx]);
    }
    return parents_[idx];
}

std::optional<int64_t> SymbolConstraintSolver::FindRootIfPresent(int64_t symbol_value) const {
    const size_t idx = GetIndex(symbol_value);
    if (idx >= parents_.size()) {
        return std::nullopt;
    }

    int64_t current = symbol_value;
    int64_t parent = parents_[idx];
    while (parent != current) {
        current = parent;
        parent = parents_[GetIndex(current)];
    }
    return current;
}

std::optional<int64_t> SymbolConstraintSolver::StaticBindingForRoot(int64_t root) const {
    const size_t idx = GetIndex(root);
    if (idx >= static_bindings_.size()) {
        return std::nullopt;
    }
    return static_bindings_[idx];
}

Status SymbolConstraintSolver::BindRoot(int64_t root, int64_t static_value) {
    const size_t idx = GetIndex(root);
    if (const std::optional<int64_t> existing = static_bindings_[idx];
        existing.has_value() && *existing != static_value) {
        return StaticConflict(*existing, static_value);
    }

    static_bindings_[idx] = static_value;
    return Status::Ok();
}

}// namespace aethermind
