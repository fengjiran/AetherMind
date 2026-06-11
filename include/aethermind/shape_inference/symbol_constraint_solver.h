#ifndef AETHERMIND_SHAPE_INFERENCE_SYMBOL_CONSTRAINT_SOLVER_H
#define AETHERMIND_SHAPE_INFERENCE_SYMBOL_CONSTRAINT_SOLVER_H

#include "aethermind/base/status.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "macros.h"

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace aethermind {

class SymbolConstraintSolver {
public:
    AM_NODISCARD Status AddEqual(ShapeSymbol lhs, ShapeSymbol rhs);
    AM_NODISCARD Status AddEqual(ShapeSymbol symbol, int64_t static_value);

    AM_NODISCARD ShapeConstraintEvaluationResult EvaluateEqual(ShapeSymbol lhs,
                                                               ShapeSymbol rhs) const;
    AM_NODISCARD bool AreEqual(ShapeSymbol lhs, ShapeSymbol rhs) const;
    AM_NODISCARD std::optional<int64_t> GetStaticBinding(ShapeSymbol symbol) const;

private:
    void EnsureSymbol(int64_t symbol_value);
    int64_t FindRoot(int64_t symbol_value);
    AM_NODISCARD std::optional<int64_t> FindRootIfPresent(int64_t symbol_value) const;
    AM_NODISCARD std::optional<int64_t> StaticBindingForRoot(int64_t root) const;
    AM_NODISCARD Status BindRoot(int64_t root, int64_t static_value);

    std::unordered_map<int64_t, int64_t> parents_;
    std::unordered_map<int64_t, size_t> ranks_;
    std::unordered_map<int64_t, int64_t> static_bindings_;
};

}// namespace aethermind

#endif// AETHERMIND_SHAPE_INFERENCE_SYMBOL_CONSTRAINT_SOLVER_H
