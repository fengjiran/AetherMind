#ifndef AETHERMIND_SHAPE_INFERENCE_SYMBOL_CONSTRAINT_SOLVER_H
#define AETHERMIND_SHAPE_INFERENCE_SYMBOL_CONSTRAINT_SOLVER_H

#include "aethermind/base/status.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/base/macros.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace aethermind {

/// @brief Plan-time symbolic equality solver using union-find.
///
/// Maintains equality classes among ShapeSymbol instances and propagates
/// static dimension bindings through those classes. Unknown(-1) symbols
/// are silently ignored — they do not participate in any equality class.
///
/// Designed for the graph compilation phase (plan time), not for runtime.
/// Proves equality facts that allow the builder to eliminate redundant
/// runtime shape checks.
/// @note Not thread-safe. All methods must be called from a single thread.
class SymbolConstraintSolver {
public:
    /// @brief Records that two symbols must be equal.
    ///
    /// Static-static mismatches return InvalidArgument. Unknown symbols
    /// are silently accepted but do not create any binding. Symbolic
    /// symbols are merged via union-find with rank-based union.
    /// @param lhs First symbol.
    /// @param rhs Second symbol.
    /// @return OkStatus on success; kInvalidArgument on static conflict.
    AM_NODISCARD Status AddEqual(ShapeSymbol lhs, ShapeSymbol rhs);

    /// @brief Binds a symbolic dimension to a known static value.
    ///
    /// If the symbol is already bound to a different static value,
    /// returns InvalidArgument. Unknown symbols are silently accepted.
    /// @param symbol       The symbolic dimension to bind.
    /// @param static_value The non-negative static value to bind.
    /// @return OkStatus on success; kInvalidArgument on conflict.
    AM_NODISCARD Status AddEqual(ShapeSymbol symbol, int64_t static_value);

    /// @brief Evaluates whether two symbols are provably equal.
    ///
    /// @param lhs First symbol.
    /// @param rhs Second symbol.
    /// @return kSatisfied when equality is proven, kViolated on static
    ///         conflict, kDeferred when facts are insufficient to decide.
    AM_NODISCARD ShapeConstraintEvaluationResult EvaluateEqual(ShapeSymbol lhs,
                                                               ShapeSymbol rhs) const;

    /// @brief Convenience check: true iff EvaluateEqual returns kSatisfied.
    /// @param lhs First symbol.
    /// @param rhs Second symbol.
    /// @return True if equality is proven.
    AM_NODISCARD bool AreEqual(ShapeSymbol lhs, ShapeSymbol rhs) const;

    /// @brief Returns the static dimension value bound to a symbol, if any.
    ///
    /// Follows union-find roots to find propagated bindings.
    /// @param symbol The symbol to query.
    /// @return The static binding, or nullopt for unknown/unbound symbols.
    AM_NODISCARD std::optional<int64_t> GetStaticBinding(ShapeSymbol symbol) const;

private:
    // Ensures dense vector storage up to the given symbol value.
    // Newly created entries are self-roots with no static binding.
    void EnsureSymbol(int64_t symbol_value);

    // Finds the root of the equality class with path compression.
    // Mutates parents_ in place.
    int64_t FindRoot(int64_t symbol_value);

    // Const variant: finds the root without path compression.
    // Returns nullopt if the symbol was never registered.
    AM_NODISCARD std::optional<int64_t> FindRootIfPresent(int64_t symbol_value) const;

    // Returns the static binding stored at a root node, if any.
    AM_NODISCARD std::optional<int64_t> StaticBindingForRoot(int64_t root) const;

    // Binds a root to a static value, rejecting conflicts.
    AM_NODISCARD Status BindRoot(int64_t root, int64_t static_value);

    // Maps symbol value to dense vector index: -2 → 0, -3 → 1, ...
    // Inverse of ShapeSymbol::Create() which generates -2, -3, ...
    AM_NODISCARD static size_t GetIndex(int64_t symbol_value) noexcept {
        AM_DCHECK(symbol_value <= -2 && "Only symbolic values can be indexed");
        return static_cast<size_t>(-symbol_value - 2);
    }

    // Union-find parent array. Each entry is either a self-root
    // (value == its own encoded symbol) or points to a parent.
    // Dense: index i corresponds to symbol value -(i+2).
    std::vector<int64_t> parents_;

    // Union-by-rank: approximate subtree depth for each root.
    // Only meaningful at root nodes; stale for non-roots.
    std::vector<uint8_t> ranks_;

    // Static dimension bindings stored at root nodes.
    // nullopt means no static binding is known for this class.
    // Only valid at root nodes; stale for non-roots.
    std::vector<std::optional<int64_t>> static_bindings_;
};

}// namespace aethermind

#endif// AETHERMIND_SHAPE_INFERENCE_SYMBOL_CONSTRAINT_SOLVER_H
