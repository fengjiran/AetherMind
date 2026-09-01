#ifndef AETHERMIND_GRAPH_OPTIMIZATION_OP_REMOVABILITY_INTERNAL_H
#define AETHERMIND_GRAPH_OPTIMIZATION_OP_REMOVABILITY_INTERNAL_H

/// @file op_removability_internal.h
/// @brief Internal removability policy shared by optimization passes.
///
/// Composes operators' public facts (OperatorTraits::has_side_effects,
/// HasStatefulOutput) into the removability policy that DCE and Commit-level
/// pruning use, without widening the operators public API. Internal header,
/// not installed to include/.

#include "aethermind/base/macros.h"
#include "aethermind/operators/op_type.h"

namespace aethermind::detail {

/// @brief Whether a node of `op_type` may be dropped when its outputs are
/// dead.
///
/// Matches DCE semantics: the op must be side-effect-free with no stateful
/// output. Unlike IsPureOperator, there is no determinism requirement: a
/// deterministic=false but side-effect-free op is still removable.
///
/// @param op_type Op type to classify. Ops without a registered schema are
///                conservatively treated as non-removable.
/// @return True when nodes of `op_type` may be removed.
AM_NODISCARD bool IsDceRemovableOp(OpType op_type) noexcept;

} // namespace aethermind::detail

#endif // AETHERMIND_GRAPH_OPTIMIZATION_OP_REMOVABILITY_INTERNAL_H