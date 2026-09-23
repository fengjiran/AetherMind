#ifndef AETHERMIND_MODEL_WEIGHT_BINDING_RESOLVER_H
#define AETHERMIND_MODEL_WEIGHT_BINDING_RESOLVER_H

/// @file weight_binding_resolver.h
/// @brief Maps structured weight bindings to their resolved raw weight storage.

#include "aethermind/graph/graph_types.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

/// @brief Resolves a direct weight binding to the raw weight it references.
///
/// Single authority for the Transformer role → ResolvedModelWeights mapping,
/// including the tied lm-head fallback: a checkpoint without an independent
/// lm_head reuses embed_tokens, so no caller special-cases tying. Resolution is
/// purely structural — it never consults debug names or graph topology.
///
/// @param binding Logical weight binding to resolve. Composite bindings
///        (QkvWeightBinding / GateUpWeightBinding) address several weights and
///        have no single resolution; resolve each component role separately.
/// @param resolved Resolved weights backing the model.
/// @return Pointer borrowed from `resolved`, valid while `resolved` lives, or
///         nullptr when the binding has no single raw weight: a composite or
///         role-less binding, kMoERouter (absent from dense checkpoints), or a
///         layer-scoped role whose decoder_layer_index is missing or out of
///         range. Callers must treat nullptr as a hard error — never as "skip
///         this weight" — because a silently unbound weight only surfaces as
///         wrong numerics at execution time.
AM_NODISCARD const RawWeightView* ResolveWeightBinding(
        const WeightBinding& binding,
        const ResolvedModelWeights& resolved) noexcept;

} // namespace aethermind

#endif
