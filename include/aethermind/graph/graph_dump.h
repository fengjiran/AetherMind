#ifndef AETHERMIND_GRAPH_GRAPH_DUMP_H
#define AETHERMIND_GRAPH_GRAPH_DUMP_H

/// @file graph_dump.h
/// @brief Human-readable dump helpers for the AetherMind computation graph.
///
/// Provides stable string labels for graph enumerators and line-oriented
/// dumps of ModelGraph values and nodes. Output is intended for diagnostics
/// and debugging; it is not a stable serialization format.
#include "aethermind/graph/graph.h"
#include "aethermind/graph/graph_types.h"

#include <iosfwd>

namespace aethermind {

/// @brief Returns a stable string label for a ParameterSlot enumerator.
///
/// If the enumerator is not handled, returns "UnknownParameterSlot".
///
/// @param slot Parameter slot to stringify.
/// @return Stable string label for `slot`.
AM_NODISCARD const char* ToString(ParameterSlot slot) noexcept;

/// @brief Returns a stable string label for a TransformerWeightRole enumerator.
///
/// If the enumerator is not handled, returns "UnknownTransformerWeightRole".
///
/// @param role Transformer weight role to stringify.
/// @return Stable string label for `role`.
AM_NODISCARD const char* ToString(TransformerWeightRole role) noexcept;

/// @brief Returns a stable string label for an optional model semantic role.
///
/// @param role Optional semantic role to stringify.
/// @return Stable string label for `role`.
AM_NODISCARD const char* ToString(const ModelSemanticRole& role) noexcept;

/// @brief Returns a stable string label for a KVCacheSlot enumerator.
///
/// If the enumerator is not handled, returns "UnknownKVCacheSlot".
///
/// @param slot KV cache slot to stringify.
/// @return Stable string label for `slot`.
AM_NODISCARD const char* ToString(KVCacheSlot slot) noexcept;

/// @brief Returns a lowercase string label identifying the active alternative
///        of a GraphValuePayload variant.
///
/// Examples: "activation", "weight", "constant", "state".
///
/// @param payload Graph value payload to inspect.
/// @return Lowercase label of the active alternative.
AM_NODISCARD const char* GraphValuePayloadKindName(const GraphValuePayload& payload) noexcept;

/// @brief Writes a human-readable representation of an OpParams variant to
///        `os`.
///
/// The format is `ParamsName{field=value, ...}`.
///
/// @param params Operator parameters to dump.
/// @param os Output stream to write to.
void DumpOpParams(const OpParams& params, std::ostream& os);

/// @brief Writes a human-readable, line-oriented dump of `graph` to `os`.
///
/// Output sections (in order):
///   1. "ModelGraph" header
///   2. "inputs:"  — one line per graph input: `v<id> name=<name>`
///   3. "outputs:" — one line per graph output: `v<id> name=<name>`
///   4. "values:"  — one line per value: kind, spec, payload, producer
///   5. "nodes:"   — one line per node: op, layer, inputs, outputs, params
///
/// @param graph Graph to dump.
/// @param os Output stream to write to.
void DumpGraph(const ModelGraph& graph, std::ostream& os);

} // namespace aethermind

#endif
