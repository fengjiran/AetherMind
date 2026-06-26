#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_DUMP_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_DUMP_H

#include "aethermind/model/graph/model_graph.h"

#include <iosfwd>

namespace aethermind {

/// Returns a stable string label for a WeightRole enumerator.
/// If the enumerator is not handled, returns "UnknownWeightRole".
AM_NODISCARD const char* ToString(WeightRole role) noexcept;

/// Returns a stable string label for a KVCacheSlot enumerator.
/// If the enumerator is not handled, returns "UnknownKVCacheSlot".
AM_NODISCARD const char* ToString(KVCacheSlot slot) noexcept;

/// Returns a lowercase string label identifying the active alternative
/// of a GraphValuePayload variant (e.g. "activation", "weight", "state").
AM_NODISCARD const char* GraphValuePayloadKindName(const GraphValuePayload& payload) noexcept;

/// Writes a human-readable representation of an OpParams variant to `os`.
/// The format is `ParamsName{field=value, ...}`.
void DumpOpParams(const OpParams& params, std::ostream& os);

/// Writes a human-readable, line-oriented dump of `graph` to `os`.
///
/// Output sections (in order):
///   1. "ModelGraph" header
///   2. "inputs:"  — one line per graph input: `v<id> name=<name>`
///   3. "outputs:" — one line per graph output: `v<id> name=<name>`
///   4. "values:"  — one line per value: kind, spec, payload, producer
///   5. "nodes:"   — one line per node: op, layer, inputs, outputs, params
void DumpGraph(const ModelGraph& graph, std::ostream& os);

}// namespace aethermind

#endif
