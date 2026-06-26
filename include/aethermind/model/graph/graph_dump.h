#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_DUMP_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_DUMP_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/model_graph.h"

#include <iosfwd>

namespace aethermind {

AM_NODISCARD const char* ToString(WeightRole role) noexcept;
AM_NODISCARD const char* ToString(KVCacheSlot slot) noexcept;
AM_NODISCARD const char* GraphValuePayloadKindName(const GraphValuePayload& payload) noexcept;

AM_NODISCARD Status DumpOpParams(const OpParams& params, std::ostream& os);
AM_NODISCARD Status DumpGraph(const ModelGraph& graph, std::ostream& os);

}// namespace aethermind

#endif
