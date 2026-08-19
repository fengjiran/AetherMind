#ifndef AETHERMIND_EXECUTION_LOWERED_GRAPH_ADAPTER_H
#define AETHERMIND_EXECUTION_LOWERED_GRAPH_ADAPTER_H

#include "aethermind/base/status.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/execution/state_alias_plan.h"

namespace aethermind {

/// Execution-private conversion of compiler semantic-port aliases into the
/// runtime-only StateAliasPlan. Compiler never includes this execution type.
AM_NODISCARD StatusOr<StateAliasPlan> ResolveStateAliasesForExecution(
        const LoweredGraph& lowered);

}// namespace aethermind

#endif
