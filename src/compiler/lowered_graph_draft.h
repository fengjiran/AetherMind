#ifndef AETHERMIND_COMPILER_LOWERED_GRAPH_DRAFT_H
#define AETHERMIND_COMPILER_LOWERED_GRAPH_DRAFT_H

// Compiler-internal construction state. This header is intentionally not
// installed under include/: a draft is never a trusted execution artifact.

#include "aethermind/compiler/lowered_graph.h"

namespace aethermind::compiler_internal {

/// Mutable compiler construction state. Finalize validates its complete
/// structure before producing the immutable LoweredGraph consumed by
/// execution. It is also the focused test seam for malformed artifact input.
struct LoweredGraphDraft {
    std::vector<LoweredStep> steps{};
    std::vector<LoweredValueDesc> values{};
    std::vector<GraphValueId> model_inputs{};
    std::vector<GraphValueId> model_outputs{};
    std::vector<LoweredStateAlias> state_aliases{};

    AM_NODISCARD Status Validate() const;
    AM_NODISCARD StatusOr<LoweredGraph> Finalize() &&;
};

}// namespace aethermind::compiler_internal

#endif
