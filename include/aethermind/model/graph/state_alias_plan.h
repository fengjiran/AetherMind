#ifndef AETHERMIND_MODEL_GRAPH_STATE_ALIAS_PLAN_H
#define AETHERMIND_MODEL_GRAPH_STATE_ALIAS_PLAN_H

#include "macros.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind {

/// Runtime-ready resolved form of a state alias constraint.
///
/// Uses step indices and port indices instead of GraphValueId,
/// making it directly usable by the execution runtime.
/// For intra-step aliases (KVCacheUpdate), input and output
/// belong to the same step_index. Cross-step aliases can be
/// added by splitting step_index in a future extension.
struct ResolvedStateAlias {
    size_t step_index = 0;
    uint32_t input_port = 0;
    uint32_t output_port = 0;
};

/// Lightweight plan for state aliasing, carried alongside
/// ExecutionPlan and enforced at runtime by the executor.
///
/// Aliases are sorted by step_index during construction so
/// that ForStep() can locate the relevant range in O(log N).
struct StateAliasPlan {
    std::vector<ResolvedStateAlias> aliases{};

    AM_NODISCARD bool empty() const noexcept;
    AM_NODISCARD size_t size() const noexcept;

    /// Returns aliases that belong to `step_index`, or an empty
    /// span if the step has no aliases.
    AM_NODISCARD std::span<const ResolvedStateAlias> ForStep(
            size_t step_index) const noexcept;
};

}// namespace aethermind

#endif
