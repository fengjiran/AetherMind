#ifndef AETHERMIND_MODEL_GRAPH_COMPILATION_STATE_ALIAS_PLAN_H
#define AETHERMIND_MODEL_GRAPH_COMPILATION_STATE_ALIAS_PLAN_H

#include "macros.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind {

/// Runtime resolved state alias record.
/// It uses step/port coordinates rather than GraphValueId, so the executor can
/// query aliases without depending on ModelGraph lowering artifacts.
struct ResolvedStateAlias {
    size_t step_index = 0;
    uint32_t input_port = 0;
    uint32_t output_port = 0;
};

/// Runtime state alias plan carried by ExecutionPlan and queried by step.
/// Aliases are sorted by step_index during construction so that ForStep() can
/// locate the relevant range in O(log N).
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
