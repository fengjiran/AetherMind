#ifndef AETHERMIND_EXECUTION_STATE_ALIAS_PLAN_H
#define AETHERMIND_EXECUTION_STATE_ALIAS_PLAN_H

/// @file state_alias_plan.h
/// @brief Runtime-resolved state alias records for plan execution.

#include "aethermind/base/macros.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind {

/// @brief Runtime resolved state alias record.
///
/// Uses step/port coordinates rather than GraphValueId so the executor can
/// query aliases without depending on ModelGraph lowering artifacts.
struct ResolvedStateAlias {
    size_t step_index = 0;
    uint32_t input_port = 0;
    uint32_t output_port = 0;
};

/// @brief Runtime state alias plan carried by ExecutionPlan and queried by
///        step.
///
/// Aliases are sorted by step_index during construction so ForStep() can
/// locate the relevant range in O(log N).
struct StateAliasPlan {
    std::vector<ResolvedStateAlias> aliases{};

    /// @brief Returns whether the plan has no aliases.
    AM_NODISCARD bool empty() const noexcept;
    /// @brief Returns the number of aliases.
    AM_NODISCARD size_t size() const noexcept;

    /// @brief Returns aliases that belong to `step_index`.
    ///
    /// @param step_index Plan step index.
    /// @return An empty span if the step has no aliases.
    AM_NODISCARD std::span<const ResolvedStateAlias> ForStep(
            size_t step_index) const noexcept;
};

}// namespace aethermind

#endif
