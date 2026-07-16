#ifndef AETHERMIND_EXECUTION_LAYER_RUNNER_H
#define AETHERMIND_EXECUTION_LAYER_RUNNER_H

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/compilation/state_alias_plan.h"

namespace aethermind {

class LayerRunner {
public:
    AM_NODISCARD static Status Run(const ExecutionPlan& plan,
                                   RuntimeBindingContext& bindings) noexcept;

private:
    AM_NODISCARD static Status RunStep(size_t step_index,
                                       const ExecutionStep& step,
                                       RuntimeBindingContext& bindings,
                                       const StateAliasPlan& alias_plan) noexcept;

    AM_NODISCARD static Status ValidateStateAliasesForStep(
            size_t step_index,
            const ExecutionStep& step,
            const StateAliasPlan& alias_plan,
            const RuntimeBindingContext& bindings) noexcept;
};

}// namespace aethermind

#endif
