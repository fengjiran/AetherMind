#ifndef AETHERMIND_EXECUTION_LAYER_RUNNER_H
#define AETHERMIND_EXECUTION_LAYER_RUNNER_H

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {

class LayerRunner {
public:
    AM_NODISCARD static Status Run(const ExecutionPlan& plan,
                                   RuntimeBindingContext& bindings) noexcept;

private:
    AM_NODISCARD static Status RunStep(const ExecutionStep& step,
                                       RuntimeBindingContext& bindings) noexcept;
};

}// namespace aethermind

#endif
