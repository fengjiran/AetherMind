#include "aethermind/execution/layer_runner.h"

namespace aethermind {

Status LayerRunner::Run(const ExecutionPlan& plan) noexcept {
    for (const ExecutionStep& step: plan.steps()) {
        if (const Status status = RunStep(step); !status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

Status LayerRunner::RunStep(const ExecutionStep& step) noexcept {
    if (step.kernel.fn == nullptr) {
        return Status::InvalidArgument("Execution step kernel function cannot be null");
    }
    return step.kernel.fn();
}

}// namespace aethermind
