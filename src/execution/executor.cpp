#include "aethermind/execution/executor.h"

#include "aethermind/execution/layer_runner.h"

namespace aethermind {

Status Executor::Execute(const ExecutionPlan& plan) noexcept {
    return LayerRunner::Run(plan);
}

}// namespace aethermind
