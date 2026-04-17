#include "aethermind/execution/executor.h"

#include "aethermind/execution/layer_runner.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {

Status Executor::Execute(const ExecutionPlan& plan,
                         RuntimeBindingContext& bindings) noexcept {
    return LayerRunner::Run(plan, bindings);
}

}// namespace aethermind
