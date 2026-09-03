#include "aethermind/execution/executor.h"
#include "aethermind/execution/execution_context.h"
#include "execution/layer_runner.h"

namespace aethermind {

Status Executor::Execute(const ExecutionPlan& plan,
                         ExecutionContext& context) noexcept {
    return LayerRunner::Run(plan, context);
}

} // namespace aethermind
