#ifndef AETHERMIND_EXECUTION_LAYER_RUNNER_H
#define AETHERMIND_EXECUTION_LAYER_RUNNER_H

/// @file layer_runner.h
/// @brief Sequential per-step execution of an ExecutionPlan (execution-private).
///
/// Execution-private implementation detail: the public execution entry point
/// is Executor::Execute; LayerRunner only backs it (and focused tests).

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/state_alias_plan.h"

namespace aethermind {

/// @brief Runs each step of an ExecutionPlan in order.
///
/// Steps are executed synchronously and single-threaded. State-alias steps
/// are validated against the plan's StateAliasPlan before kernel invocation.
class LayerRunner {
public:
    /// @brief Executes every step of the plan.
    ///
    /// @param plan Plan to execute.
    /// @param context Prepared tensor bindings and runtime resources.
    /// @return Status::Ok() on success, or the first failing step's error.
    static Status Run(const ExecutionPlan& plan,
                      const ExecutionContext& context) noexcept;

private:
    /// @brief Executes a single step after alias and binding validation.
    ///
    /// @param step_index Index of the step in the plan.
    /// @param step Step descriptor.
    /// @param context Runtime resources for the step.
    /// @param alias_plan State aliases for the whole plan.
    /// @return Status::Ok() on success.
    static Status RunStep(size_t step_index,
                          const ExecutionStep& step,
                          const ExecutionContext& context,
                          const PreparedExecutionBindings& prepared_bindings,
                          const StateAliasPlan& alias_plan,
                          const std::vector<ExecutionValueDesc>& values) noexcept;

    /// @brief Verifies that a step's state aliases have runtime backing.
    ///
    /// @param step_index Index of the step in the plan.
    /// @param step Step descriptor.
    /// @param alias_plan State aliases for the whole plan.
    /// @param context Runtime resources for the step.
    /// @param values Global value table owning the specs for this step.
    /// @return Status::Ok() on success.
    static Status ValidateStateAliasesForStep(
            size_t step_index,
            const ExecutionStep& step,
            const StateAliasPlan& alias_plan,
            const ExecutionContext& context,
            const std::vector<ExecutionValueDesc>& values) noexcept;
};

} // namespace aethermind

#endif // AETHERMIND_EXECUTION_LAYER_RUNNER_H
