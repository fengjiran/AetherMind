#ifndef AETHERMIND_EXECUTION_EXECUTOR_H
#define AETHERMIND_EXECUTION_EXECUTOR_H

/// @file executor.h
/// @brief Synchronous entry point for running an ExecutionPlan.

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"

namespace aethermind {

/// @brief Runs an ExecutionPlan against an ExecutionContext.
///
/// Execution is synchronous and single-threaded; the executor delegates to
/// LayerRunner.
class Executor {
public:
    /// @brief Executes every step of the plan in order.
    ///
    /// @param plan Plan to execute.
    /// @param context Prepared tensor bindings and borrowed execution
    ///                resources that must remain valid for the call.
    /// @return Status::Ok() on success, or the first failing step's error.
    static Status Execute(const ExecutionPlan& plan,
                          ExecutionContext& context) noexcept;
};

} // namespace aethermind

#endif
