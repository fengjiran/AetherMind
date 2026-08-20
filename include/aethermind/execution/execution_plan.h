#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_H

/// @file execution_plan.h
/// @brief Validated execution steps and the plan that owns them.

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"
#include "aethermind/execution/state_alias_plan.h"
#include "aethermind/runtime/workspace.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <cstddef>
#include <vector>

namespace aethermind {

/// @brief One resolved kernel invocation within an ExecutionPlan.
///
/// `packed_weights` is a borrowed pointer into a PackedWeightStore's storage;
/// the store must outlive this plan. Specs are the compact contributing-only
/// view consumed by runtime shape checks.
struct ExecutionStep {
    KernelSelector selector{};
    ResolvedKernel kernel{};
    /// Borrowed pointer; the owning PackedWeightStore must outlive this plan.
    const void* packed_weights = nullptr;
    WorkspaceRequirement workspace_requirement{};
    std::vector<TensorSpec> input_specs{};
    std::vector<TensorSpec> output_specs{};
    std::vector<ShapeConstraint> runtime_checks{};
};

/// @brief Immutable, validated sequence of execution steps.
///
/// Built via Create(); every step is checked for kernel op type, kernel
/// function, params-builder consistency, and workspace alignment.
class ExecutionPlan {
public:
    /// @brief Creates a validated ExecutionPlan from step metadata.
    ///
    /// @param steps Step descriptors; the plan takes ownership of the kernels
    ///              and of the borrowed packed-weight pointers' contract.
    /// @param state_alias_plan Runtime state aliases for the steps.
    /// @return The validated plan, or an error if any step fails validation.
    AM_NODISCARD static StatusOr<ExecutionPlan> Create(
            std::vector<ExecutionStep> steps,
            StateAliasPlan state_alias_plan = {});

    /// @brief Returns the ordered list of execution steps.
    AM_NODISCARD const std::vector<ExecutionStep>& steps() const noexcept;
    /// @brief Returns the number of steps in the plan.
    AM_NODISCARD size_t size() const noexcept;

    /// @brief Returns the state alias plan used to validate state updates.
    AM_NODISCARD const StateAliasPlan& state_alias_plan() const noexcept;

private:
    /// @brief Validates and appends one step.
    ///
    /// @param step Step descriptor to validate and store.
    /// @return Status::Ok() on success.
    Status AddStep(ExecutionStep step);

    std::vector<ExecutionStep> steps_{};
    StateAliasPlan state_alias_plan_{};
};

}// namespace aethermind

#endif
