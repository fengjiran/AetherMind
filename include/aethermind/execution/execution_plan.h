#ifndef AETHERMIND_EXECUTION_EXECUTION_PLAN_H
#define AETHERMIND_EXECUTION_EXECUTION_PLAN_H

/// @file execution_plan.h
/// @brief Validated execution steps and the plan that owns them.

#include "aethermind/backend/packed_weights.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/execution/state_alias_plan.h"
#include "aethermind/runtime/workspace.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aethermind {

/// @brief Execution-owned identifier for one logical plan value.
///
/// This is deliberately distinct from GraphValueId so public execution APIs
/// do not depend on graph/compiler headers.
struct ExecutionValueId {
    uint32_t index = 0;

    AM_NODISCARD friend constexpr bool operator==(
            ExecutionValueId lhs, ExecutionValueId rhs) noexcept = default;
};

/// @brief Stable identity shared by an ExecutionPlan and its PreparedExecutionBindings.
struct ExecutionPlanBindingKey {
    uint64_t value = 0;

    AM_NODISCARD friend constexpr bool operator==(
            ExecutionPlanBindingKey lhs, ExecutionPlanBindingKey rhs) noexcept = default;
};

/// @brief Storage/lifetime role of a logical execution value.
enum class ExecutionValueKind : uint8_t {
    kModelInput,
    kActivation,
    kWeight,
    kConstant,
    kState,
};

/// @brief Immutable logical metadata for one ExecutionPlan value.
struct ExecutionValueDesc {
    TensorSpec spec{};
    ExecutionValueKind kind = ExecutionValueKind::kActivation;
    std::string name{};
};

/// @brief One resolved kernel invocation within an ExecutionPlan.
///
/// `packed_weights` is a borrowed pointer into a PackedWeightStore's storage;
/// the store must outlive this plan. Each step retains semantic port operands
/// and the compact kernel-facing port projection; tensor specs are owned once
/// in ExecutionPlan::values and derived on demand via inputs/outputs.
struct ExecutionStep {
    KernelSelector selector{};
    ResolvedKernel kernel{};
    /// Shared reference into PackedWeightStore artifacts. The plan holds its
    /// own reference, so it stays executable after the store is destroyed.
    std::shared_ptr<const PackedWeights> packed_weights{};
    WorkspaceRequirement workspace_requirement{};
    /// Complete semantic schema-port-ordered input operands. State ports are
    /// retained here even though they do not become kernel TensorViews.
    std::vector<ExecutionValueId> inputs{};
    /// Complete semantic schema-port-ordered output operands. State ports are
    /// retained here even though they do not become kernel TensorViews.
    std::vector<ExecutionValueId> outputs{};
    /// Maps compact kernel-input indices to semantic input-port indices.
    std::vector<uint32_t> kernel_input_ports{};
    /// Maps compact kernel-output indices to semantic output-port indices.
    std::vector<uint32_t> kernel_output_ports{};
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
    /// @param values Logical value descriptors indexed by ExecutionValueId;
    ///               the plan takes ownership of them.
    /// @param model_inputs Logical value IDs that callers must bind as model
    ///                     inputs; the plan takes ownership of the list.
    /// @param model_outputs Logical value IDs exposed as model outputs; the
    ///                      plan takes ownership of the list.
    /// @param steps Step descriptors passed by value; the plan takes ownership
    ///              of the steps and their embedded kernels. The
    ///              `packed_weights` pointers inside each step remain borrowed
    ///              from a PackedWeightStore, which must outlive the plan.
    /// @param state_alias_plan Runtime state aliases for the steps.
    /// @param workspace_layout Total workspace size and base alignment needed
    ///                         to allocate a WorkspaceArena for this plan.
    /// @return The validated plan, or an error if any step fails validation.
    static StatusOr<ExecutionPlan> Create(
            std::vector<ExecutionValueDesc> values,
            std::vector<ExecutionValueId> model_inputs,
            std::vector<ExecutionValueId> model_outputs,
            std::vector<ExecutionStep> steps,
            StateAliasPlan state_alias_plan = {},
            WorkspacePlanLayout workspace_layout = {});

    /// @brief Returns execution-owned logical values indexed by ExecutionValueId.
    AM_NODISCARD const std::vector<ExecutionValueDesc>& values() const noexcept {
        return values_;
    }

    /// @brief Returns logical values that callers must bind as model inputs.
    AM_NODISCARD const std::vector<ExecutionValueId>& model_inputs() const noexcept {
        return model_inputs_;
    }

    /// @brief Returns logical values exposed as model outputs.
    AM_NODISCARD const std::vector<ExecutionValueId>& model_outputs() const noexcept {
        return model_outputs_;
    }

    /// @brief Returns the immutable identity used to validate prepared bindings.
    AM_NODISCARD ExecutionPlanBindingKey binding_key() const noexcept {
        return binding_key_;
    }

    /// @brief Returns the ordered list of execution steps.
    AM_NODISCARD const std::vector<ExecutionStep>& steps() const noexcept {
        return steps_;
    }

    /// @brief Returns the number of steps in the plan.
    AM_NODISCARD size_t size() const noexcept {
        return steps_.size();
    }

    /// @brief Returns the state alias plan used to validate state updates.
    AM_NODISCARD const StateAliasPlan& state_alias_plan() const noexcept {
        return state_alias_plan_;
    }

    /// @brief Returns the total workspace bytes required across all steps.
    AM_NODISCARD size_t total_workspace_bytes() const noexcept {
        return workspace_layout_.total_bytes;
    }

    /// @brief Returns the maximum base alignment required by the workspace.
    AM_NODISCARD size_t workspace_alignment() const noexcept {
        return workspace_layout_.required_alignment;
    }

private:
    // Validates and appends one step. Returns Status::Ok() on success or the
    // first validation failure.
    Status AddStep(ExecutionStep step);

    std::vector<ExecutionStep> steps_{};
    std::vector<ExecutionValueDesc> values_{};
    std::vector<ExecutionValueId> model_inputs_{};
    std::vector<ExecutionValueId> model_outputs_{};
    ExecutionPlanBindingKey binding_key_{};
    StateAliasPlan state_alias_plan_{};
    WorkspacePlanLayout workspace_layout_{};
};

} // namespace aethermind

#endif
