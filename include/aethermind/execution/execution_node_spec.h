#ifndef AETHERMIND_EXECUTION_EXECUTION_NODE_SPEC_H
#define AETHERMIND_EXECUTION_EXECUTION_NODE_SPEC_H

/// @file execution_node_spec.h
/// @brief Untrusted per-node execution request metadata for manual plan building.

#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/workspace_types.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <cstddef>
#include <vector>

namespace aethermind {

/// @brief Untrusted execution request metadata.
///
/// Callers using this type go through InferOperator verification in
/// ExecutionPlanBuilder. Compiler-produced trusted metadata uses
/// LoweredStepSpec instead.
struct ExecutionPlanNodeSpec {
    OpType op_type = OpType::kUnknown;
    /// Execution capabilities this step requires. Lowering records the
    /// configured prefix (device/ISA/weight format/phase) and fills the
    /// activation/weight dtypes from the operator's inputs and outputs.
    KernelSelector selector{};
    /// Legacy compatibility assertion for the backend-prepared workspace
    /// requirement. A default value means no assertion. When non-default, its
    /// bytes, alignment, lifetime, and reusable fields must match the concrete
    /// kernel selected by the backend; offset is ignored because planning owns
    /// it. The prepared kernel remains the sole workspace-requirement
    /// authority.
    WorkspaceRequirement workspace_requirement{};
    /// Complete schema-port-ordered input specs, including state ports that
    /// do not contribute to runtime tensor bindings. Use MakeCompactInputSpecs
    /// to derive the compact contributing-only view consumed by operator
    /// semantic inference and ExecutionStep::input_specs.
    std::vector<TensorSpec> input_specs{};
    std::vector<TensorSpec> output_specs{};
    /// Deferred runtime shape constraints derived during graph construction
    /// and carried through lowering without re-inference.
    std::vector<ShapeConstraint> runtime_checks{};
    OpParams op_params{};
};

}// namespace aethermind

#endif
