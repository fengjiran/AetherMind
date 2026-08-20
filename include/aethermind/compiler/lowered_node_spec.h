#ifndef AETHERMIND_COMPILER_LOWERED_NODE_SPEC_H
#define AETHERMIND_COMPILER_LOWERED_NODE_SPEC_H

/// @file lowered_node_spec.h
/// @brief Compiler-owned, verified node metadata passed to execution planning.

#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/workspace_types.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"

namespace aethermind {

/// @brief Semantic metadata emitted by graph lowering after ModelGraph
/// validation.
///
/// This is deliberately distinct from execution::ExecutionPlanNodeSpec: the
/// latter is an untrusted raw execution request and is revalidated with
/// InferOperator. LoweredNodeSpec preserves the already-validated semantic
/// result and is only reachable through a finalized LoweredGraph.
struct LoweredNodeSpec {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
    WorkspaceRequirement workspace_requirement{};
    /// Complete schema-port-ordered specs. State ports remain present even
    /// when they do not contribute to compact runtime tensor bindings.
    std::vector<TensorSpec> input_specs{};
    std::vector<TensorSpec> output_specs{};
    /// Deferred shape constraints copied from semantic inference; they cannot
    /// be proven until concrete runtime shapes are known, so execution
    /// re-validates them before running the kernel.
    std::vector<ShapeConstraint> runtime_checks{};
    OpParams op_params{};
};

}// namespace aethermind

#endif
