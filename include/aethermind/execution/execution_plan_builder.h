#ifndef AETHERMIND_EXECUTION_EXECUTION_PLAN_BUILDER_H
#define AETHERMIND_EXECUTION_EXECUTION_PLAN_BUILDER_H

/// @file execution_plan_builder.h
/// @brief Factory that turns node specs or compiler artifacts into ExecutionPlans.

#include "aethermind/execution/execution_node_spec.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/runtime/runtime_context.h"

#include <vector>

namespace aethermind {

class LoweredGraph;
class PackedWeightStore;

/// @brief Builds ExecutionPlan instances.
///
/// Untrusted ExecutionPlanNodeSpec metadata is validated against
/// InferOperator; compiler-produced LoweredGraph artifacts are re-checked at
/// the execution trust boundary. Kernels are resolved through the
/// RuntimeContext's registered backends.
class ExecutionPlanBuilder {
public:
    /// @brief Resolves and prepares the kernel for a single node.
    ///
    /// @param backend Backend that prepares the kernel.
    /// @param node Untrusted node metadata.
    /// @return The prepared kernel, or an error if the node is invalid.
    static StatusOr<ResolvedKernel> PrepareKernelForNode(
            const Backend& backend,
            const ExecutionPlanNodeSpec& node);

    /// @brief Builds an ExecutionPlan from independent untrusted node specs.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param nodes Untrusted per-node metadata. This API has no edge
    ///        representation: every node receives distinct external operands
    ///        and produces distinct externally-visible results. Use
    ///        Build(LoweredGraph) for graph dataflow.
    /// @return The built plan, or an error on validation or kernel failure.
    static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// @brief Builds an ExecutionPlan from untrusted node specs with packed
    ///        weights.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param packed_weight_store Storage for packed-format weights.
    /// @param nodes Untrusted per-node metadata.
    /// @return The built plan, or an error on validation or kernel failure.
    static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const PackedWeightStore& packed_weight_store,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// @brief Builds an ExecutionPlan from a finalized compiler artifact.
    ///
    /// Execution resolves its semantic state aliases into the runtime
    /// StateAliasPlan.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param lowered_graph Finalized compiler artifact.
    /// @return The built plan, or an error if the artifact is invalid.
    static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const LoweredGraph& lowered_graph);

    /// @brief Builds an ExecutionPlan from a compiler artifact with packed
    ///        weights.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param packed_weight_store Storage for packed-format weights.
    /// @param lowered Finalized compiler artifact.
    /// @return The built plan, or an error if the artifact is invalid.
    static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const PackedWeightStore& packed_weight_store,
            const LoweredGraph& lowered);
};

/// @brief Resolves compiler semantic-port state aliases into the runtime-only
///        StateAliasPlan.
///
/// Revalidates the finalized artifact and deterministically orders the aliases
/// so ExecutionStep consumers can binary-search per step. Execution-private:
/// the compiler never includes this type.
StatusOr<StateAliasPlan> ResolveStateAliasesForExecution(
        const LoweredGraph& lowered);

} // namespace aethermind

#endif
