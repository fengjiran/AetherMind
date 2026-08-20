#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H

/// @file execution_plan_builder.h
/// @brief Factory that turns node specs or compiler artifacts into ExecutionPlans.

#include "aethermind/backend/kernel_selector.h"
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
    AM_NODISCARD static StatusOr<ResolvedKernel> PrepareKernelForNode(
            const Backend& backend,
            const ExecutionPlanNodeSpec& node);

    /// @brief Builds an ExecutionPlan from untrusted node specs.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param nodes Untrusted per-node metadata.
    /// @return The built plan, or an error on validation or kernel failure.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// @brief Builds an ExecutionPlan from untrusted node specs with packed
    ///        weights.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param packed_weight_store Storage for packed-format weights.
    /// @param nodes Untrusted per-node metadata.
    /// @return The built plan, or an error on validation or kernel failure.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const PackedWeightStore& packed_weight_store,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// @brief Builds an ExecutionPlan from a finalized compiler artifact.
    ///
    /// Execution resolves its semantic state aliases into the runtime
    /// StateAliasPlan.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param lowered Finalized compiler artifact.
    /// @return The built plan, or an error if the artifact is invalid.
    /// @note Consumption scope: this overload uses only `lowered.steps()` (their
    ///       specs) and the resolved state aliases. The step bindings, value
    ///       metadata, and model inputs/outputs are not yet wired into runtime
    ///       tensor/state binding; an ExecutionPlan returned here must not be
    ///       treated as having completed runtime binding.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const LoweredGraph& lowered);

    /// @brief Builds an ExecutionPlan from a compiler artifact with packed
    ///        weights.
    ///
    /// @param runtime Runtime context providing backends.
    /// @param packed_weight_store Storage for packed-format weights.
    /// @param lowered Finalized compiler artifact.
    /// @return The built plan, or an error if the artifact is invalid.
    /// @note Same consumption-scope restriction as the Build(LoweredGraph)
    ///       overload.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const PackedWeightStore& packed_weight_store,
            const LoweredGraph& lowered);
};

}// namespace aethermind

#endif
