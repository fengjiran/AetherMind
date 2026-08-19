#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/execution/execution_node_spec.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/runtime/runtime_context.h"

#include <vector>

namespace aethermind {

class LoweredGraph;
class PackedWeightStore;

class ExecutionPlanBuilder {
public:
    AM_NODISCARD static StatusOr<ResolvedKernel> PrepareKernelForNode(
            const Backend& backend,
            const ExecutionPlanNodeSpec& node);

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const PackedWeightStore& packed_weight_store,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// Builds an ExecutionPlan from a finalized compiler artifact. Execution
    /// resolves its semantic state aliases into the runtime StateAliasPlan.
    ///
    /// @note Consumption scope: this overload uses only `lowered.steps()` (their
    /// specs) and the resolved state aliases. The step bindings, value
    /// metadata, and model inputs/outputs are not yet wired into runtime
    /// tensor/state binding; an ExecutionPlan returned here must not be treated
    /// as having completed runtime binding.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const LoweredGraph& lowered);

    /// Overload of Build(LoweredGraph) that also resolves packed weights for
    /// packed-format steps; the same consumption-scope note applies.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const PackedWeightStore& packed_weight_store,
            const LoweredGraph& lowered);
};

}// namespace aethermind

#endif
