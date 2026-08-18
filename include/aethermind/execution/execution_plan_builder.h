#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/execution/execution_node_spec.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/model/backend_sidecar.h"
#include "aethermind/runtime/runtime_context.h"

#include <vector>

namespace aethermind {

struct LoweredGraph;

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
            const BackendSidecar& sidecar,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// Builds an ExecutionPlan from a LoweredGraph by resolving lowering-time
    /// state aliases into the runtime StateAliasPlan.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const LoweredGraph& lowered);

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const BackendSidecar& sidecar,
            const LoweredGraph& lowered);
};

}// namespace aethermind

#endif
