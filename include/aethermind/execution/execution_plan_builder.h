#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/runtime/runtime_context.h"

#include <cstddef>
#include <vector>

namespace aethermind {

struct LoweredGraph;

struct ExecutionPlanNodeSpec {
    OpType op_type = OpType::kUnknown;
    DeviceType device_type = DeviceType::kCPU;
    DataType act_dtype{};
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
    WorkspaceRequirement workspace_requirement{};
    std::vector<TensorSpec> input_specs{};
    std::vector<std::byte> attrs{};
    OpParams op_params{};
};

class ExecutionPlanBuilder {
public:
    AM_NODISCARD static StatusOr<ResolvedKernel> ResolveKernelForNode(
            const Backend& backend,
            const ExecutionPlanNodeSpec& node) noexcept;

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const ModelInstance& model_instance,
            const std::vector<ExecutionPlanNodeSpec>& nodes);

    /// Builds an ExecutionPlan from a LoweredGraph by resolving lowering-time
    /// state aliases into the runtime StateAliasPlan.
    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const LoweredGraph& lowered);

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const ModelInstance& model_instance,
            const LoweredGraph& lowered);
};

}// namespace aethermind

#endif
