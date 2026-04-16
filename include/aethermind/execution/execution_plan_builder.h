#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_BUILDER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/runtime/runtime_context.h"
#include "data_type.h"

#include <span>
#include <vector>

namespace aethermind {

struct ExecutionPlanNodeSpec {
    OpType op_type = OpType::kUnknown;
    DeviceType device_type = DeviceType::kCPU;
    DataType activation_dtype{};
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
    std::span<const std::byte> attrs{};
};

class ExecutionPlanBuilder {
public:
    AM_NODISCARD static StatusOr<ResolvedKernel> ResolveKernelForNode(
            const Backend& backend,
            const ExecutionPlanNodeSpec& node) noexcept;

    AM_NODISCARD static StatusOr<ExecutionPlan> Build(
            RuntimeContext& runtime,
            const std::vector<ExecutionPlanNodeSpec>& nodes);
};

}// namespace aethermind

#endif
