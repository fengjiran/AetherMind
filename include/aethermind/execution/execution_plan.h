#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/operators/operator.h"
#include "aethermind/runtime/workspace.h"

#include <cstddef>
#include <vector>

namespace aethermind {

struct ExecutionStep {
    KernelSelector selector{};
    OperatorPtr op{};
    const void* packed_weights = nullptr;
    WorkspaceRequirement workspace_requirement{};
    std::vector<TensorSpec> output_specs{};
    std::vector<ShapeConstraint> runtime_checks{};
    const char* debug_name = nullptr;
};

class ExecutionPlan {
public:
    AM_NODISCARD const std::vector<ExecutionStep>& steps() const noexcept;
    AM_NODISCARD size_t size() const noexcept;

private:
    Status AddStep(const ExecutionStep& step);

    std::vector<ExecutionStep> steps_{};

    friend class ExecutionPlanBuilder;
};

}// namespace aethermind

#endif
