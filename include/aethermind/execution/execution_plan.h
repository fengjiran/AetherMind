#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_H

#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/base/status.h"
#include "aethermind/operators/operator.h"
#include "aethermind/runtime/workspace.h"

#include <cstddef>
#include <vector>

namespace aethermind {

struct ExecutionStep {
    OpType op_type = OpType::kUnknown;
    KernelInvocation invocation{};
    OperatorPtr op{};
    const void* packed_params = nullptr;
    WorkspaceRequirement workspace_requirement{};
    const char* debug_name = nullptr;
};

class ExecutionPlan {
public:
    Status AddStep(const ExecutionStep& step);

    AM_NODISCARD const std::vector<ExecutionStep>& steps() const noexcept;
    AM_NODISCARD size_t size() const noexcept;

private:
    std::vector<ExecutionStep> steps_{};
};

}// namespace aethermind

#endif
