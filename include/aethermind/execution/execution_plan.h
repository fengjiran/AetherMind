#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_H

#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/backend/workspace_types.h"
#include "aethermind/base/status.h"

#include <cstddef>
#include <span>
#include <vector>

namespace aethermind {

struct ExecutionStep {
    OpType op_type = OpType::kUnknown;
    KernelFunc fn = nullptr;
    const void* packed_params = nullptr;
    WorkspaceRequirement workspace_requirement{};
    std::span<const std::byte> attrs{};
    const char* debug_name = nullptr;
};

class ExecutionPlan {
public:
    Status AddStep(const ExecutionStep& step);
    Status AddStep(const ResolvedKernel& kernel,
                   const WorkspaceRequirement& workspace_requirement = {},
                   const void* packed_params = nullptr);

    AM_NODISCARD const std::vector<ExecutionStep>& steps() const noexcept;
    AM_NODISCARD size_t size() const noexcept;

private:
    std::vector<ExecutionStep> steps_{};
    std::vector<std::vector<std::byte>> owned_attrs_{};
};

}// namespace aethermind

#endif
