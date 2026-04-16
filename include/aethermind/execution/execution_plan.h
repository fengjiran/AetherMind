#ifndef AETHERMIND_BACKEND_EXECUTION_PLAN_H
#define AETHERMIND_BACKEND_EXECUTION_PLAN_H

#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"
#include "operator_name.h"

#include <cstddef>
#include <vector>

namespace aethermind {

struct ExecutionStep {
    OperatorName op_name{};
    ResolvedKernel kernel{};
};

class ExecutionPlan {
public:
    Status AddStep(const OperatorName& op_name,
                   const ResolvedKernel& kernel);

    AM_NODISCARD const std::vector<ExecutionStep>& steps() const noexcept;
    AM_NODISCARD size_t size() const noexcept;

private:
    std::vector<ExecutionStep> steps_{};
    std::vector<std::vector<std::byte>> owned_attrs_{};
};

}// namespace aethermind

#endif
