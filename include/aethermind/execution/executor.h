#ifndef AETHERMIND_EXECUTION_EXECUTOR_H
#define AETHERMIND_EXECUTION_EXECUTOR_H

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan.h"

namespace aethermind {

class Executor {
public:
    AM_NODISCARD static Status Execute(const ExecutionPlan& plan) noexcept;
};

}// namespace aethermind

#endif
