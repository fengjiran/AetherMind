#ifndef AETHERMIND_EXECUTION_EXECUTOR_H
#define AETHERMIND_EXECUTION_EXECUTOR_H

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {

class Executor {
public:
    AM_NODISCARD static Status Execute(const ExecutionPlan& plan,
                                       RuntimeBindingContext& bindings) noexcept;
};

}// namespace aethermind

#endif
