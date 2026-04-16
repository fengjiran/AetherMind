#ifndef AETHERMIND_EXECUTION_LAYER_RUNNER_H
#define AETHERMIND_EXECUTION_LAYER_RUNNER_H

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan.h"

namespace aethermind {

class LayerRunner {
public:
    AM_NODISCARD static Status Run(const ExecutionPlan& plan) noexcept;

private:
    AM_NODISCARD static Status RunStep(const ExecutionStep& step) noexcept;
};

}// namespace aethermind

#endif
