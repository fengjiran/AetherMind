#include "compiler/lowered_graph_draft.h"

#include <utility>

namespace aethermind::compiler_internal {

Status LoweredGraphDraft::Validate() const {
    LoweredGraph lowered;
    lowered.steps_ = steps;
    lowered.values_ = values;
    lowered.model_inputs_ = model_inputs;
    lowered.model_outputs_ = model_outputs;
    lowered.state_aliases_ = state_aliases;
    return ValidateLoweredGraph(lowered);
}

StatusOr<LoweredGraph> LoweredGraphDraft::Finalize() && {
    LoweredGraph lowered;
    lowered.steps_ = std::move(steps);
    lowered.values_ = std::move(values);
    lowered.model_inputs_ = std::move(model_inputs);
    lowered.model_outputs_ = std::move(model_outputs);
    lowered.state_aliases_ = std::move(state_aliases);
    AM_RETURN_IF_ERROR(ValidateLoweredGraph(lowered));
    return lowered;
}

}// namespace aethermind::compiler_internal
