#ifndef AETHERMIND_TEST_EXECUTION_OBSERVATION_HELPERS_H
#define AETHERMIND_TEST_EXECUTION_OBSERVATION_HELPERS_H

#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/runtime/runtime.h"
#include "execution/test_tensor_buffer_helpers.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace aethermind::test::observation {

// Canonical decoder tests bind token_ids and position_ids in model input order.
inline StatusOr<ExecutionContext> PrepareContext(Runtime& runtime,
                                                 const ExecutionPlan& plan,
                                                 const ExternalTensorBindings& immutable_bindings,
                                                 const TestBuffer& tokens,
                                                 const TestBuffer& positions,
                                                 WorkspaceArena* workspace,
                                                 KVCacheSessionView cache_view) {
    if (plan.model_inputs().size() != 2) {
        return Status::Internal("Decoder observation requires token and position inputs");
    }
    ExternalTensorBindings external = immutable_bindings;
    external.read_only.push_back({.value = plan.model_inputs()[0], .tensor = tokens.view()});
    external.read_only.push_back({.value = plan.model_inputs()[1], .tensor = positions.view()});
    AM_ASSIGN_OR_RETURN(auto prepared, PrepareExecutionBindings(plan, external, runtime.GetAllocator(Device::CPU())));
    return ExecutionContext::Create(plan, std::move(prepared), workspace, cache_view);
}

inline MutableTensorView FindOutput(const ExecutionContext& context,
                                    const ExecutionPlan& plan,
                                    ExecutionValueId value) {
    const auto* const prepared = context.prepared_bindings();
    if (prepared == nullptr) {
        return {};
    }
    for (size_t index = 0; index < plan.steps().size(); ++index) {
        const auto& step = plan.steps()[index];
        for (size_t output = 0; output < step.kernel_output_ports.size(); ++output) {
            const uint32_t port = step.kernel_output_ports[output];
            if (port < step.outputs.size() && step.outputs[port] == value) {
                return prepared->step(index).outputs[output];
            }
        }
    }
    return {};
}

} // namespace aethermind::test::observation

#endif
