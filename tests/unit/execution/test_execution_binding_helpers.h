#ifndef AETHERMIND_TEST_EXECUTION_BINDING_HELPERS_H
#define AETHERMIND_TEST_EXECUTION_BINDING_HELPERS_H

#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/runtime_binding_context.h"

#include <utility>
#include <vector>

namespace aethermind::test {

/// Test-only adapter from explicit test vectors to the production canonical
/// value binding API. Activation inputs are deliberately ignored: the plan
/// derives them from their producer output rather than from a second test
/// address.
class ExecutionBindingCollector {
public:
    ExecutionBindingCollector(const ExecutionPlan& plan, Allocator& allocator)
        : plan_(plan), allocator_(allocator), step_bindings_(plan.size()) {}

    void Set(size_t step_index, StepTensorBinding binding) {
        step_bindings_.at(step_index) = std::move(binding);
    }

    Status Install(RuntimeBindingContext& context) const {
        ExternalValueBindings external;
        std::vector<bool> readable_seen(plan_.values().size());
        std::vector<bool> writable_seen(plan_.values().size());
        for (size_t step_index = 0; step_index < plan_.size(); ++step_index) {
            const ExecutionStep& step = plan_.steps()[step_index];
            const StepTensorBinding& binding = step_bindings_[step_index];
            if (binding.inputs.size() != step.kernel_input_ports.size() ||
                binding.outputs.size() != step.kernel_output_ports.size()) {
                return Status::InvalidArgument("Test binding arity does not match ExecutionPlan kernel ports");
            }
            for (size_t index = 0; index < binding.inputs.size(); ++index) {
                const ExecutionValueId value = step.inputs[step.kernel_input_ports[index]];
                if (plan_.values()[value.index].kind == ExecutionValueKind::kActivation) continue;
                if (!readable_seen[value.index]) {
                    external.readable.push_back({.value = value, .tensor = binding.inputs[index]});
                    readable_seen[value.index] = true;
                }
            }
            for (size_t index = 0; index < binding.outputs.size(); ++index) {
                const ExecutionValueId value = step.outputs[step.kernel_output_ports[index]];
                if (!writable_seen[value.index] && binding.outputs[index].is_valid()) {
                    external.writable.push_back({.value = value, .tensor = binding.outputs[index]});
                    writable_seen[value.index] = true;
                }
            }
        }
        auto table = BuildExecutionBindings(plan_, external, allocator_);
        if (!table.ok()) return table.status();
        context.SetBindingTable(std::move(*table));
        return Status::Ok();
    }

private:
    const ExecutionPlan& plan_;
    Allocator& allocator_;
    std::vector<StepTensorBinding> step_bindings_;
};

} // namespace aethermind::test

#endif
