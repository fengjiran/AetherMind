#ifndef AETHERMIND_MODEL_GRAPH_COMPILATION_GRAPH_LOWERING_H
#define AETHERMIND_MODEL_GRAPH_COMPILATION_GRAPH_LOWERING_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/model/graph/compilation/state_alias_plan.h"
#include "aethermind/model/graph/graph.h"
#include "macros.h"

#include <vector>

namespace aethermind {

/// Backend-independent knobs used while translating semantic graph nodes to
/// execution-plan node specs. ModelGraph intentionally does not store these.
struct GraphLoweringConfig {
    DeviceType device_type = DeviceType::kCPU;
    IsaLevel isa = IsaLevel::kScalar;
    WeightFormat weight_format = WeightFormat::kPlain;
    ExecPhase phase = ExecPhase::kBoth;
};

/// Records a constant binding discovered during lowering for one input port.
/// Backend-specific lowering passes can use this to resolve inline data or
/// named external constants without re-walking the semantic graph.
struct LoweredConstantBinding {
    uint32_t input_port = 0;
    ConstantBinding binding{};
};

/// Records the graph values bound to one lowered execution step. The order of
/// each vector follows the operator schema port order, including state ports
/// that do not contribute tensor specs to ExecutionPlanNodeSpec::input_specs.
/// `constant_bindings` captures ConstantValue payloads encountered on input
/// ports so backend lowering can resolve them without revisiting the graph.
struct LoweredStepBinding {
    GraphNodeId node{};
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
    std::vector<LoweredConstantBinding> constant_bindings{};
};

/// Unresolved lowering-time state alias record.
/// The input/output GraphValueIds are distinct graph values, but they must map
/// to the same physical runtime state buffer after execution planning.
struct LoweredStateAlias {
    GraphValueId input{};
    GraphValueId output{};
};

/// Direct 1:1 lowering artifact from semantic ModelGraph to execution planning.
/// `steps` can be passed to ExecutionPlanBuilder; the parallel binding vectors
/// retain graph-value identity for later runtime tensor/state binding.
/// `state_aliases` is the only lowering-time collection of unresolved state
/// alias records. ResolveStateAliases() converts it into the runtime
/// StateAliasPlan consumed by ExecutionPlan and the executor.
struct LoweredGraph {
    std::vector<ExecutionPlanNodeSpec> steps{};
    std::vector<LoweredStepBinding> step_bindings{};
    std::vector<GraphValueId> model_inputs{};
    std::vector<GraphValueId> model_outputs{};
    std::vector<LoweredStateAlias> state_aliases{};
};

AM_NODISCARD StatusOr<LoweredGraph> LowerModelGraph(
        const ModelGraph& graph,
        const GraphLoweringConfig& config = {});

/// The only conversion point from lowering-time GraphValueId aliases to the
/// runtime step/port aliases stored in StateAliasPlan.
///
/// state_aliases[i] = {input GraphValueId, output GraphValueId}
/// is resolved to ResolvedStateAlias{
///   .step_index  = step whose bindings contain both values,
///   .input_port  = port index of input in that step's input_values,
///   .output_port = port index of output in that step's output_values,
/// }.
///
/// Returns an error if an alias references a GraphValueId that cannot
/// be found in any step binding.
AM_NODISCARD StatusOr<StateAliasPlan> ResolveStateAliases(
        const LoweredGraph& lowered);

}// namespace aethermind

#endif
