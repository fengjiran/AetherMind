#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_LOWERING_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_LOWERING_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/model/graph/model_graph.h"
#include "aethermind/model/graph/state_alias_plan.h"
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

/// Records the graph values bound to one lowered execution step. The order of
/// each vector follows the operator schema port order, including state ports
/// that do not contribute tensor specs to ExecutionPlanNodeSpec::input_specs.
struct LoweredStepBinding {
    GraphNodeId node{};
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
};

/// Records a state update that must alias the same physical runtime buffer.
/// KVCacheUpdate produces a new graph state version, but lowering must preserve
/// that the input and output values are versions of the same KV cache family.
struct LoweredStateAlias {
    GraphValueId input{};
    GraphValueId output{};
};

/// Direct 1:1 lowering artifact from semantic ModelGraph to execution planning.
/// `steps` can be passed to ExecutionPlanBuilder; the parallel binding vectors
/// retain graph-value identity for later runtime tensor/state binding.
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

/// Resolves graph-value-based state aliases into step/port-index-based
/// resolved aliases by cross-referencing LoweredGraph::step_bindings.
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
