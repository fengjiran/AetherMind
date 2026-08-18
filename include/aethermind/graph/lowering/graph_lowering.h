#ifndef AETHERMIND_GRAPH_LOWERING_GRAPH_LOWERING_H
#define AETHERMIND_GRAPH_LOWERING_GRAPH_LOWERING_H

/// @file graph_lowering.h
/// @brief Backend-independent lowering from semantic ModelGraph to execution-plan artifacts.
///
/// Produces a LoweredGraph of ExecutionPlanNodeSpec steps plus graph-value
/// binding vectors and unresolved state alias records. Backend-specific kernel
/// selection happens later; this stage only translates semantic IR.
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/execution/execution_node_spec.h"
#include "aethermind/execution/state_alias_plan.h"
#include "aethermind/graph/graph.h"

#include <string>

namespace aethermind {

/// @brief Backend-independent knobs used while translating semantic graph nodes
/// to execution-plan node specs.
///
/// @note ModelGraph intentionally does not store these.
struct GraphLoweringConfig {
    DeviceType device_type = DeviceType::kCPU;
    IsaLevel isa = IsaLevel::kScalar;
    WeightFormat weight_format = WeightFormat::kPlain;
    ExecPhase phase = ExecPhase::kBoth;
};

/// @brief Records a constant binding discovered during lowering for one input
/// port.
///
/// @note Backend-specific lowering passes can use this to resolve inline data
/// or named external constants without re-walking the semantic graph.
struct LoweredConstantBinding {
    uint32_t input_port = 0;
    ConstantBinding binding{};
};

/// @brief Records the graph values bound to one lowered execution step.
///
/// The order of each vector follows the operator schema port order, including
/// state ports that do not contribute to compact runtime tensor specs.
/// `constant_bindings` captures ConstantValue payloads encountered on input
/// ports so backend lowering can resolve them without revisiting the graph.
struct LoweredStepBinding {
    GraphNodeId node{};
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
    std::vector<LoweredConstantBinding> constant_bindings{};
};

/// @brief Unresolved lowering-time state alias record.
///
/// @note The input/output GraphValueIds are distinct graph values, but they
/// must map to the same physical runtime state buffer after execution planning.
struct LoweredStateAlias {
    GraphValueId input{};
    GraphValueId output{};
};

/// @brief Self-contained semantic metadata for one graph value.
///
/// The entry at `LoweredGraph::values[id.index]` describes GraphValueId `id`.
/// It copies value semantics needed after ModelGraph is destroyed; producer
/// topology remains represented by step_bindings.
struct LoweredValueDesc {
    TensorSpec spec{};
    GraphValuePayload payload{};
    QuantizationSpec quantization{};
    std::string name{};
};

/// @brief Direct 1:1 lowering artifact from semantic ModelGraph to execution
/// planning.
///
/// `steps` can be passed to ExecutionPlanBuilder; the parallel binding vectors
/// retain graph-value identity for later runtime tensor/state binding.
/// `state_aliases` is the only lowering-time collection of unresolved state
/// alias records. ResolveStateAliases() converts it into the runtime
/// StateAliasPlan consumed by ExecutionPlan and the executor.
struct LoweredGraph {
    std::vector<ExecutionPlanNodeSpec> steps{};
    std::vector<LoweredStepBinding> step_bindings{};
    std::vector<LoweredValueDesc> values{};
    std::vector<GraphValueId> model_inputs{};
    std::vector<GraphValueId> model_outputs{};
    std::vector<LoweredStateAlias> state_aliases{};
};

/// @brief Lowers a semantic ModelGraph into a LoweredGraph using the supplied
/// configuration.
///
/// @param graph Source semantic graph. Not mutated.
/// @param config Backend-independent lowering knobs.
/// @return LoweredGraph on success, or an error status describing the first
/// lowering failure.
StatusOr<LoweredGraph> LowerModelGraph(
        const ModelGraph& graph,
        const GraphLoweringConfig& config = {});

/// @brief The only conversion point from lowering-time GraphValueId aliases to
/// the runtime step/port aliases stored in StateAliasPlan.
///
/// state_aliases[i] = {input GraphValueId, output GraphValueId}
/// is resolved to ResolvedStateAlias{
///   .step_index  = step whose bindings contain both values,
///   .input_port  = port index of input in that step's input_values,
///   .output_port = port index of output in that step's output_values,
/// }.
///
/// @param lowered LoweredGraph whose state aliases should be resolved.
/// @return StateAliasPlan on success, or an error status if an alias references
/// a GraphValueId that cannot be found in any step binding.
StatusOr<StateAliasPlan> ResolveStateAliases(
        const LoweredGraph& lowered);

}// namespace aethermind

#endif
