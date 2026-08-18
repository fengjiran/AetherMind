#ifndef AETHERMIND_GRAPH_LOWERING_GRAPH_LOWERING_H
#define AETHERMIND_GRAPH_LOWERING_GRAPH_LOWERING_H

/// @file graph_lowering.h
/// @brief Backend-independent lowering from semantic ModelGraph to execution-plan artifacts.
///
/// Produces a LoweredGraph of ExecutionPlanNodeSpec steps plus graph-value
/// binding vectors and unresolved state alias records. Backend-specific kernel
/// selection happens later; this stage only translates semantic IR.
#include "aethermind/base/device.h"
#include "aethermind/base/kernel_attrs.h"
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
/// The input/output GraphValueIds are distinct graph values, but they must
/// map to the same physical runtime state buffer after execution planning.
/// The step/port coordinates are recorded at lowering time (the KVCacheUpdate
/// step index and its schema cache ports), so resolution only validates and
/// translates them instead of re-scanning step bindings.
struct LoweredStateAlias {
    /// Index of the KVCacheUpdate step in `LoweredGraph::steps` that owns this
    /// alias. Known at lowering time because steps are emitted in topological
    /// order, one per graph node.
    size_t step_index = 0;
    /// Input port of the state value (e.g. k_cache_in) in that step's
    /// schema-port order.
    uint32_t input_port = 0;
    /// Output port of the state value (e.g. k_cache_out) in that step's
    /// schema-port order.
    uint32_t output_port = 0;
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
/// Each lowering-time alias already carries its step/port coordinates (see
/// LoweredStateAlias); this function validates them against the step bindings
/// and produces the runtime ResolvedStateAlias records. It never scans
/// bindings to rediscover coordinates.
///
/// @param lowered LoweredGraph whose state aliases should be resolved.
/// @return StateAliasPlan on success, or an error status if an alias records
/// an out-of-range step or a port that does not bind the recorded value.
StatusOr<StateAliasPlan> ResolveStateAliases(
        const LoweredGraph& lowered);

}// namespace aethermind

#endif
