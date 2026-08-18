#ifndef AETHERMIND_GRAPH_LOWERING_GRAPH_LOWERING_H
#define AETHERMIND_GRAPH_LOWERING_GRAPH_LOWERING_H

/// @file graph_lowering.h
/// @brief Backend-independent lowering from semantic ModelGraph to execution-plan artifacts.
///
/// Produces a LoweredGraph of ExecutionPlanNodeSpec steps plus graph-value
/// binding vectors and unresolved state alias records. Backend-specific kernel
/// selection happens later; this stage only translates semantic IR.
#include "aethermind/base/kernel_selector.h"
#include "aethermind/execution/execution_node_spec.h"
#include "aethermind/execution/state_alias_plan.h"
#include "aethermind/graph/graph.h"

#include <string>

namespace aethermind {

/// @brief Backend-independent knobs used while translating semantic graph nodes
/// to execution-plan node specs.
///
/// @note ModelGraph intentionally does not store these. The selector's
/// act_dtype/weight_dtype fields are ignored here: lowering infers them from
/// the operator's inputs and outputs.
struct GraphLoweringConfig {
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
};

/// @brief Records the graph values bound to one lowered execution step.
///
/// The order of each vector follows the operator schema port order, including
/// state ports that do not contribute to compact runtime tensor specs.
/// Constant inputs are not indexed separately: their payloads (and any inline
/// ConstantBinding) remain reachable through LoweredGraph::values.
struct LoweredStepBinding {
    GraphNodeId node{};
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
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

/// @brief One lowered execution step paired with its graph-value binding.
///
/// Combining the execution-plan spec and its binding in a single record makes
/// the 1:1 relationship between them a type-level invariant instead of a
/// parallel-vector indexing convention.
struct LoweredStep {
    ExecutionPlanNodeSpec spec{};
    LoweredStepBinding binding{};
};

/// @brief Self-contained semantic metadata for one graph value.
///
/// The entry at `LoweredGraph::values[id.index]` describes GraphValueId `id`.
/// It copies value semantics needed after ModelGraph is destroyed; producer
/// topology remains represented by the step bindings.
struct LoweredValueDesc {
    TensorSpec spec{};
    GraphValuePayload payload{};
    QuantizationSpec quantization{};
    std::string name{};
};

/// @brief Direct 1:1 lowering artifact from semantic ModelGraph to execution
/// planning.
///
/// `steps` pairs each ExecutionPlanNodeSpec with its LoweredStepBinding, so
/// the two can never drift apart; a span of the specs can be handed to
/// ExecutionPlanBuilder, while the bindings retain graph-value identity for
/// later runtime tensor/state binding. `state_aliases` is the only
/// lowering-time collection of unresolved state alias records.
/// ResolveStateAliases() converts it into the runtime StateAliasPlan consumed
/// by ExecutionPlan and the executor.
struct LoweredGraph {
    std::vector<LoweredStep> steps{};
    std::vector<LoweredValueDesc> values{};
    std::vector<GraphValueId> model_inputs{};
    std::vector<GraphValueId> model_outputs{};
    std::vector<LoweredStateAlias> state_aliases{};
};

/// @brief Lowers a semantic ModelGraph into a LoweredGraph using the supplied
/// configuration.
///
/// @note Selector dtype derivation contract. For each step, `act_dtype` is
/// taken from the first contributing activation input port, or - when the
/// operator has no activation input - from the first activation output port.
/// `weight_dtype` is taken from the first contributing weight input port,
/// falling back to `act_dtype` when the operator has no weight input.
/// Operators are expected to keep all activation ports at the same dtype, so
/// the first port is representative. Every schema must expose at least one
/// activation input or output port; otherwise the selector would silently
/// carry an undefined activation dtype (guarded by the schema-contract test
/// `EveryOperatorSchemaExposesActivationPort`).
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
