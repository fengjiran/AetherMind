#ifndef AETHERMIND_COMPILER_GRAPH_LOWERING_H
#define AETHERMIND_COMPILER_GRAPH_LOWERING_H

/// @file graph_lowering.h
/// @brief Backend-implementation-independent lowering to compiler planning IR.

#include "aethermind/base/kernel_selector.h"
#include "aethermind/compiler/lowered_graph.h"

namespace aethermind {

/// @brief Target selection data used by lowering.
///
/// ModelGraph deliberately does not retain it; lowering derives selector
/// dtypes from validated operator specs. It carries only base-layer data
/// contracts, never Backend or KernelRegistry.
struct GraphLoweringConfig {
    KernelSelector selector{.device_type = DeviceType::kCPU,
                            .weight_format = WeightFormat::kPlain,
                            .isa = IsaLevel::kScalar,
                            .phase = ExecPhase::kBoth};
    /// When true, steps that consume a kWeight input are marked with
    /// selector.weight_format = kPacked so the execution path resolves their
    /// artifacts from a PackedWeightStore by binding-aware key. Steps without
    /// a kWeight input are always kPlain: packing describes weight storage,
    /// so weightless steps never claim kPacked regardless of this flag or
    /// selector.weight_format.
    bool enable_packed_weights = false;
};

/// @brief Lowers a validated semantic ModelGraph to a finalized compiler
/// artifact.
///
/// The complete input-spec vector follows schema-port order, including state
/// ports. Runtime consumers derive compact tensor specs through the schema.
///
/// @param graph Validated semantic graph to lower.
/// @param config Target selection data; defaults select CPU, plain weights,
///               scalar ISA, and both phases.
/// @return Structurally validated LoweredGraph. Errors are reported via the
///         returned Status, not exceptions.
StatusOr<LoweredGraph> LowerModelGraph(
        const ModelGraph& graph,
        const GraphLoweringConfig& config = {});

}// namespace aethermind

#endif
