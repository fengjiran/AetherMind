#ifndef AETHERMIND_COMPILER_GRAPH_LOWERING_H
#define AETHERMIND_COMPILER_GRAPH_LOWERING_H

/// @file graph_lowering.h
/// @brief Backend-implementation-independent lowering to compiler planning IR.

#include "aethermind/base/kernel_selector.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/graph/graph.h"

namespace aethermind {

/// Target selection data used by lowering. ModelGraph deliberately does not
/// retain it; lowering derives selector dtypes from validated operator specs.
/// It carries only base-layer data contracts, never Backend or KernelRegistry.
struct GraphLoweringConfig {
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
};

/// Lowers a validated semantic ModelGraph to a finalized compiler artifact.
/// The complete input-spec vector follows schema-port order, including state
/// ports. Runtime consumers derive compact tensor specs through the schema.
AM_NODISCARD StatusOr<LoweredGraph> LowerModelGraph(
        const ModelGraph& graph,
        const GraphLoweringConfig& config = {});

}// namespace aethermind

#endif
