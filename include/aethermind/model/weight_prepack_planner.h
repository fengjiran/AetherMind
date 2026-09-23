#ifndef AETHERMIND_MODEL_WEIGHT_PREPACK_PLANNER_H
#define AETHERMIND_MODEL_WEIGHT_PREPACK_PLANNER_H

#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/operators/op_type.h"

#include <vector>

namespace aethermind {

class PackedWeightStore;

class WeightPrepackPlanner {
public:
    struct Request {
        OpType op_type{};
        /// Source artifact id from the producing LoweredGraph.
        uint64_t source_id = 0;
        /// Artifact-local weight value id (GraphValueId) this request packs.
        uint32_t value_index = 0;
        /// Logical weight binding (layer index + role) used as artifact
        /// identity together with the selector.
        WeightBinding binding{};
        /// Raw weight view for direct bindings. Composite bindings
        /// (QkvWeightBinding / GateUpWeightBinding) leave this empty and carry
        /// the recipe-ordered components instead; the planner materializes the
        /// fused view from `components` during prepacking.
        RawWeightView raw_weight;
        /// Recipe-ordered raw components of a composite binding: Q, K, V for
        /// QkvWeightBinding; Gate, Up for GateUpWeightBinding. Empty for
        /// direct bindings, whose single view lives in `raw_weight`.
        std::vector<RawWeightView> components{};
        KernelSelector selector;
    };

    // Executes prepack for every request and stores the resulting
    // PackedWeights artifacts into a PackedWeightStore. Packing identity
    // remains {source_id, value_index, binding, selector, recipe}; the exact
    // PackingRecipe, not CPU feature detection, distinguishes layouts.
    // Production requests come from the graph-driven
    // BuildWeightPackingRequests; this planner only executes them.
    static Status PrepackAndStore(
            PackedWeightStore& packed_weight_store,
            const std::vector<Request>& requests);
};

} // namespace aethermind

#endif
