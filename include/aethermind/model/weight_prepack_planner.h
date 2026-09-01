#ifndef AETHERMIND_MODEL_WEIGHT_PREPACK_PLANNER_H
#define AETHERMIND_MODEL_WEIGHT_PREPACK_PLANNER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/operators/op_type.h"

#include <vector>

namespace aethermind {

class Backend;
class KernelRegistry;
class PackedWeightStore;

class WeightPrepackPlanner {
public:
    struct Request {
        OpType op_type{};
        /// Source artifact id from the producing LoweredModelArtifact.
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

    // Generates a list of tensors that require weight prepacking.
    // Embeddings, RMSNorm, and final_norm are intentionally excluded;
    // only linear projection weights (q/k/v/o/gate/up/down/lm_head) are requested.
    static StatusOr<std::vector<Request>> BuildRequests(
            const HfModelConfig& config,
            const ResolvedModelWeights& resolved_weights,
            const Backend& backend,
            const KernelRegistry& registry);

    // Executes prepack for every request and stores the resulting
    // PackedWeights artifacts into a PackedWeightStore. Packing identity
    // remains {source_id, value_index, binding, selector, recipe}; the exact
    // PackingRecipe, not CPU feature detection, distinguishes layouts.
    // ModelLoader does not call this legacy planner; graph-driven artifact
    // materialization is the production integration point.
    static Status PrepackAndStore(
            PackedWeightStore& packed_weight_store,
            const std::vector<Request>& requests);
};

} // namespace aethermind

#endif
