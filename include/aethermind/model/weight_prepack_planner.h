#ifndef AETHERMIND_MODEL_WEIGHT_PREPACK_PLANNER_H
#define AETHERMIND_MODEL_WEIGHT_PREPACK_PLANNER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/operators/op_type.h"

#include <vector>

namespace aethermind {

class Backend;
class KernelRegistry;
class ModelInstance;

class WeightPrepackPlanner {
public:
    struct Request {
        OpType op_type{};
        RawWeightView raw_weight;
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
    // PackedWeights artifacts into the ModelInstance backend sidecar.
    static Status PrepackAndStore(
            ModelInstance& model_instance,
            const std::vector<Request>& requests);
};

}// namespace aethermind

#endif
