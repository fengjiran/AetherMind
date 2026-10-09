#ifndef AETHERMIND_MODEL_WEIGHT_PACKING_REQUEST_H
#define AETHERMIND_MODEL_WEIGHT_PACKING_REQUEST_H

/// @file weight_packing_request.h
/// @brief Weight-packing request data shared by compilation and preparation.

#include "aethermind/backend/packing_recipe.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/operators/op_type.h"

#include <cstdint>
#include <vector>

namespace aethermind {

/// @brief One packed consumer request, keyed by its weight value in the artifact.
struct WeightPackingRequest {
    OpType op_type{};
    /// Source artifact id from the producing LoweredGraph.
    uint64_t source_id = 0;
    /// Artifact-local weight value id (GraphValueId) this request packs.
    uint32_t value_index = 0;
    /// Logical weight binding (layer index + role) used as artifact
    /// identity together with the selector.
    WeightBinding binding{};
    /// Non-empty raw weight views in logical binding order: one view for a
    /// direct binding, Q/K/V for QkvWeightBinding, Gate/Up for GateUpWeightBinding.
    /// Each view retains its backing storage; the backend materializes any
    /// composite artifact and owns the physical packing layout.
    std::vector<RawWeightView> components{};
    KernelSelector selector{};
    /// Empty when compilation builds the request. Preparation selects the
    /// exact consumer descriptor recipe before PrepackWeightRequests runs.
    PackingRecipe recipe{};
};

} // namespace aethermind

#endif
