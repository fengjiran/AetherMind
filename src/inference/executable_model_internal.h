#ifndef AETHERMIND_INFERENCE_EXECUTABLE_MODEL_INTERNAL_H
#define AETHERMIND_INFERENCE_EXECUTABLE_MODEL_INTERNAL_H

#include "aethermind/backend/backend.h"
#include "aethermind/model/weight/weight_packing.h"

#include <vector>

namespace aethermind::inference::internal {

/// @brief Injects descriptor recipes and coalesces compatible packed consumers.
StatusOr<std::vector<WeightPackingRequest>> ResolveWeightPackingRequests(
        const Backend& backend,
        std::vector<WeightPackingRequest> requests,
        DeviceType expected_device);

} // namespace aethermind::inference::internal

#endif // AETHERMIND_INFERENCE_EXECUTABLE_MODEL_INTERNAL_H
