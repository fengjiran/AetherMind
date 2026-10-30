#ifndef AETHERMIND_INFERENCE_RESOURCE_USAGE_H
#define AETHERMIND_INFERENCE_RESOURCE_USAGE_H

#include "aethermind/model/weight/model_weight_normalization.h"
#include "aethermind/model/weight/model_weight_storage_summary.h"

#include <cstddef>
#include <optional>

namespace aethermind {

/// @brief Retained model tensor payload, excluding allocator overhead and RSS.
struct ModelResourceUsage {
    ModelWeightStorageSummary weights{};
    size_t inline_constant_bytes = 0;
    size_t packed_weight_bytes = 0;
    /// Known owned raw backing + inline constants + unique packed storage.
    /// This is a partial observation when weights.backing_complete is false.
    size_t managed_payload_bytes = 0;
    bool packing_plan_complete = true;
    /// Conversion observations; these summaries do not retain source backing.
    std::optional<ModelWeightNormalizationUsage> weight_normalization{};
};

/// @brief Concrete request Buffer requirements, not an allocation or reservation.
///
/// Input/output vectors, kernel params, metadata, allocator rounding, mapped
/// backing and OS residency are outside the request payload cap. Physical KV
/// reservation bytes are independent of the committed-token watermark.
struct RequestResourceUsage {
    size_t workspace_bytes = 0;
    size_t prefill_activation_bytes = 0;
    size_t decode_activation_bytes = 0;
    size_t peak_payload_bytes = 0;
    size_t kv_reserved_bytes = 0;
    size_t model_managed_payload_bytes = 0;
    size_t peak_with_persistent_payload_bytes = 0;
    bool model_payload_complete = true;
};

} // namespace aethermind

#endif
