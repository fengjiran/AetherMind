#ifndef AETHERMIND_INFERENCE_INTERNAL_H
#define AETHERMIND_INFERENCE_INTERNAL_H

/// @file inference_internal.h
/// @brief Private preparation and Decode helpers shared with focused tests.

#include "aethermind/base/status.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace aethermind {

class Backend;
class ExecutionContext;
class ExecutionPlan;
struct ExecutionValueId;
struct WeightPackingRequest;
enum class DeviceType : uint8_t;

} // namespace aethermind

namespace aethermind::inference::internal {

/// @brief Injects descriptor recipes and coalesces compatible packed consumers.
/// @param backend Provider of descriptor-owned packing recipes.
/// @param requests Compiler requests whose recipes have not yet been selected.
/// @param expected_device Device shared by all requests.
/// @return Resolved/coalesced requests, or a device/recipe compatibility error.
StatusOr<std::vector<WeightPackingRequest>> ResolveWeightPackingRequests(
        const Backend& backend,
        std::vector<WeightPackingRequest> requests,
        DeviceType expected_device);

/// @brief Runs the prepared Decode loop shared with InferenceSession.
///
/// generated_tokens already contains the Prefill prediction and has capacity
/// for max_new_tokens. The input pointers refer to stable one-element buffers
/// prepared for the Decode context. next_position is advanced in place.
Status RunDecodeLoop(const ExecutionPlan& plan,
                     ExecutionContext& context,
                     ExecutionValueId token_output,
                     int64_t* input_token,
                     int64_t* next_position,
                     size_t vocab_size,
                     size_t max_new_tokens,
                     std::optional<uint32_t> eos_token_id,
                     std::vector<uint32_t>& generated_tokens);

} // namespace aethermind::inference::internal

#endif
