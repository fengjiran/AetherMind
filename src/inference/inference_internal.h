#ifndef AETHERMIND_INFERENCE_INTERNAL_H
#define AETHERMIND_INFERENCE_INTERNAL_H

/// @file inference_internal.h
/// @brief Private preparation and generation helpers shared with focused tests.

#include "aethermind/base/status.h"
#include "aethermind/inference/inference_session.h"

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

/// @brief Validates a plan against the Llama session I/O contract.
///
/// Input roles are positional: `model_inputs()[0]` is bound as token_ids and
/// `model_inputs()[1]` as position_ids, matching the Llama graph builder's
/// append order and lowering's order preservation. Both inputs share the same
/// dtype/rank, so a swapped pair is not detectable here; adding model families
/// or changing input order requires updating this contract.
///
/// Contract mismatches (I/O counts, dtype/rank, non-CPU device, Argmax topology,
/// or a non-activation model output) return kUnimplemented. Violations of
/// ExecutionPlan construction invariants (value ids, duplicate inputs, input
/// kinds) return kInvalidArgument. Exposed in this header so focused tests can
/// exercise the failure branches with synthetic plans.
StatusOr<detail::PhaseContract> ValidatePlanContract(
        const ExecutionPlan& plan);

/// @brief Injects descriptor recipes and coalesces compatible packed consumers.
/// @param backend Provider of descriptor-owned packing recipes.
/// @param requests Compiler requests whose recipes have not yet been selected.
/// @param expected_device Device shared by all requests.
/// @return Resolved/coalesced requests, or a device/recipe compatibility error.
StatusOr<std::vector<WeightPackingRequest>> ResolveWeightPackingRequests(
        const Backend& backend,
        std::vector<WeightPackingRequest> requests,
        DeviceType expected_device);

/// @brief Executes a prepared Prefill plan and returns its first predicted token.
/// @param plan Immutable plan matching the prepared context.
/// @param context Prepared resources borrowing a fresh KV reservation.
/// @param token_output Value containing one predicted token per prompt position.
/// @param prompt_length Non-zero prompt length, representable as Int64 and equal
///                      to the reservation's prompt length.
/// @param vocab_size Non-zero upper bound on valid token IDs.
/// @return The prediction at the final prompt position, or a phase, execution,
///         commit-position, or output-validation error.
/// @note Does not clear the context or release the reservation. If execution
///       commits before output validation fails, the committed watermark is
///       retained; the caller remains responsible for request cleanup.
/// @throws std::bad_alloc If error diagnostic allocation fails.
StatusOr<uint32_t> RunPrefill(const ExecutionPlan& plan,
                              ExecutionContext& context,
                              ExecutionValueId token_output,
                              size_t prompt_length,
                              size_t vocab_size);

/// @brief Runs the prepared Decode loop shared with InferenceSession.
///
/// generated_tokens already contains the Prefill prediction and has capacity
/// for max_new_tokens. The input pointers refer to stable one-element buffers
/// prepared for the Decode context. next_position is advanced in place.
Status RunDecodeLoop(const ExecutionPlan& plan,
                     ExecutionContext& context,
                     ExecutionValueId token_output,
                     int64_t* input_token,
                     int64_t* next_pos,
                     size_t vocab_size,
                     size_t max_new_tokens,
                     std::optional<uint32_t> eos_token_id,
                     std::vector<uint32_t>& generated_tokens);

} // namespace aethermind::inference::internal

#endif
