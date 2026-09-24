#ifndef AETHERMIND_INFERENCE_INFERENCE_SESSION_INTERNAL_H
#define AETHERMIND_INFERENCE_INFERENCE_SESSION_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace aethermind::inference::internal {

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
