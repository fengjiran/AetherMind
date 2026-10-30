#ifndef AETHERMIND_INFERENCE_INFERENCE_SESSION_H
#define AETHERMIND_INFERENCE_INFERENCE_SESSION_H

/// @file inference_session.h
/// @brief Synchronous token generation over a prepared executable model.

#include "aethermind/base/status.h"
#include "aethermind/inference/resource_usage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace aethermind::inference::detail {

/// @brief Value indices of the token/position inputs and token output within
///        one phase plan. Filled in by ValidatePlanContract.
struct PhaseContract {
    uint32_t token_input = 0;
    uint32_t position_input = 0;
    uint32_t token_output = 0;
};

} // namespace aethermind::inference::detail

namespace aethermind {

class ExecutableModel;
class Runtime;

/// @brief Greedy generation limits and stopping token.
struct GenerationConfig {
    size_t max_new_tokens = 0;
    std::optional<uint32_t> eos_token_id{};
};

/// @brief Per-request workspace + peak activation payload cap; zero is unlimited.
struct InferenceSessionOptions {
    size_t max_request_payload_bytes = 0;
};

/// @brief Synchronous, single-request orchestration over an ExecutableModel.
///
/// The session borrows Runtime, which must outlive it, and shares ownership
/// of the immutable executable model. All Generate calls sharing a Runtime
/// require external serialization, including calls on different sessions. Each
/// call starts with a fresh Prefill and releases its KV reservation before it
/// returns, including on execution or allocation failure.
class InferenceSession {
public:
    /// @brief Creates a session for a model prepared by PrepareExecutableModel.
    ///
    /// A Runtime without a KVCacheManager is allowed; Generate reports that
    /// missing resource only when a positive token limit is requested.
    static StatusOr<InferenceSession> Create(
            Runtime& runtime,
            std::shared_ptr<const ExecutableModel> model);

    /// @brief Creates a session with an explicit per-request payload cap.
    static StatusOr<InferenceSession> Create(
            Runtime& runtime,
            std::shared_ptr<const ExecutableModel> model,
            const InferenceSessionOptions& options);

    /// @brief Reports requirements for a concrete prompt length without
    ///        reserving KV, allocating tensor Buffers or executing kernels.
    /// @note Cold-path metadata/control vectors may allocate. This checks
    ///       geometry/lifecycle but not prompt token values, and does not apply
    ///       the limits, so callers can inspect an over-budget requirement.
    AM_NODISCARD StatusOr<RequestResourceUsage> PlanResources(
            size_t prompt_tokens, const GenerationConfig& config) const;

    InferenceSession(InferenceSession&&) noexcept = default;
    InferenceSession& operator=(InferenceSession&&) noexcept = default;
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
    ~InferenceSession() = default;

    /// @brief Generates token IDs, including the first token predicted by
    ///        Prefill and including EOS when it is produced.
    ///
    /// An empty prompt is invalid. A zero token limit returns an empty vector
    /// without reserving KV or executing the model. For a positive limit, the
    /// Prefill prediction is the first generated token, so at most
    /// max_new_tokens - 1 Decode KV appends are needed.
    /// std::bad_alloc propagated from request buffer, binding, workspace or
    /// activation allocation returns ResourceExhausted with an empty message,
    /// so reporting the failure does not allocate additional memory.
    AM_NODISCARD StatusOr<std::vector<uint32_t>> Generate(
            std::span<const uint32_t> prompt_tokens,
            const GenerationConfig& config) const;

private:
    using PhaseContract = inference::detail::PhaseContract;

    InferenceSession(Runtime& runtime,
                     std::shared_ptr<const ExecutableModel> model,
                     PhaseContract prefill_contract,
                     PhaseContract decode_contract,
                     size_t max_request_payload_bytes) noexcept;

    Runtime* runtime_ = nullptr;
    std::shared_ptr<const ExecutableModel> model_{};
    PhaseContract prefill_contract_{};
    PhaseContract decode_contract_{};
    size_t max_request_payload_bytes_ = 0;
};

} // namespace aethermind

#endif
