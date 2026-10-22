#ifndef AETHERMIND_INFERENCE_MODEL_EXECUTION_POLICY_H
#define AETHERMIND_INFERENCE_MODEL_EXECUTION_POLICY_H

/// @file model_execution_policy.h
/// @brief Shared first-release FP32 admission and model-to-KV configuration.

#include "aethermind/base/status.h"
#include "aethermind/runtime/runtime_options.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind {

class LoadedModel;
struct HfModelConfig;

/// @brief Converts a configuration field that must be strictly positive.
/// @param value Raw configuration value.
/// @param field Field name used in diagnostics.
/// @return The value as size_t, InvalidArgument when non-positive, or
///         OutOfRange when not representable as size_t.
/// @note Diagnostic construction may throw std::bad_alloc.
inline StatusOr<size_t> PositiveSize(int64_t value, std::string_view field) {
    if (value <= 0) {
        return Status::InvalidArgument(std::string(field) + " must be positive");
    }
    if (!std::in_range<size_t>(value)) {
        return Status::OutOfRange(
                std::string(field) + " is not representable as size_t");
    }
    return static_cast<size_t>(value);
}

/// @brief Applies the first-release scalar Float32 execution admission policy.
/// @param model Loaded model whose metadata and required resolved roles are read.
/// @return Ok, Unimplemented for unsupported precision, or an invalid metadata
///         error; diagnostic allocation failure returns empty ResourceExhausted.
/// @note Weight payloads are neither read nor converted and no borrow escapes.
///       Family and operator semantics remain the compiler's responsibility.
Status ValidateFp32ExecutionInputs(const LoadedModel& model);

/// @brief Maps validated model geometry and explicit capacity to FP32 KV options.
/// @param config Configuration already validated by ModelLoader.
/// @param context_tokens Positive KV capacity no larger than the model context.
/// @return Owning KV options with logical head dimension, scalar Float32 and
///         64-byte alignment, or a conversion/geometry/capacity error; diagnostic
///         allocation failure returns empty ResourceExhausted.
/// @note Reads metadata only. It does not allocate storage, build a Runtime,
///       inspect backends, or validate an existing Runtime/model pair. Only an
///       absent head_dim (zero) falls back to exact hidden_size/head-count division.
StatusOr<KVCacheRuntimeOptions> MakeFp32KVCacheOptions(
        const HfModelConfig& config, size_t context_tokens);

} // namespace aethermind

#endif
