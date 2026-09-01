#ifndef AETHERMIND_MODEL_MODEL_VALIDATION_OPTIONS_H
#define AETHERMIND_MODEL_MODEL_VALIDATION_OPTIONS_H

/// @file model_validation_options.h
/// @brief Validation switches for HF model directories and weight tables.

namespace aethermind {

/// Toggles controlling how strictly HF models are validated.
///
/// Validation reports mismatches through Status instead of silently fixing
/// them. The defaults accept common HF export quirks (extra tensors, missing
/// tied lm_head) while rejecting unsupported features (quantization,
/// adapters, bias, RoPE scaling).
// NOLINTBEGIN(readability-identifier-naming)
struct ModelValidationOptions {
    // Enforce exact tensor-name matching; off because HF exports may carry
    // schema-unlisted tensors.
    bool strict_tensor_names = false; // NOLINT(readability-identifier-naming)
    // Permit schema-unlisted tensors; known-ignorable ones (e.g. tied lm_head)
    // are skipped when strict_tensor_names is off.
    bool allow_unknown_tensors = true; // NOLINT(readability-identifier-naming)
    // Accept configs with rope_scaling; on because HF configs commonly ship
    // it. Scaling type values (kLinear/kNone) are validated downstream in
    // ModelGraphBuilder::MakeRoPEParams.
    bool allow_rope_scaling = true; // NOLINT(readability-identifier-naming)
    // Accept attention/MLP bias; off because Llama-family models are bias-free.
    bool allow_bias = false; // NOLINT(readability-identifier-naming)
    // Accept quantized weight tensors; off because quantization is unsupported.
    bool allow_quantized_tensors = false; // NOLINT(readability-identifier-naming)
    // Accept LoRA/adapter tensors; off because adapters are unsupported.
    bool allow_lora_or_adapter = false; // NOLINT(readability-identifier-naming)
    // Require lm_head even when embeddings are tied.
    bool require_lm_head_when_tied = false; // NOLINT(readability-identifier-naming)
    // Require one dtype across all linear weights to satisfy fused-kernel
    // assumptions.
    bool require_uniform_linear_dtype = true; // NOLINT(readability-identifier-naming)
};
// NOLINTEND(readability-identifier-naming)

} // namespace aethermind

#endif // AETHERMIND_MODEL_MODEL_VALIDATION_OPTIONS_H
