#ifndef AETHERMIND_MODEL_MODEL_VALIDATION_OPTIONS_H
#define AETHERMIND_MODEL_MODEL_VALIDATION_OPTIONS_H

namespace aethermind {

// NOLINTBEGIN(readability-identifier-naming)
struct ModelValidationOptions {
    bool strict_tensor_names = false;        // NOLINT(readability-identifier-naming)
    bool allow_unknown_tensors = true;       // NOLINT(readability-identifier-naming)
    bool allow_rope_scaling = false;         // NOLINT(readability-identifier-naming)
    bool allow_bias = false;                 // NOLINT(readability-identifier-naming)
    bool allow_quantized_tensors = false;    // NOLINT(readability-identifier-naming)
    bool allow_lora_or_adapter = false;      // NOLINT(readability-identifier-naming)
    bool require_lm_head_when_tied = false;  // NOLINT(readability-identifier-naming)
    bool require_uniform_linear_dtype = true;// NOLINT(readability-identifier-naming)
};
// NOLINTEND(readability-identifier-naming)

}// namespace aethermind

#endif// AETHERMIND_MODEL_MODEL_VALIDATION_OPTIONS_H
