#ifndef AETHERMIND_INFERENCE_CPU_INFERENCE_OPTIONS_H
#define AETHERMIND_INFERENCE_CPU_INFERENCE_OPTIONS_H

/// @file cpu_inference_options.h
/// @brief Explicit CPU compilation policy shared by application entrypoints.

#include "aethermind/compiler/model_compiler.h"

#include <cstdint>

namespace aethermind {

enum class CpuInferencePreset : uint8_t {
    kO1Plain,
    kO2Packed,
};

/// @brief Constructs both-phase CPU compile options for a supported preset.
/// @param preset Explicit O1/plain or O2/packed policy; unknown values are rejected.
/// @return Owning options; lowering derives activation/weight dtypes. Diagnostic
///         allocation failure returns ResourceExhausted with an empty message.
/// @note This does not load/admit a model, resolve kernels, or allocate storage.
StatusOr<ModelCompileOptions> MakeCpuCompileOptions(CpuInferencePreset preset);

} // namespace aethermind

#endif
