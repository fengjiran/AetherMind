#ifndef AETHERMIND_INFERENCE_INFERENCE_OPTIONS_H
#define AETHERMIND_INFERENCE_INFERENCE_OPTIONS_H

/// @file inference_options.h
/// @brief Target-independent compile presets and CPU Runtime configuration
///        shared by application entrypoints.

#include "aethermind/base/device.h"
#include "aethermind/base/status.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/runtime/runtime_options.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {

struct HfModelConfig;

enum class CompilePreset : uint8_t {
    kO1Plain,
    kO2Packed,
};

/// @brief Constructs both-phase compile options for a preset and target.
/// @param preset Explicit O1/plain or O2/packed policy; unknown values are rejected.
/// @param target Known device type, independent of backend availability.
/// @return Owning options; lowering derives activation/weight dtypes. Diagnostic
///         allocation failure returns ResourceExhausted with an empty message.
/// @note This does not validate backend/kernel support, load/admit a model,
///       or allocate storage. Invalid targets return InvalidArgument.
StatusOr<ModelCompileOptions> MakeCompileOptions(CompilePreset preset, DeviceType target);

/// @brief User settings combined with model geometry before Runtime construction.
/// @note Zero context capacity is invalid; zero byte caps mean unlimited.
///       Model preparation and session caps retain their separate option types.
struct CpuRuntimeAssemblyOptions {
    size_t context_tokens = 0;
    size_t kv_max_bytes = 0;
    size_t workspace_limit_bytes = 0;
    CpuFeaturePolicy cpu_feature_policy{};
    /// KV storage element dtype; scalar F32/F16/BF16. Defaults to FP32.
    DataType kv_dtype = DataType::Float32();
};

/// @brief Resolves CPU defaults, FP32 KV geometry, budgets and feature policy.
/// @param config Model metadata already validated by ModelLoader.
/// @param options Explicit capacity, optional payload caps and CPU policy.
/// @return Owning RuntimeOptions, or the geometry/capacity mapping error.
///         Diagnostic allocation failure returns empty ResourceExhausted.
/// @note Does not read weight payloads, allocate resource Buffers, inspect a
///       Runtime or retain borrows. RuntimeBuilder validates supported options
///       and applies the physical KV layout/budget before allocation. Changing
///       this result after TryBuild does not reconfigure the existing Runtime.
StatusOr<RuntimeOptions> MakeCpuRuntimeOptions(
        const HfModelConfig& config,
        const CpuRuntimeAssemblyOptions& options);

} // namespace aethermind

#endif
