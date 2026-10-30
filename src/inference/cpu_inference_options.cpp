#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/inference/model_execution_policy.h"

#include <new>

namespace aethermind {

StatusOr<ModelCompileOptions> MakeCpuCompileOptions(CpuInferencePreset preset) try {
    if (preset != CpuInferencePreset::kO1Plain && preset != CpuInferencePreset::kO2Packed) {
        return Status::InvalidArgument("Unknown CPU preset");
    }
    ModelCompileOptions options;
    options.optimization.opt_level = preset == CpuInferencePreset::kO1Plain ? 1 : 2;
    options.lowering.enable_packed_weights = preset == CpuInferencePreset::kO2Packed;
    options.lowering.selector = {.device_type = DeviceType::kCPU,
                                 .act_dtype = {},
                                 .weight_dtype = {},
                                 .weight_format = WeightFormat::kPlain,
                                 .phase = ExecPhase::kBoth};
    return options;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<RuntimeOptions> MakeCpuRuntimeOptions(
        const HfModelConfig& config,
        const CpuRuntimeAssemblyOptions& options) try {
    RuntimeOptions resolved;
    AM_ASSIGN_OR_RETURN(resolved.kv_cache,
                        MakeFp32KVCacheOptions(config, options.context_tokens));
    resolved.kv_cache.max_bytes = options.kv_max_bytes;
    resolved.workspace.default_workspace_limit_bytes = options.workspace_limit_bytes;
    resolved.backend.cpu_feature_policy = options.cpu_feature_policy;
    return resolved;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
