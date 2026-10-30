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
    AM_ASSIGN_OR_RETURN(auto kv_cache,
                        MakeFp32KVCacheOptions(config, options.context_tokens));
    kv_cache.max_bytes = options.kv_max_bytes;
    return RuntimeOptions{
            .allocator = {.enable_cpu = true,
                          .enable_cuda = false,
                          .enable_cann = false},
            .backend = {.enable_cpu = true,
                        .enable_cuda = false,
                        .enable_cann = false,
                        .cpu_feature_policy = options.cpu_feature_policy},
            .execution = {.enable_default_stream = true,
                          .enable_graph_executor = false},
            .workspace = {.enable_workspace_manager = false,
                          .default_workspace_limit_bytes = options.workspace_limit_bytes},
            .tracing = {.enable_profiling = false,
                        .enable_memory_profiling = false},
            .kv_cache = kv_cache,
    };
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
