#include "aethermind/inference/inference_options.h"
#include "aethermind/inference/model_execution_policy.h"

#include <new>

namespace aethermind {

StatusOr<ModelCompileOptions> MakeCompileOptions(
        CompilePreset preset, DeviceType target) try {
    if (preset != CompilePreset::kO1Plain && preset != CompilePreset::kO2Packed) {
        return Status::InvalidArgument("Unknown compile preset");
    }

    if (!IsValidDeviceType(target)) {
        return Status::InvalidArgument("Invalid compilation target");
    }

    ModelCompileOptions options;
    options.optimization.opt_level = preset == CompilePreset::kO1Plain ? 1 : 2;
    options.lowering.enable_packed_weights = preset == CompilePreset::kO2Packed;
    options.lowering.selector = {.device_type = target,
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
