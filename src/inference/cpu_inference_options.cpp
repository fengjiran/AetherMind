#include "aethermind/inference/cpu_inference_options.h"

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

} // namespace aethermind
