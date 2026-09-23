#include "aethermind/model/model_architecture.h"

namespace aethermind {

ModelArchitecture ParseModelArchitecture(const HfModelConfig& config) noexcept {
    // P1 recognizes only dense Llama checkpoints, matching the former
    // HfModelValidator family gate. Qwen / MoE spellings are added when their
    // graph builders land (P2/P3); architectures[0] becomes the primary key at
    // that point, with model_type kept as a cross-check.
    if (config.model_type == "llama") {
        return ModelArchitecture::kLlamaDense;
    }
    return ModelArchitecture::kUnknown;
}

} // namespace aethermind