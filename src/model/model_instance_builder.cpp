#include "aethermind/model/model_instance_builder.h"
#include "aethermind/model/model_instance.h"

#include <utility>

namespace aethermind {

StatusOr<std::unique_ptr<ModelInstance>> ModelInstanceBuilder::Create(
        HfModelConfig config,
        ResolvedModelWeights resolved_weights) {
    return std::make_unique<ModelInstance>(std::move(config), std::move(resolved_weights));
}

}// namespace aethermind
