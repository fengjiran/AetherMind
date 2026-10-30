#include "aethermind/model/build_model_graph.h"
#include "aethermind/model/llama_dense_graph_builder.h"
#include "aethermind/model/model_architecture.h"

#include <string>

namespace aethermind {

StatusOr<ModelGraph> BuildModelGraph(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights) {
    return BuildModelGraph(config, weights, {});
}

StatusOr<ModelGraph> BuildModelGraph(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights,
                                     const ModelGraphBuildOptions& options) {
    switch (ParseModelArchitecture(config)) {
        case ModelArchitecture::kLlamaDense:
            return BuildLlamaDense(config, weights, options);
        default:
            return Status::InvalidArgument(
                    "BuildModelGraph: unsupported model type '" + config.model_type +
                    "' (only Llama dense models are supported)");
    }
}

} // namespace aethermind
