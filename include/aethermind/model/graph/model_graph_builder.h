#ifndef AETHERMIND_MODEL_GRAPH_MODEL_GRAPH_BUILDER_H
#define AETHERMIND_MODEL_GRAPH_MODEL_GRAPH_BUILDER_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/model_graph.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

class ModelGraphBuilder {
public:
    AM_NODISCARD static StatusOr<ModelGraph> BuildLlamaDense(
            const HfModelConfig& config,
            const ResolvedModelWeights& weights);
};

}// namespace aethermind

#endif
