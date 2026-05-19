#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

size_t ResolvedModelWeights::NumLayers() const noexcept {
    return layers.size();
}

}
