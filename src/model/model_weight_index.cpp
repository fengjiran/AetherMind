#include "aethermind/model/model_weight_index.h"

namespace aethermind {

size_t ModelWeightIndex::NumLayers() const noexcept {
    return layers.size();
}

}
