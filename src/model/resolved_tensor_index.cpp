#include "aethermind/model/resolved_tensor_index.h"

namespace aethermind {

size_t ResolvedTensorIndex::NumLayers() const noexcept {
    return layers.size();
}

}
