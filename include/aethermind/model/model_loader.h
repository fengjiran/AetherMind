#ifndef AETHERMIND_MODEL_MODEL_LOADER_H
#define AETHERMIND_MODEL_MODEL_LOADER_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_load_options.h"

#include <memory>

namespace aethermind {

class LoadedModel;

class ModelLoader {
public:
    /// Loads and validates an HF model directory into backend-independent
    /// config and logical raw-weight views. It does not build graphs, resolve
    /// kernels, or prepack weights.
    static StatusOr<std::unique_ptr<LoadedModel>> Load(const ModelLoadOptions& options);
};

}// namespace aethermind

#endif
