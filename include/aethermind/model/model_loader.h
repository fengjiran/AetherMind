#ifndef AETHERMIND_MODEL_MODEL_LOADER_H
#define AETHERMIND_MODEL_MODEL_LOADER_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_load_options.h"

#include <memory>

namespace aethermind {

class Backend;
class KernelRegistry;
class ModelInstance;

class ModelLoader {
public:
    static StatusOr<std::unique_ptr<ModelInstance>> Load(
            const ModelLoadOptions& options,
            const Backend& backend,
            const KernelRegistry& registry);
};

}// namespace aethermind

#endif
