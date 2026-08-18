#ifndef AETHERMIND_MODEL_MODEL_LOADER_H
#define AETHERMIND_MODEL_MODEL_LOADER_H

#include "aethermind/base/status.h"

#include <filesystem>
#include <memory>

namespace aethermind {

class LoadedModel;

class ModelLoader {
public:
    /// Loads and validates an HF model directory into backend-independent
    /// config and logical raw-weight views. It does not build graphs, resolve
    /// kernels, or prepack weights.
    ///
    /// @param model_dir Directory containing config.json and safetensors.
    /// @return LoadedModel owning the config and resolved weights.
    static StatusOr<std::unique_ptr<LoadedModel>> Load(
            const std::filesystem::path& model_dir);
};

}// namespace aethermind

#endif
