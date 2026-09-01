#ifndef AETHERMIND_MODEL_MODEL_LOADER_H
#define AETHERMIND_MODEL_MODEL_LOADER_H

/// @file model_loader.h
/// @brief I/O-only entry point for loading HF model directories.
///
/// Loader responsibilities: directory I/O, config parsing, validation, and
/// logical weight resolution. It deliberately stays free of graph building,
/// kernel resolution, and weight prepacking.
/// @see docs/designs/model/01-model-loader.md
#include "aethermind/base/status.h"

#include <filesystem>
#include <memory>

namespace aethermind {

class LoadedModel;

/// @brief Loads and validates HuggingFace model directories.
///
/// A stateless facade over the HF loading pipeline; all failures are
/// reported through StatusOr values.
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

} // namespace aethermind

#endif
