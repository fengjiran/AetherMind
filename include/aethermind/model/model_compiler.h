#ifndef AETHERMIND_MODEL_MODEL_COMPILER_H
#define AETHERMIND_MODEL_MODEL_COMPILER_H

/// @file model_compiler.h
/// @brief Compilation orchestration from LoadedModel to LoweredModelArtifact.

#include "aethermind/base/status.h"
#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/model/lowered_model_artifact.h"
#include "aethermind/model/model_compile_options.h"

#include <filesystem>
#include <memory>

namespace aethermind {

class ModelCompiler {
public:
    /// Builds the semantic graph, optimizes it, and lowers it without any
    /// kernel resolution or weight materialization.
    static StatusOr<LoweredModelArtifact> Compile(
            std::unique_ptr<LoadedModel> model,
            const ModelCompileOptions& options = {});

    /// Thin convenience facade that loads an HF directory before invoking
    /// Compile. ModelLoader itself remains I/O-only.
    ///
    /// @param model_dir Directory containing config.json and safetensors.
    /// @param compile_options Options for optimization and lowering.
    /// @return LoweredModelArtifact on success.
    static StatusOr<LoweredModelArtifact> LoadAndCompile(
            const std::filesystem::path& model_dir,
            const ModelCompileOptions& compile_options = {});
};

}// namespace aethermind

#endif
