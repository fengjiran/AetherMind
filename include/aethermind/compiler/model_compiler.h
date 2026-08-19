#ifndef AETHERMIND_COMPILER_MODEL_COMPILER_H
#define AETHERMIND_COMPILER_MODEL_COMPILER_H

/// @file model_compiler.h
/// @brief Compiler orchestration from LoadedModel to LoweredModelArtifact.

#include "aethermind/base/status.h"
#include "aethermind/compiler/lowered_model_artifact.h"
#include "aethermind/compiler/model_compile_options.h"

#include <filesystem>
#include <memory>

namespace aethermind {

class ModelCompiler {
public:
    /// Builds semantic graph, applies compiler-selected semantic passes, and
    /// lowers without kernel resolution or weight materialization.
    AM_NODISCARD static StatusOr<LoweredModelArtifact> Compile(
            std::unique_ptr<LoadedModel> model,
            const ModelCompileOptions& options = {});

    AM_NODISCARD static StatusOr<LoweredModelArtifact> LoadAndCompile(
            const std::filesystem::path& model_dir,
            const ModelCompileOptions& compile_options = {});
};

}// namespace aethermind

#endif
