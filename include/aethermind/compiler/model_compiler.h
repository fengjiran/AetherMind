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

/// @brief Compiles LoadedModel instances into LoweredModelArtifacts.
///
/// A stateless facade over the compiler pipeline: semantic graph construction,
/// compiler-selected semantic passes, and lowering. It never resolves kernels
/// or materializes weights; those steps are deferred to execution planning.
class ModelCompiler {
public:
    /// @brief Builds the semantic graph, applies semantic optimization passes,
    /// and lowers it into a LoweredModelArtifact.
    ///
    /// @param model Model to compile. Ownership is transferred into the
    ///              returned artifact.
    /// @param options Optimization and lowering configuration.
    /// @return Compiled artifact owning `model`.
    /// @note Errors are reported via the returned Status, not exceptions.
    static StatusOr<LoweredModelArtifact> Compile(
            std::unique_ptr<LoadedModel> model,
            const ModelCompileOptions& options = {});

    /// @brief Loads a model directory and compiles it.
    ///
    /// @param model_dir Directory containing config.json and safetensors.
    /// @param compile_options Optimization and lowering configuration.
    /// @return Compiled artifact owning the loaded model.
    /// @note Errors are reported via the returned Status, not exceptions.
    static StatusOr<LoweredModelArtifact> LoadAndCompile(
            const std::filesystem::path& model_dir,
            const ModelCompileOptions& compile_options = {});
};

}// namespace aethermind

#endif
