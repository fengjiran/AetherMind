#ifndef AETHERMIND_MODEL_MODEL_COMPILER_H
#define AETHERMIND_MODEL_MODEL_COMPILER_H

/// @file model_compiler.h
/// @brief Frontend compilation orchestration from LoadedModel to LoweredModel.

#include "aethermind/base/status.h"
#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/graph/optimization/graph_pass_manager.h"
#include "aethermind/model/lowered_model.h"
#include "aethermind/model/model_load_options.h"

#include <memory>

namespace aethermind {

/// Explicit configuration for semantic optimization and backend-independent
/// graph lowering. It intentionally contains no Backend or KernelRegistry.
struct ModelLoweringOptions {
    PassContext optimization{};
    GraphLoweringConfig lowering{};
};

class ModelCompiler {
public:
    /// Builds the semantic graph, optimizes it, and lowers it without any
    /// kernel resolution or weight materialization.
    static StatusOr<LoweredModel> BuildLoweredModel(
            std::unique_ptr<LoadedModel> model,
            const ModelLoweringOptions& options = {});

    /// Thin convenience facade that loads an HF directory before invoking
    /// BuildLoweredModel. ModelLoader itself remains I/O-only.
    static StatusOr<LoweredModel> LoadAndLowerModel(
            const ModelLoadOptions& load_options,
            const ModelLoweringOptions& lowering_options = {});
};

}// namespace aethermind

#endif
