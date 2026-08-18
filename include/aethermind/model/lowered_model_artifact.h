#ifndef AETHERMIND_MODEL_LOWERED_MODEL_ARTIFACT_H
#define AETHERMIND_MODEL_LOWERED_MODEL_ARTIFACT_H

/// @file lowered_model_artifact.h
/// @brief Owning artifact for model compilation through graph lowering.

#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/model/loaded_model.h"

#include <memory>

namespace aethermind {

/// Keeps the loaded model and its backend-independent lowering artifact alive
/// as one compilation result.
///
/// The LoadedModel member owns RawWeightView backing storage needed by the
/// next, graph-driven weight-materialization phase. No backend kernel, packed
/// weight, or ExecutionPlan is resolved at this boundary.
struct LoweredModelArtifact {
    std::unique_ptr<LoadedModel> loaded_model{};
    LoweredGraph graph{};
};

}// namespace aethermind

#endif
