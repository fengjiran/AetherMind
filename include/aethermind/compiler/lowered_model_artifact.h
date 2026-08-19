#ifndef AETHERMIND_COMPILER_LOWERED_MODEL_ARTIFACT_H
#define AETHERMIND_COMPILER_LOWERED_MODEL_ARTIFACT_H

/// @file lowered_model_artifact.h
/// @brief Owning result of model compilation before execution planning.

#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/model/loaded_model.h"

#include <memory>

namespace aethermind {

struct LoweredModelArtifact {
    std::unique_ptr<LoadedModel> loaded_model{};
    LoweredGraph graph{};
};

}// namespace aethermind

#endif
