#ifndef AETHERMIND_MODEL_MODEL_COMPILE_OPTIONS_H
#define AETHERMIND_MODEL_MODEL_COMPILE_OPTIONS_H

/// @file model_compile_options.h
/// @brief Configuration for semantic optimization and graph lowering.

#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// Explicit configuration for semantic optimization and backend-independent
/// graph lowering. It intentionally contains no Backend or KernelRegistry.
struct ModelCompileOptions {
    PassContext optimization{};
    GraphLoweringConfig lowering{};
};

}// namespace aethermind

#endif
