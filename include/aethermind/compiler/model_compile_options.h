#ifndef AETHERMIND_COMPILER_MODEL_COMPILE_OPTIONS_H
#define AETHERMIND_COMPILER_MODEL_COMPILE_OPTIONS_H

/// @file model_compile_options.h
/// @brief Options for semantic optimization and compiler lowering.

#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

struct ModelCompileOptions {
    PassContext optimization{};
    GraphLoweringConfig lowering{};
};

}// namespace aethermind

#endif
