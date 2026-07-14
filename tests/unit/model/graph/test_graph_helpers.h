#ifndef AETHERMIND_MODEL_GRAPH_TEST_GRAPH_HELPERS_H
#define AETHERMIND_MODEL_GRAPH_TEST_GRAPH_HELPERS_H

#include "aethermind/shape_inference/tensor_spec.h"

#include <cstdint>
#include <vector>

namespace aethermind {

// Builds a fully-static TensorSpec from a dtype and concrete dims.
// Shared by graph test files to avoid duplicating the helper.
inline TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

}// namespace aethermind

#endif// AETHERMIND_MODEL_GRAPH_TEST_GRAPH_HELPERS_H
