#ifndef AETHERMIND_SHAPE_INFERENCE_TENSOR_SPEC_H
#define AETHERMIND_SHAPE_INFERENCE_TENSOR_SPEC_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/shape_inference/shape_symbol.h"

namespace aethermind {

struct TensorSpec {
    DataType dtype{};
    SymbolicShape shape{};
};

}// namespace aethermind

#endif
