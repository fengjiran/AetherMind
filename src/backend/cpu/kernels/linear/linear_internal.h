#ifndef AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H

#include "aethermind/base/tensor_view.h"

namespace aethermind {

namespace cpu {

/// Per-call kernel params for CPU Linear kernel.
/// Lifetime: stack-bound during LinearOp::Run, valid for the duration of fn(ctx).
struct CpuLinearParams {
    TensorView input_tensor{};
    TensorView weight_tensor{};
    MutableTensorView output_tensor{};
};

}// namespace cpu

}// namespace aethermind

#endif// AETHERMIND_BACKEND_CPU_KERNELS_LINEAR_LINEAR_INTERNAL_H
