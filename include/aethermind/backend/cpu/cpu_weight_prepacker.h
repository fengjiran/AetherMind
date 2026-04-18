#ifndef AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H
#define AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/base/tensor.h"
#include "aethermind/operators/op_type.h"

#include <memory>

namespace aethermind {

class CpuWeightPrepacker {
public:
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            const Tensor& logical_weight,
            const KernelSelector& selector) const noexcept;

    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            TensorView logical_weight,
            const KernelSelector& selector) const noexcept;
};

}// namespace aethermind

#endif
