#ifndef AETHERMIND_BACKEND_PACKED_WEIGHTS_H
#define AETHERMIND_BACKEND_PACKED_WEIGHTS_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

namespace aethermind {

// Packed weight artifacts are owned by ModelInstance backend sidecars.
// Backend/prepacker code defines the format and build path but does not own
// the packed payload lifetime.
class PackedWeights {
public:
    virtual ~PackedWeights() = default;

    AM_NODISCARD virtual OpType op_type() const noexcept = 0;
    AM_NODISCARD virtual const KernelSelector& selector() const noexcept = 0;
    AM_NODISCARD virtual const Buffer& storage() const noexcept = 0;
};

}// namespace aethermind

#endif
