#ifndef AETHERMIND_MODEL_BACKEND_SIDECAR_H
#define AETHERMIND_MODEL_BACKEND_SIDECAR_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"

#include <memory>
#include <vector>

namespace aethermind {

class BackendSidecar {
public:
    Status Store(std::unique_ptr<PackedWeights> packed_weights) noexcept;

    AM_NODISCARD const PackedWeights* Find(
            OpType op_type,
            const KernelSelector& selector) const noexcept;

    AM_NODISCARD size_t size() const noexcept {
        return packed_weights_.size();
    }

    AM_NODISCARD bool empty() const noexcept {
        return packed_weights_.empty();
    }

private:
    std::vector<std::unique_ptr<PackedWeights>> packed_weights_{};
};

}// namespace aethermind

#endif
