#include "aethermind/model/backend_sidecar.h"

namespace aethermind {

Status BackendSidecar::Store(std::unique_ptr<PackedWeights> packed_weights) noexcept {
    if (packed_weights == nullptr) {
        return Status::InvalidArgument("BackendSidecar cannot store null packed weights");
    }

    if (Find(packed_weights->op_type(), packed_weights->selector()) != nullptr) {
        return Status(StatusCode::kAlreadyExists,
                      "Packed weights already exist for the requested op/selector");
    }

    packed_weights_.push_back(std::move(packed_weights));
    return Status::Ok();
}

const PackedWeights* BackendSidecar::Find(
        OpType op_type,
        const KernelSelector& selector) const noexcept {
    for (const auto& packed_weights: packed_weights_) {
        if (packed_weights->op_type() == op_type && packed_weights->selector() == selector) {
            return packed_weights.get();
        }
    }
    return nullptr;
}

}// namespace aethermind
