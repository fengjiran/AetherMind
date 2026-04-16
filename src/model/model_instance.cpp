#include "aethermind/model/model_instance.h"

namespace aethermind {

ModelInstance::ModelInstance(BackendSidecar backend_sidecar) noexcept
    : backend_sidecar_(std::move(backend_sidecar)) {}

const BackendSidecar& ModelInstance::GetBackendSidecar() const noexcept {
    return backend_sidecar_;
}

BackendSidecar& ModelInstance::GetMutableBackendSidecar() noexcept {
    return backend_sidecar_;
}

const PackedWeights* ModelInstance::FindPackedWeights(
        OpType op_type,
        const KernelSelector& selector) const noexcept {
    return backend_sidecar_.Find(op_type, selector);
}

Status ModelInstance::StorePackedWeights(std::unique_ptr<PackedWeights> packed_weights) noexcept {
    return backend_sidecar_.Store(std::move(packed_weights));
}

}// namespace aethermind
