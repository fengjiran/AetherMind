#include "aethermind/model/model_instance.h"

namespace aethermind {

ModelInstance::ModelInstance(BackendSidecar backend_sidecar) noexcept
    : backend_sidecar_(std::move(backend_sidecar)) {}

ModelInstance::ModelInstance(HfModelConfig config, ModelWeightIndex raw_weight_index) noexcept
    : config_(std::move(config)), raw_weight_index_(std::move(raw_weight_index)) {}

const BackendSidecar& ModelInstance::GetBackendSidecar() const noexcept {
    return backend_sidecar_;
}

const HfModelConfig& ModelInstance::GetConfig() const noexcept {
    return config_;
}

const ModelWeightIndex& ModelInstance::GetRawWeightIndex() const noexcept {
    return raw_weight_index_;
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
