#ifndef AETHERMIND_MODEL_MODEL_INSTANCE_H
#define AETHERMIND_MODEL_MODEL_INSTANCE_H

#include "aethermind/model/backend_sidecar.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

class ModelInstance {
public:
    ModelInstance() = default;
    explicit ModelInstance(BackendSidecar backend_sidecar) noexcept;
    ModelInstance(HfModelConfig config, ResolvedModelWeights resolved_weights) noexcept;

    AM_NODISCARD const BackendSidecar& GetBackendSidecar() const noexcept;
    AM_NODISCARD BackendSidecar& GetMutableBackendSidecar() noexcept;

    AM_NODISCARD const HfModelConfig& GetConfig() const noexcept;
    AM_NODISCARD const ResolvedModelWeights& GetResolvedWeights() const noexcept;

    AM_NODISCARD const PackedWeights* FindPackedWeights(
            OpType op_type,
            const KernelSelector& selector) const noexcept;

    Status StorePackedWeights(std::unique_ptr<PackedWeights> packed_weights) noexcept;

private:
    BackendSidecar backend_sidecar_{};
    HfModelConfig config_{};
    ResolvedModelWeights resolved_weights_{};
};

}// namespace aethermind

#endif
