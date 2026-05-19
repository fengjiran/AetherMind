#include "aethermind/model/weight_prepack_planner.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/model_instance.h"
#include "macros.h"

#include <vector>

namespace aethermind {

namespace {

KernelSelector MakePackedSelector(const Backend& backend, const DataType& weight_dtype) {
    return KernelSelector{
            .device_type = backend.device_type(),
            .activation_dtype = DataType::Float32(),
            .weight_dtype = weight_dtype,
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
}

}// namespace

StatusOr<std::vector<WeightPrepackPlanner::Request>> WeightPrepackPlanner::BuildRequests(
        const HfModelConfig& config,
        const ResolvedModelWeights& resolved_weights,
        const Backend& backend,
        const KernelRegistry& registry) {
    UNUSED(config);
    UNUSED(registry);

    std::vector<Request> requests;
    const size_t num_layers = resolved_weights.layers.size();
    requests.reserve(num_layers * 7 + (resolved_weights.lm_head.has_value() ? 1 : 0));

    for (const auto& layer: resolved_weights.layers) {
        const auto add = [&](const RawWeightView& weight) {
            requests.push_back(Request{
                    .op_type = OpType::kLinear,
                    .raw_weight = weight,
                    .selector = MakePackedSelector(backend, weight.dtype),
            });
        };
        add(layer.attn.q_proj);
        add(layer.attn.k_proj);
        add(layer.attn.v_proj);
        add(layer.attn.o_proj);
        add(layer.mlp.gate_proj);
        add(layer.mlp.up_proj);
        add(layer.mlp.down_proj);
    }

    if (resolved_weights.lm_head.has_value()) {
        requests.push_back(Request{
                .op_type = OpType::kLinear,
                .raw_weight = *resolved_weights.lm_head,
                .selector = MakePackedSelector(backend, resolved_weights.lm_head->dtype),
        });
    }

    return requests;
}

Status WeightPrepackPlanner::PrepackAndStore(ModelInstance& model_instance,
                                             const std::vector<Request>& requests) {
    CpuWeightPrepacker prepacker;

    struct PackKey {
        OpType op_type;
        KernelSelector selector;
    };
    std::vector<PackKey> stored;

    for (const auto& req : requests) {
        bool duplicate = false;
        for (const auto& key : stored) {
            if (key.op_type == req.op_type && key.selector == req.selector) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        const auto& shape = req.raw_weight.shape;
        std::vector<int64_t> strides(shape.size());
        if (!strides.empty()) {
            strides.back() = 1;
            for (int64_t i = static_cast<int64_t>(strides.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        TensorView view(req.raw_weight.data,
                        req.raw_weight.dtype,
                        IntArrayView(shape),
                        IntArrayView(strides),
                        0);

        auto packed = prepacker.Pack(req.op_type, view, req.selector);
        if (!packed.ok()) {
            return packed.status();
        }

        AM_RETURN_IF_ERROR(model_instance.StorePackedWeights(std::move(*packed)));
        stored.push_back({req.op_type, req.selector});
    }

    return {};
}

}// namespace aethermind
