#include "aethermind/model/weight_prepack_planner.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/packed_weight_store.h"

#include <memory>
#include <vector>

namespace aethermind {

namespace {

KernelSelector MakePackedSelector(const Backend& backend, const DataType& weight_dtype) {
    return KernelSelector{
            .device_type = backend.device_type(),
            .act_dtype = DataType::Float32(),
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

    for (size_t layer_index = 0; layer_index < resolved_weights.layers.size();
         ++layer_index) {
        const auto& layer = resolved_weights.layers[layer_index];
        const auto add = [&](const RawWeightView& weight, TransformerWeightRole role) {
            requests.push_back(Request{
                    .op_type = OpType::kLinear,
                    .binding = MakeTransformerWeightBinding(layer_index, role),
                    .raw_weight = weight,
                    .selector = MakePackedSelector(backend, weight.dtype),
            });
        };
        add(layer.attn.q_proj, TransformerWeightRole::kAttentionQ);
        add(layer.attn.k_proj, TransformerWeightRole::kAttentionK);
        add(layer.attn.v_proj, TransformerWeightRole::kAttentionV);
        add(layer.attn.o_proj, TransformerWeightRole::kAttentionO);
        add(layer.mlp.gate_proj, TransformerWeightRole::kMlpGate);
        add(layer.mlp.up_proj, TransformerWeightRole::kMlpUp);
        add(layer.mlp.down_proj, TransformerWeightRole::kMlpDown);
    }

    if (resolved_weights.lm_head.has_value()) {
        requests.push_back(Request{
                .op_type = OpType::kLinear,
                .binding = MakeTransformerWeightBinding(
                        std::nullopt, TransformerWeightRole::kLmHead),
                .raw_weight = *resolved_weights.lm_head,
                .selector = MakePackedSelector(backend, resolved_weights.lm_head->dtype),
        });
    }

    return requests;
}

Status WeightPrepackPlanner::PrepackAndStore(PackedWeightStore& packed_weight_store,
                                             const std::vector<Request>& requests) {
    CpuWeightPrepacker prepacker;

    if (!requests.empty()) {
        AM_RETURN_IF_ERROR(
                packed_weight_store.SetSourceId(requests.front().source_id));
    }

    for (const auto& req: requests) {
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

        const WeightArtifactKey key{.source_id = req.source_id,
                                    .value_index = req.value_index,
                                    .binding = req.binding,
                                    .selector = req.selector,
                                    .recipe = CpuWeightPrepacker::RecipeFor(req.selector)};
        // A duplicate {binding, selector} is a planner bug: propagate as an
        // explicit error instead of silently skipping a weight.
        AM_RETURN_IF_ERROR(packed_weight_store.Store(
                key, std::shared_ptr<const PackedWeights>(std::move(*packed))));
    }

    return {};
}

}// namespace aethermind
