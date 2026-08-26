#include "aethermind/model/weight_prepack_planner.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/packed_weight_store.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
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

constexpr size_t kFusedWeightAlignment = 64;

/// @brief Owns an aligned heap buffer backing a materialized fused weight.
class OwnedAlignedStorage final : public RawStorage {
public:
    explicit OwnedAlignedStorage(void* data) noexcept : data_(data) {}

    ~OwnedAlignedStorage() override {
        std::free(data_);
    }

private:
    void* data_ = nullptr;
};

/// @brief Materializes a composite weight by concatenating its components
/// along axis 0 into one owned, aligned buffer.
///
/// The composite recipes (QKV / Gate-Up) are fixed axis-0 concatenations of
/// their components in recipe order. All components must be valid,
/// contiguous, rank-2 views of one dtype with a shared feature count; a
/// sequential byte copy then yields exactly the fused row-major layout.
StatusOr<RawWeightView> MaterializeCompositeWeight(
        const std::vector<RawWeightView>& components) {
    if (components.empty()) {
        return Status::InvalidArgument(
                "composite weight requires at least one component");
    }

    const DataType& dtype = components.front().dtype;
    int64_t feature_count = -1;
    int64_t total_rows = 0;
    size_t total_bytes = 0;
    for (const auto& component: components) {
        // Defense in depth: PrepackAndStore validates before materialization,
        // but this helper is the layout authority for fused composites.
        AM_RETURN_IF_ERROR(ValidateRawWeightView(component));
        if (!component.IsValid() || !component.is_contiguous) {
            return Status::InvalidArgument(
                    "composite weight component is not a valid contiguous view");
        }
        if (component.dtype != dtype) {
            return Status::InvalidArgument(
                    "composite weight components must share one dtype");
        }
        if (component.shape.size() != 2U) {
            return Status::InvalidArgument(
                    "composite weight components must be rank 2");
        }
        if (feature_count < 0) {
            feature_count = component.shape[1];
        } else if (feature_count != component.shape[1]) {
            return Status::InvalidArgument(
                    "composite weight components must share a feature count");
        }
        const int64_t rows = component.shape[0];
        if (rows < 0 || total_rows > std::numeric_limits<int64_t>::max() - rows) {
            return Status::InvalidArgument(
                    "composite weight row count is negative or overflows");
        }
        total_rows += rows;
        if (total_bytes >
            std::numeric_limits<size_t>::max() - component.bytes) {
            return Status::InvalidArgument(
                    "composite weight byte count overflows");
        }
        total_bytes += component.bytes;
    }

    const size_t padded_bytes = total_bytes == 0 ? 1 : total_bytes;
    void* data = nullptr;
    if (posix_memalign(&data, kFusedWeightAlignment, padded_bytes) != 0 ||
        data == nullptr) {
        return Status::ResourceExhausted(
                "failed to allocate composite weight buffer");
    }
    auto* out = static_cast<std::byte*>(data);
    for (const auto& component: components) {
        std::memcpy(out, component.data, component.bytes);
        out += component.bytes;
    }

    return RawWeightView{
            .data = static_cast<const std::byte*>(data),
            .bytes = total_bytes,
            .dtype = dtype,
            .shape = std::vector<int64_t>{total_rows, feature_count},
            .storage = std::make_shared<OwnedAlignedStorage>(data),
            .is_contiguous = true,
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
        // The prepacker copies shape-derived logical_nbytes() out of these
        // views; validate byte sizes up front so a mismatch fails eagerly
        // instead of reading out of bounds or fusing a corrupted layout.
        if (req.components.empty()) {
            AM_RETURN_IF_ERROR(ValidateRawWeightView(req.raw_weight));
        } else {
            for (const auto& component: req.components) {
                AM_RETURN_IF_ERROR(ValidateRawWeightView(component));
            }
        }

        // Composite bindings carry recipe-ordered component views; the fused
        // buffer is materialized here and kept alive through pack + store.
        RawWeightView fused{};
        const RawWeightView* weight = &req.raw_weight;
        if (!req.components.empty()) {
            auto materialized = MaterializeCompositeWeight(req.components);
            if (!materialized.ok()) {
                return materialized.status();
            }
            fused = std::move(*materialized);
            weight = &fused;
        }

        const auto& shape = weight->shape;
        std::vector<int64_t> strides(shape.size());
        if (!strides.empty()) {
            strides.back() = 1;
            for (int64_t i = static_cast<int64_t>(strides.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        TensorView view(weight->data,
                        weight->dtype,
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
