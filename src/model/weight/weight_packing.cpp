#include "aethermind/model/weight/weight_packing.h"

#include "aethermind/backend/backend.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/tensor_view.h"

#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace aethermind {

namespace {

/// Layer-scoped roles must carry an in-range decoder_layer_index; graph
/// validation already rejects a missing index (TransformerRoleRequiresLayer),
/// so nullptr here means the binding did not come from a validated graph.
const DecoderLayerRawWeights* LayerAt(const ResolvedModelWeights& resolved,
                                      std::optional<uint32_t> layer) noexcept {
    if (!layer.has_value() || *layer >= resolved.layers.size()) {
        return nullptr;
    }
    return &resolved.layers[*layer];
}

/// @brief Converts a validated raw weight view into a row-major TensorView.
///
/// The view borrows the raw weight's owned shape through `strides` only for
/// the duration of the packing call; nothing outlives the request.
StatusOr<TensorView> MakeRowMajorView(const RawWeightView& raw,
                                      std::vector<int64_t>& strides) {
    AM_RETURN_IF_ERROR(ValidateRawWeightView(raw));
    strides.resize(raw.shape.size());
    if (!strides.empty()) {
        strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(strides.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * raw.shape[i + 1];
        }
    }
    return TensorView(raw.data, raw.dtype, IntArrayView(raw.shape),
                      IntArrayView(strides), 0);
}

/// Expected byte payload of the logical weight an artifact claims to pack.
/// Undefined dtypes or empty shapes yield 0 (no size premise).
StatusOr<size_t> LogicalByteSize(const PackedWeights& artifact) noexcept {
    if (artifact.logical_dtype().IsUndefined() ||
        artifact.logical_dtype().nbytes() == 0) {
        return 0U;
    }
    size_t elements = 1;
    for (const int64_t dimension: artifact.logical_shape()) {
        if (dimension < 0) {
            return Status::InvalidArgument(
                    "Packed artifact logical shape contains a negative "
                    "dimension");
        }
        if (elements > std::numeric_limits<size_t>::max() /
                               static_cast<size_t>(dimension)) {
            return Status::Overflow(
                    "Packed artifact logical size overflowed size_t");
        }
        elements *= static_cast<size_t>(dimension);
    }
    if (elements > std::numeric_limits<size_t>::max() /
                           static_cast<size_t>(artifact.logical_dtype().nbytes())) {
        return Status::Overflow(
                "Packed artifact logical size overflowed size_t");
    }
    return elements * static_cast<size_t>(artifact.logical_dtype().nbytes());
}

} // namespace

const RawWeightView* ResolveWeightBinding(
        const WeightBinding& binding,
        const ResolvedModelWeights& resolved) noexcept {
    const std::optional<TransformerWeightRole> role =
            TryGetTransformerWeightRole(binding);
    if (!role.has_value()) {
        return nullptr;
    }
    const std::optional<uint32_t> layer = binding.decoder_layer_index;
    switch (*role) {
        case TransformerWeightRole::kTokenEmbedding:
            return &resolved.embed_tokens;
        case TransformerWeightRole::kFinalNorm:
            return &resolved.final_norm;
        case TransformerWeightRole::kLmHead:
            // Tied embeddings reuse embed_tokens when the checkpoint carries no
            // independent lm_head.
            return resolved.lm_head.has_value() ? &*resolved.lm_head
                                                : &resolved.embed_tokens;
        case TransformerWeightRole::kInputNorm:
            if (const auto* l = LayerAt(resolved, layer)) return &l->norm.input_rmsnorm;
            return nullptr;
        case TransformerWeightRole::kPostAttentionNorm:
            if (const auto* l = LayerAt(resolved, layer)) return &l->norm.post_attn_rmsnorm;
            return nullptr;
        case TransformerWeightRole::kAttentionQ:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.q_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionK:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.k_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionV:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.v_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionO:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.o_proj;
            return nullptr;
        case TransformerWeightRole::kMlpGate:
            if (const auto* l = LayerAt(resolved, layer)) return &l->mlp.gate_proj;
            return nullptr;
        case TransformerWeightRole::kMlpUp:
            if (const auto* l = LayerAt(resolved, layer)) return &l->mlp.up_proj;
            return nullptr;
        case TransformerWeightRole::kMlpDown:
            if (const auto* l = LayerAt(resolved, layer)) return &l->mlp.down_proj;
            return nullptr;
        case TransformerWeightRole::kMoERouter:
            // Dense checkpoints carry no router weight; MoE is out of the
            // current product scope.
            return nullptr;
    }
    return nullptr;
}

Status PrepackWeightRequests(const Backend& backend,
                             PackedWeightStore& packed_weight_store,
                             const std::vector<WeightPackingRequest>& requests) {
    if (!requests.empty()) {
        AM_RETURN_IF_ERROR(
                packed_weight_store.SetSourceId(requests.front().source_id));
    }

    for (const auto& req: requests) {
        // Validate byte sizes up front so a mismatch fails eagerly here with a
        // view-level message instead of surfacing deep inside the backend.
        std::vector<TensorView> components;
        std::vector<std::vector<int64_t>> strides_storage;
        const auto append_component = [&](const RawWeightView& raw) {
            strides_storage.emplace_back();
            auto view = MakeRowMajorView(raw, strides_storage.back());
            if (!view.ok()) {
                return view.status();
            }
            components.push_back(*view);
            return Status::Ok();
        };

        if (req.components.empty()) {
            AM_RETURN_IF_ERROR(append_component(req.raw_weight));
        } else {
            for (const RawWeightView& component: req.components) {
                AM_RETURN_IF_ERROR(append_component(component));
            }
        }

        auto packed = backend.PackWeights(req.op_type, components, req.selector);
        if (!packed.ok()) {
            return packed.status();
        }

        // Read the recipe back from the produced artifact instead of deriving
        // it again: the store re-verifies key/artifact consistency, so pack and
        // consume can never drift apart.
        const WeightArtifactKey key{
                .source_id = req.source_id,
                .value_index = req.value_index,
                .binding = req.binding,
                .selector = req.selector,
                .recipe = (*packed)->recipe()};
        // A duplicate {binding, selector} is a planner bug: propagate as an
        // explicit error instead of silently skipping a weight.
        AM_RETURN_IF_ERROR(packed_weight_store.Store(
                key, std::shared_ptr<const PackedWeights>(std::move(*packed))));
    }

    return {};
}

Status PackedWeightStore::SetSourceId(uint64_t source_id) noexcept {
    if (source_frozen_ && source_id != source_id_) {
        return Status::InvalidArgument(
                "PackedWeightStore is already frozen to a different source "
                "artifact");
    }
    source_id_ = source_id;
    source_frozen_ = true;
    return Status::Ok();
}

uint64_t PackedWeightStore::source_id() const noexcept {
    return source_id_;
}

Status PackedWeightStore::Store(const WeightArtifactKey& key,
                                std::shared_ptr<const PackedWeights> artifact) noexcept {
    if (artifact == nullptr) {
        return Status::InvalidArgument(
                "PackedWeightStore cannot store null packed weights");
    }

    if (Find(key) != nullptr) {
        return Status::AlreadyExists(
                "Packed weights already exist for the requested weight key");
    }

    // The store is the trust boundary where a caller may pair an arbitrary
    // artifact with a key. Reject any drift so execution never consumes a
    // mismatched payload, regardless of which recipe/selector the plan asked
    // for.
    if (key.selector != artifact->selector()) {
        return Status::InvalidArgument(
                "Packed weight key selector does not match the artifact "
                "selector");
    }
    if (key.recipe != artifact->recipe()) {
        return Status::InvalidArgument(
                "Packed weight key recipe does not match the artifact recipe");
    }
    if (artifact->storage().alignment() < key.recipe.alignment) {
        return Status::InvalidArgument(
                "Packed artifact storage alignment is below its recipe "
                "alignment");
    }
    auto expected_bytes = LogicalByteSize(*artifact);
    if (!expected_bytes.ok()) {
        return expected_bytes.status();
    }
    if (artifact->storage().nbytes() < *expected_bytes) {
        return Status::InvalidArgument(
                "Packed artifact storage is smaller than its logical weight");
    }

    if (!source_frozen_) {
        source_frozen_ = true;
    }

    entries_.emplace_back(key, std::move(artifact));
    return Status::Ok();
}

std::shared_ptr<const PackedWeights> PackedWeightStore::Find(
        const WeightArtifactKey& key) const noexcept {
    for (const auto& [entry_key, artifact]: entries_) {
        if (entry_key == key) {
            return artifact;
        }
    }
    return nullptr;
}

size_t PackedWeightStore::size() const noexcept {
    return entries_.size();
}

bool PackedWeightStore::empty() const noexcept {
    return entries_.empty();
}

} // namespace aethermind