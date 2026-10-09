#include "aethermind/model/weight/packed_weight_collection.h"
#include "aethermind/backend/backend.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/weight/weight_packing_request.h"
#include "utils/overflow_check.h"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

// The view borrows the raw shape and caller-provided stride storage only for
// the duration of the packing call; nothing outlives the request.
StatusOr<TensorView> MakeRowMajorView(const RawWeightView& raw,
                                      std::vector<int64_t>& strides) {
    AM_RETURN_IF_ERROR(ValidateRawWeightView(raw));
    strides.resize(raw.shape.size());
    if (!strides.empty()) {
        strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(strides.size()) - 2; i >= 0; --i) {
            // A zero element count does not bound the suffix products used as strides.
            if (CheckOverflowMul(strides[i + 1], raw.shape[i + 1], &strides[i])) {
                return Status::Overflow("Row-major weight stride overflows int64_t");
            }
        }
    }
    return TensorView(raw.data, raw.dtype, IntArrayView(raw.shape),
                      IntArrayView(strides), 0);
}

/// Expected byte payload of the logical weight an artifact claims to pack.
/// Undefined or zero-byte dtypes yield 0 (no size premise).
StatusOr<size_t> LogicalByteSize(const PackedWeight& artifact) noexcept {
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

        if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() ||
            CheckOverflowMul(elements, static_cast<size_t>(dimension), &elements)) {
            return Status::Overflow(
                    "Packed artifact logical size overflowed size_t");
        }
    }

    size_t bytes = 0;
    if (CheckOverflowMul(elements,
                         static_cast<size_t>(artifact.logical_dtype().nbytes()), &bytes)) {
        return Status::Overflow(
                "Packed artifact logical size overflowed size_t");
    }
    return bytes;
}

} // namespace

Status PackedWeightCollection::SetSourceId(uint64_t source_id) noexcept {
    if (source_frozen_ && source_id != source_id_) {
        return Status::InvalidArgument("PackedWeightCollection is already frozen to "
                                       "a different source artifact");
    }

    source_id_ = source_id;
    source_frozen_ = true;
    return Status::Ok();
}

uint64_t PackedWeightCollection::source_id() const noexcept {
    return source_id_;
}

Status PackedWeightCollection::Insert(const WeightArtifactKey& key,
                                      std::shared_ptr<const PackedWeight> artifact) noexcept {
    if (artifact == nullptr) {
        return Status::InvalidArgument(
                "PackedWeightCollection cannot insert null packed weights");
    }

    if (source_frozen_ && key.source_id != source_id_) {
        return Status::InvalidArgument(
                "Packed weight key belongs to a different source artifact than "
                "the collection");
    }

    if (Find(key) != nullptr) {
        return Status::AlreadyExists(
                "Packed weights already exist for the requested weight key");
    }

    // The collection is the trust boundary where a caller may pair an arbitrary
    // artifact with a key. Reject any drift so execution never consumes a
    // mismatched payload, regardless of which recipe/selector the plan asked
    // for.
    if (key.selector != artifact->selector()) {
        return Status::InvalidArgument(
                "Packed weight key selector does not match the artifact "
                "selector");
    }

    if (!IsValidPackingLayout(key.recipe.layout)) {
        return Status::InvalidArgument(
                "Packed weight key requires a known, explicit packing layout");
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
        // Binding on the first successful Insert keeps source_id_ consistent
        // with the entries, so a collection populated without an explicit
        // SetSourceId still rejects later keys from another artifact.
        source_id_ = key.source_id;
        source_frozen_ = true;
    }

    entries_.emplace_back(key, std::move(artifact));
    return Status::Ok();
}

std::shared_ptr<const PackedWeight>
PackedWeightCollection::Find(const WeightArtifactKey& key) const noexcept {
    for (const auto& [entry_key, artifact]: entries_) {
        if (entry_key == key) {
            return artifact;
        }
    }
    return nullptr;
}

size_t PackedWeightCollection::size() const noexcept {
    return entries_.size();
}

bool PackedWeightCollection::empty() const noexcept {
    return entries_.empty();
}

StatusOr<PackedWeightCollection>
PrepackWeightRequests(const Backend& backend,
                      const std::vector<WeightPackingRequest>& requests) {
    PackedWeightCollection packed_weight_collection;
    const uint64_t source_id = requests.empty() ? 0U : requests.front().source_id;
    // Validate the whole batch before packing anything: a mixed batch would
    // insert one artifact's weights under another's identity. The collection is
    // returned only on success, so a failed batch is never observable.
    for (const auto& req: requests) {
        if (req.source_id != source_id) {
            return Status::InvalidArgument(
                    "PrepackWeightRequests requires every request to share one "
                    "source artifact");
        }
    }

    if (!requests.empty()) {
        AM_RETURN_IF_ERROR(packed_weight_collection.SetSourceId(source_id));
    }

    for (const auto& req: requests) {
        if (!IsValidPackingLayout(req.recipe.layout) || req.recipe.alignment == 0) {
            return Status::InvalidArgument(
                    "PrepackWeightRequests requires an explicit packing recipe");
        }

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

        auto packed = backend.PackWeights(
                req.op_type, components, req.selector, req.recipe);
        if (!packed.ok()) {
            return packed.status();
        }

        if ((*packed)->recipe() != req.recipe) {
            return Status::InvalidArgument(
                    "Packed artifact recipe differs from its request");
        }

        const WeightArtifactKey key{
                .source_id = req.source_id,
                .value_index = req.value_index,
                .binding = req.binding,
                .selector = req.selector,
                .recipe = req.recipe};
        // A duplicate {binding, selector} is a planner bug: propagate as an
        // explicit error instead of silently skipping a weight.
        AM_RETURN_IF_ERROR(packed_weight_collection.Insert(
                key, std::shared_ptr<const PackedWeight>(std::move(*packed))));
    }

    return packed_weight_collection;
}

} // namespace aethermind
