#include "aethermind/model/weight/packed_weight_collection.h"
#include "aethermind/backend/backend.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/weight/weight_packing_request.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

// The view borrows the raw shape and caller-provided stride storage only for
// the duration of the packing call; nothing outlives the request.
StatusOr<TensorView> MakeRowMajorView(const RawWeightView& raw,
                                      std::vector<int64_t>& strides) {
    AM_RETURN_IF_ERROR(ValidateRawWeightView(raw));
    if (!raw.is_contiguous) {
        return Status::InvalidArgument("Raw weight components must be contiguous row-major views");
    }
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
StatusOr<size_t> LogicalByteSize(const PackedWeight& artifact) {
    const DataType dtype = artifact.logical_dtype();
    if (dtype.IsUndefined()) {
        return 0U;
    }
    // nbytes() requires a fixed lane encoding. Checking the raw range also
    // rejects reserved -1, which IsScalableVector() does not classify.
    if (dtype.raw_lanes() > std::numeric_limits<int16_t>::max()) {
        return Status::InvalidArgument(
                "Packed artifact logical dtype requires a fixed lane encoding");
    }
    const int element_bytes = dtype.nbytes();
    if (element_bytes == 0) {
        return 0U;
    }

    bool has_zero_dimension = false;
    for (const int64_t dimension: artifact.logical_shape()) {
        if (dimension < 0) {
            return Status::InvalidArgument(
                    "Packed artifact logical shape contains a negative "
                    "dimension");
        }
        has_zero_dimension |= dimension == 0;
    }
    if (has_zero_dimension) {
        return 0U;
    }

    size_t elements = 1;
    for (const int64_t dimension: artifact.logical_shape()) {
        if (!std::in_range<size_t>(dimension) ||
            CheckOverflowMul(elements, static_cast<size_t>(dimension), &elements)) {
            return Status::Overflow(
                    "Packed artifact logical size overflowed size_t");
        }
    }

    size_t bytes = 0;
    if (CheckOverflowMul(elements,
                         static_cast<size_t>(element_bytes), &bytes)) {
        return Status::Overflow(
                "Packed artifact logical size overflowed size_t");
    }
    return bytes;
}

} // namespace

Status PackedWeightCollection::SetSourceId(uint64_t source_id) noexcept try {
    if (source_frozen_ && source_id != source_id_) {
        return Status::InvalidArgument("PackedWeightCollection is already frozen to "
                                       "a different source artifact");
    }

    source_id_ = source_id;
    source_frozen_ = true;
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

uint64_t PackedWeightCollection::source_id() const noexcept {
    return source_id_;
}

Status PackedWeightCollection::Insert(const WeightArtifactKey& key,
                                      std::shared_ptr<const PackedWeight> artifact) noexcept try {
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

    entries_.emplace_back(key, std::move(artifact));
    if (!source_frozen_) {
        // Binding on the first successful Insert keeps source_id_ consistent
        // with the entries, so a collection populated without an explicit
        // SetSourceId still rejects later keys from another artifact.
        source_id_ = key.source_id;
        source_frozen_ = true;
    }

    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
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

StatusOr<size_t> PackedWeightCollection::total_storage_bytes() const noexcept try {
    std::vector<const BufferImpl*> owners;
    owners.reserve(entries_.size());
    for (const auto& entry: entries_) {
        if (const auto* impl = entry.second->storage().impl(); impl != nullptr) {
            owners.push_back(impl);
        }
    }
    std::sort(owners.begin(), owners.end(), std::less<const BufferImpl*>{});
    owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
    size_t total = 0;
    for (const auto* owner: owners) {
        if (CheckOverflowAdd(total, owner->nbytes(), &total)) {
            return Status::Overflow("Packed storage payload overflowed size_t");
        }
    }
    return total;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

bool PackedWeightCollection::storage_plan_complete() const noexcept {
    return storage_plan_complete_;
}

StatusOr<PackedWeightCollection> PrepackWeightRequests(
        const Backend& backend,
        const std::vector<WeightPackingRequest>& requests,
        Allocator& allocator) {
    return PrepackWeightRequests(backend, requests, allocator, std::nullopt);
}

StatusOr<PackedWeightCollection>
PrepackWeightRequests(const Backend& backend,
                      const std::vector<WeightPackingRequest>& requests,
                      Allocator& allocator,
                      std::optional<size_t> max_payload_bytes) try {
    PackedWeightCollection collection;
    const uint64_t source_id = requests.empty() ? 0U : requests.front().source_id;
    for (const auto& req: requests) {
        if (req.source_id != source_id) {
            return Status::InvalidArgument(
                    "PrepackWeightRequests requires every request to share one source artifact");
        }
    }
    if (!requests.empty()) {
        AM_RETURN_IF_ERROR(collection.SetSourceId(source_id));
    }

    struct PlannedRequest {
        std::vector<std::vector<int64_t>> strides;
        std::vector<TensorView> components;
        std::optional<AllocationRequest> storage;
    };
    std::vector<PlannedRequest> planned;
    planned.reserve(requests.size());
    size_t total = 0;
    for (const auto& req: requests) {
        if (!IsValidPackingLayout(req.recipe.layout) || req.recipe.alignment == 0) {
            return Status::InvalidArgument("PrepackWeightRequests requires an explicit packing recipe");
        }
        if (req.components.empty()) {
            return Status::InvalidArgument("PrepackWeightRequests requires non-empty weight components");
        }
        PlannedRequest item;
        item.strides.reserve(req.components.size());
        item.components.reserve(req.components.size());
        for (const auto& component: req.components) {
            item.strides.emplace_back();
            AM_ASSIGN_OR_RETURN(auto view, MakeRowMajorView(component, item.strides.back()));
            item.components.push_back(view);
        }
        auto requirement = backend.GetWeightPackingRequirements(
                req.op_type, item.components, req.selector, req.recipe);
        if (!requirement.ok()) {
            if (requirement.status().code() != StatusCode::kUnimplemented || max_payload_bytes.has_value()) {
                return requirement.status();
            }
            collection.storage_plan_complete_ = false;
        } else {
            item.storage = *requirement;
            if (requirement->alignment == 0 ||
                !std::has_single_bit(requirement->alignment) ||
                requirement->alignment < req.recipe.alignment) {
                return Status::Internal("Packing query returned an invalid storage alignment");
            }
            if (CheckOverflowAdd(total, requirement->nbytes, &total)) {
                return Status::Overflow("Packing batch payload overflowed size_t");
            }
        }
        planned.push_back(std::move(item));
    }
    if (max_payload_bytes.has_value() && total > *max_payload_bytes) {
        return Status::ResourceExhausted("Packed weight payload exceeds limit: required=" +
                                         std::to_string(total) + ", limit=" + std::to_string(*max_payload_bytes));
    }

    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];
        auto packed = backend.PackWeights(req.op_type, planned[i].components,
                                          req.selector, req.recipe, allocator);
        if (!packed.ok()) {
            return packed.status();
        }
        if (*packed == nullptr) {
            return Status::Internal("Backend weight packing returned a null artifact");
        }
        if ((*packed)->recipe() != req.recipe) {
            return Status::InvalidArgument("Packed artifact recipe differs from its request");
        }
        if (planned[i].storage.has_value() &&
            ((*packed)->storage().nbytes() != planned[i].storage->nbytes ||
             (*packed)->storage().alignment() < planned[i].storage->alignment)) {
            return Status::Internal("Packed artifact storage differs from its planned requirements");
        }
        const WeightArtifactKey key{.source_id = req.source_id,
                                    .value_index = req.value_index,
                                    .binding = req.binding,
                                    .selector = req.selector,
                                    .recipe = req.recipe};
        AM_RETURN_IF_ERROR(collection.Insert(
                key, std::shared_ptr<const PackedWeight>(std::move(*packed))));
    }
    return collection;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
