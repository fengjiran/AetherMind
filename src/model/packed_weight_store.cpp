#include "aethermind/model/packed_weight_store.h"

#include <limits>

namespace aethermind {
namespace {

// Expected byte payload of the logical weight an artifact claims to pack.
// Undefined dtypes or empty shapes yield 0 (no size premise).
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

}// namespace

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

StatusOr<std::shared_ptr<const PackedWeights>> PackedWeightStore::FindByBindingSelector(
        const WeightBinding& binding,
        const KernelSelector& selector) const noexcept {
    std::shared_ptr<const PackedWeights> match;
    for (const auto& [key, artifact]: entries_) {
        if (key.binding == binding && key.selector == selector) {
            if (match != nullptr) {
                return Status::FailedPrecondition(
                        "Multiple packing recipes exist for the same "
                        "binding/selector; specify a recipe before resolution");
            }
            match = artifact;
        }
    }
    return match;
}

size_t PackedWeightStore::size() const noexcept {
    return entries_.size();
}

bool PackedWeightStore::empty() const noexcept {
    return entries_.empty();
}

}// namespace aethermind
