#include "aethermind/model/packed_weight_store.h"

namespace aethermind {

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

    if (key.recipe != artifact->recipe()) {
        return Status::InvalidArgument(
                "Packed weight key recipe does not match the artifact recipe");
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
