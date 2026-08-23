#ifndef AETHERMIND_MODEL_PACKED_WEIGHT_STORE_H
#define AETHERMIND_MODEL_PACKED_WEIGHT_STORE_H

/// @file packed_weight_store.h
/// @brief Ownership store for legacy packed-weight artifacts.

#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"

#include <memory>
#include <vector>

namespace aethermind {

/// @brief Owns PackedWeights artifacts indexed by their legacy op/selector key.
///
/// This store remains a compatibility facility until graph-driven weight
/// materialization provides per-binding artifact identity.
///
/// @note Not thread-safe; callers must serialize concurrent access.
class PackedWeightStore {
public:
    /// @brief Takes ownership of a packed-weights artifact and stores it.
    ///
    /// The op/selector key is read from the artifact itself
    /// (`packed_weights->op_type()` / `packed_weights->selector()`), so a
    /// duplicate of an already-stored entry is rejected.
    ///
    /// @param packed_weights Artifact to store; ownership is transferred.
    /// @return Ok on success, InvalidArgument if null, or AlreadyExists if an
    ///         entry with the same op/selector key is already present.
    /// @note Errors are reported via the returned Status, not exceptions.
    Status Store(std::unique_ptr<PackedWeights> packed_weights) noexcept;

    /// @brief Returns the stored artifact matching an op/selector key, if any.
    ///
    /// @param op_type Operator type of the artifact to find.
    /// @param selector Kernel selector of the artifact to find.
    /// @return Non-owning pointer to the stored artifact, or nullptr if no
    ///         matching entry exists.
    /// @note The returned pointer is borrowed: it stays valid only while this
    ///       store owns the artifact (until the store is destroyed).
    AM_NODISCARD const PackedWeights* Find(
            OpType op_type,
            const KernelSelector& selector) const noexcept;

    AM_NODISCARD size_t size() const noexcept {
        return packed_weights_.size();
    }

    AM_NODISCARD bool empty() const noexcept {
        return packed_weights_.empty();
    }

private:
    std::vector<std::unique_ptr<PackedWeights>> packed_weights_{};
};

}// namespace aethermind

#endif
