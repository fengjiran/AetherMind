#ifndef AETHERMIND_MODEL_PACKED_WEIGHT_STORE_H
#define AETHERMIND_MODEL_PACKED_WEIGHT_STORE_H

/// @file packed_weight_store.h
/// @brief Ownership store for packed-weight artifacts.

#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"
#include "aethermind/graph/graph_types.h"

#include <memory>
#include <utility>
#include <vector>

namespace aethermind {

/// @brief Identity of one packed-weight artifact.
///
/// Replaces the legacy (OpType, KernelSelector) key: the WeightBinding
/// distinguishes Q/K/V/O, MLP gate/up/down, per-layer roles and lm_head, so
/// distinct weights no longer collide on the same selector; the recipe
/// distinguishes packing variants of the same logical weight.
struct WeightArtifactKey {
    /// Instance id of the LoweredModelArtifact this artifact was packed for.
    /// Zero means "unbound" (e.g. untrusted single-node requests).
    uint64_t source_id = 0;
    /// The lowered weight value (GraphValueId, artifact-local) this artifact
    /// serves. Together with source_id it uniquely identifies one weight
    /// instance across models.
    uint32_t value_index = 0;
    WeightBinding binding{};
    KernelSelector selector{};
    PackingRecipe recipe{};

    AM_NODISCARD friend bool operator==(const WeightArtifactKey& lhs,
                                        const WeightArtifactKey& rhs) = default;
};

/// @brief Owns PackedWeights artifacts indexed by their binding-aware key.
///
/// The store shares artifact ownership with ExecutionPlan: plan steps hold a
/// std::shared_ptr into these artifacts, so a plan stays executable after the
/// store itself is destroyed.
///
/// @note Not thread-safe; callers must serialize concurrent access.
class PackedWeightStore {
public:
    /// @brief Sets the source artifact this store was packed for.
    ///
    /// May be called before the first Store(); once frozen (after the first
    /// Store or a prior SetSourceId), a different source is rejected.
    ///
    /// @param source_id LoweredModelArtifact::artifact_id() value.
    /// @return Ok, or InvalidArgument if already frozen to a different source.
    Status SetSourceId(uint64_t source_id) noexcept;

    /// @brief Returns the bound source artifact id (0 = unbound).
    AM_NODISCARD uint64_t source_id() const noexcept;

    /// @brief Takes a shared reference to a packed-weights artifact.
    ///
    /// @param key Binding-aware artifact identity.
    /// @param artifact Artifact referenced by `key` thereafter.
    /// @return Ok on success, InvalidArgument if null, or AlreadyExists if an
    ///         entry with the same key is already present.
    Status Store(const WeightArtifactKey& key,
                 std::shared_ptr<const PackedWeights> artifact) noexcept;

    /// @brief Returns the stored artifact matching a key, if any.
    ///
    /// @param key Binding-aware artifact identity.
    /// @return Shared pointer to the stored artifact, or nullptr if no
    ///         matching entry exists.
    AM_NODISCARD std::shared_ptr<const PackedWeights> Find(
            const WeightArtifactKey& key) const noexcept;

    /// @brief Resolves the artifact for an execution step's weight value.
    ///
    /// Matches on {binding, selector} only: a plan step does not yet select a
    /// packing recipe. Returns nullptr when no artifact is present, or
    /// kFailedPrecondition when the pair maps to more than one recipe.
    AM_NODISCARD StatusOr<std::shared_ptr<const PackedWeights>>
    FindByBindingSelector(const WeightBinding& binding,
                          const KernelSelector& selector) const noexcept;

    AM_NODISCARD size_t size() const noexcept;
    AM_NODISCARD bool empty() const noexcept;

private:
    std::vector<std::pair<WeightArtifactKey, std::shared_ptr<const PackedWeights>>> entries_{};
    uint64_t source_id_ = 0;
    bool source_frozen_ = false;
};

} // namespace aethermind

#endif
