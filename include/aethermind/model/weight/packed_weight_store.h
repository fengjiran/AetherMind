#ifndef AETHERMIND_MODEL_PACKED_WEIGHT_STORE_H
#define AETHERMIND_MODEL_PACKED_WEIGHT_STORE_H

/// @file packed_weight_store.h
/// @brief Packed-weight artifact identity, shared storage, and batch construction.

#include "aethermind/backend/packed_weight.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/graph/graph_types.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace aethermind {

class Backend;
struct WeightPackingRequest;

/// @brief Identity of one packed-weight artifact.
///
/// Replaces the legacy (OpType, KernelSelector) key: the WeightBinding
/// distinguishes Q/K/V/O, MLP gate/up/down, per-layer roles and lm_head, so
/// distinct weights no longer collide on the same selector; the recipe
/// distinguishes packing variants of the same logical weight.
struct WeightArtifactKey {
    /// Instance id of the LoweredGraph this artifact was packed for.
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

/// @brief Owns PackedWeight artifacts indexed by their binding-aware key.
///
/// The store shares artifact ownership with ExecutionPlan: plan steps hold a
/// std::shared_ptr into these artifacts, so a plan stays executable after the
/// store itself is destroyed.
///
/// Single-source invariant: every entry belongs to one source artifact. The
/// source is bound by SetSourceId or, failing that, by the first successful
/// Store; afterwards a key carrying a different source_id is rejected. Zero is
/// the unbound source used by untrusted single-node requests.
///
/// Query contract: preparation-time code stores and looks up by the complete
/// WeightArtifactKey (Store / Find); execution planning resolves each step's
/// artifact through the same exact-key lookup. There is no recipe-agnostic
/// query surface — a step that does not know its recipe is a planner bug.
///
/// @note Not thread-safe; callers must serialize concurrent access.
class PackedWeightStore {
public:
    /// @brief Sets the source artifact this store was packed for.
    ///
    /// May be called before the first Store(); once frozen (after the first
    /// Store, which binds that key's source_id, or a prior SetSourceId), a
    /// different source is rejected.
    ///
    /// @param source_id LoweredGraph::artifact_id() value.
    /// @return Ok, or InvalidArgument if already frozen to a different source.
    Status SetSourceId(uint64_t source_id) noexcept;

    /// @brief Returns the bound source artifact id (0 = unbound).
    AM_NODISCARD uint64_t source_id() const noexcept;

    /// @brief Takes a shared reference to a packed-weights artifact.
    ///
    /// @param key Binding-aware artifact identity. Its source_id must match the
    ///        store's bound source; when the store is not bound yet, this call
    ///        binds it to `key.source_id`.
    /// @param artifact Artifact referenced by `key` thereafter.
    /// @return Ok on success, InvalidArgument if the artifact is null or its
    ///         key comes from a different source artifact, or AlreadyExists if
    ///         an entry with the same key is already present.
    Status Store(const WeightArtifactKey& key,
                 std::shared_ptr<const PackedWeight> artifact) noexcept;

    /// @brief Returns the stored artifact matching a key, if any.
    ///
    /// @param key Binding-aware artifact identity.
    /// @return Shared pointer to the stored artifact, or nullptr if no
    ///         matching entry exists.
    AM_NODISCARD std::shared_ptr<const PackedWeight> Find(
            const WeightArtifactKey& key) const noexcept;

    AM_NODISCARD size_t size() const noexcept;
    AM_NODISCARD bool empty() const noexcept;

private:
    std::vector<std::pair<WeightArtifactKey, std::shared_ptr<const PackedWeight>>> entries_{};
    uint64_t source_id_ = 0;
    bool source_frozen_ = false;
};

/// @brief Executes prepack for every request and returns the store owning the
/// resulting PackedWeight artifacts. Packing identity remains
/// {source_id, value_index, binding, selector, recipe}; the selected recipe is
/// passed explicitly to the backend and checked against its output.
///
/// Prepack denotes packing during model preparation, before execution. The
/// explicit recipe selects the layout, shared with Backend::PackWeights.
///
/// Production requests come from the graph-driven BuildWeightPackingRequests
/// (compiler); this function only executes them. Execution goes through the
/// Backend::PackWeights contract — this module never touches a concrete
/// backend implementation.
///
/// The returned store is bound to the batch's single source artifact.
/// Rejection or failure is all-or-nothing: no partially packed store is
/// observable.
///
/// @param backend Packing service provider. Must implement PackWeights for
///        the requests' selectors.
/// @param requests Requests to execute. Every request must carry the same
///        source_id; a mixed batch is rejected before any weight is packed.
/// @return The bound store on success, or the first validation/packing/store
///         error. An empty batch returns an unbound empty store.
StatusOr<PackedWeightStore> PrepackWeightRequests(
        const Backend& backend,
        const std::vector<WeightPackingRequest>& requests);

} // namespace aethermind

#endif
