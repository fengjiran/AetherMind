#ifndef AETHERMIND_MODEL_PACKED_WEIGHT_COLLECTION_H
#define AETHERMIND_MODEL_PACKED_WEIGHT_COLLECTION_H

/// @file packed_weight_collection.h
/// @brief Packed-weight artifact identity, collection, and batch construction.

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
/// The collection shares artifact ownership with ExecutionPlan: plan steps hold a
/// std::shared_ptr into these artifacts, so a plan stays executable after the
/// collection itself is destroyed.
///
/// Single-source invariant: every entry belongs to one source artifact. The
/// source is bound by SetSourceId or, failing that, by the first successful
/// Insert; afterwards a key carrying a different source_id is rejected. Zero is
/// the unbound source used by untrusted single-node requests.
///
/// Query contract: preparation-time code inserts and looks up by the complete
/// WeightArtifactKey (Insert / Find); execution planning resolves each step's
/// artifact through the same exact-key lookup. There is no recipe-agnostic
/// query surface — a step that does not know its recipe is a planner bug.
///
/// @note Not thread-safe; callers must serialize concurrent access.
class PackedWeightCollection {
public:
    /// @brief Sets the source artifact this collection was packed for.
    ///
    /// May be called before the first Insert(); once frozen (after the first
    /// Insert, which binds that key's source_id, or a prior SetSourceId), a
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
    ///        collection's bound source; if not yet bound, this call binds the
    ///        collection to `key.source_id` on success. Its recipe must identify
    ///        a known, explicit layout and match the artifact's recipe.
    /// @param artifact Artifact referenced by `key` thereafter.
    /// @return Ok on success, InvalidArgument if the artifact is null or its
    ///         key comes from a different source artifact or violates the recipe
    ///         contract, or AlreadyExists if
    ///         an entry with the same key is already present.
    Status Insert(const WeightArtifactKey& key,
                  std::shared_ptr<const PackedWeight> artifact) noexcept;

    /// @brief Returns the stored artifact matching a key, if any.
    ///
    /// @param key Binding-aware artifact identity.
    /// @return Shared pointer to the stored artifact, or nullptr if no
    ///         matching entry exists.
    AM_NODISCARD std::shared_ptr<const PackedWeight>
    Find(const WeightArtifactKey& key) const noexcept;

    AM_NODISCARD size_t size() const noexcept;
    AM_NODISCARD bool empty() const noexcept;

private:
    std::vector<std::pair<WeightArtifactKey, std::shared_ptr<const PackedWeight>>> entries_{};
    uint64_t source_id_ = 0;
    bool source_frozen_ = false;
};

/// @brief Executes prepack for every request and returns the collection owning the
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
/// The returned collection is bound to the batch's single source artifact.
/// Rejection or failure is all-or-nothing: no partially packed collection is
/// observable.
///
/// @param backend Packing service provider. Must implement PackWeights for
///        the requests' selectors.
/// @param requests Requests to execute. Every request must carry the same
///        source_id and a non-empty list of raw components; a mixed-source
///        batch is rejected before any weight is packed.
/// @return The bound collection on success, or the first validation, packing,
///         or insertion error. An empty batch returns an unbound empty collection.
StatusOr<PackedWeightCollection> PrepackWeightRequests(
        const Backend& backend,
        const std::vector<WeightPackingRequest>& requests);

} // namespace aethermind

#endif
