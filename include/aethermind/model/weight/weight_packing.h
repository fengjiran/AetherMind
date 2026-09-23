#ifndef AETHERMIND_MODEL_WEIGHT_PACKING_H
#define AETHERMIND_MODEL_WEIGHT_PACKING_H

/// @file weight_packing.h
/// @brief Weight assembly: binding resolution, packing requests/execution, and
///        packed-artifact storage.

#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/operators/op_type.h"

#include <memory>
#include <utility>
#include <vector>

namespace aethermind {

class Backend;

/// @brief Resolves a direct weight binding to the raw weight it references.
///
/// Single authority for the Transformer role → ResolvedModelWeights mapping,
/// including the tied lm-head fallback: a checkpoint without an independent
/// lm_head reuses embed_tokens, so no caller special-cases tying. Resolution is
/// purely structural — it never consults debug names or graph topology.
///
/// @param binding Logical weight binding to resolve. Composite bindings
///        (QkvWeightBinding / GateUpWeightBinding) address several weights and
///        have no single resolution; resolve each component role separately.
/// @param resolved Resolved weights backing the model.
/// @return Pointer borrowed from `resolved`, valid while `resolved` lives, or
///         nullptr when the binding has no single raw weight: a composite or
///         role-less binding, kMoERouter (absent from dense checkpoints), or a
///         layer-scoped role whose decoder_layer_index is missing or out of
///         range. Callers must treat nullptr as a hard error — never as "skip
///         this weight" — because a silently unbound weight only surfaces as
///         wrong numerics at execution time.
AM_NODISCARD const RawWeightView* ResolveWeightBinding(
        const WeightBinding& binding,
        const ResolvedModelWeights& resolved) noexcept;

class PackedWeightStore;

/// @brief One weight value to prepack, keyed by its position in the artifact.
struct WeightPackingRequest {
    OpType op_type{};
    /// Source artifact id from the producing LoweredGraph.
    uint64_t source_id = 0;
    /// Artifact-local weight value id (GraphValueId) this request packs.
    uint32_t value_index = 0;
    /// Logical weight binding (layer index + role) used as artifact
    /// identity together with the selector.
    WeightBinding binding{};
    /// Raw weight view for direct bindings. Composite bindings
    /// (QkvWeightBinding / GateUpWeightBinding) leave this empty and carry
    /// the recipe-ordered components instead; prepacking materializes the
    /// fused view from `components`.
    RawWeightView raw_weight;
    /// Recipe-ordered raw components of a composite binding: Q, K, V for
    /// QkvWeightBinding; Gate, Up for GateUpWeightBinding. Empty for
    /// direct bindings, whose single view lives in `raw_weight`.
    std::vector<RawWeightView> components{};
    KernelSelector selector;
};

/// @brief Executes prepack for every request and stores the resulting
/// PackedWeights artifacts into a PackedWeightStore. Packing identity
/// remains {source_id, value_index, binding, selector, recipe}; the recipe is
/// read back from each produced artifact, so pack and consume can never
/// drift apart.
///
/// Production requests come from the graph-driven BuildWeightPackingRequests
/// (compiler); this function only executes them. Execution goes through the
/// Backend::PackWeights contract — this module never touches a concrete
/// backend implementation.
///
/// @param backend Packing service provider. Must implement PackWeights for
///        the requests' selectors.
/// @param packed_weight_store Store receiving the produced artifacts.
/// @param requests Requests to execute. All requests must share one source_id.
/// @return Ok on success, or the first validation/packing/store error.
Status PrepackWeightRequests(const Backend& backend,
                             PackedWeightStore& packed_weight_store,
                             const std::vector<WeightPackingRequest>& requests);

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

/// @brief Owns PackedWeights artifacts indexed by their binding-aware key.
///
/// The store shares artifact ownership with ExecutionPlan: plan steps hold a
/// std::shared_ptr into these artifacts, so a plan stays executable after the
/// store itself is destroyed.
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
    /// Store or a prior SetSourceId), a different source is rejected.
    ///
    /// @param source_id LoweredGraph::artifact_id() value.
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

    AM_NODISCARD size_t size() const noexcept;
    AM_NODISCARD bool empty() const noexcept;

private:
    std::vector<std::pair<WeightArtifactKey, std::shared_ptr<const PackedWeights>>> entries_{};
    uint64_t source_id_ = 0;
    bool source_frozen_ = false;
};

} // namespace aethermind

#endif
