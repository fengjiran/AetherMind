#ifndef AETHERMIND_MODEL_GRAPH_OPTIMIZATION_GRAPH_PASS_MANAGER_H
#define AETHERMIND_MODEL_GRAPH_OPTIMIZATION_GRAPH_PASS_MANAGER_H

#include "aethermind/model/graph/optimization/const_evaluator.h"
#include "aethermind/model/graph/optimization/graph_rewrite.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace aethermind {

/// Configuration that controls which optimization passes run and how.
///
/// Pass registration is driven solely by `opt_level` (see the default
/// pipeline table in model_graph_design_v2.md §16.4). The `enable_*`
/// flags do NOT affect registration; each pass inspects its flag inside
/// `Run()` and returns early when disabled. This separation keeps the
/// pipeline observable and predictable: disabling a flag leaves the pass
/// in the pipeline but makes it a no-op at runtime.
///
/// `checkpoint_every` materializes an immutable snapshot of the graph every
/// N passes (0 disables checkpointing). Snapshots bound the blast radius of
/// a failed pass and match the immutable-snapshot + phase-checkpoint contract
/// in model_graph_design_v2.md §10.
struct PassContext {
    /// Optimization level. 0 = no passes, 1 = ConstantFolding→DCE,
    /// 2+ (default) = ConstantFolding→SiluMulFusion→DCE.
    uint32_t opt_level = 2;
    /// Materialize a graph snapshot every N passes (0 = never).
    uint32_t checkpoint_every = 0;

    bool enable_qkv_fusion = true;
    bool enable_swiglu_fusion = true;
    bool enable_dce = true;
    bool enable_constant_folding = true;
    bool enable_flash_attention_rewrite = true;
    bool enable_fused_add_rms_norm = true;
    ConstEvalPolicy const_eval_policy{};
};

/// A single optimization step over a graph rewrite session.
///
/// Implementations are stateless and owned by GraphPassManager. Each pass
/// reads and mutates the graph exclusively through the supplied
/// GraphRewriteSession virtual view, never directly via ModelGraph.
class GraphPass {
public:
    virtual ~GraphPass() = default;

    /// Human-readable pass name, used for logging and debugging.
    AM_NODISCARD virtual std::string_view Name() const noexcept = 0;

    /// Applies the optimization to `session`.
    ///
    /// @param session Mutable rewrite session bound to the current graph
    ///                snapshot. Passes must use this view, not the raw graph.
    /// @param ctx     Pass configuration; the pass should honor its own flag.
    /// @return Status::Ok() on success, or the first error encountered.
    ///         Returning Status::Ok() without mutating is valid (skip/no-op).
    AM_NODISCARD virtual Status Run(GraphRewriteSession& session, const PassContext& ctx) = 0;
};

/// Runs an ordered sequence of optimization passes over a ModelGraph.
///
/// The manager is immutable after construction: Add/AddSequential only
/// return a reference to *this for fluent building and never alter a
/// manager that has already been used. Run() never modifies the caller's
/// graph; it returns a new ModelGraph (or a checkpointed snapshot) that
/// owns all folded constants and rewritten nodes.
///
/// Thread-safety: a constructed manager is safe to call Run() on from
/// multiple threads concurrently, provided each call receives a distinct
/// source graph. The manager holds no mutable state across Run() calls.
class GraphPassManager {
public:
    GraphPassManager() = default;
    explicit GraphPassManager(PassContext ctx) noexcept;

    /// Appends a pass to the pipeline. Takes ownership of `pass`.
    /// @return *this for fluent chaining.
    GraphPassManager& Add(std::unique_ptr<GraphPass> pass);

    /// Appends passes in order. Equivalent to repeated Add() calls.
    /// @return *this for fluent chaining.
    GraphPassManager& AddSequential(std::vector<std::unique_ptr<GraphPass>> passes);

    /// Overrides `ctx_.checkpoint_every` for the next Run() call.
    /// @return *this for fluent chaining.
    GraphPassManager& SetCheckpointEvery(uint32_t pass_count) noexcept;

    /// Executes the pipeline over `graph` and returns the optimized graph.
    ///
    /// @param graph Source graph; must outlive the call. Never modified.
    /// @return A new ModelGraph containing the optimization result, or the
    ///         first error from any pass or checkpoint commit.
    /// @note If `checkpoint_every != 0`, intermediate snapshots are committed
    ///       and the final result is the last snapshot (or a commit of any
    ///       trailing non-checkpoint passes).
    AM_NODISCARD StatusOr<ModelGraph> Run(const ModelGraph& graph) const;

private:
    std::vector<std::unique_ptr<GraphPass>> passes_{};
    PassContext ctx_{};
};

}// namespace aethermind

#endif
