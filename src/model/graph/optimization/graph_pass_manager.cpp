// GraphPassManager executes an ordered pass pipeline over a ModelGraph.
//
// The pipeline is checkpoint-aware: when checkpoint_every != 0, the session
// is committed to an immutable ModelGraph snapshot every N passes. Each new
// session borrows from the latest snapshot, so a later pass failure cannot
// corrupt earlier results and the final return always owns a complete graph.
// See model_graph_design_v2.md §10 and §16.4 for the contract and default
// pipeline selection.

#include "aethermind/model/graph/optimization/graph_pass_manager.h"

#include <optional>
#include <utility>

namespace aethermind {

GraphPassManager::GraphPassManager(PassContext ctx) noexcept
    : ctx_(ctx) {}

GraphPassManager& GraphPassManager::Add(std::unique_ptr<GraphPass> pass) {
    passes_.push_back(std::move(pass));
    return *this;
}

GraphPassManager& GraphPassManager::AddSequential(std::vector<std::unique_ptr<GraphPass>> passes) {
    for (std::unique_ptr<GraphPass>& pass: passes) {
        Add(std::move(pass));
    }
    return *this;
}

GraphPassManager& GraphPassManager::SetCheckpointEvery(uint32_t pass_count) noexcept {
    ctx_.checkpoint_every = pass_count;
    return *this;
}

StatusOr<ModelGraph> GraphPassManager::Run(const ModelGraph& graph) const {
    // Precondition: the source graph must be semantically valid before any
    // pass observes it. ValidateAndTopologicalOrder is the single semantic
    // authority on ModelGraph (it invokes InferOperator per node); running
    // passes on an unvalidated graph would let stale/forged metadata
    // propagate into checkpoints. We discard the returned order here because
    // passes navigate the graph via the session API; the validation is the
    // side effect we want, not the topological order. Errors propagate
    // verbatim so callers see node/op semantic context (matches Task 5
    // acceptance: "Compile failure 保留 node/op semantic context").
    AM_RETURN_IF_ERROR(graph.Validate());

    // Reference the caller's graph directly to avoid an initial deep copy.
    // GraphRewriteSession only holds a const reference, so this is safe as
    // long as `graph` outlives the call. When a checkpoint fires, ownership
    // of the materialized snapshot transfers to `checkpointed`, which keeps
    // the next session's reference alive.
    auto session = std::make_unique<GraphRewriteSession>(graph);
    std::optional<ModelGraph> checkpointed;

    // Track whether the most recent pass already materialized a snapshot, so
    // the trailing return can skip a redundant Commit on an unchanged session.
    bool last_was_checkpoint = false;
    for (size_t i = 0; i < passes_.size(); ++i) {
        if (passes_[i] == nullptr) {
            return Status::InvalidArgument("GraphPassManager: pass cannot be null");
        }
        AM_RETURN_IF_ERROR(passes_[i]->Run(*session, ctx_));

        // Materialize at phase checkpoints per SetCheckpointEvery(N).
        // Includes the last pass when it lands on a checkpoint boundary,
        // matching the immutable-snapshot + phase-checkpoint contract in
        // model_graph_design_v2.md §10. The trailing return below handles
        // any accumulated mutations from non-checkpoint passes.
        if (ctx_.checkpoint_every != 0 && ((i + 1) % ctx_.checkpoint_every == 0)) {
            StatusOr<ModelGraph> checkpoint = session->Commit();
            AM_RETURN_IF_ERROR(checkpoint.status());
            checkpointed = std::move(checkpoint).value();
            session = std::make_unique<GraphRewriteSession>(*checkpointed);
            last_was_checkpoint = true;
        } else {
            last_was_checkpoint = false;
        }
    }

    // If the last pass was a checkpoint, `checkpointed` already holds the
    // final snapshot and another Commit would only deep-copy it. Otherwise
    // commit the accumulated changes from trailing non-checkpoint passes.
    if (last_was_checkpoint) {
        return std::move(*checkpointed);
    }
    return session->Commit();
}

}// namespace aethermind
