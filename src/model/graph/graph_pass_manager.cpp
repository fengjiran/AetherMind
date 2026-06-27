#include "aethermind/model/graph/graph_pass_manager.h"

#include <utility>

namespace aethermind {

GraphPassManager& GraphPassManager::Add(std::unique_ptr<GraphPass> pass) {
    passes_.push_back(std::move(pass));
    return *this;
}

GraphPassManager& GraphPassManager::SetCheckpointEvery(size_t pass_count) noexcept {
    checkpoint_every_ = pass_count;
    return *this;
}

StatusOr<ModelGraph> GraphPassManager::Run(const ModelGraph& graph) const {
    ModelGraph current = graph;
    std::unique_ptr<GraphRewriteSession> session = std::make_unique<GraphRewriteSession>(current);

    for (size_t i = 0; i < passes_.size(); ++i) {
        if (passes_[i] == nullptr) {
            return Status::InvalidArgument("GraphPassManager: pass cannot be null");
        }
        AM_RETURN_IF_ERROR(passes_[i]->Run(*session));

        const bool is_checkpoint = checkpoint_every_ != 0 && ((i + 1) % checkpoint_every_ == 0);
        const bool is_last = i + 1 == passes_.size();
        if (is_checkpoint && !is_last) {
            StatusOr<ModelGraph> checkpoint = session->Commit();
            AM_RETURN_IF_ERROR(checkpoint.status());
            current = std::move(checkpoint).value();
            session = std::make_unique<GraphRewriteSession>(current);
        }
    }

    return session->Commit();
}

}// namespace aethermind
