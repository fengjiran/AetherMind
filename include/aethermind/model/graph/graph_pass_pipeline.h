#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_PASS_PIPELINE_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_PASS_PIPELINE_H

#include "aethermind/model/graph/graph_rewrite.h"

#include <memory>
#include <string_view>
#include <vector>

namespace aethermind {

class GraphPass {
public:
    virtual ~GraphPass() = default;

    AM_NODISCARD virtual std::string_view Name() const noexcept = 0;
    AM_NODISCARD virtual Status Run(GraphRewriteSession& session) = 0;
};

class GraphPassPipeline {
public:
    GraphPassPipeline& Add(std::unique_ptr<GraphPass> pass);
    GraphPassPipeline& SetCheckpointEvery(size_t pass_count) noexcept;

    AM_NODISCARD StatusOr<ModelGraph> Run(const ModelGraph& graph) const;

private:
    std::vector<std::unique_ptr<GraphPass>> passes_{};
    size_t checkpoint_every_ = 0;
};

}// namespace aethermind

#endif
