#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_PASS_MANAGER_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_PASS_MANAGER_H

#include "aethermind/model/graph/graph_rewrite.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace aethermind {

struct PassContext {
    uint32_t opt_level = 2;
    uint32_t checkpoint_every = 0;

    bool enable_qkv_fusion = true;
    bool enable_swiglu_fusion = true;
    bool enable_flash_attention_rewrite = true;
    bool enable_fused_add_rms_norm = true;
};

class GraphPass {
public:
    virtual ~GraphPass() = default;

    AM_NODISCARD virtual std::string_view Name() const noexcept = 0;
    AM_NODISCARD virtual Status Run(GraphRewriteSession& session, const PassContext& ctx) = 0;
};

class GraphPassManager {
public:
    GraphPassManager() = default;
    explicit GraphPassManager(PassContext ctx) noexcept;

    GraphPassManager& Add(std::unique_ptr<GraphPass> pass);
    GraphPassManager& SetCheckpointEvery(uint32_t pass_count) noexcept;

    AM_NODISCARD StatusOr<ModelGraph> Run(const ModelGraph& graph) const;

private:
    std::vector<std::unique_ptr<GraphPass>> passes_{};
    PassContext ctx_{};
};

}// namespace aethermind

#endif
