#include "aethermind/graph/optimization/optimize_model_graph.h"
#include "aethermind/graph/optimization/add_rmsnorm_fusion_pass.h"
#include "aethermind/graph/optimization/constant_folding_pass.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/graph/optimization/gate_up_linear_fusion_pass.h"
#include "aethermind/graph/optimization/qkv_linear_fusion_pass.h"
#include "aethermind/graph/optimization/silu_mul_fusion_pass.h"

namespace aethermind {

namespace {

// Builds the default optimization pipeline based solely on PassContext::opt_level.
// Feature flags are carried in the context and checked internally by each pass;
// the pipeline builder does not gate pass registration on flags.
GraphPassManager BuildDefaultOptPipeline(PassContext ctx) {
    GraphPassManager pipeline(ctx);
    switch (ctx.opt_level) {
        case 0:
            break;
        case 1:
            pipeline.Add(std::make_unique<ConstantFoldingPass>());
            pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
            break;
        default:
            pipeline.Add(std::make_unique<ConstantFoldingPass>());
            pipeline.Add(std::make_unique<QkvLinearFusionPass>());
            pipeline.Add(std::make_unique<GateUpLinearFusionPass>());
            pipeline.Add(std::make_unique<SiluMulFusionPass>());
            pipeline.Add(std::make_unique<AddRmsNormFusionPass>());
            pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
            break;
    }

    return pipeline;
}

}// namespace

StatusOr<ModelGraph> OptimizeModelGraph(const ModelGraph& graph,
                                        PassContext context) {
    return BuildDefaultOptPipeline(context).Run(graph);
}

}// namespace aethermind
