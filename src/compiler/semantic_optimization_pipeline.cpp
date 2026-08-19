#include "aethermind/compiler/semantic_optimization_pipeline.h"

#include "aethermind/graph/optimization/add_rmsnorm_fusion_pass.h"
#include "aethermind/graph/optimization/constant_folding_pass.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/graph/optimization/gate_up_linear_fusion_pass.h"
#include "aethermind/graph/optimization/qkv_linear_fusion_pass.h"
#include "aethermind/graph/optimization/silu_mul_fusion_pass.h"

namespace aethermind {
namespace {

GraphPassManager BuildDefaultOptPipeline(PassContext context) {
    GraphPassManager pipeline(context);
    switch (context.opt_level) {
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
