#include "aethermind/model/graph/compilation/graph_compiler.h"
#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/model/graph/optimization/constant_folding_pass.h"
#include "aethermind/model/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/model/graph/optimization/silu_mul_fusion_pass.h"

namespace aethermind {

namespace {

/// Builds the default optimization pipeline based solely on PassContext::opt_level.
/// Feature flags are carried in the context and checked internally by each pass;
/// the pipeline builder does not gate pass registration on flags.
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
            pipeline.Add(std::make_unique<SiluMulFusionPass>());
            pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
            break;
    }

    return pipeline;
}

}// namespace

StatusOr<ModelGraph> OptimizeModelGraph(const ModelGraph& graph,
                                        PassContext context) {
    GraphPassManager pipeline = BuildDefaultOptPipeline(context);
    return pipeline.Run(graph);
}

StatusOr<CompiledModelGraph> CompileModelGraph(const ModelGraph& graph,
                                               const GraphCompileConfig& config) {
    StatusOr<ModelGraph> optimized = OptimizeModelGraph(graph, config.optimization);
    if (!optimized.ok()) {
        return optimized.status();
    }

    StatusOr<LoweredGraph> lowered = LowerModelGraph(*optimized, config.lowering);
    if (!lowered.ok()) {
        return lowered.status();
    }

    CompiledModelGraph result;
    result.optimized_graph = std::move(*optimized);
    result.lowered = std::move(*lowered);
    return result;
}

}// namespace aethermind
