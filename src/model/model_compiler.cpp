#include "aethermind/model/model_compiler.h"
#include "aethermind/graph/optimization/optimize_model_graph.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_graph_builder.h"
#include "aethermind/model/model_loader.h"

#include <string>
#include <string_view>
#include <utility>

namespace aethermind {
namespace {

Status AddStageContext(const Status& status, std::string_view stage) {
    return status.WithMessage(std::string(stage) + ": " + status.message());
}

}// namespace

StatusOr<LoweredModel> ModelCompiler::BuildLoweredModel(
        std::unique_ptr<LoadedModel> model,
        const ModelLoweringOptions& options) {
    if (model == nullptr) {
        return Status::InvalidArgument("ModelCompiler::BuildLoweredModel requires a LoadedModel");
    }

    auto graph = ModelGraphBuilder::BuildLlamaDense(
            model->GetConfig(), model->GetResolvedWeights());
    if (!graph.ok()) {
        return AddStageContext(graph.status(), "Model graph construction failed");
    }

    auto optimized = OptimizeModelGraph(*graph, options.optimization);
    if (!optimized.ok()) {
        return AddStageContext(optimized.status(), "Model graph optimization failed");
    }

    auto lowered = LowerModelGraph(*optimized, options.lowering);
    if (!lowered.ok()) {
        return AddStageContext(lowered.status(), "Model graph lowering failed");
    }

    return LoweredModel{
            .loaded_model = std::move(model),
            .graph = std::move(*lowered),
    };
}

StatusOr<LoweredModel> ModelCompiler::LoadAndLowerModel(
        const ModelLoadOptions& load_options,
        const ModelLoweringOptions& lowering_options) {
    auto model = ModelLoader::Load(load_options);
    if (!model.ok()) {
        return AddStageContext(model.status(), "Model loading failed");
    }
    return BuildLoweredModel(std::move(*model), lowering_options);
}

}// namespace aethermind
