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

StatusOr<LoweredModelArtifact> ModelCompiler::Compile(
        std::unique_ptr<LoadedModel> model,
        const ModelCompileOptions& options) {
    if (model == nullptr) {
        return Status::InvalidArgument("ModelCompiler::Compile requires a LoadedModel");
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

    return LoweredModelArtifact{
            .loaded_model = std::move(model),
            .graph = std::move(*lowered),
    };
}

StatusOr<LoweredModelArtifact> ModelCompiler::LoadAndCompile(
        const std::filesystem::path& model_dir,
        const ModelCompileOptions& compile_options) {
    auto model = ModelLoader::Load(model_dir);
    if (!model.ok()) {
        return AddStageContext(model.status(), "Model loading failed");
    }
    return Compile(std::move(*model), compile_options);
}

}// namespace aethermind
