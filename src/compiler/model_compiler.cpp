#include "aethermind/compiler/model_compiler.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/compiler/optimize_graph.h"
#include "aethermind/model/build_model_graph.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"

#include <string>
#include <string_view>
#include <utility>

namespace aethermind {
namespace {

Status AddStageContext(const Status& status, std::string_view stage) {
    return status.WithMessage(std::string(stage) + ": " + status.message());
}

// The P4-A A16 batch runs the unfused conservative baseline frozen in proposal
// 11 §4.5: only the standalone A16 kernel set exists, so graph fusions that
// would emit fused QKV/GateUp/AddRmsNorm kernels stay disabled. Explicit A16
// fusion requires separate O4 evidence and its own kernels.
PassContext EffectiveOptimization(const ModelCompileOptions& options) {
    PassContext optimization = options.optimization;
    const DataType activation = options.graph_build.activation_dtype;
    if (activation == DataType::Float(16) || activation == DataType::BFloat(16)) {
        optimization.enable_qkv_fusion = false;
        optimization.enable_gate_up_fusion = false;
        optimization.enable_fused_add_rms_norm = false;
    }
    return optimization;
}

} // namespace

StatusOr<LoweredModelArtifact> ModelCompiler::Compile(std::unique_ptr<LoadedModel> model,
                                                      const ModelCompileOptions& options) {
    if (model == nullptr) {
        return Status::InvalidArgument(
                "ModelCompiler::Compile requires a LoadedModel");
    }

    auto graph = BuildModelGraph(model->GetConfig(),
                                 model->GetResolvedWeights(),
                                 options.graph_build);
    if (!graph.ok()) {
        return AddStageContext(graph.status(), "Model graph construction failed");
    }

    auto optimized = OptimizeModelGraph(*graph,
                                        EffectiveOptimization(options));
    if (!optimized.ok()) {
        return AddStageContext(optimized.status(), "Model graph optimization failed");
    }

    auto lowered = LowerModelGraph(*optimized, options.lowering);
    if (!lowered.ok()) {
        return AddStageContext(lowered.status(), "Model graph lowering failed");
    }
    // The artifact keeps ownership of the model so consumers of the lowered
    // graph can access its weights during execution planning.
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

} // namespace aethermind
