#include "inference_runner.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "utils/overflow_check.h"

#ifdef AETHERMIND_INFERENCE_RUNNER_TESTING
#include "inference_runner_test.h"
#endif

#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind::apps::inference {
namespace {

StatusOr<size_t> PositiveSize(int64_t value, std::string_view field) {
    if (value <= 0) {
        return Status::InvalidArgument(
                std::string(field) + " must be positive");
    }

    size_t result;
    if (CheckNarrow(value, &result)) {
        return Status::OutOfRange(
                std::string(field) + " is not representable as size_t");
    }
    return result;
}

bool IsSupportedPreset(CpuInferencePreset preset) noexcept {
    return preset == CpuInferencePreset::kO1Plain || preset == CpuInferencePreset::kO2Packed;
}

Status ValidateContextCapacity(size_t capacity, size_t model_context) {
    if (capacity == 0) {
        return Status::InvalidArgument(
                "context_tokens must be positive");
    }

    if (capacity > model_context) {
        return Status::OutOfRange(
                "context_tokens exceeds max_position_embeddings");
    }
    return Status::Ok();
}

StatusOr<std::vector<uint32_t>> RunInferenceBody(const RunOptions& options,
                                                 RuntimeBuilder& builder,
                                                 RunStage& stage) {
    stage = RunStage::kLoad;
    auto loaded = ModelLoader::Load(options.model_dir);
    if (!loaded.ok()) {
        return std::move(loaded).status();
    }

    RuntimeOptions runtime_options;
    ModelCompileOptions compile_options;
    {
        // No config borrow survives Compile's transfer of LoadedModel ownership.
        const HfModelConfig& config = (*loaded)->GetConfig();
        stage = RunStage::kParameters;
        AM_RETURN_IF_ERROR(ValidateRunOptions(options, config));
        stage = RunStage::kModelAdmission;
        AM_RETURN_IF_ERROR(ValidateFp32ExecutionInputs(**loaded));

        stage = RunStage::kParameters;
        auto mapped_runtime = MakeRuntimeOptions(
                config, options.context_tokens);
        if (!mapped_runtime.ok()) {
            return std::move(mapped_runtime).status();
        }

        auto mapped_compile = MakeCpuCompileOptions(options.preset);
        if (!mapped_compile.ok()) {
            return std::move(mapped_compile).status();
        }

        runtime_options = *mapped_runtime;
        compile_options = *mapped_compile;
    }

    stage = RunStage::kCompile;
    auto artifact = ModelCompiler::Compile(
            std::move(*loaded), compile_options);
    if (!artifact.ok()) {
        return std::move(artifact).status();
    }

    stage = RunStage::kRuntimeBuild;
    builder.WithOptions(runtime_options);
    auto built_runtime = builder.TryBuild();
    if (!built_runtime.ok()) {
        return std::move(built_runtime).status();
    }

    // Prepare and every later borrower use this final Runtime address. Reverse
    // destruction releases session/model before their Runtime-owned resources.
    Runtime runtime = std::move(*built_runtime);

    stage = RunStage::kPrepare;
    auto prepared = PrepareExecutableModel(runtime, std::move(*artifact));
    if (!prepared.ok()) {
        return std::move(prepared).status();
    }

    std::shared_ptr<const ExecutableModel> model =
            std::make_shared<ExecutableModel>(std::move(*prepared));

    stage = RunStage::kSession;
    auto session = InferenceSession::Create(runtime, std::move(model));
    if (!session.ok()) {
        return std::move(session).status();
    }

    stage = RunStage::kGenerate;
    return session->Generate(options.prompt_tokens, options.generation);
}

StatusOr<std::vector<uint32_t>> RunInferenceGuarded(
        const RunOptions& options, RuntimeBuilder& builder, RunStage* failure_stage) {
    if (failure_stage != nullptr) {
        *failure_stage = RunStage::kNone;
    }

    auto stage = RunStage::kLoad;
    try {
        auto result = RunInferenceBody(options, builder, stage);
        if (!result.ok() && failure_stage != nullptr) {
            *failure_stage = stage;
        }
        return result;
    } catch (const std::bad_alloc&) {
        if (failure_stage != nullptr) {
            *failure_stage = stage;
        }
        return Status::ResourceExhausted({});
    }
}

} // namespace

const char* ToString(RunStage stage) noexcept {
    switch (stage) {
        case RunStage::kNone:
            return "none";
        case RunStage::kParameters:
            return "parameters";
        case RunStage::kLoad:
            return "load";
        case RunStage::kModelAdmission:
            return "model admission";
        case RunStage::kCompile:
            return "compile";
        case RunStage::kRuntimeBuild:
            return "runtime build";
        case RunStage::kPrepare:
            return "prepare";
        case RunStage::kSession:
            return "session";
        case RunStage::kGenerate:
            return "generate";
    }
    return "unknown";
}

Status ValidateRunOptions(const RunOptions& options, const HfModelConfig& config) try {
    if (options.model_dir.empty()) {
        return Status::InvalidArgument("model_dir must not be empty");
    }

    if (!IsSupportedPreset(options.preset)) {
        return Status::InvalidArgument("Unknown CPU preset");
    }

    if (options.prompt_tokens.empty()) {
        return Status::InvalidArgument(
                "prompt_tokens must contain at least one token");
    }

    AM_ASSIGN_OR_RETURN(const size_t vocab_size,
                        PositiveSize(config.vocab_size, "vocab_size"));
    AM_ASSIGN_OR_RETURN(const size_t model_context,
                        PositiveSize(config.max_position_embeddings,
                                     "max_position_embeddings"));
    for (const uint32_t token: options.prompt_tokens) {
        if (static_cast<size_t>(token) >= vocab_size) {
            return Status::OutOfRange(
                    "prompt_tokens contains a token outside the model vocabulary");
        }
    }

    if (options.generation.eos_token_id.has_value() &&
        static_cast<size_t>(*options.generation.eos_token_id) >= vocab_size) {
        return Status::OutOfRange(
                "eos_token_id is outside the model vocabulary");
    }

    if (options.prompt_tokens.size() > model_context) {
        return Status::OutOfRange(
                "prompt_tokens exceeds max_position_embeddings");
    }

    AM_RETURN_IF_ERROR(ValidateContextCapacity(options.context_tokens, model_context));
    if (options.generation.max_new_tokens == 0) {
        return Status::Ok();
    }

    size_t required_tokens = 0;
    if (CheckOverflowAdd(options.prompt_tokens.size(),
                         options.generation.max_new_tokens - 1,
                         &required_tokens)) {
        return Status::Overflow(
                "Requested token count overflowed size_t");
    }

    if (required_tokens > model_context) {
        return Status::OutOfRange(
                "Prompt and generation exceed max_position_embeddings");
    }

    if (required_tokens > options.context_tokens) {
        return Status::OutOfRange(
                "Prompt and generation exceed context_tokens");
    }
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<RuntimeOptions> MakeRuntimeOptions(const HfModelConfig& config,
                                            size_t context_tokens) {
    AM_ASSIGN_OR_RETURN(auto kv_options, MakeFp32KVCacheOptions(config, context_tokens));
    RuntimeOptions options;
    options.allocator = {.enable_cpu = true, .enable_cuda = false, .enable_cann = false};
    options.backend.enable_cpu = true;
    options.backend.enable_cuda = false;
    options.backend.enable_cann = false;
    options.kv_cache = kv_options;
    return options;
}

StatusOr<std::vector<uint32_t>> RunInference(const RunOptions& options,
                                             RunStage* failure_stage) {
    RuntimeBuilder builder;
    return RunInferenceGuarded(options, builder, failure_stage);
}

#ifdef AETHERMIND_INFERENCE_RUNNER_TESTING
namespace testing {

StatusOr<std::vector<uint32_t>> RunInferenceWithBuilder(
        const RunOptions& options, RuntimeBuilder& builder, RunStage* failure_stage) {
    return RunInferenceGuarded(options, builder, failure_stage);
}

} // namespace testing
#endif

} // namespace aethermind::apps::inference
