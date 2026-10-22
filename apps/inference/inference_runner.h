#ifndef AETHERMIND_APPS_INFERENCE_INFERENCE_RUNNER_H
#define AETHERMIND_APPS_INFERENCE_INFERENCE_RUNNER_H

/// @file inference_runner.h
/// @brief Private input contracts and synchronous inference application runner.
///
/// These application interfaces are not part of the public C++ ABI. Input and
/// configuration helpers borrow metadata only for the duration of a call. The
/// runner owns Runtime, model, and session until generation finishes and returns
/// only owning token IDs.

#include "aethermind/base/status.h"
#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/runtime/runtime_options.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace aethermind::apps::inference {

struct RunOptions {
    std::filesystem::path model_dir{};
    std::vector<uint32_t> prompt_tokens{};
    size_t context_tokens = 0;
    GenerationConfig generation{};
    CpuInferencePreset preset = CpuInferencePreset::kO1Plain;
};

/// @brief Stage whose operation returned the runner's error status.
enum class RunStage : uint8_t {
    kNone,
    kParameters,
    kLoad,
    kModelAdmission,
    kCompile,
    kRuntimeBuild,
    kPrepare,
    kSession,
    kGenerate,
};

/// @brief Returns a static diagnostic label without allocating.
/// @param stage Runner failure stage, or kNone on success.
/// @return Stage label, or "unknown" for an unrecognized value.
AM_NODISCARD const char* ToString(RunStage stage) noexcept;

/// @brief Validates application inputs against loaded model metadata.
/// @param options Owned request values; no filesystem access is performed.
/// @param config Configuration already validated by ModelLoader.
/// @return Ok or a parameter/range/overflow error; no model or KV execution occurs.
/// @note A zero generation limit keeps prompt, vocabulary, EOS, and model context
/// checks, but does not require the prompt to fit the physical KV capacity.
Status ValidateRunOptions(const RunOptions& options,
                          const HfModelConfig& config);

/// @brief Maps loaded model geometry and explicit capacity to CPU runtime values.
/// @param config Configuration already validated by ModelLoader.
/// @param context_tokens Positive KV capacity no larger than the model context.
/// @return Owning RuntimeOptions values or a conversion/geometry error; the
/// function allocates no KV storage and does not inspect backend availability.
StatusOr<RuntimeOptions> MakeRuntimeOptions(
        const HfModelConfig& config, size_t context_tokens);

/// @brief Loads, compiles, prepares, and generates with one request-owned Runtime.
/// @param options Owning input values borrowed for the duration of this call.
/// @param failure_stage Optional error diagnostic output; set to kNone on entry
///        and success, or to the failing operation's stage on failure.
/// @return Owning new token IDs, or the original error status. An escaping
///         std::bad_alloc becomes ResourceExhausted with an empty message.
/// @note Runtime remains at one address from preparation until all borrowing
///       model/session objects have been destroyed. Other exceptions propagate
///       to the process entrypoint after local resources are released.
StatusOr<std::vector<uint32_t>> RunInference(
        const RunOptions& options, RunStage* failure_stage = nullptr);

} // namespace aethermind::apps::inference

#endif
