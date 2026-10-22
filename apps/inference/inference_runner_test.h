#ifndef AETHERMIND_APPS_INFERENCE_INFERENCE_RUNNER_TEST_H
#define AETHERMIND_APPS_INFERENCE_INFERENCE_RUNNER_TEST_H

#ifndef AETHERMIND_INFERENCE_RUNNER_TESTING
#error "Runner builder injection is available only in test builds"
#endif

#include "inference_runner.h"

namespace aethermind {
class RuntimeBuilder;
}

namespace aethermind::apps::inference::testing {

/// @brief Runs the production orchestration using a preconfigured test builder.
/// @param options Request values borrowed for this call.
/// @param builder Builder with test allocator/backend registrations installed
///        before loading; derived RuntimeOptions replace its options after
///        Compile, while pending custom registrations retain normal semantics.
/// @param failure_stage Optional error-stage output with RunInference semantics.
/// @return Owning token IDs or the unchanged production failure status.
/// @note The builder is borrowed only for this call. Pending registrations may
///       be consumed by TryBuild; no Runtime or model borrow escapes.
AM_NODISCARD StatusOr<std::vector<uint32_t>> RunInferenceWithBuilder(
        const RunOptions& options, RuntimeBuilder& builder,
        RunStage* failure_stage = nullptr);

} // namespace aethermind::apps::inference::testing

#endif
