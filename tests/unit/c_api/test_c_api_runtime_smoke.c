#include "test_c_api_runtime_consumer.h"

#include <stdio.h>
#include <string.h>

// Independent C executable for the pinned M0 fixture, without a C++ test runner.
// Geometry is part of that test recipe, not a default of the runtime adapter.
int main(int argc, char** argv) {
    if (argc != 3 || (strcmp(argv[2], "o1-plain") != 0 && strcmp(argv[2], "o2-packed") != 0)) {
        fputs("usage: c_api_smoke <M0-FP32-directory> <o1-plain|o2-packed>\n", stderr);
        return 2;
    }
    const am_config_t runtime_config = {
            sizeof(am_config_t),
            AM_ABI_VERSION,
            2,
            4,
            4,
            8,
            AM_KV_DTYPE_FLOAT32,
            strcmp(argv[2], "o1-plain") == 0 ? AM_CPU_PRESET_O1_PLAIN : AM_CPU_PRESET_O2_PACKED,
    };
    const am_session_config_t session_config = {sizeof(am_session_config_t), AM_ABI_VERSION, 3, 0, 0};
    const uint32_t prompt[] = {7};
    uint32_t tokens[3] = {0};
    uint32_t repeated[3] = {0};
    struct AetherMindCInferenceResult result;
    AetherMindRunCInference(
            &runtime_config, argv[1], &session_config, prompt, 1,
            tokens, 3, repeated, 3, 1,
            AETHERMIND_C_RELEASE_RUNTIME | AETHERMIND_C_RELEASE_MODEL, &result);
    if (result.failure_stage != AETHERMIND_C_INFERENCE_SUCCESS ||
        result.abi_version != AM_ABI_VERSION || result.unexpected_success_error ||
        result.output_len != 3 || result.repeated_output_len != 3 ||
        memcmp(tokens, repeated, sizeof(tokens)) != 0) {
        fprintf(stderr, "C ABI smoke failed: stage=%d status=%d %s\n",
                result.failure_stage, (int) result.code, result.diagnostic);
        return 1;
    }
    printf("[%u,%u,%u]\n", (unsigned int) tokens[0], (unsigned int) tokens[1], (unsigned int) tokens[2]);
    return 0;
}
