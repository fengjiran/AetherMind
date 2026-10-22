#ifndef AETHERMIND_TEST_C_API_RUNTIME_CONSUMER_H
#define AETHERMIND_TEST_C_API_RUNTIME_CONSUMER_H

#include "c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum AetherMindCInferenceStage {
    AETHERMIND_C_INFERENCE_SUCCESS = 0,
    AETHERMIND_C_INFERENCE_RUNTIME = 1,
    AETHERMIND_C_INFERENCE_MODEL = 2,
    AETHERMIND_C_INFERENCE_SESSION = 3,
    AETHERMIND_C_INFERENCE_GENERATE = 4,
    AETHERMIND_C_INFERENCE_REPEAT = 5
};

enum AetherMindCReleaseOwners {
    AETHERMIND_C_RELEASE_NO_OWNERS = 0,
    AETHERMIND_C_RELEASE_RUNTIME = 1,
    AETHERMIND_C_RELEASE_MODEL = 2
};

struct AetherMindCInferenceResult {
    uint32_t abi_version;
    int failure_stage;
    am_status_code code;
    am_status_code diagnostic_code;
    int unexpected_success_error;
    size_t output_len;
    size_t repeated_output_len;
    char diagnostic[256];
};

// This test-only caller is compiled as C. Every lifecycle and generation call
// below resolves to the same exported ABI used by an external C application.
void AetherMindRunCInference(
        const am_config_t* runtime_config, const char* model_path,
        const am_session_config_t* session_config,
        const uint32_t* input_tokens, size_t input_len,
        uint32_t* output_tokens, size_t output_capacity,
        uint32_t* repeated_output_tokens, size_t repeated_output_capacity,
        int repeat_generation, unsigned int release_owners,
        struct AetherMindCInferenceResult* result);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
