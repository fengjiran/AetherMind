#include "test_c_api_runtime_consumer.h"

#include <string.h>

static void RecordFailure(struct AetherMindCInferenceResult* result, int stage,
                          am_status_code code, am_error_handle error) {
    result->failure_stage = stage;
    result->code = code;
    result->diagnostic_code = am_error_code(error);
    const char* message = am_error_message(error);
    size_t index = 0;
    while (message[index] != '\0' && index + 1 < sizeof(result->diagnostic)) {
        result->diagnostic[index] = message[index];
        ++index;
    }
    result->diagnostic[index] = '\0';
}

static void ReleaseUnexpectedSuccessError(struct AetherMindCInferenceResult* result,
                                          am_error_handle* error) {
    if (*error != NULL) {
        result->unexpected_success_error = 1;
        am_error_destroy(*error);
        *error = NULL;
    }
}

void AetherMindRunCInference(
        const am_config_t* runtime_config, const char* model_path,
        const am_session_config_t* session_config,
        const uint32_t* input_tokens, size_t input_len,
        uint32_t* output_tokens, size_t output_capacity,
        uint32_t* repeated_output_tokens, size_t repeated_output_capacity,
        int repeat_generation, unsigned int release_owners,
        struct AetherMindCInferenceResult* result) {
    am_runtime_t runtime = NULL;
    am_model_t model = NULL;
    am_session_t session = NULL;
    am_error_handle error = NULL;

    memset(result, 0, sizeof(*result));
    result->abi_version = am_get_abi_version();
    result->diagnostic_code = AM_STATUS_UNKNOWN;

    runtime = am_runtime_create(runtime_config, &error);
    if (runtime == NULL) {
        RecordFailure(result, AETHERMIND_C_INFERENCE_RUNTIME, am_error_code(error), error);
        goto cleanup;
    }
    ReleaseUnexpectedSuccessError(result, &error);

    model = am_model_load(runtime, model_path, &error);
    if (model == NULL) {
        RecordFailure(result, AETHERMIND_C_INFERENCE_MODEL, am_error_code(error), error);
        goto cleanup;
    }
    ReleaseUnexpectedSuccessError(result, &error);
    if ((release_owners & AETHERMIND_C_RELEASE_RUNTIME) != 0) {
        // The prepared model must retain Runtime even before a session exists.
        am_runtime_destroy(runtime);
        runtime = NULL;
    }

    session = am_session_create(model, session_config, &error);
    if (session == NULL) {
        RecordFailure(result, AETHERMIND_C_INFERENCE_SESSION, am_error_code(error), error);
        goto cleanup;
    }
    ReleaseUnexpectedSuccessError(result, &error);
    if ((release_owners & AETHERMIND_C_RELEASE_MODEL) != 0) {
        am_model_unload(model);
        model = NULL;
    }

    result->output_len = SIZE_MAX;
    am_status_code status = am_session_generate(
            session, input_tokens, input_len, output_tokens, output_capacity,
            &result->output_len, &error);
    if (status != AM_STATUS_OK) {
        RecordFailure(result, AETHERMIND_C_INFERENCE_GENERATE, status, error);
        goto cleanup;
    }
    ReleaseUnexpectedSuccessError(result, &error);

    if (repeat_generation) {
        result->repeated_output_len = SIZE_MAX;
        status = am_session_generate(
                session, input_tokens, input_len, repeated_output_tokens,
                repeated_output_capacity, &result->repeated_output_len, &error);
        if (status != AM_STATUS_OK) {
            RecordFailure(result, AETHERMIND_C_INFERENCE_REPEAT, status, error);
            goto cleanup;
        }
        ReleaseUnexpectedSuccessError(result, &error);
    }

cleanup:
    am_error_destroy(error);
    am_session_destroy(session);
    am_model_unload(model);
    am_runtime_destroy(runtime);
}
