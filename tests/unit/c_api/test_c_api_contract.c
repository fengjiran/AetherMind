#include "c_api.h"

#define AM_ASSERT_SIGNATURE(function, signature) \
    _Static_assert(_Generic(&(function), signature: 1, default: 0), #function " signature changed")

typedef uint32_t (*abi_version_signature)(void);
typedef am_runtime_t (*runtime_create_signature)(const am_config_t*, am_error_handle*);
typedef void (*runtime_destroy_signature)(am_runtime_t);
typedef am_model_t (*model_load_signature)(am_runtime_t, const char*, am_error_handle*);
typedef void (*model_unload_signature)(am_model_t);
typedef am_session_t (*session_create_signature)(am_model_t, const am_session_config_t*, am_error_handle*);
typedef void (*session_destroy_signature)(am_session_t);
typedef am_status_code (*session_generate_signature)(am_session_t, const uint32_t*, size_t,
                                                     uint32_t*, size_t, size_t*, am_error_handle*);

AM_ASSERT_SIGNATURE(am_get_abi_version, abi_version_signature);
AM_ASSERT_SIGNATURE(am_runtime_create, runtime_create_signature);
AM_ASSERT_SIGNATURE(am_runtime_destroy, runtime_destroy_signature);
AM_ASSERT_SIGNATURE(am_model_load, model_load_signature);
AM_ASSERT_SIGNATURE(am_model_unload, model_unload_signature);
AM_ASSERT_SIGNATURE(am_session_create, session_create_signature);
AM_ASSERT_SIGNATURE(am_session_destroy, session_destroy_signature);
AM_ASSERT_SIGNATURE(am_session_generate, session_generate_signature);

_Static_assert(AM_ABI_VERSION == 0x00010000u, "ABI version changed");
_Static_assert(AM_CPU_PRESET_O1_PLAIN == 0u, "O1 preset changed");
_Static_assert(AM_CPU_PRESET_O2_PACKED == 1u, "O2 preset changed");
_Static_assert(AM_KV_DTYPE_FLOAT32 == 0u, "KV dtype changed");
_Static_assert(AM_STATUS_OK == 0 && AM_STATUS_CANCELLED == 1 && AM_STATUS_UNKNOWN == 2 &&
                       AM_STATUS_INVALID_ARGUMENT == 3 && AM_STATUS_DEADLINE_EXCEEDED == 4 &&
                       AM_STATUS_NOT_FOUND == 5 && AM_STATUS_ALREADY_EXISTS == 6 &&
                       AM_STATUS_PERMISSION_DENIED == 7 && AM_STATUS_RESOURCE_EXHAUSTED == 8 &&
                       AM_STATUS_FAILED_PRECONDITION == 9 && AM_STATUS_ABORTED == 10 &&
                       AM_STATUS_OUT_OF_RANGE == 11 && AM_STATUS_UNIMPLEMENTED == 12 &&
                       AM_STATUS_INTERNAL == 13 && AM_STATUS_UNAVAILABLE == 14 &&
                       AM_STATUS_DATA_LOSS == 15 && AM_STATUS_OVERFLOW == 16 && AM_STATUS_UNAUTHENTICATED == 17,
               "Status values changed");

_Static_assert(sizeof(am_config_t) == 48, "v1 runtime config size changed");
_Static_assert(offsetof(am_config_t, struct_size) == 0, "struct_size offset changed");
_Static_assert(offsetof(am_config_t, abi_version) == 4, "abi_version offset changed");
_Static_assert(offsetof(am_config_t, num_layers) == 8, "num_layers offset changed");
_Static_assert(offsetof(am_config_t, num_kv_heads) == 16, "num_kv_heads offset changed");
_Static_assert(offsetof(am_config_t, head_dim) == 24, "head_dim offset changed");
_Static_assert(offsetof(am_config_t, max_tokens) == 32, "max_tokens offset changed");
_Static_assert(offsetof(am_config_t, kv_dtype) == 40, "kv_dtype offset changed");
_Static_assert(offsetof(am_config_t, cpu_preset) == 44, "cpu_preset offset changed");
_Static_assert(sizeof(am_session_config_t) == 24, "v1 session config size changed");
_Static_assert(offsetof(am_session_config_t, struct_size) == 0, "session struct_size offset changed");
_Static_assert(offsetof(am_session_config_t, abi_version) == 4, "session abi_version offset changed");
_Static_assert(offsetof(am_session_config_t, max_new_tokens) == 8, "max_new_tokens offset changed");
_Static_assert(offsetof(am_session_config_t, has_eos_token_id) == 16, "has_eos_token_id offset changed");
_Static_assert(offsetof(am_session_config_t, eos_token_id) == 20, "eos_token_id offset changed");

// Runtime signatures are checked above; full C generation calls are exercised
// by test_c_api_runtime_consumer.c. This function checks the error ABI separately.
int AetherMindCErrorContract(void) {
    am_error_handle error = am_error_create(AM_STATUS_INVALID_ARGUMENT, "C caller");
    if (error == NULL) {
        return 1;
    }
    const int wrong_code = am_error_code(error) != AM_STATUS_INVALID_ARGUMENT;
    const char* message = am_error_message(error);
    const int wrong_message = message == NULL || message[0] != 'C';
    am_error_destroy(error);
    am_error_destroy(NULL);
    return wrong_code || wrong_message || am_error_code(NULL) != AM_STATUS_UNKNOWN;
}

size_t AetherMindCConfigLayout(unsigned int field) {
    const size_t layout[] = {
            sizeof(am_config_t),
            offsetof(am_config_t, cpu_preset),
            sizeof(am_session_config_t),
            offsetof(am_session_config_t, eos_token_id),
    };
    return field < sizeof(layout) / sizeof(layout[0]) ? layout[field] : 0;
}
