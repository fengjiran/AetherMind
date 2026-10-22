#ifndef AETHERMIND_C_API_H
#define AETHERMIND_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Handle to Object from C API's pov */
#ifdef __cplusplus
using ObjectHandle = void*;
#else
typedef void* ObjectHandle;
#endif

#ifdef __cplusplus
using FObjectDeleter = void (*)(ObjectHandle, uint8_t);
#else
typedef void (*FObjectDeleter)(ObjectHandle, uint8_t);
#endif

struct ObjectHeader {
    /*! \brief Reference counter of the object. */
    uint32_t strong_ref_count;

    /*! \brief Weak reference counter of the object. */
    uint32_t weak_ref_count;

    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    FObjectDeleter deleter;
};

int IncObjectRef(ObjectHandle obj_ptr);

int DecObjectRef(ObjectHandle obj_ptr);

#ifdef __cplusplus
enum BacktraceUpdateMode : uint8_t {
#else
typedef enum {
#endif
    kBacktraceUpdateModeReplace = 0,
    kBacktraceUpdateModeAppend
#ifdef __cplusplus
};
#else
} BacktraceUpdateMode;
#endif

typedef enum am_status_code {
    AM_STATUS_OK = 0,
    AM_STATUS_CANCELLED = 1,
    AM_STATUS_UNKNOWN = 2,
    AM_STATUS_INVALID_ARGUMENT = 3,
    AM_STATUS_DEADLINE_EXCEEDED = 4,
    AM_STATUS_NOT_FOUND = 5,
    AM_STATUS_ALREADY_EXISTS = 6,
    AM_STATUS_PERMISSION_DENIED = 7,
    AM_STATUS_RESOURCE_EXHAUSTED = 8,
    AM_STATUS_FAILED_PRECONDITION = 9,
    AM_STATUS_ABORTED = 10,
    AM_STATUS_OUT_OF_RANGE = 11,
    AM_STATUS_UNIMPLEMENTED = 12,
    AM_STATUS_INTERNAL = 13,
    AM_STATUS_UNAVAILABLE = 14,
    AM_STATUS_DATA_LOSS = 15,
    AM_STATUS_OVERFLOW = 16,
    AM_STATUS_UNAUTHENTICATED = 17
} am_status_code;

#ifdef __cplusplus
using am_error_handle = void*;
#else
typedef void* am_error_handle;
#endif

/// @brief Creates an independently owned diagnostic; returns NULL on allocation failure.
/// @param code Status to retain without remapping.
/// @param message NUL-terminated text copied into the object; NULL means empty text.
/// @return Error handle to destroy with am_error_destroy, or NULL; no exception escapes.
am_error_handle am_error_create(am_status_code code, const char* message);

/// @brief Releases a diagnostic; NULL is a no-op.
/// @param error Live error handle, or NULL. Repeated destruction of a non-NULL handle is invalid.
void am_error_destroy(am_error_handle error);

/// @brief Reads a diagnostic's status; NULL returns AM_STATUS_UNKNOWN.
/// @param error Live error handle, or NULL.
/// @return Stored status code, or AM_STATUS_UNKNOWN when no diagnostic is available.
am_status_code am_error_code(am_error_handle error);

/// @brief Borrows diagnostic text until its error handle is destroyed.
/// @param error Live error handle, or NULL.
/// @return Non-NULL NUL-terminated text; NULL handles return the empty string.
const char* am_error_message(am_error_handle error);

// Frozen v1 runtime ABI. Shared Runtime access requires external serialization.
// See docs/api/runtime-c-abi.md for the ownership, error and buffer contracts.
#define AM_ABI_VERSION_MAJOR 1u
#define AM_ABI_VERSION_MINOR 0u
#define AM_ABI_VERSION ((AM_ABI_VERSION_MAJOR << 16u) | AM_ABI_VERSION_MINOR)

#define AM_CPU_PRESET_O1_PLAIN 0u
#define AM_CPU_PRESET_O2_PACKED 1u
#define AM_KV_DTYPE_FLOAT32 0u

typedef struct am_runtime* am_runtime_t;
typedef struct am_model* am_model_t;
typedef struct am_session* am_session_t;

/// @brief Fixed v1 CPU runtime and model preparation policy.
/// @note Zero-initialize, then set struct_size, abi_version and positive geometry.
/// CPU/FP32 and 64-byte KV alignment are the first-release resource policy.
/// cpu_preset is retained by the C adapter for later model compilation; it is
/// not a compiler dependency of the underlying Runtime.
typedef struct am_config_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t num_layers;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t max_tokens;
    uint32_t kv_dtype;
    uint32_t cpu_preset;
} am_config_t;

/// @brief Fixed v1 generation configuration copied by session creation.
/// @note has_eos_token_id is 0 or 1; eos_token_id is ignored when it is 0.
/// max_new_tokens includes the Prefill prediction and any generated EOS.
typedef struct am_session_config_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t max_new_tokens;
    uint32_t has_eos_token_id;
    uint32_t eos_token_id;
} am_session_config_t;

/// @brief Returns the implemented runtime ABI as (major << 16) | minor.
/// @return Packed ABI version; this query does not allocate or throw.
uint32_t am_get_abi_version(void);

/// @brief Creates a CPU Runtime with eagerly allocated static FP32 KV storage.
/// @param config Required fixed v1 config; copied, never retained by address.
/// @param error Optional owned diagnostic output, cleared on entry. The caller
/// must release any previous value first; this function never reads that value.
/// @return New handle, or NULL. Diagnostic OOM leaves *error NULL on failure.
/// @note No exception escapes. NULL config is InvalidArgument. Geometry must
/// fit size_t; the Runtime remains at a stable address until its final owner dies.
am_runtime_t am_runtime_create(const am_config_t* config, am_error_handle* error);

/// @brief Releases the caller's Runtime handle; existing model/session owners retain it.
/// @param runtime Live handle, or NULL for a no-op. Non-NULL handles may be released once.
void am_runtime_destroy(am_runtime_t runtime);

/// @brief Loads, admits, compiles and prepares a model against the given Runtime.
/// @param runtime Required live handle; the model retains its Runtime owner.
/// @param path Required nonempty NUL-terminated local HF directory path, borrowed
/// only during this call, using the platform's filesystem path encoding.
/// @param error Optional diagnostic output with the same ownership as runtime_create.
/// @return New model handle, or NULL; failures publish no partial model.
/// @note Compile precedes the adapter's Runtime/model KV validation and Prepare.
/// Runtime KV is already allocated; model rejection does not undo that allocation.
am_model_t am_model_load(am_runtime_t runtime, const char* path, am_error_handle* error);

/// @brief Releases the caller's model handle; existing sessions retain its owners.
/// @param model Live handle, or NULL for a no-op. Non-NULL handles may be released once.
void am_model_unload(am_model_t model);

/// @brief Creates a synchronous session sharing the prepared model and Runtime.
/// @param model Required live model handle, retained by the session.
/// @param config Required fixed v1 config, copied into immutable session state.
/// @param error Optional diagnostic output with the same ownership as runtime_create.
/// @return New session handle, or NULL; no exception escapes.
am_session_t am_session_create(am_model_t model, const am_session_config_t* config,
                               am_error_handle* error);

/// @brief Releases a session and its retained owners; NULL is a no-op.
/// @param session Live handle, or NULL. Non-NULL handles may be released once.
void am_session_destroy(am_session_t session);

/// @brief Generates only new token IDs, including generated EOS.
/// @param session Required live session; all access sharing its Runtime is serialized.
/// @param input_tokens Required nonempty prompt, borrowed only during this call.
/// @param input_len Prompt length in uint32_t elements.
/// @param output_tokens Caller-owned output. NULL requires output_capacity == 0.
/// @param output_capacity Output capacity in uint32_t elements. For positive N,
/// must be at least the configured max_new_tokens, even if EOS could stop early.
/// @param output_len Required output, set to zero on entry. Success writes the
/// actual length; short output writes N as a capacity upper bound and returns
/// OutOfRange without generating/reserving KV. Other failures leave zero.
/// @param error Optional diagnostic output with the same ownership as runtime_create.
/// @return Status independent of diagnostic availability. No exception escapes.
/// @note Output tokens are changed only after complete success. NULL/zero output
/// can query the upper bound for positive N; this does not validate token values.
/// N=0 still validates prompt/EOS/model context, returns zero, and reserves no KV.
/// Invalid/stale handles or invalid/overlapping memory regions violate caller
/// preconditions; no handle registry or rollback of invalid memory is provided.
am_status_code am_session_generate(
        am_session_t session,
        const uint32_t* input_tokens, size_t input_len,
        uint32_t* output_tokens, size_t output_capacity, size_t* output_len,
        am_error_handle* error);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
const char* AetherMindTraceback(const char* filename, int lineno, const char* func,
                                int cross_aethermind_boundary = 0);
#else
const char* AetherMindTraceback(const char* filename, int lineno, const char* func,
                                int cross_aethermind_boundary);
#endif

#endif // AETHERMIND_C_API_H
