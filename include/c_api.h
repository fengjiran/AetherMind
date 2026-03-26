//
// Created by richard on 9/29/25.
//

#ifndef AETHERMIND_C_API_H
#define AETHERMIND_C_API_H

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
    AM_STATUS_CANCELLED,
    AM_STATUS_UNKNOWN,
    AM_STATUS_INVALID_ARGUMENT,
    AM_STATUS_DEADLINE_EXCEEDED,
    AM_STATUS_NOT_FOUND,
    AM_STATUS_ALREADY_EXISTS,
    AM_STATUS_PERMISSION_DENIED,
    AM_STATUS_RESOURCE_EXHAUSTED,
    AM_STATUS_FAILED_PRECONDITION,
    AM_STATUS_ABORTED,
    AM_STATUS_OUT_OF_RANGE,
    AM_STATUS_UNIMPLEMENTED,
    AM_STATUS_INTERNAL,
    AM_STATUS_UNAVAILABLE,
    AM_STATUS_DATA_LOSS,
    AM_STATUS_UNAUTHENTICATED
} am_status_code;

#ifdef __cplusplus
using am_error_handle = void*;
#else
typedef void* am_error_handle;
#endif

am_error_handle am_error_create(am_status_code code, const char* message);

void am_error_destroy(am_error_handle error);

am_status_code am_error_code(am_error_handle error);

const char* am_error_message(am_error_handle error);

#ifdef __cplusplus
}// TVM_FFI_EXTERN_C
#endif

#ifdef __cplusplus
const char* AetherMindTraceback(const char* filename, int lineno, const char* func,
                                int cross_aethermind_boundary = 0);
#else
const char* AetherMindTraceback(const char* filename, int lineno, const char* func,
                                int cross_aethermind_boundary);
#endif

#endif// AETHERMIND_C_API_H
