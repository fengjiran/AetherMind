//
// Created by richard on 9/29/25.
//

#ifndef AETHERMIND_C_API_H
#define AETHERMIND_C_API_H

#include "macros.h"

#include <cstdint>

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
    uint32_t strong_ref_count_;

    /*! \brief Weak reference counter of the object. */
    uint32_t weak_ref_count_;

    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    FObjectDeleter deleter_;
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

#ifdef __cplusplus
}// TVM_FFI_EXTERN_C
#endif

const char* AetherMindTraceback(const char* filename, int lineno, const char* func,
    int cross_aethermind_boundary = 0);

#endif//AETHERMIND_C_API_H
