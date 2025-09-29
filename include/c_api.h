//
// Created by richard on 9/29/25.
//

#ifndef AETHERMIND_C_API_H
#define AETHERMIND_C_API_H

#include "macros.h"

#include <cstdint>

/*! \brief Handle to Object from C API's pov */
typedef void* ObjectHandle;
// using ObjectHandle = void*;

typedef void (*FDeleter)(ObjectHandle, uint8_t);
// using FDeleter = void(*)(void*, uint8_t);

struct ObjectHeader {
    /*! \brief Reference counter of the object. */
    uint32_t strong_ref_count_;

    /*! \brief Weak reference counter of the object. */
    uint32_t weak_ref_count_;

    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    FDeleter deleter_;
};

int IncObjectRef(ObjectHandle obj_ptr);

int DecObjectRef(ObjectHandle obj_ptr);

#endif//AETHERMIND_C_API_H
