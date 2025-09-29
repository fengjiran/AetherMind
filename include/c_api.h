//
// Created by richard on 9/29/25.
//

#ifndef AETHERMIND_C_API_H
#define AETHERMIND_C_API_H

#include "macros.h"

/*! \brief Handle to Object from C API's pov */
typedef void* ObjectHandle;

int IncObjectRef(ObjectHandle obj_ptr);

int DecObjectRef(ObjectHandle obj_ptr);

#endif//AETHERMIND_C_API_H
