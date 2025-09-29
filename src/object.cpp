//
// Created by richard on 9/29/25.
//
#include "object.h"

int IncObjectRef(ObjectHandle obj_ptr) {
    aethermind::details::ObjectUnsafe::IncRefObjectHandle(obj_ptr);
    return 0;
}

int DecObjectRef(ObjectHandle obj_ptr) {
    aethermind::details::ObjectUnsafe::DecRefObjectHandle(obj_ptr);
    return 0;
}