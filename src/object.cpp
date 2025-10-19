//
// Created by richard on 9/29/25.
//
#include "object.h"

namespace aethermind {

Object::Object() {
    header_.strong_ref_count = 0;
    header_.weak_ref_count = 0;
    header_.deleter = nullptr;
}

uint32_t Object::use_count() const {
    return __atomic_load_n(&header_.strong_ref_count, __ATOMIC_RELAXED);
}

uint32_t Object::weak_use_count() const {
    return __atomic_load_n(&header_.weak_ref_count, __ATOMIC_RELAXED);
}

bool Object::unique() const {
    return use_count() == 1;
}

void Object::SetDeleter(FObjectDeleter deleter) {
    header_.deleter = deleter;
}

void Object::IncRef() {
    __atomic_fetch_add(&header_.strong_ref_count, 1, __ATOMIC_RELAXED);
}

void Object::IncWeakRef() {
    __atomic_fetch_add(&header_.weak_ref_count, 1, __ATOMIC_RELAXED);
}

void Object::DecRef() {
    if (__atomic_fetch_sub(&header_.strong_ref_count, 1, __ATOMIC_RELEASE) == 1) {
        if (weak_use_count() == 1) {
            // only acquire when we need to call deleter
            // in this case we need to ensure all previous writes are visible
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (header_.deleter != nullptr) {
                header_.deleter(this, kBothPtrMask);
            }
        } else {
            // Slower path: there is still a weak reference left
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            // call destructor first, release source
            if (header_.deleter != nullptr) {
                header_.deleter(this, kStrongPtrMask);
            }

            // decrease weak ref count
            if (__atomic_fetch_sub(&header_.weak_ref_count, 1, __ATOMIC_RELEASE) == 1) {
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
                // free memory
                if (header_.deleter != nullptr) {
                    header_.deleter(this, kWeakPtrMask);
                }
            }
        }
    }
}

void Object::DecWeakRef() {
    if (__atomic_fetch_sub(&header_.weak_ref_count, 1, __ATOMIC_RELEASE) == 1) {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        // free memory
        if (header_.deleter != nullptr) {
            header_.deleter(this, kWeakPtrMask);
        }
    }
}

bool Object::TryPromoteWeakPtr() {
    uint32_t old_cnt = __atomic_load_n(&header_.strong_ref_count, __ATOMIC_RELAXED);
    // must do CAS to ensure that we are the only one that increases the reference count
    // avoid condition when two threads tries to promote weak to strong at same time
    // or when strong deletion happens between the load and the CAS
    while (old_cnt > 0) {
        uint32_t new_cnt = old_cnt + 1;
        if (__atomic_compare_exchange_n(&header_.strong_ref_count, &old_cnt, new_cnt, true,
                                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return true;
        }
    }
    return false;
}


}// namespace aethermind

int IncObjectRef(ObjectHandle obj_ptr) {
    aethermind::details::ObjectUnsafe::IncRefObjectHandle(obj_ptr);
    return 0;
}

int DecObjectRef(ObjectHandle obj_ptr) {
    aethermind::details::ObjectUnsafe::DecRefObjectHandle(obj_ptr);
    return 0;
}