// Intrusive object lifetime primitives for strong/weak handle promotion and teardown.
//
// This file contains the non-inline ownership transitions that must coordinate
// destruction order, storage release, and weak-to-strong promotion races.
#include "object.h"
#include "object_allocator.h"

namespace aethermind {

Object::Object() {
    header_.strong_ref_count = 0;
    header_.weak_ref_count = 0;
    header_.deleter = nullptr;
}

void Object::DecRef() {
    if (__atomic_fetch_sub(&header_.strong_ref_count, 1, __ATOMIC_RELEASE) == 1) {
        if (weak_use_count() == 1) {
            // Acquire only on the final teardown path so the deleter sees prior writes.
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (header_.deleter != nullptr) {
                header_.deleter(this, kBothPtrMask);
            }
        } else {
            // A weak handle still keeps the control block alive.
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            // Destroy the object payload first and leave storage reclamation to the weak path.
            if (header_.deleter != nullptr) {
                header_.deleter(this, kStrongPtrMask);
            }

            if (__atomic_fetch_sub(&header_.weak_ref_count, 1, __ATOMIC_RELEASE) == 1) {
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
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
        if (header_.deleter != nullptr) {
            header_.deleter(this, kWeakPtrMask);
        }
    }
}

bool Object::TryPromoteWeakPtr() {
    uint32_t old_cnt = __atomic_load_n(&header_.strong_ref_count, __ATOMIC_RELAXED);
    // CAS promotion avoids racing with concurrent promotions and final strong teardown.
    while (old_cnt > 0) {
        uint32_t new_cnt = old_cnt + 1;
        if (__atomic_compare_exchange_n(&header_.strong_ref_count, &old_cnt, new_cnt, true,
                                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return true;
        }
    }
    return false;
}

#ifdef AETHERMIND_ALLOCATOR_DEBUG
namespace details {
backtrace_state* AllocTracker::bt_state_ = nullptr;
}
#endif


}// namespace aethermind

int IncObjectRef(ObjectHandle obj_ptr) {
    aethermind::details::ObjectUnsafe::IncRefObjectHandle(obj_ptr);
    return 0;
}

int DecObjectRef(ObjectHandle obj_ptr) {
    aethermind::details::ObjectUnsafe::DecRefObjectHandle(obj_ptr);
    return 0;
}
