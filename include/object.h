//
// Created by 赵丹 on 2025/8/19.
//

#ifndef AETHERMIND_OBJECT_H
#define AETHERMIND_OBJECT_H

#include "macros.h"

#include <type_traits>

namespace aethermind {

template<typename T>
struct DefaultNullType final {
    static constexpr T* singleton() noexcept {
        return nullptr;
    }
};

struct DoNotIncRefCountTag final {};

class Object {
public:
    Object() : ref_count_(0), deleter(nullptr) {}

    NODISCARD uint32_t use_count() const {
        return __atomic_load_n(&ref_count_, __ATOMIC_RELAXED);
    }

    NODISCARD bool unique() const {
        return use_count() == 1;
    }

private:
    void IncRef() {
        __atomic_fetch_add(&ref_count_, 1, __ATOMIC_RELAXED);
    }

    void DecRef() {
        if (__atomic_fetch_sub(&ref_count_, 1, __ATOMIC_RELEASE) == 1) {
            // only acquire when we need to call deleter
            // in this case we need to ensure all previous writes are visible
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (deleter != nullptr) {
                deleter(this);
            }
        }
    }

    /*! \brief Reference counter of the object. */
    uint32_t ref_count_;

    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    void (*deleter)(Object* self);

    template<typename T, typename NullType>
    friend class ObjectPtr;
};

template<typename T, typename NullType = DefaultNullType<T>>
class ObjectPtr final {
    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(NullType::singleton() == NullType::singleton(), "NullType must have a constexpr singleton() method");
    static_assert(std::is_base_of_v<T, std::remove_pointer_t<decltype(NullType::singleton())>>,
                  "NullType::singleton() must return a element_type* pointer");

public:
    ObjectPtr(T* ptr, DoNotIncRefCountTag) : ptr_(ptr) {}

private:
    T* ptr_;
};

}// namespace aethermind

#endif//AETHERMIND_OBJECT_H
