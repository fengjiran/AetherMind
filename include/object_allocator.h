//
// Created by richard on 11/25/25.
//

#ifndef AETHERMIND_OBJECT_ALLOCATOR_H
#define AETHERMIND_OBJECT_ALLOCATOR_H

#include "object.h"

namespace aethermind {

namespace details {

/*!
 * \brief Allocate aligned memory.
 * \param sz The size.
 * \tparam align The alignment, must be a power of 2.
 * \return The pointer to the allocated memory.
 */
template<size_t align>
void* AllocObject(size_t sz) {
    static_assert(align != 0 && (align & align - 1) == 0, "align must be a power of 2");
    if constexpr (align <= alignof(std::max_align_t)) {
        if (void* ptr = std::malloc(sz)) {
            return ptr;
        }
        throw std::bad_alloc();
    } else {
        void* ptr;
        if (posix_memalign(&ptr, align, sz) != 0) {
            throw std::bad_alloc();
        }
        return ptr;
    }
}

inline void FreeObject(void* ptr) {
    std::free(ptr);
}

// allocate object
template<typename Derived>
class ObjectAllocatorBase {
public:
    template<typename T, typename... Args>
    ObjectPtr<T> make_object(Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "Can only be used to create Object");
        using Handler = Derived::template Handler<T>;
        T* ptr = Handler::allocate(std::forward<Args>(args)...);
        ptr->SetDeleter(Handler::deleter);
        ObjectUnsafe::IncWeakRefObjectHandle(ptr);
        return ObjectPtr<T>(ptr);
    }

    template<typename T, typename ElemType, typename... Args>
    ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "Can only be used to create Object");
        using Handler = Derived::template ArrayHandler<T, ElemType>;
        T* ptr = Handler::allocate(num_elems, std::forward<Args>(args)...);
        ptr->SetDeleter(Handler::deleter);
        ObjectUnsafe::IncWeakRefObjectHandle(ptr);
        return ObjectPtr<T>(ptr);
    }
};

class ObjectAllocator : public ObjectAllocatorBase<ObjectAllocator> {
public:
    template<typename T>
    struct Handler {
        template<typename... Args>
        static T* allocate(Args&&... args) {
            void* data = AllocObject<alignof(T)>(sizeof(T));
            new (data) T(std::forward<Args>(args)...);
            return static_cast<T*>(data);
        }

        static void deleter(void* ptr, uint8_t flag) {
            auto* p = static_cast<T*>(ptr);
            if (flag & kStrongPtrMask) {
                p->T::~T();// release source
            }

            if (flag & kWeakPtrMask) {
                FreeObject(static_cast<void*>(p));// free memory
            }
        }
    };

    template<typename ObjType, typename ElemType>
    struct ArrayHandler {
        // for now, only support elements that aligns with array header.
        static_assert(alignof(ObjType) % alignof(ElemType) == 0 &&
                              sizeof(ObjType) % alignof(ElemType) == 0,
                      "element alignment constraint");

        // static_assert((alignof(ObjType) % alignof(ElemType) == 0 || alignof(ElemType) % alignof(ObjType) == 0) &&
        //                       sizeof(ObjType) % alignof(ElemType) == 0,
        //               "element alignment constraint");

        template<typename... Args>
        static ObjType* allocate(size_t num_elems, Args&&... args) {
            const size_t size = sizeof(ObjType) + num_elems * sizeof(ElemType);
            constexpr size_t align = alignof(ObjType);
            // constexpr size_t align = std::max(alignof(ObjType), alignof(ElemType));
            // C++ standard always guarantees that alignof operator returns a power of 2
            // const size_t aligned_size = (size + (align - 1)) & ~(align - 1);
            const size_t aligned_size = (size + align - 1) / align * align;
            void* data = AllocObject<align>(aligned_size);
            new (data) ObjType(std::forward<Args>(args)...);
            return static_cast<ObjType*>(data);
        }

        static void deleter(void* ptr, uint8_t flag) {
            auto* p = static_cast<ObjType*>(ptr);
            if (flag & kStrongPtrMask) {
                p->ObjType::~ObjType();
            }

            if (flag & kWeakPtrMask) {
                FreeObject(static_cast<void*>(p));// free memory
            }
        }
    };
};
}// namespace details


template<typename T, typename... Args>
ObjectPtr<T> make_object(Args&&... args) {
    return details::ObjectAllocator().make_object<T>(std::forward<Args>(args)...);
}

template<typename T, typename ElemType, typename... Args>
ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
    return details::ObjectAllocator().make_array_object<T, ElemType>(num_elems, std::forward<Args>(args)...);
}

}// namespace aethermind

#endif//AETHERMIND_OBJECT_ALLOCATOR_H
