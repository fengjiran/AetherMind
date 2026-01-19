//
// Created by richard on 11/25/25.
//

#ifndef AETHERMIND_OBJECT_ALLOCATOR_H
#define AETHERMIND_OBJECT_ALLOCATOR_H

// #define AETHERMIND_ALLOCATOR_DEBUG

#include "backtrace.h"
#include "object.h"

#include <mutex>
#include <unordered_map>

namespace aethermind {
namespace details {

#ifdef AETHERMIND_ALLOCATOR_DEBUG

struct AllocRecord {
    size_t size;          // allocate size
    size_t align;         // alignment
    const char* type_name;// type name
    void* call_stack[16]; // call stack
    int stack_frames;     // actual stack frames
};

class AllocTracker {
public:
    static AllocTracker& GetInstance() {
        static AllocTracker inst;
        return inst;
    }

    // ptr: allocate address
    // size: object size
    // align: object alignement
    // type name: allocated object type name
    void TrackAlloc(void* ptr, size_t size, size_t align, const char* type_name) {
        std::lock_guard<std::mutex> lock(mtx_);
        AllocRecord record(size, align, type_name, {nullptr}, 0);
        // backtrace(record.call_stack, sizeof(record.call_stack) / sizeof(void*));
    }

private:
    AllocTracker() {
        bt_state_ = backtrace_create_state(nullptr, 0, nullptr, nullptr);
        atexit([] { GetInstance(); });
    }

    std::unordered_map<void*, AllocRecord> alloc_map_;
    std::mutex mtx_;
    static backtrace_state* bt_state_;
};
#endif

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
    static ObjectAllocator& GetInstance() {
        static ObjectAllocator inst;
        return inst;
    }

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

        template<typename... Args>
        static ObjType* allocate(size_t num_elems, Args&&... args) {
            const size_t size = sizeof(ObjType) + num_elems * sizeof(ElemType);
            constexpr size_t align = alignof(ObjType);
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

    ObjectAllocator(const ObjectAllocator&) = delete;
    ObjectAllocator(ObjectAllocator&&) noexcept = delete;
    ObjectAllocator& operator=(const ObjectAllocator&) = delete;
    ObjectAllocator& operator=(ObjectAllocator&&) noexcept = delete;

private:
    ObjectAllocator() = default;
};
}// namespace details


template<typename T, typename... Args>
ObjectPtr<T> make_object(Args&&... args) {
    return details::ObjectAllocator::GetInstance().make_object<T>(std::forward<Args>(args)...);
}

template<typename T, typename ElemType, typename... Args>
ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
    return details::ObjectAllocator::GetInstance().make_array_object<T, ElemType>(num_elems, std::forward<Args>(args)...);
}

}// namespace aethermind

#endif//AETHERMIND_OBJECT_ALLOCATOR_H
