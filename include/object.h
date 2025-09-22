//
// Created by 赵丹 on 2025/8/19.
//

#ifndef AETHERMIND_OBJECT_H
#define AETHERMIND_OBJECT_H

#include "macros.h"

#include <glog/logging.h>
#include <memory>
#include <type_traits>

namespace aethermind {

namespace details {
// Helper to perform
// unsafe operations related to object
struct ObjectUnsafe;

template<typename Derived>
class ObjectAllocatorBase;

}// namespace details

/*!
 * \brief Base class for reference-counted objects.
 *
 * \note Objects of this class are reference-counted, meaning that each object
 * has a reference count that tracks the number of references to the object.
 * When the reference count reaches zero, the object is automatically destroyed.
 */
class Object {
public:
    using FDeleter = void (*)(Object*);

    /*!
    * \brief Default constructor, initializes a reference-counted object.
    *
    * \note Creates a new Object instance with reference count initialized to 0
    * and deleter set to nullptr.
    * Newly created objects require manual reference count management,
    * typically wrapped using ObjectPtr.
    */
    Object() : strong_ref_count_(0), weak_ref_count_(0), deleter_(nullptr) {}

    /*!
     * \brief Get the current reference count of the object.
     *
     * \note Uses atomic operation with relaxed memory ordering (__ATOMIC_RELAXED)
     * to load the reference count value. Relaxed memory ordering is suitable
     * for performance-sensitive scenarios where strict memory synchronization
     * is not required.
     *
     * \return The current reference count value of the object
     */
    NODISCARD uint32_t use_count() const {
        return __atomic_load_n(&strong_ref_count_, __ATOMIC_RELAXED);
    }

    NODISCARD uint32_t weak_use_count() const {
        return __atomic_load_n(&weak_ref_count_, __ATOMIC_RELAXED);
    }

    /*!
     * \brief Check if the object has a unique reference count.
     *
     * \return true if the object has a reference count of 1, false otherwise.
     */
    NODISCARD bool unique() const {
        return use_count() == 1;
    }

    /*!
     * \brief Set the deleter function for the object.
     *
     * \param deleter The deleter function to be invoked when the reference count reaches zero.
     */
    void SetDeleter(FDeleter deleter) {
        deleter_ = deleter;
    }

private:
    /*!
     * \brief Increment the reference count of the object.
     *
     * \note Uses atomic operation with relaxed memory ordering (__ATOMIC_RELAXED)
     * to increment the reference count. Relaxed memory ordering is suitable
     * for performance-sensitive scenarios where strict memory synchronization
     * is not required.
     */
    void IncRef() {
        __atomic_fetch_add(&strong_ref_count_, 1, __ATOMIC_RELAXED);
    }

    void IncWeakRef() {
        __atomic_fetch_add(&weak_ref_count_, 1, __ATOMIC_RELAXED);
    }

    /*!
     * \brief Decrement the reference count of the object.
     *
     * \note Uses atomic operation with release memory ordering (__ATOMIC_RELEASE)
     * to decrement the reference count. Release memory ordering is suitable
     * for scenarios where a thread releases a lock or a resource, and other
     * threads that depend on that resource need to see the effects of the
     * release.
     * If the reference count reaches zero after decrementing, and a deleter
     * function is set, the deleter function will be invoked.
     */
    void DecRef() {
        if (__atomic_fetch_sub(&strong_ref_count_, 1, __ATOMIC_RELEASE) == 1) {
            // only acquire when we need to call deleter
            // in this case we need to ensure all previous writes are visible
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (deleter_ != nullptr) {
                deleter_(this);
            }
        }
    }

    void DecWeakRef() {
        __atomic_fetch_sub(&weak_ref_count_, 1, __ATOMIC_RELEASE);
    }

    /*! \brief Reference counter of the object. */
    uint32_t strong_ref_count_;

    uint32_t weak_ref_count_;

    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    FDeleter deleter_;

    template<typename T>
    friend class ObjectPtr;

    template<typename U>
    friend class WeakObjectPtr;

    friend struct details::ObjectUnsafe;
};

/*!
 * \brief Null type of the given type.
 *
 * \tparam T The type for which the null type is defined.
 */
template<typename T>
class NullTypeOf final : public T {
    NullTypeOf() = default;

public:
    static T* singleton() noexcept {
        static NullTypeOf inst;
        return &inst;
    }
};

template<typename T>
using null_type_of = NullTypeOf<T>;

/*!
 * \brief Tag type to indicate that reference count should not be incremented.
 */
struct DoNotIncRefCountTag final {};

/*!
 * \brief Smart pointer for reference-counted objects.
 *
 * \tparam T The type of the object being managed.
 *
 * \note The ObjectPtr class provides a way to manage the lifetime of reference-counted objects.
 * It ensures that objects are properly reference-counted and destroyed when no longer needed.
 * The ObjectPtr class follows the RAII (Resource Acquisition Is Initialization) idiom,
 * guaranteeing that resources are released when the reference count goes to zero.
 *
 */
template<typename T>
class ObjectPtr final {
    using element_type = T;
    using null_type = null_type_of<T>;

    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(std::is_base_of_v<T, std::remove_pointer_t<decltype(null_type::singleton())>>,
                  "NullType::singleton() must return a element_type* pointer");

public:
    /*!
     * \brief Default constructor.
     *
     * \note Initializes the ObjectPtr with a null pointer and does not increment the reference count.
     */
    ObjectPtr() noexcept : ObjectPtr(null_type::singleton(), DoNotIncRefCountTag()) {}

    /*!
     * \brief Constructor from nullptr.
     *
     * \note Initializes the ObjectPtr with a null pointer and does not increment the reference count.
     */
    ObjectPtr(std::nullptr_t) noexcept : ObjectPtr(null_type::singleton(), DoNotIncRefCountTag()) {}// NOLINT

    /*!
     * \brief Constructor from unique_ptr.
     *
     * \param other The unique_ptr to transfer ownership from.
     *
     * \note Initializes the ObjectPtr with the pointer from the unique_ptr and increments the reference count.
     */
    explicit ObjectPtr(std::unique_ptr<T> other) noexcept : ObjectPtr(other.release()) {}

    /*!
     * \brief Constructor from raw pointer, do not increment reference count.
     */
    ObjectPtr(T* ptr, DoNotIncRefCountTag) noexcept : ptr_(ptr) {}

    /*!
     * \brief Copy constructor.
     */
    ObjectPtr(const ObjectPtr& other) : ptr_(other.ptr_) {
        retain();
    }

    /*!
     * \brief Move constructor.
     */
    ObjectPtr(ObjectPtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = null_type::singleton();
    }

    /*!
     * \brief Copy constructor.
     *
     * \param other The other ObjectPtr.
     */
    template<typename Derived>
    ObjectPtr(const ObjectPtr<Derived>& other)// NOLINT
        : ptr_(other.ptr_ == null_type_of<Derived>::singleton() ? null_type::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        retain();
    }

    /*!
     * \brief Move constructor.
     */
    template<typename Derived>
    ObjectPtr(ObjectPtr<Derived>&& other) noexcept// NOLINT
        : ptr_(other.ptr_ == null_type_of<Derived>::singleton() ? null_type::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        other.ptr_ = null_type_of<Derived>::singleton();
    }

    /*!
     * \brief Copy assignment operator.
     */
    template<typename Derived>
    ObjectPtr& operator=(const ObjectPtr<Derived>& rhs) & {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        ObjectPtr(rhs).swap(*this);
        return *this;
    }

    /*!
     * \brief Move assignment operator.
     */
    template<typename Derived>
    ObjectPtr& operator=(ObjectPtr<Derived>&& rhs) & noexcept {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        ObjectPtr(std::move(rhs)).swap(*this);
        return *this;
    }

    ObjectPtr& operator=(const ObjectPtr& rhs) & {
        ObjectPtr(rhs).swap(*this);
        return *this;
    }

    ObjectPtr& operator=(ObjectPtr&& rhs) & noexcept {
        ObjectPtr(std::move(rhs)).swap(*this);
        return *this;
    }

    ~ObjectPtr() noexcept {
        reset();
    }

    void reset() {
        if (defined()) {
            ptr_->DecRef();
        }
        ptr_ = null_type::singleton();
    }

    void swap(ObjectPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    NODISCARD bool defined() const {
        return ptr_ != null_type::singleton();
    }

    T* get() const noexcept {
        return ptr_;
    }

    T& operator*() const noexcept {
        return *ptr_;
    }

    T* operator->() const noexcept {
        return ptr_;
    }

    operator bool() const noexcept {// NOLINT
        return defined();
    }

    NODISCARD uint32_t use_count() const noexcept {
        return ptr_->use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    /**
   * Returns an owning (!) pointer to the underlying object and makes the
   * ObjectPtr instance invalid. That means the refcount is not decreased.
   * Must put the returned pointer back into an ObjectPtr using
   * ObjectPtr::reclaim(ptr) to properly destruct it.
   *
   * @return The underlying pointer.
   */
    T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = null_type::singleton();
        return tmp;
    }

    static ObjectPtr reclaim(T* ptr) {
        return ObjectPtr(ptr, DoNotIncRefCountTag());
    }

    bool operator==(const ObjectPtr& rhs) const noexcept {
        return ptr_ == rhs.ptr_;
    }

    bool operator!=(const ObjectPtr& rhs) const noexcept {
        return ptr_ != rhs.ptr_;
    }

private:
    void retain() {
        if (defined()) {
            CHECK(ptr_->use_count() > 0)
                    << "ObjectPtr must be copy constructed with an object with ref_count_ > 0";
            ptr_->IncRef();
        }
    }

    explicit ObjectPtr(T* ptr) : ObjectPtr(ptr, DoNotIncRefCountTag()) {
        if (defined()) {
            CHECK(ptr_->use_count() == 0)
                    << "ObjectPtr must be constructed with a null_type or an object with ref_count_ == 0";
            ptr_->IncRef();
        }
    }

    T* ptr_;

    template<typename Derived>
    friend class details::ObjectAllocatorBase;

    template<typename T2>
    friend class ObjectPtr;
};

template<typename T>
class WeakObjectPtr final {
    using element_type = T;
    using null_type = null_type_of<T>;

    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(std::is_base_of_v<T, std::remove_pointer_t<decltype(null_type::singleton())>>,
                  "NullType::singleton() must return a element_type* pointer");

public:
    NODISCARD bool defined() const noexcept {
        return ptr_ != null_type::singleton();
    }

    NODISCARD uint32_t use_count() const noexcept {
        return ptr_->use_count();
    }

    NODISCARD uint32_t weak_use_count() const noexcept {
        return ptr_->weak_use_count();
    }

    NODISCARD bool expired() const noexcept {
        return use_count() == 0;
    }

    void reset() {
        if (defined()) {
            ptr_->DecWeakRef();
        }
        ptr_ = null_type::singleton();
    }

private:
    explicit WeakObjectPtr(T* ptr) : ptr_(ptr) {}


    T* ptr_;

    template<typename T2>
    friend class WeakObjectPtr;
};

namespace details {
struct ObjectUnsafe {
    static void IncRef(Object* ptr) {
        if (ptr) {
            ptr->IncRef();
        }
    }

    static void DecRef(Object* ptr) {
        if (ptr) {
            ptr->DecRef();
        }
    }

    static void IncWeakRef(Object* ptr) {
        if (ptr) {
            ptr->IncWeakRef();
        }
    }

    static void DecWeakRef(Object* ptr) {
        if (ptr) {
            ptr->DecWeakRef();
        }
    }
};
}// namespace details

template<typename T>
void swap(ObjectPtr<T>& lhs, ObjectPtr<T>& rhs) noexcept {
    lhs.swap(rhs);
}

// To allow ObjectPtr inside std::map or std::set, we need operator<
template<typename T1, typename T2>
bool operator<(const ObjectPtr<T1>& lhs, const ObjectPtr<T2>& rhs) noexcept {
    return lhs.get() < rhs.get();
}

template<typename T1, typename T2>
bool operator==(const ObjectPtr<T1>& lhs, const ObjectPtr<T2>& rhs) noexcept {
    return lhs.get() == rhs.get();
}

template<typename T>
bool operator==(const ObjectPtr<T>& lhs, std::nullptr_t) noexcept {
    return lhs.get() == nullptr;
}

template<typename T>
bool operator==(std::nullptr_t, const ObjectPtr<T>& rhs) noexcept {
    return nullptr == rhs.get();
}

template<typename T1, typename T2>
bool operator!=(const ObjectPtr<T1>& lhs, const ObjectPtr<T2>& rhs) noexcept {
    return lhs.get() != rhs.get();
}

template<typename T>
bool operator!=(const ObjectPtr<T>& lhs, std::nullptr_t) noexcept {
    return lhs.get() != nullptr;
}

template<typename T>
bool operator!=(std::nullptr_t, const ObjectPtr<T>& rhs) noexcept {
    return nullptr != rhs.get();
}

namespace details {
// allocate object
template<typename Derived>
class ObjectAllocatorBase {
public:
    template<typename T, typename... Args>
    ObjectPtr<T> make_object(Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "make can only be used to create Object");
        using Handler = Derived::template Handler<T>;
        T* ptr = Handler::allocate(std::forward<Args>(args)...);
        ptr->SetDeleter(Handler::deleter);
        ObjectUnsafe::IncWeakRef(ptr);
        return ObjectPtr<T>(ptr);
    }

    template<typename T, typename ElemType, typename... Args>
    ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "make can only be used to create Object");
        using Handler = Derived::template ArrayHandler<T, ElemType>;
        T* ptr = Handler::allocate(num_elems, std::forward<Args>(args)...);
        ptr->SetDeleter(Handler::deleter);
        ObjectUnsafe::IncWeakRef(ptr);
        return ObjectPtr<T>(ptr);
    }
};

class ObjectAllocator : public ObjectAllocatorBase<ObjectAllocator> {
public:
    template<typename T>
    struct Handler {
        struct alignas(T) StorageType {
            char data[sizeof(T)];
        };

        template<typename... Args>
        static T* allocate(Args&&... args) {
            auto* data = new StorageType();
            new (data) T(std::forward<Args>(args)...);
            return reinterpret_cast<T*>(data);
        }

        static void deleter(Object* ptr) {
            T* p = static_cast<T*>(ptr);
            p->T::~T(); // release source
            delete reinterpret_cast<StorageType*>(p); // free memory
        }
    };

    template<typename ObjType, typename ElemType>
    struct ArrayHandler {
        using StorageType = std::aligned_storage_t<sizeof(ObjType), alignof(ObjType)>;
        // for now, only support elements that aligns with array header.
        static_assert(alignof(ObjType) % alignof(ElemType) == 0 && sizeof(ObjType) % alignof(ElemType) == 0,
                      "element alignment constraint");

        template<typename... Args>
        static ObjType* allocate(size_t num_elems, Args&&... args) {
            size_t storage_size = sizeof(StorageType);
            size_t required_size = sizeof(ObjType) + num_elems * sizeof(ElemType);
            size_t num_storage_slots = (required_size + storage_size - 1) / storage_size;
            auto* data = new StorageType[num_storage_slots];
            new (data) ObjType(std::forward<Args>(args)...);
            return reinterpret_cast<ObjType*>(data);
        }

        static void deleter(Object* ptr) {
            auto* p = static_cast<ObjType*>(ptr);
            p->ObjType::~ObjType();
            delete[] reinterpret_cast<StorageType*>(p);
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

namespace std {
template<typename T>
struct hash<aethermind::ObjectPtr<T>> {
    size_t operator()(const aethermind::ObjectPtr<T>& ptr) const {
        return std::hash<T*>()(ptr.get());
    }
};
}// namespace std

#endif//AETHERMIND_OBJECT_H
