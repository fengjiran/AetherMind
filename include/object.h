//
// Created by 赵丹 on 2025/8/19.
//

#ifndef AETHERMIND_OBJECT_H
#define AETHERMIND_OBJECT_H

#include "c_api.h"
#include "utils/hash.h"

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
 * \brief Flags to indicate the type of pointer that decremented the reference count.
 */
enum DeleterFlag : uint8_t {
    kStrongPtrMask = 0x01,
    kWeakPtrMask = 0x02,
    kBothPtrMask = kStrongPtrMask | kWeakPtrMask,
};

/*!
 * \brief Base class for reference-counted objects.
 *
 * \note Objects of this class are reference-counted, meaning that each object
 * has a reference count that tracks the number of references to the object.
 * When the reference count reaches zero, the object is automatically destroyed.
 */
class Object {
public:
    /*!
    * \brief Default constructor, initializes a reference-counted object.
    *
    * \note Creates a new Object instance with reference count initialized to 0
    * and deleter set to nullptr.
    * Newly created objects require manual reference count management,
    * typically wrapped using ObjectPtr.
    */
    Object();

    virtual ~Object() = default;

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
    NODISCARD uint32_t use_count() const;

    /*!
     * \brief Get the current weak reference count of the object.
     *
     * \note Uses atomic operation with relaxed memory ordering (__ATOMIC_RELAXED)
     * to load the weak reference count value. Relaxed memory ordering is suitable
     * for performance-sensitive scenarios where strict memory synchronization
     * is not required.
     *
     * \return The current weak reference count value of the object
     */
    NODISCARD uint32_t weak_use_count() const;

    /*!
     * \brief Check if the object has a unique reference count.
     *
     * \return true if the object has a reference count of 1, false otherwise.
     */
    NODISCARD bool unique() const;

    /*!
     * \brief Set the deleter function for the object.
     *
     * \param deleter The deleter function to be invoked when the reference count reaches zero.
     */
    void SetDeleter(FObjectDeleter deleter);

    NODISCARD virtual bool IsNullTypePtr() const {
        return false;
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
    void IncRef();

    /*!
     * \brief Increment the weak reference count of the object.
     *
     * \note Uses atomic operation with relaxed memory ordering (__ATOMIC_RELAXED)
     * to increment the weak reference count. Relaxed memory ordering is suitable
     * for performance-sensitive scenarios where strict memory synchronization
     * is not required.
     */
    void IncWeakRef();

    /*!
     * \brief Decrement the reference count of the object.
     *
     * \note Uses atomic operation with release memory ordering (__ATOMIC_RELEASE)
     * to decrement the reference count. Release memory ordering is suitable
     * for scenarios where a thread releases a lock or a resource, and other
     * threads that depend on that resource need to see the effects of the
     * release.
     * If the strong reference count reaches zero after decrementing and the 
     * weak reference count is one,  and a deleter function is set, the deleter 
     * function will be invoked to destroy the object and free its memory. If the 
     * weak reference count is greater than one, the deleter function will be
     * invoked to destroy the object only.
     */
    void DecRef();

    /*!
     * \brief Decrement the weak reference count of the object.
     *
     * \note Uses atomic operation with release memory ordering (__ATOMIC_RELEASE)
     * to decrement the weak reference count. Release memory ordering is suitable
     * for scenarios where a thread releases a lock or a resource, and other
     * threads that depend on that resource need to see the effects of the
     * release.
     * If the weak reference count reaches zero after decrementing, and a deleter
     * function is set, the deleter function will be invoked to free the memory of
     * the object.
     */
    void DecWeakRef();

    /*! \brief Whether the weak pointer can be promoted to a strong pointer.
     *
     * \return true if the weak pointer can be promoted to a strong pointer, false otherwise.
     *
     * \note This method uses a compare-and-swap (CAS) operation to safely
     * increment the strong reference count if it is greater than zero.
     * This ensures that the promotion is thread-safe and avoids race conditions
     * when multiple threads try to promote a weak pointer to a strong pointer
     * at the same time.
     */
    bool TryPromoteWeakPtr();

    /*! \brief header field that is the common prefix of all objects */
    ObjectHeader header_{};

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
template<typename T, typename = std::enable_if_t<std::is_constructible_v<T>>>
class NullTypeOf final : public T {
    NullTypeOf() = default;

public:
    static T* singleton() noexcept {
        static NullTypeOf inst;
        return &inst;
    }

    NODISCARD bool IsNullTypePtr() const override {
        return true;
    }
};

inline bool IsNullTypePtr(const Object* ptr) {
    return ptr == nullptr ? true : ptr->IsNullTypePtr();
}

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
    using null_type = NullTypeOf<T>;
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
     * \note Initializes the ObjectPtr with the pointer from the other type ObjectPtr.
     * \param other The other ObjectPtr.
     */
    template<typename Derived>
    ObjectPtr(const ObjectPtr<Derived>& other)// NOLINT
        : ptr_(other.ptr_ == NullTypeOf<Derived>::singleton() ? null_type::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        retain();
    }

    /*!
     * \brief Move constructor.
     * \note Initializes the ObjectPtr with the pointer from the other type ObjectPtr.
     * \param other The other ObjectPtr.
     */
    template<typename Derived>
    ObjectPtr(ObjectPtr<Derived>&& other) noexcept// NOLINT
        : ptr_(other.ptr_ == NullTypeOf<Derived>::singleton() ? null_type::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        other.ptr_ = NullTypeOf<Derived>::singleton();
    }

    /*!
     * \brief Copy assignment operator.
     * \note Uses the copy-and-swap idiom to provide strong exception safety.
     * \param rhs The right-hand side ObjectPtr to copy from.
     * \return A reference to this ObjectPtr.
     */
    template<typename Derived>
    ObjectPtr& operator=(const ObjectPtr<Derived>& rhs) & {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        ObjectPtr(rhs).swap(*this);
        return *this;
    }

    /*!
     * \brief Move assignment operator.
     * \note Uses the copy-and-swap idiom to provide strong exception safety.
     * \param rhs The right-hand side ObjectPtr to move from.
     * \return A reference to this ObjectPtr.
     */
    template<typename Derived>
    ObjectPtr& operator=(ObjectPtr<Derived>&& rhs) & noexcept {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        ObjectPtr(std::move(rhs)).swap(*this);
        return *this;
    }

    /*!
     * \brief Copy assignment operator.
     * \note Uses the copy-and-swap idiom to provide strong exception safety.
     * \param rhs The right-hand side ObjectPtr to copy from.
     * \return A reference to this ObjectPtr.
     */
    ObjectPtr& operator=(const ObjectPtr& rhs) & {
        ObjectPtr(rhs).swap(*this);
        return *this;
    }

    /*!
     * \brief Move assignment operator.
     * \note Uses the copy-and-swap idiom to provide strong exception safety.
     * \param rhs The right-hand side ObjectPtr to move from.
     * \return A reference to this ObjectPtr.
     */
    ObjectPtr& operator=(ObjectPtr&& rhs) & noexcept {
        ObjectPtr(std::move(rhs)).swap(*this);
        return *this;
    }

    /*!
     * \brief Destructor.
     * \note Decrements the reference count of the underlying object.
     */
    ~ObjectPtr() noexcept {
        reset();
    }

    /*!
     * \brief Resets the ObjectPtr to the null type pointer.
     * \note Decrements the reference count of the underlying object if it is not the null type pointer.
     */
    void reset() {
        if (defined()) {
            ptr_->DecRef();
        }
        ptr_ = null_type::singleton();
    }

    /*!
     * \brief Swaps the contents of this ObjectPtr with the contents of another ObjectPtr.
     * \param other The other ObjectPtr to swap with.
     */
    void swap(ObjectPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    /*!
     * \brief Returns whether the ObjectPtr is defined (i.e., not null type pointer).
     * \return True if the ObjectPtr is defined, false otherwise.
     */
    NODISCARD bool defined() const {
        return ptr_ != null_type::singleton();
    }

    /*!
     * \brief Returns a raw pointer to the underlying object.
     * \return A raw pointer to the underlying object.
     */
    T* get() const noexcept {
        return ptr_;
    }

    /*!
     * \brief Dereference operator.
     * \return A reference to the underlying object.
     */
    T& operator*() const noexcept {
        return *ptr_;
    }

    /*!
     * \brief Member access operator.
     * \return A pointer to the underlying object.
     */
    T* operator->() const noexcept {
        return ptr_;
    }

    /*!
     * \brief Boolean conversion operator.
     * \return True if the ObjectPtr is defined, false otherwise.
     */
    operator bool() const noexcept {// NOLINT
        return defined();
    }

    /*!
     * \brief Returns the reference count of the underlying object.
     * \return The reference count of the underlying object.
     */
    NODISCARD uint32_t use_count() const noexcept {
        return ptr_->use_count();
    }

    /*!
     * \brief Returns whether the ObjectPtr has a unique reference to the underlying object.
     * \return True if the ObjectPtr has a unique reference to the underlying object, false otherwise.
     */
    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    /*!
     * \brief Returns an owning (!) pointer to the underlying object and makes the
     * ObjectPtr instance invalid. That means the refcount is not decreased.
     * Must put the returned pointer back into an ObjectPtr using
     * ObjectPtr::reclaim(ptr) to properly destruct it.
     * \return The underlying pointer.
     */
    T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = null_type::singleton();
        return tmp;
    }

    /*!
     * \brief Recreates an ObjectPtr from a raw pointer.
     * \note The refcount of the object is not incremented.
     * \param ptr The raw pointer to recreate the ObjectPtr from.
     * \return The recreated ObjectPtr.
     */
    static ObjectPtr reclaim(T* ptr) {
        return ObjectPtr(ptr, DoNotIncRefCountTag());
    }

    /*!
     * \brief Equality operator.
     * \param rhs The right-hand side ObjectPtr to compare with.
     * \return True if the ObjectPtrs are equal, false otherwise.
     */
    bool operator==(const ObjectPtr& rhs) const noexcept {
        return ptr_ == rhs.ptr_;
    }

    /*!
     * \brief Inequality operator.
     * \param rhs The right-hand side ObjectPtr to compare with.
     * \return True if the ObjectPtrs are not equal, false otherwise.
     */
    bool operator!=(const ObjectPtr& rhs) const noexcept {
        return ptr_ != rhs.ptr_;
    }

private:
    /*!
     * \brief Retains the underlying object by incrementing its reference count.
     */
    void retain() {
        if (defined()) {
            CHECK(ptr_->use_count() > 0)
                    << "ObjectPtr must be copy constructed with an object with ref_count_ > 0";
            ptr_->IncRef();
        }
    }

    /*!
     * \brief Constructor.
     * \param ptr The raw pointer to construct the ObjectPtr from.
     * \note The refcount of the object is not incremented if the pointer is null type pointer.
     */
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

/*!
 * \brief WeakObjectPtr is a weak reference to an Object.
 * \note It does not affect the reference count of the Object.
 * \tparam T The type of the object to reference.
 */
template<typename T>
class WeakObjectPtr final {
    using element_type = T;
    using null_type = NullTypeOf<T>;

    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(std::is_base_of_v<T, std::remove_pointer_t<decltype(null_type::singleton())>>,
                  "NullType::singleton() must return a element_type* pointer");

public:
    WeakObjectPtr() noexcept : ptr_(null_type::singleton()) {}

    explicit WeakObjectPtr(const ObjectPtr<T>& other) : WeakObjectPtr(other.get()) {
        retain();
    }

    WeakObjectPtr(const WeakObjectPtr& other) : ptr_(other.ptr_) {
        retain();
    }

    WeakObjectPtr(WeakObjectPtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = null_type::singleton();
    }

    template<typename Derived>
    WeakObjectPtr(const WeakObjectPtr<Derived>& other)// NOLINT
        : ptr_(other.ptr_ == NullTypeOf<Derived>::singleton() ? null_type::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        retain();
    }

    template<typename Derived>
    WeakObjectPtr(WeakObjectPtr<Derived>&& other) noexcept// NOLINT
        : ptr_(other.ptr_ == NullTypeOf<Derived>::singleton() ? null_type::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        other.ptr_ = NullTypeOf<Derived>::singleton();
    }

    WeakObjectPtr& operator=(const WeakObjectPtr& rhs) & {
        WeakObjectPtr(rhs).swap(*this);
        return *this;
    }

    WeakObjectPtr& operator=(WeakObjectPtr&& rhs) & noexcept {
        WeakObjectPtr(std::move(rhs)).swap(*this);
        return *this;
    }

    template<typename Derived>
    WeakObjectPtr& operator=(const WeakObjectPtr<Derived>& rhs) & {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        WeakObjectPtr(rhs).swap(*this);
        return *this;
    }

    template<typename Derived>
    WeakObjectPtr& operator=(WeakObjectPtr<Derived>&& rhs) & noexcept {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        WeakObjectPtr(std::move(rhs)).swap(*this);
        return *this;
    }

    ~WeakObjectPtr() noexcept {
        reset();
    }

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

    ObjectPtr<T> lock() const noexcept {
        if (ptr_->TryPromoteWeakPtr()) {
            return ObjectPtr<T>(ptr_, DoNotIncRefCountTag());
        }
        return nullptr;
    }

    T* release() noexcept {
        auto* tmp = ptr_;
        ptr_ = null_type::singleton();
        return tmp;
    }

    static WeakObjectPtr reclaim(T* ptr) {
        CHECK(ptr == null_type::singleton() ||
              ptr->weak_use_count() > 1 ||
              (ptr->use_count() == 0 && ptr->weak_use_count() > 0));
        return WeakObjectPtr(ptr);
    }

    void swap(WeakObjectPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    void reset() {
        if (defined()) {
            ptr_->DecWeakRef();
        }
        ptr_ = null_type::singleton();
    }

    T* unsafe_get() const noexcept {
        return ptr_;
    }

private:
    explicit WeakObjectPtr(T* ptr) : ptr_(ptr) {}

    void retain() {
        if (defined()) {
            CHECK(ptr_->weak_use_count() > 0)
                    << "ObjectPtr must be copy constructed with an object with weak_ref_count_ > 0";
            ptr_->IncWeakRef();
        }
    }

    T* ptr_;

    template<typename T2>
    friend class WeakObjectPtr;

    template<typename T1, typename T2>
    friend bool operator<(const WeakObjectPtr<T1>&, const WeakObjectPtr<T2>&) noexcept;

    template<typename T1, typename T2>
    friend bool operator==(const WeakObjectPtr<T1>&, const WeakObjectPtr<T2>&) noexcept;
};

class ObjectRef {
    ObjectPtr<Object> ptr_;
};

namespace details {
struct ObjectUnsafe {
    static ObjectHeader* GetHeader(const Object* src) {
        return const_cast<ObjectHeader*>(&src->header_);
    }

    static void IncRefObjectHandle(const ObjectHandle handle) {
        if (handle) {
            static_cast<Object*>(handle)->IncRef();
        }
    }

    static void DecRefObjectHandle(const ObjectHandle handle) {
        if (handle) {
            static_cast<Object*>(handle)->DecRef();
        }
    }

    static void IncWeakRefObjectHandle(const ObjectHandle handle) {
        if (handle) {
            static_cast<Object*>(handle)->IncWeakRef();
        }
    }

    static void DecWeakRefObjectHandle(const ObjectHandle handle) {
        if (handle) {
            static_cast<Object*>(handle)->DecWeakRef();
        }
    }
};

/*!
 * \brief TypeTraits that removes const and reference keywords.
 * \tparam T the original type
 */
// template<typename T>
// using TypeTraitsNoCR = TypeTraits<std::remove_const_t<std::remove_reference_t<T>>>;


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

template<typename T1, typename T2>
void swap(WeakObjectPtr<T1>& lhs, WeakObjectPtr<T2>& rhs) noexcept {
    lhs.swap(rhs);
}

template<typename T1, typename T2>
bool operator<(const WeakObjectPtr<T1>& lhs, const WeakObjectPtr<T2>& rhs) noexcept {
    return lhs.ptr_ < rhs.ptr_;
}

template<typename T1, typename T2>
bool operator==(const WeakObjectPtr<T1>& lhs, const WeakObjectPtr<T2>& rhs) noexcept {
    return lhs.ptr_ == rhs.ptr_;
}

template<typename T1, typename T2>
bool operator!=(const WeakObjectPtr<T1>& lhs, const WeakObjectPtr<T2>& rhs) noexcept {
    return !(lhs == rhs);
}

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
        static_assert(std::is_base_of_v<Object, T>, "make can only be used to create Object");
        using Handler = Derived::template Handler<T>;
        T* ptr = Handler::allocate(std::forward<Args>(args)...);
        ptr->SetDeleter(Handler::deleter);
        ObjectUnsafe::IncWeakRefObjectHandle(ptr);
        return ObjectPtr<T>(ptr);
    }

    template<typename T, typename ElemType, typename... Args>
    ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "make can only be used to create Object");
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
        static_assert(alignof(ObjType) % alignof(ElemType) == 0 && sizeof(ObjType) % alignof(ElemType) == 0,
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

template<typename T>
struct hash<aethermind::WeakObjectPtr<T>> {
    size_t operator()(const aethermind::WeakObjectPtr<T>& ptr) const {
        return std::hash<T*>()(ptr.unsafe_get());
    }
};
}// namespace std

#endif//AETHERMIND_OBJECT_H
