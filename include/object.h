/// Intrusive object ownership primitives and smart-handle wrappers.
///
/// This header defines the refcounted `Object` base class together with the
/// owning/non-owning handle types built on top of it. Empty handles are
/// represented by a per-type sentinel so callers should use handle-level APIs
/// such as `defined()` or `get_or_null()` instead of raw pointer null checks.

#ifndef AETHERMIND_OBJECT_H
#define AETHERMIND_OBJECT_H

#include "c_api.h"
#include "macros.h"
#include "utils/logging.h"

namespace aethermind {

namespace details {
struct ObjectUnsafe;

template<typename Derived>
class ObjectAllocatorBase;

}// namespace details

/// Flags to indicate the type of pointer that decremented the reference count.
enum DeleterFlag : uint8_t {
    kStrongPtrMask = 0x01,
    kWeakPtrMask = 0x02,
    kBothPtrMask = kStrongPtrMask | kWeakPtrMask,
};


/// Base class for intrusive strong/weak reference-counted objects.
///
/// `Object` stores the common refcount header used by `ObjectPtr` and
/// `WeakObjectPtr`. Heap-allocated instances are expected to install a deleter
/// through the allocator path. Sentinel instances such as
/// `EmptyObjectSentinel<T>` are process-lifetime objects and intentionally do
/// not participate in normal destruction.
class Object {
public:
    /// Initializes a detached object header.
    ///
    /// Newly constructed objects start with zero strong/weak references and no
    /// deleter. Production objects are typically wrapped by `ObjectPtr` through
    /// the allocator path, which installs the deleter and seeds the initial
    /// weak reference.
    Object();

    virtual ~Object() = default;

    AM_NODISCARD uint32_t use_count() const {
        return __atomic_load_n(&header_.strong_ref_count, __ATOMIC_RELAXED);
    }

    AM_NODISCARD uint32_t weak_use_count() const {
        return __atomic_load_n(&header_.weak_ref_count, __ATOMIC_RELAXED);
    }

    AM_NODISCARD bool unique() const {
        return use_count() == 1;
    }

    void SetDeleter(FObjectDeleter deleter) {
        header_.deleter = deleter;
    }

    AM_NODISCARD virtual bool IsEmptyObjectSentinel() const {
        return false;
    }

private:
    void IncRef() {
        __atomic_fetch_add(&header_.strong_ref_count, 1, __ATOMIC_RELAXED);
    }

    void IncWeakRef() {
        __atomic_fetch_add(&header_.weak_ref_count, 1, __ATOMIC_RELAXED);
    }

    /// Decrements the strong reference count and runs the appropriate teardown.
    ///
    /// When the last strong reference drops, the deleter either destroys and
    /// frees the object (`kBothPtrMask`) or performs a two-stage teardown when a
    /// weak handle still keeps the control block alive.
    void DecRef();

    void DecWeakRef();

    /// Attempts to promote a weak handle to a new strong reference.
    ///
    /// Promotion succeeds only while the strong count is still non-zero. The CAS
    /// loop prevents racing promotions from resurrecting an object after strong
    /// teardown has begun.
    bool TryPromoteWeakPtr();

    /// Common intrusive refcount header shared by all object subtypes.
    ObjectHeader header_{};

    template<typename T>
    friend class ObjectPtr;

    template<typename U>
    friend class WeakObjectPtr;

    friend struct details::ObjectUnsafe;
};

/// Process-lifetime sentinel used as the internal empty state of `ObjectPtr<T>`.
///
/// This is not a business-level Null Object pattern. `ObjectPtr` and
/// `WeakObjectPtr` store this singleton instead of raw `nullptr` so that empty
/// handles keep a stable, type-correct sentinel address. Callers should use
/// `defined()`, `operator bool()`, or `get_or_null()` instead of raw pointer
/// non-null checks.
template<typename T>
    requires std::default_initializable<T>
class EmptyObjectSentinel final : public T {
    EmptyObjectSentinel() = default;

public:
    static T* singleton() noexcept {
        static EmptyObjectSentinel inst;
        return &inst;
    }

    AM_NODISCARD bool IsEmptyObjectSentinel() const override {
        return true;
    }
};

inline bool IsEmptyObjectSentinel(const Object* ptr) {
    return ptr == nullptr ? true : ptr->IsEmptyObjectSentinel();
}

template<typename T>
    requires std::derived_from<T, Object>
bool IsDefinedObjectPtr(const T* ptr) noexcept {
    return ptr != nullptr && !IsEmptyObjectSentinel(ptr);
}

template<typename T>
concept is_valid_object_type = requires {
    requires true;
    // requires std::is_base_of_v<Object, T>;
    requires std::derived_from<T, Object>;
    typename EmptyObjectSentinel<T>;
    { EmptyObjectSentinel<T>::singleton() } -> std::same_as<T*>;
};

/// Tag type to indicate that reference count should not be incremented.
struct DoNotIncRefCountTag final {};

/// Owning intrusive smart pointer for `Object`-derived instances.
///
/// Empty `ObjectPtr` values store `EmptyObjectSentinel<T>::singleton()` rather
/// than raw `nullptr`. Sentinel-aware APIs such as `defined()`, `empty()`,
/// `operator bool()`, and `get_or_null()` should be used to observe emptiness.
template<typename T>
class ObjectPtr final {
    using element_type = T;
    using empty_sentinel_type = EmptyObjectSentinel<T>;
    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(std::is_same_v<T, std::remove_pointer_t<decltype(empty_sentinel_type::singleton())>>,
                  "EmptyObjectSentinel::singleton() must return a element_type* pointer");

public:
    ObjectPtr() noexcept : ObjectPtr(empty_sentinel_type::singleton(), DoNotIncRefCountTag()) {}

    ObjectPtr(std::nullptr_t) noexcept : ObjectPtr(empty_sentinel_type::singleton(), DoNotIncRefCountTag()) {}// NOLINT

    ObjectPtr(T* ptr, DoNotIncRefCountTag) noexcept : ptr_(ptr) {}

    ObjectPtr(const ObjectPtr& other) : ptr_(other.ptr_) {
        retain();
    }

    ObjectPtr(ObjectPtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = empty_sentinel_type::singleton();
    }

    template<typename Derived>
    ObjectPtr(const ObjectPtr<Derived>& other)// NOLINT
        : ptr_(RemapEmptyObjectSentinel(other.ptr_)) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        retain();
    }

    template<typename Derived>
    ObjectPtr(ObjectPtr<Derived>&& other) noexcept// NOLINT
        : ptr_(RemapEmptyObjectSentinel(other.ptr_)) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        other.ptr_ = EmptyObjectSentinel<Derived>::singleton();
    }

    template<typename Derived>
    ObjectPtr& operator=(const ObjectPtr<Derived>& rhs) & {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        ObjectPtr(rhs).swap(*this);
        return *this;
    }

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
        ptr_ = empty_sentinel_type::singleton();
    }

    void swap(ObjectPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    AM_NODISCARD bool defined() const noexcept {
        return ptr_ != empty_sentinel_type::singleton();
    }

    AM_NODISCARD bool empty() const noexcept {
        return ptr_ == empty_sentinel_type::singleton();
    }

    AM_NODISCARD bool is_sentinel() const noexcept {
        return ptr_ == empty_sentinel_type::singleton();
    }

    /// Returns the stored raw pointer.
    ///
    /// Empty handles return the sentinel pointer, not raw `nullptr`. Use
    /// `get_or_null()` when raw-pointer null semantics are required.
    AM_NODISCARD T* get() const noexcept {
        return ptr_;
    }

    AM_NODISCARD T* get_or_null() const noexcept {
        return defined() ? ptr_ : nullptr;
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

    AM_NODISCARD uint32_t use_count() const noexcept {
        return ptr_->use_count();
    }

    AM_NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    /// Releases the handle without decrementing the strong reference count.
    ///
    /// The caller becomes responsible for eventually re-wrapping the returned
    /// pointer with `reclaim()` or otherwise continuing intrusive ownership. An
    /// empty handle returns `nullptr`.
    T* release() noexcept {
        if (!defined()) {
            ptr_ = empty_sentinel_type::singleton();
            return nullptr;
        }

        T* tmp = ptr_;
        ptr_ = empty_sentinel_type::singleton();
        return tmp;
    }

    /// Rebuilds an owning handle from a raw pointer returned by `release()`.
    ///
    /// This path assumes the caller already owns the outstanding strong
    /// reference and therefore does not increment the refcount.
    static ObjectPtr reclaim(T* ptr) {
        return ptr ? ObjectPtr(ptr, DoNotIncRefCountTag()) : ObjectPtr(nullptr);
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
    void retain() {
        if (defined()) {
            AM_CHECK(ptr_->use_count() > 0,
                     "ObjectPtr must be copy constructed with an object with ref_count_ > 0");
            ptr_->IncRef();
        }
    }

    explicit ObjectPtr(T* ptr) : ObjectPtr(ptr, DoNotIncRefCountTag()) {
        if (defined()) {
            AM_CHECK(ptr_->use_count() == 0,
                     "ObjectPtr must be constructed with a null_type or an object with ref_count_ == 0");
            ptr_->IncRef();
        }
    }

    template<typename Derived>
    static T* RemapEmptyObjectSentinel(Derived* ptr) noexcept {
        if (ptr == EmptyObjectSentinel<Derived>::singleton()) {
            return empty_sentinel_type::singleton();
        }
        return ptr;
    }

    T* ptr_;

    template<typename Derived>
    friend class details::ObjectAllocatorBase;

    template<typename U>
    friend class ObjectPtr;
};

/// Weak intrusive handle for `Object`-derived instances.
///
/// Like `ObjectPtr`, the empty state is represented by `EmptyObjectSentinel<T>`.
/// Weak handles do not contribute to the strong refcount and must be promoted
/// through `lock()` before dereference.
template<typename T>
class WeakObjectPtr final {
    using element_type = T;
    using empty_sentinel_type = EmptyObjectSentinel<T>;

    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(std::is_base_of_v<T, std::remove_pointer_t<decltype(empty_sentinel_type::singleton())>>,
                  "EmptyObjectSentinel::singleton() must return a element_type* pointer");

public:
    WeakObjectPtr() noexcept : ptr_(empty_sentinel_type::singleton()) {}

    explicit WeakObjectPtr(const ObjectPtr<T>& other) : WeakObjectPtr(other.get()) {
        retain();
    }

    WeakObjectPtr(const WeakObjectPtr& other) : ptr_(other.ptr_) {
        retain();
    }

    WeakObjectPtr(WeakObjectPtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = empty_sentinel_type::singleton();
    }

    template<typename Derived>
    WeakObjectPtr(const WeakObjectPtr<Derived>& other)// NOLINT
        : ptr_(RemapEmptyObjectSentinel(other.ptr_)) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        retain();
    }

    template<typename Derived>
    WeakObjectPtr(WeakObjectPtr<Derived>&& other) noexcept// NOLINT
        : ptr_(RemapEmptyObjectSentinel(other.ptr_)) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        other.ptr_ = EmptyObjectSentinel<Derived>::singleton();
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

    AM_NODISCARD bool defined() const noexcept {
        return ptr_ != empty_sentinel_type::singleton();
    }

    AM_NODISCARD bool empty() const noexcept {
        return ptr_ == empty_sentinel_type::singleton();
    }

    AM_NODISCARD bool is_sentinel() const noexcept {
        return ptr_ == empty_sentinel_type::singleton();
    }

    AM_NODISCARD T* get_or_null() const noexcept {
        return defined() ? ptr_ : nullptr;
    }

    AM_NODISCARD uint32_t use_count() const noexcept {
        return ptr_->use_count();
    }

    AM_NODISCARD uint32_t weak_use_count() const noexcept {
        return ptr_->weak_use_count();
    }

    AM_NODISCARD bool expired() const noexcept {
        return use_count() == 0;
    }

    /// Promotes the weak handle when the target object is still alive.
    ObjectPtr<T> lock() const noexcept {
        if (ptr_->TryPromoteWeakPtr()) {
            return ObjectPtr<T>(ptr_, DoNotIncRefCountTag());
        }
        return nullptr;
    }

    T* release() noexcept {
        auto* tmp = ptr_;
        ptr_ = empty_sentinel_type::singleton();
        return tmp;
    }

    static WeakObjectPtr reclaim(T* ptr) {
        AM_CHECK(ptr == empty_sentinel_type::singleton() ||
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
        ptr_ = empty_sentinel_type::singleton();
    }

    T* unsafe_get() const noexcept {
        return ptr_;
    }

private:
    explicit WeakObjectPtr(T* ptr) : ptr_(ptr) {}

    void retain() {
        if (defined()) {
            AM_CHECK(ptr_->weak_use_count() > 0, "ObjectPtr must be copy constructed with an object with weak_ref_count_ > 0");
            ptr_->IncWeakRef();
        }
    }

    template<typename Derived>
    static T* RemapEmptyObjectSentinel(Derived* ptr) noexcept {
        if (ptr == EmptyObjectSentinel<Derived>::singleton()) {
            return empty_sentinel_type::singleton();
        }
        return ptr;
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
    // ObjectPtr<Object> ptr_;
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

    template<typename T>
        requires std::is_base_of_v<Object, T>
    static ObjectPtr<T> Downcast(ObjectPtr<Object>& src) {
        return ObjectPtr<T>::reclaim(static_cast<T*>(src.release()));
    }

    template<typename T>
        requires std::is_base_of_v<Object, T>
    static ObjectPtr<T> Downcast(ObjectPtr<Object>&& src) {
        return ObjectPtr<T>::reclaim(static_cast<T*>(src.release()));
    }
};

}// namespace details

template<typename T>
void swap(ObjectPtr<T>& lhs, ObjectPtr<T>& rhs) noexcept {
    lhs.swap(rhs);
}

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
    return !lhs.defined();
}

template<typename T>
bool operator==(std::nullptr_t, const ObjectPtr<T>& rhs) noexcept {
    return !rhs.defined();
}

template<typename T1, typename T2>
bool operator!=(const ObjectPtr<T1>& lhs, const ObjectPtr<T2>& rhs) noexcept {
    return lhs.get() != rhs.get();
}

template<typename T>
bool operator!=(const ObjectPtr<T>& lhs, std::nullptr_t) noexcept {
    return lhs.defined();
}

template<typename T>
bool operator!=(std::nullptr_t, const ObjectPtr<T>& rhs) noexcept {
    return rhs.defined();
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

#endif// AETHERMIND_OBJECT_H
