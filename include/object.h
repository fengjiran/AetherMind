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

template<typename T>
struct DefaultNullType final {
    static constexpr T* singleton() noexcept {
        return nullptr;
    }
};

template<typename T>
struct GetNullType {
    using type = DefaultNullType<T>;
};

template<typename T>
using null_type = GetNullType<T>::type;

struct DoNotIncRefCountTag final {};

class Object {
public:
    using FDeleter = void (*)(Object*);

    Object() : ref_count_(0), deleter_(nullptr) {}

    NODISCARD uint32_t use_count() const {
        return __atomic_load_n(&ref_count_, __ATOMIC_RELAXED);
    }

    NODISCARD bool unique() const {
        return use_count() == 1;
    }

    void SetDeleter(FDeleter deleter) {
        deleter_ = deleter;
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
            if (deleter_ != nullptr) {
                deleter_(this);
            }
        }
    }

    /*! \brief Reference counter of the object. */
    uint32_t ref_count_;

    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    FDeleter deleter_;

    template<typename T, typename NullType>
    friend class ObjectPtr;

    friend struct ObjectUnsafe;
};

template<typename T, typename NullType = null_type<T>>
class ObjectPtr final {
    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
    static_assert(NullType::singleton() == NullType::singleton(), "NullType must have a constexpr singleton() method");
    static_assert(std::is_base_of_v<T, std::remove_pointer_t<decltype(NullType::singleton())>>,
                  "NullType::singleton() must return a element_type* pointer");

public:
    using element_type = T;
    using null_type = NullType;

    /*!
     * \brief Default constructor.
     */
    ObjectPtr() noexcept : ObjectPtr(NullType::singleton(), DoNotIncRefCountTag()) {}

    /*!
     * \brief Constructor from nullptr.
     */
    ObjectPtr(std::nullptr_t) noexcept// NOLINT
        : ObjectPtr(NullType::singleton(), DoNotIncRefCountTag()) {}

    /*!
     * \brief Constructor from unique_ptr.
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
        other.ptr_ = NullType::singleton();
    }

    /*!
     * \brief Copy constructor.
     *
     * \param other The other ObjectPtr.
     */
    template<typename Derived, typename DerivedNullType>
    ObjectPtr(const ObjectPtr<Derived, DerivedNullType>& other)// NOLINT
        : ptr_(other.ptr_ == DerivedNullType::singleton() ? NullType::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        retain();
    }

    /*!
     * \brief Move constructor.
     */
    template<typename Derived, typename DerivedNullType>
    ObjectPtr(ObjectPtr<Derived, DerivedNullType>&& other) noexcept// NOLINT
        : ptr_(other.ptr_ == DerivedNullType::singleton() ? NullType::singleton() : other.ptr_) {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        other.ptr_ = DerivedNullType::singleton();
    }

    /*!
     * \brief Copy assignment operator.
     */
    template<typename Derived, typename DerivedNullType>
    ObjectPtr& operator=(const ObjectPtr<Derived, DerivedNullType>& rhs) & {
        static_assert(std::is_base_of_v<T, Derived>, "Type mismatch, Derived must be derived from T");
        ObjectPtr(rhs).swap(*this);
        return *this;
    }

    /*!
     * \brief Move assignment operator.
     */
    template<typename Derived, typename DerivedNullType>
    ObjectPtr& operator=(ObjectPtr<Derived, DerivedNullType>&& rhs) & noexcept {
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
        ptr_ = NullType::singleton();
    }

    void swap(ObjectPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    NODISCARD bool defined() const {
        return ptr_ != NullType::singleton();
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

    operator bool() const noexcept {
        return defined();
    }

    NODISCARD uint32_t use_count() const noexcept {
        return defined() ? ptr_->use_count() : 0;
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
        ptr_ = NullType::singleton();
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
        if (ptr_ != NullType::singleton()) {
            CHECK(ptr_->use_count() > 0)
                    << "ObjectPtr must be copy constructed with an object with ref_count_ > 0";
            ptr_->IncRef();
        }
    }

    explicit ObjectPtr(T* ptr) : ObjectPtr(ptr, DoNotIncRefCountTag()) {
        if (ptr_ != NullType::singleton()) {
            CHECK(ptr_->use_count() == 0)
                    << "ObjectPtr must be constructed with a NullType or an object with ref_count_ == 0";
            ptr_->IncRef();
        }
    }

    T* ptr_;

    template<typename Derived>
    friend class ObjectAllocatorBase;

    template<typename T2, typename NullType2>
    friend class ObjectPtr;

    // template<typename T3, typename T4>
    // friend struct TypeTraits;
};

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
};

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
        return ObjectPtr<T>(ptr);
    }

    template<typename T, typename ElemType, typename... Args>
    ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "make can only be used to create Object");
        using Handler = Derived::template ArrayHandler<T, ElemType>;
        T* ptr = Handler::allocate(num_elems, std::forward<Args>(args)...);
        ptr->SetDeleter(Handler::deleter);
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
            p->T::~T();
            delete reinterpret_cast<StorageType*>(p);
            // LOG(INFO) << "deleter called" << std::endl;
        }
    };

    template<typename ObjType, typename ElemType>
    struct ArrayHandler {
        using StorageType = std::aligned_storage_t<sizeof(ObjType), alignof(ObjType)>;
        // for now only support elements that aligns with array header.
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

template<typename T, typename... Args>
ObjectPtr<T> make_object(Args&&... args) {
    return ObjectAllocator().make_object<T>(std::forward<Args>(args)...);
}

template<typename T, typename ElemType, typename... Args>
ObjectPtr<T> make_array_object(size_t num_elems, Args&&... args) {
    return ObjectAllocator().make_array_object<T, ElemType>(num_elems, std::forward<Args>(args)...);
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
