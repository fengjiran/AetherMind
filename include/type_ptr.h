//
// Created by 赵丹 on 2025/8/29.
//

#ifndef AETHERMIND_TYPE_PTR_H
#define AETHERMIND_TYPE_PTR_H

#include "macros.h"

#include <glog/logging.h>

namespace aethermind {

// The wrapper for singleton type pointer, as the tag of singleton type.
template<typename T>
class SingletonTypePtr {
public:
    using element_type = std::remove_extent_t<T>;

    SingletonTypePtr(T* ptr) : ptr_(ptr) {}// NOLINT

    explicit SingletonTypePtr(std::shared_ptr<T>) = delete;

    T* get() const {
        return ptr_;
    }

    T& operator*() const {
        return *ptr_;
    }

    T* operator->() const {
        return get();
    }

    operator bool() const {//NOLINT
        return get() != nullptr;
    }

private:
    T* ptr_{nullptr};
};

template<typename T, typename U>
bool operator==(const SingletonTypePtr<T>& lhs, const SingletonTypePtr<U>& rhs) {
    return static_cast<void*>(lhs.get()) == static_cast<void*>(rhs.get());
}

template<typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& lhs, const SingletonTypePtr<U>& rhs) {
    return !(lhs == rhs);
}

template<typename T>
class SingletonOrSharedTypePtr {
public:
    using element_type = std::remove_extent_t<T>;

    SingletonOrSharedTypePtr() = default;

    SingletonOrSharedTypePtr(std::nullptr_t) : repr_(nullptr) {}// NOLINT

    SingletonOrSharedTypePtr(std::shared_ptr<T> ptr) : repr_(std::move(ptr)) {}// NOLINT

    template<typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    SingletonOrSharedTypePtr(std::shared_ptr<U> ptr) : repr_(std::move(ptr)) {}// NOLINT

    SingletonOrSharedTypePtr(SingletonTypePtr<T> ptr) : repr_(ptr) {}// NOLINT

    template<typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    SingletonOrSharedTypePtr(SingletonTypePtr<U> ptr) : repr_(SingletonTypePtr<T>(ptr.get())) {}// NOLINT

    SingletonOrSharedTypePtr(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr(SingletonOrSharedTypePtr&&) noexcept = default;
    SingletonOrSharedTypePtr& operator=(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr& operator=(SingletonOrSharedTypePtr&&) noexcept = default;
    ~SingletonOrSharedTypePtr() = default;

    T* get() const {
        return repr_.is_shared_and_not_null() ? repr_.shared_repr_.ptr_.get()
                                              : static_cast<T*>(repr_.get_raw_repr().first_);
    }

    T* operator->() const {
        return get();
    }

    operator bool() const {//NOLINT
        return repr_.is_not_null();
    }

    template<typename U = T, typename = std::enable_if_t<!std::is_same_v<std::remove_const_t<U>, void>>>
    U& operator*() const {
        return *get();
    }

    bool operator==(std::nullptr_t) const {
        return !repr_.is_not_null();
    }

    bool operator!=(std::nullptr_t) const {
        return repr_.is_not_null();
    }

private:
    struct SharedPtrWrapper {
        std::shared_ptr<T> ptr_;

        SharedPtrWrapper(std::shared_ptr<T> ptr) : ptr_(std::move(ptr)) {}// NOLINT
    };

    struct SingletonRepr {
        T* singleton_;
        void* padding_{nullptr};

        explicit SingletonRepr(T* singleton) : singleton_(singleton) {}
    };

    struct RawRepr {
        void* first_;
        void* null_if_singleton_;
    };

    union Repr {
        Repr() : Repr(nullptr) {}

        explicit Repr(std::nullptr_t) : singleton_repr_(nullptr) {}

        explicit Repr(std::shared_ptr<T> ptr) : shared_repr_(std::move(ptr)) {}

        explicit Repr(SingletonTypePtr<T> ptr) : singleton_repr_(ptr.get()) {}

        ~Repr() {
            destroy();
        }

        Repr(const Repr& other) {
            if (other.is_shared_and_not_null()) {
                new (&shared_repr_) SharedPtrWrapper(other.shared_repr_);
            } else {
                singleton_repr_.singleton_ = static_cast<T*>(other.get_raw_repr().first_);
                CHECK(other.singleton_repr_.padding_ == nullptr);
                singleton_repr_.padding_ = nullptr;
            }
        }

        Repr(Repr&& other) noexcept {
            if (other.is_shared_and_not_null()) {
                new (&shared_repr_) SharedPtrWrapper(std::move(other.shared_repr_));
            } else {
                singleton_repr_.singleton_ = static_cast<T*>(other.get_raw_repr().first_);
                CHECK(other.singleton_repr_.padding_ == nullptr);
                singleton_repr_.padding_ = nullptr;
            }
        }

        Repr& operator=(const Repr& other) {
            if (this != &other) {
                if (other.is_shared_and_not_null()) {
                    if (is_shared_and_not_null()) {
                        shared_repr_ = other.shared_repr_;
                    } else {
                        new (&shared_repr_) SharedPtrWrapper(other.shared_repr_);
                    }
                } else {
                    if (is_shared_and_not_null()) {
                        destroy();
                    }
                    singleton_repr_.singleton_ = static_cast<T*>(other.get_raw_repr().first_);
                    CHECK(other.get_raw_repr().null_if_singleton_ == nullptr);
                    singleton_repr_.padding_ = nullptr;
                }
            }
            return *this;
        }

        Repr& operator=(Repr&& other) noexcept {
            if (this != &other) {
                if (other.is_shared_and_not_null()) {
                    if (is_shared_and_not_null()) {
                        shared_repr_ = std::move(other.shared_repr_);
                    } else {
                        new (&shared_repr_) SharedPtrWrapper(std::move(other.shared_repr_));
                    }
                } else {
                    if (is_shared_and_not_null()) {
                        destroy();
                    }
                    singleton_repr_.singleton_ = static_cast<T*>(other.get_raw_repr().first_);
                    CHECK(other.get_raw_repr().null_if_singleton_ == nullptr);
                    singleton_repr_.padding_ = nullptr;
                }
            }
            return *this;
        }

        RawRepr get_raw_repr() const {
            RawRepr repr;
            std::memcpy(&repr, reinterpret_cast<const char*>(this), sizeof(RawRepr));
            return repr;
        }

        NODISCARD bool is_not_null() const {
            auto repr = get_raw_repr();
            CHECK(repr.null_if_singleton_ == nullptr || repr.first_ != nullptr);
            return repr.first_ != nullptr;
        }

        NODISCARD bool is_shared_and_not_null() const {
            return get_raw_repr().null_if_singleton_ != nullptr;
        }

        SharedPtrWrapper shared_repr_;
        SingletonRepr singleton_repr_;

    private:
        void destroy() {
            if (is_shared_and_not_null()) {
                shared_repr_.~SharedPtrWrapper();
            }
        }
    };

    Repr repr_;
};

template<typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& lhs, const SingletonOrSharedTypePtr<U>& rhs) {
    return lhs.get() == rhs.get();
}

template<typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& lhs, const std::shared_ptr<U>& rhs) {
    return lhs.get() == rhs.get();
}

template<typename T, typename U>
bool operator==(const std::shared_ptr<T>& lhs, const SingletonOrSharedTypePtr<U>& rhs) {
    return lhs.get() == rhs.get();
}

template<typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& lhs, const SingletonTypePtr<U>& rhs) {
    return lhs.get() == rhs.get();
}

template<typename T, typename U>
bool operator==(const SingletonTypePtr<T>& lhs, const SingletonOrSharedTypePtr<U>& rhs) {
    return lhs.get() == rhs.get();
}

template<typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& lhs, const SingletonOrSharedTypePtr<U>& rhs) {
    return !(lhs == rhs);
}

template<typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& lhs, const std::shared_ptr<U>& rhs) {
    return !(lhs == rhs);
}

template<typename T, typename U>
bool operator!=(const std::shared_ptr<T>& lhs, const SingletonOrSharedTypePtr<U>& rhs) {
    return !(lhs == rhs);
}

template<typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& lhs, const SingletonTypePtr<U>& rhs) {
    return !(lhs == rhs);
}

template<typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& lhs, const SingletonOrSharedTypePtr<U>& rhs) {
    return !(lhs == rhs);
}

}// namespace aethermind

namespace std {
template<typename T>
struct hash<aethermind::SingletonOrSharedTypePtr<T>> {
    size_t operator()(const aethermind::SingletonOrSharedTypePtr<T>& x) const {
        return std::hash<T*>()(x.get());
    }
};
}// namespace std

#endif//AETHERMIND_TYPE_PTR_H
