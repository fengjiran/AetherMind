//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "any_utils.h"
#include "data_type.h"
#include "device.h"
#include "error.h"

#include <typeindex>

namespace aethermind {

class HolderBase {
public:
    virtual ~HolderBase() = default;
    NODISCARD virtual std::unique_ptr<HolderBase> Clone() const = 0;
    NODISCARD virtual const std::type_index& type() const = 0;
    NODISCARD virtual uint32_t use_count() const = 0;
    NODISCARD virtual bool IsObjectRef() const = 0;
    NODISCARD virtual bool IsMap() const = 0;
    NODISCARD virtual void* GetUnderlyingPtr() = 0;
};

template<typename T>
class Holder final : public HolderBase {
public:
    explicit Holder(T value) : value_(std::move(value)), type_index_(typeid(T)) {}

    NODISCARD std::unique_ptr<HolderBase> Clone() const override {
        return std::make_unique<Holder>(value_);
    }

    NODISCARD const std::type_index& type() const override {
        return type_index_;
    }

    NODISCARD uint32_t use_count() const override {
        if constexpr (details::has_use_count_method_v<T>) {
            return value_.use_count();
        }
        return 1;
    }

    NODISCARD bool IsObjectRef() const override {
        if constexpr (std::is_base_of_v<ObjectRef, T>) {
            return true;
        }
        return false;
    }

    NODISCARD bool IsMap() const override {
        if constexpr (details::is_map_v<T>) {
            return true;
        }
        return false;
    }

    NODISCARD void* GetUnderlyingPtr() override {
        return &value_;
    }

private:
    T value_;
    std::type_index type_index_;

    friend class Any;
};

class Tensor;
class Function;
template<typename FType>
class TypedFunction;
template<typename>
class SingletonOrSharedTypePtr;
class Type;

class Any {
public:
    Any() = default;

#ifdef CPP20
    template<typename T, typename U = std::decay_t<T>>
        requires(!details::is_plain_type<U> && !std::same_as<U, Any>)
#else
    template<typename T,
             typename U = std::decay_t<T>,
             typename = std::enable_if_t<!details::is_plain_type<U> && !std::is_same_v<U, Any>>>
#endif
    Any(T&& value) : ptr_(std::make_unique<Holder<U>>(std::forward<T>(value))) {// NOLINT
    }

// integer ctor
#ifdef CPP20
    template<details::is_integral T>
#else
    template<typename T, std::enable_if_t<details::is_integral<T>>* = nullptr>
#endif
    Any(T value) : ptr_(std::make_unique<Holder<int64_t>>(value)) {//NOLINT
    }

    // floating point ctor
#ifdef CPP20
    template<details::is_floating_point T>
#else
    template<typename T, std::enable_if_t<details::is_floating_point<T>>* = nullptr>
#endif
    Any(T value) : ptr_(std::make_unique<Holder<double>>(value)) {//NOLINT
    }

    // string ctor
#ifdef CPP20
    template<details::is_string T>
#else
    template<typename T, std::enable_if_t<details::is_string<T>>* = nullptr>
#endif
    Any(T value) : ptr_(std::make_unique<Holder<String>>(std::move(value))) {//NOLINT
    }

    Any(const Any& other);

    Any(Any&& other) noexcept;

    Any& operator=(const Any& other) &;

    Any& operator=(Any&& other) & noexcept;

    template<typename T>
    Any& operator=(T value) & {
        Any(std::move(value)).swap(*this);
        return *this;
    }

    NODISCARD void* GetUnderlyingPtr() const;

    /**
   * \brief Try to reinterpret the Any as a type T, return std::nullopt if it is not possible.
   *
   * \tparam T The type to cast to.
   * \return The cast value, or std::nullopt if the cast is not possible.
   * \note This function won't try to run type conversion (use try_cast for that purpose).
   */
#ifdef CPP20
    template<typename T>
        requires(!details::is_plain_type<T>)
#else
    template<typename T, std::enable_if_t<!details::is_plain_type<T>>* = nullptr>
#endif
    NODISCARD std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            if (has_value()) {
                if (auto* p = dynamic_cast<Holder<T>*>(ptr_.get())) {
                    return p->value_;
                }
            }
            return std::nullopt;
        }
    }

#ifdef CPP20
    template<typename T>
        requires(!details::is_plain_type<T>)
#else
    template<typename T, std::enable_if_t<!details::is_plain_type<T>>* = nullptr>
#endif
    std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Any>) {
            return std::move(*this);
        } else {
            if (has_value()) {
                if (auto* p = dynamic_cast<Holder<T>*>(ptr_.get())) {
                    return std::move(p->value_);
                }
            }
            return std::nullopt;
        }
    }

#ifdef CPP20
    template<typename T>
        requires details::is_plain_type<T>
#else
    template<typename T, std::enable_if_t<details::is_plain_type<T>>* = nullptr>
#endif
    NODISCARD std::optional<T> as() const {
        if (has_value()) {
            if constexpr (details::is_integral<T>) {
                if (auto* p = dynamic_cast<Holder<int64_t>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            } else if constexpr (details::is_floating_point<T>) {
                if (auto* p = dynamic_cast<Holder<double>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            } else if constexpr (details::is_string<T>) {
                if (auto* p = dynamic_cast<Holder<String>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            }
        }
        return std::nullopt;
    }

    template<typename T>
    std::optional<T> try_cast() const {
        return as<T>();
    }

    template<typename T>
    NODISCARD T cast() const& {
        auto opt = as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError) << "cast failed.";
        }
        return opt.value();
    }

    template<typename T>
    T cast() && {
        auto opt = std::move(*this).as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError);
        }
        reset();
        return std::move(opt.value());
    }

    template<typename T>
    operator T() {//NOLINT
        return cast<T>();
    }


    template<typename T>
    T MoveFromAny() {
        return std::move(*this).cast<T>();
    }

    template<typename T>
        requires requires {
            typename T::size_type;
            requires details::is_array_subscript<T>;
        }
    decltype(auto) operator[](T::size_type i) {
        return (*static_cast<T*>(ptr_->GetUnderlyingPtr()))[i];
    }

    template<typename T>
        requires requires {
            typename T::key_type;
            typename T::mapped_type;
            requires details::is_map_subscript<T>;
        }
    decltype(auto) operator[](const T::key_type& key) {
        return (*static_cast<T*>(ptr_->GetUnderlyingPtr()))[key];
    }

    void reset();

    void swap(Any& other) noexcept;

    NODISCARD const std::type_index& type() const;

    NODISCARD SingletonOrSharedTypePtr<Type> GetTypePtr() const noexcept;

    NODISCARD bool has_value() const noexcept;

    NODISCARD bool IsNone() const noexcept;

    NODISCARD bool IsBool() const noexcept;

    NODISCARD bool IsInteger() const noexcept;

    NODISCARD bool IsFloatingPoint() const noexcept;

    NODISCARD bool IsString() const noexcept;

    NODISCARD bool IsVoidPtr() const noexcept;

    NODISCARD bool IsDevice() const noexcept;

    NODISCARD bool IsTensor() const noexcept;

    NODISCARD bool IsObjectRef() const noexcept;

    NODISCARD bool IsMap() const noexcept;

    NODISCARD String ToNone() const noexcept;

    NODISCARD int64_t ToInt() const;

    NODISCARD double ToDouble() const;

    NODISCARD bool ToBool() const;

    NODISCARD void* ToVoidPtr() const;

    NODISCARD Device ToDevice() const;

    NODISCARD String ToString() const;

    NODISCARD Tensor ToTensor() const;

    NODISCARD uint32_t use_count() const noexcept;

    NODISCARD bool unique() const noexcept;

    bool operator==(std::nullptr_t) const noexcept;

    bool operator!=(std::nullptr_t p) const noexcept;

private:
    std::unique_ptr<HolderBase> ptr_;

    friend class AnyEqual;
};

class Any1 {
public:
    Any1() = default;

    // not plain type ctor
    template<typename T, typename U = std::decay_t<T>>
        requires(!details::is_plain_type<U> && !std::same_as<U, Any>)
    Any1(T&& value) {//NOLINT
        if constexpr (sizeof(U) <= kSmallObjectSize) {
            // small object, construct at local buffer
            is_small_object_ = true;
            new (local_buffer_) U(std::forward<T>(value));
            small_type_index_ = typeid(U);
        } else {
            is_small_object_ = false;
            ptr_ = std::make_unique<Holder<U>>(std::forward<T>(value));
        }
    }

    // integer ctor
    template<details::is_integral T>
    Any1(T value) {//NOLINT
        using U = int64_t;
        if constexpr (sizeof(U) <= kSmallObjectSize) {
            // small object, construct at local buffer
            is_small_object_ = true;
            new (local_buffer_) U(static_cast<U>(value));
            small_type_index_ = typeid(U);
        } else {
            is_small_object_ = false;
            ptr_ = std::make_unique<Holder<U>>(static_cast<U>(value));
        }
    }

    // floating ctor
    template<details::is_floating_point T>
    Any1(T value) {//NOLINT
        using U = double;
        if constexpr (sizeof(U) <= kSmallObjectSize) {
            is_small_object_ = true;
            new (local_buffer_) U(static_cast<U>(value));
            small_type_index_ = typeid(U);
        } else {
            is_small_object_ = false;
            ptr_ = std::make_unique<Holder<U>>(static_cast<U>(value));
        }
    }

    // string ctor
    template<details::is_string T>
    Any1(T value) {//NOLINT
        using U = String;
        if constexpr (sizeof(U) <= kSmallObjectSize) {
            is_small_object_ = true;
            new (local_buffer_) U(static_cast<U>(value));
            small_type_index_ = typeid(U);
        } else {
            is_small_object_ = false;
            ptr_ = std::make_unique<Holder<U>>(static_cast<U>(value));
        }
    }

    Any1(const Any1& other) {
        if (other.is_small_object_) {
            std::memcpy(local_buffer_, other.local_buffer_, kSmallObjectSize);
            is_small_object_ = true;
            small_type_index_ = other.small_type_index_;
        } else {
            if (other.has_value()) {
                ptr_ = other.ptr_->Clone();
            }
            is_small_object_ = false;
        }
    }

    NODISCARD bool has_value() const noexcept {
        return is_small_object_ ? small_type_index_ != typeid(std::nullptr_t) : ptr_ != nullptr;
    }

private:
    static constexpr size_t kSmallObjectSize = sizeof(void*) * 2;
    union {
        alignas(std::max_align_t) uint8_t local_buffer_[kSmallObjectSize];// small object buffer
        std::unique_ptr<HolderBase> ptr_{nullptr};                        //big object pointer
    };

    bool is_small_object_{true};
    std::type_index small_type_index_{typeid(std::nullptr_t)};
};

// std::ostream& operator<<(std::ostream& os, const Any& any);

class AnyEqual {
public:
    bool operator()(const Any& lhs, const Any& rhs) const;
};

inline bool operator==(const Any& lhs, const Any& rhs) noexcept {
    return AnyEqual()(lhs, rhs);
}

inline bool operator!=(const Any& lhs, const Any& rhs) noexcept {
    return !AnyEqual()(lhs, rhs);
}

class AnyHash {
public:
    size_t operator()(const Any& v) const;
};

namespace details {

template<typename T>
struct TypeName {
    static String value() {
        return typeid(T).name();
    }
};

#define DEFINE_TYPE_NAME(code, bits, lanes, T, name) \
    template<>                                       \
    struct TypeName<T> {                             \
        static String value() {                      \
            return #name;                            \
        }                                            \
    };

SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_TYPE_NAME);
DEFINE_TYPE_NAME(_, _, _, Tensor, Tensor);
DEFINE_TYPE_NAME(_, _, _, Device, Device);
DEFINE_TYPE_NAME(_, _, _, Any, Any);
DEFINE_TYPE_NAME(_, _, _, Any*, Any*);
DEFINE_TYPE_NAME(_, _, _, const Any*, const Any*);
DEFINE_TYPE_NAME(_, _, _, const Any&, const Any&);
DEFINE_TYPE_NAME(_, _, _, void, void);
DEFINE_TYPE_NAME(_, _, _, Function, Function);

template<typename FType>
struct TypeName<TypedFunction<FType>> {
    static String value() {
        return "Function";
    }
};

#undef DEFINE_TYPE_NAME

template<typename T>
struct Type2Str {
    using U = std::remove_const_t<std::remove_reference_t<T>>;
    static String value() {
        return TypeName<U>::value();
    }
};

}// namespace details

// template<typename T>
//     requires std::default_initializable<T>
// class Test {
// public:
//     Test() = default;
//
//     void print();
//
// private:
//     T value;
// };
//
// template<typename T>
//     requires std::default_initializable<T>
// void Test<T>::print() {
//     std::cout << value;
// }

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::Any> {
    size_t operator()(const aethermind::Any& v) const noexcept {
        return aethermind::AnyHash()(v);
    }
};
}// namespace std

#endif//AETHERMIND_ANY_H
