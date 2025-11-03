//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "any_utils.h"
#include "data_type.h"
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
        } else {
            return 1;
        }
    }

    NODISCARD bool IsObjectRef() const override {
        if constexpr (std::is_base_of_v<ObjectRef, T>) {
            return true;
        } else {
            return false;
        }
    }

    NODISCARD void* GetUnderlyingPtr() override {
        return &value_;
    }

private:
    T value_;
    std::type_index type_index_;

    friend class Any;
};

class Device;
class Tensor;
class Function;
template<typename FType>
class TypedFunction;

class Any {
public:
    Any() = default;

    template<typename T,
             typename U = std::decay_t<T>,
             typename = std::enable_if_t<!details::is_plain_v<U> && !std::is_same_v<U, Any>>>
    Any(T&& value) : ptr_(std::make_unique<Holder<U>>(std::forward<T>(value))) {}// NOLINT

    // integer ctor
    template<typename T, std::enable_if_t<details::is_integral_v<T>>* = nullptr>
    Any(T value) : ptr_(std::make_unique<Holder<int64_t>>(value)) {}//NOLINT

    // floating point ctor
    template<typename T, std::enable_if_t<details::is_floating_point_v<T>>* = nullptr>
    Any(T value) : ptr_(std::make_unique<Holder<double>>(value)) {}//NOLINT

    // string ctor
    template<typename T, std::enable_if_t<details::is_string_v<T>>* = nullptr>
    Any(T value) : ptr_(std::make_unique<Holder<String>>(std::move(value))) {}//NOLINT

    Any(const Any& other);

    Any(Any&& other) noexcept;

    Any& operator=(const Any& other) &;

    Any& operator=(Any&& other) & noexcept;

    template<typename T>
    Any& operator=(T value) & {
        Any(std::move(value)).swap(*this);
        return *this;
    }

    /**
   * \brief Try to reinterpret the Any as a type T, return std::nullopt if it is not possible.
   *
   * \tparam T The type to cast to.
   * \return The cast value, or std::nullopt if the cast is not possible.
   * \note This function won't try to run type conversion (use try_cast for that purpose).
   */
    template<typename T, std::enable_if_t<!details::is_plain_v<T>>* = nullptr>
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

    template<typename T, std::enable_if_t<!details::is_plain_v<T>>* = nullptr>
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

    NODISCARD void* GetUnderlyingPtr() const;

    template<typename T, std::enable_if_t<details::is_plain_v<T>>* = nullptr>
    NODISCARD std::optional<T> as() const {
        if (has_value()) {
            if constexpr (details::is_integral_v<T>) {
                if (auto* p = dynamic_cast<Holder<int64_t>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            } else if constexpr (details::is_floating_point_v<T>) {
                if (auto* p = dynamic_cast<Holder<double>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            } else if constexpr (details::is_string_v<T>) {
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

    void reset();

    void swap(Any& other) noexcept;

    NODISCARD const std::type_index& type() const;

    NODISCARD bool has_value() const noexcept;

    NODISCARD bool IsBool() const noexcept;

    NODISCARD bool IsInteger() const noexcept;

    NODISCARD bool IsFloatingPoint() const noexcept;

    NODISCARD bool IsString() const noexcept;

    NODISCARD bool IsVoidPtr() const noexcept;

    NODISCARD bool IsDevice() const noexcept;

    NODISCARD bool IsTensor() const noexcept;

    NODISCARD bool IsObjectRef() const noexcept;

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

    bool operator==(const Any& other) const noexcept;
    bool operator!=(const Any& other) const noexcept;

private:
    std::unique_ptr<HolderBase> ptr_;

    friend class AnyEqual;
};

class AnyEqual {
public:
    bool operator()(const Any& lhs, const Any& rhs) const;
};

// template<typename T, typename = std::enable_if_t<details::is_integral_v<T>>>
// bool operator==(const Any& lhs, const T& rhs) {
//     return lhs.has_value() ? lhs.cast<T>() == rhs : false;
// }

// template<typename T>
// bool operator!=(const Any& lhs, const T& rhs) {
//     return !operator==(lhs, rhs);
// }
//
// template<typename T>
// bool operator==(const T& lhs, const Any& rhs) {
//     return rhs.has_value() ? lhs == rhs.cast<T>() : false;
// }
//
// template<typename T>
// bool operator!=(const T& lhs, const Any& rhs) {
//     return !operator==(lhs, rhs);
// }

/*
class Any {
public:
    Any() = default;

    Any(const Any& other);

    Any(Any&& other) noexcept;

    // constructor from general types
    template<typename T>
    Any(T other) {// NOLINT
        TypeTraits<T>::MoveToAny(std::move(other), &data_);
    }

    ~Any() {
        reset();
    }

    template<typename T>
    Any& operator=(T other) & {
        Any(std::move(other)).swap(*this);
        return *this;
    }

    Any& operator=(const Any& other) & {
        // copy and swap idiom
        Any(other).swap(*this);
        return *this;
    }

    Any& operator=(Any&& other) & noexcept {
        // copy and swap idiom
        Any(std::move(other)).swap(*this);
        return *this;
    }


    template<typename T>
    std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            if (TypeTraits<T>::check(&data_)) {
                return TypeTraits<T>::CopyFromAnyAfterCheck(&data_);
            }
            return std::nullopt;
        }
    }

    template<typename T>
    std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            if (TypeTraits<T>::check(&data_)) {
                return TypeTraits<T>::MoveFromAnyAfterCheck(&data_);
            }
            return std::nullopt;
        }
    }

    template<typename T>
    std::optional<T> try_cast() const {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            return TypeTraits<T>::TryCastFromAny(&data_);
        }
    }

    template<typename T>
    T cast() const& {
        auto opt = TypeTraits<T>::TryCastFromAny(&data_);
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError) << "Cannot convert from type `"
                                        << AnyTagToString(tag()) << "` to `"
                                        << TypeTraits<T>::TypeStr() << "`.";
        }
        return opt.value();
    }

    template<typename T>
    T cast() && {
        if (TypeTraits<T>::check(&data_)) {
            return TypeTraits<T>::MoveFromAnyAfterCheck(&data_);
        }
        auto opt = TypeTraits<T>::TryCastFromAny(&data_);
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError) << "Cannot convert from type `"
                                        << AnyTagToString(tag()) << "` to `"
                                        << TypeTraits<T>::TypeStr() << "`.";
        }
        return *std::move(opt);
    }

    NODISCARD uint32_t use_count() const noexcept;

    NODISCARD int64_t to_int() const;

    NODISCARD double to_double() const;

    NODISCARD bool to_bool() const;

    NODISCARD void* to_void_ptr() const;

    NODISCARD Device to_device() const;

    NODISCARD String to_string() const&;

    NODISCARD String to_string() &&;

    NODISCARD Tensor to_tensor() const&;

    NODISCARD Tensor to_tensor() &&;

    NODISCARD bool is_unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD bool is_bool() const noexcept {
        return tag() == AnyTag::Bool;
    }

    NODISCARD bool is_int() const noexcept {
        return tag() == AnyTag::Int;
    }

    NODISCARD bool is_double() const noexcept {
        return tag() == AnyTag::Double;
    }

    NODISCARD bool is_void_ptr() const noexcept {
        return tag() == AnyTag::OpaquePtr;
    }

    NODISCARD bool is_string() const noexcept {
        return tag() == AnyTag::String;
    }

    NODISCARD bool is_device() const noexcept {
        return tag() == AnyTag::Device;
    }

    NODISCARD bool is_tensor() const noexcept {
        return tag() == AnyTag::Tensor;
    }

    NODISCARD bool is_object_ptr() const noexcept {
        return IsObjectPtr(tag());
    }

    NODISCARD AnyTag tag() const noexcept {
        return data_.tag_;
    }

    void swap(Any& other) noexcept {
        std::swap(data_, other.data_);
    }

    void reset();

    bool operator==(std::nullptr_t) const noexcept {
        return data_.tag_ == AnyTag::None;
    }

    bool operator!=(std::nullptr_t) const noexcept {
        return data_.tag_ != AnyTag::None;
    }

private:
    AetherMindAny data_;

#define COUNT_TAG(x, _) 1 +
    static constexpr auto kNumTags = AETHERMIND_FORALL_ANY_TAGS(COUNT_TAG) 0;
#undef COUNT_TAG
};
*/

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

}// namespace aethermind

#endif//AETHERMIND_ANY_H
