//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "error.h"
#include "type_traits.h"

namespace aethermind {

class HolderBase {
public:
    virtual ~HolderBase() = default;
    NODISCARD virtual std::unique_ptr<HolderBase> Clone() const = 0;
    NODISCARD virtual const std::type_index& type() const = 0;
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

    T value_;
    std::type_index type_index_;
};


class Param {
public:
    Param() = default;

    template<typename T,
             typename U = std::decay_t<T>,
             typename = std::enable_if_t<!details::is_plain_v<U> && !std::is_same_v<U, Param>>>
    Param(T&& value) : ptr_(std::make_unique<Holder<U>>(std::forward<T>(value))) {}// NOLINT

    template<typename T, std::enable_if_t<details::is_integral_v<T>>* = nullptr>
    Param(T value) : ptr_(std::make_unique<Holder<int64_t>>(value)) {}//NOLINT

    template<typename T, std::enable_if_t<details::is_floating_point_v<T>>* = nullptr>
    Param(T value) : ptr_(std::make_unique<Holder<double>>(value)) {}//NOLINT

    template<typename T, std::enable_if_t<details::is_string_v<T>>* = nullptr>
    Param(T value) : ptr_(std::make_unique<Holder<String>>(std::move(value))) {}//NOLINT

    Param(const Param& other) {
        ptr_ = other.ptr_ ? other.ptr_->Clone() : nullptr;
    }

    Param(Param&& other) noexcept {
        ptr_ = std::move(other.ptr_);
    }

    Param& operator=(const Param& other) & {
        Param(other).swap(*this);
        return *this;
    }

    Param& operator=(Param&& other) & noexcept {
        Param(std::move(other)).swap(*this);
        return *this;
    }

    template<typename T>
    Param& operator=(T value) & {
        Param(std::move(value)).swap(*this);
        return *this;
    }

    template<typename T, std::enable_if_t<!details::is_plain_v<T>>* = nullptr>
    std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Param>) {
            return *this;
        } else {
            if (ptr_) {
                if (auto* p = dynamic_cast<Holder<T>*>(ptr_.get())) {
                    return p->value_;
                }
                return std::nullopt;
            }
            return std::nullopt;
        }
    }

    template<typename T, std::enable_if_t<!details::is_plain_v<T>>* = nullptr>
    std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Param>) {
            return *this;
        } else {
            if (ptr_) {
                if (auto* p = dynamic_cast<Holder<T>*>(ptr_.get())) {
                    return std::move(p->value_);
                }
                return std::nullopt;
            }
            return std::nullopt;
        }
    }

    template<typename T, std::enable_if_t<details::is_integral_v<T>>* = nullptr>
    std::optional<T> as() const {
        if (ptr_) {
            if (auto* p = dynamic_cast<Holder<int64_t>*>(ptr_.get())) {
                return static_cast<T>(p->value_);
            }
            return std::nullopt;
        }
        return std::nullopt;
    }

    template<typename T, std::enable_if_t<details::is_floating_point_v<T>>* = nullptr>
    std::optional<T> as() const {
        if (ptr_) {
            if (auto* p = dynamic_cast<Holder<double>*>(ptr_.get())) {
                return static_cast<T>(p->value_);
            }
            return std::nullopt;
        }
        return std::nullopt;
    }

    template<typename T, std::enable_if_t<details::is_string_v<T>>* = nullptr>
    std::optional<T> as() const {
        if (ptr_) {
            if (auto* p = dynamic_cast<Holder<String>*>(ptr_.get())) {
                return static_cast<T>(p->value_);
            }
            return std::nullopt;
        }
        return std::nullopt;
    }

    template<typename T>
    std::optional<T> try_cast() {
        return as<T>();
    }

    template<typename T>
    T cast() const& {
        auto opt = as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError);
        }
        return opt.value();
    }

    template<typename T>
    T cast() && {
        auto opt = std::move(*this).as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError);
        }
        return std::move(opt.value());
    }

    void swap(Param& other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    NODISCARD const std::type_index& type() const {
        if (ptr_) {
            return ptr_->type();
        }
        AETHERMIND_THROW(BadAnyCast) << "Any has no value.";
        AETHERMIND_UNREACHABLE();
    }

    NODISCARD bool is_bool() const noexcept {
        return type() == std::type_index(typeid(bool));
    }

    NODISCARD bool is_int() const noexcept {
        return type() == std::type_index(typeid(int64_t));
    }

    NODISCARD bool is_floating_point() const noexcept {
        return type() == std::type_index(typeid(double));
    }

    NODISCARD bool is_string() const noexcept {
        return type() == std::type_index(typeid(String));
    }

    NODISCARD bool is_void_ptr() const noexcept {
        return type() == std::type_index(typeid(void*));
    }


private:
    std::unique_ptr<HolderBase> ptr_;
};

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

    /**
   * \brief Try to reinterpret the Any as a type T, return std::nullopt if it is not possible.
   *
   * \tparam T The type to cast to.
   * \return The cast value, or std::nullopt if the cast is not possible.
   * \note This function won't try to run type conversion (use try_cast for that purpose).
   */
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

namespace details {

// template<typename T>
// struct Type2Str {
//     static std::string value() {
//         return TypeTraitsNoCR<T>::TypeStr();
//     }
// };

template<>
struct Type2Str<Any> {
    static std::string value() {
        return "Any";
    }
};

template<>
struct Type2Str<Any*> {
    static std::string value() {
        return "Any*";
    }
};

template<>
struct Type2Str<const Any*> {
    static std::string value() {
        return "const Any*";
    }
};

template<>
struct Type2Str<const Any&> {
    static std::string value() {
        return "const Any&";
    }
};

template<>
struct Type2Str<void> {
    static std::string value() {
        return "void";
    }
};
}// namespace details

}// namespace aethermind

#endif//AETHERMIND_ANY_H
