//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "error.h"
#include "container/string.h"
#include "type_traits.h"

namespace aethermind {

class Any {
public:
    Any() = default;

    Any(const Any& other);

    Any(Any&& other) noexcept;

    // constructor from general types
    template<typename T>
    Any(T other) {//NOLINT
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
    std::optional<T> try_cast() {
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

}// namespace aethermind

#endif//AETHERMIND_ANY_H
