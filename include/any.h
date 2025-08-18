//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "error.h"
#include "type_traits.h"

namespace aethermind {

class Any {
public:
    Any() = default;

    Any(const Any&) = default;

    Any(Any&& other) noexcept : data_(other.data_) {
        other.reset();
    }

    // constructor from general types
    template<typename T>
    Any(T other) {
        TypeTraits<T>::MoveToAny(std::move(other), &data_);
    }

    template<typename T>
    Any& operator=(T other) {
        Any(std::move(other)).swap(*this);
        return *this;
    }

    Any& operator=(const Any& other) {
        // copy and swap idiom
        Any(other).swap(*this);
        return *this;
    }

    Any& operator=(Any&& other) noexcept {
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
            return std::optional<T>(std::nullopt);
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
            return std::optional<T>(std::nullopt);
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
                                        << TagToString(data_.tag_) << "` to `"
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
                                        << TagToString(data_.tag_) << "` to `"
                                        << TypeTraits<T>::TypeStr() << "`.";
        }
        return *std::move(opt);
    }

    ~Any() {
        reset();
    }

    Tag tag() const noexcept {
        return data_.tag_;
    }


    void swap(Any& other) noexcept {
        std::swap(data_, other.data_);
    }

    void reset() {
        data_.payload_ = 0;
        data_.tag_ = Tag::None;
    }

private:
    AetherMindAny data_;

#define COUNT_TAG(x) 1 +
    static constexpr auto kNumTags = AETHERMIND_FORALL_TAGS(COUNT_TAG) 0;
#undef COUNT_TAG
};

}// namespace aethermind

#endif//AETHERMIND_ANY_H
