//
// Created by 赵丹 on 2025/8/22.
//

#ifndef AETHERMIND_STRING_H
#define AETHERMIND_STRING_H

#include "object.h"

namespace aethermind {

class StringImpl : public Object {
public:
    StringImpl();

    NODISCARD size_t size() const noexcept;

    NODISCARD const char* data() const noexcept;

private:
    const char* data_;
    size_t size_;

    friend class String;
};

class StringImplNullType final : public StringImpl {
    static StringImplNullType singleton_;
    StringImplNullType() = default;

public:
    static constexpr StringImpl* singleton() noexcept {
        return &singleton_;
    }
};

template<>
struct GetNullType<StringImpl> {
    using type = StringImplNullType;
};

static_assert(std::is_same_v<null_type<StringImpl>, StringImplNullType>);

class String {
public:
    String() = default;
    String(std::nullopt_t) = delete;// NOLINT

    /*!
     * \brief constructor from raw string
     *
     * \param other a char array.
     * \param size the size of the char array.
     */
    String(const char* other, size_t size);

    /*!
     * \brief constructor from raw string
     *
     * \param other a char array.
     * \note This constructor is marked as explicit to avoid implicit conversion
     *       of nullptr value here to string, which then was used in comparison
     */
    explicit String(const char* other);

    /*!
     * \brief Construct a new string object
     * \param other The std::string object to be copied
     */
    String(const std::string& other);// NOLINT


    String(const String&) = default;
    String(String&&) noexcept = default;
    // String& operator=(const String&) & = default;
    String& operator=(String&&) & noexcept = default;
    // String& operator=(const String&) && = default;
    String& operator=(String&&) && noexcept = default;

    String& operator=(const String& other) & {
        String(other).swap(*this);
        return *this;
    }

    String& operator=(const String& other) && {
        String(other).swap(*this);
        return *this;
    }

    void swap(String& other) noexcept;

    NODISCARD const char* data() const noexcept;

    NODISCARD const char* c_str() const noexcept;

    NODISCARD bool defined() const noexcept;

    NODISCARD size_t size() const noexcept;

    NODISCARD bool empty() const noexcept;

private:
    ObjectPtr<StringImpl> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_STRING_H
