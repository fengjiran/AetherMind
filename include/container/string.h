//
// Created by 赵丹 on 2025/8/22.
//

#ifndef AETHERMIND_CONTAINER_STRING_H
#define AETHERMIND_CONTAINER_STRING_H

#include "object.h"

namespace aethermind {

class StringImpl : public Object {
public:
    StringImpl();

    // ~StringImpl() override = default;

    NODISCARD size_t size() const noexcept;

    NODISCARD const char* data() const noexcept;

private:
    const char* data_;
    size_t size_;

    friend class String;
};

class String : public ObjectRef {
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
    String(const char* other);// NOLINT

    /*!
     * \brief Construct a new string object
     * \param other The std::string object to be copied
     */
    String(const std::string& other);// NOLINT

    String(std::string&& other);// NOLINT

    String(std::string_view other);//NOLINT

    explicit String(ObjectPtr<StringImpl>);

    String(const String&) = default;
    String(String&&) noexcept = default;

    String& operator=(const String& other);

    String& operator=(String&&) noexcept;

    String& operator=(const std::string&);

    String& operator=(const char*);

    void swap(String& other) noexcept;

    NODISCARD const char* data() const noexcept;

    NODISCARD const char* c_str() const noexcept;

    NODISCARD bool defined() const noexcept;

    NODISCARD size_t size() const noexcept;

    NODISCARD bool empty() const noexcept;

    NODISCARD char operator[](size_t i) const noexcept;

    NODISCARD char at(size_t i) const;

    NODISCARD uint32_t use_count() const noexcept;

    NODISCARD bool unique() const noexcept;

    NODISCARD StringImpl* get_impl_ptr_unsafe() const noexcept;

    NODISCARD StringImpl* release_impl_unsafe();

    NODISCARD const ObjectPtr<StringImpl>& get_object_ptr() const;

    operator std::string() const;// NOLINT

    operator const char*() const;//NOLINT

    /*!
     * \brief Compares this String object to other
     *
     * \param other The String to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int compare(const String& other) const;

    /*!
     * \brief Compares this String object to other
     *
     * \param other The string to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int compare(const std::string& other) const;

    /*!
     * \brief Compares this to other
     *
     * \param other The character array to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int compare(const char* other) const;

    /*!
     * \brief Compare two char sequence
     *
     * \param lhs Pointers to the char array to compare
     * \param lhs_cnt Length of the char array to compare
     * \param rhs Pointers to the char array to compare
     * \param rhs_cnt Length of the char array to compare
     * \return int zero if both char sequences compare equal. negative if this
     * appears before other, positive otherwise.
     */
    static int memncmp(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt);

    /*!
     * \brief Compare two char sequence for equality
     *
     * \param lhs Pointers to the char array to compare
     * \param rhs Pointers to the char array to compare
     * \param lhs_cnt Length of the char array to compare
     * \param rhs_cnt Length of the char array to compare
     *
     * \return true if the two char sequences are equal, false otherwise.
     */
    static bool memequal(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt);

private:
    ObjectPtr<StringImpl> impl_;

    /*!
     * \brief Concatenate two char sequences
     *
     * \param lhs Pointers to the lhs char array
     * \param lhs_cnt The size of the lhs char array
     * \param rhs Pointers to the rhs char array
     * \param rhs_cnt The size of the rhs char array
     *
     * \return The concatenated char sequence
     */
    static String concat(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt);

    // Overload + operator
    friend String operator+(const String& lhs, const String& rhs);
    friend String operator+(const String& lhs, const std::string& rhs);
    friend String operator+(const std::string& lhs, const String& rhs);
    friend String operator+(const String& lhs, const char* rhs);
    friend String operator+(const char* lhs, const String& rhs);
};

// Overload < operator
bool operator<(std::nullptr_t, const String& rhs) = delete;
bool operator<(const String& lhs, std::nullptr_t) = delete;
inline bool operator<(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) < 0; }
inline bool operator<(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) > 0; }
inline bool operator<(const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }
inline bool operator<(const String& lhs, const char* rhs) { return lhs.compare(rhs) < 0; }
inline bool operator<(const char* lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

// Overload > operator
bool operator>(std::nullptr_t, const String& rhs) = delete;
bool operator>(const String& lhs, std::nullptr_t) = delete;
inline bool operator>(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) > 0; }
inline bool operator>(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) < 0; }
inline bool operator>(const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }
inline bool operator>(const String& lhs, const char* rhs) { return lhs.compare(rhs) > 0; }
inline bool operator>(const char* lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

// Overload <= operator
bool operator<=(std::nullptr_t, const String& rhs) = delete;
bool operator<=(const String& lhs, std::nullptr_t) = delete;
inline bool operator<=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) <= 0; }
inline bool operator<=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }
inline bool operator<=(const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }
inline bool operator<=(const String& lhs, const char* rhs) { return lhs.compare(rhs) <= 0; }
inline bool operator<=(const char* lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

// Overload >= operator
bool operator>=(std::nullptr_t, const String& rhs) = delete;
bool operator>=(const String& lhs, std::nullptr_t) = delete;
inline bool operator>=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) >= 0; }
inline bool operator>=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }
inline bool operator>=(const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }
inline bool operator>=(const String& lhs, const char* rhs) { return lhs.compare(rhs) >= 0; }
inline bool operator>=(const char* lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

// Overload == operator
bool operator==(std::nullptr_t, const String& rhs) = delete;
bool operator==(const String& lhs, std::nullptr_t) = delete;
inline bool operator==(const String& lhs, const std::string& rhs) {
    return String::memequal(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const std::string& lhs, const String& rhs) {
    return String::memequal(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const String& lhs, const String& rhs) {
    return String::memequal(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const String& lhs, const char* rhs) { return lhs.compare(rhs) == 0; }
inline bool operator==(const char* lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

// Overload != operator
bool operator!=(const String& lhs, std::nullptr_t) = delete;
bool operator!=(std::nullptr_t, const String& rhs) = delete;
inline bool operator!=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) != 0; }
inline bool operator!=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) != 0; }
inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }
inline bool operator!=(const String& lhs, const char* rhs) { return lhs.compare(rhs) != 0; }
inline bool operator!=(const char* lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

std::ostream& operator<<(std::ostream& os, const String&);

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::String> {
    std::size_t operator()(const aethermind::String& str) const noexcept {
        return std::hash<std::string_view>()(std::string_view(str.data(), str.size()));
    }
};
}// namespace std

#endif//AETHERMIND_CONTAINER_STRING_H
