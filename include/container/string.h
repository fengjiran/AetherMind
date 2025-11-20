//
// Created by 赵丹 on 2025/8/22.
//

#ifndef AETHERMIND_CONTAINER_STRING_H
#define AETHERMIND_CONTAINER_STRING_H

#include "container/container_utils.h"
#include "object.h"

namespace aethermind {

class StringImpl : public Object {
public:
    StringImpl() : data_(nullptr) {}

    NODISCARD char* data() const noexcept {
        return data_;
    }

private:
    static ObjectPtr<StringImpl> Create(size_t cap) {
        auto impl = make_array_object<StringImpl, char>(cap + 1);
        impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(StringImpl);
        std::memset(impl->data_, '\0', cap + 1);
        return impl;
    }

    char* data_;

    friend class String;
};

//TODO: implementation iterator(iterator adapter)
class String : public ObjectRef {
public:
    using traits_type = std::char_traits<char>;
    using allocator_type = std::allocator<char>;
    using allocator_traits = std::allocator_traits<allocator_type>;

    using value_type = traits_type::char_type;
    using size_type = allocator_traits::size_type;
    using difference_type = allocator_traits::difference_type;

    using pointer = allocator_traits::pointer;
    using const_pointer = allocator_traits::const_pointer;
    using reference = value_type&;
    using const_reference = const value_type&;

    class CharProxy;
    class Converter;

    // using iterator = details::IteratorAdapter<pointer, Converter, String>;
    // using const_iterator = details::IteratorAdapter<const_pointer, Converter, const String>;

    using iterator = value_type*;
    using const_iterator = const value_type*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    String() = default;
    String(std::nullopt_t) = delete;// NOLINT

    /*!
     * \brief constructor from raw string
     *
     * \param other a char array.
     * \param size the size of the char array.
     */
    String(const_pointer other, size_type size);

    /*!
     * \brief constructor from raw string
     *
     * \param other a char array.
     */
    String(const_pointer other);//NOLINT
    String(size_type size, value_type c);
    template<typename Iter>
    String(Iter first, Iter last) {
        Construct<>(first, last);
    }

    /*!
     * \brief Construct a new string object
     * \param other The std::string object to be copied
     */
    String(const std::string& other) : String(other.begin(), other.end()) {}//NOLINT
    String(std::string_view other) : String(other.begin(), other.end()) {}  //NOLINT
    String(const String& other);
    String(String&& other) noexcept;
    String(const String& other, size_type pos);
    String(const String& other, size_type pos, size_type n);
    String(std::initializer_list<char> list) : String(list.begin(), list.end()) {}

    String& operator=(const String& other);
    String& operator=(String&& other) noexcept;
    String& operator=(const std::string& other);
    String& operator=(const char* other);

    NODISCARD const_pointer data() const noexcept;
    NODISCARD pointer data() noexcept;
    NODISCARD const_pointer c_str() const noexcept;

    NODISCARD iterator begin() noexcept;
    NODISCARD const_iterator begin() const noexcept;
    NODISCARD iterator end() noexcept;
    NODISCARD const_iterator end() const noexcept;

    void swap(String& other) noexcept;

    NODISCARD bool defined() const noexcept;
    NODISCARD bool IsLocal() const noexcept;

    NODISCARD size_type size() const noexcept;
    NODISCARD size_type length() const noexcept;
    NODISCARD size_type capacity() const noexcept;
    NODISCARD bool empty() const noexcept;
    void clear() noexcept;

    NODISCARD uint32_t use_count() const noexcept;
    NODISCARD bool unique() const noexcept;
    NODISCARD StringImpl* GetImplPtrUnsafe() const noexcept;
    NODISCARD StringImpl* ReleaseImplUnsafe();
    NODISCARD const ObjectPtr<StringImpl>& GetObjectPtr() const;

    void push_back(value_type c);
    void pop_back() noexcept;

    NODISCARD static size_type max_size() noexcept;

    const_reference operator[](size_type i) const noexcept;
    CharProxy operator[](size_type i) noexcept;
    NODISCARD const_reference at(size_type i) const;
    CharProxy at(size_type i);
    CharProxy front() noexcept;
    NODISCARD const_reference front() const noexcept;
    CharProxy back() noexcept;
    NODISCARD const_reference back() const noexcept;

    NODISCARD String substr(size_type pos = 0, size_type n = npos) const;
    operator std::string() const;// NOLINT
    operator const char*() const;//NOLINT

    String& replace(size_type pos, size_type n1, const_pointer str, size_type n2);
    String& replace(size_type pos, size_type n1, const_pointer src);
    String& replace(size_type pos, size_type n, const String& src);
    String& replace(size_type pos1, size_type n1, const String& src, size_type pos2, size_type n2 = npos);
    String& replace(size_type pos, size_type n1, size_type n2, value_type c);
    String& replace(const_iterator first, const_iterator last, const_pointer src, size_type n);
    String& replace(const_iterator first, const_iterator last, const String& src);
    String& replace(const_iterator first, const_iterator last, const_pointer src);
    String& replace(const_iterator first, const_iterator last, size_type n, value_type c);
    template<typename Iter>
    String& replace(const_iterator first, const_iterator last, Iter k1, Iter k2) {
        CHECK(first >= begin() && first <= last && last <= end());
        size_t pos = first - begin();
        const size_type n1 = std::distance(first, last);
        const size_type n2 = std::distance(k1, k2);

        replace_aux(pos, n1, n2);
        for (auto it = k1; it != k2; ++it) {
            traits_type::assign(data()[pos++], *it);
        }
        return *this;
    }
    String& replace(const_iterator first, const_iterator last, pointer p1, pointer p2);
    String& replace(const_iterator first, const_iterator last, const_pointer p1, const_pointer p2);
    String& replace(const_iterator first, const_iterator last, std::initializer_list<value_type> l);

    void resize(size_type n, value_type c = value_type());
    void reserve(size_type n);
    void shrink_to_fit() noexcept;

    String& erase(size_type pos = 0, size_type n = npos);
    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

    String& append(const_pointer src, size_type n);
    String& append(const String& str);
    String& append(const String& str, size_type pos, size_type n = npos);
    String& append(const_pointer src);
    String& append(size_type n, value_type c);
    String& append(std::initializer_list<value_type> l);
    template<typename Iter,
             typename = std::enable_if_t<std::is_convertible_v<
                     typename std::iterator_traits<Iter>::iterator_category,
                     std::forward_iterator_tag>>>
    String& append(Iter first, Iter last) {
        return replace(end(), end(), first, last);
    }
    String& operator+=(const String& str);
    String& operator+=(const_pointer str);
    String& operator+=(value_type c);
    String& operator+=(std::initializer_list<value_type> l);

    iterator insert(const_iterator p, size_type n, value_type c);
    template<typename Iter>
    iterator insert(const_iterator p, Iter first, Iter last) {
        CHECK(p >= begin() && p <= end());
        const size_type pos = p - begin();
        replace(p, p, first, last);
        return iterator(data() + pos);
    }
    iterator insert(const_iterator p, std::initializer_list<char> l);
    iterator insert(const_iterator p, value_type c);
    String& insert(size_type pos, const String& other);
    String& insert(size_type pos1, const String& other, size_type pos2, size_type n = npos);
    String& insert(size_type pos, const_pointer str, size_type n);
    String& insert(size_type pos, const_pointer str);
    String& insert(size_type pos, size_type n, value_type c);

    /**
       *  @brief  Find position of a C substring.
       *  @param s  C string to locate.
       *  @param pos  Index of character to search from.
       *  @param n  Number of characters from @a s to search for.
       *  @return  Index of start of first occurrence.
       *
       *  Starting from @a pos, searches forward for the first @a n
       *  characters in @a s within this string. If found,
       *  returns the index where it begins.
       *  If not found, returns npos.
      */
    NODISCARD size_type find(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type find_kmp(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type find(const String& str, size_type pos = 0) const noexcept;
    NODISCARD size_type find(const_pointer str, size_type pos = 0) const noexcept;
    NODISCARD size_type find(value_type c, size_type pos = 0) const noexcept;

    NODISCARD size_type rfind(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type rfind(const String& str, size_type pos = npos) const noexcept;
    NODISCARD size_type rfind(const_pointer str, size_type pos = npos) const noexcept;
    NODISCARD size_type rfind(value_type c, size_type pos = npos) const noexcept;

    NODISCARD size_type find_first_of(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type find_first_of(const String& str, size_type pos = 0) const noexcept;
    NODISCARD size_type find_first_of(const_pointer str, size_type pos = 0) const noexcept;
    NODISCARD size_type find_first_of(value_type c, size_type pos = 0) const noexcept;

    NODISCARD size_type find_first_not_of(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type find_first_not_of(const String& str, size_type pos = 0) const noexcept;
    NODISCARD size_type find_first_not_of(const_pointer str, size_type pos = 0) const noexcept;
    NODISCARD size_type find_first_not_of(value_type c, size_type pos = 0) const noexcept;

    NODISCARD size_type find_last_of(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type find_last_of(const String& str, size_type pos = npos) const noexcept;
    NODISCARD size_type find_last_of(const_pointer str, size_type pos = npos) const noexcept;
    NODISCARD size_type find_last_of(value_type c, size_type pos = npos) const noexcept;

    NODISCARD size_type find_last_not_of(const_pointer s, size_type pos, size_type n) const noexcept;
    NODISCARD size_type find_last_not_of(const String& str, size_type pos = npos) const noexcept;
    NODISCARD size_type find_last_not_of(const_pointer str, size_type pos = npos) const noexcept;
    NODISCARD size_type find_last_not_of(value_type c, size_type pos = npos) const noexcept;

    NODISCARD bool starts_with(const String& str) const noexcept;
    NODISCARD bool starts_with(const_pointer str) const noexcept;
    NODISCARD bool starts_with(value_type c) const noexcept;

    NODISCARD bool ends_with(const String& str) const noexcept;
    NODISCARD bool ends_with(const_pointer str) const noexcept;
    NODISCARD bool ends_with(value_type c) const noexcept;

    /*!
     * \brief Compares this String object to other
     *
     * \param other The String to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int compare(const String& other) const;
    NODISCARD int compare(size_type pos, size_type n, const String& other) const;
    NODISCARD int compare(size_type pos1, size_type n1,
                          const String& other, size_type pos2, size_type n2 = npos) const;

    /*!
     * \brief Compares this String object to other
     *
     * \param other The string to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int compare(const std::string& other) const;
    NODISCARD int compare(size_type pos, size_type n, const std::string& other) const;
    NODISCARD int compare(size_type pos1, size_type n1,
                          const std::string& other, size_type pos2, size_type n2 = npos) const;

    /*!
     * \brief Compares this to other
     *
     * \param other The character array to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int compare(const_pointer other) const;
    NODISCARD int compare(size_type pos, size_type n, const_pointer other) const;
    NODISCARD int compare(size_type pos, size_type n1, const_pointer other, size_type n2) const;

    static constexpr size_type npos = static_cast<size_type>(-1);
    static constexpr size_type kIncFactor = 2;

private:
    enum {
        local_capacity_ = 15
    };

    union {
        value_type local_buffer_[local_capacity_ + 1];
        size_type capacity_ = 0;
    };
    size_type size_ = 0;
    ObjectPtr<StringImpl> impl_;

    void InitLocalBuffer() noexcept;
    NODISCARD size_type Limit(size_type pos, size_type limit) const noexcept;
    NODISCARD size_type CheckPos(size_type pos) const;
    void CheckSize(size_type delta) const;
    String& replace_aux(size_type pos, size_type n1, size_type n2);

    template<typename Iter,
             typename = std::enable_if_t<std::is_convertible_v<
                     typename std::iterator_traits<Iter>::iterator_category,
                     std::forward_iterator_tag>>>
    void Construct(Iter first, Iter last) {
        const size_type cap = std::distance(first, last);
        pointer dst = nullptr;
        if (cap > static_cast<size_type>(local_capacity_)) {
            impl_ = StringImpl::Create(cap);
            capacity_ = cap;
            dst = impl_->data();
        } else {
            InitLocalBuffer();
            dst = local_buffer_;
        }

        for (size_type i = 0; i < cap; ++i) {
            dst[i] = *first++;
        }

        size_ = cap;
    }

    void Construct(size_type n, value_type c);
    void SwitchContainer(size_type new_cap);
    void COW(int64_t delta);

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
    static int MemoryCompare(const_pointer lhs, size_type lhs_cnt, const_pointer rhs, size_type rhs_cnt);

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
    static String Concat(const_pointer lhs, size_t lhs_cnt, const_pointer rhs, size_t rhs_cnt);

    // Overload + operator
    friend String operator+(const String& lhs, const String& rhs);
    friend String operator+(const String& lhs, const std::string& rhs);
    friend String operator+(const String& lhs, const_pointer rhs);
    friend String operator+(const String& lhs, value_type rhs);

    friend String operator+(const std::string& lhs, const String& rhs);
    friend String operator+(const_pointer lhs, const String& rhs);
    friend String operator+(value_type lhs, const String& rhs);
};

class String::CharProxy {
public:
    CharProxy(String& str, size_type idx) : str_(str), idx_(idx) {}
    CharProxy& operator=(value_type c) {
        str_.replace(idx_, 1, 1, c);
        return *this;
    }

    friend bool operator==(const CharProxy& lhs, const CharProxy& rhs) {
        return *(lhs.str_.data() + lhs.idx_) == *(rhs.str_.data() + rhs.idx_);
    }

    friend bool operator!=(const CharProxy& lhs, const CharProxy& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const CharProxy& lhs, value_type rhs) {
        return *(lhs.str_.data() + lhs.idx_) == rhs;
    }

    friend bool operator!=(const CharProxy& lhs, value_type rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(value_type lhs, const CharProxy& rhs) {
        return rhs == lhs;
    }

    friend bool operator!=(value_type lhs, const CharProxy& rhs) {
        return rhs != lhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const CharProxy& c) {
        os << *(c.str_.data() + c.idx_);
        return os;
    }

private:
    String& str_;
    size_type idx_;
};

class String::Converter {
public:
    using value_type = String::value_type;
    static const value_type& convert(const String&, const_pointer ptr) {
        return *ptr;
    }

    static CharProxy convert(String& str, pointer ptr) {
        return {str, static_cast<size_type>(ptr - str.data())};
    }
};

inline std::ostream& operator<<(std::ostream& os, const String& str) {
    os.write(str.data(), str.size());
    return os;
}

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
inline bool operator==(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) == 0; }
inline bool operator==(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) == 0; }
inline bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }
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
