//
// Created by 赵丹 on 2025/8/22.
//

#ifndef AETHERMIND_CONTAINER_STRING_H
#define AETHERMIND_CONTAINER_STRING_H

#include "object.h"
#include "string.h"

namespace aethermind_test {
using namespace aethermind;

class StringImpl : public Object {
public:
    StringImpl();

    NODISCARD size_t size() const noexcept;

    NODISCARD size_t capacity() const noexcept;

    NODISCARD const char* data() const noexcept;

private:
    static ObjectPtr<StringImpl> Create(size_t cap);

    char* data_;
    size_t size_;
    size_t capacity_;

    friend class String;
};

class String : public ObjectRef {
public:
    String();
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
     */
    String(const char* other);// NOLINT

    String(size_t size, char c);

    template<typename Iter>
    String(Iter first, Iter last) {
        impl_ = StringImpl::Create(std::distance(first, last));
        size_t i = 0;
        for (auto it = first; it != last; ++it) {
            impl_->data_[i++] = *it;
            ++impl_->size_;
        }
    }

    String(std::initializer_list<char> list);

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

    NODISCARD StringImpl* GetImplPtrUnsafe() const noexcept;

    NODISCARD StringImpl* ReleaseImplUnsafe();

    NODISCARD const ObjectPtr<StringImpl>& GetObjectPtr() const;

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
    NODISCARD int Compare(const String& other) const;

    /*!
     * \brief Compares this String object to other
     *
     * \param other The string to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int Compare(const std::string& other) const;

    /*!
     * \brief Compares this to other
     *
     * \param other The character array to compare with.
     *
     * \return zero if both char sequences compare equal. negative if this appears
     * before other, positive otherwise.
     */
    NODISCARD int Compare(const char* other) const;

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
    static int MemoryCompare(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt);

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
    static bool MemoryEqual(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt);

    static constexpr size_t npos = static_cast<size_t>(-1);

private:
    enum {
        local_capacity_ = 15
    };

    union {
        char local_buffer_[local_capacity_ + 1];
        size_t capacity_;
    };

    size_t size_;
    ObjectPtr<StringImpl> impl_;

    void InitLocalBuffer() noexcept;

    NODISCARD bool IsLocal() const noexcept;
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
    static String Concat(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt);

    // Overload + operator
    friend String operator+(const String& lhs, const String& rhs);
    friend String operator+(const String& lhs, const std::string& rhs);
    friend String operator+(const std::string& lhs, const String& rhs);
    friend String operator+(const String& lhs, const char* rhs);
    friend String operator+(const char* lhs, const String& rhs);
};

class string_test {
public:
    string_test() = default;

private:
    enum {
        local_capacity_ = 15
    };

    union {
        char local_buf[local_capacity_ + 1];
        size_t capacity_;
    };

    ObjectPtr<StringImpl> impl_;
    size_t size_;
};

// Overload < operator
bool operator<(std::nullptr_t, const String& rhs) = delete;
bool operator<(const String& lhs, std::nullptr_t) = delete;
inline bool operator<(const String& lhs, const std::string& rhs) { return lhs.Compare(rhs) < 0; }
inline bool operator<(const std::string& lhs, const String& rhs) { return rhs.Compare(lhs) > 0; }
inline bool operator<(const String& lhs, const String& rhs) { return lhs.Compare(rhs) < 0; }
inline bool operator<(const String& lhs, const char* rhs) { return lhs.Compare(rhs) < 0; }
inline bool operator<(const char* lhs, const String& rhs) { return rhs.Compare(lhs) > 0; }

// Overload > operator
bool operator>(std::nullptr_t, const String& rhs) = delete;
bool operator>(const String& lhs, std::nullptr_t) = delete;
inline bool operator>(const String& lhs, const std::string& rhs) { return lhs.Compare(rhs) > 0; }
inline bool operator>(const std::string& lhs, const String& rhs) { return rhs.Compare(lhs) < 0; }
inline bool operator>(const String& lhs, const String& rhs) { return lhs.Compare(rhs) > 0; }
inline bool operator>(const String& lhs, const char* rhs) { return lhs.Compare(rhs) > 0; }
inline bool operator>(const char* lhs, const String& rhs) { return rhs.Compare(lhs) < 0; }

// Overload <= operator
bool operator<=(std::nullptr_t, const String& rhs) = delete;
bool operator<=(const String& lhs, std::nullptr_t) = delete;
inline bool operator<=(const String& lhs, const std::string& rhs) { return lhs.Compare(rhs) <= 0; }
inline bool operator<=(const std::string& lhs, const String& rhs) { return rhs.Compare(lhs) >= 0; }
inline bool operator<=(const String& lhs, const String& rhs) { return lhs.Compare(rhs) <= 0; }
inline bool operator<=(const String& lhs, const char* rhs) { return lhs.Compare(rhs) <= 0; }
inline bool operator<=(const char* lhs, const String& rhs) { return rhs.Compare(lhs) >= 0; }

// Overload >= operator
bool operator>=(std::nullptr_t, const String& rhs) = delete;
bool operator>=(const String& lhs, std::nullptr_t) = delete;
inline bool operator>=(const String& lhs, const std::string& rhs) { return lhs.Compare(rhs) >= 0; }
inline bool operator>=(const std::string& lhs, const String& rhs) { return rhs.Compare(lhs) <= 0; }
inline bool operator>=(const String& lhs, const String& rhs) { return lhs.Compare(rhs) >= 0; }
inline bool operator>=(const String& lhs, const char* rhs) { return lhs.Compare(rhs) >= 0; }
inline bool operator>=(const char* lhs, const String& rhs) { return rhs.Compare(lhs) <= 0; }

// Overload == operator
bool operator==(std::nullptr_t, const String& rhs) = delete;
bool operator==(const String& lhs, std::nullptr_t) = delete;
inline bool operator==(const String& lhs, const std::string& rhs) {
    return String::MemoryEqual(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const std::string& lhs, const String& rhs) {
    return String::MemoryEqual(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const String& lhs, const String& rhs) {
    return String::MemoryEqual(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const String& lhs, const char* rhs) { return lhs.Compare(rhs) == 0; }
inline bool operator==(const char* lhs, const String& rhs) { return rhs.Compare(lhs) == 0; }

// Overload != operator
bool operator!=(const String& lhs, std::nullptr_t) = delete;
bool operator!=(std::nullptr_t, const String& rhs) = delete;
inline bool operator!=(const String& lhs, const std::string& rhs) { return lhs.Compare(rhs) != 0; }
inline bool operator!=(const std::string& lhs, const String& rhs) { return rhs.Compare(lhs) != 0; }
inline bool operator!=(const String& lhs, const String& rhs) { return lhs.Compare(rhs) != 0; }
inline bool operator!=(const String& lhs, const char* rhs) { return lhs.Compare(rhs) != 0; }
inline bool operator!=(const char* lhs, const String& rhs) { return rhs.Compare(lhs) != 0; }

std::ostream& operator<<(std::ostream& os, const String&);
}// namespace aethermind_test


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

class String : public ObjectRef {
public:
    using traits_type = std::char_traits<char>;
    using allocator_type = std::allocator<char>;
    using allocator_traits = std::allocator_traits<allocator_type>;

    using value_type = traits_type::char_type;
    using size_type = allocator_traits::size_type;
    using difference_type = allocator_traits::difference_type;
    using iterator = value_type*;
    using const_iterator = const value_type*;
    using pointer = allocator_traits::pointer;
    using const_pointer = allocator_traits::const_pointer;
    using reference = std::iterator_traits<iterator>::reference;
    using const_reference = std::iterator_traits<const_iterator>::reference;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    class CharProxy {
    public:
        CharProxy(String& str, size_type idx) : str_(str), idx_(idx) {}
        CharProxy& operator=(value_type c) {
            str_.COW(0);
            *(str_.data() + idx_) = c;
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

    String(size_type size, char c);

    template<typename Iter>
    String(Iter first, Iter last) {
        Construct<>(first, last);
    }

    String(std::initializer_list<char> list) : String(list.begin(), list.end()) {}

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

    String& operator=(const String& other);
    String& operator=(String&& other) noexcept;
    String& operator=(const std::string& other);
    String& operator=(const char* other);

    NODISCARD iterator begin() noexcept {
        return data();
    }

    NODISCARD const_iterator begin() const noexcept {
        return data();
    }

    NODISCARD iterator end() noexcept {
        return data() + size();
    }

    NODISCARD const_iterator end() const noexcept {
        return data() + size();
    }

    void swap(String& other) noexcept;

    NODISCARD const_pointer data() const noexcept {
        return IsLocal() ? local_buffer_ : impl_->data();
    }

    NODISCARD pointer data() noexcept {
        return IsLocal() ? local_buffer_ : impl_->data();
    }

    NODISCARD const_pointer c_str() const noexcept {
        return data();
    }

    NODISCARD bool defined() const noexcept {
        return impl_;
    }

    NODISCARD size_type size() const noexcept {
        return size_;
    }

    NODISCARD size_type length() const noexcept {
        return size_;
    }

    NODISCARD size_type capacity() const noexcept {
        return IsLocal() ? static_cast<size_type>(local_capacity_) : capacity_;
    }

    NODISCARD bool empty() const noexcept {
        return size() == 0;
    }

    void clear() noexcept {
        size_ = 0;
    }

    void push_back(char c);

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
        String src(k1, k2);
        return replace(first - begin(), last - first, src.data(), src.size());
    }

    String& replace(const_iterator first, const_iterator last, pointer k1, pointer k2);
    String& replace(const_iterator first, const_iterator last, const_pointer k1, const_pointer k2);
    String& replace(const_iterator first, const_iterator last, std::initializer_list<value_type> l);

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


    NODISCARD uint32_t use_count() const noexcept {
        return IsLocal() ? 1 : impl_.use_count();
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    NODISCARD StringImpl* GetImplPtrUnsafe() const noexcept {
        return impl_.get();
    }

    NODISCARD StringImpl* ReleaseImplUnsafe() {
        return impl_.release();
    }

    NODISCARD const ObjectPtr<StringImpl>& GetObjectPtr() const {
        return impl_;
    }

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
    static bool MemoryEqual(const_pointer lhs, size_type lhs_cnt, const_pointer rhs, size_type rhs_cnt);

    static constexpr size_type npos = static_cast<size_type>(-1);
    static constexpr size_type kIncFactor = 2;

private:
    enum {
        local_capacity_ = 15
    };

    union {
        char local_buffer_[local_capacity_ + 1];
        size_type capacity_ = 0;
    };
    size_type size_ = 0;
    ObjectPtr<StringImpl> impl_;

    void InitLocalBuffer() noexcept {
        std::memset(local_buffer_, '\0', local_capacity_ + 1);
    }

    NODISCARD bool IsLocal() const noexcept {
        return !defined();
    }

    NODISCARD size_type Limit(size_type pos, size_type limit) const noexcept;
    NODISCARD size_type CheckPos(size_type pos) const;
    void CheckSize(size_type delta) const;
    String& append_aux(const_pointer src, size_type n);

    template<typename Iter,
             typename = std::enable_if_t<std::is_convertible_v<
                     typename std::iterator_traits<Iter>::iterator_category,
                     std::forward_iterator_tag>>>
    void Construct(Iter first, Iter last) {
        const size_type cap = std::distance(first, last);
        char* dst = nullptr;
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

    void Construct(size_type n, char c);
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
inline bool operator==(const String& lhs, const std::string& rhs) {
    return String::MemoryEqual(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const std::string& lhs, const String& rhs) {
    return String::MemoryEqual(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}
inline bool operator==(const String& lhs, const String& rhs) {
    return String::MemoryEqual(lhs.data(), lhs.size(), rhs.data(), rhs.size());
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
