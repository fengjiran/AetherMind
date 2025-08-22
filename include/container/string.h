//
// Created by 赵丹 on 2025/8/22.
//

#ifndef AETHERMIND_STRING_H
#define AETHERMIND_STRING_H

#include "object.h"

namespace aethermind {

class StringImpl : public Object {
public:
    StringImpl() : data_(nullptr), size_(0) {}

    StringImpl(const char* data, size_t size);

private:
    const char* data_;
    size_t size_;
};

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

    String(const String&) = default;
    String(String&&) noexcept = default;
    String& operator=(const String&) & = default;
    String& operator=(String&&) & noexcept = default;
    String& operator=(const String&) && = default;
    String& operator=(String&&) && noexcept = default;

    void swap(String& other) noexcept {
        std::swap(impl_, other.impl_);
    }



private:
    ObjectPtr<StringImpl> impl_;


};

}// namespace aethermind

#endif//AETHERMIND_STRING_H
