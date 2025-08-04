//
// Created by 赵丹 on 25-7-29.
//

#ifndef AETHERMIND_ERROR_H
#define AETHERMIND_ERROR_H

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace aethermind {

class Error : public std::exception {
public:
    Error(const std::string& kind, const std::string& message)
        : kind_(kind), message_(message) {}

    const char* what() const noexcept override {
        thread_local std::string what_str = kind_ + ": " + message_;
        return what_str.c_str();
    }

private:
    std::string kind_;
    std::string message_;
};

class ErrorBuilder {
public:
    ErrorBuilder(const std::string& kind, bool log_before_throw)
        : kind_(kind), log_before_throw_(log_before_throw) {}

    std::ostringstream& stream() {
        return stream_;
    }

    ~ErrorBuilder() noexcept(false) {
        Error error(kind_, stream_.str());
        if (log_before_throw_) {
            std::cerr << error.what() << std::endl;
        }
        throw error;
    }

private:
    std::string kind_;
    std::ostringstream stream_;
    bool log_before_throw_;
};

#define ATP_THROW(ErrorKind) ErrorBuilder(#ErrorKind, true).stream()

}// namespace aethermind

#endif//AETHERMIND_ERROR_H
