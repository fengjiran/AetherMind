//
// Created by 赵丹 on 25-7-29.
//

#ifndef AETHERMIND_ERROR_H
#define AETHERMIND_ERROR_H

#include "traceback.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace aethermind {

class Error : public std::exception {
public:
    Error(std::string kind, std::string message, std::string traceback)
        : kind_(std::move(kind)), message_(std::move(message)), traceback_(std::move(traceback)) {}

    NODISCARD const char* what() const noexcept override {
        thread_local std::string what_str = kind_ + ": " + message_;
        what_str = std::string("Traceback (most recent call last):\n") +
                   traceback_ + kind_ + ": " + message_;
        return what_str.c_str();
    }

private:
    std::string kind_;
    std::string message_;
    std::string traceback_;
};

class ErrorBuilder {
public:
    ErrorBuilder(std::string kind, std::string traceback, bool log_before_throw)
        : kind_(std::move(kind)), traceback_(std::move(traceback)), log_before_throw_(log_before_throw) {}

    std::ostringstream& stream() {
        return stream_;
    }

    ~ErrorBuilder() AETHERMIND_THROW_EXCEPTION {
        Error error(kind_, stream_.str(), traceback_);
        if (log_before_throw_) {
            std::cerr << error.what() << "\n\n";
        }
        throw error;
    }

private:
    std::string kind_;
    std::string traceback_;
    std::ostringstream stream_;
    bool log_before_throw_;
};

// define traceback here as call into traceback function
#define TRACEBACK_HERE AetherMindTraceback(__FILE__, __LINE__, FUNC_SIG)

#define AETHERMIND_THROW(ErrorKind) ErrorBuilder(#ErrorKind, TRACEBACK_HERE, true).stream()

}// namespace aethermind

#endif//AETHERMIND_ERROR_H
