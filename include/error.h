//
// Created by 赵丹 on 25-7-29.
//

#ifndef AETHERMIND_ERROR_H
#define AETHERMIND_ERROR_H

#include "c_api.h"
#include "traceback.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <utility>

namespace aethermind {

class Error : public std::exception {
public:
    Error(String kind, String message, String traceback)
        : kind_(std::move(kind)), message_(std::move(message)), traceback_(std::move(traceback)) {}

    NODISCARD String kind() const {
        return kind_;
    }

    NODISCARD String message() const {
        return message_;
    }

    NODISCARD String traceback() const {
        return traceback_;
    }

    NODISCARD const char* what() const noexcept override {
        thread_local String what_str = kind_ + ": " + message_;
        what_str = "Traceback (most recent call last):\n" +
                   TracebackMostRecent() + kind_ + ": " + message_;
        return what_str.c_str();
    }

    static void UpdateTraceback(Error* e, const String& traceback, uint8_t update_mode) {
        if (update_mode == kBacktraceUpdateModeReplace) {
            e->traceback_ = traceback;
        } else {
            e->traceback_ = e->traceback_ + traceback;
        }
    }

    /*!
   * \brief Get the traceback in the order of most recent call last.
   *
   * \return The traceback of the error object.
   */
    NODISCARD String TracebackMostRecent() const {
        std::vector<int64_t> delimiter = {-1};
        for (size_t i = 0; i < traceback_.size(); ++i) {
            if (traceback_[i] == '\n') {
                delimiter.push_back(static_cast<int64_t>(i));
            }
        }

        String res;
        for (size_t i = delimiter.size() - 1; i > 0; i--) {
            int64_t start = delimiter[i - 1] + 1;
            int64_t end = delimiter[i];
            if (start == end) {
                continue;
            }
            res = res + String(traceback_.data() + start, end - start) + "\n";
        }
        return res;
    }

private:
    String kind_;
    String message_;
    String traceback_;
};

class ErrorBuilder {
public:
    ErrorBuilder(String kind, String traceback, bool log_before_throw)
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
    String kind_;
    String traceback_;
    std::ostringstream stream_;
    bool log_before_throw_;
};

// define traceback here as call into traceback function
#define TRACEBACK_HERE AetherMindTraceback(__FILE__, __LINE__, FUNC_SIG, 0)

#define AM_THROW(ErrorKind) ErrorBuilder(#ErrorKind, TRACEBACK_HERE, true).stream()

}// namespace aethermind

#endif//AETHERMIND_ERROR_H
