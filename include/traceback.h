//
// Created by richard on 8/9/25.
//

#ifndef AETHERMIND_TRACEBACK_H
#define AETHERMIND_TRACEBACK_H

#include "macros.h"

#include <cstring>
#include <ranges>
#include <sstream>
#include <string>

namespace aethermind {

int32_t GetTracebackLimit();

/**
 * \brief List frames that should stop the backtrace.
 * \param filename The filename of the frame.
 * \param symbol The symbol name of the frame.
 * \return true if the frame should stop the backtrace.
 * \note We stop backtrace at the boundary.
 */
bool DetectBoundary(const char* filename, const char* symbol);

struct TraceBackStorage {
    TraceBackStorage() = default;

    void Append(const char* filename, int lineno, const char* func);

    NODISCARD bool ExceedTracebackLimit() const {
        return line_count_ >= max_frame_size_;
    }

    NODISCARD std::string GetTraceback() const {
        return traceback_stream_.str();
    }

    /*! \brief The stream to store the backtrace. */
    std::ostringstream traceback_stream_;
    /*! \brief The number of lines in the backtrace. */
    size_t line_count_ = 0;
    size_t max_frame_size_ = GetTracebackLimit();
    /*! \brief Number of frames to skip. */
    size_t skip_frame_count_ = 0;
    /*! \brief Whether to stop at the boundary. */
    bool stop_at_boundary_ = true;
};

}// namespace aethermind

#endif//AETHERMIND_TRACEBACK_H
