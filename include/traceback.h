/// Traceback collection and formatting utilities.
///
/// Declares `TraceBackStorage`, `GetTracebackLimit()`, and `DetectBoundary()`
/// used by the backtrace implementation in `src/traceback.cpp`. When
/// `USE_LIBBACKTRACE` is not enabled, `src/traceback.cpp` provides stub
/// definitions so that translation units including this header still link.

#ifndef AETHERMIND_TRACEBACK_H
#define AETHERMIND_TRACEBACK_H

#include "macros.h"

#include <sstream>
#include <string>

namespace aethermind {

/// Default backtrace frame limit when `TRACEBACK_LIMIT` env var is unset.
constexpr int32_t kDefaultTracebackLimit = 512;

/// Returns the maximum number of frames to collect.
///
/// Reads the `TRACEBACK_LIMIT` environment variable if set; otherwise
/// returns `kDefaultTracebackLimit`.
int32_t GetTracebackLimit();

/// Returns true if backtrace collection should stop at the given frame.
///
/// Used to terminate collection at language boundary frames (e.g., Python
/// interpreter entries) so that higher-level stacks are reported by the
/// caller's runtime rather than duplicated here.
///
/// @param filename Source filename of the frame, may be null.
/// @param symbol Demangled symbol name of the frame, may be null.
/// @return true if collection should stop at this frame.
bool DetectBoundary(const char* filename, const char* symbol);

/// Accumulates backtrace frames into a formatted string.
///
/// Populated by the libbacktrace full callback (`BacktraceFullCallback` in
/// `src/traceback.cpp`) via `Append()`. Not thread-safe; each call to
/// `AetherMindTraceback` creates a local instance, and concurrent calls are
/// serialized by an internal mutex.
struct TraceBackStorage {
    TraceBackStorage() = default;

    /// Appends a single frame to the traceback stream.
    ///
    /// @param filename Source filename, may be null (rendered as "<unknown>").
    /// @param lineno  Source line number; 0 suppresses the line field.
    /// @param func    Symbol or function name, may be null (frames whose
    ///                symbol starts with "0x0" are skipped).
    void Append(const char* filename, int lineno, const char* func);

    AM_NODISCARD bool ExceedTracebackLimit() const {
        return line_count_ >= max_frame_size_;
    }

    AM_NODISCARD std::string GetTraceback() const {
        return traceback_stream_.str();
    }

    std::ostringstream traceback_stream_;
    size_t line_count_ = 0;
    size_t max_frame_size_ = GetTracebackLimit();
    // Number of leading frames to skip before collecting.
    size_t skip_frame_count_ = 0;
    // If true, stop collection when DetectBoundary returns true.
    bool stop_at_boundary_ = true;
};

}// namespace aethermind

#endif// AETHERMIND_TRACEBACK_H
