// Backtrace collection and formatting implementation.
//
// When USE_LIBBACKTRACE is enabled, provides full symbolicated backtraces via
// libbacktrace + cxxabi demangling with frame filtering. Otherwise, provides
// stub definitions for GetTracebackLimit/DetectBoundary and a minimal
// single-frame traceback using only the caller-provided location.

#include "traceback.h"
#include "env.h"
#include "error.h"

#if USE_LIBBACKTRACE

#include <backtrace.h>
#include <cxxabi.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>

#endif

#if BACKTRACE_ON_SEGFAULT
#include <csignal>
#endif

namespace aethermind {
#if USE_LIBBACKTRACE

int32_t GetTracebackLimit() {
    if (has_env("TRACEBACK_LIMIT")) {
        return std::stoi(get_env("TRACEBACK_LIMIT").value());
    }
    return kDefaultTracebackLimit;
}

bool DetectBoundary(AM_MAYBE_UNUSED const char* filename, const char* symbol) {
    if (symbol) {
        // Python C ABI entry points.
        if (strncmp(symbol, "slot_tp_call", 12) == 0) {
            return true;
        }

        if (strcmp(symbol, "object_is_not_callable") == 0) {
            return true;
        }

        // Python interpreter frames — stop here; the Python runtime
        // reports these from its own side.
        if (strncmp(symbol, "_Py", 3) == 0 ||
            strncmp(symbol, "PyObject", 8) == 0) {
            return true;
        }
    }
    return false;
}

void TraceBackStorage::Append(const char* filename, int lineno, const char* func) {
    // Frames with no filename are only useful if they carry a symbol.
    if (!filename) {
        if (func) {
            if (strncmp(func, "0x0", 3) == 0) {
                return;
            }
            filename = "<unknown>";
        } else {
            return;
        }
    }

    traceback_stream_ << "  File \"" << filename << "\"";
    if (lineno != 0) {
        traceback_stream_ << ", line " << lineno;
    }
    traceback_stream_ << ", in " << func << '\n';
    line_count_++;
}

// libbacktrace callback: invoked when backtrace_create_state fails.
void BacktraceCreateErrorCallback(AM_MAYBE_UNUSED void* data, const char* msg, AM_MAYBE_UNUSED int errnum) {
    std::cerr << "Could not initialize backtrace state: " << msg << std::endl;
}

// Creates the process-wide libbacktrace state.
backtrace_state* BacktraceCreate() {
    return backtrace_create_state(nullptr, 1,
                                  BacktraceCreateErrorCallback, nullptr);
}

// Returns the process-wide backtrace state, initialized on first use.
// Thread-safe per C++11 function-local static initialization rules.
backtrace_state* GetBacktraceState() {
    static backtrace_state* state = BacktraceCreate();
    return state;
}

// Demangles a C++ mangled name via abi::__cxa_demangle.
// Returns the original name if demangling fails.
std::string DemangleName(const char* name) {
    int status = 0;
    size_t length = std::strlen(name);
    char* demangled_name = abi::__cxa_demangle(name, nullptr, &length, &status);
    std::string result(name);
    if (demangled_name && status == 0 && length > 0) {
        result = demangled_name;
    }
    if (demangled_name) {
        std::free(demangled_name);
    }
    return result;
}

// Returns true for low-information frames (implementation internals, test
// harness) to exclude from the backtrace.
bool ExcludeFrame(const char* filename, const char* symbol) {
    if (filename) {
        if (strstr(filename, "src/traceback.cpp")) {
            return true;
        }

        // C++ stdlib frames
        if (strstr(filename, "include/c++/")) {
            return true;
        }
    }

    if (symbol) {
        // C++ stdlib frames
        if (strstr(symbol, "__libc_")) {
            return true;
        }

        // google test frames: match the `testing::` namespace or the bare
        // namespace name to avoid false positives on user symbols that happen
        // to contain the substring "testing".
        if (strstr(symbol, "testing::") || strcmp(symbol, "testing") == 0) {
            return true;
        }

        if (strstr(symbol, "RUN_ALL_TESTS")) {
            return true;
        }
    }

    return false;
}

// libbacktrace callback: invoked on per-frame errors. Intentionally no-op.
void BacktraceErrorCallback(AM_MAYBE_UNUSED void* data, AM_MAYBE_UNUSED const char* msg, AM_MAYBE_UNUSED int errnum) {
}

// libbacktrace callback: resolves a PC to a symbol name, writing the
// result into the `std::string*` passed via `data`.
void BacktraceSyminfoCallback(void* data, uintptr_t pc, const char* symname, AM_MAYBE_UNUSED uintptr_t symval, AM_MAYBE_UNUSED uintptr_t symsize) {
    if (data == nullptr) { return; }
    auto str = static_cast<std::string*>(data);

    if (symname != nullptr) {
        *str = DemangleName(symname);
    } else {
        std::ostringstream s;
        s << "0x" << std::setfill('0') << std::setw(sizeof(uintptr_t) * 2) << std::hex << pc;
        *str = s.str();
    }
}

// libbacktrace callback: invoked per stack frame.
// Returns 1 to stop collection, 0 to continue.
int BacktraceFullCallback(void* data, uintptr_t pc, const char* filename, int lineno,
                          const char* symbol) {
    if (data == nullptr) { return 0; }
    auto* backtrace_stk = static_cast<TraceBackStorage*>(data);
    std::string symbol_str = "<unknown>";
    if (symbol) {
        symbol_str = DemangleName(symbol);
    } else {
        backtrace_syminfo(GetBacktraceState(), pc, BacktraceSyminfoCallback,
                          BacktraceErrorCallback, &symbol_str);
    }
    symbol = symbol_str.data();

    if (backtrace_stk->ExceedTracebackLimit()) {
        return 1;
    }

    if (backtrace_stk->stop_at_boundary_ && DetectBoundary(filename, symbol)) {
        return 1;
    }

    if (backtrace_stk->skip_frame_count_ > 0) {
        backtrace_stk->skip_frame_count_--;
        return 0;
    }

    if (ExcludeFrame(filename, symbol)) {
        return 0;
    }

    backtrace_stk->Append(filename, lineno, symbol);
    return 0;
}

}// namespace aethermind

#if BACKTRACE_ON_SEGFAULT
void backtrace_handler(int sig) {
    // WARNING: Signal handlers must only call async-signal-safe functions.
    // `AetherMindTraceback` and `std::cerr <<` may allocate heap memory and
    // use non-reentrant routines, which is technically undefined behavior.
    // We accept this risk because the process is already crashing and
    // diagnostic output is more valuable than strict conformance.
    const char* backtrace = AetherMindTraceback(nullptr, 0, nullptr, 1);
    std::cerr << "!!!!!!! AetherMind encountered a Segfault !!!!!!!\n"
              << backtrace << std::endl;
    // Re-raise signal with default handler. `= {}` zero-initializes the
    // struct; no explicit memset needed.
    struct sigaction act = {};
    act.sa_flags = SA_RESETHAND;
    act.sa_handler = SIG_DFL;
    sigaction(sig, &act, nullptr);
    raise(sig);
}

__attribute__((constructor)) void install_signal_handler() {
    // NOTE: This may override previously installed signal handlers.
    struct sigaction act = {};
    act.sa_flags = 0;
    act.sa_handler = backtrace_handler;
    sigemptyset(&act.sa_mask);
    sigaction(SIGSEGV, &act, nullptr);
}
#endif

const char* AetherMindTraceback(AM_MAYBE_UNUSED const char* filename,
                                AM_MAYBE_UNUSED int lineno,
                                AM_MAYBE_UNUSED const char* func,
                                int cross_aethermind_boundary) {
    thread_local std::string traceback_str;
    aethermind::TraceBackStorage traceback;
    traceback.stop_at_boundary_ = cross_aethermind_boundary == 0;
    if (filename != nullptr && func != nullptr) {
        // Skip AetherMindTraceback and its caller — already captured above.
        traceback.skip_frame_count_ = 2;
        if (!aethermind::ExcludeFrame(filename, func)) {
            traceback.Append(filename, lineno, func);
        }
    }

    if (aethermind::GetBacktraceState() != nullptr) {
        static std::mutex m;
        std::scoped_lock<std::mutex> lock(m);
        backtrace_full(aethermind::GetBacktraceState(), 0, aethermind::BacktraceFullCallback,
                       aethermind::BacktraceErrorCallback, &traceback);
    }

    traceback_str = traceback.GetTraceback();
    return traceback_str.c_str();
}

#else

}// namespace aethermind

// Stubs for declarations in traceback.h. The non-libbacktrace build does not
// perform backtrace collection, so these are never exercised at runtime, but
// must be defined to satisfy linkers for translation units that include
// traceback.h (TraceBackStorage::max_frame_size_ references GetTracebackLimit).
int32_t aethermind::GetTracebackLimit() {
    return aethermind::kDefaultTracebackLimit;
}

bool aethermind::DetectBoundary(AM_MAYBE_UNUSED const char* filename,
                                AM_MAYBE_UNUSED const char* symbol) {
    return false;
}

// Minimal fallback: emits only the caller-provided frame without
// symbolication. Used when USE_LIBBACKTRACE is not enabled.
const char* AetherMindTraceback(const char* filename,
                                int lineno,
                                const char* func,
                                AM_MAYBE_UNUSED int cross_aethermind_boundary) {
    thread_local std::string traceback_str;
    std::ostringstream traceback_stream;
    traceback_stream << "  File \"" << (filename ? filename : "<unknown>") << "\"";
    if (lineno != 0) {
        traceback_stream << ", line " << lineno;
    }
    traceback_stream << ", in " << (func ? func : "<unknown>") << '\n';
    traceback_str = traceback_stream.str();
    return traceback_str.c_str();
}

#endif
