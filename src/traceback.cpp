//
// Created by richard on 8/9/25.
//

#include "traceback.h"
#include "env.h"
#include "error.h"

#if USE_LIBBACKTRACE

#include <backtrace.h>
#include <cxxabi.h>

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
    return 512;
}

bool DetectBoundary(MAYBE_UNUSED const char* filename, const char* symbol) {
    if (symbol) {
        // if (strncmp(symbol, "TVMFFIFunctionCall", 18) == 0) {
        //     return true;
        // }

        // python ABI functions
        if (strncmp(symbol, "slot_tp_call", 12) == 0) {
            return true;
        }

        if (strncmp(symbol, "object_is_not_callable", 11) == 0) {
            return true;
        }

        // Python interpreter stack frames
        // we stop backtrace at the Python interpreter stack frames
        // since these frame will be handled from by the python side.
        if (strncmp(symbol, "_Py", 3) == 0 ||
            strncmp(symbol, "PyObject", 8) == 0) {
            return true;
        }
    }
    return false;
}

void TraceBackStorage::Append(const char* filename, int lineno, const char* func) {
    // skip frames with empty filename
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

void BacktraceCreateErrorCallback(void*, const char* msg, int) {
    std::cerr << "Could not initialize backtrace state: " << msg << std::endl;
}

backtrace_state* BacktraceCreate() {
    return backtrace_create_state(nullptr, 1,
                                  BacktraceCreateErrorCallback, nullptr);
}

static backtrace_state* _bt_state = BacktraceCreate();

String DemangleName(String name) {
    int status = 0;
    size_t length = name.size();
    char* demangled_name = abi::__cxa_demangle(name.c_str(), nullptr, &length, &status);
    if (demangled_name && status == 0 && length > 0) {
        name = demangled_name;
    }
    if (demangled_name) {
        std::free(demangled_name);
    }
    return name;
}

/*!
 * \brief List frame patterns that should be excluded as they contain less information
 */
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

        // google test frames
        if (strstr(symbol, "testing")) {
            return true;
        }

        if (strstr(symbol, "RUN_ALL_TESTS")) {
            return true;
        }
    }

    return false;
}

void BacktraceErrorCallback(void*, const char*, int) {
    // do nothing
}

void BacktraceSyminfoCallback(void* data, uintptr_t pc, const char* symname, uintptr_t, uintptr_t) {
    auto str = static_cast<String*>(data);

    if (symname != nullptr) {
        *str = DemangleName(symname);
    } else {
        std::ostringstream s;
        s << "0x" << std::setfill('0') << std::setw(sizeof(uintptr_t) * 2) << std::hex << pc;
        *str = s.str();
    }
}

int BacktraceFullCallback(void* data, uintptr_t pc, const char* filename, int lineno,
                          const char* symbol) {
    auto* backtrace_stk = static_cast<TraceBackStorage*>(data);
    String symbol_str = "<unknown>";
    if (symbol) {
        symbol_str = DemangleName(symbol);
    } else {
        backtrace_syminfo(_bt_state, pc, BacktraceSyminfoCallback,
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
    // Technically we shouldn't do any allocation in a signal handler, but
    // Backtrace may allocate. What's the worst it could do? We're already
    // crashing.
    const char* backtrace = AetherMindTraceback(nullptr, 0, nullptr, 1);
    std::cerr << "!!!!!!! AetherMind encountered a Segfault !!!!!!!\n"
              << backtrace << std::endl;
    // Re-raise signal with default handler
    struct sigaction act = {};
    std::memset(&act, 0, sizeof(struct sigaction));
    act.sa_flags = SA_RESETHAND;
    act.sa_handler = SIG_DFL;
    sigaction(sig, &act, nullptr);
    raise(sig);
}

__attribute__((constructor)) void install_signal_handler() {
    // this may override already installed signal handlers
    std::signal(SIGSEGV, backtrace_handler);
}
#endif

const char* AetherMindTraceback(MAYBE_UNUSED const char* filename,
                                MAYBE_UNUSED int lineno,
                                MAYBE_UNUSED const char* func,
                                int cross_aethermind_boundary) {
    thread_local aethermind::String traceback_str;
    aethermind::TraceBackStorage traceback;
    traceback.stop_at_boundary_ = cross_aethermind_boundary == 0;
    if (filename != nullptr && func != nullptr) {
        // need to skip AetherMindTraceback and the caller function
        // which is already included in filename and func.
        traceback.skip_frame_count_ = 2;
        if (!aethermind::ExcludeFrame(filename, func)) {
            traceback.Append(filename, lineno, func);
        }
    }

    if (aethermind::_bt_state != nullptr) {
        static std::mutex m;
        std::scoped_lock<std::mutex> lock(m);
        backtrace_full(aethermind::_bt_state, 0, aethermind::BacktraceFullCallback,
                       aethermind::BacktraceErrorCallback, &traceback);
    }

    traceback_str = traceback.GetTraceback();
    return traceback_str.c_str();
}

#else

const char* AetherMindTraceback(const char* filename,
                                int lineno,
                                const char* func,
                                int cross_aethermind_boundary) {
    thread_local std::string traceback_str;
    std::ostringstream traceback_stream;
    traceback_stream << "  File \"" << filename << "\", line " << lineno << ", in " << func << '\n';
    traceback_str = traceback_stream.str();
    return traceback_str.c_str();
}

#endif
