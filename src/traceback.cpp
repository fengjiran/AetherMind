//
// Created by richard on 8/9/25.
//

#include "traceback.h"
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

void BacktraceCreateErrorCallback(void*, const char* msg, int) {
    std::cerr << "Could not initialize backtrace state: " << msg << std::endl;
}

backtrace_state* BacktraceCreate() {
    return backtrace_create_state(nullptr, 1, BacktraceCreateErrorCallback, nullptr);
}

static backtrace_state* _bt_state = BacktraceCreate();

std::string DemangleName(std::string name) {
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
    auto str = static_cast<std::string*>(data);

    if (symname != nullptr) {
        *str = DemangleName(symname);
    } else {
        std::ostringstream s;
        s << "0x" << std::setfill('0') << std::setw(sizeof(uintptr_t) * 2) << std::hex << pc;
        *str = s.str();
    }
}

int BacktraceFullCallback(void* data, uintptr_t pc, const char* filename, int lineno, const char* symbol) {
    auto* trace_stk = static_cast<TraceBackStorage*>(data);
    std::string symbol_str = "<unknown>";
    if (symbol) {
        symbol_str = DemangleName(symbol);
    } else {
        backtrace_syminfo(_bt_state, pc, BacktraceSyminfoCallback, BacktraceErrorCallback, &symbol_str);
    }
    symbol = symbol_str.data();

    if (trace_stk->ExceedTracebackLimit()) {
        return 1;
    }

    if (ExcludeFrame(filename, symbol)) {
        return 0;
    }

    trace_stk->Append(filename, symbol, lineno);
    return 0;
}

std::string Traceback() {
    TraceBackStorage traceback;
    if (_bt_state == nullptr) {
        return "";
    }

    // libbacktrace eats memory if run on multiple threads at the same time, so we guard against it
    {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        backtrace_full(_bt_state, 0, BacktraceFullCallback, BacktraceErrorCallback, &traceback);
    }
    return traceback.GetTraceback();
}

#if BACKTRACE_ON_SEGFAULT
void backtrace_handler(int sig) {
    // Technically we shouldn't do any allocation in a signal handler, but
    // Backtrace may allocate. What's the worst it could do? We're already
    // crashing.
    std::cerr << "!!!!!!! AetherMind encountered a Segfault !!!!!!!\n"
              << Traceback() << std::endl;
    // Re-raise signal with default handler
    struct sigaction act = {};
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

const char* AetherMindTraceback(const char*, int, const char*) {
    thread_local std::string traceback_str = Traceback();
    return traceback_str.c_str();
}

#else

const char* AetherMindTraceback(const char* filename, int lineno, const char* func) {
    thread_local std::string traceback_str;
    std::ostringstream traceback_stream;
    traceback_stream << "  File \"" << filename << "\", line " << lineno << ", in " << func << '\n';
    traceback_str = traceback_stream.str();
    return traceback_str.c_str();
}

#endif

}// namespace aethermind
