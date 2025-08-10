//
// Created by richard on 8/9/25.
//

#ifndef AETHERMIND_TRACEBACK_H
#define AETHERMIND_TRACEBACK_H

#include "env.h"
#include "macros.h"

#include <cstring>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

namespace aethermind {

inline int32_t GetTracebackLimit() {
    if (has_env("TRACEBACK_LIMIT")) {
        return std::stoi(get_env("TRACEBACK_LIMIT").value());
    }
    return 512;
}

class TraceBackStorage {
public:
    TraceBackStorage() : max_frame_size(GetTracebackLimit()) {}

    void Append(const char* filename, const char* func, int lineno) {
        // skip frames with empty filename
        if (filename == nullptr) {
            if (func != nullptr) {
                if (strncmp(func, "0x0", 3) == 0) {
                    return;
                }
                filename = "<unknown>";
            } else {
                return;
            }
        }

        std::ostringstream traceback_stream;
        traceback_stream << "  File \"" << filename << "\"";
        if (lineno != 0) {
            traceback_stream << ", line " << lineno;
        }
        traceback_stream << ", in " << func << '\n';
        lines.push_back(traceback_stream.str());
    }

    NODISCARD bool ExceedTracebackLimit() const {
        return lines.size() >= max_frame_size;
    }

    NODISCARD std::string GetTraceback() const {
        std::string traceback;
        for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
            traceback += *it;
        }
        return traceback;
    }

private:
    std::vector<std::string> lines;
    size_t max_frame_size;
};

const char* AetherMindTraceback(const char* filename, int lineno, const char* func);

}// namespace aethermind

#endif//AETHERMIND_TRACEBACK_H
