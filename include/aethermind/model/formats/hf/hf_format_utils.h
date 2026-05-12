#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_FORMAT_UTILS_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_FORMAT_UTILS_H

#include <filesystem>
#include <string>
#include <string_view>

namespace aethermind {
namespace hf {

inline std::string FormatPathMessage(std::string_view prefix,
                                     const std::filesystem::path& path) {
    return std::string(prefix) + ": " + path.string();
}

}// namespace hf
}// namespace aethermind

#endif
