#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_UTILS_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_UTILS_H

#include "aethermind/base/status.h"
#include "data_type.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace aethermind {
namespace hf {

inline std::string FormatPathMessage(std::string_view prefix,
                                      const std::filesystem::path& path) {
    return std::string(prefix) + ": " + path.string();
}

AM_NODISCARD StatusOr<DataType> ParseSafetensorsDType(std::string_view dtype_text);

AM_NODISCARD StatusOr<std::string> ReadFileText(const std::filesystem::path& path);

}// namespace hf
}// namespace aethermind

#endif
