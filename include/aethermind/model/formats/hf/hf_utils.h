#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_UTILS_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_UTILS_H

#include "aethermind/base/status.h"
#include "data_type.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace aethermind::hf {

inline std::string FormatPathMessage(std::string_view prefix,
                                     const std::filesystem::path& path) {
    return std::string(prefix) + ": " + path.string();
}

AM_NODISCARD StatusOr<DataType> ParseSafetensorsDType(std::string_view dtype_text);

/// Rejects symlinked HF model input files before later checks follow the path.
///
/// This keeps config, index, and safetensors file handling aligned with the
/// sharded loader policy: model files must be regular filesystem entries, not
/// links to another location.
AM_NODISCARD Status RejectExistingPathIfSymlink(const std::filesystem::path& path,
                                             std::string_view file_role);

AM_NODISCARD StatusOr<std::string> ReadFileText(const std::filesystem::path& path);

}// namespace aethermind::hf

#endif
