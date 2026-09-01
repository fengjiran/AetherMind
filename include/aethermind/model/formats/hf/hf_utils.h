#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_UTILS_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_UTILS_H

/// @file hf_utils.h
/// @brief Small shared helpers for HF model directory I/O.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace aethermind::hf {

inline std::string FormatPathMessage(std::string_view prefix,
                                     const std::filesystem::path& path) {
    return std::string(prefix) + ": " + path.string();
}

/// @brief Maps a safetensors dtype string (e.g. "F32", "BF16") to DataType.
///
/// @param dtype_text Dtype string from the safetensors header.
/// @return Matching DataType, or InvalidArgument for unknown names.
StatusOr<DataType> ParseSafetensorsDType(std::string_view dtype_text);

/// @brief Rejects symlinked HF model input files before later checks follow the path.
///
/// This keeps config, index, and safetensors file handling aligned with the
/// sharded loader policy: model files must be regular filesystem entries, not
/// links to another location.
///
/// @param path Candidate file path.
/// @param file_role Human-readable file role (e.g. "config.json") for errors.
/// @return InvalidArgument when the path exists and is a symlink.
Status RejectExistingPathIfSymlink(const std::filesystem::path& path,
                                   std::string_view file_role);

/// @brief Reads a file as text.
///
/// @param path File to read.
/// @return File contents, or an error when the file cannot be read.
StatusOr<std::string> ReadFileText(const std::filesystem::path& path);

} // namespace aethermind::hf

#endif
