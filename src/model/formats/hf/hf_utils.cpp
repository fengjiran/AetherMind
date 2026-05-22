#include "aethermind/model/formats/hf/hf_utils.h"

#include <fstream>
#include <string>
#include <system_error>

namespace aethermind::hf {

StatusOr<DataType> ParseSafetensorsDType(std::string_view dtype_text) {
    const auto equals = [dtype_text](std::string_view expected) noexcept {
        return dtype_text == expected;
    };

    if (equals("F16")) {
        return DataType::Float(16);
    }

    if (equals("BF16")) {
        return DataType::BFloat(16);
    }

    if (equals("F32")) {
        return DataType::Float32();
    }

    if (equals("F64")) {
        return DataType::Float(64);
    }

    if (equals("I16")) {
        return DataType::Int(16);
    }

    if (equals("I32")) {
        return DataType::Int(32);
    }

    if (equals("I64")) {
        return DataType::Int(64);
    }

    if (equals("I8")) {
        return DataType::Int(8);
    }

    if (equals("U8")) {
        return DataType::UInt(8);
    }

    if (equals("U16")) {
        return DataType::UInt(16);
    }

    if (equals("U32")) {
        return DataType::UInt(32);
    }

    if (equals("U64")) {
        return DataType::UInt(64);
    }

    if (equals("BOOL")) {
        return DataType::Bool();
    }

    if (equals("F8_E5M2")) {
        return DataType::Float8E5M2();
    }

    if (equals("F8_E4M3")) {
        return DataType::Float8E4M3();
    }

    return Status::InvalidArgument(
            std::string("Unsupported safetensors dtype: ") + std::string(dtype_text));
}

StatusOr<std::string> ReadFileText(const std::filesystem::path& path) {
    AM_RETURN_IF_ERROR(RejectExistingPathIfSymlink(path, "HF text file"));

    std::error_code error;
    if (!std::filesystem::exists(path, error) || error) {
        if (error) {
            return Status::Internal(FormatPathMessage("Failed to stat file", path));
        }
        return Status::NotFound(FormatPathMessage("File not found", path));
    }

    if (!std::filesystem::is_regular_file(path, error) || error) {
        if (error) {
            return Status::Internal(FormatPathMessage("Failed to inspect file type", path));
        }
        return Status::InvalidArgument(FormatPathMessage("Path is not a regular file", path));
    }

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        return Status::Internal(FormatPathMessage("Failed to open file", path));
    }

    const auto end_pos = stream.tellg();
    if (end_pos < 0) {
        return Status::Internal(FormatPathMessage("Failed to determine file size", path));
    }

    if (constexpr size_t kMaxTextFileSize = 16ULL * 1024 * 1024; static_cast<size_t>(end_pos) > kMaxTextFileSize) {
        return Status::InvalidArgument(
                FormatPathMessage("File exceeds maximum size", path));
    }

    std::string text(end_pos, '\0');
    stream.seekg(0, std::ios::beg);
    if (!stream) {
        return Status::Internal(FormatPathMessage("Failed to seek file", path));
    }

    if (!text.empty()) {
        stream.read(text.data(), static_cast<std::streamsize>(text.size()));
        if (!stream) {
            return Status::Internal(FormatPathMessage("Failed to read file", path));
        }
    }
    return text;
}

Status RejectExistingPathIfSymlink(const std::filesystem::path& path,
                                std::string_view file_role) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error) {
        if (error == std::errc::no_such_file_or_directory ||
            error == std::errc::not_a_directory) {
            return Status::Ok();
        }
        std::string message = "Failed to inspect ";
        message += file_role;
        message += " symlink status";
        return Status::Internal(FormatPathMessage(message, path));
    }

    if (std::filesystem::is_symlink(status)) {
        std::string message(file_role);
        message += " must not be a symlink";
        return Status::InvalidArgument(FormatPathMessage(message, path));
    }

    return Status::Ok();
}

}// namespace aethermind::hf
