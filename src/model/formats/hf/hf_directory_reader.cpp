#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_format_utils.h"
#include "aethermind/model/formats/hf/hf_safetensors_index.h"

#include <system_error>

namespace aethermind {

StatusOr<HfDirectoryReader> HfDirectoryReader::Open(const std::filesystem::path& model_dir) {
    auto dir_desc = hf::InspectDirectory(model_dir);
    if (!dir_desc.ok()) {
        return dir_desc.status();
    }

    return HfDirectoryReader(std::move(*dir_desc));
}

StatusOr<RawTensorTable> HfDirectoryReader::LoadTensorTable() const {
    if (!dir_desc_.IsSingleFile()) {
        return Status(StatusCode::kUnimplemented,
                      hf::FormatPathMessage("Only single-file HF safetensors layout is implemented", dir_desc_.model_dir));
    }

    auto index = HfSafetensorsIndex::LoadSingleFile(dir_desc_.safetensors_path);
    if (!index.ok()) {
        return index.status();
    }

    RawTensorTable tensor_table;
    tensor_table.reserve(index->Entries().size());
    for (const auto& entry: index->Entries()) {
        tensor_table.emplace(entry.name, entry.view);
    }
    return tensor_table;
}

namespace hf {

StatusOr<HfDirectoryDescriptor> InspectDirectory(const std::filesystem::path& model_dir) {
    if (model_dir.empty()) {
        return Status::InvalidArgument("HF model directory path must not be empty");
    }

    std::error_code error;
    if (!std::filesystem::exists(model_dir, error) || error) {
        if (error) {
            return Status::Internal(
                    FormatPathMessage("Failed to stat HF model directory", model_dir));
        }
        return Status::NotFound(
                FormatPathMessage("HF model directory not found", model_dir));
    }

    if (!std::filesystem::is_directory(model_dir, error) || error) {
        if (error) {
            return Status::Internal(
                    FormatPathMessage("Failed to inspect HF model directory type", model_dir));
        }
        return Status::InvalidArgument(
                FormatPathMessage("HF model path is not a directory", model_dir));
    }

    const auto config_path = model_dir / "config.json";
    const auto safetensors_path = model_dir / "model.safetensors";
    const auto safetensors_index_path = model_dir / "model.safetensors.index.json";

    if (!std::filesystem::exists(config_path, error) || error) {
        if (error) {
            return Status::Internal(
                    FormatPathMessage("Failed to stat config.json", config_path));
        }
        return Status::NotFound(
                FormatPathMessage("HF model directory is missing config.json", config_path));
    }

    const bool has_single_file = std::filesystem::exists(safetensors_path, error);
    if (error) {
        return Status::Internal(
                FormatPathMessage("Failed to stat model.safetensors", safetensors_path));
    }

    const bool has_sharded_index = std::filesystem::exists(safetensors_index_path, error);
    if (error) {
        return Status::Internal(
                FormatPathMessage("Failed to stat model.safetensors.index.json", safetensors_index_path));
    }

    if (has_single_file && has_sharded_index) {
        return Status(StatusCode::kFailedPrecondition,
                      FormatPathMessage("HF model directory has conflicting single-file and sharded safetensors layouts",
                                        model_dir));
    }

    if (has_single_file) {
        return HfDirectoryDescriptor{
                .layout = HfDirectoryLayout::kSingleSafetensors,
                .model_dir = model_dir,
                .config_path = config_path,
                .safetensors_path = safetensors_path,
                .safetensors_index_path = {},
        };
    }

    if (has_sharded_index) {
        return Status(StatusCode::kUnimplemented,
                      FormatPathMessage("Sharded HF safetensors layout is not implemented yet", model_dir));
    }

    return Status::NotFound(
            FormatPathMessage("HF model directory is missing both model.safetensors and model.safetensors.index.json",
                              model_dir));
}

}// namespace hf
}// namespace aethermind
