#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_DIRECTORY_READER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_DIRECTORY_READER_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_config.h"
#include "aethermind/model/resolved_tensor_index.h"

#include <filesystem>
#include <utility>

namespace aethermind {

enum class HfDirectoryLayout {
    kUnknown = 0,
    kSingleSafetensors,
    kShardedSafetensors,
};

struct HfDirectoryDescriptor {
    HfDirectoryLayout layout = HfDirectoryLayout::kUnknown;
    std::filesystem::path model_dir{};
    std::filesystem::path config_path{};
    std::filesystem::path safetensors_path{};
    std::filesystem::path safetensors_index_path{};

    AM_NODISCARD bool IsSingleFile() const noexcept {
        return layout == HfDirectoryLayout::kSingleSafetensors;
    }

    AM_NODISCARD bool IsSharded() const noexcept {
        return layout == HfDirectoryLayout::kShardedSafetensors;
    }
};

class HfDirectoryReader {
public:
    AM_NODISCARD static StatusOr<HfDirectoryReader> Open(const std::filesystem::path& model_dir);

    AM_NODISCARD const HfDirectoryDescriptor& Layout() const noexcept {
        return dir_desc_;
    }

    AM_NODISCARD StatusOr<RawTensorTable> LoadTensorTable() const;

    AM_NODISCARD StatusOr<ModelConfig> ParseConfig() const;

private:
    explicit HfDirectoryReader(HfDirectoryDescriptor dir_desc) noexcept
        : dir_desc_(std::move(dir_desc)) {}

    HfDirectoryDescriptor dir_desc_{};
};

namespace hf {
StatusOr<HfDirectoryDescriptor> InspectDirectory(const std::filesystem::path& model_dir);
}// namespace hf

}// namespace aethermind

#endif
