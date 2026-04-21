#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_DIRECTORY_READER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_DIRECTORY_READER_H

#include "aethermind/base/status.h"

#include <filesystem>

namespace aethermind {

enum class HfDirectoryLayout {
    kUnknown = 0,
    kSingleSafetensors,
    kShardedSafetensors,
};

struct HfDirectoryLayoutInfo {
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
    static StatusOr<HfDirectoryLayoutInfo> DiscoverLayout(
            const std::filesystem::path& model_dir);
};

}// namespace aethermind

#endif
