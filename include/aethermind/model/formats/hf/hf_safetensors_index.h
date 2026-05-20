#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H

#include "aethermind/base/status.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aethermind {

class HfSafetensorsIndex {
public:
    AM_NODISCARD static StatusOr<HfSafetensorsIndex> Load(
            const std::filesystem::path& index_path);

    AM_NODISCARD const std::unordered_map<std::string, std::string>& WeightMap() const noexcept {
        return weight_map_;
    }

    AM_NODISCARD std::optional<uint64_t> TotalSize() const noexcept {
        return total_size_;
    }

    AM_NODISCARD std::vector<std::string> UniqueShardFilenames() const;

private:
    HfSafetensorsIndex(std::unordered_map<std::string, std::string> weight_map,
                       std::optional<uint64_t> total_size) noexcept
        : weight_map_(std::move(weight_map)), total_size_(total_size) {}

    std::unordered_map<std::string, std::string> weight_map_{};
    std::optional<uint64_t> total_size_{};
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
