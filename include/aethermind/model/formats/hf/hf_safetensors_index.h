#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H

/// @file hf_safetensors_index.h
/// @brief Parsed model.safetensors.index.json for sharded checkpoints.

#include "aethermind/base/status.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aethermind {

/// @brief Sharded-checkpoint index mapping tensor names to shard files.
///
/// Owns only parsed metadata (weight map and optional total size); file I/O
/// happens inside Load. Constructed through the static Load factory; copies
/// are cheap and intentionally allowed.
class HfSafetensorsIndex {
public:
    /// @brief Parses a safetensors index file.
    ///
    /// @param index_path Path to model.safetensors.index.json.
    /// @return Parsed index, or an error for missing or malformed files.
    AM_NODISCARD static StatusOr<HfSafetensorsIndex> Load(
            const std::filesystem::path& index_path);

    /// @brief Tensor name to shard filename mapping.
    ///
    /// @return Borrowed map; valid while this index object is alive.
    AM_NODISCARD const std::unordered_map<std::string, std::string>& WeightMap() const noexcept {
        return weight_map_;
    }

    /// @brief Optional total size metadata from the index.
    ///
    /// @return Total checkpoint size when the index file provides it.
    AM_NODISCARD std::optional<uint64_t> TotalSize() const noexcept {
        return total_size_;
    }

    /// @brief Unique shard filenames referenced by the weight map.
    ///
    /// @return Unique shard names in sorted order.
    AM_NODISCARD std::vector<std::string> UniqueShardFilenames() const;

    // Rule of 0: no resource-managing members, but explicit to document intent
    // against the private-constructor + static-factory pattern.
    HfSafetensorsIndex(const HfSafetensorsIndex&) = default;
    HfSafetensorsIndex(HfSafetensorsIndex&&) = default;
    HfSafetensorsIndex& operator=(const HfSafetensorsIndex&) = default;
    HfSafetensorsIndex& operator=(HfSafetensorsIndex&&) = default;

    ~HfSafetensorsIndex() = default;

private:
    HfSafetensorsIndex(std::unordered_map<std::string, std::string> weight_map,
                       std::optional<uint64_t> total_size) noexcept
        : weight_map_(std::move(weight_map)), total_size_(total_size) {}

    std::unordered_map<std::string, std::string> weight_map_{};
    std::optional<uint64_t> total_size_{};
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
