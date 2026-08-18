#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_FILE_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_FILE_H

/// @file hf_safetensors_file.h
/// @brief Zero-copy reader for a single safetensors checkpoint file.

#include "aethermind/base/status.h"
#include "aethermind/model/raw_weight.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aethermind {

/// @brief One tensor entry from a safetensors file header.
///
/// data_offset_begin/end are absolute byte offsets within the file's data
/// region; the file reader validates that they tile the region contiguously.
/// `view` shares the file's mmap-backed storage.
struct HfSafetensorsEntry {
    std::string name{};
    DataType dtype{};
    std::vector<int64_t> shape{};
    uint64_t data_offset_begin = 0;
    uint64_t data_offset_end = 0;
    RawWeightView view{};

    AM_NODISCARD size_t ByteSize() const noexcept {
        return data_offset_end - data_offset_begin;
    }
};

/// @brief Opens a safetensors checkpoint as an immutable mmap-backed table.
///
/// All entries share one RawStorage; views remain valid as long as this
/// object (or a copy of its storage) is alive. The mapped file must not be
/// modified in place (see Open).
class HfSafetensorsFile {
public:
    /// @brief Loads a single-file safetensors checkpoint via zero-copy mmap.
    ///
    /// The checkpoint file is treated as immutable for the full lifetime of the
    /// returned file object and any RawWeightView copied from it. Writers must
    /// publish updates by writing a new file and atomically renaming it into
    /// place; truncating or modifying the mapped file in place can make later
    /// view access fault with SIGBUS.
    ///
    /// @param safetensors_path Path to the .safetensors checkpoint.
    /// @return File object with parsed entries, or an error for missing or
    /// malformed files.
    static StatusOr<HfSafetensorsFile> Open(const std::filesystem::path& safetensors_path);

    /// @brief Looks up a tensor entry by name.
    ///
    /// @param tensor_name Tensor name as recorded in the file header.
    /// @return Borrowed entry pointer, or nullptr when absent. The pointer
    /// stays valid while this file object is alive.
    AM_NODISCARD const HfSafetensorsEntry* Find(std::string_view tensor_name) const;

    AM_NODISCARD const std::vector<HfSafetensorsEntry>& Entries() const noexcept {
        return entries_;
    }

    AM_NODISCARD const std::filesystem::path& Path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_{};
    std::shared_ptr<const RawStorage> storage_{};
    std::vector<HfSafetensorsEntry> entries_{};
    std::unordered_map<std::string, size_t> name_index_{};
};

}// namespace aethermind

#endif
