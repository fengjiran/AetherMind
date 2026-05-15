#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_FILE_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_FILE_H

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

class HfSafetensorsFile {
public:
    /// Loads a single-file safetensors checkpoint using the default mmap zero-copy path.
    ///
    /// The checkpoint file is treated as immutable for the full lifetime of the returned
    /// file object and any RawWeightView copied from it. Writers must publish updates by writing
    /// a new file and atomically renaming it into place; truncating or modifying the mapped
    /// file in place can make later view access fault with SIGBUS.
    static StatusOr<HfSafetensorsFile> Open(const std::filesystem::path& safetensors_path);

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
