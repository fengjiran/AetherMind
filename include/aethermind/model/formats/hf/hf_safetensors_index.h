#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_SAFETENSORS_INDEX_H

#include "aethermind/base/status.h"
#include "aethermind/model/resolved_tensor_index.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {

struct HfSafetensorEntry {
    std::string name{};
    DataType dtype{};
    std::vector<int64_t> shape{};
    uint64_t data_offset_begin = 0;
    uint64_t data_offset_end = 0;
    RawTensorView view{};

    AM_NODISCARD size_t ByteSize() const noexcept {
        return data_offset_end - data_offset_begin;
    }
};

class HfSafetensorsIndex {
public:
    static StatusOr<HfSafetensorsIndex> LoadSingleFile(
            const std::filesystem::path& safetensors_path);

    AM_NODISCARD const HfSafetensorEntry* Find(
            std::string_view tensor_name) const noexcept;

    AM_NODISCARD const std::vector<HfSafetensorEntry>& Entries() const noexcept {
        return entries_;
    }

    AM_NODISCARD const std::filesystem::path& Path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_{};
    std::vector<HfSafetensorEntry> entries_{};
};

}// namespace aethermind

#endif
