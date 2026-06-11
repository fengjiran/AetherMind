#include "aethermind/model/formats/hf/hf_safetensors_file.h"
#include "aethermind/base/mmap_file.h"
#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "aethermind/model/formats/hf/hf_utils.h"
#include "aethermind/utils/overflow_check.h"
#include "utils/logging.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aethermind {

namespace {

class MmapRawStorage final : public RawStorage {
public:
    explicit MmapRawStorage(MemoryMappedFile mmap) noexcept
        : mmap_(std::move(mmap)) {}

    AM_NODISCARD const std::byte* data() const noexcept {
        return mmap_.ByteData();
    }

    AM_NODISCARD size_t size() const noexcept {
        return mmap_.size();
    }

    AM_NODISCARD Status Advise(MemoryMappedFile::Advice advice) const {
        return mmap_.Advise(advice);
    }

private:
    // RawWeightView instances may keep this mapping alive after Open returns.
    // The checkpoint file must not be truncated while mapped; later reads could SIGBUS.
    MemoryMappedFile mmap_;
};

StatusOr<uint64_t> ParseLittleEndianU64(std::span<const std::byte> bytes) {
    if (bytes.size() < sizeof(uint64_t)) {
        return Status::InvalidArgument(
                "Safetensors file is too small to contain header length");
    }

    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        value |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[i])) << (i * 8U);
    }
    return value;
}

StatusOr<uint64_t> CheckedMultiply(uint64_t lhs, uint64_t rhs, std::string_view context) {
    uint64_t product = 0;
    if (CheckOverflowMul(lhs, rhs, &product)) {
        return Status::OutOfRange(
                std::string("Integer overflow while computing ") + std::string(context));
    }
    return product;
}

}// namespace

namespace hf {

class SafetensorsHeaderParser : public HfJsonReader {
public:
    SafetensorsHeaderParser(std::string_view input,
                            const std::shared_ptr<const RawStorage>& storage,
                            const std::byte* data_base,
                            size_t data_size) noexcept
        : HfJsonReader(input), storage_(storage), data_base_(data_base), data_size_(data_size) {
        AM_CHECK(data_base_ != nullptr || data_size_ == 0,
                 "data_base_ is null but data_size_ is non-zero");
    }

    StatusOr<std::vector<HfSafetensorsEntry>> Parse() {
        SkipWhitespace();
        if (!TryConsume('{')) {
            return Status::InvalidArgument("Safetensors header must start with a JSON object");
        }

        std::vector<HfSafetensorsEntry> entries;
        SkipWhitespace();
        if (TryConsume('}')) {
            return entries;
        }

        while (true) {
            const auto key = ParseString();
            if (!key.ok()) {
                return key.status();
            }

            AM_RETURN_IF_ERROR(Expect(':', "after safetensors header key"));

            if (*key == "__metadata__") {
                Status skip_status = SkipValue();
                if (!skip_status.ok()) {
                    return skip_status;
                }
            } else {
                if (!seen_names_.insert(*key).second) {
                    return Status::InvalidArgument(
                            std::string("Duplicate safetensors tensor name: '") + *key + "'");
                }

                HfSafetensorsEntry entry;
                if (const Status entry_status = ParseTensorEntry(*key, &entry); !entry_status.ok()) {
                    return entry_status.WithMessage(
                            std::string("Safetensors tensor '") + *key + "': " + entry_status.message());
                }
                entries.push_back(std::move(entry));
            }

            SkipWhitespace();
            if (TryConsume('}')) {
                break;
            }
            AM_RETURN_IF_ERROR(Expect(',', "between safetensors header entries"));
        }

        SkipWhitespace();
        if (!AtEnd()) {
            return Status::InvalidArgument(
                    "Safetensors header contains trailing JSON content");
        }

        if (data_size_ > 0 || !entries.empty()) {
            std::vector<const HfSafetensorsEntry*> sorted;
            sorted.reserve(entries.size());
            for (const auto& e: entries) {
                sorted.push_back(&e);
            }
            std::ranges::sort(sorted,
                              [](const HfSafetensorsEntry* a, const HfSafetensorsEntry* b) noexcept {
                                  return a->data_offset_begin < b->data_offset_begin;
                              });

            for (size_t i = 0; i < sorted.size(); ++i) {
                if (i == 0 && sorted[i]->data_offset_begin != 0) {
                    return Status::InvalidArgument(
                            "First safetensors tensor data_offset_begin is not 0");
                }

                if (i > 0 && sorted[i]->data_offset_begin != sorted[i - 1]->data_offset_end) {
                    return Status::InvalidArgument(
                            "Safetensors tensor data_offsets overlap or contain gaps");
                }
            }

            if (sorted.back()->data_offset_end != data_size_) {
                return Status::InvalidArgument(
                        "Last safetensors tensor data_offset_end does not cover entire data region");
            }
        }

        return entries;
    }

private:
    Status ParseTensorEntry(const std::string& name, HfSafetensorsEntry* entry) {
        AM_RETURN_IF_ERROR(Expect('{', "at start of safetensors tensor entry"));

        std::optional<DataType> dtype;
        std::vector<int64_t> shape;
        std::optional<std::pair<uint64_t, uint64_t>> data_offsets;

        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }

                AM_RETURN_IF_ERROR(Expect(':', "after safetensors tensor field name"));

                if (*key == "dtype") {
                    const auto dtype_text = ParseString();
                    if (!dtype_text.ok()) {
                        return dtype_text.status();
                    }

                    const auto parsed_dtype = hf::ParseSafetensorsDType(*dtype_text);
                    if (!parsed_dtype.ok()) {
                        return parsed_dtype.status();
                    }
                    dtype = *parsed_dtype;
                } else if (*key == "shape") {
                    auto parsed_shape = ParseInt64Array();
                    if (!parsed_shape.ok()) {
                        return parsed_shape.status();
                    }
                    shape = std::move(*parsed_shape);
                } else if (*key == "data_offsets") {
                    const auto parsed_offsets = ParseOffsetPair();
                    if (!parsed_offsets.ok()) {
                        return parsed_offsets.status();
                    }
                    data_offsets = *parsed_offsets;
                } else {
                    Status skip_status = SkipValue();
                    if (!skip_status.ok()) {
                        return skip_status;
                    }
                }

                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between safetensors tensor fields"));
            }
        }

        if (!dtype.has_value()) {
            return Status::InvalidArgument("Safetensors tensor entry is missing required 'dtype' field");
        }

        if (!data_offsets.has_value()) {
            return Status::InvalidArgument("Safetensors tensor entry is missing required 'data_offsets' field");
        }

        const uint64_t begin = data_offsets->first;
        const uint64_t end = data_offsets->second;
        if (begin > end) {
            return Status::InvalidArgument("Safetensors tensor entry has data_offsets begin > end");
        }

        if (end > data_size_) {
            return Status::OutOfRange(
                    "Safetensors tensor entry points outside the raw data region");
        }

        uint64_t numel = 1;
        for (const int64_t dim: shape) {
            if (dim < 0) {
                return Status::InvalidArgument("Safetensors tensor entry has negative shape dimension");
            }

            const auto updated_numel = CheckedMultiply(numel,
                                                       static_cast<uint64_t>(dim), "safetensors shape");
            if (!updated_numel.ok()) {
                return updated_numel.status();
            }
            numel = *updated_numel;
        }

        const auto expected_nbytes = CheckedMultiply(numel,
                                                     static_cast<uint64_t>(dtype->nbytes()), "safetensors tensor byte size");
        if (!expected_nbytes.ok()) {
            return expected_nbytes.status();
        }

        if (*expected_nbytes != end - begin) {
            return Status::InvalidArgument(
                    "Safetensors tensor shape and dtype do not match data_offsets byte size");
        }

        entry->name = name;
        entry->dtype = *dtype;
        entry->shape = shape;
        entry->data_offset_begin = begin;
        entry->data_offset_end = end;
        entry->view = RawWeightView{
                .data = data_base_ + begin,
                .bytes = end - begin,
                .dtype = *dtype,
                .shape = shape,
                .storage = storage_,
        };
        return Status::Ok();
    }

    StatusOr<std::pair<uint64_t, uint64_t>> ParseOffsetPair() {
        AM_RETURN_IF_ERROR(Expect('[', "at start of data_offsets"));

        const auto first = ParseUInt64();
        if (!first.ok()) {
            return first.status();
        }
        AM_RETURN_IF_ERROR(Expect(',', "between data_offsets values"));
        const auto second = ParseUInt64();
        if (!second.ok()) {
            return second.status();
        }

        SkipWhitespace();
        if (!TryConsume(']')) {
            return Status::InvalidArgument("Expected closing ']' after data_offsets values");
        }
        return std::make_pair(*first, *second);
    }

    std::unordered_set<std::string> seen_names_{};
    std::shared_ptr<const RawStorage> storage_{};
    const std::byte* data_base_ = nullptr;
    size_t data_size_ = 0;
};

}// namespace hf

StatusOr<HfSafetensorsFile> HfSafetensorsFile::Open(const std::filesystem::path& safetensors_path) {
    AM_RETURN_IF_ERROR(hf::RejectExistingPathIfSymlink(safetensors_path, "HF safetensors file"));

    auto mmap = MemoryMappedFile::Map(safetensors_path);
    if (!mmap.ok()) {
        return mmap.status();
    }

    const auto storage = std::make_shared<MmapRawStorage>(std::move(*mmap));
    // Safetensors views are retained for later tensor-level access. The random-access hint
    // avoids aggressive whole-file readahead, but remains best-effort and non-fatal.
    (void) storage->Advise(MemoryMappedFile::Advice::kRandom);
    if (storage->size() < sizeof(uint64_t)) {
        return Status::InvalidArgument(
                hf::FormatPathMessage("Safetensors file is too small", safetensors_path));
    }

    const auto header_length = ParseLittleEndianU64(
            {storage->data(), sizeof(uint64_t)});
    if (!header_length.ok()) {
        return Status::InvalidArgument(
                hf::FormatPathMessage(header_length.status().message(), safetensors_path));
    }

    constexpr uint64_t kMaxHeaderSize = 16ULL * 1024 * 1024;
    if (*header_length > kMaxHeaderSize) {
        return Status::InvalidArgument(
                hf::FormatPathMessage("Safetensors header exceeds maximum size (16MB)", safetensors_path));
    }

    constexpr uint64_t kHeaderBegin = sizeof(uint64_t);
    uint64_t header_end = 0;
    if (CheckOverflowAdd(kHeaderBegin, *header_length, &header_end)) {
        return Status::InvalidArgument(
                hf::FormatPathMessage("Safetensors header length overflow", safetensors_path));
    }

    if (header_end > storage->size()) {
        return Status::InvalidArgument(
                hf::FormatPathMessage("Safetensors header length exceeds file size", safetensors_path));
    }

    const auto* header_chars = reinterpret_cast<const char*>(storage->data() + kHeaderBegin);
    const std::string_view header_json(header_chars, *header_length);
    const std::byte* data_base = storage->data() + header_end;
    const size_t data_size = storage->size() - header_end;

    auto parsed_entries = hf::SafetensorsHeaderParser(header_json, storage, data_base, data_size).Parse();
    if (!parsed_entries.ok()) {
        return parsed_entries.status().WithMessage(
                hf::FormatPathMessage(parsed_entries.status().message(), safetensors_path));
    }

    HfSafetensorsFile file;
    file.path_ = safetensors_path;
    file.storage_ = storage;
    file.entries_ = std::move(*parsed_entries);
    for (size_t i = 0; i < file.entries_.size(); ++i) {
        file.name_index_[file.entries_[i].name] = i;
    }
    return file;
}

const HfSafetensorsEntry* HfSafetensorsFile::Find(std::string_view tensor_name) const {
    if (const auto it = name_index_.find(std::string(tensor_name));
        it != name_index_.end()) {
        return &entries_[it->second];
    }
    return nullptr;
}

}// namespace aethermind
