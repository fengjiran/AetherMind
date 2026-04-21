#include "aethermind/model/formats/hf/hf_safetensors_index.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace aethermind {

namespace {

class OwnedBytesBacking final : public RawTensorBacking {
public:
    explicit OwnedBytesBacking(std::vector<std::byte> bytes) noexcept
        : bytes_(std::move(bytes)) {}

    AM_NODISCARD const std::byte* Data() const noexcept {
        return bytes_.data();
    }

    AM_NODISCARD size_t Size() const noexcept {
        return bytes_.size();
    }

private:
    std::vector<std::byte> bytes_{};
};

std::string FormatPathMessage(
        std::string_view prefix,
        const std::filesystem::path& path) {
    return std::string(prefix) + ": " + path.string();
}

StatusOr<std::vector<std::byte>> ReadFileBytes(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        return Status::NotFound(FormatPathMessage("Safetensors file not found", path));
    }
    if (error) {
        return Status::Internal(FormatPathMessage("Failed to stat safetensors file", path));
    }
    if (!std::filesystem::is_regular_file(path, error)) {
        return Status::InvalidArgument(FormatPathMessage("Safetensors path is not a regular file", path));
    }
    if (error) {
        return Status::Internal(FormatPathMessage("Failed to inspect safetensors file type", path));
    }

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        return Status::Internal(FormatPathMessage("Failed to open safetensors file", path));
    }

    const std::streampos end_pos = stream.tellg();
    if (end_pos < 0) {
        return Status::Internal(FormatPathMessage("Failed to determine safetensors file size", path));
    }

    const size_t size = static_cast<size_t>(end_pos);
    stream.seekg(0, std::ios::beg);

    std::vector<std::byte> bytes(size);
    if (size > 0) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        if (!stream) {
            return Status::Internal(FormatPathMessage("Failed to read safetensors file bytes", path));
        }
    }
    return bytes;
}

StatusOr<uint64_t> ParseLittleEndianU64(std::span<const std::byte> bytes) {
    if (bytes.size() < sizeof(uint64_t)) {
        return Status::InvalidArgument("Safetensors file is too small to contain header length");
    }

    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        value |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[i])) << (i * 8U);
    }
    return value;
}

Status ParseSafetensorsDType(
        std::string_view dtype_text,
        DataType* dtype) {
    if (dtype_text.compare("F16") == 0) {
        *dtype = DataType::Float(16);
        return Status::Ok();
    }
    if (dtype_text.compare("BF16") == 0) {
        *dtype = DataType::BFloat(16);
        return Status::Ok();
    }
    if (dtype_text.compare("F32") == 0) {
        *dtype = DataType::Float32();
        return Status::Ok();
    }
    if (dtype_text.compare("I32") == 0) {
        *dtype = DataType::Int(32);
        return Status::Ok();
    }
    if (dtype_text.compare("I64") == 0) {
        *dtype = DataType::Int(64);
        return Status::Ok();
    }
    return Status::InvalidArgument(std::string("Unsupported safetensors dtype: ") + std::string(dtype_text));
}

StatusOr<uint64_t> CheckedMultiply(uint64_t lhs, uint64_t rhs, std::string_view context) {
    if (lhs == 0 || rhs == 0) {
        return uint64_t{0};
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return Status(StatusCode::kOutOfRange,
                      std::string("Integer overflow while computing ") + std::string(context));
    }
    return lhs * rhs;
}

class HeaderParser {
public:
    HeaderParser(
            std::string_view input,
            const std::shared_ptr<const RawTensorBacking>& backing,
            const std::byte* data_base,
            size_t data_size) noexcept
        : input_(input), backing_(backing), data_base_(data_base), data_size_(data_size) {}

    StatusOr<std::vector<HfSafetensorEntry>> Parse() {
        SkipWhitespace();
        if (!Consume('{')) {
            return Status::InvalidArgument("Safetensors header must start with a JSON object");
        }

        std::vector<HfSafetensorEntry> entries;
        SkipWhitespace();
        if (Consume('}')) {
            return entries;
        }

        while (true) {
            const auto key = ParseString();
            if (!key.ok()) {
                return key.status();
            }
            if (!Expect(':', "Expected ':' after safetensors header key")) {
                return Status::InvalidArgument("Expected ':' after safetensors header key");
            }

            if (key->compare("__metadata__") == 0) {
                const Status skip_status = SkipValue();
                if (!skip_status.ok()) {
                    return skip_status;
                }
            } else {
                HfSafetensorEntry entry;
                const Status entry_status = ParseTensorEntry(*key, &entry);
                if (!entry_status.ok()) {
                    return entry_status;
                }
                entries.push_back(std::move(entry));
            }

            SkipWhitespace();
            if (Consume('}')) {
                break;
            }
            if (!Expect(',', "Expected ',' between safetensors header entries")) {
                return Status::InvalidArgument("Expected ',' between safetensors header entries");
            }
        }

        SkipWhitespace();
        if (!AtEnd()) {
            return Status::InvalidArgument("Safetensors header contains trailing JSON content");
        }
        return entries;
    }

private:
    Status ParseTensorEntry(
            const std::string& name,
            HfSafetensorEntry* entry) {
        if (!Expect('{', "Expected '{' at start of safetensors tensor entry")) {
            return Status::InvalidArgument("Expected '{' at start of safetensors tensor entry");
        }

        std::optional<DataType> dtype;
        std::vector<int64_t> shape;
        std::optional<std::pair<uint64_t, uint64_t>> data_offsets;

        SkipWhitespace();
        if (!Consume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }
                if (!Expect(':', "Expected ':' after safetensors tensor field name")) {
                    return Status::InvalidArgument("Expected ':' after safetensors tensor field name");
                }

                if (key->compare("dtype") == 0) {
                    const auto dtype_text = ParseString();
                    if (!dtype_text.ok()) {
                        return dtype_text.status();
                    }
                    DataType parsed_dtype;
                    const Status dtype_status = ParseSafetensorsDType(*dtype_text, &parsed_dtype);
                    if (!dtype_status.ok()) {
                        return dtype_status;
                    }
                    dtype = parsed_dtype;
                } else if (key->compare("shape") == 0) {
                    const auto parsed_shape = ParseInt64Array();
                    if (!parsed_shape.ok()) {
                        return parsed_shape.status();
                    }
                    shape = std::move(*parsed_shape);
                } else if (key->compare("data_offsets") == 0) {
                    const auto parsed_offsets = ParseOffsetPair();
                    if (!parsed_offsets.ok()) {
                        return parsed_offsets.status();
                    }
                    data_offsets = *parsed_offsets;
                } else {
                    const Status skip_status = SkipValue();
                    if (!skip_status.ok()) {
                        return skip_status;
                    }
                }

                SkipWhitespace();
                if (Consume('}')) {
                    break;
                }
                if (!Expect(',', "Expected ',' between safetensors tensor fields")) {
                    return Status::InvalidArgument("Expected ',' between safetensors tensor fields");
                }
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
            return Status(StatusCode::kOutOfRange,
                          "Safetensors tensor entry points outside the raw data region");
        }

        uint64_t numel = 1;
        for (const int64_t dim : shape) {
            if (dim < 0) {
                return Status::InvalidArgument("Safetensors tensor entry has negative shape dimension");
            }
            const auto updated_numel = CheckedMultiply(numel, static_cast<uint64_t>(dim), "safetensors shape");
            if (!updated_numel.ok()) {
                return updated_numel.status();
            }
            numel = *updated_numel;
        }

        const auto expected_nbytes = CheckedMultiply(numel, static_cast<uint64_t>(dtype->nbytes()), "safetensors tensor byte size");
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
        entry->view = RawTensorView{
                .data = data_base_ + begin,
                .byte_size = static_cast<size_t>(end - begin),
                .dtype = *dtype,
                .shape = shape,
                .backing = backing_,
        };
        return Status::Ok();
    }

    StatusOr<std::vector<int64_t>> ParseInt64Array() {
        if (!Expect('[', "Expected '[' at start of integer array")) {
            return Status::InvalidArgument("Expected '[' at start of integer array");
        }

        std::vector<int64_t> values;
        SkipWhitespace();
        if (Consume(']')) {
            return values;
        }

        while (true) {
            const auto value = ParseInt64();
            if (!value.ok()) {
                return value.status();
            }
            values.push_back(*value);

            SkipWhitespace();
            if (Consume(']')) {
                break;
            }
            if (!Expect(',', "Expected ',' between integer array elements")) {
                return Status::InvalidArgument("Expected ',' between integer array elements");
            }
        }
        return values;
    }

    StatusOr<std::pair<uint64_t, uint64_t>> ParseOffsetPair() {
        if (!Expect('[', "Expected '[' at start of data_offsets")) {
            return Status::InvalidArgument("Expected '[' at start of data_offsets");
        }

        const auto first = ParseUInt64();
        if (!first.ok()) {
            return first.status();
        }
        if (!Expect(',', "Expected ',' between data_offsets values")) {
            return Status::InvalidArgument("Expected ',' between data_offsets values");
        }
        const auto second = ParseUInt64();
        if (!second.ok()) {
            return second.status();
        }

        SkipWhitespace();
        if (!Consume(']')) {
            return Status::InvalidArgument("Expected closing ']' after data_offsets values");
        }
        return std::make_pair(*first, *second);
    }

    StatusOr<std::string> ParseString() {
        SkipWhitespace();
        if (AtEnd() || input_[position_] != '"') {
            return Status::InvalidArgument("Expected JSON string");
        }
        ++position_;

        std::string result;
        while (!AtEnd()) {
            const char current = input_[position_++];
            if (current == '"') {
                return result;
            }
            if (current == '\\') {
                if (AtEnd()) {
                    return Status::InvalidArgument("Unexpected end of JSON escape sequence");
                }
                const char escaped = input_[position_++];
                switch (escaped) {
                    case '"':
                    case '\\':
                    case '/':
                        result.push_back(escaped);
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    default:
                        return Status::InvalidArgument("Unsupported JSON escape sequence in safetensors header");
                }
                continue;
            }
            result.push_back(current);
        }
        return Status::InvalidArgument("Unterminated JSON string in safetensors header");
    }

    StatusOr<int64_t> ParseInt64() {
        SkipWhitespace();
        const size_t start = position_;
        if (!AtEnd() && (input_[position_] == '-' || input_[position_] == '+')) {
            ++position_;
        }
        while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
        if (start == position_ || (position_ == start + 1 && (input_[start] == '-' || input_[start] == '+'))) {
            return Status::InvalidArgument("Expected integer value in safetensors header");
        }

        int64_t value = 0;
        const auto token = input_.substr(start, position_ - start);
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec != std::errc{} || ptr != token.data() + token.size()) {
            return Status::InvalidArgument("Invalid integer value in safetensors header");
        }
        return value;
    }

    StatusOr<uint64_t> ParseUInt64() {
        const auto parsed = ParseInt64();
        if (!parsed.ok()) {
            return parsed.status();
        }
        if (*parsed < 0) {
            return Status::InvalidArgument("Expected non-negative integer value in safetensors header");
        }
        return static_cast<uint64_t>(*parsed);
    }

    Status SkipValue() {
        SkipWhitespace();
        if (AtEnd()) {
            return Status::InvalidArgument("Unexpected end of JSON while skipping value");
        }

        const char current = input_[position_];
        if (current == '{') {
            ++position_;
            SkipWhitespace();
            if (Consume('}')) {
                return Status::Ok();
            }
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }
                if (!Expect(':', "Expected ':' inside JSON object while skipping value")) {
                    return Status::InvalidArgument("Expected ':' inside JSON object while skipping value");
                }
                AM_RETURN_IF_ERROR(SkipValue());
                SkipWhitespace();
                if (Consume('}')) {
                    return Status::Ok();
                }
                if (!Expect(',', "Expected ',' inside JSON object while skipping value")) {
                    return Status::InvalidArgument("Expected ',' inside JSON object while skipping value");
                }
            }
        }

        if (current == '[') {
            ++position_;
            SkipWhitespace();
            if (Consume(']')) {
                return Status::Ok();
            }
            while (true) {
                AM_RETURN_IF_ERROR(SkipValue());
                SkipWhitespace();
                if (Consume(']')) {
                    return Status::Ok();
                }
                if (!Expect(',', "Expected ',' inside JSON array while skipping value")) {
                    return Status::InvalidArgument("Expected ',' inside JSON array while skipping value");
                }
            }
        }

        if (current == '"') {
            const auto string_value = ParseString();
            if (!string_value.ok()) {
                return string_value.status();
            }
            return Status::Ok();
        }

        if (std::isdigit(static_cast<unsigned char>(current)) || current == '-' || current == '+') {
            const auto number = ParseInt64();
            if (!number.ok()) {
                return number.status();
            }
            return Status::Ok();
        }

        if (ConsumeLiteral("true") || ConsumeLiteral("false") || ConsumeLiteral("null")) {
            return Status::Ok();
        }

        return Status::InvalidArgument("Unsupported JSON value while skipping safetensors metadata");
    }

    void SkipWhitespace() noexcept {
        while (!AtEnd() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    AM_NODISCARD bool AtEnd() const noexcept {
        return position_ >= input_.size();
    }

    bool Consume(char expected) noexcept {
        SkipWhitespace();
        if (!AtEnd() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    bool Expect(char expected, std::string_view) noexcept {
        return Consume(expected);
    }

    bool ConsumeLiteral(std::string_view literal) noexcept {
        SkipWhitespace();
        if (input_.substr(position_, literal.size()) == literal) {
            position_ += literal.size();
            return true;
        }
        return false;
    }

    std::string_view input_;
    std::shared_ptr<const RawTensorBacking> backing_{};
    const std::byte* data_base_ = nullptr;
    size_t data_size_ = 0;
    size_t position_ = 0;
};

}// namespace

StatusOr<HfSafetensorsIndex> HfSafetensorsIndex::LoadSingleFile(
        const std::filesystem::path& safetensors_path) {
    const auto file_bytes = ReadFileBytes(safetensors_path);
    if (!file_bytes.ok()) {
        return file_bytes.status();
    }

    const auto backing = std::make_shared<OwnedBytesBacking>(std::move(*file_bytes));
    if (backing->Size() < sizeof(uint64_t)) {
        return Status::InvalidArgument(FormatPathMessage("Safetensors file is too small", safetensors_path));
    }

    const auto header_length = ParseLittleEndianU64(
            std::span<const std::byte>(backing->Data(), sizeof(uint64_t)));
    if (!header_length.ok()) {
        return Status(StatusCode::kInvalidArgument,
                      FormatPathMessage(header_length.status().message(), safetensors_path));
    }

    const uint64_t header_begin = sizeof(uint64_t);
    const uint64_t header_end = header_begin + *header_length;
    if (header_end > backing->Size()) {
        return Status::InvalidArgument(
                FormatPathMessage("Safetensors header length exceeds file size", safetensors_path));
    }

    const auto* header_chars = reinterpret_cast<const char*>(backing->Data() + header_begin);
    const std::string_view header_json(header_chars, static_cast<size_t>(*header_length));
    const std::byte* data_base = backing->Data() + header_end;
    const size_t data_size = backing->Size() - static_cast<size_t>(header_end);

    HeaderParser parser(header_json, backing, data_base, data_size);
    const auto parsed_entries = parser.Parse();
    if (!parsed_entries.ok()) {
        return Status(parsed_entries.status().code(),
                      FormatPathMessage(parsed_entries.status().message(), safetensors_path));
    }

    HfSafetensorsIndex index;
    index.path_ = safetensors_path;
    index.entries_ = std::move(*parsed_entries);
    return index;
}

const HfSafetensorEntry* HfSafetensorsIndex::Find(
        std::string_view tensor_name) const noexcept {
    for (const auto& entry : entries_) {
        if (entry.name.size() == tensor_name.size() &&
            std::char_traits<char>::compare(entry.name.data(), tensor_name.data(), tensor_name.size()) == 0) {
            return &entry;
        }
    }
    return nullptr;
}

}// namespace aethermind
