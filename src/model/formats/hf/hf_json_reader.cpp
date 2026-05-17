#include "aethermind/model/formats/hf/hf_json_reader.h"

#include <cctype>
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace aethermind {
namespace hf {

HfJsonReader::HfJsonReader(std::string_view input) noexcept
    : input_(input) {}

void HfJsonReader::SkipWhitespace() noexcept {
    while (!AtEnd() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
    }
}

bool HfJsonReader::AtEnd() const noexcept {
    return position_ >= input_.size();
}

bool HfJsonReader::TryConsume(char expected) noexcept {
    SkipWhitespace();
    if (!AtEnd() && input_[position_] == expected) {
        ++position_;
        return true;
    }
    return false;
}

Status HfJsonReader::Expect(char expected, std::string_view context) {
    if (TryConsume(expected)) {
        return Status::Ok();
    }
    return Status::InvalidArgument(
            "Expected '" + std::string(1, expected) + "' " + std::string(context));
}

bool HfJsonReader::TryConsumeLiteral(std::string_view literal) noexcept {
    SkipWhitespace();
    if (position_ + literal.size() > input_.size()) {
        return false;
    }

    if (input_.substr(position_, literal.size()) == literal) {
        position_ += literal.size();
        return true;
    }
    return false;
}

StatusOr<std::string> HfJsonReader::ParseString() {
    SkipWhitespace();
    if (AtEnd() || input_[position_] != '"') {
        return Status::InvalidArgument("Expected JSON string");
    }
    ++position_;

    std::string result;
    while (!AtEnd()) {
        const char cur = input_[position_++];
        if (cur == '"') {
            return result;
        }

        if (cur == '\\') {
            if (AtEnd()) {
                return Status::InvalidArgument("Unexpected end of JSON escape sequence");
            }

            switch (const char escaped = input_[position_++]) {
                case '"':
                case '\\':
                case '/':
                    result += escaped;
                    break;
                case 'b':
                    result += '\b';
                    break;
                case 'f':
                    result += '\f';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    return Status::InvalidArgument("Unsupported JSON escape sequence");
            }
            continue;
        }
        result += cur;
    }
    return Status::InvalidArgument("Unterminated JSON string");
}

StatusOr<int64_t> HfJsonReader::ParseInt64() {
    SkipWhitespace();
    const size_t start = position_;
    if (!AtEnd() && input_[position_] == '-') {
        ++position_;
    }

    while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
    }

    if (start == position_ || (position_ == start + 1 && input_[start] == '-')) {
        return Status::InvalidArgument("Expected integer value");
    }

    int64_t value = 0;
    const auto token = input_.substr(start, position_ - start);
    if (const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        ec != std::errc{} || ptr != token.data() + token.size()) {
        return Status::InvalidArgument("Invalid integer value");
    }
    return value;
}

StatusOr<uint64_t> HfJsonReader::ParseUInt64() {
    SkipWhitespace();
    const size_t start = position_;
    while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
    }
    if (start == position_) {
        return Status::InvalidArgument("Expected non-negative integer value");
    }

    uint64_t value = 0;
    const auto token = input_.substr(start, position_ - start);
    if (const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        ec != std::errc{} || ptr != token.data() + token.size()) {
        return Status::InvalidArgument("Invalid unsigned integer value");
    }
    return value;
}

StatusOr<double> HfJsonReader::ParseDouble() {
    SkipWhitespace();
    const size_t start = position_;
    if (!AtEnd() && input_[position_] == '-') {
        ++position_;
    }

    while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
    }
    if (!AtEnd() && input_[position_] == '.') {
        ++position_;
        while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    if (!AtEnd() && (input_[position_] == 'e' || input_[position_] == 'E')) {
        ++position_;
        if (!AtEnd() && (input_[position_] == '+' || input_[position_] == '-')) {
            ++position_;
        }
        while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    if (start == position_ || (position_ == start + 1 && input_[start] == '-')) {
        return Status::InvalidArgument("Expected floating point value");
    }

    double value = 0.0;
    const auto token = input_.substr(start, position_ - start);
    if (const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        ec != std::errc{} || ptr != token.data() + token.size()) {
        return Status::InvalidArgument("Invalid floating point value");
    }
    return value;
}

StatusOr<bool> HfJsonReader::ParseBool() {
    if (TryConsumeLiteral("true")) {
        return true;
    }
    if (TryConsumeLiteral("false")) {
        return false;
    }
    return Status::InvalidArgument("Expected boolean value");
}

StatusOr<std::vector<std::string>> HfJsonReader::ParseStringArray() {
    AM_RETURN_IF_ERROR(Expect('[', "at start of string array"));

    std::vector<std::string> values;
    SkipWhitespace();
    if (TryConsume(']')) {
        return values;
    }

    while (true) {
        auto value = ParseString();
        if (!value.ok()) {
            return value.status();
        }
        values.push_back(std::move(*value));

        SkipWhitespace();
        if (TryConsume(']')) {
            break;
        }
        AM_RETURN_IF_ERROR(Expect(',', "between string array elements"));
    }
    return values;
}

StatusOr<std::vector<int64_t>> HfJsonReader::ParseInt64Array() {
    AM_RETURN_IF_ERROR(Expect('[', "at start of integer array"));

    std::vector<int64_t> values;
    SkipWhitespace();
    if (TryConsume(']')) {
        return values;
    }

    while (true) {
        const auto value = ParseInt64();
        if (!value.ok()) {
            return value.status();
        }
        values.push_back(*value);

        SkipWhitespace();
        if (TryConsume(']')) {
            break;
        }
        AM_RETURN_IF_ERROR(Expect(',', "between integer array elements"));
    }
    return values;
}

Status HfJsonReader::SkipValue() {
    return SkipValueInternal(0);
}

Status HfJsonReader::SkipValueInternal(uint32_t depth) {
    if (depth > kMaxSkipDepth) {
        return Status::InvalidArgument("JSON nesting depth exceeds maximum");
    }

    SkipWhitespace();
    if (AtEnd()) {
        return Status::InvalidArgument("Unexpected end of JSON while skipping value");
    }

    const char cur = input_[position_];
    if (cur == '{') {
        ++position_;
        SkipWhitespace();
        if (TryConsume('}')) {
            return Status::Ok();
        }

        while (true) {
            const auto key = ParseString();
            if (!key.ok()) {
                return key.status();
            }

            AM_RETURN_IF_ERROR(Expect(':', "inside JSON object while skipping value"));

            AM_RETURN_IF_ERROR(SkipValueInternal(depth + 1));
            SkipWhitespace();
            if (TryConsume('}')) {
                return Status::Ok();
            }

            AM_RETURN_IF_ERROR(Expect(',', "inside JSON object while skipping value"));
        }
    }

    if (cur == '[') {
        ++position_;
        SkipWhitespace();
        if (TryConsume(']')) {
            return Status::Ok();
        }

        while (true) {
            AM_RETURN_IF_ERROR(SkipValueInternal(depth + 1));
            SkipWhitespace();
            if (TryConsume(']')) {
                return Status::Ok();
            }

            AM_RETURN_IF_ERROR(Expect(',', "inside JSON array while skipping value"));
        }
    }

    if (cur == '"') {
        if (const auto string_value = ParseString(); !string_value.ok()) {
            return string_value.status();
        }
        return Status::Ok();
    }

    if (std::isdigit(static_cast<unsigned char>(cur)) || cur == '-') {
        if (const auto number = ParseDouble(); !number.ok()) {
            return number.status();
        }
        return Status::Ok();
    }

    if (TryConsumeLiteral("true") || TryConsumeLiteral("false") || TryConsumeLiteral("null")) {
        return Status::Ok();
    }

    return Status::InvalidArgument("Unsupported JSON value while skipping");
}

}// namespace hf
}// namespace aethermind
