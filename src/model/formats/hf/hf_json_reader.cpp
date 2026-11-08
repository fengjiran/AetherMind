#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "utils/parse_number.h"

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace aethermind::hf {
namespace {

constexpr bool IsJsonWhitespace(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

constexpr bool IsDigit(char c) noexcept {
    return c >= '0' && c <= '9';
}

constexpr bool IsValueTerminator(char c) noexcept {
    return IsJsonWhitespace(c) || c == ',' || c == ']' || c == '}';
}

void AppendUtf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        output += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7FFU) {
        output += static_cast<char>(0xC0U | (codepoint >> 6U));
        output += static_cast<char>(0x80U | (codepoint & 0x3FU));
    } else if (codepoint <= 0xFFFFU) {
        output += static_cast<char>(0xE0U | (codepoint >> 12U));
        output += static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU));
        output += static_cast<char>(0x80U | (codepoint & 0x3FU));
    } else {
        output += static_cast<char>(0xF0U | (codepoint >> 18U));
        output += static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU));
        output += static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU));
        output += static_cast<char>(0x80U | (codepoint & 0x3FU));
    }
}

} // namespace

HfJsonReader::HfJsonReader(std::string_view input) noexcept
    : input_(input) {}

void HfJsonReader::SkipWhitespace() noexcept {
    while (!AtEnd() && IsJsonWhitespace(input_[position_])) {
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
    return Error(
            "Expected '" + std::string(1, expected) + "' " + std::string(context));
}

Status HfJsonReader::Error(std::string_view message) const {
    return ErrorAt(message, position_);
}

Status HfJsonReader::ErrorAt(std::string_view message, size_t offset) const {
    return Status::InvalidArgument(std::string(message) + " at byte offset " + std::to_string(offset));
}

bool HfJsonReader::TryConsumeLiteral(std::string_view literal) noexcept {
    SkipWhitespace();
    if (literal.size() > input_.size() - position_) {
        return false;
    }

    if (input_.substr(position_, literal.size()) == literal) {
        position_ += literal.size();
        return true;
    }
    return false;
}

StatusOr<std::string> HfJsonReader::ParseString() {
    std::string decoded;
    AM_RETURN_IF_ERROR(ScanString(&decoded));
    return decoded;
}

StatusOr<uint32_t> HfJsonReader::ParseHexQuad() {
    if (input_.size() - position_ < 4) {
        return Error("Truncated JSON Unicode escape");
    }
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        const char c = input_[position_];
        uint32_t digit = 0;
        if (c >= '0' && c <= '9') digit = static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f')
            digit = static_cast<uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            digit = static_cast<uint32_t>(c - 'A' + 10);
        else
            return Error("Invalid JSON Unicode escape");
        ++position_;
        value = (value << 4U) | digit;
    }
    return value;
}

Status HfJsonReader::ScanString(std::string* decoded) {
    SkipWhitespace();
    if (AtEnd() || input_[position_] != '"') {
        return Error("Expected JSON string");
    }
    ++position_;

    while (!AtEnd()) {
        const size_t start = position_;
        const char cur = input_[position_++];
        if (cur == '"') {
            return Status::Ok();
        }

        if (cur == '\\') {
            if (AtEnd()) {
                return Error("Unexpected end of JSON escape sequence");
            }

            switch (const char escaped = input_[position_++]) {
                case '"':
                case '\\':
                case '/':
                    if (decoded) *decoded += escaped;
                    break;
                case 'b':
                    if (decoded) *decoded += '\b';
                    break;
                case 'f':
                    if (decoded) *decoded += '\f';
                    break;
                case 'n':
                    if (decoded) *decoded += '\n';
                    break;
                case 'r':
                    if (decoded) *decoded += '\r';
                    break;
                case 't':
                    if (decoded) *decoded += '\t';
                    break;
                case 'u': {
                    AM_ASSIGN_OR_RETURN(auto codepoint, ParseHexQuad());
                    if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                        if (input_.size() - position_ < 6 ||
                            input_[position_] != '\\' || input_[position_ + 1] != 'u') {
                            return Error("JSON high surrogate requires a low surrogate");
                        }
                        position_ += 2;
                        const size_t low_start = position_;
                        AM_ASSIGN_OR_RETURN(auto low, ParseHexQuad());
                        if (low < 0xDC00U || low > 0xDFFFU) {
                            return ErrorAt("Invalid JSON low surrogate", low_start);
                        }
                        codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
                    } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
                        return ErrorAt("Unpaired JSON low surrogate", start);
                    }
                    if (decoded) AppendUtf8(*decoded, codepoint);
                    break;
                }
                default:
                    return ErrorAt("Unsupported JSON escape sequence", position_ - 1);
            }
            continue;
        }
        const auto first = static_cast<unsigned char>(cur);
        if (first < 0x20U) {
            return ErrorAt("Unescaped control character in JSON string", start);
        }
        if (first < 0x80U) {
            if (decoded) *decoded += cur;
            continue;
        }

        size_t continuation_count = 0;
        uint32_t codepoint = 0;
        uint32_t minimum = 0;
        if (first >= 0xC2U && first <= 0xDFU) {
            continuation_count = 1;
            codepoint = first & 0x1FU;
            minimum = 0x80U;
        } else if (first >= 0xE0U && first <= 0xEFU) {
            continuation_count = 2;
            codepoint = first & 0x0FU;
            minimum = 0x800U;
        } else if (first >= 0xF0U && first <= 0xF4U) {
            continuation_count = 3;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        } else {
            return ErrorAt("Invalid UTF-8 in JSON string", start);
        }
        if (input_.size() - position_ < continuation_count) {
            return ErrorAt("Truncated UTF-8 in JSON string", start);
        }
        for (size_t i = 0; i < continuation_count; ++i) {
            const auto next = static_cast<unsigned char>(input_[position_]);
            if ((next & 0xC0U) != 0x80U) {
                return Error("Invalid UTF-8 continuation in JSON string");
            }
            ++position_;
            codepoint = (codepoint << 6U) | (next & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            return ErrorAt("Invalid UTF-8 codepoint in JSON string", start);
        }
        if (decoded) decoded->append(input_.substr(start, continuation_count + 1));
    }
    return Error("Unterminated JSON string");
}

StatusOr<std::string_view> HfJsonReader::ScanNumberToken() {
    SkipWhitespace();
    const size_t start = position_;
    if (!AtEnd() && input_[position_] == '-') {
        ++position_;
    }
    if (AtEnd() || !IsDigit(input_[position_])) {
        return Error("Expected integer value in JSON number");
    }
    if (input_[position_] == '0') {
        ++position_;
        if (!AtEnd() && IsDigit(input_[position_])) {
            return Error("Leading zeros are not allowed in JSON numbers");
        }
    } else {
        while (!AtEnd() && IsDigit(input_[position_])) {
            ++position_;
        }
    }
    if (!AtEnd() && input_[position_] == '.') {
        ++position_;
        if (AtEnd() || !IsDigit(input_[position_])) {
            return Error("Invalid floating point value: fraction requires a digit");
        }
        while (!AtEnd() && IsDigit(input_[position_])) {
            ++position_;
        }
    }
    if (!AtEnd() && (input_[position_] == 'e' || input_[position_] == 'E')) {
        ++position_;
        if (!AtEnd() && (input_[position_] == '+' || input_[position_] == '-')) {
            ++position_;
        }
        if (AtEnd() || !IsDigit(input_[position_])) {
            return Error("Invalid floating point value: exponent requires a digit");
        }
        while (!AtEnd() && IsDigit(input_[position_])) {
            ++position_;
        }
    }
    if (!AtEnd() && !IsValueTerminator(input_[position_])) {
        return Error("Invalid character after JSON number");
    }
    return input_.substr(start, position_ - start);
}

StatusOr<int64_t> HfJsonReader::ParseInt64() {
    SkipWhitespace();
    const size_t start = position_;
    if (AtEnd() || (!IsDigit(input_[position_]) && input_[position_] != '-')) {
        return Error("Expected integer value");
    }
    AM_ASSIGN_OR_RETURN(auto token, ScanNumberToken());
    int64_t value = 0;
    if (const auto [ptr, ec] =
                std::from_chars(token.data(), token.data() + token.size(), value);
        ec != std::errc{} || ptr != token.data() + token.size()) {
        return ErrorAt("Invalid integer value", start);
    }
    return value;
}

StatusOr<uint64_t> HfJsonReader::ParseUInt64() {
    SkipWhitespace();
    const size_t start = position_;
    if (AtEnd() || !IsDigit(input_[position_])) {
        return Error("Expected non-negative integer value");
    }
    AM_ASSIGN_OR_RETURN(auto token, ScanNumberToken());
    uint64_t value = 0;
    if (const auto [ptr, ec] =
                std::from_chars(token.data(), token.data() + token.size(), value);
        ec != std::errc{} || ptr != token.data() + token.size()) {
        return ErrorAt("Invalid unsigned integer value", start);
    }
    return value;
}

StatusOr<double> HfJsonReader::ParseDouble() {
    SkipWhitespace();
    const size_t start = position_;
    if (AtEnd() || (!IsDigit(input_[position_]) && input_[position_] != '-')) {
        return Error("Expected floating point value");
    }
    AM_ASSIGN_OR_RETURN(auto token, ScanNumberToken());
    double value = 0.0;
    if (!utils::ParseDouble(token, value)) {
        return ErrorAt("Invalid floating point value", start);
    }
    return value;
}

StatusOr<bool> HfJsonReader::ParseBool() {
    bool value = false;
    if (TryConsumeLiteral("true")) {
        value = true;
    } else if (!TryConsumeLiteral("false")) {
        return Error("Expected boolean value");
    }
    if (!AtEnd() && !IsValueTerminator(input_[position_])) {
        return Error("Invalid character after JSON boolean");
    }
    return value;
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

StatusOr<std::vector<double>> HfJsonReader::ParseDoubleArray() {
    if (!TryConsume('[')) {
        return Error("Expected JSON array");
    }
    std::vector<double> values;
    SkipWhitespace();
    if (TryConsume(']')) {
        return values;
    }
    while (true) {
        AM_ASSIGN_OR_RETURN(double value, ParseDouble());
        values.push_back(value);
        SkipWhitespace();
        if (TryConsume(']')) {
            break;
        }
        AM_RETURN_IF_ERROR(Expect(',', "between JSON array values"));
    }
    return values;
}

Status HfJsonReader::SkipValue() {
    return SkipValueInternal(0);
}

Status HfJsonReader::SkipValueInternal(uint32_t depth) {
    if (depth > kMaxSkipDepth) {
        return Error("JSON nesting depth exceeds maximum");
    }

    SkipWhitespace();
    if (AtEnd()) {
        return Error("Unexpected end of JSON while skipping value");
    }

    const char cur = input_[position_];
    if (cur == '{') {
        ++position_;
        SkipWhitespace();
        if (TryConsume('}')) {
            return Status::Ok();
        }

        while (true) {
            AM_RETURN_IF_ERROR(ScanString(nullptr));

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
        return ScanString(nullptr);
    }

    if (IsDigit(cur) || cur == '-') {
        if (const auto number = ScanNumberToken(); !number.ok()) {
            return number.status();
        }
        return Status::Ok();
    }

    if (TryConsumeLiteral("true") || TryConsumeLiteral("false") || TryConsumeLiteral("null")) {
        if (!AtEnd() && !IsValueTerminator(input_[position_])) {
            return Error("Invalid character after JSON literal");
        }
        return Status::Ok();
    }

    return Error("Unsupported JSON value while skipping");
}

} // namespace aethermind::hf
