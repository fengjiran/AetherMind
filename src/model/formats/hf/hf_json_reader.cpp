#include "aethermind/model/formats/hf/hf_json_reader.h"

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

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

bool HfJsonReader::Consume(char expected) noexcept {
    SkipWhitespace();
    if (!AtEnd() && input_[position_] == expected) {
        ++position_;
        return true;
    }
    return false;
}

bool HfJsonReader::Expect(char expected) noexcept {
    return Consume(expected);
}

bool HfJsonReader::ConsumeLiteral(std::string_view literal) noexcept {
    SkipWhitespace();
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
        const char current = input_[position_++];
        if (current == '"') {
            return result;
        }

        if (current == '\\') {
            if (AtEnd()) {
                return Status::InvalidArgument("Unexpected end of JSON escape sequence");
            }

            switch (const char escaped = input_[position_++]) {
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
                    return Status::InvalidArgument("Unsupported JSON escape sequence");
            }
            continue;
        }
        result.push_back(current);
    }
    return Status::InvalidArgument("Unterminated JSON string");
}

StatusOr<int64_t> HfJsonReader::ParseInt64() {
    SkipWhitespace();
    const size_t start = position_;
    if (!AtEnd() && (input_[position_] == '-' || input_[position_] == '+')) {
        ++position_;
    }
    while (!AtEnd() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
    }
    if (start == position_ || (position_ == start + 1 && (input_[start] == '-' || input_[start] == '+'))) {
        return Status::InvalidArgument("Expected integer value");
    }

    int64_t value = 0;
    const auto token = input_.substr(start, position_ - start);
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size()) {
        return Status::InvalidArgument("Invalid integer value");
    }
    return value;
}

Status HfJsonReader::SkipValue() {
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
            if (!Expect(':')) {
                return Status::InvalidArgument("Expected ':' inside JSON object while skipping value");
            }
            AM_RETURN_IF_ERROR(SkipValue());
            SkipWhitespace();
            if (Consume('}')) {
                return Status::Ok();
            }
            if (!Expect(',')) {
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
            if (!Expect(',')) {
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

    return Status::InvalidArgument("Unsupported JSON value while skipping");
}

}// namespace hf
}// namespace aethermind
