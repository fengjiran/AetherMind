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

            AM_RETURN_IF_ERROR(SkipValueInternal(depth + 1));
            SkipWhitespace();
            if (Consume('}')) {
                return Status::Ok();
            }

            if (!Expect(',')) {
                return Status::InvalidArgument("Expected ',' inside JSON object while skipping value");
            }
        }
    }

    if (cur == '[') {
        ++position_;
        SkipWhitespace();
        if (Consume(']')) {
            return Status::Ok();
        }

        while (true) {
            AM_RETURN_IF_ERROR(SkipValueInternal(depth + 1));
            SkipWhitespace();
            if (Consume(']')) {
                return Status::Ok();
            }

            if (!Expect(',')) {
                return Status::InvalidArgument("Expected ',' inside JSON array while skipping value");
            }
        }
    }

    if (cur == '"') {
        if (const auto string_value = ParseString(); !string_value.ok()) {
            return string_value.status();
        }
        return Status::Ok();
    }

    if (std::isdigit(static_cast<unsigned char>(cur)) || cur == '-') {
        if (const auto number = ParseInt64(); !number.ok()) {
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
