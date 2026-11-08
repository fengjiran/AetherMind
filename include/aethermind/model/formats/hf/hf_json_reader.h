#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_JSON_READER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_JSON_READER_H

/// @file hf_json_reader.h
/// @brief Sequential JSON parser primitives over borrowed HF metadata text.

#include "aethermind/base/status.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind::hf {

/// @brief Sequential JSON parser over a borrowed string view.
///
/// The parser walks the input with a single position cursor and never copies
/// it; decoded strings and arrays own their results. Expected format errors
/// surface as InvalidArgument with a zero-based byte offset into the input.
/// After a parse failure, discard the reader; cursor rollback is not promised.
class HfJsonReader {
public:
    explicit HfJsonReader(std::string_view input) noexcept;
    virtual ~HfJsonReader() = default;

    void SkipWhitespace() noexcept;

    AM_NODISCARD bool AtEnd() const noexcept;

    bool TryConsume(char expected) noexcept;
    Status Expect(char expected, std::string_view context);
    /// Matches a raw prefix after whitespace; callers validate token boundaries.
    bool TryConsumeLiteral(std::string_view literal) noexcept;

    /// Decodes JSON escapes, including UTF-16 surrogate pairs, into UTF-8.
    /// Rejects malformed UTF-8 and unescaped control characters.
    StatusOr<std::string> ParseString();
    /// Parse a complete JSON number token and check the requested representation.
    /// Integer readers reject fractions/exponents; the unsigned reader rejects
    /// a minus sign. Floating-point conversion rejects unrepresentable values.
    StatusOr<int64_t> ParseInt64();
    StatusOr<uint64_t> ParseUInt64();
    StatusOr<double> ParseDouble();
    StatusOr<bool> ParseBool();
    StatusOr<std::vector<std::string>> ParseStringArray();
    StatusOr<std::vector<int64_t>> ParseInt64Array();
    StatusOr<std::vector<double>> ParseDoubleArray();
    /// Validates and skips one value without materializing strings or numbers.
    /// Numeric syntax is checked independently of machine numeric range. The
    /// enclosing parser must check its separator or the end of the document.
    Status SkipValue();

    /// Depth cap for nested values during SkipValue, bounding stack usage on
    /// adversarial inputs.
    static constexpr uint32_t kMaxSkipDepth = 32;

protected:
    /// Remaining input and cursor; subclasses may read but not rewind.
    std::string_view input_;
    size_t position_ = 0;

private:
    Status Error(std::string_view message) const;
    Status ErrorAt(std::string_view message, size_t offset) const;
    StatusOr<uint32_t> ParseHexQuad();
    // A null output keeps the same validation while discarding decoded bytes.
    Status ScanString(std::string* decoded);
    StatusOr<std::string_view> ScanNumberToken();
    Status SkipValueInternal(uint32_t depth);
};

} // namespace aethermind::hf


#endif
