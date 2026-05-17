#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_JSON_READER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_JSON_READER_H

#include "aethermind/base/status.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aethermind {
namespace hf {

class HfJsonReader {
public:
    explicit HfJsonReader(std::string_view input) noexcept;
    virtual ~HfJsonReader() = default;

    void SkipWhitespace() noexcept;

    AM_NODISCARD bool AtEnd() const noexcept;

    bool TryConsume(char expected) noexcept;
    Status Expect(char expected, std::string_view context);
    bool TryConsumeLiteral(std::string_view literal) noexcept;

    StatusOr<std::string> ParseString();
    StatusOr<int64_t> ParseInt64();
    StatusOr<uint64_t> ParseUInt64();
    StatusOr<double> ParseDouble();
    StatusOr<bool> ParseBool();
    StatusOr<std::vector<std::string>> ParseStringArray();
    StatusOr<std::vector<int64_t>> ParseInt64Array();
    Status SkipValue();

    static constexpr uint32_t kMaxSkipDepth = 32;

protected:
    std::string_view input_;
    size_t position_ = 0;

private:
    Status SkipValueInternal(uint32_t depth);
};

}// namespace hf
}// namespace aethermind

#endif
