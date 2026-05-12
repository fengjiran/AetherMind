#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_JSON_READER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_JSON_READER_H

#include "aethermind/base/status.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace aethermind {
namespace hf {

class HfJsonReader {
public:
    explicit HfJsonReader(std::string_view input) noexcept;
    virtual ~HfJsonReader() = default;

    void SkipWhitespace() noexcept;

    AM_NODISCARD bool AtEnd() const noexcept;

    bool Consume(char expected) noexcept;
    bool Expect(char expected) noexcept;
    bool ConsumeLiteral(std::string_view literal) noexcept;

    StatusOr<std::string> ParseString();
    StatusOr<int64_t> ParseInt64();
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
