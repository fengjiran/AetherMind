#include "aethermind/dtypes/bf16_conversion.h"
#include "utils/overflow_check.h"

#include <cstdint>
#include <cstring>

namespace aethermind {
namespace {

static_assert(sizeof(float) == sizeof(uint32_t));

Status Convert(std::span<const std::byte> src, std::span<std::byte> dst,
               size_t src_width, size_t dst_width, Bf16ConversionFunc func) {
    if (src.size() % src_width != 0) {
        return Status::InvalidArgument(
                "BF16 conversion requires complete source elements");
    }

    const size_t count = src.size() / src_width;
    size_t dst_bytes = 0;
    if (CheckOverflowMul(count, dst_width, &dst_bytes)) {
        return Status::Overflow("BF16 conversion byte count overflow");
    }

    if (dst.size() != dst_bytes) {
        return Status::InvalidArgument(
                "BF16 conversion destination byte count mismatch");
    }

    if (count == 0) {
        return Status::Ok();
    }

    if (func == nullptr || src.data() == nullptr || dst.data() == nullptr) {
        return Status::InvalidArgument(
                "BF16 conversion requires non-null ranges and function");
    }

    const auto begin_src = reinterpret_cast<uintptr_t>(src.data());
    const auto begin_dst = reinterpret_cast<uintptr_t>(dst.data());
    // Ordered subtraction avoids overflow from forming address-range ends.
    if (begin_src <= begin_dst
                ? begin_dst - begin_src < src.size()
                : begin_src - begin_dst < dst.size()) {
        return Status::InvalidArgument(
                "BF16 conversion ranges must not overlap");
    }

    func(src.data(), dst.data(), count);
    return Status::Ok();
}

} // namespace

void ConvertBf16ToFp32Scalar(const std::byte* src,
                             std::byte* dst,
                             size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        uint16_t input = 0;
        std::memcpy(&input, src + i * sizeof(input), sizeof(input));
        uint32_t output = static_cast<uint32_t>(input) << 16;
        if ((input & 0x7F80U) == 0x7F80U && (input & 0x007FU) != 0) {
            output |= 0x00400000U;
        }
        std::memcpy(dst + i * sizeof(output), &output, sizeof(output));
    }
}

void ConvertFp32ToBf16Scalar(const std::byte* src,
                             std::byte* dst,
                             size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        uint32_t input = 0;
        std::memcpy(&input, src + i * sizeof(input), sizeof(input));
        uint16_t output = 0;
        if ((input & 0x7FFFFFFFU) > 0x7F800000U) {
            output = static_cast<uint16_t>((input >> 16) | 0x0040U);
        } else {
            // Bias implements RNE across subnormal, normal and overflow boundaries.
            const uint32_t bias = 0x7FFFU + ((input >> 16) & 1U);
            output = static_cast<uint16_t>((input + bias) >> 16);
        }
        std::memcpy(dst + i * sizeof(output), &output, sizeof(output));
    }
}

Status ConvertBf16ToFp32(std::span<const std::byte> src, std::span<std::byte> dst,
                         const Bf16ConversionFuncs& functions) {
    return Convert(src, dst, sizeof(uint16_t),
                   sizeof(float), functions.to_fp32);
}

Status ConvertFp32ToBf16(std::span<const std::byte> src, std::span<std::byte> dst,
                         const Bf16ConversionFuncs& functions) {
    return Convert(src, dst, sizeof(float),
                   sizeof(uint16_t), functions.from_fp32);
}

} // namespace aethermind
