#include "aethermind/dtypes/fp16_conversion.h"
#include "utils/overflow_check.h"

#include <cstdint>

namespace aethermind {
namespace {

Status Convert(std::span<const std::byte> src, std::span<std::byte> dst,
               size_t src_width, size_t dst_width, Fp16ConversionFunc func) {
    if (src.size() % src_width != 0) {
        return Status::InvalidArgument(
                "FP16 conversion requires complete source elements");
    }

    const size_t count = src.size() / src_width;
    size_t dst_bytes = 0;
    if (CheckOverflowMul(count, dst_width, &dst_bytes)) {
        return Status::Overflow("FP16 conversion byte count overflow");
    }

    if (dst.size() != dst_bytes) {
        return Status::InvalidArgument(
                "FP16 conversion destination byte count mismatch");
    }

    if (count == 0) {
        return Status::Ok();
    }

    if (func == nullptr || src.data() == nullptr || dst.data() == nullptr) {
        return Status::InvalidArgument(
                "FP16 conversion requires non-null ranges and function");
    }

    const auto begin_src = reinterpret_cast<uintptr_t>(src.data());
    const auto begin_dst = reinterpret_cast<uintptr_t>(dst.data());
    // Ordered subtraction avoids overflow from forming address-range ends.
    if (begin_src <= begin_dst
                ? begin_dst - begin_src < src.size()
                : begin_src - begin_dst < dst.size()) {
        return Status::InvalidArgument(
                "FP16 conversion ranges must not overlap");
    }

    func(src.data(), dst.data(), count);
    return Status::Ok();
}

} // namespace

Status ConvertFp16ToFp32(std::span<const std::byte> src, std::span<std::byte> dst,
                         const Fp16ConversionFuncs& functions) {
    return Convert(src, dst, sizeof(uint16_t),
                   sizeof(float), functions.to_fp32);
}

Status ConvertFp32ToFp16(std::span<const std::byte> src, std::span<std::byte> dst,
                         const Fp16ConversionFuncs& functions) {
    return Convert(src, dst, sizeof(float),
                   sizeof(uint16_t), functions.from_fp32);
}

} // namespace aethermind
