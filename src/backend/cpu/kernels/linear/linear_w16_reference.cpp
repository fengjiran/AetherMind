#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "linear_internal.h"

#include <cstring>

namespace aethermind::cpu::detail {
namespace {

float LoadW16(const std::byte* data, DataType dtype) noexcept {
    uint16_t bits = 0;
    std::memcpy(&bits, data, sizeof(bits));
    if (dtype == DataType::Float(16)) {
        return Half(bits, Half::from_bits());
    }
    return BFloat16(bits, BFloat16::from_bits());
}

} // namespace

Status RunLinearF32W16Reference(const LinearF32W16KernelArgs& args) noexcept {
    if (args.row_count == 0 || args.out_features == 0) {
        return Status::Ok();
    }

    if (args.in_features == 0) {
        for (int64_t r = 0; r < args.row_count; ++r) {
            float* const output_row = args.output + r * args.output_row_stride;
            for (int64_t c = 0; c < args.out_features; ++c) {
                output_row[c * args.output_col_stride] = 0.0F;
            }
        }
        return Status::Ok();
    }

    for (int64_t row = 0; row < args.row_count; ++row) {
        const float* const input_row = args.input + row * args.input_row_stride;
        float* const output_row = args.output + row * args.output_row_stride;
        for (int64_t col = 0; col < args.out_features; ++col) {
            const std::byte* const weight_row =
                    args.weight +
                    static_cast<size_t>(col * args.weight_row_stride) * sizeof(uint16_t);
            float sum = 0.0F;
            for (int64_t k = 0; k < args.in_features; ++k) {
                const std::byte* const weight_element =
                        weight_row +
                        static_cast<size_t>(k * args.weight_col_stride) * sizeof(uint16_t);
                sum += input_row[k * args.input_col_stride] *
                       LoadW16(weight_element, args.weight_dtype);
            }
            output_row[col * args.output_col_stride] = sum;
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
