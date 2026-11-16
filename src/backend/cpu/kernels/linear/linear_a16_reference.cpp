#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "linear_internal.h"

#include <cstring>

namespace aethermind::cpu::detail {

bool IsBf16NativeDotProductSafe(const std::byte* input,
                                const std::byte* weight,
                                int64_t element_count) noexcept {
    for (int64_t index = 0; index < element_count; ++index) {
        uint16_t input_bits = 0;
        uint16_t weight_bits = 0;
        std::memcpy(&input_bits, input + static_cast<size_t>(index) * sizeof(uint16_t), sizeof(input_bits));
        std::memcpy(&weight_bits, weight + static_cast<size_t>(index) * sizeof(uint16_t), sizeof(weight_bits));
        const unsigned input_exponent = (input_bits >> 7U) & 0xFFU;
        const unsigned weight_exponent = (weight_bits >> 7U) & 0xFFU;
        if (input_exponent == 0xFFU || weight_exponent == 0xFFU) {
            return false;
        }
        if ((input_bits & 0x7FFFU) == 0 || (weight_bits & 0x7FFFU) == 0) {
            continue;
        }
        if (input_exponent == 0 || weight_exponent == 0) {
            return false;
        }

        // A normal BF16 product is an integer multiple of
        // 2^(e_input + e_weight - 14). Requiring e_input + e_weight >= -112
        // makes every product, and every rounded partial sum, a multiple of
        // FLT_MIN. Even cancellation then cannot create a nonzero subnormal.
        // Biased exponents add 254. The upper bound also excludes products
        // near F32 overflow; unusual magnitudes use the reference unchanged.
        const unsigned exponent_sum = input_exponent + weight_exponent;
        if (exponent_sum < 142U || exponent_sum > 379U) {
            return false;
        }
    }
    return true;
}

Status RunLinearA16Reference(const LinearA16KernelArgs& args) noexcept {
    if (args.row_count == 0 || args.out_features == 0) {
        return Status::Ok();
    }

    const size_t input_bytes = A16ElementBytes(args.act_dtype);
    const size_t weight_bytes = A16ElementBytes(args.act_dtype);
    const size_t output_bytes = A16ElementBytes(args.output_dtype);

    if (args.in_features == 0) {
        const float zero = 0.0F;
        for (int64_t row = 0; row < args.row_count; ++row) {
            for (int64_t col = 0; col < args.out_features; ++col) {
                std::byte* const destination =
                        args.output +
                        (static_cast<size_t>(row * args.output_row_stride + col * args.output_col_stride)) *
                                output_bytes;
                if (args.output_dtype == DataType::Float32()) {
                    std::memcpy(destination, &zero, sizeof(zero));
                } else {
                    const uint16_t zero_bits = 0U;
                    std::memcpy(destination, &zero_bits, sizeof(zero_bits));
                }
            }
        }
        return Status::Ok();
    }

    for (int64_t row = 0; row < args.row_count; ++row) {
        for (int64_t col = 0; col < args.out_features; ++col) {
            const std::byte* const weight_row =
                    args.weight +
                    static_cast<size_t>(col * args.weight_row_stride) * weight_bytes;
            float sum = 0.0F;
            for (int64_t k = 0; k < args.in_features; ++k) {
                const std::byte* const input_element =
                        args.input +
                        (static_cast<size_t>(row * args.input_row_stride + k * args.input_col_stride)) *
                                input_bytes;
                const std::byte* const weight_element =
                        weight_row +
                        static_cast<size_t>(k * args.weight_col_stride) * weight_bytes;
                sum += LoadA16Element(input_element, args.act_dtype) *
                       LoadA16Element(weight_element, args.act_dtype);
            }
            std::byte* const destination =
                    args.output +
                    (static_cast<size_t>(row * args.output_row_stride + col * args.output_col_stride)) *
                            output_bytes;
            StoreA16Element(destination, sum, args.output_dtype);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
