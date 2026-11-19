#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "gemm_a16_w16_internal.h"

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

Status RunGemmA16W16Reference(const GemmA16W16Args& params) noexcept {
    const GemmArgs& args = params.gemm;
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }

    const size_t input_bytes = A16ElementBytes(params.operand_dtype);
    const size_t weight_bytes = A16ElementBytes(params.operand_dtype);
    const size_t output_bytes = A16ElementBytes(params.output_dtype);

    if (args.k == 0) {
        const float zero = 0.0F;
        for (int64_t row = 0; row < args.m; ++row) {
            for (int64_t col = 0; col < args.n; ++col) {
                std::byte* const destination =
                        args.output +
                        (static_cast<size_t>(row * args.output_m_stride + col * args.output_n_stride)) *
                                output_bytes;
                if (params.output_dtype == DataType::Float32()) {
                    std::memcpy(destination, &zero, sizeof(zero));
                } else {
                    const uint16_t zero_bits = 0U;
                    std::memcpy(destination, &zero_bits, sizeof(zero_bits));
                }
            }
        }
        return Status::Ok();
    }

    for (int64_t row = 0; row < args.m; ++row) {
        for (int64_t col = 0; col < args.n; ++col) {
            const std::byte* const weight_row =
                    args.rhs +
                    static_cast<size_t>(col * args.rhs_n_stride) * weight_bytes;
            float sum = 0.0F;
            for (int64_t k = 0; k < args.k; ++k) {
                const std::byte* const input_element =
                        args.lhs +
                        (static_cast<size_t>(row * args.lhs_m_stride + k * args.lhs_k_stride)) *
                                input_bytes;
                const std::byte* const weight_element =
                        weight_row +
                        static_cast<size_t>(k * args.rhs_k_stride) * weight_bytes;
                sum += LoadA16Element(input_element, params.operand_dtype) *
                       LoadA16Element(weight_element, params.operand_dtype);
            }
            std::byte* const destination =
                    args.output +
                    (static_cast<size_t>(row * args.output_m_stride + col * args.output_n_stride)) *
                            output_bytes;
            StoreA16Element(destination, sum, params.output_dtype);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
