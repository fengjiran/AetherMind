#include "aethermind/backend/cpu/cpu_bpanel_packing.h"
#include "aethermind/base/macros.h"
#include "gemm_internal.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::cpu::detail {
namespace {

size_t PackedBIndex(int64_t n, int64_t k, int64_t n_blocks) noexcept {
    const size_t panel = static_cast<size_t>(k / cpu::kCpuBPanelF32V1KC);
    const size_t block = static_cast<size_t>(n / cpu::kCpuBPanelF32V1NR);
    const size_t panel_row = static_cast<size_t>(k % cpu::kCpuBPanelF32V1KC);
    const size_t column = static_cast<size_t>(n % cpu::kCpuBPanelF32V1NR);
    return (((panel * static_cast<size_t>(n_blocks) + block) *
                     static_cast<size_t>(cpu::kCpuBPanelF32V1KC) +
             panel_row) *
            static_cast<size_t>(cpu::kCpuBPanelF32V1NR)) +
           column;
}

} // namespace

Status RunGemmF32PackedBReference(const PackedGemmF32Args& packed) noexcept {
    const GemmF32Args& args = packed.gemm;
    if (args.m < 0 || args.n < 0 || args.k < 0 || packed.logical_n < 0 ||
        packed.logical_k < 0 || args.k != packed.logical_k ||
        packed.weight_n_offset < 0 || packed.n_blocks < 0 ||
        packed.weight_n_offset > packed.logical_n ||
        args.n > packed.logical_n - packed.weight_n_offset) {
        return Status::InvalidArgument(
                "Packed GEMM received invalid logical dimensions or output slice");
    }
    const int64_t expected_blocks =
            packed.logical_n / cpu::kCpuBPanelF32V1NR +
            (packed.logical_n % cpu::kCpuBPanelF32V1NR != 0);
    if (packed.n_blocks != expected_blocks) {
        return Status::InvalidArgument(
                "Packed GEMM n_blocks does not match its logical N dimension");
    }
    AM_ASSIGN_OR_RETURN(const size_t required_bytes,
                        cpu::CpuBPanelF32V1PackedByteSize(
                                packed.logical_n, packed.logical_k));
    if (packed.packed_nbytes != required_bytes ||
        (required_bytes != 0 &&
         (packed.packed_b == nullptr ||
          reinterpret_cast<std::uintptr_t>(packed.packed_b) %
                          cpu::kCpuBPanelF32V1Alignment !=
                  0))) {
        return Status::InvalidArgument(
                "Packed GEMM artifact does not match its bpanel byte/alignment contract");
    }
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }
    if (args.output == nullptr) {
        return Status::InvalidArgument("Packed GEMM requires non-null output data");
    }
    if (args.k == 0) {
        for (int64_t row = 0; row < args.m; ++row) {
            for (int64_t col = 0; col < args.n; ++col) {
                args.output[row * args.output_m_stride +
                            col * args.output_n_stride] = 0.0F;
            }
        }
        return Status::Ok();
    }
    if (args.lhs == nullptr) {
        return Status::InvalidArgument(
                "Packed GEMM requires non-null input data");
    }

    for (int64_t row = 0; row < args.m; ++row) {
        const float* const lhs_row = args.lhs + row * args.lhs_m_stride;
        float* const output_row = args.output + row * args.output_m_stride;
        for (int64_t col = 0; col < args.n; ++col) {
            const int64_t weight_n = packed.weight_n_offset + col;
            double sum = 0.0;
            for (int64_t k = 0; k < args.k; ++k) {
                sum += static_cast<double>(lhs_row[k * args.lhs_k_stride]) *
                       static_cast<double>(packed.packed_b[PackedBIndex(weight_n, k, packed.n_blocks)]);
            }
            output_row[col * args.output_n_stride] = static_cast<float>(sum);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
