#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A32_W32_POLICY_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A32_W32_POLICY_H

/// @file gemm_a32_w32_policy.h
/// @brief ISA-neutral layout and blocking policy for the A32W32 AVX2 GEMM
///        drivers.
///
/// Pure decisions only: no intrinsics, allocation, Status or host capability
/// checks. Callers must already know AVX2 and FMA are available, so this
/// header stays safe to include from translation units compiled without
/// per-ISA flags.

#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "gemm_internal.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// AVX2 B-panel lane count; must match kGemmAvx2NR in gemm_avx2_internal.h.
inline constexpr int64_t kF32Avx2NR = 16;

/// @brief Layout family selected for an F32 AVX2 GEMM.
enum class F32Avx2Layout {
    kUnsupported,
    kRhsKContiguous,
    kRhsNContiguous
};

/// @brief Compute strategy inside a selected layout family.
enum class F32Avx2Strategy {
    kNone,
    kRowPairs,
    kColumnBlockedRowPairs,
    kBlockedTile6
};

/// @brief Pure dispatch decision for RunGemmA32W32Avx2Fma.
///
/// `m_full`/`n_full` describe the rectangle covered by the selected driver;
/// the caller owns the remaining N and M tails. `include_odd_row` asks the
/// blocked driver to fold the final M row into its B-panel loop.
struct F32Avx2Plan {
    F32Avx2Layout layout = F32Avx2Layout::kUnsupported;
    F32Avx2Strategy strategy = F32Avx2Strategy::kNone;
    int64_t m_full = 0;
    int64_t n_full = 0;
    bool include_odd_row = false;
};

/// @brief Whether the N-contiguous AVX2 driver amortizes its tile setup.
constexpr bool CanUseF32NContiguousAvx2(int64_t m, int64_t k, int64_t n) noexcept {
    return m >= 4 && k >= 128 && n >= 64;
}

/// @brief Selects the AVX2 layout and blocking strategy for pre-validated args.
inline F32Avx2Plan PlanF32Avx2(const GemmArgs& args) noexcept {
    if (args.m <= 0 || args.n <= 0 || args.k <= 0 ||
        args.lhs_k_stride != 1 || args.output_n_stride != 1) {
        return {};
    }

    if (args.rhs_k_stride == 1) {
        constexpr int64_t kMaxRowPairMRows = 8;
        constexpr int64_t kMaxColumnBlockedRows = 9;
        constexpr int64_t kMinColumnBlockedK = 128;
        constexpr int64_t kBlockedOddRowMinK = 2048;
        constexpr int64_t kBlockedOddRowMinN = 1024;

        if (args.n < kF32Avx2NR) {
            return {F32Avx2Layout::kRhsKContiguous,
                    F32Avx2Strategy::kRowPairs,
                    args.m, args.n, false};
        }

        // Sharing each 16-row weight group across all row pairs amortizes the
        // extra calls only once K is long enough.
        const bool col_blocked = args.m > 2 && args.m <= kMaxColumnBlockedRows &&
                                 args.k >= kMinColumnBlockedK && args.n > kF32Avx2NR;
        if (col_blocked) {
            return {F32Avx2Layout::kRhsKContiguous,
                    F32Avx2Strategy::kColumnBlockedRowPairs,
                    args.m, args.n, false};
        }

        if (args.m <= kMaxRowPairMRows) {
            return {F32Avx2Layout::kRhsKContiguous,
                    F32Avx2Strategy::kRowPairs,
                    args.m, args.n, false};
        }

        // Wide projections amortize folding the final M row into the B-panel
        // loop; shorter shapes keep it on the row-pair tail, where rereading
        // the RHS costs less.
        bool include_odd_row = args.k >= kBlockedOddRowMinK && args.n >= kBlockedOddRowMinN;
        return {F32Avx2Layout::kRhsKContiguous,
                F32Avx2Strategy::kBlockedTile6,
                include_odd_row ? args.m : args.m - args.m % 2,
                args.n - args.n % kF32Avx2NR, include_odd_row};
    }

    if (args.rhs_n_stride == 1 && CanUseF32NContiguousAvx2(args.m, args.k, args.n)) {
        return {F32Avx2Layout::kRhsNContiguous,
                F32Avx2Strategy::kNone,
                args.m, args.n - args.n % kF32Avx2NR, false};
    }
    return {};
}

enum class PackedBStrategy {
    kScan,
    kBlocked,
    kReference
};

/// @brief Selects a consumer for a validated packed-B slice with nonempty output.
constexpr PackedBStrategy PlanPackedB(const PackedGemmF32Args& packed) noexcept {
    const auto& args = packed.gemm;
    if (args.k == 0 || args.output_n_stride != 1) {
        return PackedBStrategy::kReference;
    }

    constexpr int64_t kBlockedMinM = 9;
    constexpr int64_t nr = kCpuBPanelF32Kc512Nr16NR;
    if (args.m >= kBlockedMinM && args.n >= nr && args.lhs_k_stride == 1 &&
        packed.weight_n_offset % nr == 0) {
        return PackedBStrategy::kBlocked;
    }
    return PackedBStrategy::kScan;
}

} // namespace aethermind::cpu::detail

#endif
