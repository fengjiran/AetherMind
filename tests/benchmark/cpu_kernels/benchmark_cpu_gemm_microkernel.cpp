#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using namespace aethermind;

using GemmRunner = Status (*)(const cpu::detail::GemmF32Args&) noexcept;

enum class RhsLayout {
    /// Logical B[K,N] is stored row-major; N is contiguous.
    kNContiguous,
    /// Logical B[K,N] is backed by row-major weight W[N,K]; K is contiguous.
    kKContiguous
};

float DeterministicValue(size_t index) {
    constexpr size_t kPeriod = 251;
    return static_cast<float>(static_cast<int64_t>(index % kPeriod) -
                              static_cast<int64_t>(kPeriod / 2U)) *
           0.0078125F;
}

void FillDeterministic(std::vector<float>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = DeterministicValue(index);
    }
}

const char* LayoutName(RhsLayout layout) noexcept {
    switch (layout) {
        case RhsLayout::kNContiguous:
            return "n-contiguous";
        case RhsLayout::kKContiguous:
            return "k-contiguous";
    }
    return "unknown";
}

Status ValidateGemmOutput(const cpu::detail::GemmF32Args& args,
                          float error_tolerance) {
    for (int64_t row = 0; row < args.m; ++row) {
        for (int64_t col = 0; col < args.n; ++col) {
            double expected = 0.0;
            for (int64_t inner = 0; inner < args.k; ++inner) {
                expected += static_cast<double>(
                                    args.lhs[row * args.lhs_m_stride +
                                             inner * args.lhs_k_stride]) *
                            static_cast<double>(
                                    args.rhs[inner * args.rhs_k_stride +
                                             col * args.rhs_n_stride]);
            }

            const float actual = args.output[row * args.output_m_stride +
                                             col * args.output_n_stride];
            const auto reference = static_cast<float>(expected);
            if (!std::isfinite(actual) || !std::isfinite(reference)) {
                return Status::Internal(
                        "GEMM microkernel benchmark correctness guard produced NaN or Inf");
            }

            const float absolute_error = std::fabs(actual - reference);
            const float relative_error = absolute_error / std::max(std::fabs(reference), 1.0e-6F);
            if (absolute_error > error_tolerance && relative_error > error_tolerance) {
                return Status::Internal(
                        "GEMM microkernel benchmark disagrees with its independent oracle");
            }
        }
    }
    return Status::Ok();
}

void SetGemmCounters(benchmark::State& state,
                     int64_t m,
                     int64_t k,
                     int64_t n) {
    const int64_t output_elements = m * n;
    const int64_t logical_bytes = static_cast<int64_t>(sizeof(float)) *
                                  (m * k + k * n + output_elements);
    state.counters["M"] = static_cast<double>(m);
    state.counters["K"] = static_cast<double>(k);
    state.counters["N"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * output_elements);
    state.SetBytesProcessed(state.iterations() * logical_bytes);

    const double flops = static_cast<double>(state.iterations()) * 2.0 *
                         static_cast<double>(m) * static_cast<double>(n) *
                         static_cast<double>(k);
    const double bytes = static_cast<double>(state.iterations()) *
                         static_cast<double>(logical_bytes);
    state.counters["GFLOP/s"] = benchmark::Counter(
            flops, benchmark::Counter::kIsRate,
            benchmark::Counter::OneK::kIs1000);
    state.counters["logical GB/s"] = benchmark::Counter(
            bytes, benchmark::Counter::kIsRate,
            benchmark::Counter::OneK::kIs1000);
}

void BenchmarkGemmF32(benchmark::State& state,
                      RhsLayout rhs_layout,
                      GemmRunner runner,
                      float error_tolerance) {
    const int64_t m = state.range(0);
    const int64_t k = state.range(1);
    const int64_t n = state.range(2);
    const auto lhs_elements = static_cast<size_t>(m * k);
    const auto rhs_elements = static_cast<size_t>(k * n);
    const auto output_elements = static_cast<size_t>(m * n);

    std::vector<float> lhs(lhs_elements);
    std::vector<float> rhs(rhs_elements);
    std::vector<float> output(output_elements,
                              std::numeric_limits<float>::quiet_NaN());
    FillDeterministic(lhs);
    FillDeterministic(rhs);

    const cpu::detail::GemmF32Args args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = output.data(),
            .m = m,
            .n = n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = rhs_layout == RhsLayout::kNContiguous ? n : 1,
            .rhs_n_stride = rhs_layout == RhsLayout::kNContiguous ? 1 : k,
            .output_m_stride = n,
            .output_n_stride = 1,
    };

    const Status correctness_run = runner(args);
    if (!correctness_run.ok()) {
        state.SkipWithError(correctness_run.ToString());
        return;
    }

    const Status correctness = ValidateGemmOutput(args, error_tolerance);
    if (!correctness.ok()) {
        state.SkipWithError(correctness.ToString());
        return;
    }

    state.SetLabel(LayoutName(rhs_layout));
    for (auto _: state) {
        const Status status = runner(args);
        if (!status.ok()) {
            state.SkipWithError(status.ToString());
            break;
        }
        benchmark::DoNotOptimize(output.data());
    }

    SetGemmCounters(state, m, k, n);
}

void BM_GemmF32ReferenceNContiguous(benchmark::State& state) {
    BenchmarkGemmF32(state, RhsLayout::kNContiguous,
                     &cpu::detail::RunGemmF32Reference, 1.0e-5F);
}

void BM_GemmF32ReferenceKContiguous(benchmark::State& state) {
    BenchmarkGemmF32(state, RhsLayout::kKContiguous,
                     &cpu::detail::RunGemmF32Reference, 1.0e-5F);
}

void BM_GemmF32ScalarOptimizedNContiguous(benchmark::State& state) {
    BenchmarkGemmF32(state, RhsLayout::kNContiguous,
                     &cpu::detail::RunGemmF32ScalarOptimized, 1.0e-4F);
}

void BM_GemmF32ScalarOptimizedKContiguous(benchmark::State& state) {
    BenchmarkGemmF32(state, RhsLayout::kKContiguous,
                     &cpu::detail::RunGemmF32ScalarOptimized, 1.0e-4F);
}

benchmark::Benchmark* RegisterGemmShapes(benchmark::Benchmark* benchmark) {
    return benchmark
            // Decode and small-M workloads.
            ->Args({1, 4096, 4096})
            ->Args({1, 4096, 11008})
            ->Args({4, 4096, 4096})
            // A bounded Prefill diagnostic baseline.
            ->Args({16, 4096, 4096})
            // Tail shapes for future tile-specific implementations.
            ->Args({1, 33, 31})
            ->Args({2, 32, 33})
            ->ArgNames({"M", "K", "N"});
}

benchmark::Benchmark* RegisterScalarGemmShapes(benchmark::Benchmark* benchmark) {
    return benchmark
            // The G1S candidate specializes M=1; keep direct comparisons
            // shape-identical across reference, portable, and strict variants.
            ->Args({1, 4096, 4096})
            ->Args({1, 4096, 11008})
            ->Args({1, 33, 31})
            ->Args({1, 32, 33})
            ->ArgNames({"M", "K", "N"});
}

const auto* const kGemmF32ReferenceNContiguous = RegisterGemmShapes(
        benchmark::RegisterBenchmark("BM_GemmF32ReferenceNContiguous",
                                     &BM_GemmF32ReferenceNContiguous));
const auto* const kGemmF32ReferenceKContiguous = RegisterGemmShapes(
        benchmark::RegisterBenchmark("BM_GemmF32ReferenceKContiguous",
                                     &BM_GemmF32ReferenceKContiguous));
const auto* const kGemmF32ScalarOptimizedNContiguous = RegisterScalarGemmShapes(
        benchmark::RegisterBenchmark("BM_GemmF32ScalarOptimizedNContiguous",
                                     &BM_GemmF32ScalarOptimizedNContiguous));
const auto* const kGemmF32ScalarOptimizedKContiguous = RegisterScalarGemmShapes(
        benchmark::RegisterBenchmark("BM_GemmF32ScalarOptimizedKContiguous",
                                     &BM_GemmF32ScalarOptimizedKContiguous));

} // namespace
