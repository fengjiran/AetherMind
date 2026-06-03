#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/cpu/kernels/cpu_simd_utils.h"
#include "aethermind/backend/kernel_context.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace aethermind {
namespace {

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
}

bool HasUnitColumnStrides(const CpuRmsNormKernelArgs& args) noexcept {
    return args.input_col_stride_ == 1 && args.weight_stride_ == 1 && args.output_col_stride_ == 1;
}

void ProcessContiguousRmsNormRowAvx2(const CpuRmsNormKernelArgs& args, int64_t row_idx) noexcept {
    const auto hidden_size = static_cast<size_t>(args.hidden_size_);
    const float* const row_in = args.input_ + row_idx * args.input_row_stride_;
    float* const row_out = args.output_ + row_idx * args.output_row_stride_;

    // Phase1: sum of squares
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    size_t j = 0;
    for (; j + 32 <= hidden_size; j += 32) {
        const __m256 x0 = _mm256_loadu_ps(row_in + j);
        const __m256 x1 = _mm256_loadu_ps(row_in + j + 8);
        const __m256 x2 = _mm256_loadu_ps(row_in + j + 16);
        const __m256 x3 = _mm256_loadu_ps(row_in + j + 24);

        vsum0 = _mm256_fmadd_ps(x0, x0, vsum0);
        vsum1 = _mm256_fmadd_ps(x1, x1, vsum1);
        vsum2 = _mm256_fmadd_ps(x2, x2, vsum2);
        vsum3 = _mm256_fmadd_ps(x3, x3, vsum3);
    }

    __m256 vres = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1), _mm256_add_ps(vsum2, vsum3));
    for (; j + 8 <= hidden_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(row_in + j);
        vres = _mm256_fmadd_ps(x0, x0, vres);
    }

    float sum_sq = HorizontalSumAvx2(vres);
    for (; j < hidden_size; ++j) {
        sum_sq += row_in[j] * row_in[j];
    }

    // Phase2: InvRms
    const float mean_sq = sum_sq / static_cast<float>(hidden_size);
    const float inv_rms = 1.0F / std::sqrt(mean_sq + args.epsilon_);
    const __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

    // Phase3: Normalize + Scale
    j = 0;
    for (; j + 32 <= hidden_size; j += 32) {
        __m256 x0 = _mm256_loadu_ps(row_in + j);
        __m256 x1 = _mm256_loadu_ps(row_in + j + 8);
        __m256 x2 = _mm256_loadu_ps(row_in + j + 16);
        __m256 x3 = _mm256_loadu_ps(row_in + j + 24);

        x0 = _mm256_mul_ps(x0, inv_rms_vec);
        x1 = _mm256_mul_ps(x1, inv_rms_vec);
        x2 = _mm256_mul_ps(x2, inv_rms_vec);
        x3 = _mm256_mul_ps(x3, inv_rms_vec);

        const __m256 w0 = _mm256_loadu_ps(args.weight_ + j);
        const __m256 w1 = _mm256_loadu_ps(args.weight_ + j + 8);
        const __m256 w2 = _mm256_loadu_ps(args.weight_ + j + 16);
        const __m256 w3 = _mm256_loadu_ps(args.weight_ + j + 24);

        const __m256 out0 = _mm256_mul_ps(x0, w0);
        const __m256 out1 = _mm256_mul_ps(x1, w1);
        const __m256 out2 = _mm256_mul_ps(x2, w2);
        const __m256 out3 = _mm256_mul_ps(x3, w3);

        _mm256_storeu_ps(row_out + j, out0);
        _mm256_storeu_ps(row_out + j + 8, out1);
        _mm256_storeu_ps(row_out + j + 16, out2);
        _mm256_storeu_ps(row_out + j + 24, out3);
    }

    for (; j + 8 <= hidden_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(row_in + j);
        const __m256 w0 = _mm256_loadu_ps(args.weight_ + j);
        x0 = _mm256_mul_ps(x0, inv_rms_vec);
        _mm256_storeu_ps(row_out + j, _mm256_mul_ps(x0, w0));
    }

    for (; j < hidden_size; ++j) {
        row_out[j] = row_in[j] * inv_rms * args.weight_[j];
    }
}

}// namespace

void ProcessStridedRmsNormRowScalar(const CpuRmsNormKernelArgs& args, int64_t row_idx) noexcept {
    const float* const row_in = args.input_ + row_idx * args.input_row_stride_;
    float* const row_out = args.output_ + row_idx * args.output_row_stride_;

    double sum_sq = 0.0;
    for (int64_t j = 0; j < args.hidden_size_; ++j) {
        const auto x = static_cast<double>(row_in[j * args.input_col_stride_]);
        sum_sq += x * x;
    }

    const double mean_sq = sum_sq / static_cast<double>(args.hidden_size_);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(args.epsilon_));
    for (int64_t j = 0; j < args.hidden_size_; ++j) {
        row_out[j * args.output_col_stride_] = static_cast<float>(static_cast<double>(row_in[j * args.input_col_stride_]) *
                                                                  inv_rms * static_cast<double>(args.weight_[j * args.weight_stride_]));
    }
}

Status CpuRmsNormKernel(const CpuRmsNormKernelArgs& args) noexcept {
    const bool use_avx2 = HasUnitColumnStrides(args);
    auto process_row = [&args, use_avx2](int64_t row_idx) noexcept {
        if (use_avx2) {
            ProcessContiguousRmsNormRowAvx2(args, row_idx);
            return;
        }
        ProcessStridedRmsNormRowScalar(args, row_idx);
    };

    if (constexpr int64_t kOmpParallelThreshold = 16; args.seq_len_ <= kOmpParallelThreshold) {
        for (int64_t i = 0; i < args.seq_len_; ++i) {
            process_row(i);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < args.seq_len_; ++i) {
            process_row(i);
        }
    }

    return Status::Ok();
}

Status CpuRmsNormKernelEntry(const KernelContext& ctx) noexcept {
    float epsilon;
    if (ctx.attrs.size() != sizeof(float)) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }
    std::memcpy(&epsilon, ctx.attrs.data(), sizeof(float));

    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires finite positive epsilon");
    }

    const CpuRmsNormParams* params = GetParams(ctx.packed_params);
    if (params == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires CpuRmsNormParams in KernelContext.packed_params");
    }

    const TensorView& input = params->input_tensor;
    const TensorView& weight = params->weight_tensor;
    const MutableTensorView& output = params->output_tensor;

    if (!input.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires a valid input TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires a valid output MutableTensorView");
    }

    if (input.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires float32 input TensorView");
    }

    if (weight.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires float32 weight TensorView");
    }

    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires float32 output MutableTensorView");
    }

    if (input.rank() != 2) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires rank-2 input TensorView");
    }

    if (weight.rank() != 1) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires rank-1 weight TensorView");
    }

    if (output.rank() != 2) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires rank-2 output MutableTensorView");
    }

    if (!input.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires contiguous input TensorView");
    }

    if (!weight.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires contiguous weight TensorView");
    }

    if (!output.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires contiguous output MutableTensorView");
    }

    const int64_t seq_len = input.dim(0);
    const int64_t hidden_size = input.dim(1);
    if (seq_len <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive seq_len");
    }

    if (hidden_size <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive hidden_size");
    }

    if (weight.dim(0) != hidden_size) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires weight length to match hidden_size");
    }

    if (output.dim(0) != seq_len || output.dim(1) != hidden_size) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires output shape to match input shape");
    }

    if (input.data() == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires non-null input data pointer");
    }

    if (weight.data() == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires non-null weight data pointer");
    }

    if (output.data() == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires non-null output data pointer");
    }

    if (input.stride(0) <= 0 || input.stride(1) <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive input strides");
    }

    if (weight.stride(0) <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive weight stride");
    }

    if (output.stride(0) <= 0 || output.stride(1) <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernelEntry requires positive output strides");
    }

    return CpuRmsNormKernel(CpuRmsNormKernelArgs{
            .input_ = input.data<float>(),
            .weight_ = weight.data<float>(),
            .output_ = output.data<float>(),
            .seq_len_ = seq_len,
            .hidden_size_ = hidden_size,
            .input_row_stride_ = input.stride(0),
            .input_col_stride_ = input.stride(1),
            .weight_stride_ = weight.stride(0),
            .output_row_stride_ = output.stride(0),
            .output_col_stride_ = output.stride(1),
            .epsilon_ = epsilon,
    });
}

}// namespace aethermind
