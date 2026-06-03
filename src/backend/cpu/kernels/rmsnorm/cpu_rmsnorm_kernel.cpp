#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/cpu/kernels/common/cpu_simd_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "cpu_rmsnorm_internal.h"

#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind {
namespace {

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
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

#if defined(__AVX2__) && defined(__FMA__)
AM_ALWAYS_INLINE void rmsnorm_micro_kernel_avx2(float* __restrict__ output,
                                                const float* __restrict__ input,
                                                const float* __restrict__ weight,
                                                int64_t hidden_size,
                                                float epsilon) {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    int64_t j = 0;
    for (; j + 32 <= hidden_size; j += 32) {
        const __m256 x0 = _mm256_loadu_ps(input + j);
        const __m256 x1 = _mm256_loadu_ps(input + j + 8);
        const __m256 x2 = _mm256_loadu_ps(input + j + 16);
        const __m256 x3 = _mm256_loadu_ps(input + j + 24);

        vsum0 = _mm256_fmadd_ps(x0, x0, vsum0);
        vsum1 = _mm256_fmadd_ps(x1, x1, vsum1);
        vsum2 = _mm256_fmadd_ps(x2, x2, vsum2);
        vsum3 = _mm256_fmadd_ps(x3, x3, vsum3);
    }

    __m256 vres = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1), _mm256_add_ps(vsum2, vsum3));
    for (; j + 8 <= hidden_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(input + j);
        vres = _mm256_fmadd_ps(x0, x0, vres);
    }

    float sum_sq = HorizontalSumAvx2(vres);
    for (; j < hidden_size; ++j) {
        sum_sq += input[j] * input[j];
    }

    const float mean_sq = sum_sq / static_cast<float>(hidden_size);
    const float inv_rms = 1.0F / std::sqrt(mean_sq + epsilon);
    const __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

    j = 0;
    for (; j + 32 <= hidden_size; j += 32) {
        __m256 x0 = _mm256_loadu_ps(input + j);
        __m256 x1 = _mm256_loadu_ps(input + j + 8);
        __m256 x2 = _mm256_loadu_ps(input + j + 16);
        __m256 x3 = _mm256_loadu_ps(input + j + 24);

        x0 = _mm256_mul_ps(x0, inv_rms_vec);
        x1 = _mm256_mul_ps(x1, inv_rms_vec);
        x2 = _mm256_mul_ps(x2, inv_rms_vec);
        x3 = _mm256_mul_ps(x3, inv_rms_vec);

        const __m256 w0 = _mm256_loadu_ps(weight + j);
        const __m256 w1 = _mm256_loadu_ps(weight + j + 8);
        const __m256 w2 = _mm256_loadu_ps(weight + j + 16);
        const __m256 w3 = _mm256_loadu_ps(weight + j + 24);

        const __m256 out0 = _mm256_mul_ps(x0, w0);
        const __m256 out1 = _mm256_mul_ps(x1, w1);
        const __m256 out2 = _mm256_mul_ps(x2, w2);
        const __m256 out3 = _mm256_mul_ps(x3, w3);

        _mm256_storeu_ps(output + j, out0);
        _mm256_storeu_ps(output + j + 8, out1);
        _mm256_storeu_ps(output + j + 16, out2);
        _mm256_storeu_ps(output + j + 24, out3);
    }

    for (; j + 8 <= hidden_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(input + j);
        const __m256 w0 = _mm256_loadu_ps(weight + j);
        x0 = _mm256_mul_ps(x0, inv_rms_vec);
        _mm256_storeu_ps(output + j, _mm256_mul_ps(x0, w0));
    }

    for (; j < hidden_size; ++j) {
        output[j] = input[j] * inv_rms * weight[j];
    }
}
#endif

Status CpuRmsNormKernel(const CpuRmsNormKernelArgs& args) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    auto process_row = [&args](int64_t row_idx) noexcept {
        rmsnorm_micro_kernel_avx2(args.output_ + row_idx * args.output_row_stride_,
                                  args.input_ + row_idx * args.input_row_stride_,
                                  args.weight_,
                                  args.hidden_size_,
                                  args.epsilon_);
    };
#else
    auto process_row = [&args](int64_t row_idx) noexcept {
        ProcessStridedRmsNormRowScalar(args, row_idx);
    };
#endif

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
