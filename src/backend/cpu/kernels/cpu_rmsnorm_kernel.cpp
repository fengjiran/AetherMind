#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/cpu/kernels/cpu_simd_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/operators/op_type.h"

#include <cstdint>
#include <immintrin.h>

namespace aethermind {
namespace {

const CpuRmsNormAttrs* GetAttrs(std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(CpuRmsNormAttrs)) {
        return nullptr;
    }
    return reinterpret_cast<const CpuRmsNormAttrs*>(attrs.data());
}

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
}

}// namespace

Status CpuRmsNormKernel(const KernelInvocation& invocation,
                        const KernelContext& op_ctx,
                        const WorkspaceBinding&) noexcept {
    if (invocation.op_type != OpType::kRmsNorm) {
        return Status::InvalidArgument("CpuRmsNormKernel only supports OpType::kRmsNorm");
    }

    if (!op_ctx.device.is_cpu()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CPU device");
    }

    const CpuRmsNormAttrs* attrs = GetAttrs(op_ctx.attrs);
    if (attrs == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CpuRmsNormAttrs in KernelContext.attrs");
    }

    const CpuRmsNormParams* params = GetParams(op_ctx.packed_params);
    if (params == nullptr) {
        return Status::InvalidArgument("CpuRmsNormKernel requires CpuRmsNormParams in KernelContext.packed_params");
    }

    const TensorView& input = params->Input;
    const TensorView& weight = params->Weight;
    const MutableTensorView& output = params->Output;

    if (!input.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires a valid input TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires a valid output MutableTensorView");
    }

    if (input.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires input TensorView to have float dtype");
    }

    if (weight.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight TensorView to have float dtype");
    }

    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires output MutableTensorView to have float dtype");
    }

    if (input.rank() != 2 || !input.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires input TensorView to be contiguous 2D [seq_len, hidden]");
    }

    if (weight.rank() != 1 || !weight.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight TensorView to be contiguous 1D");
    }

    if (output.rank() != 2 || !output.is_contiguous()) {
        return Status::InvalidArgument("CpuRmsNormKernel requires output MutableTensorView to be contiguous 2D [seq_len, hidden]");
    }

    const int64_t seq_len = input.dim(0);
    const int64_t hidden_size_i64 = input.dim(1);
    if (seq_len <= 0 || hidden_size_i64 <= 0) {
        return Status::InvalidArgument("CpuRmsNormKernel requires positive seq_len and hidden_size");
    }

    if (output.dim(0) != seq_len || output.dim(1) != hidden_size_i64) {
        return Status::InvalidArgument("CpuRmsNormKernel output shape must match input [seq_len, hidden]");
    }

    if (weight.numel() != hidden_size_i64) {
        return Status::InvalidArgument("CpuRmsNormKernel requires weight length to match hidden_size");
    }

    const auto hidden_size = static_cast<size_t>(hidden_size_i64);
    const auto* const input_data = input.data<float>();
    const auto* const weight_data = weight.data<float>();
    auto* const output_data = output.data<float>();
    const float epsilon = attrs->Epsilon;

    auto process_row = [input_data, output_data, weight_data, hidden_size, epsilon](int64_t row_idx) noexcept {
        const float* const row_in = input_data + row_idx * hidden_size;
        float* const row_out = output_data + row_idx * hidden_size;

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
        float mean_sq = sum_sq / hidden_size;
        float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

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

            __m256 w0 = _mm256_loadu_ps(weight_data + j);
            __m256 w1 = _mm256_loadu_ps(weight_data + j + 8);
            __m256 w2 = _mm256_loadu_ps(weight_data + j + 16);
            __m256 w3 = _mm256_loadu_ps(weight_data + j + 24);

            __m256 out0 = _mm256_mul_ps(x0, w0);
            __m256 out1 = _mm256_mul_ps(x1, w1);
            __m256 out2 = _mm256_mul_ps(x2, w2);
            __m256 out3 = _mm256_mul_ps(x3, w3);

            _mm256_storeu_ps(row_out + j, out0);
            _mm256_storeu_ps(row_out + j + 8, out1);
            _mm256_storeu_ps(row_out + j + 16, out2);
            _mm256_storeu_ps(row_out + j + 24, out3);
        }

        for (; j + 8 <= hidden_size; j += 8) {
            __m256 x0 = _mm256_loadu_ps(row_in + j);
            __m256 w0 = _mm256_loadu_ps(weight_data + j);
            x0 = _mm256_mul_ps(x0, inv_rms_vec);
            _mm256_storeu_ps(row_out + j, _mm256_mul_ps(x0, w0));
        }

        for (; j < hidden_size; ++j) {
            row_out[j] = row_in[j] * inv_rms * weight_data[j];
        }
    };

    if (constexpr int64_t kOmpParallelThreshold = 16; seq_len <= kOmpParallelThreshold) {
        for (int64_t i = 0; i < seq_len; ++i) {
            process_row(i);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < seq_len; ++i) {
            process_row(i);
        }
    }

    return Status::Ok();
}

}// namespace aethermind
