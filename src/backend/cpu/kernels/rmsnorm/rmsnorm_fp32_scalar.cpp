#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "rmsnorm_internal.h"

#include <cmath>

namespace aethermind {
namespace {

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
}

void ProcessRmsNormRowScalar(const RmsNormFp32KernelArgs& args, int64_t row_idx) noexcept {
    const float* const row_in = args.input + row_idx * args.input_row_stride;
    float* const row_out = args.output + row_idx * args.output_row_stride;

    double sum_sq = 0.0;
    for (int64_t j = 0; j < args.hidden_size; ++j) {
        const auto x = static_cast<double>(row_in[j * args.input_col_stride]);
        sum_sq += x * x;
    }

    const double mean_sq = sum_sq / static_cast<double>(args.hidden_size);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(args.epsilon));
    for (int64_t j = 0; j < args.hidden_size; ++j) {
        row_out[j * args.output_col_stride] = static_cast<float>(static_cast<double>(row_in[j * args.input_col_stride]) *
                                                                 inv_rms * static_cast<double>(args.weight[j * args.weight_stride]));
    }
}

}// namespace


Status RmsNormKernel_CPU_FP32_Scalar(const RmsNormFp32KernelArgs& args) noexcept {
    if (constexpr int64_t kOmpParallelThreshold = 16; args.seq_len <= kOmpParallelThreshold) {
        for (int64_t i = 0; i < args.seq_len; ++i) {
            ProcessRmsNormRowScalar(args, i);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < args.seq_len; ++i) {
            ProcessRmsNormRowScalar(args, i);
        }
    }

    return Status::Ok();
}

}// namespace aethermind
