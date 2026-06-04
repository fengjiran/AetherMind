#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_registration.h"

#include <cmath>
#include <cstring>

namespace aethermind {
namespace {

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
}

void ProcessRmsNormRowScalar(const CpuRmsNormKernelArgs& args, int64_t row_idx) noexcept {
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

Status CpuRmsNormKernelScalar(const CpuRmsNormKernelArgs& args) noexcept {
    if (constexpr int64_t kOmpParallelThreshold = 16; args.seq_len_ <= kOmpParallelThreshold) {
        for (int64_t i = 0; i < args.seq_len_; ++i) {
            ProcessRmsNormRowScalar(args, i);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < args.seq_len_; ++i) {
            ProcessRmsNormRowScalar(args, i);
        }
    }

    return Status::Ok();
}

}// namespace

Status CpuRmsNormKernelEntry_FP32_Scalar(const KernelContext& ctx) noexcept {
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

    return CpuRmsNormKernelScalar(CpuRmsNormKernelArgs{
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

AM_REGISTER_KERNEL(CpuRmsNormFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kRmsNorm,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .activation_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuRmsNormKernelEntry_FP32_Scalar,
                           .name = "cpu::rmsnorm_f32_scalar",
                           .priority = 10,
                   })

}// namespace aethermind
