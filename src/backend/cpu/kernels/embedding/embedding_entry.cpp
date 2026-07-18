#include "embedding_internal.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/utils/overflow_check.h"

#include <algorithm>
#include <cstdint>

namespace aethermind {
namespace {

const cpu::detail::EmbeddingParams* GetParams(const void* kernel_params) noexcept {
    return static_cast<const cpu::detail::EmbeddingParams*>(kernel_params);
}

bool IsSupportedTokenIdDType(const DataType& dtype) noexcept {
    return dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::UInt(32);
}

int64_t ReadTokenId(const TensorView& token_ids, size_t index) noexcept {
    if (token_ids.dtype() == DataType::Int(32)) {
        return token_ids.data<int32_t>()[index];
    }
    if (token_ids.dtype() == DataType::UInt(32)) {
        return static_cast<int64_t>(token_ids.data<uint32_t>()[index]);
    }
    return token_ids.data<int64_t>()[index];
}

}// namespace

Status cpu::detail::EmbeddingKernel(const KernelContext& ctx) noexcept {
    const cpu::detail::EmbeddingParams* params = GetParams(ctx.kernel_params);
    if (params == nullptr) {
        return Status::InvalidArgument("EmbeddingKernel requires cpu::detail::EmbeddingParams in KernelContext.kernel_params");
    }

    const TensorView& token_ids = params->token_ids;
    const TensorView& weight = params->weight;
    const MutableTensorView& output = params->output;

    if (!token_ids.is_valid()) {
        return Status::InvalidArgument("EmbeddingKernel requires a valid token id TensorView");
    }
    if (!weight.is_valid()) {
        return Status::InvalidArgument("EmbeddingKernel requires a valid weight TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument("EmbeddingKernel requires a valid output MutableTensorView");
    }

    if (!IsSupportedTokenIdDType(token_ids.dtype())) {
        return Status::InvalidArgument("EmbeddingKernel token ids must be int32, int64, or uint32");
    }
    if (weight.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("EmbeddingKernel requires float32 weight TensorView");
    }
    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("EmbeddingKernel requires float32 output MutableTensorView");
    }

    if (token_ids.rank() != 1 || !token_ids.is_contiguous()) {
        return Status::InvalidArgument("EmbeddingKernel requires contiguous 1D token ids");
    }
    if (weight.rank() != 2 || !weight.is_contiguous()) {
        return Status::InvalidArgument("EmbeddingKernel requires contiguous 2D weight");
    }
    if (output.rank() != 2 || !output.is_contiguous()) {
        return Status::InvalidArgument("EmbeddingKernel requires contiguous 2D output");
    }

    const int64_t token_count = token_ids.numel();
    const int64_t vocab_size = weight.dim(0);
    const int64_t hidden_size = weight.dim(1);
    if (token_count <= 0 || vocab_size <= 0 || hidden_size <= 0) {
        return Status::InvalidArgument("EmbeddingKernel requires positive token, vocab, and hidden sizes");
    }
    if (output.dim(0) != token_count || output.dim(1) != hidden_size) {
        return Status::InvalidArgument("EmbeddingKernel output shape must be [token_count, hidden_size]");
    }

    // Null data guard: is_valid() does not reject nullptr data_ with positive shapes.
    if (token_ids.data() == nullptr) {
        return Status::InvalidArgument("EmbeddingKernel requires non-null token id data");
    }
    if (weight.data() == nullptr) {
        return Status::InvalidArgument("EmbeddingKernel requires non-null weight data");
    }
    if (output.data() == nullptr) {
        return Status::InvalidArgument("EmbeddingKernel requires non-null output data");
    }

    // Check that dimension products do not overflow size_t before computing offsets.
    const auto hidden = static_cast<size_t>(hidden_size);
    {
        size_t weight_total = 0;
        if (CheckOverflowMul(static_cast<size_t>(vocab_size), hidden, &weight_total)) {
            return Status::InvalidArgument("EmbeddingKernel weight dimensions overflow");
        }
        size_t output_total = 0;
        if (CheckOverflowMul(static_cast<size_t>(token_count), hidden, &output_total)) {
            return Status::InvalidArgument("EmbeddingKernel output dimensions overflow");
        }
    }
    const auto* const weight_data = weight.data<float>();
    auto* const output_data = output.data<float>();

    for (int64_t token = 0; token < token_count; ++token) {
        const int64_t token_id = ReadTokenId(token_ids, static_cast<size_t>(token));
        if (token_id < 0 || token_id >= vocab_size) {
            return Status::OutOfRange("EmbeddingKernel token id is out of vocabulary range");
        }

        const auto source_offset = static_cast<size_t>(token_id) * hidden;
        const auto target_offset = static_cast<size_t>(token) * hidden;
        std::copy_n(weight_data + source_offset, hidden, output_data + target_offset);
    }

    return Status::Ok();
}

Status BuildEmbeddingParams(std::span<const TensorView> inputs,
                                   std::span<const MutableTensorView> outputs,
                                   void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Embedding requires 2 inputs and 1 output");
    }
    ::new (params_buffer) cpu::detail::EmbeddingParams{
            .token_ids = inputs[0],
            .weight = inputs[1],
            .output = outputs[0],
    };
    return Status::Ok();
}

AM_REGISTER_KERNEL(EmbeddingFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &cpu::detail::EmbeddingKernel,
                           .name = "cpu::embedding_f32_scalar",
                           .priority = 10,
                           .params_builder = &BuildEmbeddingParams,
                           .params_size = sizeof(cpu::detail::EmbeddingParams),
                   })

}// namespace aethermind
