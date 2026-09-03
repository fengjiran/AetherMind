#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "embedding_internal.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <cstdint>

namespace aethermind::cpu::detail {
namespace {

int64_t ReadTokenId(const void* token_ids_data,
                    const DataType& dtype,
                    size_t index) noexcept {
    // BuildEmbeddingArgs validates dtype against the semantic contract first,
    // keeping the read path tied to kEmbeddingSupportedTokenIdDTypes.
    AM_DCHECK(aethermind::IsSupportedTokenIdDType(dtype));
    if (dtype == DataType::Int(32)) {
        return *(static_cast<const int32_t*>(token_ids_data) + index);
    }

    if (dtype == DataType::UInt(32)) {
        return *(static_cast<const uint32_t*>(token_ids_data) + index);
    }

    if (dtype == DataType::Int(64)) {
        return *(static_cast<const int64_t*>(token_ids_data) + index);
    }
    // Unreachable after BuildEmbeddingArgs validation; fall back without reading.
    return 0;
}

Status BuildEmbeddingArgs(const KernelParamsBuildContext& context, void* params_buffer) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires 2 inputs and 1 output");
    }

    const TensorView& token_ids = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];

    if (!token_ids.is_valid()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires a valid token id TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires a valid output MutableTensorView");
    }

    if (!IsSupportedTokenIdDType(token_ids.dtype())) {
        return Status::InvalidArgument(
                "EmbeddingKernel token ids must be int32, int64, or uint32");
    }

    if (weight.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires float32 weight TensorView");
    }

    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires float32 output MutableTensorView");
    }

    if (token_ids.rank() < 1 || !token_ids.is_contiguous()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires contiguous token ids with rank >= 1");
    }

    if (weight.rank() != 2 || !weight.is_contiguous()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires contiguous 2D weight");
    }

    const int64_t token_count = token_ids.numel();
    const int64_t vocab_size = weight.dim(0);
    const int64_t hidden_size = weight.dim(1);
    if (vocab_size <= 0 || hidden_size <= 0) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires positive vocab and hidden sizes");
    }

    // Output preserves all token-id axes and appends the hidden axis, matching
    // the semantic output shape [token_ids..., hidden_size]. The gather itself
    // is linear over the flattened token ids, so any contiguous rank works.
    if (output.rank() != token_ids.rank() + 1 || !output.is_contiguous()) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires contiguous output with rank = token ids rank + 1");
    }

    for (int32_t i = 0; i < token_ids.rank(); ++i) {
        if (output.dim(i) != token_ids.dim(i)) {
            return Status::InvalidArgument(
                    "EmbeddingKernel output shape must be [token ids..., hidden_size]");
        }
    }

    if (output.dim(token_ids.rank()) != hidden_size) {
        return Status::InvalidArgument(
                "EmbeddingKernel output shape must be [token ids..., hidden_size]");
    }

    // Empty output: no token ids to gather, nothing to write. Null data is
    // permitted for zero-element tensors (see TensorView [0] semantics).
    if (token_count == 0) {
        ::new (params_buffer) EmbeddingKernelArgs{
                .token_dtype = token_ids.dtype(),
                .token_count = 0,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
        };
        return Status::Ok();
    }

    // Null data guard: is_valid() does not reject nullptr data_ with positive shapes.
    if (token_ids.data() == nullptr) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires non-null token id data");
    }

    if (weight.data() == nullptr) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires non-null weight data");
    }

    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                "EmbeddingKernel requires non-null output data");
    }

    // Check that dimension products do not overflow size_t before computing offsets.
    size_t weight_total = 0;
    if (CheckOverflowMul(static_cast<size_t>(vocab_size),
                         static_cast<size_t>(hidden_size), &weight_total)) {
        return Status::InvalidArgument(
                "EmbeddingKernel weight dimensions overflow");
    }

    size_t output_total = 0;
    if (CheckOverflowMul(static_cast<size_t>(token_count),
                         static_cast<size_t>(hidden_size), &output_total)) {
        return Status::InvalidArgument(
                "EmbeddingKernel output dimensions overflow");
    }

    ::new (params_buffer) EmbeddingKernelArgs{
            .token_ids_data = token_ids.data(),
            .token_dtype = token_ids.dtype(),
            .weight_data = weight.data<float>(),
            .output_data = output.data<float>(),
            .token_count = token_count,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
    };
    return Status::Ok();
}

} // namespace

Status EmbeddingKernel(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const EmbeddingKernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);

    if (args->token_count == 0) {
        return Status::Ok();
    }

    for (int64_t i = 0; i < args->token_count; ++i) {
        const int64_t token_id = ReadTokenId(args->token_ids_data,
                                             args->token_dtype,
                                             static_cast<size_t>(i));
        if (token_id < 0 || token_id >= args->vocab_size) {
            return Status::OutOfRange(
                    "EmbeddingKernel token id is out of vocabulary range");
        }

        const auto source_offset = static_cast<size_t>(token_id) *
                                   static_cast<size_t>(args->hidden_size);
        const auto target_offset = static_cast<size_t>(i) *
                                   static_cast<size_t>(args->hidden_size);
        std::copy_n(args->weight_data + source_offset, args->hidden_size,
                    args->output_data + target_offset);
    }

    return Status::Ok();
}

// The prepared args must satisfy the PreparedExecutionBindings params arena contract.
static_assert(std::is_trivially_destructible_v<EmbeddingKernelArgs>);
static_assert(alignof(EmbeddingKernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(EmbeddingFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &EmbeddingKernel,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingKernelArgs),
                           .params_builder = &BuildEmbeddingArgs,
                           .name = "cpu::embedding_f32_scalar",
                   })

} // namespace aethermind::cpu::detail
