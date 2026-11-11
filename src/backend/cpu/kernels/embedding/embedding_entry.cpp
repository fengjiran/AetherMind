#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/packed_weight_validation.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/check.h"
#include "aethermind/dtypes/bf16_conversion.h"
#include "aethermind/dtypes/fp16_conversion.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "backend/cpu/bf16_conversion_internal.h"
#include "backend/cpu/fp16_conversion_internal.h"
#include "embedding_internal.h"
#include "utils/overflow_check.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

// ── FP32 weight ──────────────────────────────────────────────────────────────

Status BuildEmbeddingF32ReferenceArgs(const KernelParamsBuildContext& context, void* params_buffer) noexcept {
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
        ::new (params_buffer) EmbeddingF32KernelArgs{
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

    AM_ASSIGN_OR_RETURN(const ByteAddressRange token_ids_range,
                        BuildContiguousByteRange(token_ids, "EmbeddingKernel token ids"));
    AM_ASSIGN_OR_RETURN(const ByteAddressRange weight_range,
                        BuildContiguousByteRange(weight, "EmbeddingKernel weight"));
    AM_ASSIGN_OR_RETURN(const ByteAddressRange output_range,
                        BuildContiguousByteRange(output, "EmbeddingKernel output"));

    // The gather reads token ids and weight rows progressively, so writing into
    // either one can corrupt values later iterations still need. All three views
    // are contiguous, hence an intersecting range is proven byte overlap.
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU Embedding", output_range,
            "output", token_ids_range, "token ids"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU Embedding", output_range,
            "output", weight_range, "weight"));

    ::new (params_buffer) EmbeddingF32KernelArgs{
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

Status BuildEmbeddingF32PackedIdentityArgs(const KernelParamsBuildContext& context,
                                           void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed Embedding requires token ids, output, and a weight artifact");
    }

    const TensorView& token_ids = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!token_ids.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "Packed Embedding requires valid token id and output views");
    }

    if (!IsSupportedTokenIdDType(token_ids.dtype()) ||
        token_ids.rank() < 1 || !token_ids.is_contiguous()) {
        return Status::InvalidArgument(
                "Packed Embedding requires contiguous supported token ids with rank >= 1");
    }

    if (output.dtype() != DataType::Float32() ||
        output.rank() != token_ids.rank() + 1 || !output.is_contiguous()) {
        return Status::InvalidArgument(
                "Packed Embedding requires contiguous float32 output with an appended hidden axis");
    }

    if (packed.logical_shape.size() != 2U ||
        packed.logical_shape[0] <= 0 || packed.logical_shape[1] <= 0) {
        return Status::InvalidArgument(
                "Packed Embedding requires positive rank-2 logical weight dimensions");
    }

    const int64_t vocab_size = packed.logical_shape[0];
    const int64_t hidden_size = packed.logical_shape[1];
    const std::array<int64_t, 2> expected_shape{vocab_size, hidden_size};
    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            packed, expected_shape, "EmbeddingKernelEntry"));
    for (int32_t i = 0; i < token_ids.rank(); ++i) {
        if (output.dim(i) != token_ids.dim(i)) {
            return Status::InvalidArgument(
                    "Packed Embedding output shape must be [token ids..., hidden_size]");
        }
    }

    if (output.dim(token_ids.rank()) != hidden_size) {
        return Status::InvalidArgument(
                "Packed Embedding output shape must be [token ids..., hidden_size]");
    }

    const int64_t token_count = token_ids.numel();
    if (token_count == 0) {
        ::new (params_buffer) EmbeddingF32KernelArgs{
                .token_dtype = token_ids.dtype(),
                .token_count = 0,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
        };
        return Status::Ok();
    }

    if (token_ids.data() == nullptr || output.data() == nullptr ||
        packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed Embedding requires non-null storage for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange token_range,
                        BuildContiguousByteRange(token_ids,
                                                 "Packed Embedding token ids"));
    AM_ASSIGN_OR_RETURN(const ByteAddressRange output_range,
                        BuildContiguousByteRange(output,
                                                 "Packed Embedding output"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU Embedding", output_range,
            "output", token_range, "token ids"));
    if (!std::in_range<int64_t>(packed.nbytes)) {
        return Status::Overflow(
                "Packed Embedding artifact byte range exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                packed.data, static_cast<int64_t>(packed.nbytes),
                                size_t{1}, "Packed Embedding weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU Embedding", output_range, "output", packed_range,
            "packed weight"));

    ::new (params_buffer) EmbeddingF32KernelArgs{
            .token_ids_data = token_ids.data(),
            .token_dtype = token_ids.dtype(),
            .weight_data = static_cast<const float*>(packed.data),
            .output_data = output.data<float>(),
            .token_count = token_count,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
    };
    return Status::Ok();
}

Status EmbeddingF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const EmbeddingF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunEmbeddingF32Reference(*args);
}

// ── FP16/BF16 weight ─────────────────────────────────────────────────────────

Status BuildEmbeddingF32W16Metadata(const OpParams& params,
                                    std::vector<std::byte>& attrs) {
    if (const auto* embedding = std::get_if<EmbeddingParams>(&params);
        embedding == nullptr || embedding->output_dtype.raw_lanes() != 1 ||
        embedding->output_dtype != DataType::Float32()) {
        return Status::InvalidArgument(
                "W16 Embedding requires explicit float32 output_dtype");
    }
    attrs.clear();
    return Status::Ok();
}

template<bool Bfloat, EmbeddingRowWiden Widen = Bfloat ? ConvertBf16ToFp32Scalar : ConvertFp16ToFp32Scalar>
Status BuildEmbeddingF32W16Args(const KernelParamsBuildContext& context,
                                void* params_buffer) noexcept {
    if (context.inputs.size() != 2 || context.outputs.size() != 1 ||
        context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Plain W16 Embedding requires token ids, weight, and output");
    }

    const auto& tokens = context.inputs[0];
    const auto& weight = context.inputs[1];
    const auto& output = context.outputs[0];
    if (!tokens.is_valid() || !weight.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument("W16 Embedding requires valid tensor views");
    }

    if (!IsSupportedTokenIdDType(tokens.dtype()) || tokens.rank() < 1 ||
        !tokens.is_contiguous()) {
        return Status::InvalidArgument(
                "W16 Embedding requires contiguous int32, uint32, "
                "or int64 token ids with rank >= 1");
    }

    if (const DataType expected_weight = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
        weight.dtype() != expected_weight || weight.rank() != 2 ||
        !weight.is_contiguous()) {
        return Status::InvalidArgument(
                "W16 Embedding requires contiguous rank-2 "
                "weight matching the kernel dtype");
    }

    if (output.dtype() != DataType::Float32() ||
        output.rank() != tokens.rank() + 1 || !output.is_contiguous()) {
        return Status::InvalidArgument(
                "W16 Embedding requires contiguous float32 "
                "output with an appended hidden axis");
    }

    const int64_t vocab_size = weight.dim(0);
    const int64_t hidden_size = weight.dim(1);
    if (vocab_size <= 0 || hidden_size <= 0) {
        return Status::InvalidArgument(
                "W16 Embedding requires positive vocab and hidden sizes");
    }

    for (int32_t axis = 0; axis < tokens.rank(); ++axis) {
        if (output.dim(axis) != tokens.dim(axis)) {
            return Status::InvalidArgument(
                    "W16 Embedding output shape must be [token ids..., hidden_size]");
        }
    }

    if (output.dim(tokens.rank()) != hidden_size) {
        return Status::InvalidArgument(
                "W16 Embedding output shape must be [token ids..., hidden_size]");
    }

    const int64_t token_count = tokens.numel();
    if (token_count == 0) {
        ::new (params_buffer) EmbeddingF32W16KernelArgs{};
        return Status::Ok();
    }

    if (tokens.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "W16 Embedding requires non-null storage for non-empty tensors");
    }

    size_t count = 0;
    size_t vocab = 0;
    size_t hidden = 0;
    size_t weight_row_bytes = 0;
    size_t output_row_bytes = 0;
    if (CheckNarrow(token_count, &count) || CheckNarrow(vocab_size, &vocab) ||
        CheckNarrow(hidden_size, &hidden) ||
        CheckOverflowMul(hidden, sizeof(uint16_t), &weight_row_bytes) ||
        CheckOverflowMul(hidden, sizeof(float), &output_row_bytes)) {
        return Status::Overflow("W16 Embedding dimensions exceed host byte geometry");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange tokens_range,
                        BuildContiguousByteRange(tokens, "W16 Embedding token ids"));
    AM_ASSIGN_OR_RETURN(const ByteAddressRange weight_range,
                        BuildContiguousByteRange(weight, "W16 Embedding weight"));
    AM_ASSIGN_OR_RETURN(const ByteAddressRange output_range,
                        BuildContiguousByteRange(output, "W16 Embedding output"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU W16 Embedding", output_range,
            "output", tokens_range, "token ids"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU W16 Embedding", output_range,
            "output", weight_range, "weight"));

    // The complete tensor footprints prove all selected row offsets are in range.
    // The prepared conversion function has no borrowed backend or feature state.
    ::new (params_buffer) EmbeddingF32W16KernelArgs{
            .token_ids = static_cast<const std::byte*>(tokens.data()),
            .token_dtype = tokens.dtype(),
            .weight = static_cast<const std::byte*>(weight.data()),
            .output = static_cast<std::byte*>(output.data()),
            .token_count = count,
            .vocab_size = vocab,
            .hidden_size = hidden,
            .weight_row_bytes = weight_row_bytes,
            .output_row_bytes = output_row_bytes,
            .widen = Widen,
    };
    return Status::Ok();
}

template<bool Bfloat, EmbeddingRowWiden Widen = Bfloat ? ConvertBf16ToFp32Scalar : ConvertFp16ToFp32Scalar>
Status BuildEmbeddingF32W16PackedIdentityArgs(const KernelParamsBuildContext& context,
                                              void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed W16 Embedding requires token ids, output, and a weight artifact");
    }

    const TensorView& tokens = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!tokens.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "Packed W16 Embedding requires valid token id and output views");
    }

    if (!IsSupportedTokenIdDType(tokens.dtype()) || tokens.rank() < 1 ||
        !tokens.is_contiguous()) {
        return Status::InvalidArgument(
                "Packed W16 Embedding requires contiguous supported token ids with rank >= 1");
    }

    if (output.dtype() != DataType::Float32() ||
        output.rank() != tokens.rank() + 1 || !output.is_contiguous()) {
        return Status::InvalidArgument(
                "Packed W16 Embedding requires contiguous float32 "
                "output with an appended hidden axis");
    }

    if (packed.logical_shape.size() != 2U ||
        packed.logical_shape[0] <= 0 || packed.logical_shape[1] <= 0) {
        return Status::InvalidArgument(
                "Packed W16 Embedding requires positive rank-2 logical weight dimensions");
    }

    const int64_t vocab_size = packed.logical_shape[0];
    const int64_t hidden_size = packed.logical_shape[1];
    const DataType expected_weight = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            packed, std::array{vocab_size, hidden_size},
            "EmbeddingW16KernelEntry", expected_weight));
    for (int32_t axis = 0; axis < tokens.rank(); ++axis) {
        if (output.dim(axis) != tokens.dim(axis)) {
            return Status::InvalidArgument(
                    "Packed W16 Embedding output shape must be [token ids..., hidden_size]");
        }
    }

    if (output.dim(tokens.rank()) != hidden_size) {
        return Status::InvalidArgument(
                "Packed W16 Embedding output shape must be [token ids..., hidden_size]");
    }

    const int64_t token_count = tokens.numel();
    if (token_count == 0) {
        ::new (params_buffer) EmbeddingF32W16KernelArgs{};
        return Status::Ok();
    }

    if (tokens.data() == nullptr || output.data() == nullptr || packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed W16 Embedding requires non-null storage for non-empty tensors");
    }

    size_t count = 0;
    size_t vocab = 0;
    size_t hidden = 0;
    size_t weight_row_bytes = 0;
    size_t output_row_bytes = 0;
    if (CheckNarrow(token_count, &count) || CheckNarrow(vocab_size, &vocab) ||
        CheckNarrow(hidden_size, &hidden) ||
        CheckOverflowMul(hidden, sizeof(uint16_t), &weight_row_bytes) ||
        CheckOverflowMul(hidden, sizeof(float), &output_row_bytes)) {
        return Status::Overflow("Packed W16 Embedding dimensions exceed host byte geometry");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange tokens_range,
                        BuildContiguousByteRange(tokens, "Packed W16 Embedding token ids"));
    AM_ASSIGN_OR_RETURN(const ByteAddressRange output_range,
                        BuildContiguousByteRange(output, "Packed W16 Embedding output"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU W16 Embedding", output_range,
            "output", tokens_range, "token ids"));
    if (!std::in_range<int64_t>(packed.nbytes)) {
        return Status::Overflow("Packed W16 Embedding artifact byte range exceeds int64_t");
    }
    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(packed.data,
                                                 static_cast<int64_t>(packed.nbytes),
                                                 size_t{1}, "Packed W16 Embedding weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU W16 Embedding", output_range,
            "output", packed_range, "packed weight"));

    ::new (params_buffer) EmbeddingF32W16KernelArgs{
            .token_ids = static_cast<const std::byte*>(tokens.data()),
            .token_dtype = tokens.dtype(),
            .weight = static_cast<const std::byte*>(packed.data),
            .output = static_cast<std::byte*>(output.data()),
            .token_count = count,
            .vocab_size = vocab,
            .hidden_size = hidden,
            .weight_row_bytes = weight_row_bytes,
            .output_row_bytes = output_row_bytes,
            .widen = Widen,
    };
    return Status::Ok();
}

Status EmbeddingF32W16ReferenceEntry(const KernelContext& context) noexcept {
    const auto* args = static_cast<const EmbeddingF32W16KernelArgs*>(context.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunEmbeddingF32W16Reference(*args);
}

template<bool Bfloat, EmbeddingRowWiden Widen>
KernelDef VectorEmbedding(bool packed, CpuFeatureSet requirements, std::string_view name) {
    return {
            .op_type = OpType::kEmbedding,
            .selector = {.device_type = DeviceType::kCPU,
                         .act_dtype = DataType::Float32(),
                         .weight_dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16),
                         .weight_format = packed ? WeightFormat::kPacked : WeightFormat::kPlain,
                         .phase = ExecPhase::kBoth},
            .packing_recipe = packed ? CpuIdentityPackingRecipe() : PackingRecipe{},
            .cpu_requirements = requirements,
            .kernel_func = &EmbeddingF32W16ReferenceEntry,
            .priority = 20,
            .params_size = sizeof(EmbeddingF32W16KernelArgs),
            .params_builder = packed ? &BuildEmbeddingF32W16PackedIdentityArgs<Bfloat, Widen>
                                     : &BuildEmbeddingF32W16Args<Bfloat, Widen>,
            .metadata_builder = &BuildEmbeddingF32W16Metadata,
            .name = name,
    };
}

} // namespace

// The prepared args must satisfy the PreparedExecutionBindings params arena contract.
static_assert(std::is_trivially_destructible_v<EmbeddingF32KernelArgs>);
static_assert(alignof(EmbeddingF32KernelArgs) <= alignof(std::max_align_t));
static_assert(std::is_trivially_destructible_v<EmbeddingF32W16KernelArgs>);
static_assert(alignof(EmbeddingF32W16KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(EmbeddingF32W16KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(CpuEmbeddingF32Reference,
                   KernelDef{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &EmbeddingF32ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingF32KernelArgs),
                           .params_builder = &BuildEmbeddingF32ReferenceArgs,
                           .name = "cpu::embedding_f32_reference",
                   })

AM_REGISTER_KERNEL(CpuEmbeddingF32PackedIdentityReference,
                   KernelDef{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPacked,
                                   .phase = ExecPhase::kBoth,
                           },
                           .packing_recipe = CpuIdentityPackingRecipe(),
                           .kernel_func = &EmbeddingF32ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingF32KernelArgs),
                           .params_builder = &BuildEmbeddingF32PackedIdentityArgs,
                           .name = "cpu::embedding_f32_packed_identity_reference",
                   })

#if defined(FP16_HAS_F16C_CONVERSION)
AM_REGISTER_KERNEL(CpuEmbeddingF32F16F16c,
                   (VectorEmbedding<false, ConvertFp16ToFp32F16c>(
                           false, CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kF16c}),
                           "cpu::embedding_f32_f16_f16c")))
AM_REGISTER_KERNEL(CpuEmbeddingF32F16PackedF16c,
                   (VectorEmbedding<false, ConvertFp16ToFp32F16c>(
                           true, CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kF16c}),
                           "cpu::embedding_f32_f16_packed_identity_f16c")))
#endif
#if defined(BF16_HAS_AVX2_CONVERSION)
AM_REGISTER_KERNEL(CpuEmbeddingF32Bf16Avx2,
                   (VectorEmbedding<true, ConvertBf16ToFp32Avx2>(
                           false, CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kAvx2}),
                           "cpu::embedding_f32_bf16_avx2")))
AM_REGISTER_KERNEL(CpuEmbeddingF32Bf16PackedAvx2,
                   (VectorEmbedding<true, ConvertBf16ToFp32Avx2>(
                           true, CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kAvx2}),
                           "cpu::embedding_f32_bf16_packed_identity_avx2")))
#endif

AM_REGISTER_KERNEL(CpuEmbeddingF32F16Reference,
                   KernelDef{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float(16),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &EmbeddingF32W16ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingF32W16KernelArgs),
                           .params_builder = &BuildEmbeddingF32W16Args<false>,
                           .metadata_builder = &BuildEmbeddingF32W16Metadata,
                           .name = "cpu::embedding_f32_f16_reference",
                   })

AM_REGISTER_KERNEL(CpuEmbeddingF32Bf16Reference,
                   KernelDef{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::BFloat(16),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &EmbeddingF32W16ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingF32W16KernelArgs),
                           .params_builder = &BuildEmbeddingF32W16Args<true>,
                           .metadata_builder = &BuildEmbeddingF32W16Metadata,
                           .name = "cpu::embedding_f32_bf16_reference",
                   })

AM_REGISTER_KERNEL(CpuEmbeddingF32F16PackedIdentityReference,
                   KernelDef{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float(16),
                                   .weight_format = WeightFormat::kPacked,
                                   .phase = ExecPhase::kBoth,
                           },
                           .packing_recipe = CpuIdentityPackingRecipe(),
                           .kernel_func = &EmbeddingF32W16ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingF32W16KernelArgs),
                           .params_builder = &BuildEmbeddingF32W16PackedIdentityArgs<false>,
                           .metadata_builder = &BuildEmbeddingF32W16Metadata,
                           .name = "cpu::embedding_f32_f16_packed_identity_reference",
                   })

AM_REGISTER_KERNEL(CpuEmbeddingF32Bf16PackedIdentityReference,
                   KernelDef{
                           .op_type = OpType::kEmbedding,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::BFloat(16),
                                   .weight_format = WeightFormat::kPacked,
                                   .phase = ExecPhase::kBoth,
                           },
                           .packing_recipe = CpuIdentityPackingRecipe(),
                           .kernel_func = &EmbeddingF32W16ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(EmbeddingF32W16KernelArgs),
                           .params_builder = &BuildEmbeddingF32W16PackedIdentityArgs<true>,
                           .metadata_builder = &BuildEmbeddingF32W16Metadata,
                           .name = "cpu::embedding_f32_bf16_packed_identity_reference",
                   })

} // namespace aethermind::cpu::detail
