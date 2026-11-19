#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/packed_weight_validation.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/check.h"
#include "backend/cpu/kernels/gemm/gemm_a16_w16_internal.h"
#include "linear_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace aethermind::cpu::detail {
namespace {

Status BuildLinearF32Args(const KernelParamsBuildContext& context,
                          void* params_buffer) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Linear requires 2 inputs and 1 output");
    }

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid()) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires a valid input TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires a valid output MutableTensorView");
    }

    if (const DataType fp32 = DataType::Float32();
        input.dtype() != fp32 || weight.dtype() != fp32 || output.dtype() != fp32) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires float32 input, "
                "weight, and output TensorViews");
    }

    const int32_t rank = input.rank();
    if (rank < 1) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires input rank >= 1");
    }

    if (weight.rank() != 2) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires rank-2 weight TensorView");
    }

    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires output rank to match input rank");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "LinearKernelEntry requires output "
                    "leading dimensions to match input");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    const int64_t out_features = weight.dim(0);
    if (weight.dim(1) != in_features) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires weight input dimension "
                "to match input last dimension");
    }

    if (output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires output last dimension "
                "to match weight output dimension");
    }

    const auto row_count = ComputeFlattenedRowCount(
            input, "LinearKernelEntry");
    if (!row_count.ok()) {
        return row_count.status();
    }

    if (row_count.value() == 0 || out_features == 0) {
        ::new (params_buffer) LinearA32W32KernelArgs{
                .row_count = row_count.value(),
                .in_features = in_features,
                .out_features = out_features,
        };
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout("LinearKernelEntry", output_analysis));

    if (in_features == 0) {
        ::new (params_buffer) LinearA32W32KernelArgs{
                .output = output.data<float>(),
                .row_count = output_analysis.row_count(),
                .in_features = 0,
                .out_features = out_features,
                .output_row_stride = output_analysis.row_stride(),
                .output_col_stride = output_analysis.column_stride(),
        };
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires non-null "
                "data pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearKernelEntry input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis weight_analysis,
                        AnalyzeRowwiseView(weight, "LinearKernelEntry weight"));

    // Linear has no valid in-place form, not even when in_features equals
    // out_features: writing one output row can clobber input or weight elements
    // that later dot products have not read yet.
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU Linear",
            output_analysis.footprint(),
            "output",
            input_analysis.footprint(),
            "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU Linear",
            output_analysis.footprint(),
            "output",
            weight_analysis.footprint(),
            "weight"));

    ::new (params_buffer) LinearA32W32KernelArgs{
            .input = input.data<float>(),
            .weight = weight.data<float>(),
            .output = output.data<float>(),
            .row_count = input_analysis.row_count(),
            .in_features = in_features,
            .out_features = out_features,
            .input_row_stride = input_analysis.row_stride(),
            .input_col_stride = input_analysis.column_stride(),
            .weight_row_stride = weight_analysis.row_stride(),
            .weight_col_stride = weight_analysis.column_stride(),
            .output_row_stride = output_analysis.row_stride(),
            .output_col_stride = output_analysis.column_stride(),
    };
    return Status::Ok();
}

template<bool Bfloat>
Status BuildLinearF32W16Args(const KernelParamsBuildContext& context,
                             void* params_buffer) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Linear requires 2 inputs and 1 output");
    }

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid() || !weight.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "LinearF32W16 requires valid input, weight, and output TensorViews");
    }

    const DataType weight_dtype = weight.dtype();
    const DataType expected_weight = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    if (input.dtype() != DataType::Float32() || output.dtype() != DataType::Float32() ||
        weight_dtype != expected_weight) {
        return Status::InvalidArgument(
                "LinearF32W16 requires float32 input/output "
                "and weight dtype matching its selector");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || weight.rank() != 2 || output.rank() != rank) {
        return Status::InvalidArgument(
                "LinearF32W16 requires rank >= 1 input/output "
                "and rank-2 weight TensorViews");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "LinearF32W16 requires output leading dimensions to match input");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    const int64_t out_features = weight.dim(0);
    if (weight.dim(1) != in_features || output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "LinearF32W16 input, weight, and output "
                "feature dimensions do not match");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input,
                                                 "LinearF32W16 input"));
    if (row_count == 0 || out_features == 0) {
        ::new (params_buffer) LinearA32W16KernelArgs{
                .weight_dtype = weight_dtype,
                .row_count = row_count,
                .in_features = in_features,
                .out_features = out_features,
        };
        return Status::Ok();
    }

    // Unknown view alignment does not prove float alignment. Zero K still
    // writes output, but does not read input or the byte-addressed W16 weight.
    if (reinterpret_cast<std::uintptr_t>(output.data()) % alignof(float) != 0 ||
        (in_features != 0 && reinterpret_cast<std::uintptr_t>(input.data()) % alignof(float) != 0)) {
        return Status::InvalidArgument(
                "LinearF32W16 requires naturally aligned float32 input/output data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearF32W16 output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout("LinearF32W16", output_analysis));
    if (in_features == 0) {
        ::new (params_buffer) LinearA32W16KernelArgs{
                .output = output.data<float>(),
                .weight_dtype = weight_dtype,
                .row_count = output_analysis.row_count(),
                .in_features = 0,
                .out_features = out_features,
                .output_row_stride = output_analysis.row_stride(),
                .output_col_stride = output_analysis.column_stride(),
        };
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "LinearF32W16 requires non-null data "
                "pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearF32W16 input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis weight_analysis,
                        AnalyzeRowwiseView(weight, "LinearF32W16 weight"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU LinearF32W16", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU LinearF32W16", output_analysis.footprint(), "output",
            weight_analysis.footprint(), "weight"));

    ::new (params_buffer) LinearA32W16KernelArgs{
            .input = input.data<float>(),
            .weight = static_cast<const std::byte*>(weight.data()),
            .output = output.data<float>(),
            .weight_dtype = weight_dtype,
            .row_count = input_analysis.row_count(),
            .in_features = in_features,
            .out_features = out_features,
            .input_row_stride = input_analysis.row_stride(),
            .input_col_stride = input_analysis.column_stride(),
            .weight_row_stride = weight_analysis.row_stride(),
            .weight_col_stride = weight_analysis.column_stride(),
            .output_row_stride = output_analysis.row_stride(),
            .output_col_stride = output_analysis.column_stride(),
    };
    return Status::Ok();
}

template<bool Bfloat>
Status BuildLinearF32W16PackedIdentityArgs(const KernelParamsBuildContext& context,
                                           void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires one activation, one output, and an artifact");
    }

    const TensorView& input = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!input.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires valid input and output TensorViews");
    }

    if (input.dtype() != DataType::Float32() || output.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires float32 input and output TensorViews");
    }

    if (input.rank() < 1 || packed.logical_shape.size() != 2U) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires rank >= 1 input and a rank-2 weight artifact");
    }

    const DataType weight_dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    const int64_t in_features = input.dim(input.rank() - 1);
    const int64_t out_features = packed.logical_shape[0];
    if (out_features < 0 || packed.logical_shape[1] != in_features) {
        return Status::InvalidArgument(
                "Packed W16 Linear weight shape does not match the input feature dimension");
    }

    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            packed, std::array{out_features, in_features},
            "LinearW16KernelEntry", weight_dtype));

    const int32_t rank = input.rank();
    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires output rank to match input rank");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "Packed W16 Linear requires output leading dimensions to match input");
        }
    }

    if (output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "Packed W16 Linear output feature dimension does not match the weight");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "LinearW16KernelEntry"));
    LinearA32W16KernelArgs args{
            .weight_dtype = weight_dtype,
            .row_count = row_count,
            .in_features = in_features,
            .out_features = out_features,
    };

    if (row_count == 0 || out_features == 0) {
        ::new (params_buffer) LinearA32W16KernelArgs(args);
        return Status::Ok();
    }

    if (reinterpret_cast<std::uintptr_t>(output.data()) % alignof(float) != 0 ||
        (in_features != 0 && reinterpret_cast<std::uintptr_t>(input.data()) % alignof(float) != 0)) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires naturally aligned float32 input/output data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearW16KernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "LinearW16KernelEntry output", output_analysis));
    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires non-null output for non-empty tensors");
    }

    args.output = output.data<float>();
    args.output_row_stride = output_analysis.row_stride();
    args.output_col_stride = output_analysis.column_stride();
    if (in_features == 0) {
        ::new (params_buffer) LinearA32W16KernelArgs(args);
        return Status::Ok();
    }

    if (input.data() == nullptr || packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed W16 Linear requires non-null input and packed weight data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearW16KernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU LinearF32W16", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    if (!std::in_range<int64_t>(packed.nbytes)) {
        return Status::Overflow(
                "Packed W16 Linear artifact byte range exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                packed.data, static_cast<int64_t>(packed.nbytes),
                                size_t{1}, "LinearW16KernelEntry packed weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU LinearF32W16", output_analysis.footprint().envelope(), "output",
            packed_range, "packed weight"));

    args.input = input.data<float>();
    args.weight = static_cast<const std::byte*>(packed.data);
    args.row_count = input_analysis.row_count();
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    args.weight_row_stride = in_features;
    args.weight_col_stride = 1;
    ::new (params_buffer) LinearA32W16KernelArgs(args);
    return Status::Ok();
}

template<bool Bfloat>
Status BuildLinearA16Args(const KernelParamsBuildContext& context,
                          void* params_buffer) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("A16 Linear requires 2 inputs and 1 output");
    }

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid() || !weight.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "A16 Linear requires valid input, weight, and output TensorViews");
    }

    const DataType act_dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    if (input.dtype() != act_dtype || weight.dtype() != act_dtype) {
        return Status::InvalidArgument(
                "A16 Linear requires activation and weight dtype matching its selector");
    }

    const DataType output_dtype = output.dtype();
    if (output_dtype != act_dtype && output_dtype != DataType::Float32()) {
        return Status::InvalidArgument(
                "A16 Linear output must be the activation dtype or float32");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || weight.rank() != 2 || output.rank() != rank) {
        return Status::InvalidArgument(
                "A16 Linear requires rank >= 1 input/output and rank-2 weight");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "A16 Linear requires output leading dimensions to match input");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    const int64_t out_features = weight.dim(0);
    if (weight.dim(1) != in_features || output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "A16 Linear input, weight, and output feature dimensions do not match");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "A16 Linear input"));
    GemmA16W16Args args{
            .gemm = {.m = row_count, .n = out_features, .k = in_features},
            .operand_dtype = act_dtype,
            .output_dtype = output_dtype,
    };
    GemmArgs& gemm = args.gemm;
    if (row_count == 0 || out_features == 0) {
        ::new (params_buffer) GemmA16W16Args(args);
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "A16 Linear output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout("A16 Linear output", output_analysis));
    gemm.output = static_cast<std::byte*>(output.data());
    gemm.output_m_stride = output_analysis.row_stride();
    gemm.output_n_stride = output_analysis.column_stride();
    if (in_features == 0) {
        gemm.m = output_analysis.row_count();
        ::new (params_buffer) GemmA16W16Args(args);
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "A16 Linear requires non-null data pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "A16 Linear input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis weight_analysis,
                        AnalyzeRowwiseView(weight, "A16 Linear weight"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU A16 Linear", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU A16 Linear", output_analysis.footprint(), "output",
            weight_analysis.footprint(), "weight"));

    gemm.lhs = static_cast<const std::byte*>(input.data());
    gemm.rhs = static_cast<const std::byte*>(weight.data());
    gemm.m = input_analysis.row_count();
    gemm.lhs_m_stride = input_analysis.row_stride();
    gemm.lhs_k_stride = input_analysis.column_stride();
    gemm.rhs_k_stride = weight_analysis.column_stride();
    gemm.rhs_n_stride = weight_analysis.row_stride();
    ::new (params_buffer) GemmA16W16Args(args);
    return Status::Ok();
}

template<bool Bfloat>
Status BuildLinearA16PackedIdentityArgs(const KernelParamsBuildContext& context,
                                        void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires one activation, one output, and an artifact");
    }

    const TensorView& input = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!input.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires valid input and output TensorViews");
    }

    const DataType act_dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    if (input.dtype() != act_dtype) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires the activation dtype of its selector");
    }

    const DataType output_dtype = output.dtype();
    if (output_dtype != act_dtype && output_dtype != DataType::Float32()) {
        return Status::InvalidArgument(
                "Packed A16 Linear output must be the activation dtype or float32");
    }

    if (input.rank() < 1 || packed.logical_shape.size() != 2U) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires rank >= 1 input and a rank-2 weight artifact");
    }

    const int64_t in_features = input.dim(input.rank() - 1);
    const int64_t out_features = packed.logical_shape[0];
    if (out_features < 0 || packed.logical_shape[1] != in_features) {
        return Status::InvalidArgument(
                "Packed A16 Linear weight shape does not match the input feature dimension");
    }

    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            packed, std::array{out_features, in_features},
            "LinearA16KernelEntry", act_dtype));

    const int32_t rank = input.rank();
    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires output rank to match input rank");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "Packed A16 Linear requires output leading dimensions to match input");
        }
    }

    if (output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "Packed A16 Linear output feature dimension does not match the weight");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "LinearA16KernelEntry"));
    GemmA16W16Args args{
            .gemm = {.m = row_count, .n = out_features, .k = in_features},
            .operand_dtype = act_dtype,
            .output_dtype = output_dtype,
    };
    GemmArgs& gemm = args.gemm;
    if (row_count == 0 || out_features == 0) {
        ::new (params_buffer) GemmA16W16Args(args);
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearA16KernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "LinearA16KernelEntry output", output_analysis));
    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires non-null output for non-empty tensors");
    }

    gemm.output = static_cast<std::byte*>(output.data());
    gemm.output_m_stride = output_analysis.row_stride();
    gemm.output_n_stride = output_analysis.column_stride();

    // Zero K only zero-fills the output and reads neither input nor weight, so
    // their empty buffers may legally carry null data pointers.
    if (in_features == 0) {
        ::new (params_buffer) GemmA16W16Args(args);
        return Status::Ok();
    }

    if (input.data() == nullptr || packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed A16 Linear requires non-null input and packed weight data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearA16KernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU A16 Linear", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    if (!std::in_range<int64_t>(packed.nbytes)) {
        return Status::Overflow(
                "Packed A16 Linear artifact byte range exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                packed.data, static_cast<int64_t>(packed.nbytes),
                                size_t{1}, "LinearA16KernelEntry packed weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU A16 Linear", output_analysis.footprint().envelope(), "output",
            packed_range, "packed weight"));

    gemm.lhs = static_cast<const std::byte*>(input.data());
    gemm.rhs = static_cast<const std::byte*>(packed.data);
    gemm.m = input_analysis.row_count();
    gemm.lhs_m_stride = input_analysis.row_stride();
    gemm.lhs_k_stride = input_analysis.column_stride();
    gemm.rhs_k_stride = 1;
    gemm.rhs_n_stride = in_features;
    ::new (params_buffer) GemmA16W16Args(args);
    return Status::Ok();
}

StatusOr<LinearA32W32KernelArgs> BuildLinearF32PackedBase(
        const KernelParamsBuildContext& context,
        bool bpanel) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed Linear requires one activation, one output, and an artifact");
    }

    const TensorView& input = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!input.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "Packed Linear requires valid input and output TensorViews");
    }

    if (input.dtype() != DataType::Float32() ||
        output.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "Packed Linear requires float32 input and output TensorViews");
    }

    if (input.rank() < 1 || packed.logical_shape.size() != 2U) {
        return Status::InvalidArgument(
                "Packed Linear requires rank >= 1 input and a rank-2 weight artifact");
    }

    const int64_t in_features = input.dim(input.rank() - 1);
    const int64_t out_features = packed.logical_shape[0];
    if (out_features < 0 || packed.logical_shape[1] != in_features) {
        return Status::InvalidArgument(
                "Packed Linear weight shape does not match the input feature dimension");
    }

    const std::array<int64_t, 2> expected_shape{out_features, in_features};
    if (bpanel) {
        AM_RETURN_IF_ERROR(ValidateBPanelF32PackedWeight(
                packed, expected_shape, "LinearKernelEntry"));
    } else {
        AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
                packed, expected_shape, "LinearKernelEntry"));
    }

    const int32_t rank = input.rank();
    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "Packed Linear requires output rank to match input rank");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "Packed Linear requires output leading dimensions to match input");
        }
    }

    if (output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "Packed Linear output feature dimension does not match the weight");
    }
    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "LinearKernelEntry"));

    LinearA32W32KernelArgs args{
            .row_count = row_count,
            .in_features = in_features,
            .out_features = out_features,
    };

    if (row_count == 0 || out_features == 0) {
        return args;
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "LinearKernelEntry output", output_analysis));
    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                "Packed Linear requires non-null output for non-empty tensors");
    }

    args.output = output.data<float>();
    args.output_row_stride = output_analysis.row_stride();
    args.output_col_stride = output_analysis.column_stride();

    if (in_features == 0) {
        return args;
    }

    if (input.data() == nullptr || packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed Linear requires non-null input and packed weight data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearKernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU Linear", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    if (!std::in_range<int64_t>(packed.nbytes)) {
        return Status::Overflow(
                "Packed Linear artifact byte range exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                packed.data, static_cast<int64_t>(packed.nbytes),
                                size_t{1}, "LinearKernelEntry packed weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU Linear", output_analysis.footprint().envelope(),
            "output", packed_range, "packed weight"));

    args.input = input.data<float>();
    args.weight = static_cast<const float*>(packed.data);
    args.row_count = input_analysis.row_count();
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    args.weight_row_stride = in_features;
    args.weight_col_stride = 1;
    return args;
}

Status BuildLinearF32PackedIdentityArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const LinearA32W32KernelArgs args,
                        BuildLinearF32PackedBase(context, false));
    ::new (params_buffer) LinearA32W32KernelArgs(args);
    return Status::Ok();
}

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status BuildLinearF32BpanelArgs(const KernelParamsBuildContext& context,
                                void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const LinearA32W32KernelArgs base,
                        BuildLinearF32PackedBase(context, true));
    const int64_t n_blocks = base.out_features /
                                     kCpuBPanelF32Kc512Nr16NR +
                             (base.out_features % kCpuBPanelF32Kc512Nr16NR != 0);
    const PackedGemmF32Args args{
            .gemm = GemmArgs{
                    .lhs = reinterpret_cast<const std::byte*>(base.input),
                    .output = reinterpret_cast<std::byte*>(base.output),
                    .m = base.row_count,
                    .n = base.out_features,
                    .k = base.in_features,
                    .lhs_m_stride = base.input_row_stride,
                    .lhs_k_stride = base.input_col_stride,
                    .output_m_stride = base.output_row_stride,
                    .output_n_stride = base.output_col_stride,
            },
            .packed_b = reinterpret_cast<const std::byte*>(base.weight),
            .packed_nbytes = context.packed_weight->nbytes,
            .logical_n = base.out_features,
            .logical_k = base.in_features,
            .weight_n_offset = 0,
            .n_blocks = n_blocks,
    };
    ::new (params_buffer) PackedGemmF32Args(args);
    return Status::Ok();
}

Status LinearF32PackedBEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const PackedGemmF32Args*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmA32W32PackedB(*args);
}
#endif

template<bool Packed, bool Native>
Status BuildLinearF32OptimizedArgs(const KernelParamsBuildContext& context, void* params_buffer) noexcept {
    LinearA32W32KernelArgs bound;
    if constexpr (Packed) {
        AM_RETURN_IF_ERROR(BuildLinearF32PackedIdentityArgs(context, &bound));
    } else {
        AM_RETURN_IF_ERROR(BuildLinearF32Args(context, &bound));
    }
    auto driver = &RunLinearF32Scalar;
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
    if constexpr (Native) {
        if (bound.weight_col_stride == 1) {
            driver = &RunLinearF32Avx2Fma;
        }
    }
#endif
    // No K-vector body exists here; prepared A/B evidence showed a material
    // small-M regression. Freeze the compatible oracle rather than dispatching
    // through the SIMD driver on every execution. Empty outputs also avoid all
    // input pointer arithmetic, including the native driver's row setup.
    if (bound.row_count == 0 || bound.out_features == 0 || bound.in_features == 0 ||
        (bound.in_features < 8 && (bound.row_count <= 8 || bound.out_features < 16)) ||
        bound.input_col_stride != 1 || bound.output_col_stride != 1 ||
        (bound.weight_col_stride != 1 && bound.weight_row_stride != 1)) {
        driver = &RunLinearF32Reference;
    }
    ::new (params_buffer) LinearA32W32PreparedArgs{bound, driver};
    return Status::Ok();
}

Status LinearF32ScalarEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearA32W32PreparedArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr && args->driver != nullptr);
    return args->driver(args->linear);
}

Status LinearF32W16ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearA32W16KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunLinearF32W16Reference(*args);
}

Status LinearA16ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GemmA16W16Args*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmA16W16Reference(*args);
}

#if defined(A16_HAS_AVX2_FMA_KERNEL)
Status LinearA16Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GemmA16W16Args*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmA16W16Avx2Fma(*args);
}
#endif

#if defined(A16_HAS_AVX512BF16_KERNEL)
Status LinearA16Bf16Avx512Bf16Entry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GemmA16W16Args*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmABf16WBf16Avx512Bf16(*args);
}
#endif

#if defined(A16_HAS_AVX512FP16_KERNEL)
Status LinearA16F16Avx512Fp16Entry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GemmA16W16Args*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmAF16WF16Avx512Fp16(*args);
}
#endif

} // namespace

static_assert(std::is_trivially_destructible_v<LinearA32W32KernelArgs>);
static_assert(std::is_trivially_copyable_v<LinearA32W32PreparedArgs>);
static_assert(sizeof(LinearA32W32PreparedArgs) <= kMaxKernelParamsSize);
static_assert(alignof(LinearA32W32PreparedArgs) <= alignof(std::max_align_t));
static_assert(alignof(LinearA32W32KernelArgs) <= alignof(std::max_align_t));
static_assert(std::is_trivially_destructible_v<LinearA32W16KernelArgs>);
static_assert(alignof(LinearA32W16KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(LinearA32W16KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(
        CpuLinearF32Scalar,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearF32ScalarEntry,
                .priority = 10,
                .params_size = sizeof(LinearA32W32PreparedArgs),
                .params_builder = &BuildLinearF32OptimizedArgs<false, false>,
                .name = "cpu::linear_f32_w32_scalar"})

AM_REGISTER_KERNEL(
        CpuLinearF32F16Reference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearF32W16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(LinearA32W16KernelArgs),
                .params_builder = &BuildLinearF32W16Args<false>,
                .name = "cpu::linear_f32_wf16_reference"})

AM_REGISTER_KERNEL(
        CpuLinearF32Bf16Reference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearF32W16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(LinearA32W16KernelArgs),
                .params_builder = &BuildLinearF32W16Args<true>,
                .name = "cpu::linear_f32_wbf16_reference"})

AM_REGISTER_KERNEL(
        CpuLinearF32F16PackedIdentityReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearF32W16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(LinearA32W16KernelArgs),
                .params_builder = &BuildLinearF32W16PackedIdentityArgs<false>,
                .name = "cpu::linear_f32_wf16_packed_identity_reference"})

AM_REGISTER_KERNEL(
        CpuLinearF32Bf16PackedIdentityReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearF32W16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(LinearA32W16KernelArgs),
                .params_builder = &BuildLinearF32W16PackedIdentityArgs<true>,
                .name = "cpu::linear_f32_wbf16_packed_identity_reference"})


AM_REGISTER_KERNEL(
        CpuLinearF32PackedIdentityScalar,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearF32ScalarEntry,
                .priority = 10,
                .params_size = sizeof(LinearA32W32PreparedArgs),
                .params_builder = &BuildLinearF32OptimizedArgs<true, false>,
                .name = "cpu::linear_f32_w32_packed_identity_scalar"})

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
static_assert(std::is_trivially_destructible_v<PackedGemmF32Args>);
static_assert(sizeof(PackedGemmF32Args) <= kMaxKernelParamsSize);
static_assert(alignof(PackedGemmF32Args) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuLinearF32PackedBpanelCandidate,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = cpu::CpuBPanelF32Kc512Nr16Recipe(),
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &LinearF32PackedBEntry,
                .priority = 10,
                .params_size = sizeof(PackedGemmF32Args),
                .params_builder = &BuildLinearF32BpanelArgs,
                .name = "cpu::linear_f32_w32_packed_bpanel_candidate"})
#endif

static_assert(std::is_trivially_destructible_v<GemmA16W16Args>);
static_assert(alignof(GemmA16W16Args) <= alignof(std::max_align_t));
static_assert(sizeof(GemmA16W16Args) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(
        CpuLinearA16F16Reference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float(16),
                        .output_dtype = DataType::Float(16),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16Args<false>,
                .name = "cpu::linear_f16_wf16_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Reference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::BFloat(16),
                        .output_dtype = DataType::BFloat(16),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16Args<true>,
                .name = "cpu::linear_bf16_wbf16_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16F16Fp32OutputReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float(16),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16Args<false>,
                .name = "cpu::linear_f16_wf16_out32_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Fp32OutputReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::BFloat(16),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16Args<true>,
                .name = "cpu::linear_bf16_wbf16_out32_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16F16PackedIdentityReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float(16),
                        .output_dtype = DataType::Float(16),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16PackedIdentityArgs<false>,
                .name = "cpu::linear_f16_wf16_packed_identity_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16Bf16PackedIdentityReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::BFloat(16),
                        .output_dtype = DataType::BFloat(16),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16PackedIdentityArgs<true>,
                .name = "cpu::linear_bf16_wbf16_packed_identity_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16F16Fp32OutputPackedIdentityReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float(16),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16PackedIdentityArgs<false>,
                .name = "cpu::linear_f16_wf16_out32_packed_identity_reference"})

AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Fp32OutputPackedIdentityReference,
        KernelDef{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::BFloat(16),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearA16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GemmA16W16Args),
                .params_builder = &BuildLinearA16PackedIdentityArgs<true>,
                .name = "cpu::linear_bf16_wbf16_out32_packed_identity_reference"})

// --- A16 ISA kernels ---
// The reference stays registered at priority 10. AVX2/FMA widening fallback and
// the AVX-512 native kernels are added at higher priority with explicit CPU
// requirements, so CpuBackend selects them only when the effective feature set
// contains every required feature. On hardware without AVX-512 the native
// descriptors are never eligible.
namespace {

template<bool Bfloat, bool Packed, bool Fp32Output>
KernelDef A16LinearDef(Status (*entry)(const KernelContext&) noexcept,
                       CpuFeatureSet requirements,
                       int priority,
                       std::string_view name) {
    const DataType act = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    return KernelDef{
            .op_type = OpType::kLinear,
            .selector = KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .act_dtype = act,
                    .output_dtype = Fp32Output ? DataType::Float32() : act,
                    .weight_dtype = act,
                    .weight_format = Packed ? WeightFormat::kPacked : WeightFormat::kPlain,
                    .phase = ExecPhase::kBoth,
            },
            .packing_recipe = Packed ? CpuIdentityPackingRecipe() : PackingRecipe{},
            .cpu_requirements = requirements,
            .kernel_func = entry,
            .priority = priority,
            .params_size = sizeof(GemmA16W16Args),
            .params_builder = Packed ? &BuildLinearA16PackedIdentityArgs<Bfloat> : &BuildLinearA16Args<Bfloat>,
            .name = name,
    };
}

} // namespace

#if defined(A16_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearA16F16Avx2Fma,
        (A16LinearDef<false, false, false>(&LinearA16Avx2FmaEntry,
                                           CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                           20, "cpu::linear_f16_wf16_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Avx2Fma,
        (A16LinearDef<true, false, false>(&LinearA16Avx2FmaEntry,
                                          CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                          20, "cpu::linear_bf16_wbf16_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16F16Fp32OutputAvx2Fma,
        (A16LinearDef<false, false, true>(&LinearA16Avx2FmaEntry,
                                          CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                          20, "cpu::linear_f16_wf16_out32_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Fp32OutputAvx2Fma,
        (A16LinearDef<true, false, true>(&LinearA16Avx2FmaEntry,
                                         CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                         20, "cpu::linear_bf16_wbf16_out32_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16F16PackedIdentityAvx2Fma,
        (A16LinearDef<false, true, false>(&LinearA16Avx2FmaEntry,
                                          CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                          20, "cpu::linear_f16_wf16_packed_identity_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16PackedIdentityAvx2Fma,
        (A16LinearDef<true, true, false>(&LinearA16Avx2FmaEntry,
                                         CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                         20, "cpu::linear_bf16_wbf16_packed_identity_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16F16Fp32OutputPackedIdentityAvx2Fma,
        (A16LinearDef<false, true, true>(&LinearA16Avx2FmaEntry,
                                         CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                         20,
                                         "cpu::linear_f16_wf16_out32_packed_identity_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Fp32OutputPackedIdentityAvx2Fma,
        (A16LinearDef<true, true, true>(&LinearA16Avx2FmaEntry,
                                        CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                        20,
                                        "cpu::linear_bf16_wbf16_out32_packed_identity_avx2_fma")))
#endif

#if defined(A16_HAS_AVX512BF16_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Avx512Bf16,
        (A16LinearDef<true, false, false>(
                &LinearA16Bf16Avx512Bf16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Bf16}),
                30, "cpu::linear_bf16_wbf16_avx512_bf16")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Fp32OutputAvx512Bf16,
        (A16LinearDef<true, false, true>(
                &LinearA16Bf16Avx512Bf16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Bf16}),
                30, "cpu::linear_bf16_wbf16_out32_avx512_bf16")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16PackedIdentityAvx512Bf16,
        (A16LinearDef<true, true, false>(
                &LinearA16Bf16Avx512Bf16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Bf16}),
                30, "cpu::linear_bf16_wbf16_packed_identity_avx512_bf16")))
AM_REGISTER_KERNEL(
        CpuLinearA16Bf16Fp32OutputPackedIdentityAvx512Bf16,
        (A16LinearDef<true, true, true>(
                &LinearA16Bf16Avx512Bf16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Bf16}),
                30, "cpu::linear_bf16_wbf16_out32_packed_identity_avx512_bf16")))
#endif

#if defined(A16_HAS_AVX512FP16_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearA16F16Avx512Fp16,
        (A16LinearDef<false, false, false>(
                &LinearA16F16Avx512Fp16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Fp16}),
                30, "cpu::linear_f16_wf16_avx512_fp16")))
AM_REGISTER_KERNEL(
        CpuLinearA16F16Fp32OutputAvx512Fp16,
        (A16LinearDef<false, false, true>(
                &LinearA16F16Avx512Fp16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Fp16}),
                30, "cpu::linear_f16_wf16_out32_avx512_fp16")))
AM_REGISTER_KERNEL(
        CpuLinearA16F16PackedIdentityAvx512Fp16,
        (A16LinearDef<false, true, false>(
                &LinearA16F16Avx512Fp16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Fp16}),
                30, "cpu::linear_f16_wf16_packed_identity_avx512_fp16")))
AM_REGISTER_KERNEL(
        CpuLinearA16F16Fp32OutputPackedIdentityAvx512Fp16,
        (A16LinearDef<false, true, true>(
                &LinearA16F16Avx512Fp16Entry,
                CpuFeatureSet::From({CpuFeature::kAvx512F, CpuFeature::kAvx512Fp16}),
                30, "cpu::linear_f16_wf16_out32_packed_identity_avx512_fp16")))
#endif


namespace {

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status LinearF32Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearA32W32PreparedArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr && args->driver != nullptr);
    return args->driver(args->linear);
}
#endif
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status LinearF32Bf16Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearA32W16KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunLinearF32Bf16Avx2Fma(*args);
}
#endif
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
Status LinearF32F16Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearA32W16KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunLinearF32F16Avx2Fma(*args);
}
#endif

template<bool Reduced, bool Packed>
KernelDef LinearOptimizedDef(DataType weight_dtype, KernelFunc entry,
                             KernelParamsBuilder builder, CpuFeatureSet requirements,
                             std::string_view name) {
    return KernelDef{
            .op_type = OpType::kLinear,
            .selector = KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .output_dtype = DataType::Float32(),
                    .weight_dtype = weight_dtype,
                    .weight_format = Packed ? WeightFormat::kPacked : WeightFormat::kPlain,
                    .phase = ExecPhase::kBoth,
            },
            .packing_recipe = Packed ? CpuIdentityPackingRecipe() : PackingRecipe{},
            .cpu_requirements = requirements,
            .kernel_func = entry,
            .priority = 20,
            .params_size = Reduced ? sizeof(LinearA32W16KernelArgs) : sizeof(LinearA32W32PreparedArgs),
            .params_builder = builder,
            .name = name,
    };
}

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearF32Avx2Optimized,
        (LinearOptimizedDef<false, false>(DataType::Float32(), &LinearF32Avx2FmaEntry,
                                          &BuildLinearF32OptimizedArgs<false, true>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                          "cpu::linear_f32_w32_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearF32PackedIdentityAvx2Optimized,
        (LinearOptimizedDef<false, true>(DataType::Float32(), &LinearF32Avx2FmaEntry,
                                         &BuildLinearF32OptimizedArgs<true, true>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                         "cpu::linear_f32_w32_packed_identity_avx2_fma")))
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearBf16Avx2Optimized,
        (LinearOptimizedDef<true, false>(DataType::BFloat(16), &LinearF32Bf16Avx2FmaEntry,
                                         &BuildLinearF32W16Args<true>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                         "cpu::linear_f32_wbf16_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearBf16PackedIdentityAvx2Optimized,
        (LinearOptimizedDef<true, true>(DataType::BFloat(16), &LinearF32Bf16Avx2FmaEntry,
                                        &BuildLinearF32W16PackedIdentityArgs<true>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                        "cpu::linear_f32_wbf16_packed_identity_avx2_fma")))
#endif

#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearF16Avx2Optimized,
        (LinearOptimizedDef<true, false>(DataType::Float(16), &LinearF32F16Avx2FmaEntry,
                                         &BuildLinearF32W16Args<false>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                         "cpu::linear_f32_wf16_avx2_fma")))
AM_REGISTER_KERNEL(
        CpuLinearF16PackedIdentityAvx2Optimized,
        (LinearOptimizedDef<true, true>(DataType::Float(16), &LinearF32F16Avx2FmaEntry,
                                        &BuildLinearF32W16PackedIdentityArgs<false>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                        "cpu::linear_f32_wf16_packed_identity_avx2_fma")))
#endif

} // namespace aethermind::cpu::detail
