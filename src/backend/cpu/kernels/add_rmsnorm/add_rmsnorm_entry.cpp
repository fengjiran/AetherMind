#include "add_rmsnorm_internal.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

StatusOr<float> ReadEpsilon(std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(float)) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }

    float eps = 0.0F;
    std::memcpy(&eps, attrs.data(), sizeof(eps));
    if (!std::isfinite(eps) || eps <= 0.0F) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires finite positive epsilon");
    }
    return eps;
}

StatusOr<int64_t> ValidateActivationViews(
        const TensorView& input,
        const TensorView& residual,
        const MutableTensorView& output,
        const MutableTensorView& new_residual) noexcept {
    if (!input.is_valid() || !residual.is_valid() || !output.is_valid() ||
        !new_residual.is_valid()) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires valid input and output TensorViews");
    }

    if (input.dtype() != DataType::Float32() ||
        residual.dtype() != DataType::Float32() ||
        output.dtype() != DataType::Float32() ||
        new_residual.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires float32 input, residual, and outputs");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || residual.rank() != rank || output.rank() != rank ||
        new_residual.rank() != rank) {
        return Status::InvalidArgument("AddRmsNormKernelEntry requires input,"
                                       " residual, and outputs with matching rank >= 1");
    }

    for (int32_t dimension = 0; dimension < rank; ++dimension) {
        if (residual.dim(dimension) != input.dim(dimension) ||
            output.dim(dimension) != input.dim(dimension) ||
            new_residual.dim(dimension) != input.dim(dimension)) {
            return Status::InvalidArgument("AddRmsNormKernelEntry requires "
                                           "residual and outputs to match input shape");
        }
    }

    const int64_t hidden_size = input.dim(rank - 1);
    if (hidden_size <= 0) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires positive hidden_size");
    }
    return hidden_size;
}

StatusOr<AddRmsNormF32KernelArgs> BuildNonEmptyArgs(const TensorView& input,
                                                    const TensorView& residual,
                                                    const MutableTensorView& output,
                                                    const MutableTensorView& new_residual,
                                                    const float* weight,
                                                    int64_t weight_stride,
                                                    const RowwiseViewAnalysis* plain_weight_analysis,
                                                    const PackedWeightView* packed_weight,
                                                    int64_t hidden_size,
                                                    float eps) noexcept {
    if (input.data() == nullptr || residual.data() == nullptr || weight == nullptr ||
        output.data() == nullptr || new_residual.data() == nullptr) {
        return Status::InvalidArgument("AddRmsNormKernelEntry requires "
                                       "non-null data pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "AddRmsNormKernelEntry input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis residual_analysis,
                        AnalyzeRowwiseView(residual, "AddRmsNormKernelEntry residual"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "AddRmsNormKernelEntry output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis new_residual_analysis,
                        AnalyzeRowwiseView(new_residual,
                                           "AddRmsNormKernelEntry new_residual"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "AddRmsNormKernelEntry output", output_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "AddRmsNormKernelEntry new_residual", new_residual_analysis));

    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU AddRmsNorm", output_analysis.footprint(), "output",
            new_residual_analysis.footprint(), "new_residual"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU AddRmsNorm", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU AddRmsNorm", output_analysis.footprint(), "output",
            residual_analysis.footprint(), "residual"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU AddRmsNorm", new_residual_analysis.footprint(), "new_residual",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU AddRmsNorm", new_residual_analysis.footprint(), "new_residual",
            residual_analysis.footprint(), "residual"));

    if (plain_weight_analysis != nullptr) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
                "CPU AddRmsNorm", output_analysis.footprint(), "output",
                plain_weight_analysis->footprint(), "weight"));
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
                "CPU AddRmsNorm", new_residual_analysis.footprint(), "new_residual",
                plain_weight_analysis->footprint(), "weight"));
    } else {
        AM_DCHECK(packed_weight != nullptr);
        AM_ASSIGN_OR_RETURN(
                const ByteAddressRange packed_range,
                BuildContiguousByteRange(
                        packed_weight->data,
                        static_cast<int64_t>(packed_weight->nbytes), size_t{1},
                        "AddRmsNormKernelEntry packed weight"));
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU AddRmsNorm", output_analysis.footprint(), "output",
                packed_range, "packed weight"));
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU AddRmsNorm", new_residual_analysis.footprint(), "new_residual",
                packed_range, "packed weight"));
    }

    return AddRmsNormF32KernelArgs{
            .input = input.data<float>(),
            .residual = residual.data<float>(),
            .weight = weight,
            .output = output.data<float>(),
            .new_residual = new_residual.data<float>(),
            .row_count = input_analysis.row_count(),
            .hidden_size = hidden_size,
            .input_row_stride = input_analysis.row_stride(),
            .input_col_stride = input_analysis.column_stride(),
            .residual_row_stride = residual_analysis.row_stride(),
            .residual_col_stride = residual_analysis.column_stride(),
            .weight_stride = weight_stride,
            .output_row_stride = output_analysis.row_stride(),
            .output_col_stride = output_analysis.column_stride(),
            .new_residual_row_stride = new_residual_analysis.row_stride(),
            .new_residual_col_stride = new_residual_analysis.column_stride(),
            .eps = eps,
    };
}

Status BuildAddRmsNormF32PlainArgs(const KernelParamsBuildContext& context,
                                   void* params_buffer) noexcept {
    if (context.inputs.size() != 3U || context.outputs.size() != 2U) {
        return Status::InvalidArgument(
                "AddRmsNorm requires input, residual, weight and two outputs");
    }

    AM_ASSIGN_OR_RETURN(const float eps, ReadEpsilon(context.attrs));
    const TensorView& input = context.inputs[0];
    const TensorView& residual = context.inputs[1];
    const TensorView& weight = context.inputs[2];
    const MutableTensorView& output = context.outputs[0];
    const MutableTensorView& new_residual = context.outputs[1];
    AM_ASSIGN_OR_RETURN(const int64_t hidden_size,
                        ValidateActivationViews(input, residual, output, new_residual));

    if (!weight.is_valid() || weight.dtype() != DataType::Float32() ||
        weight.rank() != 1 || weight.dim(0) != hidden_size) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires rank-1 float32 weight matching hidden_size");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "AddRmsNormKernelEntry"));
    if (row_count == 0) {
        ::new (params_buffer) AddRmsNormF32KernelArgs{
                .row_count = 0,
                .hidden_size = hidden_size,
                .eps = eps,
        };
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis weight_analysis,
                        AnalyzeRowwiseView(weight, "AddRmsNormKernelEntry weight"));
    AM_ASSIGN_OR_RETURN(const AddRmsNormF32KernelArgs args,
                        BuildNonEmptyArgs(input, residual, output, new_residual,
                                          weight.data<float>(), weight_analysis.column_stride(),
                                          &weight_analysis, nullptr, hidden_size, eps));
    ::new (params_buffer) AddRmsNormF32KernelArgs(args);
    return Status::Ok();
}

Status BuildAddRmsNormF32PackedIdentityArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    if (context.inputs.size() != 2U || context.outputs.size() != 2U) {
        return Status::InvalidArgument(
                "AddRmsNorm packed kernel requires input, residual and two outputs");
    }
    if (!context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "AddRmsNorm packed kernel requires a packed weight artifact");
    }

    AM_ASSIGN_OR_RETURN(const float eps, ReadEpsilon(context.attrs));
    const TensorView& input = context.inputs[0];
    const TensorView& residual = context.inputs[1];
    const MutableTensorView& output = context.outputs[0];
    const MutableTensorView& new_residual = context.outputs[1];
    AM_ASSIGN_OR_RETURN(const int64_t hidden_size,
                        ValidateActivationViews(input, residual, output, new_residual));

    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            *context.packed_weight, std::array{hidden_size},
            "AddRmsNormKernelEntry"));
    if (context.packed_weight->nbytes >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow(
                "AddRmsNormKernelEntry packed storage size exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "AddRmsNormKernelEntry"));
    if (row_count == 0) {
        ::new (params_buffer) AddRmsNormF32KernelArgs{
                .row_count = 0,
                .hidden_size = hidden_size,
                .eps = eps,
        };
        return Status::Ok();
    }

    if (context.packed_weight->data == nullptr) {
        return Status::InvalidArgument(
                "AddRmsNormKernelEntry requires packed weight data for non-empty tensors");
    }
    AM_ASSIGN_OR_RETURN(const AddRmsNormF32KernelArgs args,
                        BuildNonEmptyArgs(
                                input, residual, output, new_residual,
                                static_cast<const float*>(context.packed_weight->data), 1,
                                nullptr, &*context.packed_weight, hidden_size, eps));
    ::new (params_buffer) AddRmsNormF32KernelArgs(args);
    return Status::Ok();
}

Status BuildAddRmsNormF32Metadata(const OpParams& params,
                                  std::vector<std::byte>& attrs) {
    const auto* add_rmsnorm_params = std::get_if<AddRmsNormParams>(&params);
    if (add_rmsnorm_params == nullptr) {
        return Status::InvalidArgument("AddRmsNorm kernel requires AddRmsNormParams");
    }

    if (!std::isfinite(add_rmsnorm_params->eps) || add_rmsnorm_params->eps <= 0.0F) {
        return Status::InvalidArgument("AddRmsNorm kernel requires finite positive epsilon");
    }

    const auto eps_bytes = std::as_bytes(std::span{&add_rmsnorm_params->eps, size_t{1}});
    attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status AddRmsNormF32ReferenceEntry(const KernelContext& context) noexcept {
    const auto* args = static_cast<const AddRmsNormF32KernelArgs*>(context.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunAddRmsNormF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<AddRmsNormF32KernelArgs>);
static_assert(sizeof(AddRmsNormF32KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(AddRmsNormF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuAddRmsNormF32Reference,
        KernelDescriptor{
                .op_type = OpType::kAddRmsNorm,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &AddRmsNormF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(AddRmsNormF32KernelArgs),
                .params_builder = &BuildAddRmsNormF32PlainArgs,
                .metadata_builder = &BuildAddRmsNormF32Metadata,
                .name = "cpu::add_rmsnorm_f32_reference"})

AM_REGISTER_KERNEL(
        CpuAddRmsNormF32PackedIdentityReference,
        KernelDescriptor{
                .op_type = OpType::kAddRmsNorm,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &AddRmsNormF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(AddRmsNormF32KernelArgs),
                .params_builder = &BuildAddRmsNormF32PackedIdentityArgs,
                .metadata_builder = &BuildAddRmsNormF32Metadata,
                .name = "cpu::add_rmsnorm_f32_packed_identity_reference"})

} // namespace aethermind::cpu::detail
