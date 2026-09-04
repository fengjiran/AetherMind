#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "rope_internal.h"
#include "utils/overflow_check.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

struct RoPEF32KernelMetadata {
    int64_t head_dim{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};
    double theta{};
    double position_divisor{};
};

bool HasIdenticalMapping(const TensorView& input,
                         const MutableTensorView& output) noexcept {
    if (input.data() != output.data() || input.dtype() != output.dtype() ||
        input.rank() != output.rank()) {
        return false;
    }

    for (int32_t dim = 0; dim < input.rank(); ++dim) {
        if (input.dim(dim) != output.dim(dim) || input.stride(dim) != output.stride(dim)) {
            return false;
        }
    }
    return true;
}

Status ValidateRoPEParamsForKernel(const RoPEParams& params,
                                   RoPEF32KernelMetadata& metadata) noexcept {
    if (params.head_dim <= 0 || params.num_attention_heads <= 0 ||
        params.num_key_value_heads <= 0 || params.max_position_embeddings <= 0) {
        return Status::InvalidArgument("CPU RoPE requires positive dimensions and head counts");
    }
    if (params.head_dim % 2 != 0) {
        return Status::InvalidArgument("CPU RoPE requires an even head_dim");
    }
    if (!std::isfinite(params.theta) || params.theta <= 0.0) {
        return Status::InvalidArgument("CPU RoPE requires finite positive theta");
    }

    int64_t q_width = 0;
    if (CheckOverflowMul(params.num_attention_heads, params.head_dim, &q_width)) {
        return Status::InvalidArgument("CPU RoPE q width overflows int64_t");
    }
    int64_t k_width = 0;
    if (CheckOverflowMul(params.num_key_value_heads, params.head_dim, &k_width)) {
        return Status::InvalidArgument("CPU RoPE k width overflows int64_t");
    }

    double position_divisor = 1.0;
    switch (params.scaling_type) {
        case RoPEScalingType::kNone:
            if (params.scaling_factor.has_value()) {
                return Status::InvalidArgument(
                        "CPU RoPE kNone scaling must not carry a scaling_factor");
            }
            break;
        case RoPEScalingType::kLinear:
            if (!params.scaling_factor.has_value() ||
                !std::isfinite(*params.scaling_factor) || *params.scaling_factor <= 0.0) {
                return Status::InvalidArgument(
                        "CPU RoPE kLinear scaling requires a finite positive scaling_factor");
            }
            position_divisor = *params.scaling_factor;
            break;
        default:
            return Status::InvalidArgument("CPU RoPE has an unsupported scaling_type");
    }

    metadata = RoPEF32KernelMetadata{
            .head_dim = params.head_dim,
            .num_q_heads = params.num_attention_heads,
            .num_kv_heads = params.num_key_value_heads,
            .theta = params.theta,
            .position_divisor = position_divisor,
    };
    return Status::Ok();
}

Status ValidateMetadata(const RoPEF32KernelMetadata& metadata) noexcept {
    if (metadata.head_dim <= 0 || metadata.head_dim % 2 != 0 ||
        metadata.num_q_heads <= 0 || metadata.num_kv_heads <= 0 ||
        !std::isfinite(metadata.theta) || metadata.theta <= 0.0 ||
        !std::isfinite(metadata.position_divisor) || metadata.position_divisor <= 0.0) {
        return Status::InvalidArgument("CPU RoPE kernel attrs are invalid");
    }

    int64_t q_width = 0;
    if (CheckOverflowMul(metadata.num_q_heads, metadata.head_dim, &q_width)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs q width overflows int64_t");
    }
    int64_t k_width = 0;
    if (CheckOverflowMul(metadata.num_kv_heads, metadata.head_dim, &k_width)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs k width overflows int64_t");
    }
    return Status::Ok();
}

Status BuildRoPEF32ReferenceArgs(const KernelParamsBuildContext& context,
                                 void* params_buffer) noexcept {
    if (context.attrs.size() != sizeof(RoPEF32KernelMetadata)) {
        return Status::InvalidArgument("CPU RoPE requires frozen metadata attrs");
    }

    RoPEF32KernelMetadata metadata{};
    std::memcpy(&metadata, context.attrs.data(), sizeof(metadata));
    AM_RETURN_IF_ERROR(ValidateMetadata(metadata));

    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 3 || outputs.size() != 2) {
        return Status::InvalidArgument("RoPE requires 3 inputs and 2 outputs");
    }

    const TensorView& q = inputs[0];
    const TensorView& k = inputs[1];
    const TensorView& position_ids = inputs[2];
    const MutableTensorView& q_output = outputs[0];
    const MutableTensorView& k_output = outputs[1];
    if (!q.is_valid() || !k.is_valid() || !position_ids.is_valid() ||
        !q_output.is_valid() || !k_output.is_valid()) {
        return Status::InvalidArgument("CPU RoPE requires valid TensorViews");
    }

    const DataType fp32 = DataType::Float32();
    if (q.dtype() != fp32 || k.dtype() != fp32 || q_output.dtype() != fp32 ||
        k_output.dtype() != fp32 || position_ids.dtype() != DataType::Int(64)) {
        return Status::InvalidArgument(
                "CPU RoPE reference requires float32 q/k/outputs and int64 position_ids");
    }
    if (q.rank() != 2 || k.rank() != 2 || q_output.rank() != 2 ||
        k_output.rank() != 2 || position_ids.rank() != 1) {
        return Status::InvalidArgument(
                "CPU RoPE requires rank-2 q/k/outputs and rank-1 position_ids");
    }

    const int64_t seq_len = q.dim(0);
    if (seq_len <= 0) {
        return Status::InvalidArgument("CPU RoPE requires positive seq_len");
    }
    if (k.dim(0) != seq_len || position_ids.dim(0) != seq_len ||
        q_output.dim(0) != seq_len || k_output.dim(0) != seq_len) {
        return Status::InvalidArgument("CPU RoPE requires matching sequence lengths");
    }

    int64_t q_width = 0;
    if (CheckOverflowMul(metadata.num_q_heads, metadata.head_dim, &q_width)) {
        return Status::InvalidArgument("CPU RoPE q width overflows int64_t");
    }
    int64_t k_width = 0;
    if (CheckOverflowMul(metadata.num_kv_heads, metadata.head_dim, &k_width)) {
        return Status::InvalidArgument("CPU RoPE k width overflows int64_t");
    }
    if (q.dim(1) != q_width || q_output.dim(1) != q_width ||
        k.dim(1) != k_width || k_output.dim(1) != k_width) {
        return Status::InvalidArgument("CPU RoPE tensor widths do not match RoPEParams");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            q, "CPU RoPE requires positive q strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            k, "CPU RoPE requires positive k strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            position_ids, "CPU RoPE requires positive position_ids strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            q_output, "CPU RoPE requires positive q output strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            k_output, "CPU RoPE requires positive k output strides"));

    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, q_width, q.stride(0), q.stride(1), "q"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, k_width, k.stride(0), k.stride(1), "k"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, 1, position_ids.stride(0), 1, "position_ids"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, q_width, q_output.stride(0), q_output.stride(1), "q output"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, k_width, k_output.stride(0), k_output.stride(1), "k output"));
    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            "CPU RoPE", seq_len, q_width, q_output.stride(0), q_output.stride(1)));
    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            "CPU RoPE", seq_len, k_width, k_output.stride(0), k_output.stride(1)));

    if (q.data() == q_output.data() && !HasIdenticalMapping(q, q_output)) {
        return Status::InvalidArgument(
                "CPU RoPE q in-place execution requires identical input and output mapping");
    }
    if (k.data() == k_output.data() && !HasIdenticalMapping(k, k_output)) {
        return Status::InvalidArgument(
                "CPU RoPE k in-place execution requires identical input and output mapping");
    }
    if (q_output.data() == k.data() || q_output.data() == k_output.data() ||
        q_output.data() == position_ids.data()) {
        return Status::InvalidArgument(
                "CPU RoPE q output must not alias k, k output, or position_ids");
    }
    if (k_output.data() == q.data() || k_output.data() == q_output.data() ||
        k_output.data() == position_ids.data()) {
        return Status::InvalidArgument(
                "CPU RoPE k output must not alias q, q output, or position_ids");
    }

    ::new (params_buffer) RoPEF32KernelArgs{
            .q = q.data<float>(),
            .k = k.data<float>(),
            .position_ids = position_ids.data<int64_t>(),
            .q_output = q_output.data<float>(),
            .k_output = k_output.data<float>(),
            .seq_len = seq_len,
            .head_dim = metadata.head_dim,
            .num_q_heads = metadata.num_q_heads,
            .num_kv_heads = metadata.num_kv_heads,
            .q_row_stride = q.stride(0),
            .q_col_stride = q.stride(1),
            .k_row_stride = k.stride(0),
            .k_col_stride = k.stride(1),
            .position_stride = position_ids.stride(0),
            .q_output_row_stride = q_output.stride(0),
            .q_output_col_stride = q_output.stride(1),
            .k_output_row_stride = k_output.stride(0),
            .k_output_col_stride = k_output.stride(1),
            .theta = metadata.theta,
            .position_divisor = metadata.position_divisor,
    };
    return Status::Ok();
}

Status BuildRoPEF32Metadata(const OpParams& params,
                            std::vector<std::byte>& attrs) {
    const auto* rope_params = std::get_if<RoPEParams>(&params);
    if (rope_params == nullptr) {
        return Status::InvalidArgument("CPU RoPE kernel requires RoPEParams");
    }

    RoPEF32KernelMetadata metadata{};
    AM_RETURN_IF_ERROR(ValidateRoPEParamsForKernel(*rope_params, metadata));
    const auto bytes = std::as_bytes(std::span{&metadata, size_t{1}});
    attrs.assign(bytes.begin(), bytes.end());
    return Status::Ok();
}

Status RoPEF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const RoPEF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunRoPEF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<RoPEF32KernelArgs>);
static_assert(alignof(RoPEF32KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(RoPEF32KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(
        CpuRoPEF32Reference,
        KernelDescriptor{
                .op_type = OpType::kRoPE,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &RoPEF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(RoPEF32KernelArgs),
                .params_builder = &BuildRoPEF32ReferenceArgs,
                .metadata_builder = &BuildRoPEF32Metadata,
                .name = "cpu::rope_f32_reference"})

} // namespace aethermind::cpu::detail
