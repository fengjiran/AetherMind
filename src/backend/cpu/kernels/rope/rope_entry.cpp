#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/rope_frequency_resolver.h"
#include "rope_internal.h"
#include "utils/overflow_check.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

struct AddressRange {
    std::uintptr_t begin{};
    std::uintptr_t end{};
};

struct RowwiseAddressLayout {
    AddressRange envelope{};
    std::uintptr_t row_stride_bytes{};
    std::uintptr_t row_bytes{};
    int64_t row_count{};
};

bool ToAddressValue(int64_t value, std::uintptr_t* result) noexcept {
    if (value < 0 ||
        static_cast<std::uintmax_t>(value) > std::numeric_limits<std::uintptr_t>::max()) {
        return false;
    }

    *result = static_cast<std::uintptr_t>(value);
    return true;
}

StatusOr<RowwiseAddressLayout> BuildRowwiseAddressLayout(
        const void* data,
        int64_t row_count,
        int64_t column_count,
        int64_t row_stride,
        int64_t column_stride,
        size_t item_size,
        const char* role) noexcept {
    std::uintptr_t row_count_minus_one = 0;
    std::uintptr_t column_count_minus_one = 0;
    std::uintptr_t row_stride_elements = 0;
    std::uintptr_t column_stride_elements = 0;
    if (!ToAddressValue(row_count - 1, &row_count_minus_one) ||
        !ToAddressValue(column_count - 1, &column_count_minus_one) ||
        !ToAddressValue(row_stride, &row_stride_elements) ||
        !ToAddressValue(column_stride, &column_stride_elements) ||
        item_size > std::numeric_limits<std::uintptr_t>::max()) {
        return Status::InvalidArgument(std::string("CPU RoPE ") + role +
                                       " address range overflow");
    }

    const auto item_size_bytes = item_size;
    std::uintptr_t last_column_offset = 0;
    std::uintptr_t row_span_elements = 0;
    std::uintptr_t row_bytes = 0;
    std::uintptr_t row_stride_bytes = 0;
    std::uintptr_t last_row_offset = 0;
    if (CheckOverflowMul(column_count_minus_one, column_stride_elements, &last_column_offset) ||
        CheckOverflowAdd(last_column_offset, std::uintptr_t{1}, &row_span_elements) ||
        CheckOverflowMul(row_span_elements, item_size_bytes, &row_bytes) ||
        CheckOverflowMul(row_stride_elements, item_size_bytes, &row_stride_bytes) ||
        CheckOverflowMul(row_count_minus_one, row_stride_bytes, &last_row_offset)) {
        return Status::InvalidArgument(std::string("CPU RoPE ") + role +
                                       " address range overflow");
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    std::uintptr_t last_row_begin = 0;
    std::uintptr_t end = 0;
    if (CheckOverflowAdd(begin, last_row_offset, &last_row_begin) ||
        CheckOverflowAdd(last_row_begin, row_bytes, &end)) {
        return Status::InvalidArgument(std::string("CPU RoPE ") + role +
                                       " address range overflow");
    }

    return RowwiseAddressLayout{
            .envelope = AddressRange{.begin = begin, .end = end},
            .row_stride_bytes = row_stride_bytes,
            .row_bytes = row_bytes,
            .row_count = row_count,
    };
}

AddressRange RowAddressRange(const RowwiseAddressLayout& layout,
                             int64_t row) noexcept {
    // BuildRowwiseAddressLayout checked the last row's end address, so every
    // earlier row address and end is representable as well.
    const std::uintptr_t row_offset =
            static_cast<std::uintptr_t>(row) * layout.row_stride_bytes;
    const std::uintptr_t begin = layout.envelope.begin + row_offset;
    return AddressRange{.begin = begin, .end = begin + layout.row_bytes};
}

bool RangesOverlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool RowwiseRangesOverlap(const RowwiseAddressLayout& lhs,
                          const RowwiseAddressLayout& rhs) noexcept {
    // Positive row strides keep both row ranges in increasing-address order.
    // A row range includes column-stride holes, deliberately rejecting a few
    // otherwise safe strided views instead of requiring set-wise alias checks.
    int64_t lhs_row = 0;
    int64_t rhs_row = 0;
    while (lhs_row < lhs.row_count && rhs_row < rhs.row_count) {
        const AddressRange lhs_range = RowAddressRange(lhs, lhs_row);
        const AddressRange rhs_range = RowAddressRange(rhs, rhs_row);
        if (RangesOverlap(lhs_range, rhs_range)) {
            return true;
        }

        if (lhs_range.end <= rhs_range.begin) {
            ++lhs_row;
        } else {
            ++rhs_row;
        }
    }
    return false;
}

Status ValidateNoRowwiseOverlap(const RowwiseAddressLayout& output,
                                const char* output_role,
                                const RowwiseAddressLayout& input,
                                const char* input_role) noexcept {
    if (!RangesOverlap(output.envelope, input.envelope) ||
        !RowwiseRangesOverlap(output, input)) {
        return Status::Ok();
    }

    return Status::InvalidArgument(std::string("CPU RoPE ") + output_role +
                                   " must not overlap " + input_role);
}

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

uint8_t ExpectedFrequencyTableCount(RoPEAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case RoPEAlgorithm::kDynamicNtk:
            return 0;
        case RoPEAlgorithm::kLongRope:
            return 2;
        case RoPEAlgorithm::kStandard:
        case RoPEAlgorithm::kLinear:
        case RoPEAlgorithm::kYarn:
        case RoPEAlgorithm::kLlama3:
            return 1;
    }
    return 255;
}

Status ValidateMetadataLayout(const RoPEF32KernelMetadata& metadata,
                              std::span<const std::byte> attrs) noexcept {
    if (metadata.head_dim <= 0 || metadata.rotary_dim <= 0 ||
        metadata.rotary_dim > metadata.head_dim || metadata.rotary_dim % 2 != 0 ||
        metadata.num_q_heads <= 0 || metadata.num_kv_heads <= 0 ||
        !std::isfinite(metadata.theta) || metadata.theta <= 0.0 ||
        (metadata.pairing != RoPEPairing::kSplitHalf &&
         metadata.pairing != RoPEPairing::kInterleaved)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs are invalid");
    }
    const uint8_t expected_table_count = ExpectedFrequencyTableCount(metadata.algorithm);
    if (expected_table_count == 255 || metadata.frequency_table_count != expected_table_count ||
        metadata.frequency_count != static_cast<uint32_t>(metadata.rotary_dim / 2)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs have invalid frequency tables");
    }
    size_t table_bytes = 0;
    if (CheckOverflowMul(static_cast<size_t>(metadata.frequency_count), sizeof(double),
                         &table_bytes) ||
        CheckOverflowMul(table_bytes, static_cast<size_t>(metadata.frequency_table_count),
                         &table_bytes) ||
        table_bytes > attrs.size() - sizeof(metadata) ||
        attrs.size() != sizeof(metadata) + table_bytes) {
        return Status::InvalidArgument("CPU RoPE kernel attrs are truncated");
    }
    return Status::Ok();
}

Status ValidateRoPEParamsForKernel(const RoPEParams& params,
                                   RoPEF32KernelMetadata& metadata) noexcept {
    if (params.head_dim <= 0 || params.num_attention_heads <= 0 ||
        params.num_key_value_heads <= 0 || params.max_pos_embeddings <= 0) {
        return Status::InvalidArgument(
                "CPU RoPE requires positive dimensions and head counts");
    }

    AM_RETURN_IF_ERROR(ValidateRoPEFrequencyParameters(params));
    if (static_cast<uint64_t>(EffectiveRoPERotaryDim(params) / 2) >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Status::InvalidArgument("CPU RoPE rotary pair count exceeds metadata range");
    }

    int64_t q_width = 0;
    if (CheckOverflowMul(params.num_attention_heads, params.head_dim, &q_width)) {
        return Status::InvalidArgument("CPU RoPE q width overflows int64_t");
    }

    int64_t k_width = 0;
    if (CheckOverflowMul(params.num_key_value_heads, params.head_dim, &k_width)) {
        return Status::InvalidArgument("CPU RoPE k width overflows int64_t");
    }

    metadata = RoPEF32KernelMetadata{
            .head_dim = params.head_dim,
            .rotary_dim = EffectiveRoPERotaryDim(params),
            .num_q_heads = params.num_attention_heads,
            .num_kv_heads = params.num_key_value_heads,
            .theta = params.theta,
            .frequency_count = static_cast<uint32_t>(EffectiveRoPERotaryDim(params) / 2),
            .pairing = params.pairing,
            .algorithm = GetRoPEAlgorithm(params.algorithm),
            .truncate_correction_range = false,
            .frequency_table_count = ExpectedFrequencyTableCount(GetRoPEAlgorithm(params.algorithm)),
    };
    std::visit(
            [&](const auto& algorithm) {
                using T = std::decay_t<decltype(algorithm)>;
                if constexpr (std::is_same_v<T, LinearRoPE>) {
                    metadata.factor = algorithm.factor;
                } else if constexpr (std::is_same_v<T, DynamicNtkRoPE>) {
                    metadata.factor = algorithm.factor;
                    metadata.original_context_length = algorithm.original_context_length;
                } else if constexpr (std::is_same_v<T, YarnRoPE>) {
                    metadata.factor = algorithm.factor;
                    metadata.original_context_length = algorithm.original_context_length;
                    metadata.beta_fast = algorithm.beta_fast;
                    metadata.beta_slow = algorithm.beta_slow;
                    metadata.rotary_output_scale = algorithm.rotary_output_scale;
                    metadata.truncate_correction_range = algorithm.truncate_correction_range;
                } else if constexpr (std::is_same_v<T, Llama3RoPE>) {
                    metadata.factor = algorithm.factor;
                    metadata.low_frequency_factor = algorithm.low_frequency_factor;
                    metadata.high_frequency_factor = algorithm.high_frequency_factor;
                    metadata.original_context_length = algorithm.original_context_length;
                } else if constexpr (std::is_same_v<T, LongRoPE>) {
                    metadata.original_context_length = algorithm.original_context_length;
                    metadata.rotary_output_scale = algorithm.rotary_output_scale;
                }
            },
            params.algorithm);
    return Status::Ok();
}

Status BuildRoPEF32ReferenceArgs(const KernelParamsBuildContext& context,
                                 void* params_buffer) noexcept {
    if (context.attrs.size() < sizeof(RoPEF32KernelMetadata)) {
        return Status::InvalidArgument("CPU RoPE requires frozen metadata attrs");
    }

    RoPEF32KernelMetadata metadata{};
    std::memcpy(&metadata, context.attrs.data(), sizeof(metadata));
    AM_RETURN_IF_ERROR(ValidateMetadataLayout(metadata, context.attrs));
    int64_t checked_q_width = 0;
    int64_t checked_k_width = 0;
    if (CheckOverflowMul(metadata.num_q_heads, metadata.head_dim,
                         &checked_q_width) ||
        CheckOverflowMul(metadata.num_kv_heads, metadata.head_dim,
                         &checked_k_width)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs tensor width overflows int64_t");
    }

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

    if (const DataType fp32 = DataType::Float32();
        q.dtype() != fp32 || k.dtype() != fp32 || q_output.dtype() != fp32 ||
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

    const int64_t q_width = checked_q_width;
    const int64_t k_width = checked_k_width;
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
            "CPU RoPE", seq_len, q_width, q.stride(0),
            q.stride(1), "q"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, k_width, k.stride(0),
            k.stride(1), "k"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, 1,
            position_ids.stride(0), 1, "position_ids"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, q_width,
            q_output.stride(0), q_output.stride(1), "q output"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "CPU RoPE", seq_len, k_width,
            k_output.stride(0), k_output.stride(1), "k output"));
    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            "CPU RoPE", seq_len, q_width,
            q_output.stride(0), q_output.stride(1)));
    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            "CPU RoPE", seq_len, k_width,
            k_output.stride(0), k_output.stride(1)));

    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout q_layout,
                        BuildRowwiseAddressLayout(q.data(), seq_len, q_width,
                                                  q.stride(0), q.stride(1),
                                                  q.itemsize(), "q"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout k_layout,
                        BuildRowwiseAddressLayout(k.data(), seq_len, k_width,
                                                  k.stride(0), k.stride(1),
                                                  k.itemsize(), "k"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout position_layout,
                        BuildRowwiseAddressLayout(position_ids.data(), seq_len, 1,
                                                  position_ids.stride(0), 1,
                                                  position_ids.itemsize(), "position_ids"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout q_output_layout,
                        BuildRowwiseAddressLayout(q_output.data(), seq_len, q_width,
                                                  q_output.stride(0), q_output.stride(1),
                                                  q_output.itemsize(), "q output"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout k_output_layout,
                        BuildRowwiseAddressLayout(k_output.data(), seq_len, k_width,
                                                  k_output.stride(0), k_output.stride(1),
                                                  k_output.itemsize(), "k output"));

    if (!HasIdenticalMapping(q, q_output)) {
        AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
                q_output_layout, "q output", q_layout, "q"));
    }
    if (!HasIdenticalMapping(k, k_output)) {
        AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
                k_output_layout, "k output", k_layout, "k"));
    }

    AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
            q_output_layout, "q output", k_layout, "k"));
    AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
            q_output_layout, "q output", k_output_layout, "k output"));
    AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
            q_output_layout, "q output", position_layout, "position_ids"));
    AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
            k_output_layout, "k output", q_layout, "q"));
    AM_RETURN_IF_ERROR(ValidateNoRowwiseOverlap(
            k_output_layout, "k output", position_layout, "position_ids"));

    ::new (params_buffer) RoPEF32KernelArgs{
            .q = q.data<float>(),
            .k = k.data<float>(),
            .pos_ids = position_ids.data<int64_t>(),
            .q_output = q_output.data<float>(),
            .k_output = k_output.data<float>(),
            .seq_len = seq_len,
            .head_dim = metadata.head_dim,
            .rotary_dim = metadata.rotary_dim,
            .num_q_heads = metadata.num_q_heads,
            .num_kv_heads = metadata.num_kv_heads,
            .q_row_stride = q.stride(0),
            .q_col_stride = q.stride(1),
            .k_row_stride = k.stride(0),
            .k_col_stride = k.stride(1),
            .pos_stride = position_ids.stride(0),
            .q_output_row_stride = q_output.stride(0),
            .q_output_col_stride = q_output.stride(1),
            .k_output_row_stride = k_output.stride(0),
            .k_output_col_stride = k_output.stride(1),
            .pairing = metadata.pairing,
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
    metadata.rotary_output_scale = 1.0;
    std::vector<ResolvedRoPEFreqs> tables;
    if (const auto* long_rope = std::get_if<LongRoPE>(&rope_params->algorithm)) {
        if (long_rope->original_context_length == std::numeric_limits<int64_t>::max()) {
            return Status::InvalidArgument("CPU RoPE LongRoPE original context is too large");
        }
        AM_ASSIGN_OR_RETURN(auto short_table,
                            ResolveDynamicRoPEFrequencies(*rope_params,
                                                          long_rope->original_context_length));
        AM_ASSIGN_OR_RETURN(auto long_table,
                            ResolveDynamicRoPEFrequencies(*rope_params,
                                                          long_rope->original_context_length + 1));
        metadata.rotary_output_scale = short_table.rotary_output_scale;
        tables.push_back(std::move(short_table));
        tables.push_back(std::move(long_table));
    } else if (!std::holds_alternative<DynamicNtkRoPE>(rope_params->algorithm)) {
        RoPEParams static_params = *rope_params;
        // Retain Linear's position-divisor formulation in the CPU reference.
        // It avoids rejecting a finite tiny factor when every current position
        // is zero, while preserving the exact mathematical result.
        if (std::holds_alternative<LinearRoPE>(static_params.algorithm)) {
            static_params.algorithm = StandardRoPE{};
        }
        AM_ASSIGN_OR_RETURN(auto table, ResolveStaticRoPEFrequencies(static_params));
        metadata.rotary_output_scale = table.rotary_output_scale;
        tables.push_back(std::move(table));
    }
    if (tables.size() != metadata.frequency_table_count) {
        return Status::Internal("CPU RoPE metadata table count does not match algorithm");
    }
    const auto bytes = std::as_bytes(std::span{&metadata, size_t{1}});
    attrs.assign(bytes.begin(), bytes.end());
    for (const ResolvedRoPEFreqs& table: tables) {
        if (table.inv_freqs.size() != metadata.frequency_count) {
            return Status::Internal("CPU RoPE resolver returned invalid frequency count");
        }
        const auto append = [&](const std::vector<double>& frequencies) {
            const auto factor_bytes = std::as_bytes(std::span{frequencies});
            attrs.insert(attrs.end(), factor_bytes.begin(), factor_bytes.end());
        };
        append(table.inv_freqs);
    }
    return Status::Ok();
}

Status RoPEF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const RoPEF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunRoPEF32Reference(*args, ctx.attrs);
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
