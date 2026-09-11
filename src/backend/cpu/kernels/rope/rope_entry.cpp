#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/rope_frequency_resolver.h"
#include "rope_internal.h"
#include "utils/numeric_utils.h"
#include "utils/variant_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

Status ValidateNoRowwiseOverlap(const RowwiseAddressLayout& output,
                                const char* output_role,
                                const RowwiseAddressLayout& input,
                                const char* input_role) noexcept {
    switch (ClassifyRowwiseLayoutOverlap(output, input)) {
        case RowwiseLayoutOverlap::kDisjoint:
            return Status::Ok();
        case RowwiseLayoutOverlap::kProvenOverlap:
            return Status::InvalidArgument(std::string("CPU RoPE ") + output_role +
                                           " must not overlap " + input_role);
        case RowwiseLayoutOverlap::kMayOverlap:
            return Status::Unimplemented(std::string("CPU RoPE cannot prove ") + output_role +
                                         " is disjoint from " + input_role +
                                         " for the requested strided layouts");
    }
    return Status::Internal("CPU RoPE row-wise overlap classification is invalid");
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
    if (attrs.size() < sizeof(metadata) || metadata.head_dim <= 0 || metadata.rotary_dim <= 0 ||
        metadata.rotary_dim > metadata.head_dim || metadata.rotary_dim % 2 != 0 ||
        metadata.num_q_heads <= 0 || metadata.num_kv_heads <= 0 ||
        !IsFinitePositive(metadata.theta) || !IsFinitePositive(metadata.position_divisor) ||
        !IsFinitePositive(metadata.rotary_output_scale) ||
        (metadata.pairing != RoPEPairing::kSplitHalf &&
         metadata.pairing != RoPEPairing::kInterleaved)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs are invalid");
    }

    if (const uint8_t expected_table_count = ExpectedFrequencyTableCount(metadata.algorithm);
        expected_table_count == 255 || metadata.freq_table_count != expected_table_count ||
        metadata.freq_count != static_cast<uint32_t>(metadata.rotary_dim / 2)) {
        return Status::InvalidArgument("CPU RoPE kernel attrs have invalid frequency tables");
    }

    switch (metadata.algorithm) {
        case RoPEAlgorithm::kStandard:
        case RoPEAlgorithm::kLinear:
            break;
        case RoPEAlgorithm::kDynamicNtk:
            if (metadata.rotary_dim <= 2 || !IsFinitePositive(metadata.factor) ||
                metadata.original_context_length <= 0) {
                return Status::InvalidArgument("CPU RoPE Dynamic NTK attrs are invalid");
            }
            break;
        case RoPEAlgorithm::kYarn:
            if (metadata.theta <= 1.0 || !IsFinitePositive(metadata.factor) ||
                !IsFinitePositive(metadata.beta_fast) || !IsFinitePositive(metadata.beta_slow) ||
                metadata.beta_fast <= metadata.beta_slow || metadata.original_context_length <= 0) {
                return Status::InvalidArgument("CPU RoPE YaRN attrs are invalid");
            }
            break;
        case RoPEAlgorithm::kLlama3:
            if (!IsFinitePositive(metadata.factor) ||
                !IsFinitePositive(metadata.low_freq_factor) ||
                !IsFinitePositive(metadata.high_freq_factor) ||
                metadata.high_freq_factor <= metadata.low_freq_factor ||
                metadata.original_context_length <= 0) {
                return Status::InvalidArgument("CPU RoPE Llama3 attrs are invalid");
            }
            break;
        case RoPEAlgorithm::kLongRope:
            if (metadata.original_context_length <= 0) {
                return Status::InvalidArgument("CPU RoPE LongRoPE attrs are invalid");
            }
            break;
    }

    size_t table_bytes = 0;
    if (CheckOverflowMul(static_cast<size_t>(metadata.freq_count), sizeof(double),
                         &table_bytes) ||
        CheckOverflowMul(table_bytes, static_cast<size_t>(metadata.freq_table_count),
                         &table_bytes) ||
        table_bytes > attrs.size() - sizeof(metadata) ||
        attrs.size() != sizeof(metadata) + table_bytes) {
        return Status::InvalidArgument("CPU RoPE kernel attrs are truncated");
    }
    return Status::Ok();
}

StatusOr<std::array<double, 2>> ValidateFrequencyTablesAndFindMax(
        const RoPEF32KernelMetadata& metadata,
        std::span<const std::byte> attrs) noexcept {
    std::array<double, 2> max_inv_freqs{};
    for (uint8_t table = 0; table < metadata.freq_table_count; ++table) {
        for (uint32_t pair = 0; pair < metadata.freq_count; ++pair) {
            auto freq = ReadStaticInvFreqUnchecked(metadata.freq_count, attrs, table, pair);
            if (!IsFinitePositive(freq)) {
                return Status::Overflow("CPU RoPE inverse frequency attrs are invalid");
            }
            max_inv_freqs[table] = std::max(max_inv_freqs[table], freq);
        }
    }
    return max_inv_freqs;
}

StatusOr<RoPEF32KernelMetadata> MakeRoPEF32KernelMetadata(const RoPEParams& params) noexcept {
    if (params.head_dim <= 0 || params.num_q_heads <= 0 ||
        params.num_kv_heads <= 0 || params.max_pos_embeddings <= 0) {
        return Status::InvalidArgument(
                "CPU RoPE requires positive dimensions and head counts");
    }

    AM_RETURN_IF_ERROR(ValidateRoPEFreqParams(params));
    const int64_t rotary_dim = EffectiveRoPERotaryDim(params);
    const int64_t pair_count = rotary_dim / 2;
    const RoPEAlgorithm rope_algorithm = GetRoPEAlgorithm(params.algorithm);
    if (static_cast<uint64_t>(pair_count) >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return Status::InvalidArgument("CPU RoPE rotary pair count exceeds metadata range");
    }

    int64_t q_width = 0;
    if (CheckOverflowMul(params.num_q_heads, params.head_dim, &q_width)) {
        return Status::InvalidArgument("CPU RoPE q width overflows int64_t");
    }

    int64_t k_width = 0;
    if (CheckOverflowMul(params.num_kv_heads, params.head_dim, &k_width)) {
        return Status::InvalidArgument("CPU RoPE k width overflows int64_t");
    }

    RoPEF32KernelMetadata metadata{
            .head_dim = params.head_dim,
            .rotary_dim = rotary_dim,
            .num_q_heads = params.num_q_heads,
            .num_kv_heads = params.num_kv_heads,
            .theta = params.theta,
            .position_divisor = 1.0,
            .rotary_output_scale = 1.0,
            .freq_count = static_cast<uint32_t>(pair_count),
            .truncate_correction_range = false,
            .freq_table_count = ExpectedFrequencyTableCount(rope_algorithm),
            .pairing = params.pairing,
            .algorithm = rope_algorithm};

    auto visitor = overloaded{
            [](const StandardRoPE&) {},
            [](const LinearRoPE&) {},
            [&](const DynamicNtkRoPE& algorithm) {
                metadata.factor = algorithm.factor;
                metadata.original_context_length = algorithm.original_context_length;
            },
            [&](const YarnRoPE& algorithm) {
                metadata.factor = algorithm.factor;
                metadata.original_context_length = algorithm.original_context_length;
                metadata.beta_fast = algorithm.beta_fast;
                metadata.beta_slow = algorithm.beta_slow;
                metadata.truncate_correction_range = algorithm.truncate_correction_range;
            },
            [&](const Llama3RoPE& algorithm) {
                metadata.factor = algorithm.factor;
                metadata.low_freq_factor = algorithm.low_frequency_factor;
                metadata.high_freq_factor = algorithm.high_frequency_factor;
                metadata.original_context_length = algorithm.original_context_length;
            },
            [&](const LongRoPE& algorithm) {
                metadata.original_context_length = algorithm.original_context_length;
            }};
    std::visit(visitor, params.algorithm);
    return metadata;
}

Status BuildRoPEF32Metadata(const OpParams& params, std::vector<std::byte>& attrs) {
    const auto* rope_params = std::get_if<RoPEParams>(&params);
    if (rope_params == nullptr) {
        return Status::InvalidArgument("CPU RoPE kernel requires RoPEParams");
    }

    AM_ASSIGN_OR_RETURN(RoPEF32KernelMetadata metadata, MakeRoPEF32KernelMetadata(*rope_params));
    std::vector<RoPERotationCoefficients> tables;
    auto append_static_coefficients = [&]() -> Status {
        AM_ASSIGN_OR_RETURN(auto coefficients,
                            ResolveStaticRoPERotationCoefficients(*rope_params));
        metadata.position_divisor = coefficients.position_divisor;
        metadata.rotary_output_scale = coefficients.rotary_output_scale;
        tables.push_back(std::move(coefficients));
        return Status::Ok();
    };

    auto visitor = overloaded{
            [&](const LongRoPE& algorithm) -> Status {
                if (algorithm.original_context_length == std::numeric_limits<int64_t>::max()) {
                    return Status::InvalidArgument(
                            "CPU RoPE LongRoPE original context is too large");
                }

                AM_ASSIGN_OR_RETURN(auto short_coefficients,
                                    ResolveDynamicRoPERotationCoefficients(
                                            *rope_params, algorithm.original_context_length));
                AM_ASSIGN_OR_RETURN(auto long_coefficients,
                                    ResolveDynamicRoPERotationCoefficients(
                                            *rope_params, algorithm.original_context_length + 1));
                if (short_coefficients.position_divisor != long_coefficients.position_divisor ||
                    short_coefficients.rotary_output_scale != long_coefficients.rotary_output_scale) {
                    return Status::Internal(
                            "CPU RoPE LongRoPE resolved inconsistent scalar coefficients");
                }
                metadata.position_divisor = short_coefficients.position_divisor;
                metadata.rotary_output_scale = short_coefficients.rotary_output_scale;
                tables.push_back(std::move(short_coefficients));
                tables.push_back(std::move(long_coefficients));
                return Status::Ok();
            },
            [](const DynamicNtkRoPE&) -> Status {
                // Dynamic NTK derives frequencies at invocation time; no frozen table.
                return Status::Ok();
            },
            [&](const StandardRoPE&) -> Status {
                return append_static_coefficients();
            },
            [&](const LinearRoPE&) -> Status {
                return append_static_coefficients();
            },
            [&](const YarnRoPE&) -> Status {
                return append_static_coefficients();
            },
            [&](const Llama3RoPE&) -> Status {
                return append_static_coefficients();
            }};
    AM_RETURN_IF_ERROR(std::visit(visitor, rope_params->algorithm));

    if (tables.size() != metadata.freq_table_count) {
        return Status::Internal("CPU RoPE metadata table count does not match algorithm");
    }

    const auto bytes = std::as_bytes(std::span{&metadata, size_t{1}});
    attrs.assign(bytes.begin(), bytes.end());
    for (const auto& table: tables) {
        const auto& inv_freqs = table.inv_freqs;
        if (inv_freqs.size() != metadata.freq_count) {
            return Status::Internal("CPU RoPE resolver returned invalid frequency count");
        }

        const auto append = [&](const std::vector<double>& freqs) {
            const auto factor_bytes = std::as_bytes(std::span{freqs});
            attrs.insert(attrs.end(), factor_bytes.begin(), factor_bytes.end());
        };
        append(inv_freqs);
    }
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
    AM_ASSIGN_OR_RETURN(const auto max_inv_freqs,
                        ValidateFrequencyTablesAndFindMax(metadata, context.attrs));
    int64_t checked_q_width = 0;
    int64_t checked_kv_width = 0;
    if (CheckOverflowMul(metadata.num_q_heads, metadata.head_dim, &checked_q_width) ||
        CheckOverflowMul(metadata.num_kv_heads, metadata.head_dim, &checked_kv_width)) {
        return Status::InvalidArgument(
                "CPU RoPE kernel attrs tensor width overflows int64_t");
    }

    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 3 || outputs.size() != 2) {
        return Status::InvalidArgument("RoPE requires 3 inputs and 2 outputs");
    }

    const TensorView& q = inputs[0];
    const TensorView& k = inputs[1];
    const TensorView& pos_ids = inputs[2];
    const MutableTensorView& q_output = outputs[0];
    const MutableTensorView& k_output = outputs[1];
    if (!q.is_valid() || !k.is_valid() || !pos_ids.is_valid() ||
        !q_output.is_valid() || !k_output.is_valid()) {
        return Status::InvalidArgument("CPU RoPE requires valid TensorViews");
    }

    if (const auto fp32 = DataType::Float32();
        q.dtype() != fp32 || k.dtype() != fp32 || q_output.dtype() != fp32 ||
        k_output.dtype() != fp32 || pos_ids.dtype() != DataType::Int(64)) {
        return Status::InvalidArgument(
                "CPU RoPE reference requires float32 q/k/outputs and int64 position_ids");
    }

    if (q.rank() != 2 || k.rank() != 2 || q_output.rank() != 2 ||
        k_output.rank() != 2 || pos_ids.rank() != 1) {
        return Status::InvalidArgument(
                "CPU RoPE requires rank-2 q/k/outputs and rank-1 position_ids");
    }

    const int64_t seq_len = q.dim(0);
    if (seq_len <= 0) {
        return Status::InvalidArgument("CPU RoPE requires positive seq_len");
    }

    if (k.dim(0) != seq_len || pos_ids.dim(0) != seq_len ||
        q_output.dim(0) != seq_len || k_output.dim(0) != seq_len) {
        return Status::InvalidArgument("CPU RoPE requires matching sequence lengths");
    }

    const int64_t q_width = checked_q_width;
    const int64_t k_width = checked_kv_width;
    if (q.dim(1) != q_width || q_output.dim(1) != q_width ||
        k.dim(1) != k_width || k_output.dim(1) != k_width) {
        return Status::InvalidArgument("CPU RoPE tensor widths do not match RoPEParams");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            q, "CPU RoPE requires positive q strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            k, "CPU RoPE requires positive k strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            pos_ids, "CPU RoPE requires positive position_ids strides"));
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
            pos_ids.stride(0), 1, "position_ids"));
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
                                                  q.itemsize(), "CPU RoPE q"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout k_layout,
                        BuildRowwiseAddressLayout(k.data(), seq_len, k_width,
                                                  k.stride(0), k.stride(1),
                                                  k.itemsize(), "CPU RoPE k"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout position_layout,
                        BuildRowwiseAddressLayout(pos_ids.data(), seq_len, 1,
                                                  pos_ids.stride(0), 1,
                                                  pos_ids.itemsize(), "CPU RoPE position_ids"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout q_output_layout,
                        BuildRowwiseAddressLayout(q_output.data(), seq_len, q_width,
                                                  q_output.stride(0), q_output.stride(1),
                                                  q_output.itemsize(), "CPU RoPE q output"));
    AM_ASSIGN_OR_RETURN(const RowwiseAddressLayout k_output_layout,
                        BuildRowwiseAddressLayout(k_output.data(), seq_len, k_width,
                                                  k_output.stride(0), k_output.stride(1),
                                                  k_output.itemsize(), "CPU RoPE k output"));

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
            .pos_ids = pos_ids.data<int64_t>(),
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
            .pos_stride = pos_ids.stride(0),
            .q_output_row_stride = q_output.stride(0),
            .q_output_col_stride = q_output.stride(1),
            .k_output_row_stride = k_output.stride(0),
            .k_output_col_stride = k_output.stride(1),
            .original_context_length = metadata.original_context_length,
            .theta = metadata.theta,
            .factor = metadata.factor,
            .position_divisor = metadata.position_divisor,
            .rotary_output_scale = metadata.rotary_output_scale,
            .short_max_inv_freq = max_inv_freqs[0],
            .long_max_inv_freq = max_inv_freqs[1],
            .freq_count = metadata.freq_count,
            .pairing = metadata.pairing,
            .algorithm = metadata.algorithm,
    };
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
