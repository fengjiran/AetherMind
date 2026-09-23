#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "attention_internal.h"
#include "utils/overflow_check.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

constexpr std::string_view kKernelName = "CPU Attention";

Status ValidateMetadata(const AttentionF32KernelMetadata& metadata) noexcept {
    if (metadata.num_q_heads <= 0 || metadata.num_kv_heads <= 0 ||
        metadata.head_dim <= 0 ||
        metadata.num_q_heads % metadata.num_kv_heads != 0 ||
        !std::isfinite(metadata.scale) || metadata.scale <= 0.0F) {
        return Status::InvalidArgument("CPU Attention has invalid frozen head geometry");
    }

    int64_t hidden = 0;
    if (CheckOverflowMul(metadata.num_q_heads, metadata.head_dim, &hidden)) {
        return Status::InvalidArgument("CPU Attention hidden width overflows int64_t");
    }
    return Status::Ok();
}

StatusOr<AttentionF32KernelMetadata> ReadMetadata(std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(AttentionF32KernelMetadata)) {
        return Status::InvalidArgument(
                "CPU Attention requires frozen metadata in attrs");
    }

    AttentionF32KernelMetadata metadata{};
    std::memcpy(&metadata, attrs.data(), sizeof(metadata));
    AM_RETURN_IF_ERROR(ValidateMetadata(metadata));
    return metadata;
}

template<typename TensorLike>
Status ValidateActivation(const TensorLike& tensor, std::string_view role) noexcept {
    if (!tensor.is_valid()) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires a valid " +
                std::string(role) + " TensorView");
    }

    if (tensor.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires float32 " +
                std::string(role) + " TensorView");
    }

    if (tensor.rank() != 2 || tensor.dim(0) <= 0 || tensor.dim(1) <= 0) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires " +
                std::string(role) +
                " shape [seq_len, hidden] with positive dimensions");
    }

    if (tensor.stride(0) <= 0 || tensor.stride(1) <= 0) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires positive " +
                std::string(role) + " strides");
    }

    if (tensor.data() == nullptr ||
        reinterpret_cast<std::uintptr_t>(tensor.data()) % alignof(float) != 0) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires non-null float-aligned " +
                std::string(role) + " data");
    }
    return Status::Ok();
}

Status BuildAttentionF32ReferenceArgs(const KernelParamsBuildContext& context,
                                      void* params_buffer) noexcept {
    if (context.inputs.size() != 1 || context.outputs.size() != 1) {
        return Status::InvalidArgument(
                "CPU Attention requires one query input and one output");
    }

    AM_ASSIGN_OR_RETURN(const AttentionF32KernelMetadata metadata, ReadMetadata(context.attrs));
    const TensorView& query = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    AM_RETURN_IF_ERROR(ValidateActivation(query, "query"));
    AM_RETURN_IF_ERROR(ValidateActivation(output, "output"));
    if (query.shape() != output.shape()) {
        return Status::InvalidArgument(
                "CPU Attention requires output shape to match query shape");
    }

    int64_t hidden = 0;
    if (CheckOverflowMul(metadata.num_q_heads, metadata.head_dim, &hidden) ||
        query.dim(1) != hidden) {
        return Status::InvalidArgument(
                "CPU Attention query hidden width does not match frozen head geometry");
    }

    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint query_footprint,
                        BuildStridedAddressFootprint(query, "CPU Attention query"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint output_footprint,
                        BuildStridedAddressFootprint(output, "CPU Attention output"));
    AM_RETURN_IF_ERROR(ValidateLayoutInjectivity(
            kKernelName, output_footprint.injectivity(), "output"));
    AM_RETURN_IF_ERROR(ValidateStridedDisjoint(
            kKernelName, output_footprint, "output",
            query_footprint, "query"));

    ::new (params_buffer) AttentionF32KernelArgs{
            .query = query.data<float>(),
            .output = output.data<float>(),
            .seq_len = query.dim(0),
            .num_q_heads = metadata.num_q_heads,
            .num_kv_heads = metadata.num_kv_heads,
            .head_dim = metadata.head_dim,
            .query_row_stride = query.stride(0),
            .query_col_stride = query.stride(1),
            .output_row_stride = output.stride(0),
            .output_col_stride = output.stride(1),
            .scale = metadata.scale,
    };
    return Status::Ok();
}

StatusOr<StridedAddressFootprint> BuildKVFootprint(const KVCacheLayerStorageBinding& storage,
                                                   const std::byte* data,
                                                   size_t visible_end,
                                                   std::string_view role) noexcept {
    const size_t token_stride = storage.token_stride_bytes / sizeof(float);
    const size_t head_stride = storage.head_stride_bytes / sizeof(float);
    if (token_stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        head_stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        storage.num_kv_heads > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        storage.head_dim > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        visible_end > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " " + std::string(role) +
                " cache geometry exceeds int64 range");
    }

    const std::array<int64_t, 3> shape = {
            static_cast<int64_t>(storage.num_kv_heads),
            static_cast<int64_t>(visible_end),
            static_cast<int64_t>(storage.head_dim),
    };

    const std::array<int64_t, 3> strides = {
            static_cast<int64_t>(head_stride),
            static_cast<int64_t>(token_stride),
            1,
    };
    // Avoid formatting a diagnostic label for every successful KV read.
    const std::string_view cache_label = role == "key"
                                                 ? "CPU Attention key cache"
                                                 : "CPU Attention value cache";
    return BuildStridedAddressFootprint(data, shape, strides,
                                        sizeof(float), cache_label);
}

StatusOr<StridedAddressFootprint> BuildOutputFootprint(
        const AttentionF32KernelArgs& args) noexcept {
    int64_t hidden = 0;
    if (CheckOverflowMul(args.num_q_heads, args.head_dim, &hidden)) {
        return Status::InvalidArgument(
                "CPU Attention output hidden width overflow");
    }

    const std::array<int64_t, 2> shape = {args.seq_len, hidden};
    const std::array<int64_t, 2> strides = {
            args.output_row_stride,
            args.output_col_stride,
    };
    return BuildStridedAddressFootprint(
            args.output, shape, strides, sizeof(float), "CPU Attention output");
}

Status ValidateReadBinding(const AttentionF32KernelArgs& args,
                           const KVCacheReadBinding& read) noexcept {
    const KVCacheLayerStorageBinding& storage = read.storage;
    if (storage.dtype != DataType::Float32() || storage.key_data == nullptr ||
        storage.value_data == nullptr || storage.num_kv_heads == 0 ||
        storage.head_dim == 0 || storage.token_capacity == 0) {
        return Status::InvalidArgument(
                "CPU Attention has invalid KV cache storage binding");
    }

    if (reinterpret_cast<std::uintptr_t>(storage.key_data) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(storage.value_data) % alignof(float) != 0) {
        return Status::InvalidArgument(
                "CPU Attention requires float-aligned KV cache pointers");
    }

    if (storage.token_stride_bytes % sizeof(float) != 0 ||
        storage.head_stride_bytes % sizeof(float) != 0) {
        return Status::InvalidArgument(
                "CPU Attention requires float-aligned KV cache strides");
    }

    if (storage.num_kv_heads != static_cast<size_t>(args.num_kv_heads) ||
        storage.head_dim != static_cast<size_t>(args.head_dim)) {
        return Status::InvalidArgument(
                "CPU Attention KV cache geometry does not match head metadata");
    }

    if (read.committed_end > read.visible_end ||
        read.visible_end > storage.token_capacity ||
        read.query_begin > read.query_end ||
        read.query_end > read.visible_end) {
        return Status::InvalidArgument(
                "CPU Attention has invalid KV cache read frontiers");
    }

    if (read.query_end - read.query_begin != static_cast<size_t>(args.seq_len)) {
        return Status::InvalidArgument(
                "CPU Attention query interval does not match query sequence length");
    }

    size_t minimum_token_stride = 0;
    size_t minimum_head_stride = 0;
    if (CheckOverflowMul(storage.head_dim, sizeof(float), &minimum_token_stride) ||
        CheckOverflowMul(storage.token_capacity, storage.token_stride_bytes,
                         &minimum_head_stride) ||
        storage.token_stride_bytes < minimum_token_stride ||
        storage.head_stride_bytes < minimum_head_stride) {
        return Status::InvalidArgument(
                "CPU Attention KV cache strides do not cover its geometry");
    }

    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint key_footprint,
                        BuildKVFootprint(storage, storage.key_data, read.visible_end, "key"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint value_footprint,
                        BuildKVFootprint(storage, storage.value_data, read.visible_end, "value"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint output_footprint,
                        BuildOutputFootprint(args));
    AM_RETURN_IF_ERROR(ValidateStridedDisjoint(
            kKernelName, key_footprint, "key cache",
            value_footprint, "value cache"));
    AM_RETURN_IF_ERROR(ValidateStridedDisjoint(
            kKernelName, output_footprint, "output",
            key_footprint, "key cache"));
    return ValidateStridedDisjoint(
            kKernelName, output_footprint, "output",
            value_footprint, "value cache");
}

Status BuildAttentionF32Metadata(const OpParams& params,
                                 std::vector<std::byte>& attrs) {
    const auto* attention = std::get_if<AttentionParams>(&params);
    if (attention == nullptr) {
        return Status::InvalidArgument("CPU Attention kernel requires AttentionParams");
    }

    const AttentionF32KernelMetadata metadata{
            .num_q_heads = attention->num_q_heads,
            .num_kv_heads = attention->num_kv_heads,
            .head_dim = attention->head_dim,
            .scale = attention->head_dim > 0
                             ? 1.0F / std::sqrt(static_cast<float>(attention->head_dim))
                             : 0.0F,
    };

    AM_RETURN_IF_ERROR(ValidateMetadata(metadata));
    const auto bytes = std::as_bytes(std::span{&metadata, size_t{1}});
    attrs.assign(bytes.begin(), bytes.end());
    return Status::Ok();
}

Status AttentionF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* const args = static_cast<const AttentionF32KernelArgs*>(ctx.kernel_params);
    if (args == nullptr || ctx.kv_read == nullptr) {
        return Status::FailedPrecondition(
                "CPU Attention requires prepared args and a KV read binding");
    }
    AM_RETURN_IF_ERROR(ValidateReadBinding(*args, *ctx.kv_read));
    return RunAttentionF32Reference(*args, *ctx.kv_read);
}

} // namespace

static_assert(std::is_trivially_copyable_v<AttentionF32KernelMetadata>);
static_assert(std::is_trivially_destructible_v<AttentionF32KernelArgs>);
static_assert(alignof(AttentionF32KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(AttentionF32KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(
        CpuAttentionF32Reference,
        KernelDescriptor{
                .op_type = OpType::kAttention,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &AttentionF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(AttentionF32KernelArgs),
                .params_builder = &BuildAttentionF32ReferenceArgs,
                .metadata_builder = &BuildAttentionF32Metadata,
                .name = "cpu::attention_f32_reference"})

} // namespace aethermind::cpu::detail
