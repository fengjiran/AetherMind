#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "kvcache_update_internal.h"
#include "utils/overflow_check.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

constexpr std::string_view kKernelName = "CPU KVCacheUpdate";

Status ValidateActivation(const TensorView& tensor, std::string_view role) noexcept {
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
                std::string(role) + " shape [T, hidden] with positive dimensions");
    }

    if (tensor.stride(0) <= 0 || tensor.stride(1) <= 0) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires positive " +
                std::string(role) + " strides");
    }

    if (tensor.data() == nullptr) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires non-null " +
                std::string(role) + " data");
    }

    if (reinterpret_cast<std::uintptr_t>(tensor.data()) % alignof(float) != 0) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires float-aligned " +
                std::string(role) + " data");
    }
    return Status::Ok();
}

Status BuildKVCacheUpdateF32ReferenceArgs(const KernelParamsBuildContext& context,
                                          void* params_buffer) noexcept {
    if (context.inputs.size() != 2 || !context.outputs.empty()) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate requires 2 inputs and 0 TensorView outputs");
    }

    const TensorView& key = context.inputs[0];
    const TensorView& value = context.inputs[1];
    AM_RETURN_IF_ERROR(ValidateActivation(key, "key"));
    AM_RETURN_IF_ERROR(ValidateActivation(value, "value"));
    if (key.shape() != value.shape()) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate requires key and value shapes to match");
    }

    AM_RETURN_IF_ERROR(BuildStridedAddressFootprint(key, "CPU KVCacheUpdate key"));
    AM_RETURN_IF_ERROR(BuildStridedAddressFootprint(value, "CPU KVCacheUpdate value"));

    ::new (params_buffer) KVCacheUpdateF32KernelArgs{
            .key_data = static_cast<const float*>(key.data()),
            .value_data = static_cast<const float*>(value.data()),
            .seq_len = key.dim(0),
            .hidden = key.dim(1),
            .key_row_stride = key.stride(0),
            .key_col_stride = key.stride(1),
            .value_row_stride = value.stride(0),
            .value_col_stride = value.stride(1),
    };
    return Status::Ok();
}

StatusOr<StridedAddressFootprint> BuildTargetFootprint(
        const KVCacheLayerStorageBinding& storage,
        std::byte* data,
        size_t begin,
        size_t end,
        std::string_view role) noexcept {
    if (data == nullptr || end <= begin) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " has invalid " +
                std::string(role) + " cache geometry");
    }

    if (storage.token_stride_bytes % sizeof(float) != 0 ||
        storage.head_stride_bytes % sizeof(float) != 0) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " requires float-aligned " +
                std::string(role) + " cache strides");
    }

    const size_t token_stride = storage.token_stride_bytes / sizeof(float);
    const size_t head_stride = storage.head_stride_bytes / sizeof(float);
    if (token_stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        head_stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        storage.num_kv_heads > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        storage.head_dim > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        end - begin > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::InvalidArgument(
                std::string(kKernelName) + " " +
                std::string(role) + " cache geometry exceeds int64 range");
    }

    size_t begin_bytes = 0;
    if (CheckOverflowMul(begin, storage.token_stride_bytes, &begin_bytes)) {
        return Status::InvalidArgument(std::string(kKernelName) + " " +
                                       std::string(role) + " cache offset overflow");
    }

    const std::array<int64_t, 3> shape = {static_cast<int64_t>(storage.num_kv_heads),
                                          static_cast<int64_t>(end - begin),
                                          static_cast<int64_t>(storage.head_dim)};
    const std::array<int64_t, 3> strides = {static_cast<int64_t>(head_stride),
                                            static_cast<int64_t>(token_stride), 1};
    // Keep the diagnostic label allocation-free on the successful append path.
    const std::string_view cache_label = role == "key"
                                                 ? "CPU KVCacheUpdate key cache"
                                                 : "CPU KVCacheUpdate value cache";
    return BuildStridedAddressFootprint(
            data + begin_bytes, shape, strides, sizeof(float), cache_label);
}

Status ValidateNoOverlap(const KVCacheUpdateF32KernelArgs& args,
                         const KVCacheAppendBinding& append) noexcept {
    const std::array<int64_t, 2> src_shape = {args.seq_len, args.hidden};
    const std::array<int64_t, 2> key_strides = {args.key_row_stride, args.key_col_stride};
    const std::array<int64_t, 2> value_strides = {args.value_row_stride, args.value_col_stride};
    AM_ASSIGN_OR_RETURN(
            const StridedAddressFootprint key_source,
            BuildStridedAddressFootprint(
                    args.key_data, src_shape, key_strides,
                    sizeof(float), "CPU KVCacheUpdate key activation"));
    AM_ASSIGN_OR_RETURN(
            const StridedAddressFootprint value_source,
            BuildStridedAddressFootprint(
                    args.value_data, src_shape, value_strides,
                    sizeof(float), "CPU KVCacheUpdate value activation"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint key_target,
                        BuildTargetFootprint(append.storage, append.storage.key_data,
                                             append.begin, append.end, "key"));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint value_target,
                        BuildTargetFootprint(append.storage, append.storage.value_data,
                                             append.begin, append.end, "value"));

    if (ByteRangesOverlap(key_target.envelope(), value_target.envelope())) {
        return Status::InvalidArgument("CPU KVCacheUpdate cannot prove key "
                                       "and value cache destinations are disjoint");
    }

    const auto require_disjoint = [](const StridedAddressFootprint& source,
                                     std::string_view source_name,
                                     const StridedAddressFootprint& destination,
                                     std::string_view destination_name) -> Status {
        if (ByteRangesOverlap(source.envelope(), destination.envelope())) {
            return Status::InvalidArgument(
                    std::string(kKernelName) + " cannot prove " + std::string(source_name) +
                    " activation is disjoint from " + std::string(destination_name) +
                    " cache destination");
        }
        return Status::Ok();
    };

    AM_RETURN_IF_ERROR(require_disjoint(key_source, "key",
                                        key_target, "key"));
    AM_RETURN_IF_ERROR(require_disjoint(key_source, "key",
                                        value_target, "value"));
    AM_RETURN_IF_ERROR(require_disjoint(value_source, "value",
                                        key_target, "key"));
    return require_disjoint(value_source, "value",
                            value_target, "value");
}

Status ValidateAppend(const KVCacheUpdateF32KernelArgs& args,
                      const KVCacheAppendBinding& append) noexcept {
    const KVCacheLayerStorageBinding& storage = append.storage;
    if (storage.dtype != DataType::Float32()) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate requires float32 KV cache storage");
    }

    if (storage.key_data == nullptr || storage.value_data == nullptr ||
        storage.num_kv_heads == 0 || storage.head_dim == 0 || storage.token_capacity == 0) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate has invalid KV cache storage binding");
    }

    if (reinterpret_cast<std::uintptr_t>(storage.key_data) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(storage.value_data) % alignof(float) != 0) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate requires float-aligned KV cache pointers");
    }

    if (append.end <= append.begin || append.end > storage.token_capacity) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate append range is outside the cache binding");
    }

    if (append.end - append.begin != static_cast<size_t>(args.seq_len)) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate append length does not match key/value sequence length");
    }

    size_t hidden = 0;
    if (CheckOverflowMul(storage.num_kv_heads, storage.head_dim, &hidden) ||
        hidden > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        args.hidden != static_cast<int64_t>(hidden)) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate hidden size does not match KV cache geometry");
    }

    size_t minimum_token_stride = 0;
    size_t minimum_head_stride = 0;
    if (CheckOverflowMul(storage.head_dim, sizeof(float), &minimum_token_stride) ||
        CheckOverflowMul(storage.token_capacity, storage.token_stride_bytes, &minimum_head_stride) ||
        storage.token_stride_bytes < minimum_token_stride ||
        storage.head_stride_bytes < minimum_head_stride) {
        return Status::InvalidArgument(
                "CPU KVCacheUpdate cache strides do not cover its geometry");
    }
    return ValidateNoOverlap(args, append);
}

Status KVCacheUpdateF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* const args = static_cast<const KVCacheUpdateF32KernelArgs*>(ctx.kernel_params);
    if (args == nullptr || ctx.kv_append == nullptr) {
        return Status::FailedPrecondition(
                "CPU KVCacheUpdate requires prepared args and an append binding");
    }
    AM_RETURN_IF_ERROR(ValidateAppend(*args, *ctx.kv_append));
    return RunKVCacheUpdateF32Reference(*args, *ctx.kv_append);
}

} // namespace

static_assert(std::is_trivially_destructible_v<KVCacheUpdateF32KernelArgs>);
static_assert(alignof(KVCacheUpdateF32KernelArgs) <= alignof(std::max_align_t));
static_assert(sizeof(KVCacheUpdateF32KernelArgs) <= kMaxKernelParamsSize);

AM_REGISTER_KERNEL(CpuKVCacheUpdateF32Reference,
                   KernelDescriptor{
                           .op_type = OpType::kKVCacheUpdate,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &KVCacheUpdateF32ReferenceEntry,
                           .priority = 10,
                           .params_size = sizeof(KVCacheUpdateF32KernelArgs),
                           .params_builder = &BuildKVCacheUpdateF32ReferenceArgs,
                           .name = "cpu::kvcache_update_f32_reference",
                   })

} // namespace aethermind::cpu::detail
