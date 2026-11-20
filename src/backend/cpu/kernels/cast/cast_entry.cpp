#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/dtypes/bf16_conversion.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/fp16_conversion.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <cstring>
#include <new>
#include <type_traits>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

struct CastArgs {
    const std::byte* source{};
    std::byte* destination{};
    int64_t count{};
    int32_t rank{};
    bool contiguous{};
    std::array<int64_t, ShapeAndStride::kMaxRank> shape{};
    std::array<int64_t, ShapeAndStride::kMaxRank> source_strides{};
    std::array<int64_t, ShapeAndStride::kMaxRank> destination_strides{};
};

template<typename To>
Status BuildMetadata(const OpParams& params, std::vector<std::byte>&) {
    const auto* cast = std::get_if<CastParams>(&params);
    if (cast == nullptr || cast->target_dtype != DataType::Make<To>()) {
        return Status::InvalidArgument("CPU Cast target_dtype must match the resolved kernel");
    }
    return Status::Ok();
}

template<typename From, typename To>
Status Bind(const KernelParamsBuildContext& context, void* storage) noexcept {
    if (context.inputs.size() != 1 || context.outputs.size() != 1) {
        return Status::InvalidArgument("CPU Cast requires one input and one output");
    }
    const auto& source = context.inputs.front();
    const auto& destination = context.outputs.front();
    if (!source.is_valid() || !destination.is_valid() ||
        source.dtype() != DataType::Make<From>() || destination.dtype() != DataType::Make<To>() ||
        source.shape() != destination.shape()) {
        return Status::InvalidArgument("CPU Cast requires valid views with matching shapes and resolved dtypes");
    }
    AM_ASSIGN_OR_RETURN(const auto input, BuildStridedAddressFootprint(source, "CPU Cast input"));
    AM_ASSIGN_OR_RETURN(const auto output, BuildStridedAddressFootprint(destination, "CPU Cast output"));
    if (output.is_empty()) {
        ::new (storage) CastArgs{};
        return Status::Ok();
    }
    AM_RETURN_IF_ERROR(ValidateLayoutInjectivity("CPU Cast", output.injectivity(), "output"));
    AM_RETURN_IF_ERROR(ValidateStridedDisjoint("CPU Cast", output, "output", input, "input"));
    CastArgs args{
            .source = static_cast<const std::byte*>(source.data()),
            .destination = static_cast<std::byte*>(destination.data()),
            .count = source.numel(),
            .rank = source.rank(),
            .contiguous = source.is_contiguous() && destination.is_contiguous(),
    };
    for (int32_t i = 0; i < args.rank; ++i) {
        args.shape[i] = source.dim(i);
        args.source_strides[i] = source.stride(i);
        args.destination_strides[i] = destination.stride(i);
    }
    ::new (storage) CastArgs(args);
    return Status::Ok();
}

template<typename From, typename To>
void Convert(const std::byte* source, std::byte* destination, size_t count) noexcept {
    if constexpr (std::is_same_v<From, float>) {
        if constexpr (std::is_same_v<To, Half>) {
            ConvertFp32ToFp16Scalar(source, destination, count);
        } else {
            ConvertFp32ToBf16Scalar(source, destination, count);
        }
    } else if constexpr (std::is_same_v<To, float>) {
        if constexpr (std::is_same_v<From, Half>) {
            ConvertFp16ToFp32Scalar(source, destination, count);
        } else {
            ConvertBf16ToFp32Scalar(source, destination, count);
        }
    } else {
        // Every finite F16/BF16 value is exactly representable in F32.
        // Widening and narrowing also reuse the existing NaN bit contracts.
        for (size_t i = 0; i < count; ++i) {
            From value{};
            std::memcpy(&value, source + i * sizeof(From), sizeof(From));
            const To converted(static_cast<float>(value));
            std::memcpy(destination + i * sizeof(To), &converted, sizeof(To));
        }
    }
}

template<typename From, typename To>
Status Run(const KernelContext& context) noexcept {
    const auto& args = *static_cast<const CastArgs*>(context.kernel_params);
    if (args.count == 0) {
        return Status::Ok();
    }
    if (args.contiguous) {
        Convert<From, To>(args.source, args.destination, static_cast<size_t>(args.count));
        return Status::Ok();
    }
    for (int64_t i = 0; i < args.count; ++i) {
        int64_t remaining = i;
        size_t source_offset = 0;
        size_t destination_offset = 0;
        for (int32_t axis = args.rank - 1; axis >= 0; --axis) {
            const int64_t coord = remaining % args.shape[axis];
            remaining /= args.shape[axis];
            source_offset += static_cast<size_t>(coord * args.source_strides[axis]);
            destination_offset += static_cast<size_t>(coord * args.destination_strides[axis]);
        }
        Convert<From, To>(args.source + source_offset * sizeof(From),
                          args.destination + destination_offset * sizeof(To), 1);
    }
    return Status::Ok();
}

template<typename From, typename To>
KernelDef MakeCast(std::string_view name) {
    return {
            .op_type = OpType::kCast,
            .selector = {.device_type = DeviceType::kCPU,
                         .act_dtype = DataType::Make<From>(),
                         .output_dtype = DataType::Make<To>(),
                         .weight_dtype = DataType::Make<From>(),
                         .weight_format = WeightFormat::kPlain,
                         .phase = ExecPhase::kBoth},
            .kernel_func = &Run<From, To>,
            .priority = 10,
            .params_size = sizeof(CastArgs),
            .params_builder = &Bind<From, To>,
            .metadata_builder = &BuildMetadata<To>,
            .name = name,
    };
}

static_assert(std::is_trivially_destructible_v<CastArgs>);
static_assert(sizeof(CastArgs) <= kMaxKernelParamsSize);
static_assert(alignof(CastArgs) <= alignof(std::max_align_t));

} // namespace

AM_REGISTER_KERNEL(CpuCastF32ToF16Reference, (MakeCast<float, Half>("cpu::cast_f32_to_f16_reference")))
AM_REGISTER_KERNEL(CpuCastF16ToF32Reference, (MakeCast<Half, float>("cpu::cast_f16_to_f32_reference")))
AM_REGISTER_KERNEL(CpuCastF32ToBf16Reference, (MakeCast<float, BFloat16>("cpu::cast_f32_to_bf16_reference")))
AM_REGISTER_KERNEL(CpuCastBf16ToF32Reference, (MakeCast<BFloat16, float>("cpu::cast_bf16_to_f32_reference")))
AM_REGISTER_KERNEL(CpuCastF16ToBf16Reference, (MakeCast<Half, BFloat16>("cpu::cast_f16_to_bf16_reference")))
AM_REGISTER_KERNEL(CpuCastBf16ToF16Reference, (MakeCast<BFloat16, Half>("cpu::cast_bf16_to_f16_reference")))

} // namespace aethermind::cpu::detail
