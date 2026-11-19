#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/dtypes/fp16_conversion.h"
#include "aethermind/operators/op_params.h"

#include <array>
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

template<bool Narrow>
DataType SourceType() {
    return Narrow ? DataType::Float32() : DataType::Float(16);
}

template<bool Narrow>
DataType DestinationType() {
    return Narrow ? DataType::Float(16) : DataType::Float32();
}

template<bool Narrow>
Status BuildMetadata(const OpParams& params, std::vector<std::byte>&) {
    const auto* cast = std::get_if<CastParams>(&params);
    if (cast == nullptr || cast->target_dtype != DestinationType<Narrow>()) {
        return Status::InvalidArgument("CPU Cast target_dtype must match the resolved kernel");
    }
    return Status::Ok();
}

template<bool Narrow>
Status Bind(const KernelParamsBuildContext& context, void* storage) noexcept {
    if (context.inputs.size() != 1 || context.outputs.size() != 1) {
        return Status::InvalidArgument("CPU Cast requires one input and one output");
    }
    const auto& source = context.inputs.front();
    const auto& destination = context.outputs.front();
    if (!source.is_valid() || !destination.is_valid() ||
        source.dtype() != SourceType<Narrow>() || destination.dtype() != DestinationType<Narrow>() ||
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

template<bool Narrow>
Status Run(const KernelContext& context) noexcept {
    const auto& args = *static_cast<const CastArgs*>(context.kernel_params);
    if (args.count == 0) {
        return Status::Ok();
    }
    constexpr auto convert = Narrow ? ConvertFp32ToFp16Scalar : ConvertFp16ToFp32Scalar;
    if (args.contiguous) {
        convert(args.source, args.destination, static_cast<size_t>(args.count));
        return Status::Ok();
    }
    constexpr size_t source_bytes = Narrow ? sizeof(float) : sizeof(uint16_t);
    constexpr size_t destination_bytes = Narrow ? sizeof(uint16_t) : sizeof(float);
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
        convert(args.source + source_offset * source_bytes,
                args.destination + destination_offset * destination_bytes, 1);
    }
    return Status::Ok();
}

template<bool Narrow>
KernelDef MakeCast(std::string_view name) {
    return {
            .op_type = OpType::kCast,
            .selector = {.device_type = DeviceType::kCPU,
                         .act_dtype = SourceType<Narrow>(),
                         .output_dtype = DestinationType<Narrow>(),
                         .weight_dtype = SourceType<Narrow>(),
                         .weight_format = WeightFormat::kPlain,
                         .phase = ExecPhase::kBoth},
            .kernel_func = &Run<Narrow>,
            .priority = 10,
            .params_size = sizeof(CastArgs),
            .params_builder = &Bind<Narrow>,
            .metadata_builder = &BuildMetadata<Narrow>,
            .name = name,
    };
}

static_assert(std::is_trivially_destructible_v<CastArgs>);
static_assert(sizeof(CastArgs) <= kMaxKernelParamsSize);
static_assert(alignof(CastArgs) <= alignof(std::max_align_t));

} // namespace

AM_REGISTER_KERNEL(CpuCastF32ToF16Reference, MakeCast<true>("cpu::cast_f32_to_f16_reference"))
AM_REGISTER_KERNEL(CpuCastF16ToF32Reference, MakeCast<false>("cpu::cast_f16_to_f32_reference"))

} // namespace aethermind::cpu::detail
