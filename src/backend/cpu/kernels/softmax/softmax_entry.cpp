#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

struct SoftmaxArgs {
    const std::byte* source{};
    std::byte* destination{};
    size_t slices{};
    size_t length{};
    int32_t rank{};
    int32_t axis{};
    std::array<int64_t, ShapeAndStride::kMaxRank> shape{};
    std::array<int64_t, ShapeAndStride::kMaxRank> input_strides{};
    std::array<int64_t, ShapeAndStride::kMaxRank> output_strides{};
};

Status BuildMetadata(const OpParams& params, std::vector<std::byte>& attrs) {
    const auto* softmax = std::get_if<SoftmaxParams>(&params);
    if (softmax == nullptr) {
        return Status::InvalidArgument("CPU Softmax requires SoftmaxParams");
    }
    const auto bytes = std::as_bytes(std::span(&softmax->axis, 1));
    attrs.assign(bytes.begin(), bytes.end());
    return Status::Ok();
}

template<typename Element>
Status Bind(const KernelParamsBuildContext& context, void* storage) noexcept {
    if (context.inputs.size() != 1 || context.outputs.size() != 1 ||
        context.attrs.size() != sizeof(int64_t)) {
        return Status::InvalidArgument("CPU Softmax requires one input, one output and an axis");
    }
    const auto& input = context.inputs.front();
    const auto& output = context.outputs.front();
    const auto dtype = DataType::Make<Element>();
    if (!input.is_valid() || !output.is_valid() || input.dtype() != dtype ||
        output.dtype() != dtype || input.shape() != output.shape()) {
        return Status::InvalidArgument("CPU Softmax requires valid matching shapes and dtypes");
    }
    int64_t axis{};
    std::memcpy(&axis, context.attrs.data(), sizeof(axis));
    if (axis < 0) {
        axis += input.rank();
    }
    if (axis < 0 || axis >= input.rank()) {
        return Status::InvalidArgument("CPU Softmax axis is out of range");
    }

    AM_ASSIGN_OR_RETURN(const auto source, BuildStridedAddressFootprint(input, "CPU Softmax input"));
    AM_ASSIGN_OR_RETURN(const auto destination, BuildStridedAddressFootprint(output, "CPU Softmax output"));
    if (destination.is_empty()) {
        ::new (storage) SoftmaxArgs{};
        return Status::Ok();
    }
    AM_RETURN_IF_ERROR(ValidateLayoutInjectivity("CPU Softmax", destination.injectivity(), "output"));
    if (!HaveIdenticalViewMapping(input, output)) {
        AM_RETURN_IF_ERROR(ValidateStridedDisjoint("CPU Softmax", destination, "output", source, "input"));
    }
    const auto length = static_cast<size_t>(input.dim(static_cast<int32_t>(axis)));
    SoftmaxArgs args{.source = static_cast<const std::byte*>(input.data()),
                     .destination = static_cast<std::byte*>(output.data()),
                     .slices = static_cast<size_t>(input.numel()) / length,
                     .length = length,
                     .rank = input.rank(),
                     .axis = static_cast<int32_t>(axis)};
    for (int32_t i = 0; i < args.rank; ++i) {
        args.shape[i] = input.dim(i);
        args.input_strides[i] = input.stride(i);
        args.output_strides[i] = output.stride(i);
    }
    ::new (storage) SoftmaxArgs(args);
    return Status::Ok();
}

template<typename Element>
double Read(const std::byte* source, size_t index) noexcept {
    Element value{};
    std::memcpy(&value, source + index * sizeof(Element), sizeof(Element));
    return static_cast<float>(value);
}

template<typename Element>
Status Run(const KernelContext& context) noexcept {
    const auto& args = *static_cast<const SoftmaxArgs*>(context.kernel_params);
    const auto source_stride = static_cast<size_t>(args.input_strides[args.axis]);
    const auto destination_stride = static_cast<size_t>(args.output_strides[args.axis]);
    for (size_t slice = 0; slice < args.slices; ++slice) {
        size_t remaining = slice;
        size_t from = 0;
        size_t to = 0;
        for (int32_t axis = args.rank - 1; axis >= 0; --axis) {
            if (axis == args.axis) {
                continue;
            }
            const auto dim = static_cast<size_t>(args.shape[axis]);
            const auto coord = remaining % dim;
            remaining /= dim;
            from += coord * static_cast<size_t>(args.input_strides[axis]);
            to += coord * static_cast<size_t>(args.output_strides[axis]);
        }
        const auto* source = args.source + from * sizeof(Element);
        auto* destination = args.destination + to * sizeof(Element);
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < args.length; ++i) {
            maximum = std::max(maximum, Read<Element>(source, i * source_stride));
        }
        double sum = 0.0;
        for (size_t i = 0; i < args.length; ++i) {
            sum += std::exp(Read<Element>(source, i * source_stride) - maximum);
        }
        // All input reads contributing to shared state precede stores, so
        // an exact in-place binding cannot change the reduction.
        for (size_t i = 0; i < args.length; ++i) {
            const auto probability = static_cast<float>(
                    std::exp(Read<Element>(source, i * source_stride) - maximum) / sum);
            const Element value(probability);
            std::memcpy(destination + i * destination_stride * sizeof(Element), &value, sizeof(Element));
        }
    }
    return Status::Ok();
}

template<typename Element>
KernelDef MakeSoftmax(std::string_view name) {
    const auto dtype = DataType::Make<Element>();
    return {.op_type = OpType::kSoftmax,
            .selector = {.device_type = DeviceType::kCPU, .act_dtype = dtype, .output_dtype = dtype, .weight_dtype = dtype, .weight_format = WeightFormat::kPlain, .phase = ExecPhase::kBoth},
            .kernel_func = &Run<Element>,
            .priority = 10,
            .params_size = sizeof(SoftmaxArgs),
            .params_builder = &Bind<Element>,
            .metadata_builder = &BuildMetadata,
            .name = name};
}
static_assert(std::is_trivially_destructible_v<SoftmaxArgs> && sizeof(SoftmaxArgs) <= kMaxKernelParamsSize);
static_assert(alignof(SoftmaxArgs) <= alignof(std::max_align_t));

} // namespace

AM_REGISTER_KERNEL(CpuSoftmaxF32, MakeSoftmax<float>("cpu::softmax_f32_reference"))
AM_REGISTER_KERNEL(CpuSoftmaxF16, MakeSoftmax<Half>("cpu::softmax_f16_reference"))
AM_REGISTER_KERNEL(CpuSoftmaxBf16, MakeSoftmax<BFloat16>("cpu::softmax_bf16_reference"))

} // namespace aethermind::cpu::detail
