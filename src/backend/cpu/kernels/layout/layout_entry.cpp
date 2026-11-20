#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <cstring>
#include <new>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

constexpr size_t kRank = ShapeAndStride::kMaxRank;

struct CopyArgs {
    const std::byte* source{};
    std::byte* destination{};
    size_t count{};
    size_t item_bytes{};
    int32_t rank{};
    bool flat{};
    std::array<int64_t, kRank> shape{};
    std::array<int64_t, kRank> source_strides{};
    std::array<int64_t, kRank> destination_strides{};
};

struct ReshapeMetadata {
    enum class Kind : uint8_t { kLiteral,
                                kInput,
                                kInfer };
    std::array<Kind, kRank> kinds{};
    std::array<int64_t, kRank> values{};
    size_t rank{};
};

template<typename T>
StatusOr<T> ReadMetadata(std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(T)) {
        return Status::InvalidArgument("CPU layout operator has invalid prepared metadata");
    }
    T metadata{};
    std::memcpy(&metadata, attrs.data(), sizeof(T));
    return metadata;
}

template<OpType Op>
Status BuildMetadata(const OpParams& params, std::vector<std::byte>& attrs) {
    if constexpr (Op == OpType::kReshape) {
        const auto* reshape = std::get_if<ReshapeParams>(&params);
        if (reshape == nullptr || reshape->target_shape.size() > kRank) {
            return Status::InvalidArgument("CPU Reshape requires a supported target rank");
        }
        ReshapeMetadata metadata{};
        metadata.rank = reshape->target_shape.size();
        bool seen_infer = false;
        for (size_t i = 0; i < metadata.rank; ++i) {
            const auto& dim = reshape->target_shape[i];
            if (const auto* literal = std::get_if<ReshapeLiteralDim>(&dim)) {
                if (literal->value < 0) {
                    return Status::InvalidArgument("CPU Reshape requires non-negative literal dimensions");
                }
                metadata.kinds[i] = ReshapeMetadata::Kind::kLiteral;
                metadata.values[i] = literal->value;
            } else if (const auto* input = std::get_if<ReshapeInputDim>(&dim)) {
                if (input->axis >= kRank) {
                    return Status::InvalidArgument("CPU Reshape input axis exceeds maximum rank");
                }
                metadata.kinds[i] = ReshapeMetadata::Kind::kInput;
                metadata.values[i] = input->axis;
            } else {
                if (seen_infer) {
                    return Status::InvalidArgument("CPU Reshape accepts at most one inferred dimension");
                }
                seen_infer = true;
                metadata.kinds[i] = ReshapeMetadata::Kind::kInfer;
            }
        }
        const auto bytes = std::as_bytes(std::span(&metadata, 1));
        attrs.assign(bytes.begin(), bytes.end());
    } else if constexpr (Op == OpType::kPermute) {
        const auto* permute = std::get_if<PermuteParams>(&params);
        if (permute == nullptr || permute->permutation.size() > kRank) {
            return Status::InvalidArgument("CPU Permute requires a supported permutation");
        }
        std::array<bool, kRank> seen{};
        for (const auto axis: permute->permutation) {
            if (axis >= permute->permutation.size() || seen[axis]) {
                return Status::InvalidArgument("CPU Permute requires a bijection over the input axes");
            }
            seen[axis] = true;
        }
        const auto bytes = std::as_bytes(std::span(permute->permutation));
        attrs.assign(bytes.begin(), bytes.end());
    } else {
        if (!std::holds_alternative<ReorderParams>(params)) {
            return Status::InvalidArgument("CPU Reorder requires ReorderParams");
        }
    }
    return Status::Ok();
}

Status ValidateReshape(const TensorView& source, const MutableTensorView& destination,
                       std::span<const std::byte> attrs) noexcept {
    AM_ASSIGN_OR_RETURN(const auto target, ReadMetadata<ReshapeMetadata>(attrs));
    if (target.rank != static_cast<size_t>(destination.rank()) ||
        source.numel() != destination.numel()) {
        return Status::InvalidArgument("CPU Reshape requires the target rank and equal volumes");
    }
    bool has_infer = false;
    bool target_axis_is_zero = false;
    for (size_t i = 0; i < target.rank; ++i) {
        const auto dim = destination.dim(static_cast<int32_t>(i));
        switch (target.kinds[i]) {
            case ReshapeMetadata::Kind::kLiteral:
                if (target.values[i] != dim) {
                    return Status::InvalidArgument("CPU Reshape output does not match a literal target dimension");
                }
                break;
            case ReshapeMetadata::Kind::kInput:
                if (target.values[i] < 0 || target.values[i] >= source.rank() ||
                    source.dim(static_cast<int32_t>(target.values[i])) != dim) {
                    return Status::InvalidArgument("CPU Reshape output does not match a referenced input axis");
                }
                break;
            case ReshapeMetadata::Kind::kInfer:
                if (has_infer) {
                    return Status::InvalidArgument("CPU Reshape accepts at most one inferred dimension");
                }
                has_infer = true;
                continue;
            default:
                return Status::InvalidArgument("CPU Reshape has an unknown target dimension kind");
        }
        target_axis_is_zero |= dim == 0;
    }
    if (has_infer && target_axis_is_zero) {
        return Status::InvalidArgument("CPU Reshape cannot infer an axis from an ambiguous zero-volume target");
    }
    if (source.numel() != 0 && (!source.is_contiguous() || !destination.is_contiguous())) {
        return Status::Unimplemented("CPU Reshape copy requires contiguous input and output");
    }
    return Status::Ok();
}

template<OpType Op, typename Element>
Status Bind(const KernelParamsBuildContext& context, void* storage) noexcept {
    if (context.inputs.size() != 1 || context.outputs.size() != 1) {
        return Status::InvalidArgument("CPU layout operator requires one input and one output");
    }
    const auto& source = context.inputs.front();
    const auto& destination = context.outputs.front();
    const auto dtype = DataType::Make<Element>();
    if (!source.is_valid() || !destination.is_valid() ||
        source.dtype() != dtype || destination.dtype() != dtype) {
        return Status::InvalidArgument("CPU layout operator requires valid same-dtype views");
    }

    CopyArgs args{.source = static_cast<const std::byte*>(source.data()),
                  .destination = static_cast<std::byte*>(destination.data()),
                  .count = static_cast<size_t>(destination.numel()),
                  .item_bytes = sizeof(Element),
                  .rank = destination.rank(),
                  .flat = source.is_contiguous() && destination.is_contiguous()};
    if constexpr (Op == OpType::kReshape) {
        AM_RETURN_IF_ERROR(ValidateReshape(source, destination, context.attrs));
        args.flat = true;
    } else if constexpr (Op == OpType::kReorder) {
        if (source.shape() != destination.shape()) {
            return Status::InvalidArgument("CPU Reorder preserves the input shape");
        }
        if (args.count != 0 && !destination.is_contiguous()) {
            return Status::InvalidArgument("CPU Reorder requires a contiguous output");
        }
    } else {
        if (source.rank() != destination.rank() ||
            context.attrs.size() != static_cast<size_t>(args.rank) * sizeof(uint32_t)) {
            return Status::InvalidArgument("CPU Permute rank must match the prepared permutation");
        }
    }

    if constexpr (Op != OpType::kReshape) {
        std::array<bool, kRank> seen{};
        for (int32_t i = 0; i < args.rank; ++i) {
            auto axis = static_cast<uint32_t>(i);
            if constexpr (Op == OpType::kPermute) {
                std::memcpy(&axis, context.attrs.data() + i * sizeof(axis), sizeof(axis));
                if (axis >= static_cast<uint32_t>(args.rank) || seen[axis] ||
                    source.dim(static_cast<int32_t>(axis)) != destination.dim(i)) {
                    return Status::InvalidArgument("CPU Permute views must match the prepared axis mapping");
                }
                seen[axis] = true;
                args.flat &= axis == static_cast<uint32_t>(i);
            }
            args.shape[i] = destination.dim(i);
            args.source_strides[i] = source.stride(static_cast<int32_t>(axis));
            args.destination_strides[i] = destination.stride(i);
        }
    }

    AM_ASSIGN_OR_RETURN(const auto input, BuildStridedAddressFootprint(source, "CPU layout input"));
    AM_ASSIGN_OR_RETURN(const auto output, BuildStridedAddressFootprint(destination, "CPU layout output"));
    if (!output.is_empty()) {
        AM_RETURN_IF_ERROR(ValidateLayoutInjectivity("CPU layout", output.injectivity(), "output"));
        AM_RETURN_IF_ERROR(ValidateStridedDisjoint("CPU layout", output, "output", input, "input"));
    }
    ::new (storage) CopyArgs(args);
    return Status::Ok();
}

Status Copy(const KernelContext& context) noexcept {
    const auto& args = *static_cast<const CopyArgs*>(context.kernel_params);
    if (args.count == 0) {
        return Status::Ok();
    }
    if (args.flat) {
        std::memcpy(args.destination, args.source, args.count * args.item_bytes);
        return Status::Ok();
    }
    for (size_t i = 0; i < args.count; ++i) {
        size_t remaining = i;
        size_t from = 0;
        size_t to = 0;
        for (int32_t axis = args.rank - 1; axis >= 0; --axis) {
            const auto dim = static_cast<size_t>(args.shape[axis]);
            const auto coord = remaining % dim;
            remaining /= dim;
            from += coord * static_cast<size_t>(args.source_strides[axis]);
            to += coord * static_cast<size_t>(args.destination_strides[axis]);
        }
        std::memcpy(args.destination + to * args.item_bytes,
                    args.source + from * args.item_bytes, args.item_bytes);
    }
    return Status::Ok();
}

template<OpType Op, typename Element>
KernelDef MakeLayout(std::string_view name) {
    const auto dtype = DataType::Make<Element>();
    return {.op_type = Op,
            .selector = {.device_type = DeviceType::kCPU, .act_dtype = dtype, .output_dtype = dtype, .weight_dtype = dtype, .weight_format = WeightFormat::kPlain, .phase = ExecPhase::kBoth},
            .kernel_func = &Copy,
            .priority = 10,
            .params_size = sizeof(CopyArgs),
            .params_builder = &Bind<Op, Element>,
            .metadata_builder = &BuildMetadata<Op>,
            .name = name};
}
static_assert(std::is_trivially_destructible_v<CopyArgs> && sizeof(CopyArgs) <= kMaxKernelParamsSize);
static_assert(alignof(CopyArgs) <= alignof(std::max_align_t));

} // namespace

AM_REGISTER_KERNEL(CpuReshapeF32, (MakeLayout<OpType::kReshape, float>("cpu::reshape_f32_reference")))
AM_REGISTER_KERNEL(CpuReshapeF16, (MakeLayout<OpType::kReshape, Half>("cpu::reshape_f16_reference")))
AM_REGISTER_KERNEL(CpuReshapeBf16, (MakeLayout<OpType::kReshape, BFloat16>("cpu::reshape_bf16_reference")))
AM_REGISTER_KERNEL(CpuPermuteF32, (MakeLayout<OpType::kPermute, float>("cpu::permute_f32_reference")))
AM_REGISTER_KERNEL(CpuPermuteF16, (MakeLayout<OpType::kPermute, Half>("cpu::permute_f16_reference")))
AM_REGISTER_KERNEL(CpuPermuteBf16, (MakeLayout<OpType::kPermute, BFloat16>("cpu::permute_bf16_reference")))
AM_REGISTER_KERNEL(CpuReorderF32, (MakeLayout<OpType::kReorder, float>("cpu::reorder_f32_reference")))
AM_REGISTER_KERNEL(CpuReorderF16, (MakeLayout<OpType::kReorder, Half>("cpu::reorder_f16_reference")))
AM_REGISTER_KERNEL(CpuReorderBf16, (MakeLayout<OpType::kReorder, BFloat16>("cpu::reorder_bf16_reference")))

} // namespace aethermind::cpu::detail
