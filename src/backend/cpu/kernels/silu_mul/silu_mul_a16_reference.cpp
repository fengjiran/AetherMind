#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/activation_utils.h"
#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "silu_mul_internal.h"

#include <array>
#include <span>

namespace aethermind::cpu::detail {
namespace {

float LoadElement(const std::byte* data, int64_t offset, DataType dtype) noexcept {
    return LoadA16Element(data + static_cast<size_t>(offset) * A16ElementBytes(dtype), dtype);
}

} // namespace

void RunSiluMulA16FlatReference(const std::byte* gate, const std::byte* up,
                                std::byte* output, DataType dtype, int64_t numel) noexcept {
    const size_t element_bytes = A16ElementBytes(dtype);
    for (int64_t index = 0; index < numel; ++index) {
        const size_t offset = static_cast<size_t>(index) * element_bytes;
        const float product = ComputeSiluF32(LoadA16Element(gate + offset, dtype)) *
                              LoadA16Element(up + offset, dtype);
        StoreA16Element(output + offset, product, dtype);
    }
}

Status RunSiluMulA16Reference(const SiluMulA16KernelArgs& args) noexcept {
    if (args.numel == 0) {
        return Status::Ok();
    }

    if (args.is_flat) {
        RunSiluMulA16FlatReference(args.gate_data, args.up_data, args.output_data, args.act_dtype, args.numel);
        return Status::Ok();
    }

    if (args.output_rank == 0) {
        const float product =
                ComputeSiluF32(LoadElement(args.gate_data, 0, args.act_dtype)) *
                LoadElement(args.up_data, 0, args.act_dtype);
        StoreA16Element(args.output_data, product, args.act_dtype);
        return Status::Ok();
    }

    const std::span gate_shape(args.gate_shape.data(), args.gate_rank);
    const std::span gate_strides(args.gate_strides.data(), args.gate_rank);
    const std::span up_shape(args.up_shape.data(), args.up_rank);
    const std::span up_strides(args.up_strides.data(), args.up_rank);

    std::array<int64_t, kMaxRank> coord{};
    for (int64_t flat_index = 0; flat_index < args.numel; ++flat_index) {
        int64_t remaining = flat_index;
        for (int32_t axis = args.output_rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % args.output_shape[axis];
            remaining /= args.output_shape[axis];
        }

        const int64_t gate_offset = MapCoordToOffset(
                gate_shape, args.output_rank, gate_strides, coord);
        const int64_t up_offset = MapCoordToOffset(
                up_shape, args.output_rank, up_strides, coord);

        int64_t output_offset = 0;
        for (int32_t axis = 0; axis < args.output_rank; ++axis) {
            output_offset += coord[axis] * args.output_strides[axis];
        }

        const float product =
                ComputeSiluF32(LoadElement(args.gate_data, gate_offset, args.act_dtype)) *
                LoadElement(args.up_data, up_offset, args.act_dtype);
        StoreA16Element(args.output_data +
                                static_cast<size_t>(output_offset) *
                                        A16ElementBytes(args.act_dtype),
                        product, args.act_dtype);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
