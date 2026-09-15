#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "backend/cpu/kernels/common/activation_utils.h"
#include "silu_mul_internal.h"

#include <array>
#include <span>

namespace aethermind::cpu::detail {

Status RunSiluMulF32Reference(const SiluMulF32KernelArgs& args) noexcept {
    if (args.numel == 0) {
        return Status::Ok();
    }

    if (args.is_flat) {
        for (int64_t index = 0; index < args.numel; ++index) {
            args.output_data[index] = ComputeSiluF32(args.gate_data[index]) * args.up_data[index];
        }
        return Status::Ok();
    }

    if (args.output_rank == 0) {
        args.output_data[0] = ComputeSiluF32(args.gate_data[0]) * args.up_data[0];
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
        args.output_data[output_offset] =
                ComputeSiluF32(args.gate_data[gate_offset]) * args.up_data[up_offset];
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
