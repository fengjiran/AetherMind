#include "aethermind/backend/cpu/kernels/common/activation_utils.h"
#include "silu_internal.h"

#include <array>

namespace aethermind::cpu::detail {

Status RunSiluF32Reference(const SiluF32KernelArgs& args) noexcept {
    if (args.numel == 0) {
        return Status::Ok();
    }

    if (args.is_flat) {
        for (int64_t index = 0; index < args.numel; ++index) {
            args.output_data[index] = ComputeSiluF32(args.input_data[index]);
        }
        return Status::Ok();
    }

    if (args.rank == 0) {
        args.output_data[0] = ComputeSiluF32(args.input_data[0]);
        return Status::Ok();
    }

    std::array<int64_t, kMaxRank> coord{};
    for (int64_t flat_index = 0; flat_index < args.numel; ++flat_index) {
        int64_t remaining = flat_index;
        for (int32_t axis = args.rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % args.shape[axis];
            remaining /= args.shape[axis];
        }

        int64_t input_offset = 0;
        int64_t output_offset = 0;
        for (int32_t axis = 0; axis < args.rank; ++axis) {
            input_offset += coord[axis] * args.input_strides[axis];
            output_offset += coord[axis] * args.output_strides[axis];
        }
        args.output_data[output_offset] = ComputeSiluF32(args.input_data[input_offset]);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
