#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "elementwise_mul_internal.h"

#include <array>
#include <span>

namespace aethermind::cpu::detail {

Status RunElementwiseMulF32Reference(const ElementwiseMulF32KernelArgs& args) noexcept {
    if (args.numel == 0) {
        return Status::Ok();
    }

    if (args.output_rank == 0) {
        args.output_data[0] = args.lhs_data[0] * args.rhs_data[0];
        return Status::Ok();
    }

    std::array<int64_t, kMaxRank> coord{};
    for (int64_t i = 0; i < args.numel; ++i) {
        int64_t remaining = i;
        for (int32_t axis = args.output_rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % args.output_shape[axis];
            remaining /= args.output_shape[axis];
        }

        const auto lhs_offset = MapCoordToOffset(
                std::span(args.lhs_shape.data(), args.lhs_rank),
                args.output_rank,
                std::span(args.lhs_strides.data(), args.lhs_rank),
                coord);
        const auto rhs_offset = MapCoordToOffset(
                std::span(args.rhs_shape.data(), args.rhs_rank),
                args.output_rank,
                std::span(args.rhs_strides.data(), args.rhs_rank),
                coord);

        int64_t out_offset = 0;
        for (int32_t axis = 0; axis < args.output_rank; ++axis) {
            out_offset += coord[axis] * args.output_strides[axis];
        }

        args.output_data[out_offset] = args.lhs_data[lhs_offset] * args.rhs_data[rhs_offset];
    }

    return Status::Ok();
}

} // namespace aethermind::cpu::detail
