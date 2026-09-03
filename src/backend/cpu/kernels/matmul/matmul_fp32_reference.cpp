#include "matmul_internal.h"

#include "backend/cpu/kernels/gemm/gemm_internal.h"

namespace aethermind::cpu::detail {

Status RunMatMulFp32Reference(const MatMulFp32KernelArgs& args) noexcept {
    if (args.batch_count == 0 || args.m == 0 || args.n == 0) {
        return Status::Ok();
    }

    for (int64_t batch = 0; batch < args.batch_count; ++batch) {
        int64_t remainder = batch;
        int64_t lhs_offset = 0;
        int64_t rhs_offset = 0;
        int64_t output_offset = 0;
        for (int32_t axis = args.batch_rank - 1; axis >= 0; --axis) {
            const int64_t coordinate = remainder % args.batch_dims[axis];
            remainder /= args.batch_dims[axis];
            lhs_offset += coordinate * args.lhs_batch_strides[axis];
            rhs_offset += coordinate * args.rhs_batch_strides[axis];
            output_offset += coordinate * args.output_batch_strides[axis];
        }

        AM_RETURN_IF_ERROR(RunGemmFp32Reference(GemmFp32Args{
                .lhs = args.k == 0 ? nullptr : args.lhs + lhs_offset,
                .rhs = args.k == 0 ? nullptr : args.rhs + rhs_offset,
                .output = args.output + output_offset,
                .m = args.m,
                .n = args.n,
                .k = args.k,
                .lhs_m_stride = args.lhs_m_stride,
                .lhs_k_stride = args.lhs_k_stride,
                .rhs_k_stride = args.rhs_k_stride,
                .rhs_n_stride = args.rhs_n_stride,
                .output_m_stride = args.output_m_stride,
                .output_n_stride = args.output_n_stride,
        }));
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
