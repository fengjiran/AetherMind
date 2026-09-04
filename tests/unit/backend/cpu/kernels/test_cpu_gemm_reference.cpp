#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <gtest/gtest.h>

#include <array>

namespace {

using namespace aethermind;

TEST(CPUKernelGemmReference, ComputesStridedMatrices) {
    std::array<float, 12> lhs{};
    std::array<float, 14> rhs{};
    std::array<float, 20> output{};
    lhs[0] = 1.0F;
    lhs[2] = 2.0F;
    lhs[4] = 3.0F;
    lhs[7] = -1.0F;
    lhs[9] = 0.0F;
    lhs[11] = 4.0F;
    rhs[0] = 2.0F;
    rhs[3] = -1.0F;
    rhs[5] = 0.5F;
    rhs[8] = 3.0F;
    rhs[10] = -2.0F;
    rhs[13] = 1.0F;

    const Status status = cpu::detail::RunGemmF32Reference(cpu::detail::GemmF32Args{
            .lhs = lhs.data(),
            .rhs = rhs.data(),
            .output = output.data(),
            .m = 2,
            .n = 2,
            .k = 3,
            .lhs_m_stride = 7,
            .lhs_k_stride = 2,
            .rhs_k_stride = 5,
            .rhs_n_stride = 3,
            .output_m_stride = 9,
            .output_n_stride = 4,
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], -3.0F);
    EXPECT_FLOAT_EQ(output[4], 8.0F);
    EXPECT_FLOAT_EQ(output[9], -10.0F);
    EXPECT_FLOAT_EQ(output[13], 5.0F);
}

TEST(CPUKernelGemmReference, ZeroInnerDimensionWritesZeroWithoutInputPointers) {
    std::array<float, 10> output;
    output.fill(3.0F);

    const Status status = cpu::detail::RunGemmF32Reference(cpu::detail::GemmF32Args{
            .output = output.data(),
            .m = 2,
            .n = 3,
            .k = 0,
            .output_m_stride = 5,
            .output_n_stride = 1,
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 0.0F);
    EXPECT_FLOAT_EQ(output[1], 0.0F);
    EXPECT_FLOAT_EQ(output[2], 0.0F);
    EXPECT_FLOAT_EQ(output[5], 0.0F);
    EXPECT_FLOAT_EQ(output[6], 0.0F);
    EXPECT_FLOAT_EQ(output[7], 0.0F);
}

TEST(CPUKernelGemmReference, EmptyOutputIsSuccessfulNoOp) {
    const Status status = cpu::detail::RunGemmF32Reference(cpu::detail::GemmF32Args{
            .m = 0,
            .n = 3,
            .k = 2,
    });
    EXPECT_TRUE(status.ok()) << status.ToString();
}

} // namespace
