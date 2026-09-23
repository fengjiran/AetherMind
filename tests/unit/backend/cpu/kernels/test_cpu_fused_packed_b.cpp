#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_context.h"
#include "backend/cpu/cpu_backend_internal.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)

bool CanExecutePackedAvx2() {
    const auto caps = cpu::DetectCpuCapabilities();
    return caps.ok() && caps->effective_features.Contains(CpuFeature::kAvx2) &&
           caps->effective_features.Contains(CpuFeature::kFma);
}

float TestValue(size_t index) {
    return static_cast<float>(static_cast<int64_t>((index * 19U + 3U) % 37U) - 18) *
           0.03125F;
}

void Fill(std::vector<float>& values) {
    for (size_t i = 0; i < values.size(); ++i) values[i] = TestValue(i);
}

StatusOr<const KernelDescriptor*> ResolveCandidate(
        OpType op_type,
        std::string_view name,
        const KernelSelector& selector,
        const CpuCapabilities& caps,
        KernelRegistry& isolated_registry) {
    KernelRegistry& global = KernelRegistry::Global();
    AM_RETURN_IF_ERROR(global.Freeze());
    AM_ASSIGN_OR_RETURN(const auto registered, global.FindByOpType(op_type));
    const auto candidate = std::find_if(
            registered.begin(), registered.end(), [&](const KernelDescriptor* d) {
                return d->name == name;
            });
    if (candidate == registered.end()) {
        return Status::NotFound("Packed-B fused candidate descriptor is not registered");
    }
    AM_RETURN_IF_ERROR(isolated_registry.Register(**candidate));
    AM_RETURN_IF_ERROR(isolated_registry.Freeze());
    return cpu::internal::ResolveEligibleDescriptor(
            isolated_registry, op_type, selector, caps.effective_features);
}

void ExpectSegmentNearReference(const std::vector<float>& input,
                                int64_t rows,
                                int64_t k,
                                const std::vector<float>& weights,
                                int64_t weight_row_offset,
                                int64_t output_n,
                                int64_t output_stride,
                                const std::vector<float>& output,
                                float guard) {
    std::vector<float> expected(static_cast<size_t>(rows * output_stride), guard);
    const cpu::detail::GemmF32Args reference{
            .lhs = input.data(),
            .rhs = weights.data() + weight_row_offset * k,
            .output = expected.data(),
            .m = rows,
            .n = output_n,
            .k = k,
            .lhs_m_stride = k,
            .lhs_k_stride = 1,
            .rhs_k_stride = 1,
            .rhs_n_stride = k,
            .output_m_stride = output_stride,
            .output_n_stride = 1,
    };
    const Status status = cpu::detail::RunGemmF32Reference(reference);
    ASSERT_TRUE(status.ok()) << status.ToString();
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < output_n; ++col) {
            const size_t index = static_cast<size_t>(row * output_stride + col);
            const float absolute_error = std::fabs(output[index] - expected[index]);
            const float relative_error = absolute_error /
                                         std::max(std::fabs(expected[index]), 1.0e-6F);
            EXPECT_TRUE(absolute_error <= 2.0e-4F + 2.0e-5F * std::sqrt(float(k)) ||
                        relative_error <= 2.0e-4F)
                    << "row=" << row << " col=" << col
                    << " actual=" << output[index] << " expected=" << expected[index];
        }
        for (int64_t col = output_n; col < output_stride; ++col) {
            EXPECT_EQ(output[static_cast<size_t>(row * output_stride + col)], guard)
                    << "guard row=" << row << " col=" << col;
        }
    }
}

TEST(CpuQkvLinearPackedB, UnalignedComponentBoundariesUseCorrectSlices) {
    if (!CanExecutePackedAvx2()) {
        GTEST_SKIP() << "AVX2+FMA GEMM candidate is unavailable on this host";
    }
    constexpr int64_t rows = 9;
    constexpr int64_t k = 33;
    constexpr int64_t q_n = 17;
    constexpr int64_t key_n = 3;
    constexpr int64_t value_n = 15;
    constexpr int64_t total_n = q_n + key_n + value_n;
    constexpr int64_t q_stride = q_n + 2;
    constexpr int64_t key_stride = key_n + 2;
    constexpr int64_t value_stride = value_n + 2;
    std::vector<float> input(static_cast<size_t>(rows * k));
    std::vector<float> weights(static_cast<size_t>(total_n * k));
    std::vector<float> q_output(static_cast<size_t>(rows * q_stride), -11.0F);
    std::vector<float> key_output(static_cast<size_t>(rows * key_stride), -12.0F);
    std::vector<float> value_output(static_cast<size_t>(rows * value_stride), -13.0F);
    Fill(input);
    Fill(weights);

    const int64_t input_shape[] = {rows, k};
    const int64_t input_strides[] = {k, 1};
    const TensorView input_view(
            input.data(), DataType::Float32(), input_shape, input_strides);
    const int64_t q_shape[] = {q_n, k};
    const int64_t q_strides[] = {k, 1};
    const int64_t key_shape[] = {key_n, k};
    const int64_t value_shape[] = {value_n, k};
    const TensorView components[] = {
            TensorView(weights.data(), DataType::Float32(), q_shape, q_strides),
            TensorView(weights.data() + q_n * k, DataType::Float32(), key_shape, q_strides),
            TensorView(weights.data() + (q_n + key_n) * k,
                       DataType::Float32(), value_shape, q_strides),
    };
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
    const auto caps = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(caps.ok()) << caps.status().ToString();
    KernelRegistry isolated_registry;
    const auto descriptor = ResolveCandidate(
            OpType::kQkvLinear,
            "cpu::qkv_linear_f32_packed_bpanel_candidate",
            selector, *caps, isolated_registry);
    ASSERT_TRUE(descriptor.ok()) << descriptor.status().ToString();
    const auto recipe = cpu::internal::ResolvePackingRecipeFromRegistry(
            isolated_registry, OpType::kQkvLinear, selector,
            caps->effective_features);
    ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
    const QkvLinearParams op_params{
            .q_out_features = q_n,
            .k_out_features = key_n,
            .v_out_features = value_n,
            .has_bias = false,
    };
    std::vector<std::byte> attrs;
    ASSERT_TRUE((*descriptor)->metadata_builder(OpParams{op_params}, attrs).ok());

    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            OpType::kQkvLinear, components, selector, *recipe);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    const PackedWeightView packed_view{
            .data = (*packed)->storage().data(),
            .nbytes = (*packed)->storage().nbytes(),
            .logical_dtype = (*packed)->logical_dtype(),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };
    const int64_t q_output_shape[] = {rows, q_n};
    const int64_t q_output_strides[] = {q_stride, 1};
    const int64_t key_output_shape[] = {rows, key_n};
    const int64_t key_output_strides[] = {key_stride, 1};
    const int64_t value_output_shape[] = {rows, value_n};
    const int64_t value_output_strides[] = {value_stride, 1};
    const std::array<MutableTensorView, 3> outputs{
            MutableTensorView(q_output.data(), DataType::Float32(),
                              q_output_shape, q_output_strides),
            MutableTensorView(key_output.data(), DataType::Float32(),
                              key_output_shape, key_output_strides),
            MutableTensorView(value_output.data(), DataType::Float32(),
                              value_output_shape, value_output_strides),
    };
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
    ASSERT_TRUE((*descriptor)->params_builder(KernelParamsBuildContext{
                                                      .inputs = std::span<const TensorView>(&input_view, 1),
                                                      .outputs = outputs,
                                                      .attrs = attrs,
                                                      .packed_weight = packed_view,
                                              },
                                              params.data())
                        .ok());
    ASSERT_TRUE((*descriptor)->kernel_func(KernelContext{
                                                   .device_type = DeviceType::kCPU,
                                                   .kernel_params = params.data(),
                                                   .attrs = attrs,
                                           })
                        .ok());

    ExpectSegmentNearReference(input, rows, k, weights, 0, q_n, q_stride,
                               q_output, -11.0F);
    ExpectSegmentNearReference(input, rows, k, weights, q_n, key_n, key_stride,
                               key_output, -12.0F);
    ExpectSegmentNearReference(input, rows, k, weights, q_n + key_n,
                               value_n, value_stride, value_output, -13.0F);
}

TEST(CpuGateUpLinearPackedB, UnalignedComponentBoundaryUsesCorrectSlices) {
    if (!CanExecutePackedAvx2()) {
        GTEST_SKIP() << "AVX2+FMA GEMM candidate is unavailable on this host";
    }
    constexpr int64_t rows = 3;
    constexpr int64_t k = 31;
    constexpr int64_t gate_n = 5;
    constexpr int64_t up_n = 7;
    constexpr int64_t total_n = gate_n + up_n;
    constexpr int64_t gate_stride = gate_n + 2;
    constexpr int64_t up_stride = up_n + 2;
    std::vector<float> input(static_cast<size_t>(rows * k));
    std::vector<float> weights(static_cast<size_t>(total_n * k));
    std::vector<float> gate_output(static_cast<size_t>(rows * gate_stride), -21.0F);
    std::vector<float> up_output(static_cast<size_t>(rows * up_stride), -22.0F);
    Fill(input);
    Fill(weights);

    const int64_t input_shape[] = {rows, k};
    const int64_t input_strides[] = {k, 1};
    const TensorView input_view(
            input.data(), DataType::Float32(), input_shape, input_strides);
    const int64_t gate_shape[] = {gate_n, k};
    const int64_t component_strides[] = {k, 1};
    const int64_t up_shape[] = {up_n, k};
    const TensorView components[] = {
            TensorView(weights.data(), DataType::Float32(), gate_shape,
                       component_strides),
            TensorView(weights.data() + gate_n * k, DataType::Float32(),
                       up_shape, component_strides),
    };
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
    const auto caps = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(caps.ok()) << caps.status().ToString();
    KernelRegistry isolated_registry;
    const auto descriptor = ResolveCandidate(
            OpType::kGateUpLinear,
            "cpu::gate_up_linear_f32_packed_bpanel_candidate",
            selector, *caps, isolated_registry);
    ASSERT_TRUE(descriptor.ok()) << descriptor.status().ToString();
    const auto recipe = cpu::internal::ResolvePackingRecipeFromRegistry(
            isolated_registry, OpType::kGateUpLinear, selector,
            caps->effective_features);
    ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
    const GateUpLinearParams op_params{
            .gate_out_features = gate_n,
            .up_out_features = up_n,
            .has_bias = false,
    };
    std::vector<std::byte> attrs;
    ASSERT_TRUE((*descriptor)->metadata_builder(OpParams{op_params}, attrs).ok());

    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            OpType::kGateUpLinear, components, selector, *recipe);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    const PackedWeightView packed_view{
            .data = (*packed)->storage().data(),
            .nbytes = (*packed)->storage().nbytes(),
            .logical_dtype = (*packed)->logical_dtype(),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };
    const int64_t gate_output_shape[] = {rows, gate_n};
    const int64_t gate_output_strides[] = {gate_stride, 1};
    const int64_t up_output_shape[] = {rows, up_n};
    const int64_t up_output_strides[] = {up_stride, 1};
    const std::array<MutableTensorView, 2> outputs{
            MutableTensorView(gate_output.data(), DataType::Float32(),
                              gate_output_shape, gate_output_strides),
            MutableTensorView(up_output.data(), DataType::Float32(),
                              up_output_shape, up_output_strides),
    };
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
    ASSERT_TRUE((*descriptor)->params_builder(KernelParamsBuildContext{
                                                      .inputs = std::span<const TensorView>(&input_view, 1),
                                                      .outputs = outputs,
                                                      .attrs = attrs,
                                                      .packed_weight = packed_view,
                                              },
                                              params.data())
                        .ok());
    ASSERT_TRUE((*descriptor)->kernel_func(KernelContext{
                                                   .device_type = DeviceType::kCPU,
                                                   .kernel_params = params.data(),
                                                   .attrs = attrs,
                                           })
                        .ok());

    ExpectSegmentNearReference(input, rows, k, weights, 0, gate_n,
                               gate_stride, gate_output, -21.0F);
    ExpectSegmentNearReference(input, rows, k, weights, gate_n, up_n,
                               up_stride, up_output, -22.0F);
}

#endif

} // namespace
