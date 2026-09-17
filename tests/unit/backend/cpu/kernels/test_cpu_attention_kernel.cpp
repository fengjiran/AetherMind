#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/macros.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/attention/attention_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeAttentionSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareAttentionKernel(
        int64_t num_q_heads = 1,
        int64_t num_kv_heads = 1,
        int64_t head_dim = 2) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kAttention, MakeAttentionSelector(),
            OpParams{AttentionParams{
                    .num_q_heads = num_q_heads,
                    .num_kv_heads = num_kv_heads,
                    .head_dim = head_dim,
            }});
}

struct AttentionTestViews {
    TensorView query{};
    MutableTensorView output{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunAttentionEntry(const ResolvedKernel& kernel,
                         const AttentionTestViews& views,
                         const KVCacheReadBinding& read) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 1> inputs = {views.query};
    const std::array<MutableTensorView, 1> outputs = {views.output};
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
            },
            prepared.storage.data()));
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .kv_read = &read,
            .attrs = kernel.attrs,
    });
}

KVCacheReadBinding MakeReadBinding(float* key,
                                   float* value,
                                   size_t num_kv_heads,
                                   size_t token_capacity,
                                   size_t head_dim,
                                   size_t token_stride_elements,
                                   size_t head_stride_elements,
                                   size_t visible_end,
                                   size_t query_begin,
                                   size_t query_end) {
    return KVCacheReadBinding{
            .storage = {
                    .key_data = reinterpret_cast<std::byte*>(key),
                    .value_data = reinterpret_cast<std::byte*>(value),
                    .dtype = DataType::Float32(),
                    .num_kv_heads = num_kv_heads,
                    .head_dim = head_dim,
                    .token_capacity = token_capacity,
                    .token_stride_bytes = token_stride_elements * sizeof(float),
                    .head_stride_bytes = head_stride_elements * sizeof(float),
            },
            .committed_end = visible_end,
            .visible_end = visible_end,
            .query_begin = query_begin,
            .query_end = query_end,
    };
}

TEST(CPUKernelAttention, CpuBackendPreparesPlainF32ReferenceKernel) {
    const auto kernel = PrepareAttentionKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->op_type, OpType::kAttention);
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::attention_f32_reference");
    EXPECT_NE(kernel->fn, nullptr);
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::AttentionF32KernelArgs));

    CpuBackend backend;
    KernelSelector unsupported = MakeAttentionSelector();
    unsupported.act_dtype = DataType::Float(16);
    EXPECT_EQ(backend.PrepareKernel(OpType::kAttention, unsupported,
                                    OpParams{AttentionParams{.num_q_heads = 1,
                                                             .num_kv_heads = 1,
                                                             .head_dim = 2}})
                      .status()
                      .code(),
              StatusCode::kNotFound);
}

TEST(CPUKernelAttention, ReferenceComputesCausalPrefillGolden) {
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t strides[2] = {2, 1};
    const std::array<float, 4> query = {1.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 4> output{};
    std::array<float, 4> key = {1.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 4> value = {10.0F, 0.0F, 0.0F, 20.0F};

    const auto kernel = PrepareAttentionKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunAttentionEntry(
                        *kernel,
                        {.query = TensorView(query.data(), DataType::Float32(), shape, strides),
                         .output = MutableTensorView(output.data(), DataType::Float32(), shape, strides)},
                        MakeReadBinding(key.data(), value.data(), 1, 2, 2, 2, 4, 2, 0, 2))
                        .ok());

    EXPECT_FLOAT_EQ(output[0], 10.0F);
    EXPECT_FLOAT_EQ(output[1], 0.0F);
    const float second_weight = std::exp(1.0F / std::sqrt(2.0F)) /
                                (1.0F + std::exp(1.0F / std::sqrt(2.0F)));
    EXPECT_NEAR(output[2], 10.0F * (1.0F - second_weight), 1.0e-5F);
    EXPECT_NEAR(output[3], 20.0F * second_weight, 1.0e-5F);
}

TEST(CPUKernelAttention, ReferenceMapsGroupedQueryHeadsToTheirKVHeads) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    const std::array<float, 4> query = {1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 4> output{};
    std::array<float, 4> key = {1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 4> value = {10.0F, 11.0F, 20.0F, 21.0F};

    const auto kernel = PrepareAttentionKernel(4, 2, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunAttentionEntry(
                        *kernel,
                        {.query = TensorView(query.data(), DataType::Float32(), shape, strides),
                         .output = MutableTensorView(output.data(), DataType::Float32(), shape, strides)},
                        MakeReadBinding(key.data(), value.data(), 2, 1, 1, 1, 2, 1, 0, 1))
                        .ok());
    EXPECT_FLOAT_EQ(output[0], 10.0F);
    EXPECT_FLOAT_EQ(output[1], 10.0F);
    EXPECT_FLOAT_EQ(output[2], 20.0F);
    EXPECT_FLOAT_EQ(output[3], 20.0F);
}

TEST(CPUKernelAttention, ReferenceSupportsStableSoftmaxAndPaddedStrides) {
    constexpr int64_t query_shape[2] = {1, 1};
    constexpr int64_t output_shape[2] = {1, 1};
    constexpr int64_t query_strides[2] = {5, 2};
    constexpr int64_t output_strides[2] = {7, 3};
    std::array<float, 5> query{};
    std::array<float, 7> output{};
    query[0] = 1.0F;
    std::array<float, 6> key{};
    std::array<float, 6> value{};
    key[0] = 1000.0F;
    key[3] = 999.0F;
    value[0] = 0.0F;
    value[3] = 10.0F;

    const auto kernel = PrepareAttentionKernel(1, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunAttentionEntry(
                        *kernel,
                        {.query = TensorView(query.data(), DataType::Float32(), query_shape, query_strides),
                         .output = MutableTensorView(output.data(), DataType::Float32(), output_shape,
                                                     output_strides)},
                        MakeReadBinding(key.data(), value.data(), 1, 2, 1, 3, 6, 2, 1, 2))
                        .ok());
    const float expected = 10.0F / (1.0F + std::exp(1.0F));
    EXPECT_NEAR(output[0], expected, 1.0e-5F);
}

TEST(CPUKernelAttention, ReferenceUsesHistoryAndCurrentTokenForDecode) {
    constexpr int64_t shape[2] = {1, 1};
    constexpr int64_t strides[2] = {1, 1};
    const std::array<float, 1> query = {1.0F};
    std::array<float, 1> output{};
    std::array<float, 2> key = {0.0F, 1.0F};
    std::array<float, 2> value = {2.0F, 6.0F};

    const auto kernel = PrepareAttentionKernel(1, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunAttentionEntry(
                        *kernel,
                        {.query = TensorView(query.data(), DataType::Float32(), shape, strides),
                         .output = MutableTensorView(output.data(), DataType::Float32(), shape, strides)},
                        MakeReadBinding(key.data(), value.data(), 1, 2, 1, 1, 2, 2, 1, 2))
                        .ok());
    EXPECT_NEAR(output[0], (2.0F + 6.0F * std::exp(1.0F)) / (1.0F + std::exp(1.0F)),
                1.0e-5F);
}

TEST(CPUKernelAttention, ReferenceHandlesFirstTokenDecode) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    const std::array<float, 2> query = {3.0F, -2.0F};
    std::array<float, 2> output{};
    std::array<float, 2> key = {1.0F, 1.0F};
    std::array<float, 2> value = {7.0F, 9.0F};

    const auto kernel = PrepareAttentionKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunAttentionEntry(
                        *kernel,
                        {.query = TensorView(query.data(), DataType::Float32(), shape, strides),
                         .output = MutableTensorView(output.data(), DataType::Float32(), shape, strides)},
                        MakeReadBinding(key.data(), value.data(), 1, 1, 2, 2, 2, 1, 0, 1))
                        .ok());
    EXPECT_FLOAT_EQ(output[0], 7.0F);
    EXPECT_FLOAT_EQ(output[1], 9.0F);
}

TEST(CPUKernelAttention, ReferenceRejectsInvalidMetadataAndConcreteBindings) {
    EXPECT_EQ(PrepareAttentionKernel(3, 2, 1).status().code(), StatusCode::kInvalidArgument);

    const auto kernel = PrepareAttentionKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    std::array<float, 2> query{};
    std::array<float, 2> output{};
    std::array<float, 2> key{};
    std::array<float, 2> value{};

    constexpr int64_t wrong_width_shape[2] = {1, 1};
    EXPECT_EQ(RunAttentionEntry(
                      *kernel,
                      {.query = TensorView(query.data(), DataType::Float32(), wrong_width_shape, strides),
                       .output = MutableTensorView(output.data(), DataType::Float32(), wrong_width_shape,
                                                   strides)},
                      MakeReadBinding(key.data(), value.data(), 1, 1, 2, 2, 2, 1, 0, 1))
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr int64_t colliding_output_strides[2] = {2, 0};
    EXPECT_EQ(RunAttentionEntry(
                      *kernel,
                      {.query = TensorView(query.data(), DataType::Float32(), shape, strides),
                       .output = MutableTensorView(output.data(), DataType::Float32(), shape,
                                                   colliding_output_strides)},
                      MakeReadBinding(key.data(), value.data(), 1, 1, 2, 2, 2, 1, 0, 1))
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr int64_t multirow_shape[2] = {2, 2};
    constexpr int64_t positive_colliding_strides[2] = {1, 1};
    std::array<float, 4> multirow_query{};
    std::array<float, 3> colliding_output{};
    std::array<float, 4> multirow_key{};
    std::array<float, 4> multirow_value{};
    EXPECT_EQ(RunAttentionEntry(
                      *kernel,
                      {.query = TensorView(multirow_query.data(), DataType::Float32(),
                                           multirow_shape, strides),
                       .output = MutableTensorView(colliding_output.data(), DataType::Float32(),
                                                   multirow_shape,
                                                   positive_colliding_strides)},
                      MakeReadBinding(multirow_key.data(), multirow_value.data(), 1, 2, 2,
                                      2, 4, 2, 0, 2))
                      .code(),
              StatusCode::kInvalidArgument);

    EXPECT_EQ(RunAttentionEntry(
                      *kernel,
                      {.query = TensorView(query.data(), DataType::Float32(), shape, strides),
                       .output = MutableTensorView(query.data(), DataType::Float32(), shape, strides)},
                      MakeReadBinding(key.data(), value.data(), 1, 1, 2, 2, 2, 1, 0, 1))
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelAttention, ReferenceRejectsInvalidReadGeometryAndAliases) {
    const auto kernel = PrepareAttentionKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    std::array<float, 2> query = {1.0F, 0.0F};
    std::array<float, 2> output{};
    std::array<float, 4> key{};
    std::array<float, 4> value{};
    const AttentionTestViews views{
            .query = TensorView(query.data(), DataType::Float32(), shape, strides),
            .output = MutableTensorView(output.data(), DataType::Float32(), shape, strides),
    };

    auto invalid_frontier = MakeReadBinding(key.data(), value.data(), 1, 2, 2, 2, 4, 1, 1, 1);
    EXPECT_EQ(RunAttentionEntry(*kernel, views, invalid_frontier).code(),
              StatusCode::kInvalidArgument);

    auto invalid_stride = MakeReadBinding(key.data(), value.data(), 1, 2, 2, 2, 4, 1, 0, 1);
    invalid_stride.storage.token_stride_bytes = 10;
    EXPECT_EQ(RunAttentionEntry(*kernel, views, invalid_stride).code(),
              StatusCode::kInvalidArgument);

    auto output_alias = MakeReadBinding(output.data(), value.data(), 1, 1, 2, 2, 2, 1, 0, 1);
    EXPECT_EQ(RunAttentionEntry(*kernel, views, output_alias).code(),
              StatusCode::kInvalidArgument);
}

} // namespace
