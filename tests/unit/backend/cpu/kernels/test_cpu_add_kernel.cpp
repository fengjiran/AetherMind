#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/add_op.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/shape_inference/broadcast.h"
#include "backend/cpu/kernels/add/add_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

template<size_t DataN, size_t Rank>
TensorView MakeFloatTensorView(const float (&data)[DataN],
                               const int64_t (&shape)[Rank],
                               const int64_t (&strides)[Rank]) {
    return TensorView{data, DataType::Float32(), shape, strides};
}

template<size_t DataN, size_t Rank>
MutableTensorView MakeMutableFloatTensorView(float (&data)[DataN],
                                             const int64_t (&shape)[Rank],
                                             const int64_t (&strides)[Rank]) {
    return MutableTensorView{data, DataType::Float32(), shape, strides};
}

template<typename T, size_t DataN, size_t Rank>
TensorView MakeTensorView(const T (&data)[DataN],
                          const int64_t (&shape)[Rank],
                          const int64_t (&strides)[Rank]) {
    return TensorView{data, DataType::Make<T>(), shape, strides};
}

template<typename T, size_t DataN, size_t Rank>
MutableTensorView MakeMutableTensorView(T (&data)[DataN],
                                        const int64_t (&shape)[Rank],
                                        const int64_t (&strides)[Rank]) {
    return MutableTensorView{data, DataType::Make<T>(), shape, strides};
}

template<typename T, size_t N>
void RunSameShapeAddAndExpect(const T (&lhs)[N],
                              const T (&rhs)[N],
                              const T (&expected)[N]) {
    T output[N]{};
    const int64_t shape[1] = {static_cast<int64_t>(N)};
    const int64_t strides[1] = {1};
    cpu::detail::AddParams params{
            .lhs_tensor = MakeTensorView(lhs, shape, strides),
            .rhs_tensor = MakeTensorView(rhs, shape, strides),
            .output_tensor = MakeMutableTensorView(output, shape, strides),
    };
    KernelContext ctx{.kernel_params = &params};

    const Status status = cpu::detail::AddKernel(ctx);
    ASSERT_TRUE(status.ok()) << status.ToString();
    for (size_t i = 0; i < N; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            EXPECT_FLOAT_EQ(output[i], expected[i]);
        } else if constexpr (std::is_same_v<T, double>) {
            EXPECT_DOUBLE_EQ(output[i], expected[i]);
        } else if constexpr (std::is_same_v<T, BFloat16>) {
            EXPECT_FLOAT_EQ(static_cast<float>(output[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(output[i], expected[i]);
        }
    }
}

// --- Direct kernel tests ---

TEST(AddKernel, RejectsNullParams) {
    KernelContext ctx{};
    ctx.kernel_params = nullptr;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddKernel, RejectsInvalidViews) {
    KernelContext ctx{};
    cpu::detail::AddParams params{};
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddKernel, RejectsMismatchedDTypes) {
    KernelContext ctx{};
    const int32_t lhs[6] = {1, 2, 3, 4, 5, 6};
    const float rhs[6] = {10, 20, 30, 40, 50, 60};
    float out[6] = {};
    const int64_t shape[2] = {2, 3};
    const int64_t strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = TensorView{lhs, DataType::Int(32), shape, strides},
            .rhs_tensor = MakeFloatTensorView(rhs, shape, strides),
            .output_tensor = MakeMutableFloatTensorView(out, shape, strides),
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddKernel, RejectsUnsupportedDType) {
    KernelContext ctx{};
    const Half lhs[1] = {Half{1.0F}};
    const Half rhs[1] = {Half{2.0F}};
    Half out[1]{};
    const int64_t shape[1] = {1};
    const int64_t strides[1] = {1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeTensorView(lhs, shape, strides),
            .rhs_tensor = MakeTensorView(rhs, shape, strides),
            .output_tensor = MakeMutableTensorView(out, shape, strides),
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddKernel, RejectsIncompatibleBroadcast) {
    KernelContext ctx{};
    const float lhs[6] = {1, 2, 3, 4, 5, 6};
    const float rhs[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float out[6] = {};
    const int64_t lhs_shape[2] = {2, 3};
    const int64_t lhs_strides[2] = {3, 1};
    const int64_t rhs_shape[2] = {4, 3};
    const int64_t rhs_strides[2] = {3, 1};
    const int64_t out_shape[2] = {2, 3};
    const int64_t out_strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, lhs_shape, lhs_strides),
            .rhs_tensor = MakeFloatTensorView(rhs, rhs_shape, rhs_strides),
            .output_tensor = MakeMutableFloatTensorView(out, out_shape, out_strides),
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddKernel, RejectsWrongOutputShape) {
    KernelContext ctx{};
    const float lhs[6] = {1, 2, 3, 4, 5, 6};
    const float rhs[6] = {10, 20, 30, 40, 50, 60};
    float out[6] = {};
    const int64_t lhs_shape[2] = {2, 3};
    const int64_t lhs_strides[2] = {3, 1};
    const int64_t rhs_shape[2] = {2, 3};
    const int64_t rhs_strides[2] = {3, 1};
    const int64_t out_shape[2] = {3, 2};
    const int64_t out_strides[2] = {2, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, lhs_shape, lhs_strides),
            .rhs_tensor = MakeFloatTensorView(rhs, rhs_shape, rhs_strides),
            .output_tensor = MakeMutableFloatTensorView(out, out_shape, out_strides),
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(AddKernel, ZeroElementOutputReturnsOk) {
    KernelContext ctx{};
    const int64_t shape[2] = {2, 0};
    const int64_t strides[2] = {0, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = TensorView{static_cast<const float*>(nullptr), DataType::Float32(), shape, strides},
            .rhs_tensor = TensorView{static_cast<const float*>(nullptr), DataType::Float32(), shape, strides},
            .output_tensor = MutableTensorView{static_cast<float*>(nullptr), DataType::Float32(), shape, strides},
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(AddKernel, ZeroPlusOneYieldsZero) {
    KernelContext ctx{};
    const float rhs[3] = {1, 2, 3};
    const int64_t zero_shape[2] = {0, 3};
    const int64_t zero_strides[2] = {3, 1};
    const int64_t rhs_shape[2] = {1, 3};
    const int64_t rhs_strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = TensorView{static_cast<const float*>(nullptr), DataType::Float32(), zero_shape, zero_strides},
            .rhs_tensor = MakeFloatTensorView(rhs, rhs_shape, rhs_strides),
            .output_tensor = MutableTensorView{static_cast<float*>(nullptr), DataType::Float32(), zero_shape, zero_strides},
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(AddKernel, OnePlusZeroYieldsZero) {
    KernelContext ctx{};
    const float lhs[3] = {1, 2, 3};
    const int64_t zero_shape[2] = {0, 3};
    const int64_t zero_strides[2] = {3, 1};
    const int64_t lhs_shape[2] = {1, 3};
    const int64_t lhs_strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, lhs_shape, lhs_strides),
            .rhs_tensor = TensorView{static_cast<const float*>(nullptr), DataType::Float32(), zero_shape, zero_strides},
            .output_tensor = MutableTensorView{static_cast<float*>(nullptr), DataType::Float32(), zero_shape, zero_strides},
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(AddKernel, RankZeroAdd) {
    KernelContext ctx{};
    const float lhs[1] = {3.0F};
    const float rhs[1] = {5.0F};
    float out[1] = {0.0F};

    cpu::detail::AddParams params{
            .lhs_tensor = TensorView{lhs, DataType::Float32(), IntArrayView{}, IntArrayView{}},
            .rhs_tensor = TensorView{rhs, DataType::Float32(), IntArrayView{}, IntArrayView{}},
            .output_tensor = MutableTensorView{out, DataType::Float32(), IntArrayView{}, IntArrayView{}},
    };
    ctx.kernel_params = &params;
    const Status status = cpu::detail::AddKernel(ctx);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(out[0], 8.0F);
}

TEST(AddKernel, SameShapeContiguous) {
    KernelContext ctx{};
    const float lhs[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float rhs[6] = {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F};
    float out[6] = {};
    const int64_t shape[2] = {2, 3};
    const int64_t strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, shape, strides),
            .rhs_tensor = MakeFloatTensorView(rhs, shape, strides),
            .output_tensor = MakeMutableFloatTensorView(out, shape, strides),
    };
    ctx.kernel_params = &params;
    ASSERT_TRUE(cpu::detail::AddKernel(ctx).ok());

    EXPECT_FLOAT_EQ(out[0], 11.0F);
    EXPECT_FLOAT_EQ(out[1], 22.0F);
    EXPECT_FLOAT_EQ(out[2], 33.0F);
    EXPECT_FLOAT_EQ(out[3], 44.0F);
    EXPECT_FLOAT_EQ(out[4], 55.0F);
    EXPECT_FLOAT_EQ(out[5], 66.0F);
}

TEST(AddKernel, SupportsAllRegisteredDTypes) {
    const float float_lhs[3] = {1.0F, 2.0F, 3.0F};
    const float float_rhs[3] = {0.5F, 1.5F, 2.5F};
    const float float_expected[3] = {1.5F, 3.5F, 5.5F};
    RunSameShapeAddAndExpect(float_lhs, float_rhs, float_expected);

    const double double_lhs[3] = {1.0, 2.0, 3.0};
    const double double_rhs[3] = {0.25, 0.5, 0.75};
    const double double_expected[3] = {1.25, 2.5, 3.75};
    RunSameShapeAddAndExpect(double_lhs, double_rhs, double_expected);

    const BFloat16 bfloat_lhs[3] = {BFloat16{1.0F}, BFloat16{2.0F}, BFloat16{3.0F}};
    const BFloat16 bfloat_rhs[3] = {BFloat16{0.5F}, BFloat16{1.5F}, BFloat16{2.5F}};
    const BFloat16 bfloat_expected[3] = {BFloat16{1.5F}, BFloat16{3.5F}, BFloat16{5.5F}};
    RunSameShapeAddAndExpect(bfloat_lhs, bfloat_rhs, bfloat_expected);

    const int32_t int32_lhs[3] = {1, -2, 3};
    const int32_t int32_rhs[3] = {4, 5, -6};
    const int32_t int32_expected[3] = {5, 3, -3};
    RunSameShapeAddAndExpect(int32_lhs, int32_rhs, int32_expected);

    const int64_t int64_lhs[3] = {1, -2, 3};
    const int64_t int64_rhs[3] = {4, 5, -6};
    const int64_t int64_expected[3] = {5, 3, -3};
    RunSameShapeAddAndExpect(int64_lhs, int64_rhs, int64_expected);
}

TEST(AddKernel, ReportsIntegerOverflow) {
    const int32_t lhs[1] = {std::numeric_limits<int32_t>::max()};
    const int32_t rhs[1] = {1};
    int32_t output[1]{};
    const int64_t shape[1] = {1};
    const int64_t strides[1] = {1};
    cpu::detail::AddParams params{
            .lhs_tensor = MakeTensorView(lhs, shape, strides),
            .rhs_tensor = MakeTensorView(rhs, shape, strides),
            .output_tensor = MakeMutableTensorView(output, shape, strides),
    };
    KernelContext ctx{.kernel_params = &params};

    const Status status = cpu::detail::AddKernel(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

TEST(AddKernel, ScalarBroadcast) {
    KernelContext ctx{};
    const float lhs[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float rhs[1] = {10.0F};
    float out[6] = {};
    const int64_t lhs_shape[2] = {2, 3};
    const int64_t lhs_strides[2] = {3, 1};
    const int64_t out_shape[2] = {2, 3};
    const int64_t out_strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, lhs_shape, lhs_strides),
            .rhs_tensor = TensorView{rhs, DataType::Float32(), IntArrayView{}, IntArrayView{}},
            .output_tensor = MakeMutableFloatTensorView(out, out_shape, out_strides),
    };
    ctx.kernel_params = &params;
    ASSERT_TRUE(cpu::detail::AddKernel(ctx).ok());

    EXPECT_FLOAT_EQ(out[0], 11.0F);
    EXPECT_FLOAT_EQ(out[1], 12.0F);
    EXPECT_FLOAT_EQ(out[2], 13.0F);
    EXPECT_FLOAT_EQ(out[3], 14.0F);
    EXPECT_FLOAT_EQ(out[4], 15.0F);
    EXPECT_FLOAT_EQ(out[5], 16.0F);
}

TEST(AddKernel, TrailingBroadcast) {
    KernelContext ctx{};
    const float lhs[2] = {10.0F, 20.0F};
    const float rhs[3] = {1.0F, 2.0F, 3.0F};
    float out[6] = {};
    const int64_t lhs_shape[2] = {2, 1};
    const int64_t lhs_strides[2] = {1, 1};
    const int64_t rhs_shape[1] = {3};
    const int64_t rhs_strides[1] = {1};
    const int64_t out_shape[2] = {2, 3};
    const int64_t out_strides[2] = {3, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, lhs_shape, lhs_strides),
            .rhs_tensor = MakeFloatTensorView(rhs, rhs_shape, rhs_strides),
            .output_tensor = MakeMutableFloatTensorView(out, out_shape, out_strides),
    };
    ctx.kernel_params = &params;
    ASSERT_TRUE(cpu::detail::AddKernel(ctx).ok());

    EXPECT_FLOAT_EQ(out[0], 11.0F);
    EXPECT_FLOAT_EQ(out[1], 12.0F);
    EXPECT_FLOAT_EQ(out[2], 13.0F);
    EXPECT_FLOAT_EQ(out[3], 21.0F);
    EXPECT_FLOAT_EQ(out[4], 22.0F);
    EXPECT_FLOAT_EQ(out[5], 23.0F);
}

TEST(AddKernel, HandlesStridedInput) {
    KernelContext ctx{};
    const float big[16] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                           8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F};
    const float rhs[4] = {100.0F, 200.0F, 300.0F, 400.0F};
    float out[4] = {};
    const int64_t lhs_shape[2] = {2, 2};
    const int64_t lhs_strides[2] = {8, 2};
    const int64_t rhs_shape[2] = {2, 2};
    const int64_t rhs_strides[2] = {2, 1};
    const int64_t out_shape[2] = {2, 2};
    const int64_t out_strides[2] = {2, 1};

    cpu::detail::AddParams params{
            .lhs_tensor = TensorView{big + 1, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs_tensor = MakeFloatTensorView(rhs, rhs_shape, rhs_strides),
            .output_tensor = MakeMutableFloatTensorView(out, out_shape, out_strides),
    };
    ctx.kernel_params = &params;
    ASSERT_TRUE(cpu::detail::AddKernel(ctx).ok());

    EXPECT_FLOAT_EQ(out[0], 101.0F);
    EXPECT_FLOAT_EQ(out[1], 203.0F);
    EXPECT_FLOAT_EQ(out[2], 309.0F);
    EXPECT_FLOAT_EQ(out[3], 411.0F);
}

TEST(AddKernel, HandlesStridedOutput) {
    KernelContext ctx{};
    const float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float rhs[4] = {10.0F, 20.0F, 30.0F, 40.0F};
    float big_out[16] = {};
    const int64_t shape[2] = {2, 2};
    const int64_t contig_strides[2] = {2, 1};
    const int64_t out_shape[2] = {2, 2};
    const int64_t out_strides[2] = {8, 2};

    cpu::detail::AddParams params{
            .lhs_tensor = MakeFloatTensorView(lhs, shape, contig_strides),
            .rhs_tensor = MakeFloatTensorView(rhs, shape, contig_strides),
            .output_tensor = MutableTensorView{big_out + 1, DataType::Float32(), out_shape, out_strides},
    };
    ctx.kernel_params = &params;
    ASSERT_TRUE(cpu::detail::AddKernel(ctx).ok());

    EXPECT_FLOAT_EQ(big_out[1], 11.0F);
    EXPECT_FLOAT_EQ(big_out[3], 22.0F);
    EXPECT_FLOAT_EQ(big_out[9], 33.0F);
    EXPECT_FLOAT_EQ(big_out[11], 44.0F);
    EXPECT_FLOAT_EQ(big_out[0], 0.0F);
    EXPECT_FLOAT_EQ(big_out[8], 0.0F);
}

// --- Backend registration test ---

TEST(AddKernel, ResolvesThroughCpuBackend) {
    CpuBackend backend;
    const std::array cases{
            std::pair{DataType::Float32(), "cpu::add_f32_scalar"},
            std::pair{DataType::Double(), "cpu::add_f64_scalar"},
            std::pair{DataType::BFloat(16), "cpu::add_bf16_scalar"},
            std::pair{DataType::Int(32), "cpu::add_i32_scalar"},
            std::pair{DataType::Int(64), "cpu::add_i64_scalar"},
    };
    static_assert(cases.size() == kAddSupportedDTypes.size(),
                  "ResolvesThroughCpuBackend test cases must cover the same number of dtypes as kAddSupportedDTypes");

    for (const auto& [dtype, expected_name]: cases) {
        SCOPED_TRACE(ToString(dtype));
        const auto selector = KernelSelector{
                .device_type = DeviceType::kCPU,
                .act_dtype = dtype,
                .weight_dtype = dtype,
                .weight_format = WeightFormat::kPlain,
                .isa = IsaLevel::kScalar,
                .phase = ExecPhase::kBoth,
        };
        const auto resolved = backend.ResolveKernelInfo(OpType::kAdd, selector);
        ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
        EXPECT_NE(resolved.value().fn, nullptr);
        EXPECT_STREQ(resolved.value().debug_name, expected_name);
    }
}

// --- End-to-end test through LowerModelGraph / ExecutionPlanBuilder / Executor ---

TEST(AddKernel, EndToEndThroughGraphLoweringAndExecutor) {
    ModelGraph graph;

    const GraphValueId lhs_tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{
                               ShapeSymbol::CreateFromValue(2)})},
            "lhs_tokens");
    const GraphValueId lhs_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{
                               ShapeSymbol::CreateFromValue(1),
                               ShapeSymbol::CreateFromValue(1)})},
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    auto emb_lhs_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {lhs_tokens, lhs_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});
    ASSERT_TRUE(emb_lhs_or.ok()) << emb_lhs_or.status().ToString();
    const AddedNode& emb_lhs = *emb_lhs_or;

    const GraphValueId rhs_tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{
                               ShapeSymbol::CreateFromValue(1)})},
            "rhs_tokens");
    const GraphValueId rhs_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{
                               ShapeSymbol::CreateFromValue(1),
                               ShapeSymbol::CreateFromValue(2)})},
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    auto emb_rhs_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {rhs_tokens, rhs_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});
    ASSERT_TRUE(emb_rhs_or.ok()) << emb_rhs_or.status().ToString();
    const AddedNode& emb_rhs = *emb_rhs_or;

    auto add_node_or = graph.AddNode(
            OpType::kAdd,
            std::nullopt,
            {emb_lhs.outputs[0], emb_rhs.outputs[0]},
            {NodeOutputDesc{.payload = ActivationValue{}, .name = "out"}},
            AddParams{});
    ASSERT_TRUE(add_node_or.ok()) << add_node_or.status().ToString();
    const AddedNode& add_node = *add_node_or;
    graph.MarkOutput(add_node.outputs[0]);

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 3U);
    ASSERT_EQ(lowered->step_bindings.size(), 3U);

    ASSERT_EQ(lowered->steps[0].op_type, OpType::kEmbedding);
    ASSERT_EQ(lowered->steps[1].op_type, OpType::kEmbedding);
    ASSERT_EQ(lowered->steps[2].op_type, OpType::kAdd);
    EXPECT_EQ(lowered->steps[2].act_dtype, DataType::Float32());
    EXPECT_EQ(lowered->steps[2].weight_dtype, DataType::Float32());

    RuntimeBuilder runtime_builder;
    RuntimeContext runtime = runtime_builder.Build();
    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 3U);

    RuntimeBindingContext bindings;

    const int64_t emb0_tokens[2] = {0, 0};
    float emb0_weight[1] = {2.0F};
    float emb0_out[2] = {};
    const int64_t emb0_tok_shape[1] = {2};
    const int64_t emb0_tok_strides[1] = {1};
    const int64_t emb0_w_shape[2] = {1, 1};
    const int64_t emb0_w_strides[2] = {1, 1};
    const int64_t emb0_out_shape[2] = {2, 1};
    const int64_t emb0_out_strides[2] = {1, 1};
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView{emb0_tokens, DataType::Int(64), emb0_tok_shape, emb0_tok_strides},
                                                     TensorView{emb0_weight, DataType::Float32(), emb0_w_shape, emb0_w_strides},
                                             },
                                             .outputs = {
                                                     MutableTensorView{emb0_out, DataType::Float32(), emb0_out_shape, emb0_out_strides},
                                             },
                                     });

    const int64_t emb1_tokens[1] = {0};
    float emb1_weight[2] = {3.0F, 5.0F};
    float emb1_out[2] = {};
    const int64_t emb1_tok_shape[1] = {1};
    const int64_t emb1_tok_strides[1] = {1};
    const int64_t emb1_w_shape[2] = {1, 2};
    const int64_t emb1_w_strides[2] = {2, 1};
    const int64_t emb1_out_shape[2] = {1, 2};
    const int64_t emb1_out_strides[2] = {2, 1};
    bindings.SetStepTensorBinding(1, StepTensorBinding{
                                             .inputs = {
                                                     TensorView{emb1_tokens, DataType::Int(64), emb1_tok_shape, emb1_tok_strides},
                                                     TensorView{emb1_weight, DataType::Float32(), emb1_w_shape, emb1_w_strides},
                                             },
                                             .outputs = {
                                                     MutableTensorView{emb1_out, DataType::Float32(), emb1_out_shape, emb1_out_strides},
                                             },
                                     });

    float add_out[4] = {};
    const int64_t lhs_shape[2] = {2, 1};
    const int64_t lhs_strides[2] = {1, 1};
    const int64_t rhs_shape[2] = {1, 2};
    const int64_t rhs_strides[2] = {2, 1};
    const int64_t out_shape[2] = {2, 2};
    const int64_t out_strides[2] = {2, 1};
    bindings.SetStepTensorBinding(2, StepTensorBinding{
                                             .inputs = {
                                                     TensorView{emb0_out, DataType::Float32(), lhs_shape, lhs_strides},
                                                     TensorView{emb1_out, DataType::Float32(), rhs_shape, rhs_strides},
                                             },
                                             .outputs = {
                                                     MutableTensorView{add_out, DataType::Float32(), out_shape, out_strides},
                                             },
                                     });

    const Status status = Executor::Execute(*plan, bindings);
    ASSERT_TRUE(status.ok()) << status.ToString();

    EXPECT_FLOAT_EQ(add_out[0], 5.0F);
    EXPECT_FLOAT_EQ(add_out[1], 7.0F);
    EXPECT_FLOAT_EQ(add_out[2], 5.0F);
    EXPECT_FLOAT_EQ(add_out[3], 7.0F);
}

TEST(AddKernel, Int64EndToEndThroughLoweredGraphAndExecutor) {
    const TensorSpec lhs_spec{.dtype = DataType::Int(64), .shape = StaticShape({2, 1})};
    const TensorSpec rhs_spec{.dtype = DataType::Int(64), .shape = StaticShape({1, 2})};
    const TensorSpec output_spec{.dtype = DataType::Int(64), .shape = StaticShape({2, 2})};
    LoweredGraph lowered;
    lowered.steps.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kAdd,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Int(64),
            .weight_dtype = DataType::Int(64),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .input_specs = {lhs_spec, rhs_spec},
            .output_specs = {output_spec},
            .op_params = OpParams(AddParams{}),
    });

    RuntimeContext runtime = RuntimeBuilder{}.Build();
    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    const int64_t lhs[2] = {10, 20};
    const int64_t rhs[2] = {1, 2};
    int64_t output[4]{};
    const int64_t lhs_shape[2] = {2, 1};
    const int64_t lhs_strides[2] = {1, 1};
    const int64_t rhs_shape[2] = {1, 2};
    const int64_t rhs_strides[2] = {2, 1};
    const int64_t output_shape[2] = {2, 2};
    const int64_t output_strides[2] = {2, 1};
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     MakeTensorView(lhs, lhs_shape, lhs_strides),
                                                     MakeTensorView(rhs, rhs_shape, rhs_strides),
                                             },
                                             .outputs = {
                                                     MakeMutableTensorView(output, output_shape, output_strides),
                                             },
                                     });

    const Status status = Executor::Execute(*plan, bindings);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(output[0], 11);
    EXPECT_EQ(output[1], 12);
    EXPECT_EQ(output[2], 21);
    EXPECT_EQ(output[3], 22);
}

// --- Runtime constraint integration test ---

TEST(AddKernel, RejectsIncompatibleRuntimeBroadcastShapes) {
    const ShapeSymbol s0 = ShapeSymbol::Create();
    const ShapeSymbol s1 = ShapeSymbol::Create();
    SymbolicShape lhs_shape(std::vector<ShapeSymbol>{s0, ShapeSymbol::CreateFromValue(3)});
    SymbolicShape rhs_shape(std::vector<ShapeSymbol>{s1, ShapeSymbol::CreateFromValue(3)});

    auto inferred = InferBroadcastShape(lhs_shape, rhs_shape);
    ASSERT_TRUE(inferred.ok());

    std::vector<TensorSpec> add_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = lhs_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = rhs_shape},
    };
    auto analyzed = InferOperator(OpType::kAdd, OpParams{AddParams{}}, add_inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    RuntimeBuilder runtime_builder;
    RuntimeContext runtime = runtime_builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kAdd,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .input_specs = add_inputs,
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams(AddParams{}),
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_EQ(plan->steps().front().runtime_checks.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<DimBroadcastableConstraint>(
            plan->steps().front().runtime_checks[0].condition));

    const float lhs_data[6] = {1, 2, 3, 4, 5, 6};
    const float rhs_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float out_data[6] = {};
    const int64_t lhs_concrete_shape[2] = {2, 3};
    const int64_t lhs_strides[2] = {3, 1};
    const int64_t rhs_concrete_shape[2] = {4, 3};
    const int64_t rhs_strides[2] = {3, 1};
    const int64_t out_concrete_shape[2] = {2, 3};
    const int64_t out_strides[2] = {3, 1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView{lhs_data, DataType::Float32(), lhs_concrete_shape, lhs_strides},
                                                     TensorView{rhs_data, DataType::Float32(), rhs_concrete_shape, rhs_strides},
                                             },
                                             .outputs = {
                                                     MutableTensorView{out_data, DataType::Float32(), out_concrete_shape, out_strides},
                                             },
                                     });

    const Status status = Executor::Execute(*plan, bindings);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

// TDD red-proof: after consolidation, the frozen registry must contain exactly
// five canonical Add descriptors with weight_dtype == act_dtype and no
// undefined-weight / v2 selector.
TEST(AddKernel, CanonicalAddRegistryHasExactlyFiveDescriptors) {
    // Constructing a CpuBackend implicitly freezes the global registry.
    CpuBackend backend;
    auto& registry = KernelRegistry::Global();
    auto result = registry.FindByOpType(OpType::kAdd);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& descriptors = result.value();

    EXPECT_EQ(descriptors.size(), 5U);

    // Canonical descriptor names (order-independent).
    const std::array<const char*, 5> canonical_names = {
            "cpu::add_f32_scalar",
            "cpu::add_f64_scalar",
            "cpu::add_bf16_scalar",
            "cpu::add_i32_scalar",
            "cpu::add_i64_scalar",
    };

    for (const auto& name: canonical_names) {
        bool found = false;
        for (const auto* desc: descriptors) {
            if (desc->name == name) {
                // Each canonical selector must have weight_dtype == act_dtype.
                EXPECT_EQ(desc->selector.weight_dtype, desc->selector.act_dtype)
                        << "Descriptor \"" << name << "\" has weight_dtype != act_dtype";
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Missing canonical descriptor: " << name;
    }

    // No descriptor should have undefined weight_dtype.
    for (const auto* desc: descriptors) {
        EXPECT_NE(desc->selector.weight_dtype, DataType{})
                << "Descriptor \"" << desc->name << "\" has undefined weight_dtype";
    }
}

// TDD red-proof: resolving Add with Float32 activation and undefined
// weight_dtype must return NotFound after consolidation.
TEST(AddKernel, ResolveAddWithUndefinedWeightDtypeReturnsNotFound) {
    CpuBackend backend;
    const auto selector = KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType{},
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    const auto resolved = backend.ResolveKernelInfo(OpType::kAdd, selector);
    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kNotFound);
}

}// namespace
}// namespace aethermind
