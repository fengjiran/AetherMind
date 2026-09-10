#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/rope_frequency_resolver.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/rope/rope_internal.h"
#include "execution/test_execution_binding_helpers.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

RoPEParams MakeRoPEParams(int64_t head_dim = 4,
                          int64_t num_q_heads = 2,
                          int64_t num_kv_heads = 1,
                          double theta = 4.0) {
    return RoPEParams{
            .head_dim = head_dim,
            .num_attention_heads = num_q_heads,
            .num_key_value_heads = num_kv_heads,
            .max_pos_embeddings = 8,
            .theta = theta,
            .algorithm = StandardRoPE{},
    };
}

KernelSelector MakeRoPESelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareRoPEKernel(const RoPEParams& params = MakeRoPEParams()) {
    CpuBackend backend;
    return backend.PrepareKernel(OpType::kRoPE, MakeRoPESelector(), OpParams{params});
}

struct RoPETestViews {
    TensorView q{};
    TensorView k{};
    TensorView position_ids{};
    MutableTensorView q_output{};
    MutableTensorView k_output{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

StatusOr<PreparedKernelParams> BuildRoPEPreparedParams(
        const ResolvedKernel& kernel,
        const RoPETestViews& views) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 3> inputs{views.q, views.k, views.position_ids};
    const std::array<MutableTensorView, 2> outputs{views.q_output, views.k_output};
    const Status status = kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
            },
            prepared.storage.data());
    if (!status.ok()) {
        return status;
    }
    return prepared;
}

Status RunRoPEEntry(const ResolvedKernel& kernel,
                    const PreparedKernelParams& prepared) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .attrs = kernel.attrs,
    });
}

Status RunRoPEEntryWith(const ResolvedKernel& kernel,
                        const RoPETestViews& views) noexcept {
    const auto prepared = BuildRoPEPreparedParams(kernel, views);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return RunRoPEEntry(kernel, *prepared);
}

Status RunRoPEEntry(const RoPEParams& params,
                    const RoPETestViews& views) noexcept {
    const auto kernel = PrepareRoPEKernel(params);
    if (!kernel.ok()) {
        return kernel.status();
    }
    return RunRoPEEntryWith(*kernel, views);
}

void ExpectRoPENear(const float* input,
                    const float* output,
                    int64_t seq_len,
                    int64_t num_heads,
                    int64_t head_dim,
                    int64_t input_row_stride,
                    int64_t input_col_stride,
                    int64_t output_row_stride,
                    int64_t output_col_stride,
                    const int64_t* position_ids,
                    int64_t position_stride,
                    double theta,
                    double position_divisor) {
    const int64_t half = head_dim / 2;
    for (int64_t token = 0; token < seq_len; ++token) {
        const auto position = static_cast<double>(position_ids[token * position_stride]) /
                              position_divisor;
        for (int64_t head = 0; head < num_heads; ++head) {
            const int64_t head_offset = head * head_dim;
            for (int64_t pair = 0; pair < half; ++pair) {
                const double angle = position * std::pow(
                                                        theta, -2.0 * static_cast<double>(pair) / static_cast<double>(head_dim));
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double first = input[token * input_row_stride +
                                           (head_offset + pair) * input_col_stride];
                const double second = input[token * input_row_stride +
                                            (head_offset + half + pair) * input_col_stride];
                EXPECT_NEAR(output[token * output_row_stride +
                                   (head_offset + pair) * output_col_stride],
                            static_cast<float>(first * cosine - second * sine),
                            1.0e-6F);
                EXPECT_NEAR(output[token * output_row_stride +
                                   (head_offset + half + pair) * output_col_stride],
                            static_cast<float>(second * cosine + first * sine),
                            1.0e-6F);
            }
        }
    }
}

void ExpectPairSquaredNormPreserved(const float* input,
                                    const float* output,
                                    int64_t seq_len,
                                    int64_t num_heads,
                                    int64_t head_dim,
                                    int64_t input_row_stride,
                                    int64_t output_row_stride) {
    const int64_t half = head_dim / 2;
    for (int64_t token = 0; token < seq_len; ++token) {
        for (int64_t head = 0; head < num_heads; ++head) {
            const int64_t head_offset = head * head_dim;
            for (int64_t pair = 0; pair < half; ++pair) {
                const int64_t first_offset = head_offset + pair;
                const int64_t second_offset = head_offset + half + pair;
                const double input_squared_norm =
                        static_cast<double>(input[token * input_row_stride + first_offset]) *
                                input[token * input_row_stride + first_offset] +
                        static_cast<double>(input[token * input_row_stride + second_offset]) *
                                input[token * input_row_stride + second_offset];
                const double output_squared_norm =
                        static_cast<double>(output[token * output_row_stride + first_offset]) *
                                output[token * output_row_stride + first_offset] +
                        static_cast<double>(output[token * output_row_stride + second_offset]) *
                                output[token * output_row_stride + second_offset];
                EXPECT_NEAR(output_squared_norm, input_squared_norm, 2.0e-5);
            }
        }
    }
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(CPUKernelRoPE, CpuBackendPreparesPlainF32ReferenceKernel) {
    const auto kernel = PrepareRoPEKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->op_type, OpType::kRoPE);
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::rope_f32_reference");
    EXPECT_NE(kernel->fn, nullptr);
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::RoPEF32KernelArgs));

    CpuBackend backend;
    KernelSelector unsupported = MakeRoPESelector();
    unsupported.act_dtype = DataType::Float(16);
    EXPECT_EQ(backend.PrepareKernel(OpType::kRoPE, unsupported, OpParams{MakeRoPEParams()})
                      .status()
                      .code(),
              StatusCode::kNotFound);
    unsupported = MakeRoPESelector();
    unsupported.weight_format = WeightFormat::kPacked;
    EXPECT_EQ(backend.PrepareKernel(OpType::kRoPE, unsupported, OpParams{MakeRoPEParams()})
                      .status()
                      .code(),
              StatusCode::kNotFound);
}

TEST(CPUKernelRoPE, ReferenceUsesLlamaSplitHalfGoldenWithGqa) {
    constexpr int64_t q_shape[2] = {2, 8};
    constexpr int64_t q_strides[2] = {8, 1};
    constexpr int64_t k_shape[2] = {2, 4};
    constexpr int64_t k_strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr float q[16] = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, 0.5F, 2.0F, -3.0F,
                             2.0F, -1.0F, 0.5F, 3.0F, 4.0F, -2.0F, 1.5F, 0.25F};
    constexpr float k[8] = {-2.0F, 1.0F, 0.5F, 3.0F, 1.5F, -0.5F, 4.0F, -2.0F};
    constexpr int64_t position_ids[2] = {0, 1};
    std::array<float, 16> q_output{};
    std::array<float, 8> k_output{};

    const Status status = RunRoPEEntry(MakeRoPEParams(), RoPETestViews{
                                                                 .q = TensorView{q, DataType::Float32(), q_shape, q_strides},
                                                                 .k = TensorView{k, DataType::Float32(), k_shape, k_strides},
                                                                 .position_ids = TensorView{position_ids, DataType::Int(64), position_shape, position_strides},
                                                                 .q_output = MutableTensorView{q_output.data(), DataType::Float32(), q_shape, q_strides},
                                                                 .k_output = MutableTensorView{k_output.data(), DataType::Float32(), k_shape, k_strides},
                                                         });
    ASSERT_TRUE(status.ok()) << status.ToString();
    for (size_t index = 0; index < 8; ++index) {
        EXPECT_EQ(q_output[index], q[index]);
    }
    for (size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(k_output[index], k[index]);
    }
    // token 1/head 0 locks x[0] <-> x[2] and x[1] <-> x[3], not adjacent pairs.
    EXPECT_NEAR(q_output[8], 0.6598691F, 1.0e-6F);
    EXPECT_NEAR(q_output[9], -2.3158591F, 1.0e-6F);
    EXPECT_NEAR(q_output[10], 1.9530932F, 1.0e-6F);
    EXPECT_NEAR(q_output[11], 2.1533222F, 1.0e-6F);
    ExpectRoPENear(q, q_output.data(), 2, 2, 4, 8, 1, 8, 1, position_ids, 1, 4.0, 1.0);
    ExpectRoPENear(k, k_output.data(), 2, 1, 4, 4, 1, 4, 1, position_ids, 1, 4.0, 1.0);
}

TEST(CPUKernelRoPE, ReferenceSupportsLinearScalingAndStridedPaddedLayouts) {
    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t q_strides[2] = {11, 2};
    constexpr int64_t k_strides[2] = {12, 2};
    constexpr int64_t q_output_strides[2] = {13, 3};
    constexpr int64_t k_output_strides[2] = {14, 3};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {2};
    std::array<float, 18> q{};
    std::array<float, 19> k{};
    std::array<float, 23> q_output{};
    std::array<float, 24> k_output{};
    std::array<int64_t, 3> positions{};
    for (int64_t token = 0; token < 2; ++token) {
        for (int64_t col = 0; col < 4; ++col) {
            q[token * q_strides[0] + col * q_strides[1]] =
                    static_cast<float>(token * 4 + col - 2) * 0.5F;
            k[token * k_strides[0] + col * k_strides[1]] =
                    static_cast<float>(token * 4 + col + 1) * -0.25F;
        }
    }
    positions[0] = 0;
    positions[2] = 2;
    auto params = MakeRoPEParams(4, 1, 1);
    params.algorithm = LinearRoPE{.factor = 2.0};

    ASSERT_TRUE(RunRoPEEntry(params, RoPETestViews{
                                             .q = TensorView{q.data(), DataType::Float32(), shape, q_strides},
                                             .k = TensorView{k.data(), DataType::Float32(), shape, k_strides},
                                             .position_ids = TensorView{positions.data(), DataType::Int(64), position_shape, position_strides},
                                             .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, q_output_strides},
                                             .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, k_output_strides},
                                     })
                        .ok());
    ExpectRoPENear(q.data(), q_output.data(), 2, 1, 4, 11, 2, 13, 3,
                   positions.data(), 2, 4.0, 2.0);
    ExpectRoPENear(k.data(), k_output.data(), 2, 1, 4, 12, 2, 14, 3,
                   positions.data(), 2, 4.0, 2.0);
}

TEST(CPUKernelRoPE, ReferenceSupportsExactInPlaceOutputs) {
    constexpr int64_t q_shape[2] = {2, 8};
    constexpr int64_t q_strides[2] = {11, 1};
    constexpr int64_t k_shape[2] = {2, 4};
    constexpr int64_t k_strides[2] = {6, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    std::array<float, 19> q{};
    std::array<float, 10> k{};
    constexpr int64_t position_ids[2] = {1, 3};
    for (int64_t token = 0; token < 2; ++token) {
        for (int64_t col = 0; col < 8; ++col) {
            q[token * q_strides[0] + col] = static_cast<float>(token * 8 + col - 5) * 0.25F;
        }
        for (int64_t col = 0; col < 4; ++col) {
            k[token * k_strides[0] + col] = static_cast<float>(token * 4 + col - 3) * -0.5F;
        }
    }
    const auto original_q = q;
    const auto original_k = k;
    ASSERT_TRUE(RunRoPEEntry(MakeRoPEParams(), RoPETestViews{
                                                       .q = TensorView{q.data(), DataType::Float32(), q_shape, q_strides},
                                                       .k = TensorView{k.data(), DataType::Float32(), k_shape, k_strides},
                                                       .position_ids = TensorView{position_ids, DataType::Int(64), position_shape, position_strides},
                                                       .q_output = MutableTensorView{q.data(), DataType::Float32(), q_shape, q_strides},
                                                       .k_output = MutableTensorView{k.data(), DataType::Float32(), k_shape, k_strides},
                                               })
                        .ok());
    ExpectRoPENear(original_q.data(), q.data(), 2, 2, 4, 11, 1, 11, 1, position_ids, 1, 4.0, 1.0);
    ExpectRoPENear(original_k.data(), k.data(), 2, 1, 4, 6, 1, 6, 1, position_ids, 1, 4.0, 1.0);
}

TEST(CPUKernelRoPE, ReferenceRejectsPartialAndCrossTensorAliases) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[1] = {0};
    std::array<float, 8> q{};
    std::array<float, 8> k{};
    std::array<float, 8> q_output{};
    std::array<float, 8> k_output{};
    const auto make_views = [&] {
        return RoPETestViews{
                .q = TensorView{q.data(), DataType::Float32(), shape, strides},
                .k = TensorView{k.data(), DataType::Float32(), shape, strides},
                .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
        };
    };
    const auto expect_invalid = [&](RoPETestViews views) {
        EXPECT_EQ(RunRoPEEntry(MakeRoPEParams(4, 1, 1), views).code(),
                  StatusCode::kInvalidArgument);
    };

    auto views = make_views();
    views.q_output = MutableTensorView{q.data() + 1, DataType::Float32(), shape, strides};
    expect_invalid(views);
    views = make_views();
    views.q = TensorView{q.data() + 1, DataType::Float32(), shape, strides};
    views.q_output = MutableTensorView{q.data(), DataType::Float32(), shape, strides};
    expect_invalid(views);
    views = make_views();
    views.q_output = MutableTensorView{k.data() + 1, DataType::Float32(), shape, strides};
    expect_invalid(views);
    views = make_views();
    views.k_output = MutableTensorView{q.data() + 1, DataType::Float32(), shape, strides};
    expect_invalid(views);
    views = make_views();
    views.k_output = MutableTensorView{q_output.data() + 1, DataType::Float32(), shape, strides};
    expect_invalid(views);

    std::array<int64_t, 4> position_storage{};
    views = make_views();
    views.position_ids = TensorView{position_storage.data() + 1, DataType::Int(64), position_shape, position_strides};
    views.q_output = MutableTensorView{reinterpret_cast<float*>(position_storage.data()),
                                       DataType::Float32(), shape, strides};
    expect_invalid(views);
}

TEST(CPUKernelRoPE, ReferenceAllowsSameAllocationWithRowSeparatedViews) {
    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t strides[2] = {8, 1};
    constexpr int64_t contiguous_strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[2] = {0, 1};
    std::array<float, 16> q_and_q_output{};
    std::array<float, 8> k{};
    std::array<float, 8> k_output{};
    for (int64_t token = 0; token < 2; ++token) {
        for (int64_t column = 0; column < 4; ++column) {
            q_and_q_output[token * strides[0] + column] =
                    static_cast<float>(token * 4 + column - 2);
            k[token * contiguous_strides[0] + column] =
                    static_cast<float>(token * 4 + column + 1);
        }
    }

    ASSERT_TRUE(RunRoPEEntry(MakeRoPEParams(4, 1, 1), RoPETestViews{
                                                              .q = TensorView{q_and_q_output.data(), DataType::Float32(), shape, strides},
                                                              .k = TensorView{k.data(), DataType::Float32(), shape, contiguous_strides},
                                                              .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                              .q_output = MutableTensorView{q_and_q_output.data() + 4, DataType::Float32(), shape, strides},
                                                              .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, contiguous_strides},
                                                      })
                        .ok());
    ExpectRoPENear(q_and_q_output.data(), q_and_q_output.data() + 4, 2, 1, 4,
                   8, 1, 8, 1, positions, 1, 4.0, 1.0);
    ExpectRoPENear(k.data(), k_output.data(), 2, 1, 4, 4, 1, 4, 1,
                   positions, 1, 4.0, 1.0);
}

TEST(CPUKernelRoPE, ReferenceSupportsSubunitLinearScaling) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr float q[4] = {1.0F, -2.0F, 3.0F, -4.0F};
    constexpr float k[4] = {-0.5F, 1.5F, -2.5F, 3.5F};
    constexpr int64_t positions[1] = {3};
    std::array<float, 4> q_output{};
    std::array<float, 4> k_output{};
    auto params = MakeRoPEParams(4, 1, 1);
    params.algorithm = LinearRoPE{.factor = 0.5};

    ASSERT_TRUE(RunRoPEEntry(params, RoPETestViews{
                                             .q = TensorView{q, DataType::Float32(), shape, strides},
                                             .k = TensorView{k, DataType::Float32(), shape, strides},
                                             .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                             .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                             .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                     })
                        .ok());
    ExpectRoPENear(q, q_output.data(), 1, 1, 4, 4, 1, 4, 1,
                   positions, 1, 4.0, 0.5);
    ExpectRoPENear(k, k_output.data(), 1, 1, 4, 4, 1, 4, 1,
                   positions, 1, 4.0, 0.5);
}

TEST(CPUKernelRoPE, ReferenceSupportsSharedQkvStorageInPlace) {
    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t strides[2] = {12, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[2] = {1, 17};
    // Each row holds Q, K, then V. Q/K envelopes overlap, but accessed rows
    // do not; V must remain untouched while both rotated outputs alias inputs.
    std::array<float, 24> qkv{};
    for (size_t index = 0; index < qkv.size(); ++index) {
        qkv[index] = static_cast<float>(static_cast<int>(index) - 9) * 0.25F;
    }
    const auto original = qkv;

    ASSERT_TRUE(RunRoPEEntry(MakeRoPEParams(4, 1, 1), RoPETestViews{
                                                              .q = TensorView{qkv.data(), DataType::Float32(), shape, strides},
                                                              .k = TensorView{qkv.data() + 4, DataType::Float32(), shape, strides},
                                                              .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                              .q_output = MutableTensorView{qkv.data(), DataType::Float32(), shape, strides},
                                                              .k_output = MutableTensorView{qkv.data() + 4, DataType::Float32(), shape, strides},
                                                      })
                        .ok());

    ExpectRoPENear(original.data(), qkv.data(), 2, 1, 4, 12, 1, 12, 1, positions, 1, 4.0, 1.0);
    ExpectRoPENear(original.data() + 4, qkv.data() + 4, 2, 1, 4, 12, 1, 12, 1, positions, 1, 4.0, 1.0);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 8; column < 12; ++column) {
            EXPECT_EQ(qkv[row * 12 + column], original[row * 12 + column]);
        }
    }
}

TEST(CPUKernelRoPE, ReferencePreservesPairSquaredNorm) {
    constexpr int64_t q_shape[2] = {3, 128};
    constexpr int64_t q_strides[2] = {128, 1};
    constexpr int64_t k_shape[2] = {3, 64};
    constexpr int64_t k_strides[2] = {64, 1};
    constexpr int64_t position_shape[1] = {3};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[3] = {0, 17, 4096};
    std::array<float, 384> q{};
    std::array<float, 192> k{};
    std::array<float, 384> q_output{};
    std::array<float, 192> k_output{};
    for (size_t index = 0; index < q.size(); ++index) {
        q[index] = static_cast<float>(static_cast<int64_t>(index % 17U) - 8) * 0.25F;
    }
    for (size_t index = 0; index < k.size(); ++index) {
        k[index] = static_cast<float>(static_cast<int64_t>(index % 13U) - 6) * -0.5F;
    }

    ASSERT_TRUE(RunRoPEEntry(MakeRoPEParams(64, 2, 1), RoPETestViews{
                                                               .q = TensorView{q.data(), DataType::Float32(), q_shape, q_strides},
                                                               .k = TensorView{k.data(), DataType::Float32(), k_shape, k_strides},
                                                               .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                               .q_output = MutableTensorView{q_output.data(), DataType::Float32(), q_shape, q_strides},
                                                               .k_output = MutableTensorView{k_output.data(), DataType::Float32(), k_shape, k_strides},
                                                       })
                        .ok());
    ExpectPairSquaredNormPreserved(q.data(), q_output.data(), 3, 2, 64, 128, 128);
    ExpectPairSquaredNormPreserved(k.data(), k_output.data(), 3, 1, 64, 64, 64);
}

TEST(CPUKernelRoPEEntry, RejectsByteAndAddressRangeOverflow) {
    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t byte_overflow_strides[2] = {
            std::numeric_limits<int64_t>::max() / 2 + 1, 1};
    constexpr int64_t positions[2] = {0, 1};
    std::array<float, 8> q{};
    std::array<float, 8> k{};
    std::array<float, 8> q_output{};
    std::array<float, 8> k_output{};
    const auto make_views = [&] {
        return RoPETestViews{
                .q = TensorView{q.data(), DataType::Float32(), shape, strides},
                .k = TensorView{k.data(), DataType::Float32(), shape, strides},
                .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
        };
    };

    auto views = make_views();
    views.q = TensorView{q.data(), DataType::Float32(), shape, byte_overflow_strides};
    EXPECT_EQ(RunRoPEEntry(MakeRoPEParams(4, 1, 1), views).code(),
              StatusCode::kInvalidArgument);

    const auto near_address_limit = reinterpret_cast<const float*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    views = make_views();
    views.q = TensorView{near_address_limit, DataType::Float32(), shape, strides};
    EXPECT_EQ(RunRoPEEntry(MakeRoPEParams(4, 1, 1), views).code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelRoPE, PreparedParamsRejectUnrepresentableInverseFrequency) {
    constexpr int64_t shape[2] = {1, 64};
    constexpr int64_t strides[2] = {64, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[1] = {0};
    std::array<float, 64> q{};
    std::array<float, 64> k{};
    std::array<float, 64> q_output{};
    std::array<float, 64> k_output{};
    const auto params = MakeRoPEParams(64, 1, 1,
                                       std::numeric_limits<double>::denorm_min());

    EXPECT_EQ(RunRoPEEntry(params, RoPETestViews{
                                           .q = TensorView{q.data(), DataType::Float32(), shape, strides},
                                           .k = TensorView{k.data(), DataType::Float32(), shape, strides},
                                           .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                           .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                           .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                   })
                      .code(),
              StatusCode::kOverflow);
}

TEST(CPUKernelRoPEEntry, ParamsBuilderRejectsCorruptFrozenFrequencyAttrs) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[1] = {0};
    std::array<float, 4> q{};
    std::array<float, 4> k{};
    std::array<float, 4> q_output{};
    std::array<float, 4> k_output{};
    const auto views = RoPETestViews{
            .q = TensorView{q.data(), DataType::Float32(), shape, strides},
            .k = TensorView{k.data(), DataType::Float32(), shape, strides},
            .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
            .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
            .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
    };
    const auto kernel = PrepareRoPEKernel(MakeRoPEParams(4, 1, 1));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    auto truncated = *kernel;
    truncated.attrs.pop_back();
    EXPECT_EQ(BuildRoPEPreparedParams(truncated, views).status().code(),
              StatusCode::kInvalidArgument);

    auto invalid_frequency = *kernel;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::memcpy(invalid_frequency.attrs.data() +
                        sizeof(cpu::detail::RoPEF32KernelMetadata),
                &nan, sizeof(nan));
    EXPECT_EQ(BuildRoPEPreparedParams(invalid_frequency, views).status().code(),
              StatusCode::kOverflow);
}

TEST(CPUKernelRoPEEntry, RejectsInvalidParamsLayoutsAndAliases) {
    CpuBackend backend;
    EXPECT_EQ(backend.PrepareKernel(OpType::kRoPE, MakeRoPESelector(), OpParams{RmsNormParams{}})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
    auto invalid = MakeRoPEParams();
    invalid.head_dim = 3;
    EXPECT_EQ(backend.PrepareKernel(OpType::kRoPE, MakeRoPESelector(), OpParams{invalid})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
    invalid = MakeRoPEParams();
    invalid.rotary_dim = 3;
    EXPECT_EQ(backend.PrepareKernel(OpType::kRoPE, MakeRoPESelector(), OpParams{invalid})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t zero_position_stride[1] = {0};
    constexpr int64_t overlapping_rows[2] = {1, 1};
    constexpr int64_t different_mapping[2] = {5, 1};
    constexpr int64_t overflow_stride[2] = {std::numeric_limits<int64_t>::max(), 1};
    std::array<float, 8> q{};
    std::array<float, 8> k{};
    std::array<float, 8> q_output{};
    std::array<float, 8> k_output{};
    constexpr int64_t positions[2] = {0, 1};
    const auto make_views = [&] {
        return RoPETestViews{
                .q = TensorView{q.data(), DataType::Float32(), shape, strides},
                .k = TensorView{k.data(), DataType::Float32(), shape, strides},
                .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
        };
    };
    const auto expect_invalid = [&](RoPETestViews views) {
        EXPECT_EQ(RunRoPEEntry(MakeRoPEParams(4, 1, 1), views).code(),
                  StatusCode::kInvalidArgument);
    };
    auto views = make_views();
    views.q = TensorView{};
    expect_invalid(views);
    views = make_views();
    views.position_ids = TensorView{positions, DataType::Int(32), position_shape, position_strides};
    expect_invalid(views);
    views = make_views();
    views.position_ids = TensorView{positions, DataType::Int(64), position_shape, zero_position_stride};
    expect_invalid(views);
    views = make_views();
    views.q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, overlapping_rows};
    expect_invalid(views);
    views = make_views();
    views.q = TensorView{q.data(), DataType::Float32(), shape, overflow_stride};
    expect_invalid(views);
    views = make_views();
    views.q_output = MutableTensorView{q.data(), DataType::Float32(), shape, different_mapping};
    expect_invalid(views);
    views = make_views();
    views.q_output = MutableTensorView{k.data(), DataType::Float32(), shape, strides};
    expect_invalid(views);
    views = make_views();
    views.k_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides};
    expect_invalid(views);
}

TEST(CPUKernelRoPE, PreparedParamsRevalidateMutablePositionContentsBeforeWrites) {
    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr float q[8] = {1.0F, 2.0F, 3.0F, 4.0F,
                            -0.5F, 1.5F, -2.5F, 3.5F};
    constexpr float k[8] = {-1.0F, 0.5F, 2.0F, -3.0F,
                            4.0F, -4.0F, 0.25F, -0.25F};
    int64_t positions[2] = {0, 1};
    std::array<float, 8> q_output{};
    std::array<float, 8> k_output{};
    const auto kernel = PrepareRoPEKernel(MakeRoPEParams(4, 1, 1));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const auto prepared = BuildRoPEPreparedParams(*kernel, RoPETestViews{
                                                                   .q = TensorView{q, DataType::Float32(), shape, strides},
                                                                   .k = TensorView{k, DataType::Float32(), shape, strides},
                                                                   .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                                   .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                                                   .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                                           });
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    ASSERT_TRUE(RunRoPEEntry(*kernel, *prepared).ok());
    EXPECT_EQ(q_output[0], q[0]);
    // max_position_embeddings is not a coordinate upper bound.
    positions[1] = 999;
    ASSERT_TRUE(RunRoPEEntry(*kernel, *prepared).ok());
    ExpectRoPENear(q, q_output.data(), 2, 1, 4, 4, 1, 4, 1, positions, 1, 4.0, 1.0);
    q_output.fill(11.0F);
    k_output.fill(13.0F);
    positions[1] = -1;
    EXPECT_EQ(RunRoPEEntry(*kernel, *prepared).code(), StatusCode::kInvalidArgument);
    for (float value: q_output) EXPECT_EQ(value, 11.0F);
    for (float value: k_output) EXPECT_EQ(value, 13.0F);
}

TEST(CPUKernelRoPE, PreparedParamsRevalidateDynamicAngleRangeBeforeWrites) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr float q[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float k[4] = {-1.0F, 0.5F, 2.0F, -3.0F};
    int64_t positions[1] = {0};
    std::array<float, 4> q_output{};
    std::array<float, 4> k_output{};
    auto params = MakeRoPEParams(4, 1, 1);
    params.algorithm = LinearRoPE{.factor = std::numeric_limits<double>::denorm_min()};
    const auto kernel = PrepareRoPEKernel(params);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const auto prepared = BuildRoPEPreparedParams(*kernel, RoPETestViews{
                                                                   .q = TensorView{q, DataType::Float32(), shape, strides},
                                                                   .k = TensorView{k, DataType::Float32(), shape, strides},
                                                                   .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                                   .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                                                   .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                                           });
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    ASSERT_TRUE(RunRoPEEntry(*kernel, *prepared).ok());
    q_output.fill(17.0F);
    k_output.fill(19.0F);
    positions[0] = 1;
    EXPECT_EQ(RunRoPEEntry(*kernel, *prepared).code(), StatusCode::kOverflow);
    for (float value: q_output) EXPECT_EQ(value, 17.0F);
    for (float value: k_output) EXPECT_EQ(value, 19.0F);
}

TEST(CPUKernelRoPE, PreparedParamsRejectFinitePositionAngleOverflowBeforeInPlaceWrites) {
    constexpr int64_t shape[2] = {2, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    std::array<float, 8> q = {1.0F, 2.0F, 3.0F, 4.0F,
                              -1.0F, -2.0F, -3.0F, -4.0F};
    std::array<float, 8> k = {-0.5F, 0.5F, -1.5F, 1.5F,
                              -2.5F, 2.5F, -3.5F, 3.5F};
    const auto original_q = q;
    const auto original_k = k;
    int64_t positions[2] = {0, 1};
    auto params = MakeRoPEParams(4, 1, 1, 1.0 / 16.0);
    params.algorithm = LinearRoPE{.factor = std::numeric_limits<double>::min()};
    const auto kernel = PrepareRoPEKernel(params);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const auto prepared = BuildRoPEPreparedParams(*kernel, RoPETestViews{
                                                                   .q = TensorView{q.data(), DataType::Float32(), shape, strides},
                                                                   .k = TensorView{k.data(), DataType::Float32(), shape, strides},
                                                                   .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                                   .q_output = MutableTensorView{q.data(), DataType::Float32(), shape, strides},
                                                                   .k_output = MutableTensorView{k.data(), DataType::Float32(), shape, strides},
                                                           });
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    EXPECT_EQ(RunRoPEEntry(*kernel, *prepared).code(), StatusCode::kOverflow);
    EXPECT_EQ(q, original_q);
    EXPECT_EQ(k, original_k);

    positions[1] = 0;
    ASSERT_TRUE(RunRoPEEntry(*kernel, *prepared).ok());
    EXPECT_EQ(q, original_q);
    EXPECT_EQ(k, original_k);
}

TEST(CPUKernelRoPE, ReferenceSupportsInterleavedPairingAndPartialRotaryTail) {
    constexpr int64_t shape[2] = {1, 6};
    constexpr int64_t strides[2] = {6, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[1] = {1};
    constexpr float q[6] = {1.0F, 2.0F, 3.0F, 4.0F, 9.0F, -5.0F};
    constexpr float k[6] = {-1.0F, 0.5F, 2.0F, -3.0F, 7.0F, 11.0F};
    std::array<float, 6> q_output{};
    std::array<float, 6> k_output{};
    auto params = MakeRoPEParams(6, 1, 1);
    params.rotary_dim = 4;
    params.pairing = RoPEPairing::kInterleaved;
    ASSERT_TRUE(RunRoPEEntry(params, RoPETestViews{
                                             .q = TensorView{q, DataType::Float32(), shape, strides},
                                             .k = TensorView{k, DataType::Float32(), shape, strides},
                                             .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                             .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                             .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                     })
                        .ok());
    const double c0 = std::cos(1.0);
    const double s0 = std::sin(1.0);
    const double c1 = std::cos(0.5);
    const double s1 = std::sin(0.5);
    EXPECT_NEAR(q_output[0], q[0] * c0 - q[1] * s0, 1.0e-6F);
    EXPECT_NEAR(q_output[1], q[1] * c0 + q[0] * s0, 1.0e-6F);
    EXPECT_NEAR(q_output[2], q[2] * c1 - q[3] * s1, 1.0e-6F);
    EXPECT_NEAR(q_output[3], q[3] * c1 + q[2] * s1, 1.0e-6F);
    EXPECT_EQ(q_output[4], q[4]);
    EXPECT_EQ(q_output[5], q[5]);
    EXPECT_EQ(k_output[4], k[4]);
    EXPECT_EQ(k_output[5], k[5]);
}

TEST(CPUKernelRoPE, DynamicNtkReuseAndLongRopeContextSwitchUseRuntimePositions) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr float q[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float k[4] = {-1.0F, 0.5F, 2.0F, -3.0F};
    int64_t positions[1] = {3};
    std::array<float, 4> q_output{};
    std::array<float, 4> k_output{};
    const auto views = RoPETestViews{
            .q = TensorView{q, DataType::Float32(), shape, strides},
            .k = TensorView{k, DataType::Float32(), shape, strides},
            .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
            .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
            .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
    };
    auto dynamic = MakeRoPEParams(4, 1, 1);
    dynamic.algorithm = DynamicNtkRoPE{.factor = 2.0, .original_context_length = 4};
    const auto dynamic_kernel = PrepareRoPEKernel(dynamic);
    ASSERT_TRUE(dynamic_kernel.ok()) << dynamic_kernel.status().ToString();
    const auto dynamic_prepared = BuildRoPEPreparedParams(*dynamic_kernel, views);
    ASSERT_TRUE(dynamic_prepared.ok()) << dynamic_prepared.status().ToString();
    ASSERT_TRUE(RunRoPEEntry(*dynamic_kernel, *dynamic_prepared).ok());
    const float before_threshold = q_output[1];
    positions[0] = 7;
    ASSERT_TRUE(RunRoPEEntry(*dynamic_kernel, *dynamic_prepared).ok());
    EXPECT_NE(before_threshold, q_output[1]);
    const double dynamic_angle = 7.0 / 6.0;
    EXPECT_NEAR(q_output[1], q[1] * std::cos(dynamic_angle) - q[3] * std::sin(dynamic_angle),
                1.0e-6F);

    auto long_rope = MakeRoPEParams(4, 1, 1);
    long_rope.algorithm = LongRoPE{.short_factors = {1.0, 1.0},
                                   .long_factors = {2.0, 4.0},
                                   .original_context_length = 4,
                                   .rotary_output_scale = 1.5};
    const auto long_kernel = PrepareRoPEKernel(long_rope);
    ASSERT_TRUE(long_kernel.ok()) << long_kernel.status().ToString();
    positions[0] = 3;
    const auto long_prepared = BuildRoPEPreparedParams(*long_kernel, views);
    ASSERT_TRUE(long_prepared.ok()) << long_prepared.status().ToString();
    ASSERT_TRUE(RunRoPEEntry(*long_kernel, *long_prepared).ok());
    const float short_value = q_output[1];
    positions[0] = 4;
    ASSERT_TRUE(RunRoPEEntry(*long_kernel, *long_prepared).ok());
    EXPECT_NE(short_value, q_output[1]);
    const double long_angle = 4.0 * 0.125;
    EXPECT_NEAR(q_output[1],
                1.5 * (q[1] * std::cos(long_angle) - q[3] * std::sin(long_angle)),
                1.0e-6F);
}

TEST(CPUKernelRoPE, DynamicNtkRejectsSubunitBaseFrequencyOverflowBeforeWrites) {
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t shape[2] = {1, kHeadDim};
    constexpr int64_t strides[2] = {kHeadDim, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[1] = {0};
    std::array<float, kHeadDim> q{};
    std::array<float, kHeadDim> k{};
    std::array<float, kHeadDim> q_output{};
    std::array<float, kHeadDim> k_output{};
    q_output.fill(17.0F);
    k_output.fill(19.0F);

    auto params = MakeRoPEParams(kHeadDim, 1, 1,
                                 std::numeric_limits<double>::denorm_min());
    params.algorithm = DynamicNtkRoPE{.factor = 1.0, .original_context_length = 4};
    const Status status = RunRoPEEntry(params, RoPETestViews{
                                                       .q = TensorView{q.data(), DataType::Float32(), shape, strides},
                                                       .k = TensorView{k.data(), DataType::Float32(), shape, strides},
                                                       .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                       .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                                       .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                               });
    EXPECT_EQ(status.code(), StatusCode::kOverflow);
    for (float value: q_output) EXPECT_EQ(value, 17.0F);
    for (float value: k_output) EXPECT_EQ(value, 19.0F);
}

TEST(CPUKernelRoPE, ReferenceConsumesYarnAndLlama3StaticFrequencyTables) {
    constexpr int64_t shape[2] = {1, 4};
    constexpr int64_t strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {1};
    constexpr int64_t position_strides[1] = {1};
    constexpr int64_t positions[1] = {3};
    constexpr float values[4] = {1.0F, 2.0F, 3.0F, 4.0F};

    auto params = MakeRoPEParams(4, 1, 1, 4.0);
    const auto run_and_check = [&](RoPEAlgorithmParams algorithm) {
        params.algorithm = std::move(algorithm);
        const auto frequencies = ResolveStaticRoPEFreqs(params);
        ASSERT_TRUE(frequencies.ok()) << frequencies.status().ToString();
        std::array<float, 4> q_output{};
        std::array<float, 4> k_output{};
        ASSERT_TRUE(RunRoPEEntry(params, RoPETestViews{
                                                 .q = TensorView{values, DataType::Float32(), shape, strides},
                                                 .k = TensorView{values, DataType::Float32(), shape, strides},
                                                 .position_ids = TensorView{positions, DataType::Int(64), position_shape, position_strides},
                                                 .q_output = MutableTensorView{q_output.data(), DataType::Float32(), shape, strides},
                                                 .k_output = MutableTensorView{k_output.data(), DataType::Float32(), shape, strides},
                                         })
                            .ok());
        for (int64_t pair = 0; pair < 2; ++pair) {
            const double angle = 3.0 * frequencies->inv_freqs[static_cast<size_t>(pair)];
            const double cosine = std::cos(angle) * frequencies->rotary_output_scale;
            const double sine = std::sin(angle) * frequencies->rotary_output_scale;
            const int64_t second = pair + 2;
            const float expected_first = static_cast<float>(values[pair] * cosine - values[second] * sine);
            const float expected_second = static_cast<float>(values[second] * cosine + values[pair] * sine);
            EXPECT_NEAR(q_output[pair], expected_first, 1.0e-6F);
            EXPECT_NEAR(q_output[second], expected_second, 1.0e-6F);
            EXPECT_EQ(k_output[pair], q_output[pair]);
            EXPECT_EQ(k_output[second], q_output[second]);
        }
    };

    run_and_check(YarnRoPE{.factor = 2.0,
                           .original_context_length = 8,
                           .beta_fast = 32.0,
                           .beta_slow = 1.0,
                           .rotary_output_scale = 1.5,
                           .truncate_correction_range = false});
    run_and_check(Llama3RoPE{.factor = 2.0,
                             .low_frequency_factor = 1.0,
                             .high_frequency_factor = 4.0,
                             .original_context_length = 16});
}

TEST(CPUKernelRoPE, ExecutionPlanBuilderRunsPreparedReferenceKernelWithTwoOutputs) {
    RuntimeBuilder builder;
    Runtime runtime = builder.Build();
    const auto params = MakeRoPEParams(4, 2, 1);
    const std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 4})},
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({2})},
    };
    const auto analyzed = InferOperator(OpType::kRoPE, OpParams{params}, inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    const std::vector<ExecutionPlanNodeSpec> nodes = {
            ExecutionPlanNodeSpec{
                    .op_type = OpType::kRoPE,
                    .selector = MakeRoPESelector(),
                    .input_specs = inputs,
                    .output_specs = analyzed->outputs,
                    .runtime_checks = analyzed->runtime_checks,
                    .op_params = OpParams{params},
            },
    };
    const auto plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    constexpr int64_t q_shape[2] = {2, 8};
    constexpr int64_t q_strides[2] = {8, 1};
    constexpr int64_t k_shape[2] = {2, 4};
    constexpr int64_t k_strides[2] = {4, 1};
    constexpr int64_t position_shape[1] = {2};
    constexpr int64_t position_strides[1] = {1};
    constexpr float q[16] = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, 0.5F, 2.0F, -3.0F,
                             2.0F, -1.0F, 0.5F, 3.0F, 4.0F, -2.0F, 1.5F, 0.25F};
    constexpr float k[8] = {-2.0F, 1.0F, 0.5F, 3.0F, 1.5F, -0.5F, 4.0F, -2.0F};
    constexpr int64_t positions[2] = {0, 1};
    std::array<float, 16> q_output{};
    std::array<float, 8> k_output{};
    test::ExecutionBindingCollector collector(*plan, runtime.GetAllocator(Device::CPU()));
    collector.Set(0, StepTensorBinding{
                             .inputs = {
                                     TensorView{q, DataType::Float32(), q_shape, q_strides},
                                     TensorView{k, DataType::Float32(), k_shape, k_strides},
                                     TensorView{positions, DataType::Int(64), position_shape, position_strides},
                             },
                             .outputs = {
                                     MutableTensorView{q_output.data(), DataType::Float32(), q_shape, q_strides},
                                     MutableTensorView{k_output.data(), DataType::Float32(), k_shape, k_strides},
                             },
                     });
    auto context = collector.CreateContext();
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());
    ExpectRoPENear(q, q_output.data(), 2, 2, 4, 8, 1, 8, 1, positions, 1, 4.0, 1.0);
    ExpectRoPENear(k, k_output.data(), 2, 1, 4, 4, 1, 4, 1, positions, 1, 4.0, 1.0);
}

} // namespace
