#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "fixtures/rope_hf_v4_57_1.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

enum class Layout { kContiguous,
                    kStrided,
                    kInPlace };

struct FixtureParam {
    size_t case_index;
    Layout layout;
};

// Covers the fixed fixture domain, including FP32 HF phase rounding at long
// positions. This is a compatibility gate, not a claim of HF bit equivalence.
constexpr double kHfAbsoluteTolerance = 1.0e-3;
constexpr double kHfRelativeTolerance = 1.0e-5;
constexpr float kPadding = -9876.0F;

struct StridedBuffer {
    std::array<int64_t, 2> shape;
    std::array<int64_t, 2> strides;
    std::vector<float> data;

    StridedBuffer(int64_t rows, int64_t columns, int64_t column_stride, int64_t padding)
        : shape{rows, columns},
          strides{columns * column_stride + padding, column_stride},
          data(static_cast<size_t>(rows * strides[0]), kPadding) {}

    void Assign(std::span<const float> logical) {
        for (int64_t row = 0; row < shape[0]; ++row) {
            for (int64_t col = 0; col < shape[1]; ++col) {
                data[row * strides[0] + col * strides[1]] = logical[row * shape[1] + col];
            }
        }
    }

    TensorView ReadOnly() const {
        return TensorView{data.data(), DataType::Float32(), shape, strides};
    }

    MutableTensorView Writable() {
        return MutableTensorView{data.data(), DataType::Float32(), shape, strides};
    }

    double ExpectMatches(std::span<const float> expected) const {
        double max_error = 0.0;
        for (int64_t row = 0; row < shape[0]; ++row) {
            for (int64_t offset = 0; offset < strides[0]; ++offset) {
                const float actual = data[row * strides[0] + offset];
                const int64_t col = offset / strides[1];
                if (offset % strides[1] != 0 || col >= shape[1]) {
                    EXPECT_EQ(actual, kPadding) << "padding row=" << row << " offset=" << offset;
                    continue;
                }
                const double golden = expected[row * shape[1] + col];
                const double error = std::abs(static_cast<double>(actual) - golden);
                EXPECT_TRUE(std::isfinite(actual)) << "row=" << row << " col=" << col;
                EXPECT_LE(error, kHfAbsoluteTolerance + kHfRelativeTolerance * std::abs(golden))
                        << "row=" << row << " col=" << col << " actual=" << actual << " HF=" << golden;
                max_error = std::max(max_error, error);
            }
        }
        return max_error;
    }
};

class CPUKernelRoPEHf : public ::testing::TestWithParam<FixtureParam> {};

TEST_P(CPUKernelRoPEHf, MatchesPinnedCpuFloat32Golden) {
    const auto param = GetParam();
    const auto& fixture = test::rope_hf::kCases[param.case_index];
    const auto seq_len = static_cast<int64_t>(fixture.positions.size());
    const bool strided = param.layout != Layout::kContiguous;
    StridedBuffer q(seq_len, fixture.q_heads * fixture.head_dim, strided ? 2 : 1, strided ? 5 : 0);
    StridedBuffer k(seq_len, fixture.kv_heads * fixture.head_dim, strided ? 2 : 1, strided ? 7 : 0);
    q.Assign(fixture.q);
    k.Assign(fixture.k);
    StridedBuffer q_output(seq_len, q.shape[1], strided ? 3 : 1, strided ? 4 : 0);
    StridedBuffer k_output(seq_len, k.shape[1], strided ? 3 : 1, strided ? 6 : 0);
    auto& actual_q = param.layout == Layout::kInPlace ? q : q_output;
    auto& actual_k = param.layout == Layout::kInPlace ? k : k_output;
    const std::array<int64_t, 1> position_shape{seq_len};
    const std::array<int64_t, 1> position_stride{strided ? 2 : 1};
    std::vector<int64_t> positions(static_cast<size_t>(seq_len * position_stride[0]), -1);
    for (int64_t row = 0; row < seq_len; ++row) {
        positions[row * position_stride[0]] = fixture.positions[row];
    }
    const auto original_positions = positions;

    CpuBackend backend;
    RoPEParams params{
            .head_dim = fixture.head_dim,
            .num_attention_heads = fixture.q_heads,
            .num_key_value_heads = fixture.kv_heads,
            .max_pos_embeddings = 8192,
            .theta = fixture.theta,
    };
    if (fixture.linear) {
        params.algorithm = LinearRoPE{.factor = fixture.factor};
    }
    const auto kernel = backend.PrepareKernel(
            OpType::kRoPE,
            KernelSelector{.device_type = DeviceType::kCPU,
                           .act_dtype = DataType::Float32(),
                           .weight_dtype = DataType::Float32(),
                           .weight_format = WeightFormat::kPlain,
                           .phase = ExecPhase::kBoth},
            OpParams{params});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const std::array inputs{q.ReadOnly(), k.ReadOnly(),
                            TensorView{positions.data(), DataType::Int(64), position_shape, position_stride}};
    const std::array outputs{actual_q.Writable(), actual_k.Writable()};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> prepared{};
    const auto built = kernel->params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs},
            prepared.data());
    ASSERT_TRUE(built.ok()) << built.ToString();
    const auto status = kernel->fn(KernelContext{.kernel_params = prepared.data(), .attrs = kernel->attrs});
    ASSERT_TRUE(status.ok()) << status.ToString();

    const double max_q_error = actual_q.ExpectMatches(fixture.q_output);
    const double max_k_error = actual_k.ExpectMatches(fixture.k_output);
    RecordProperty("MaxAbsoluteError", std::to_string(std::max(max_q_error, max_k_error)));
    EXPECT_EQ(positions, original_positions);
    if (param.layout != Layout::kInPlace) {
        // Read-only inputs must survive out-of-place execution exactly.
        for (int64_t row = 0; row < seq_len; ++row) {
            for (int64_t col = 0; col < q.shape[1]; ++col) {
                EXPECT_EQ(q.data[row * q.strides[0] + col * q.strides[1]], fixture.q[row * q.shape[1] + col]);
            }
            for (int64_t col = 0; col < k.shape[1]; ++col) {
                EXPECT_EQ(k.data[row * k.strides[0] + col * k.strides[1]], fixture.k[row * k.shape[1] + col]);
            }
        }
    }
}

std::vector<FixtureParam> FixtureParams() {
    std::vector<FixtureParam> params;
    for (size_t index = 0; index < std::size(test::rope_hf::kCases); ++index) {
        for (auto layout: {Layout::kContiguous, Layout::kStrided, Layout::kInPlace}) {
            params.push_back({index, layout});
        }
    }
    return params;
}

std::string FixtureName(const ::testing::TestParamInfo<FixtureParam>& info) {
    constexpr const char* layouts[] = {"Contiguous", "Strided", "InPlace"};
    return std::string(test::rope_hf::kCases[info.param.case_index].name) + layouts[static_cast<size_t>(info.param.layout)];
}

INSTANTIATE_TEST_SUITE_P(Layouts, CPUKernelRoPEHf, ::testing::ValuesIn(FixtureParams()), FixtureName);

} // namespace
