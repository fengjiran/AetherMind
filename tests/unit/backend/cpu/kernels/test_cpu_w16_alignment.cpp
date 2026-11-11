#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <tuple>

namespace {

using namespace aethermind;

enum class ProjectionKind { kPlainLinear,
                            kPackedLinear,
                            kQkv,
                            kGateUp };

class W16ProjectionAlignment : public ::testing::TestWithParam<std::tuple<ProjectionKind, bool>> {
protected:
    size_t OutputCount() const {
        switch (std::get<0>(GetParam())) {
            case ProjectionKind::kQkv:
                return 3;
            case ProjectionKind::kGateUp:
                return 2;
            default:
                return 1;
        }
    }

    Status Prepare(size_t input_offset, std::array<size_t, 3> output_offsets,
                   int64_t rows = 1, int64_t inner = 1,
                   std::array<int64_t, 3> features = {1, 1, 1}) {
        const auto kind = std::get<0>(GetParam());
        const bool bfloat = std::get<1>(GetParam());
        const DataType dtype = bfloat ? DataType::BFloat(16) : DataType::Float(16);
        const KernelSelector selector{
                .device_type = DeviceType::kCPU,
                .act_dtype = DataType::Float32(),
                .weight_dtype = dtype,
                .weight_format = kind == ProjectionKind::kPlainLinear ? WeightFormat::kPlain : WeightFormat::kPacked,
                .phase = ExecPhase::kBoth,
        };
        OpType op = OpType::kLinear;
        OpParams attrs = LinearParams{};
        if (kind == ProjectionKind::kQkv) {
            op = OpType::kQkvLinear;
            attrs = QkvLinearParams{.q_out_features = features[0],
                                    .k_out_features = features[1],
                                    .v_out_features = features[2]};
        } else if (kind == ProjectionKind::kGateUp) {
            op = OpType::kGateUpLinear;
            attrs = GateUpLinearParams{.gate_out_features = features[0],
                                       .up_out_features = features[1]};
        }
        CpuBackend backend;
        AM_ASSIGN_OR_RETURN(kernel_, backend.PrepareKernel(op, selector, attrs));

        const float input = 2.0F;
        std::memcpy(input_.bytes.data() + input_offset, &input, sizeof(input));
        for (auto& output: outputs_) {
            output.bytes.fill(std::byte{0x5A});
        }
        int64_t total_features = 0;
        for (size_t i = 0; i < OutputCount(); ++i) {
            total_features += features[i];
        }
        const uint16_t bits = bfloat ? 0x3F80U : 0x3C00U;
        for (int64_t i = 0; i < total_features * inner; ++i) {
            std::memcpy(weights_.data() + static_cast<size_t>(i) * sizeof(bits), &bits, sizeof(bits));
        }

        const std::array<int64_t, 2> input_shape{rows, inner};
        const std::array<int64_t, 2> input_strides{inner == 0 ? 1 : inner, 1};
        const std::array<int64_t, 2> packed_shape{total_features, inner};
        const std::array<TensorView, 2> inputs{
                TensorView{input_.bytes.data() + input_offset, DataType::Float32(), input_shape, input_strides},
                TensorView{weights_.data(), dtype, packed_shape, input_strides},
        };
        std::array<std::array<int64_t, 2>, 3> output_shapes{};
        std::array<std::array<int64_t, 2>, 3> output_strides{};
        std::array<MutableTensorView, 3> outputs;
        for (size_t i = 0; i < OutputCount(); ++i) {
            output_shapes[i] = {rows, features[i]};
            output_strides[i] = {features[i] == 0 ? 1 : features[i], 1};
            outputs[i] = MutableTensorView{outputs_[i].bytes.data() + output_offsets[i], DataType::Float32(),
                                           output_shapes[i], output_strides[i]};
        }
        KernelParamsBuildContext context{
                .inputs = std::span{inputs}.first(kind == ProjectionKind::kPlainLinear ? 2 : 1),
                .outputs = std::span{outputs}.first(OutputCount()),
                .attrs = kernel_.attrs,
        };
        if (kind != ProjectionKind::kPlainLinear) {
            context.packed_weight = PackedWeightView{
                    .data = weights_.data(),
                    .nbytes = static_cast<size_t>(total_features * inner) * sizeof(bits),
                    .logical_dtype = dtype,
                    .logical_shape = packed_shape,
                    .recipe_layout = PackingLayout::kCpuIdentity,
                    .recipe_alignment = 64,
                    .alignment = 64,
            };
        }
        return kernel_.params_builder(context, params_.data());
    }

    Status Run() const {
        return kernel_.fn(KernelContext{.kernel_params = params_.data(), .attrs = kernel_.attrs});
    }

    float Output(size_t index, size_t offset = 0) const {
        float result = 0;
        std::memcpy(&result, outputs_[index].bytes.data() + offset, sizeof(result));
        return result;
    }

    void ExpectOutputUnchanged(size_t index) const {
        for (const auto byte: outputs_[index].bytes) {
            EXPECT_EQ(byte, std::byte{0x5A});
        }
    }

private:
    struct alignas(64) FloatStorage {
        std::array<std::byte, 2 * sizeof(float)> bytes{};
    };
    FloatStorage input_;
    std::array<FloatStorage, 3> outputs_;
    alignas(64) std::array<std::byte, 64> weights_{};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params_{};
    ResolvedKernel kernel_{};
};

TEST_P(W16ProjectionAlignment, RejectsMisalignedInputBeforeExecution) {
    EXPECT_EQ(Prepare(1, {}).code(), StatusCode::kInvalidArgument);
    for (size_t i = 0; i < OutputCount(); ++i) {
        ExpectOutputUnchanged(i);
    }
}

TEST_P(W16ProjectionAlignment, RejectsEachMisalignedOutputBeforeExecution) {
    for (size_t i = 0; i < OutputCount(); ++i) {
        SCOPED_TRACE(i);
        std::array<size_t, 3> offsets{};
        offsets[i] = 1;
        EXPECT_EQ(Prepare(0, offsets).code(), StatusCode::kInvalidArgument);
        ExpectOutputUnchanged(i);
    }
}

TEST_P(W16ProjectionAlignment, ZeroInnerDimensionStillRejectsMisalignedOutputs) {
    for (size_t i = 0; i < OutputCount(); ++i) {
        SCOPED_TRACE(i);
        std::array<size_t, 3> offsets{};
        offsets[i] = 1;
        EXPECT_EQ(Prepare(0, offsets, 1, 0).code(), StatusCode::kInvalidArgument);
        ExpectOutputUnchanged(i);
    }
}

TEST_P(W16ProjectionAlignment, ZeroInnerDimensionDoesNotRequireInputAlignment) {
    ASSERT_TRUE(Prepare(1, {}, 1, 0).ok());
    EXPECT_TRUE(Run().ok());
    for (size_t i = 0; i < OutputCount(); ++i) {
        EXPECT_EQ(Output(i), 0.0F);
    }
}

TEST_P(W16ProjectionAlignment, EmptyProjectionsDoNotRequireAlignment) {
    for (const bool zero_rows: {false, true}) {
        SCOPED_TRACE(zero_rows);
        const auto features = zero_rows ? std::array<int64_t, 3>{1, 1, 1} : std::array<int64_t, 3>{0, 0, 0};
        ASSERT_TRUE(Prepare(1, {1, 1, 1}, zero_rows ? 0 : 1, 1, features).ok());
        EXPECT_TRUE(Run().ok());
        for (size_t i = 0; i < OutputCount(); ++i) {
            ExpectOutputUnchanged(i);
        }
    }
}

TEST_P(W16ProjectionAlignment, EmptyOutputSplitDoesNotRequireAlignment) {
    for (size_t i = 0; i < OutputCount(); ++i) {
        SCOPED_TRACE(i);
        std::array<size_t, 3> offsets{};
        offsets[i] = 1;
        std::array<int64_t, 3> features{1, 1, 1};
        features[i] = 0;
        ASSERT_TRUE(Prepare(0, offsets, 1, 1, features).ok());
        EXPECT_TRUE(Run().ok());
        ExpectOutputUnchanged(i);
        for (size_t j = 0; j < OutputCount(); ++j) {
            if (j != i) {
                EXPECT_EQ(Output(j), 2.0F);
            }
        }
    }
}

TEST_P(W16ProjectionAlignment, AcceptsNaturalAlignmentWithoutSimdAlignment) {
    ASSERT_TRUE(Prepare(alignof(float), {alignof(float), alignof(float), alignof(float)}).ok());
    EXPECT_TRUE(Run().ok());
    for (size_t i = 0; i < OutputCount(); ++i) {
        EXPECT_EQ(Output(i, alignof(float)), 2.0F);
    }
}

INSTANTIATE_TEST_SUITE_P(
        W16Projections, W16ProjectionAlignment,
        ::testing::Combine(::testing::Values(ProjectionKind::kPlainLinear, ProjectionKind::kPackedLinear,
                                             ProjectionKind::kQkv, ProjectionKind::kGateUp),
                           ::testing::Bool()),
        ([](const auto& info) {
            const std::array names{"PlainLinear", "PackedLinear", "Qkv", "GateUp"};
            return std::string(names[static_cast<size_t>(std::get<0>(info.param))]) +
                   (std::get<1>(info.param) ? "Bfloat16" : "Float16");
        }));

} // namespace
