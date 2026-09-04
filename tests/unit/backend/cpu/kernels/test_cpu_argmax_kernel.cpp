#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/argmax/argmax_internal.h"
#include "execution/test_execution_binding_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {

using namespace aethermind;
using cpu::detail::ArgmaxF32KernelArgs;

/// Value written to output slots the kernel must never touch, so a stray write
/// to padding shows up as a comparison failure.
constexpr int64_t kSentinel = -987654321;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

KernelSelector MakeArgmaxSelector(const DataType& act_dtype = DataType::Float32()) {
    // ArgMax has no weight port, so DeriveSelectorDTypes falls back to act_dtype
    // for weight_dtype; the descriptor advertises the same pair.
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .weight_dtype = act_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareArgmaxKernel(int64_t axis = -1) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kArgmax,
            MakeArgmaxSelector(),
            OpParams{ArgmaxParams{.axis = axis}});
}

/// Mirror of the two views the ArgMax params builder validates at binding time.
struct ArgmaxTestViews {
    TensorView input_tensor{};
    MutableTensorView output_tensor{};
};

/// Buffer that plays the role of the PreparedExecutionBindings params arena slot.
struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

StatusOr<PreparedKernelParams> BuildArgmaxPreparedParams(
        const ResolvedKernel& kernel,
        const ArgmaxTestViews& views) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 1> inputs{views.input_tensor};
    const std::array<MutableTensorView, 1> outputs{views.output_tensor};
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

Status RunArgmaxEntry(const ResolvedKernel& kernel,
                      const PreparedKernelParams& prepared) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .attrs = kernel.attrs,
    });
}

Status RunArgmaxEntryWith(const ResolvedKernel& kernel,
                          const ArgmaxTestViews& views) noexcept {
    const auto prepared = BuildArgmaxPreparedParams(kernel, views);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return RunArgmaxEntry(kernel, *prepared);
}

Status RunArgmaxWithAxis(int64_t axis, const ArgmaxTestViews& views) noexcept {
    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(axis);
    if (!kernel.ok()) {
        return kernel.status();
    }
    return RunArgmaxEntryWith(*kernel, views);
}

int64_t DotOffset(const std::vector<int64_t>& coord,
                  const std::vector<int64_t>& strides) noexcept {
    int64_t offset = 0;
    for (size_t dim = 0; dim < coord.size(); ++dim) {
        offset += coord[dim] * strides[dim];
    }
    return offset;
}

/// Enumerates every coordinate of `shape` in row-major order; an empty shape
/// yields the single rank-0 coordinate.
std::vector<std::vector<int64_t>> EnumerateCoordinates(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (const int64_t extent: shape) {
        total *= extent;
    }

    std::vector<std::vector<int64_t>> coords;
    coords.reserve(static_cast<size_t>(total));
    std::vector<int64_t> coord(shape.size(), 0);
    for (int64_t index = 0; index < total; ++index) {
        coords.push_back(coord);
        for (size_t dim = shape.size(); dim-- > 0;) {
            if (++coord[dim] < shape[dim]) {
                break;
            }
            coord[dim] = 0;
        }
    }
    return coords;
}

/// Picks the winning index of one slice: the first NaN wins, otherwise the
/// lowest index of the strict maximum.
int64_t SelectExpectedWinner(const std::vector<float>& slice) noexcept {
    for (size_t index = 0; index < slice.size(); ++index) {
        if (std::isnan(slice[index])) {
            return static_cast<int64_t>(index);
        }
    }

    int64_t best_index = 0;
    for (size_t index = 1; index < slice.size(); ++index) {
        if (slice[index] > slice[static_cast<size_t>(best_index)]) {
            best_index = static_cast<int64_t>(index);
        }
    }
    return best_index;
}

/// Fills `expected` by enumerating output coordinates and scanning each
/// reduction slice, an independent path from the kernel's flat-index decoding.
/// Slots no output coordinate maps to keep their initial sentinel value.
void FillExpectedArgmax(const std::vector<float>& input,
                        const std::vector<int64_t>& input_shape,
                        const std::vector<int64_t>& input_strides,
                        int64_t axis,
                        const std::vector<int64_t>& output_shape,
                        const std::vector<int64_t>& output_strides,
                        std::vector<int64_t>& expected) {
    const auto input_rank = static_cast<int64_t>(input_shape.size());
    const int64_t reduction_axis = axis < 0 ? axis + input_rank : axis;
    const int64_t reduction_size = input_shape[static_cast<size_t>(reduction_axis)];

    for (const auto& output_coord: EnumerateCoordinates(output_shape)) {
        std::vector<int64_t> input_coord(input_shape.size(), 0);
        size_t output_axis = 0;
        for (size_t dim = 0; dim < input_shape.size(); ++dim) {
            if (static_cast<int64_t>(dim) == reduction_axis) {
                continue;
            }
            input_coord[dim] = output_coord[output_axis++];
        }

        std::vector<float> slice;
        slice.reserve(static_cast<size_t>(reduction_size));
        for (int64_t reduction_index = 0; reduction_index < reduction_size;
             ++reduction_index) {
            input_coord[static_cast<size_t>(reduction_axis)] = reduction_index;
            slice.push_back(input[static_cast<size_t>(
                    DotOffset(input_coord, input_strides))]);
        }

        expected[static_cast<size_t>(DotOffset(output_coord, output_strides))] =
                SelectExpectedWinner(slice);
    }
}

/// One fully described layout case: physical storage, logical geometry, axis.
struct ArgmaxLayoutCase {
    std::vector<float> input;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_strides;
    int64_t axis = -1;
    std::vector<int64_t> output_shape;
    std::vector<int64_t> output_strides;
    int64_t output_storage_size = 0;
};

/// Runs one layout case through the prepared kernel and compares every physical
/// output slot against the independent reference, padding slots included.
void ExpectArgmaxMatchesReference(const ArgmaxLayoutCase& test_case) {
    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(test_case.axis);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    std::vector<int64_t> output(
            static_cast<size_t>(test_case.output_storage_size), kSentinel);
    const Status status = RunArgmaxEntryWith(
            *kernel,
            ArgmaxTestViews{
                    .input_tensor = TensorView{test_case.input.data(),
                                               DataType::Float32(),
                                               IntArrayView{test_case.input_shape},
                                               IntArrayView{test_case.input_strides}},
                    .output_tensor = MutableTensorView{output.data(),
                                                       DataType::Int(64),
                                                       IntArrayView{test_case.output_shape},
                                                       IntArrayView{test_case.output_strides}},
            });
    ASSERT_TRUE(status.ok()) << status.ToString();

    std::vector<int64_t> expected(output.size(), kSentinel);
    FillExpectedArgmax(test_case.input,
                       test_case.input_shape,
                       test_case.input_strides,
                       test_case.axis,
                       test_case.output_shape,
                       test_case.output_strides,
                       expected);
    EXPECT_EQ(output, expected);
}

/// Runs a rank-1 input with unit stride and a rank-0 output, the scalar case.
void ExpectScalarArgmax(std::initializer_list<float> values, int64_t expected_index) {
    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const std::vector<float> input(values);
    const int64_t input_shape[1] = {static_cast<int64_t>(input.size())};
    const int64_t input_strides[1] = {1};
    int64_t output = kSentinel;

    const Status status = RunArgmaxEntryWith(
            *kernel,
            ArgmaxTestViews{
                    .input_tensor = TensorView{input.data(), DataType::Float32(),
                                               input_shape, input_strides},
                    .output_tensor = MutableTensorView{&output, DataType::Int(64), {}, {}},
            });
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(output, expected_index);
}

/// Fills storage with a deterministic pattern whose slices contain ties, so the
/// lowest-index tie rule is exercised together with the layout.
std::vector<float> MakeTieHeavyInput(size_t storage_size) {
    std::vector<float> input(storage_size);
    for (size_t index = 0; index < storage_size; ++index) {
        input[index] = static_cast<float>((index * 37) % 11) - 5.0F;
    }
    return input;
}

TEST(CPUKernelArgmax, ResolvesReferenceDescriptor) {
    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    EXPECT_STREQ(kernel->name, "cpu::argmax_f32_reference");
    EXPECT_EQ(kernel->op_type, OpType::kArgmax);
    EXPECT_EQ(kernel->params_size, sizeof(ArgmaxF32KernelArgs));
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->attrs.size(), sizeof(int64_t));
    EXPECT_EQ(kernel->workspace_requirement.bytes, 0U);

    int64_t frozen_axis = 0;
    std::memcpy(&frozen_axis, kernel->attrs.data(), sizeof(frozen_axis));
    EXPECT_EQ(frozen_axis, -1);
}

TEST(CPUKernelArgmax, DescriptorResolvesForEveryLegalAxis) {
    for (const int64_t axis: {int64_t{-8}, int64_t{-1}, int64_t{0}, int64_t{7}}) {
        const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(axis);
        ASSERT_TRUE(kernel.ok()) << "axis " << axis << ": " << kernel.status().ToString();
    }
}

TEST(CPUKernelArgmax, PrepareRejectsFloat16Selector) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kArgmax,
            MakeArgmaxSelector(DataType::Float(16)),
            OpParams{ArgmaxParams{.axis = -1}});
    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kNotFound) << kernel.status().ToString();
}

TEST(CPUKernelArgmax, PrepareRejectsBFloat16Selector) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kArgmax,
            MakeArgmaxSelector(DataType::BFloat(16)),
            OpParams{ArgmaxParams{.axis = -1}});
    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kNotFound) << kernel.status().ToString();
}

TEST(CPUKernelArgmaxEntry, PrepareRejectsWrongOpParams) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kArgmax,
            MakeArgmaxSelector(),
            OpParams{RmsNormParams{.eps = 1.0e-5F}});
    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kInvalidArgument)
            << kernel.status().ToString();
}

TEST(CPUKernelArgmax, RankOneInputProducesScalarOutput) {
    ExpectScalarArgmax({1.0F, 3.0F, 2.0F, 0.5F, -1.0F}, 1);
}

TEST(CPUKernelArgmax, TiesSelectLowestIndex) {
    ExpectScalarArgmax({1.0F, 3.0F, 3.0F, 3.0F, 2.0F}, 1);
}

TEST(CPUKernelArgmax, AllEqualValuesSelectFirstIndex) {
    ExpectScalarArgmax({2.0F, 2.0F, 2.0F}, 0);
}

TEST(CPUKernelArgmax, NegativeValuesSelectMaximum) {
    ExpectScalarArgmax({-5.0F, -1.0F, -3.0F}, 1);
}

TEST(CPUKernelArgmax, PositiveInfinityWins) {
    const float infinity = std::numeric_limits<float>::infinity();
    ExpectScalarArgmax({-infinity, 0.0F, infinity, 5.0F}, 2);
}

TEST(CPUKernelArgmax, AllNegativeInfinitySelectsFirstIndex) {
    const float infinity = std::numeric_limits<float>::infinity();
    ExpectScalarArgmax({-infinity, -infinity, -infinity}, 0);
}

TEST(CPUKernelArgmax, LeadingNanWins) {
    ExpectScalarArgmax({std::nanf(""), 9.0F, 8.0F}, 0);
}

TEST(CPUKernelArgmax, MiddleNanWins) {
    ExpectScalarArgmax({1.0F, std::nanf(""), 9.0F}, 1);
}

TEST(CPUKernelArgmax, FirstOfMultipleNansWins) {
    ExpectScalarArgmax({1.0F, std::nanf(""), std::nanf(""), 9.0F}, 1);
}

TEST(CPUKernelArgmax, NanWinsOverInfinity) {
    const float infinity = std::numeric_limits<float>::infinity();
    ExpectScalarArgmax({infinity, std::nanf(""), infinity}, 1);
}

TEST(CPUKernelArgmax, TrailingNanWinsOverEarlierMaximum) {
    ExpectScalarArgmax({1.0F, 2.0F, 3.0F, std::nanf("")}, 3);
}

TEST(CPUKernelArgmax, SingleElementSliceSelectsIndexZero) {
    ExpectScalarArgmax({std::nanf("")}, 0);
    ExpectScalarArgmax({42.0F}, 0);
}

TEST(CPUKernelArgmax, PerRowSemanticsWithTiesAndNans) {
    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const float input[12] = {
            1.0F,
            2.0F,
            2.0F,
            1.0F,
            std::numeric_limits<float>::quiet_NaN(),
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            4.0F,
            1.0F,
            1.0F,
    };
    constexpr int64_t input_shape[2] = {3, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[1] = {3};
    constexpr int64_t output_strides[1] = {1};
    std::array<int64_t, 3> output{kSentinel, kSentinel, kSentinel};

    const Status status = RunArgmaxEntryWith(
            *kernel,
            ArgmaxTestViews{
                    .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                               input_strides},
                    .output_tensor = MutableTensorView{output.data(), DataType::Int(64),
                                                       output_shape, output_strides},
            });
    ASSERT_TRUE(status.ok()) << status.ToString();

    // Row 0 ties at index 1, row 1 starts with NaN, row 2 ties at index 0.
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 0);
}

TEST(CPUKernelArgmax, ReducesFirstAxisOfContiguousRankTwo) {
    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = MakeTieHeavyInput(12),
            .input_shape = {3, 4},
            .input_strides = {4, 1},
            .axis = 0,
            .output_shape = {4},
            .output_strides = {1},
            .output_storage_size = 4,
    });
}

TEST(CPUKernelArgmax, ReducesMiddleAxisOfContiguousRankThree) {
    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = MakeTieHeavyInput(24),
            .input_shape = {2, 3, 4},
            .input_strides = {12, 4, 1},
            .axis = 1,
            .output_shape = {2, 4},
            .output_strides = {4, 1},
            .output_storage_size = 8,
    });
}

TEST(CPUKernelArgmax, NegativeAxisMatchesEquivalentPositiveAxis) {
    const ArgmaxLayoutCase positive{
            .input = MakeTieHeavyInput(24),
            .input_shape = {2, 3, 4},
            .input_strides = {12, 4, 1},
            .axis = 1,
            .output_shape = {2, 4},
            .output_strides = {4, 1},
            .output_storage_size = 8,
    };
    ArgmaxLayoutCase negative = positive;
    negative.axis = -2;

    ExpectArgmaxMatchesReference(positive);
    ExpectArgmaxMatchesReference(negative);
}

TEST(CPUKernelArgmax, SupportsPaddedInputAndPaddedOutput) {
    // Rows live at offsets 0 and 5 with slots 3, 4, 8, 9 holding poison values
    // that no slice may read; output slots 1 and 2 must stay untouched.
    constexpr float poison = 1.0e30F;
    std::vector<float> input(10, poison);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
            input[static_cast<size_t>(row * 5 + col)] =
                    static_cast<float>((row * 3 + col) * 29 % 7) - 3.0F;
        }
    }

    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = input,
            .input_shape = {2, 3},
            .input_strides = {5, 1},
            .axis = -1,
            .output_shape = {2},
            .output_strides = {3},
            .output_storage_size = 4,
    });
}

TEST(CPUKernelArgmax, SupportsTransposedInput) {
    // Column-major storage: element (i, j) lives at offset i + 2 * j.
    std::vector<float> input(6);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
            input[static_cast<size_t>(row + 2 * col)] =
                    static_cast<float>((row * 5 + col * 3) % 7) - 2.0F;
        }
    }

    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = input,
            .input_shape = {2, 3},
            .input_strides = {1, 2},
            .axis = -1,
            .output_shape = {2},
            .output_strides = {1},
            .output_storage_size = 2,
    });
}

TEST(CPUKernelArgmax, SupportsStridedReductionAxisWithGaps) {
    constexpr float poison = 1.0e30F;
    std::vector<float> input(13, poison);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
            input[static_cast<size_t>(row * 8 + col * 2)] =
                    static_cast<float>((row + col * 11) % 5) - 2.0F;
        }
    }

    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = input,
            .input_shape = {2, 3},
            .input_strides = {8, 2},
            .axis = -1,
            .output_shape = {2},
            .output_strides = {1},
            .output_storage_size = 2,
    });
}

TEST(CPUKernelArgmax, SupportsStridedRankOneInputWithScalarOutput) {
    constexpr float poison = 1.0e30F;
    std::vector<float> input(9, poison);
    for (int64_t index = 0; index < 5; ++index) {
        input[static_cast<size_t>(index * 2)] = static_cast<float>(index % 3) - 1.0F;
    }

    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = input,
            .input_shape = {5},
            .input_strides = {2},
            .axis = 0,
            .output_shape = {},
            .output_strides = {},
            .output_storage_size = 1,
    });
}

TEST(CPUKernelArgmax, SupportsUnitExtentsOnBothSidesOfTheAxis) {
    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = MakeTieHeavyInput(12),
            .input_shape = {1, 3, 1, 4, 1},
            .input_strides = {12, 4, 4, 1, 1},
            .axis = 1,
            .output_shape = {1, 1, 4, 1},
            .output_strides = {12, 4, 1, 1},
            .output_storage_size = 4,
    });
}

TEST(CPUKernelArgmax, SupportsRankEightInputReducingLastAxis) {
    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = MakeTieHeavyInput(48),
            .input_shape = {2, 1, 2, 1, 2, 1, 2, 3},
            .input_strides = {24, 24, 12, 12, 6, 6, 3, 1},
            .axis = -1,
            .output_shape = {2, 1, 2, 1, 2, 1, 2},
            .output_strides = {8, 8, 4, 4, 2, 2, 1},
            .output_storage_size = 16,
    });
}

TEST(CPUKernelArgmax, SupportsRankEightInputReducingFirstAxis) {
    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = MakeTieHeavyInput(48),
            .input_shape = {2, 1, 2, 1, 2, 1, 2, 3},
            .input_strides = {24, 24, 12, 12, 6, 6, 3, 1},
            .axis = 0,
            .output_shape = {1, 2, 1, 2, 1, 2, 3},
            .output_strides = {24, 12, 12, 6, 6, 3, 1},
            .output_storage_size = 24,
    });
}

TEST(CPUKernelArgmax, SupportsEveryRankFromOneToEight) {
    for (int32_t rank = 1; rank <= 8; ++rank) {
        const auto rank_size = static_cast<size_t>(rank);
        std::vector<int64_t> input_shape(rank_size, 2);
        std::vector<int64_t> input_strides(rank_size);
        int64_t stride = 1;
        for (int32_t dim = rank - 1; dim >= 0; --dim) {
            input_strides[static_cast<size_t>(dim)] = stride;
            stride *= 2;
        }

        const int64_t axis = rank / 2;
        std::vector<int64_t> output_shape;
        for (int32_t dim = 0; dim < rank; ++dim) {
            if (static_cast<int64_t>(dim) != axis) {
                output_shape.push_back(2);
            }
        }

        // The output uses its own compact layout rather than inheriting the hole
        // the reduction axis leaves in the input strides.
        std::vector<int64_t> output_strides(output_shape.size());
        int64_t output_stride = 1;
        for (size_t dim = output_shape.size(); dim-- > 0;) {
            output_strides[dim] = output_stride;
            output_stride *= 2;
        }

        ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
                .input = MakeTieHeavyInput(static_cast<size_t>(stride)),
                .input_shape = input_shape,
                .input_strides = input_strides,
                .axis = axis,
                .output_shape = output_shape,
                .output_strides = output_strides,
                .output_storage_size = output_stride,
        });
    }
}

TEST(CPUKernelArgmaxEntry, ZeroNonReductionDimensionIsSuccessfulNoOp) {
    // Empty output: the builder must accept null data and unvalidated strides.
    constexpr int64_t input_shape[3] = {2, 0, 4};
    constexpr int64_t input_strides[3] = {0, 4, 1};
    constexpr int64_t output_shape[2] = {2, 0};
    constexpr int64_t output_strides[2] = {0, 0};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{nullptr, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{nullptr, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, ZeroReductionExtentIsRejected) {
    constexpr float input = 1.0F;
    int64_t output[3] = {kSentinel, kSentinel, kSentinel};
    constexpr int64_t input_shape[2] = {3, 0};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t output_shape[1] = {3};
    constexpr int64_t output_strides[1] = {1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{&input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
    EXPECT_EQ(output[0], kSentinel);
}

TEST(CPUKernelArgmaxEntry, ZeroReductionExtentOnFirstAxisIsRejected) {
    constexpr float input = 1.0F;
    int64_t output[3] = {kSentinel, kSentinel, kSentinel};
    constexpr int64_t input_shape[2] = {0, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t output_shape[1] = {3};
    constexpr int64_t output_strides[1] = {1};

    const Status status = RunArgmaxWithAxis(0, ArgmaxTestViews{
                                                       .input_tensor = TensorView{&input, DataType::Float32(), input_shape,
                                                                                  input_strides},
                                                       .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                          output_strides},
                                               });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsRankZeroInput) {
    constexpr float input = 1.0F;
    int64_t output = kSentinel;

    const Status status = RunArgmaxWithAxis(0, ArgmaxTestViews{
                                                       .input_tensor = TensorView{&input, DataType::Float32(), {}, {}},
                                                       .output_tensor = MutableTensorView{&output, DataType::Int(64), {}, {}},
                                               });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsAxisBeyondInputRank) {
    constexpr float input[8] = {};
    int64_t output[4] = {};
    constexpr int64_t input_shape[2] = {2, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[1] = {4};
    constexpr int64_t output_strides[1] = {1};
    const ArgmaxTestViews views{
            .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                       input_strides},
            .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                               output_strides},
    };

    EXPECT_EQ(RunArgmaxWithAxis(5, views).code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(RunArgmaxWithAxis(-3, views).code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(RunArgmaxWithAxis(std::numeric_limits<int64_t>::min(), views).code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelArgmaxEntry, RejectsInvalidViews) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    int64_t output = kSentinel;
    constexpr int64_t input_shape[1] = {4};
    constexpr int64_t input_strides[1] = {1};

    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const Status missing_input = RunArgmaxEntryWith(
            *kernel,
            ArgmaxTestViews{
                    .input_tensor = TensorView{},
                    .output_tensor = MutableTensorView{&output, DataType::Int(64), {}, {}},
            });
    EXPECT_EQ(missing_input.code(), StatusCode::kInvalidArgument)
            << missing_input.ToString();

    const Status missing_output = RunArgmaxEntryWith(
            *kernel,
            ArgmaxTestViews{
                    .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                               input_strides},
                    .output_tensor = MutableTensorView{},
            });
    EXPECT_EQ(missing_output.code(), StatusCode::kInvalidArgument)
            << missing_output.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsWrongArity) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    int64_t output = kSentinel;
    constexpr int64_t input_shape[1] = {4};
    constexpr int64_t input_strides[1] = {1};

    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    PreparedKernelParams prepared;
    const std::array<TensorView, 2> inputs{
            TensorView{input, DataType::Float32(), input_shape, input_strides},
            TensorView{input, DataType::Float32(), input_shape, input_strides},
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{&output, DataType::Int(64), {}, {}}};
    const Status status = kernel->params_builder(
            KernelParamsBuildContext{.inputs = inputs,
                                     .outputs = outputs,
                                     .attrs = kernel->attrs},
            prepared.storage.data());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsMissingAttrs) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    int64_t output = kSentinel;
    constexpr int64_t input_shape[1] = {4};
    constexpr int64_t input_strides[1] = {1};

    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    PreparedKernelParams prepared;
    const std::array<TensorView, 1> inputs{
            TensorView{input, DataType::Float32(), input_shape, input_strides}};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{&output, DataType::Int(64), {}, {}}};
    const Status status = kernel->params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = {}},
            prepared.storage.data());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsWrongInputDType) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    int64_t output = kSentinel;
    constexpr int64_t input_shape[1] = {4};
    constexpr int64_t input_strides[1] = {1};

    const Status float64_status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                                .input_tensor = TensorView{input, DataType::Double(), input_shape,
                                                                                           input_strides},
                                                                .output_tensor = MutableTensorView{&output, DataType::Int(64), {}, {}},
                                                        });
    EXPECT_EQ(float64_status.code(), StatusCode::kInvalidArgument)
            << float64_status.ToString();

    const Status float16_status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                                .input_tensor = TensorView{input, DataType::Float(16), input_shape,
                                                                                           input_strides},
                                                                .output_tensor = MutableTensorView{&output, DataType::Int(64), {}, {}},
                                                        });
    EXPECT_EQ(float16_status.code(), StatusCode::kInvalidArgument)
            << float16_status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsWrongOutputDType) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    int32_t output = 0;
    constexpr int64_t input_shape[1] = {4};
    constexpr int64_t input_strides[1] = {1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{&output, DataType::Int(32), {}, {}},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsWrongOutputRank) {
    constexpr float input[8] = {};
    int64_t output[4] = {};
    constexpr int64_t input_shape[2] = {2, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsWrongOutputShape) {
    constexpr float input[8] = {};
    int64_t output[3] = {};
    constexpr int64_t input_shape[2] = {2, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[1] = {3};
    constexpr int64_t output_strides[1] = {1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsNonPositiveOutputStride) {
    constexpr float input[8] = {};
    int64_t output[2] = {};
    constexpr int64_t input_shape[2] = {2, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[1] = {2};
    constexpr int64_t output_strides[1] = {0};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, DoesNotSupportUnprovenOutputLayout) {
    // Shape [2, 2] with strides [1, 1] maps four coordinates onto two slots.
    constexpr float input[8] = {};
    int64_t output[2] = {};
    constexpr int64_t input_shape[3] = {2, 2, 2};
    constexpr int64_t input_strides[3] = {4, 2, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {1, 1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, ReportsInjectiveIrregularOutputAsUnimplemented) {
    // Shape [2, 3] with strides [3, 2] maps to offsets
    // {0, 2, 4, 3, 5, 7}. The layout is injective, but it does not satisfy the
    // kernel's inexpensive stride-span proof and therefore lies outside the
    // explicitly supported output-layout subset.
    constexpr float input[12] = {};
    std::array<int64_t, 8> output{};
    constexpr int64_t input_shape[3] = {2, 3, 2};
    constexpr int64_t input_strides[3] = {6, 2, 1};
    constexpr int64_t output_shape[2] = {2, 3};
    constexpr int64_t output_strides[2] = {3, 2};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output.data(), DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, AcceptsTransposedOutputLayout) {
    // A pure transpose is injective, so it must not be rejected as overlapping.
    ExpectArgmaxMatchesReference(ArgmaxLayoutCase{
            .input = MakeTieHeavyInput(24),
            .input_shape = {2, 3, 4},
            .input_strides = {12, 4, 1},
            .axis = -1,
            .output_shape = {2, 3},
            .output_strides = {1, 2},
            .output_storage_size = 6,
    });
}

TEST(CPUKernelArgmaxEntry, RejectsAliasedInputOutputBasePointer) {
    // ArgMax changes dtype and rank, so a shared base pointer is never a legal
    // in-place execution. Only identical base pointers are detectable: the
    // TensorView API carries no storage bounds, so partial overlap between two
    // distinct pointers can be proven neither way.
    std::array<float, 4> storage{};
    constexpr int64_t input_shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t output_shape[1] = {2};
    constexpr int64_t output_strides[1] = {2};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{storage.data(), DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{storage.data(), DataType::Int(64),
                                                                                           output_shape, output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsOutputElementCountOverflow) {
    constexpr float input = 1.0F;
    int64_t output = kSentinel;
    constexpr int64_t input_shape[3] = {std::numeric_limits<int64_t>::max(), 3, 2};
    constexpr int64_t input_strides[3] = {6, 2, 1};
    constexpr int64_t output_shape[2] = {std::numeric_limits<int64_t>::max(), 3};
    constexpr int64_t output_strides[2] = {3, 1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{&input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{&output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsInputOffsetOverflow) {
    constexpr float input = 1.0F;
    int64_t output[2] = {kSentinel, kSentinel};
    constexpr int64_t input_shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {std::numeric_limits<int64_t>::max(), 1};
    constexpr int64_t output_shape[1] = {2};
    constexpr int64_t output_strides[1] = {1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{&input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmaxEntry, RejectsOutputOffsetOverflow) {
    constexpr float input[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int64_t output = kSentinel;
    constexpr int64_t input_shape[2] = {3, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t output_shape[1] = {3};
    constexpr int64_t output_strides[1] = {std::numeric_limits<int64_t>::max() / 2 + 1};

    const Status status = RunArgmaxWithAxis(-1, ArgmaxTestViews{
                                                        .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                                                                   input_strides},
                                                        .output_tensor = MutableTensorView{&output, DataType::Int(64), output_shape,
                                                                                           output_strides},
                                                });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelArgmax, PreparedParamsExecuteRepeatedlyWithoutRevalidation) {
    const StatusOr<ResolvedKernel> kernel = PrepareArgmaxKernel(-1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    constexpr int64_t input_shape[2] = {2, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[1] = {2};
    constexpr int64_t output_strides[1] = {1};
    float input[8] = {};
    std::array<int64_t, 2> output{kSentinel, kSentinel};

    // Build the compute-ready params once, as PrepareExecutionBindings would.
    const auto prepared = BuildArgmaxPreparedParams(
            *kernel,
            ArgmaxTestViews{
                    .input_tensor = TensorView{input, DataType::Float32(), input_shape,
                                               input_strides},
                    .output_tensor = MutableTensorView{output.data(), DataType::Int(64),
                                                       output_shape, output_strides},
            });
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();

    // Repeated execution reuses the params pinned to `input`; only the input
    // contents change between runs.
    const float batches[3][8] = {
            {1.0F, 2.0F, 3.0F, 4.0F, 9.0F, 8.0F, 7.0F, 6.0F},
            {4.0F, 3.0F, 2.0F, 1.0F, 6.0F, 7.0F, 8.0F, 9.0F},
            {2.0F, 2.0F, 2.0F, 2.0F, 1.0F, 5.0F, 5.0F, 0.0F},
    };
    const int64_t expected[3][2] = {{3, 0}, {0, 3}, {0, 1}};
    for (size_t batch = 0; batch < 3; ++batch) {
        std::copy_n(batches[batch], 8, input);
        ASSERT_TRUE(RunArgmaxEntry(*kernel, *prepared).ok());
        EXPECT_EQ(output[0], expected[batch][0]) << "batch " << batch;
        EXPECT_EQ(output[1], expected[batch][1]) << "batch " << batch;
    }
}

TEST(CPUKernelArgmax, ExecutionPlanBuilderRunsResolvedKernel) {
    RuntimeBuilder builder;
    Runtime runtime = builder.Build();

    const OpParams params{ArgmaxParams{.axis = -1}};
    const std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 4})},
    };
    const auto analyzed = InferOperator(OpType::kArgmax, params, inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    ASSERT_EQ(analyzed->outputs.size(), 1U);
    EXPECT_EQ(analyzed->outputs[0].dtype, DataType::Int(64));

    const std::vector<ExecutionPlanNodeSpec> nodes = {
            ExecutionPlanNodeSpec{
                    .op_type = OpType::kArgmax,
                    .selector = MakeArgmaxSelector(),
                    .input_specs = inputs,
                    .output_specs = analyzed->outputs,
                    .runtime_checks = analyzed->runtime_checks,
                    .op_params = params,
            },
    };
    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    constexpr float input[8] = {1.0F, 2.0F, 4.0F, 3.0F, 9.0F, 8.0F, 7.0F, 6.0F};
    std::array<int64_t, 2> output{kSentinel, kSentinel};
    constexpr int64_t input_shape[2] = {2, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[1] = {2};
    constexpr int64_t output_strides[1] = {1};

    test::ExecutionBindingCollector collector(*plan, runtime.GetAllocator(Device::CPU()));
    collector.Set(0, StepTensorBinding{
                             .inputs = {
                                     TensorView{input, DataType::Float32(), input_shape,
                                                input_strides},
                             },
                             .outputs = {
                                     MutableTensorView{output.data(), DataType::Int(64), output_shape, output_strides},
                             },
                     });
    auto bindings = collector.CreateContext();
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();

    const Status status = Executor::Execute(*plan, *bindings);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 0);
}

TEST(CPUKernelArgmaxEntry, OverlappingOutputFailsAtPrepareExecutionBindings) {
    RuntimeBuilder builder;
    Runtime runtime = builder.Build();

    const OpParams params{ArgmaxParams{.axis = -1}};
    const std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 2, 2})},
    };
    const auto analyzed = InferOperator(OpType::kArgmax, params, inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    const std::vector<ExecutionPlanNodeSpec> nodes = {
            ExecutionPlanNodeSpec{
                    .op_type = OpType::kArgmax,
                    .selector = MakeArgmaxSelector(),
                    .input_specs = inputs,
                    .output_specs = analyzed->outputs,
                    .runtime_checks = analyzed->runtime_checks,
                    .op_params = params,
            },
    };
    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    constexpr float input[8] = {};
    std::array<int64_t, 2> output{kSentinel, kSentinel};
    constexpr int64_t input_shape[3] = {2, 2, 2};
    constexpr int64_t input_strides[3] = {4, 2, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t overlapping_output_strides[2] = {1, 1};
    const ExecutionStep& step = plan->steps().front();

    // The params builder runs inside PrepareExecutionBindings: the overlapping
    // output layout must be rejected here, before any kernel execution.
    const auto table = PrepareExecutionBindings(
            *plan,
            {.readable = {{.value = step.inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), input_shape,
                                                input_strides)}},
             .writable = {{.value = step.outputs[0],
                           .tensor = MutableTensorView(output.data(), DataType::Int(64),
                                                       output_shape,
                                                       overlapping_output_strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_FALSE(table.ok());
    EXPECT_EQ(table.status().code(), StatusCode::kUnimplemented)
            << table.status().ToString();
    EXPECT_EQ(output[0], kSentinel);
    EXPECT_EQ(output[1], kSentinel);
}

} // namespace
