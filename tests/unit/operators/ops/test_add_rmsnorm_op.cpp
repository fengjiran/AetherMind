#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/ops/add_rmsnorm_op.h"

#include <cstring>
#include <gtest/gtest.h>

namespace {
using namespace aethermind;

struct StubKernelState {
    bool called = false;
    std::span<const std::byte> attrs{};
};

StubKernelState g_stub_state{};

Status StubAddRmsNormKernel(const KernelContext& ctx) noexcept {
    g_stub_state.called = true;
    g_stub_state.attrs = ctx.attrs;
    return Status::Ok();
}

void ResetStubState() {
    g_stub_state = StubKernelState{};
}

class FakeBackend final : public Backend {
public:
    StatusOr<ResolvedKernel> resolve_result{Status::NotFound("unconfigured")};

    AM_NODISCARD DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    AM_NODISCARD const BackendCapabilities& capabilities() const noexcept override {
        static const BackendCapabilities kCaps{};
        return kCaps;
    }
    AM_NODISCARD KernelFunc ResolveKernel(OpType, const KernelSelector&) const noexcept override {
        return resolve_result.ok() ? resolve_result.value().fn : nullptr;
    }
    AM_NODISCARD StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType, const KernelSelector&) const noexcept override {
        return resolve_result;
    }
    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kAddRmsNorm,
            .fn = &StubAddRmsNormKernel,
            .attrs = {},
            .debug_name = "test::stub_add_rmsnorm",
    };
}

StepTensorBinding MakeBinding() {
    static float input[8]{};
    static float residual[8]{};
    static float weight[2]{};
    static float output[8]{};
    static float new_residual[8]{};
    static const std::array<int64_t, 2> kActivationShape{4, 2};
    static const std::array<int64_t, 2> kActivationStrides{2, 1};
    static const std::array<int64_t, 1> kWeightShape{2};
    static const std::array<int64_t, 1> kWeightStrides{1};
    return {
            .inputs = {
                    TensorView(input, DataType::Float32(), kActivationShape, kActivationStrides),
                    TensorView(residual, DataType::Float32(), kActivationShape, kActivationStrides),
                    TensorView(weight, DataType::Float32(), kWeightShape, kWeightStrides),
            },
            .outputs = {
                    MutableTensorView(output, DataType::Float32(), kActivationShape, kActivationStrides),
                    MutableTensorView(new_residual, DataType::Float32(), kActivationShape, kActivationStrides),
            },
    };
}

TEST(AddRmsNormOpPrepare, ResolvesKernelWritesEpsilonAndRequiresNoWorkspace) {
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();
    AddRmsNormOp op{AddRmsNormParams{.eps = 3.0e-5F}};
    OperatorContext context{.backend = &backend};

    ASSERT_TRUE(op.Prepare(context).ok());
    EXPECT_EQ(op.ComputeWorkspaceRequirement({}).bytes, 0U);
    ASSERT_EQ(op.GetResolvedKernel().attrs.size(), sizeof(float));
    float eps = 0.0F;
    std::memcpy(&eps, op.GetResolvedKernel().attrs.data(), sizeof(eps));
    EXPECT_FLOAT_EQ(eps, 3.0e-5F);
}

TEST(AddRmsNormOpPrepare, RejectsMissingBackendAndNullKernel) {
    AddRmsNormOp op{AddRmsNormParams{}};
    OperatorContext null_context{};
    EXPECT_EQ(op.Prepare(null_context).code(), StatusCode::kInvalidArgument);

    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{.op_type = OpType::kAddRmsNorm};
    OperatorContext context{.backend = &backend};
    EXPECT_EQ(op.Prepare(context).code(), StatusCode::kInternal);
}

TEST(AddRmsNormOpRun, ValidatesArityAndDispatchesThroughResolvedKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();
    AddRmsNormOp op{AddRmsNormParams{.eps = 2.0e-5F}};
    OperatorContext context{.backend = &backend};
    ASSERT_TRUE(op.Prepare(context).ok());

    RuntimeBindingContext bindings;
    StepTensorBinding wrong = MakeBinding();
    wrong.inputs.pop_back();
    bindings.SetStepTensorBinding(0, std::move(wrong));
    KernelContext kernel_ctx;
    EXPECT_EQ(op.Run(kernel_ctx, bindings, 0).code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);

    StepTensorBinding wrong_output = MakeBinding();
    wrong_output.outputs.pop_back();
    bindings.SetStepTensorBinding(0, std::move(wrong_output));
    EXPECT_EQ(op.Run(kernel_ctx, bindings, 0).code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);

    bindings.SetStepTensorBinding(0, MakeBinding());
    kernel_ctx.attrs = op.GetResolvedKernel().attrs;
    ASSERT_TRUE(op.Run(kernel_ctx, bindings, 0).ok());
    EXPECT_TRUE(g_stub_state.called);
    ASSERT_EQ(g_stub_state.attrs.size(), sizeof(float));
}

TEST(AddRmsNormOpRun, RejectsCallBeforePrepare) {
    const AddRmsNormOp op{AddRmsNormParams{}};
    KernelContext kernel_ctx;
    RuntimeBindingContext bindings;
    EXPECT_EQ(op.Run(kernel_ctx, bindings, 0).code(), StatusCode::kFailedPrecondition);
}

TEST(AddRmsNormOp, RegistersTypedFactoryAndDefaultParams) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kAddRmsNorm,
            OpParams{AddRmsNormParams{}});
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    EXPECT_EQ(created.value()->Type(), OpType::kAddRmsNorm);
    EXPECT_STREQ(created.value()->Name(), "AddRmsNorm");

    const StatusOr<OpParams> defaults = OperatorRegistry::CreateDefaultParams(OpType::kAddRmsNorm);
    ASSERT_TRUE(defaults.ok()) << defaults.status().ToString();
    EXPECT_NE(std::get_if<AddRmsNormParams>(&*defaults), nullptr);
}

}// namespace
