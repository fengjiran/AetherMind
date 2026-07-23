#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

StatusOr<InferenceResult> InferRmsNorm(float eps,
                                       const SymbolicShape& act_shape,
                                       const SymbolicShape& weight_shape) {
    std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    return InferOperator(OpType::kRmsNorm,
                         OpParams{RmsNormParams{.eps = eps}},
                         inputs);
}

template<typename T>
concept HasPublicAddStep = requires(T& t, const ExecutionStep& s) {
    t.AddStep(s);
};

static_assert(!HasPublicAddStep<ExecutionPlan>,
              "External code must not be able to call AddStep. "
              "ExecutionPlan is immutable after builder construction.");

void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeTestBuffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), alignment)};
}

Status ImmutableKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

class ImmutablePackedWeights final : public PackedWeights {
public:
    ImmutablePackedWeights(OpType op_type,
                           KernelSelector selector,
                           Buffer storage,
                           bool* destroyed_flag) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)),
          destroyed_flag_(destroyed_flag) {}

    ~ImmutablePackedWeights() override {
        if (destroyed_flag_ != nullptr) {
            *destroyed_flag_ = true;
        }
    }

    OpType op_type() const noexcept override {
        return op_type_;
    }

    const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    const Buffer& storage() const noexcept override {
        return storage_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
    bool* destroyed_flag_ = nullptr;
};

class ImmutableTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }

    KernelFunc ResolveKernel(OpType,
                             const KernelSelector&) const noexcept override {
        return &ImmutableKernel;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector&) const noexcept override {
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &ImmutableKernel,
                .attrs = {},
                .debug_name = "test::immutable_kernel",
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities capabilities_{};
};

class ImmutableTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<ImmutableTestBackend>();
    }
};

TEST(ExecutionPlanImmutability, StepsReturnsConstViewAfterConstruction) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({1, 4});
    const SymbolicShape weight_shape = StaticShape({4});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .input_specs = {TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape}},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const std::vector<ExecutionStep>& steps = plan->steps();
    EXPECT_EQ(steps.size(), 1U);
    EXPECT_EQ(steps.front().op->Type(), OpType::kRmsNorm);
    ASSERT_NE(steps.front().op, nullptr);
    EXPECT_NE(steps.front().op->GetResolvedKernel().fn, nullptr);
}

TEST(ExecutionPlanImmutability, WorkspaceOffsetsAreFrozenAfterBuilderPlanning) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({1, 4});
    const SymbolicShape weight_shape = StaticShape({4});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {.bytes = 64, .alignment = 32, .offset = 999},
            .input_specs = {TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape}},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}},
    });
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {.bytes = 128, .alignment = 64, .offset = 123},
            .input_specs = {TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape}},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 2U);

    const ExecutionStep& step0 = plan->steps()[0];
    const ExecutionStep& step1 = plan->steps()[1];

    EXPECT_EQ(step0.workspace_requirement.offset, 0U);
    EXPECT_EQ(step1.workspace_requirement.offset, 64U);

    EXPECT_EQ(step0.workspace_requirement.bytes, 64U);
    EXPECT_EQ(step1.workspace_requirement.bytes, 128U);
}

TEST(ExecutionPlanImmutability, PackedWeightsLifetimeManagedByModelInstance) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    bool packed_destroyed = false;
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };

    ModelInstance model_instance;
    ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<ImmutablePackedWeights>(
                                                          OpType::kRmsNorm,
                                                          selector,
                                                          MakeTestBuffer(256),
                                                          &packed_destroyed))
                        .ok());

    const SymbolicShape act_shape = StaticShape({1, 4});
    const SymbolicShape weight_shape = StaticShape({4});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .input_specs = {TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape}},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, model_instance, nodes);

    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const ExecutionStep& step = plan->steps().front();
    ASSERT_NE(step.packed_weights, nullptr);

    const void* packed_ptr = step.packed_weights;
    EXPECT_FALSE(packed_destroyed);

    {
        ExecutionPlan local_plan = plan.value();
        const ExecutionStep& local_step = local_plan.steps().front();
        EXPECT_EQ(local_step.packed_weights, packed_ptr);
    }

    EXPECT_FALSE(packed_destroyed);
}

TEST(ExecutionPlanImmutability, ExecutorConsumesFrozenPlanWithoutModification) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({1, 4});
    const SymbolicShape weight_shape = StaticShape({4});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .input_specs = {TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape}},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());

    const size_t original_size = plan->size();
    const OpType original_op_type = plan->steps().front().op->Type();
    ASSERT_NE(plan->steps().front().op, nullptr);
    const KernelFunc original_fn = plan->steps().front().op->GetResolvedKernel().fn;

    float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {};
    constexpr int64_t io_shape[2] = {1, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t w_shape[1] = {4};
    constexpr int64_t w_strides[1] = {1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView{input, DataType::Float32(), io_shape, io_strides},
                                                     TensorView{weight, DataType::Float32(), w_shape, w_strides},
                                             },
                                             .outputs = {
                                                     MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
                                             },
                                     });
    const Status status = Executor::Execute(*plan, bindings);

    ASSERT_TRUE(status.ok());

    EXPECT_EQ(plan->size(), original_size);
    EXPECT_EQ(plan->steps().front().op->Type(), original_op_type);
    EXPECT_EQ(plan->steps().front().op->GetResolvedKernel().fn, original_fn);
}

TEST(ExecutionPlanImmutability, PlanDoesNotContainRuntimeBindings) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({1, 4});
    const SymbolicShape weight_shape = StaticShape({4});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .input_specs = {TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape}},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const ExecutionStep& step = plan->steps().front();

    EXPECT_EQ(step.packed_weights, nullptr);

    EXPECT_GT(step.workspace_requirement.alignment, 0U);
    EXPECT_TRUE((step.workspace_requirement.alignment &
                 (step.workspace_requirement.alignment - 1)) == 0);
}

}// namespace
}// namespace aethermind
