#include "../../include/aethermind/execution/execution_plan_builder.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/model/graph/graph.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/rmsnorm_op.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

struct TestAttrs {
    int epsilon;
};

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

class TestPackedWeights final : public PackedWeights {
public:
    TestPackedWeights(OpType op_type,
                      KernelSelector selector,
                      Buffer storage) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)) {}

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
};

Status PackedTestKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

class PackedTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }

    KernelFunc ResolveKernel(OpType,
                             const KernelSelector& selector) const noexcept override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return nullptr;
        }
        return &PackedTestKernel;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector& selector) const noexcept override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return Status::NotFound("Packed test backend only resolves packed selectors");
        }
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &PackedTestKernel,
                .attrs = {},
                .debug_name = "test::packed_kernel",
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities capabilities_{};
};

class PackedTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<PackedTestBackend>();
    }
};

Status SoftmaxTestKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

class SoftmaxTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override { return caps_; }
    KernelFunc ResolveKernel(OpType op_type, const KernelSelector&) const noexcept override {
        return op_type == OpType::kSoftmax ? &SoftmaxTestKernel : nullptr;
    }
    StatusOr<ResolvedKernel> ResolveKernelInfo(OpType op_type,
                                               const KernelSelector&) const noexcept override {
        if (op_type != OpType::kSoftmax) {
            return Status::NotFound("SoftmaxTestBackend only resolves kSoftmax");
        }
        return ResolvedKernel{.op_type = op_type, .fn = &SoftmaxTestKernel, .attrs = {}, .debug_name = "test::softmax_kernel"};
    }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }

private:
    BackendCapabilities caps_{};
};

class SoftmaxTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<SoftmaxTestBackend>();
    }
};

ExecutionPlanNodeSpec MakeRmsNormNodeSpec(std::span<const std::byte> attrs = {}) {
    return ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .attrs = std::vector<std::byte>(attrs.begin(), attrs.end()),
    };
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

// Helper: derive RmsNorm output_specs and runtime_checks via the semantic
// authority InferOperator, so tests can fill caller-provided metadata
// fields without duplicating inference logic.
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

TEST(ExecutionPlanBuilder, ResolveKernelForNodeUsesOpTypeDirectly) {
    CpuBackend backend;
    TestAttrs attrs{.epsilon = 42};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::ResolveKernelForNode(backend,
                                                       MakeRmsNormNodeSpec(attrs_bytes));

    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ(resolved->op_type, OpType::kRmsNorm);
    ASSERT_NE(resolved->fn, nullptr);
    EXPECT_NE(resolved->attrs.data(), attrs_bytes.data());
    EXPECT_EQ(resolved->attrs.size(), sizeof(attrs));
    EXPECT_EQ(resolved->attrs, std::vector<std::byte>(attrs_bytes.begin(), attrs_bytes.end()));
    EXPECT_STREQ(resolved->debug_name, "cpu::rmsnorm_f32_scalar");
}

TEST(ExecutionPlanBuilder, BuildFreezesResolvedKernelIntoExecutionPlan) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(11.0F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 11.0F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;
    nodes.push_back(node);

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    const auto& step = plan->steps().front();
    ASSERT_NE(step.op, nullptr);
    const ResolvedKernel step_kernel = step.op->GetResolvedKernel();
    ASSERT_EQ(step_kernel.attrs.size(), sizeof(float));
    const auto* stored_epsilon = reinterpret_cast<const float*>(step_kernel.attrs.data());
    ASSERT_NE(stored_epsilon, nullptr);
    EXPECT_EQ(step.op->Type(), OpType::kRmsNorm);
    EXPECT_EQ(step.selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(step.packed_weights, nullptr);
    EXPECT_EQ(step.workspace_requirement.bytes, 0U);
    EXPECT_EQ(step.workspace_requirement.alignment, 64U);
    EXPECT_EQ(step.workspace_requirement.offset, 0U);
    EXPECT_FLOAT_EQ(*stored_epsilon, 11.0F);
    EXPECT_EQ(step.debug_name, nullptr);
    EXPECT_STREQ(step_kernel.debug_name, "cpu::rmsnorm_f32_scalar");
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesValidatesInferredMetadata) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const SymbolicShape act_shape(std::vector<ShapeSymbol>{seq_len, input_hidden});
    const SymbolicShape weight_shape(std::vector<ShapeSymbol>{weight_hidden});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    const ExecutionStep& step = plan->steps().front();
    ASSERT_EQ(step.input_specs.size(), 2U);
    EXPECT_EQ(step.input_specs[0], node.input_specs[0]);
    EXPECT_EQ(step.input_specs[1], node.input_specs[1]);
    ASSERT_EQ(step.output_specs.size(), 1U);
    EXPECT_EQ(step.output_specs[0].dtype, DataType::Float32());
    ASSERT_EQ(step.output_specs[0].shape.rank(), 2U);
    // Spy: RmsNorm output shape echoes input[0] shape, so the inferred
    // ShapeSymbol IDs in step.output_specs match the input ShapeSymbol IDs.
    // This proves the untrusted path called InferOperator (which echoes
    // input symbols) rather than skipping inference.
    EXPECT_EQ(step.output_specs[0].shape[0], seq_len);
    EXPECT_EQ(step.output_specs[0].shape[1], input_hidden);

    ASSERT_EQ(step.runtime_checks.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(step.runtime_checks[0].condition));
    const auto& equal = std::get<DimEqualConstraint>(step.runtime_checks[0].condition);
    EXPECT_EQ(equal.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(equal.lhs.dim_index, 1U);
    EXPECT_EQ(equal.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(equal.rhs.dim_index, 0U);
}

TEST(ExecutionPlanBuilder, BuildRejectsMismatchedOutputSpecs) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    // Deliberately set output_specs to a wrong shape ([4,16] instead of [4,8]).
    node.output_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 16})},
    };
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlanBuilder, BuildRejectsInvalidInputSpecsBeforePrepare) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    // Semantically invalid: activation last dim (8) != weight length (16).
    // InferRmsNorm will fail, and Build rejects with that error's code.
    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16})},
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsWrongInputDtype) {
    // Untrusted path must reject wrong dtype via InferOperator re-validation.
    // RmsNorm only accepts floating-point dtypes; supplying Int32 for input[0]
    // must fail at the semantic authority layer (not at kernel resolution or Prepare).
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Int(32), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    // Caller-provided output_specs/runtime_checks would be mismatched anyway;
    // the dtype check fires first inside InferOperator.
    node.output_specs = {
            TensorSpec{.dtype = DataType::Int(32), .shape = act_shape},
    };
    node.runtime_checks = {};

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    // The rejection must come from InferRmsNorm's dtype check, not from
    // an output-spec mismatch or kernel resolution failure.
    EXPECT_NE(plan.status().message().find("RmsNorm"), std::string::npos);
    EXPECT_NE(plan.status().message().find("dtype"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsMismatchedRuntimeChecks) {
    // Untrusted path must reject caller-provided runtime_checks that differ
    // from InferOperator-derived constraints. Using symbolic shapes for
    // activation hidden dim and weight length forces InferRmsNorm to emit
    // a DimEqualConstraint; supplying an empty runtime_checks vector must
    // fail the strict-equality check in ValidateCallerMetadata.
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const SymbolicShape act_shape(std::vector<ShapeSymbol>{seq_len, input_hidden});
    const SymbolicShape weight_shape(std::vector<ShapeSymbol>{weight_hidden});

    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    ASSERT_EQ(analyzed->runtime_checks.size(), 1U)
            << "expected InferRmsNorm to emit a DimEqualConstraint for "
               "symbolic-hidden != symbolic-weight-length";

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    // Deliberately empty: caller omits the constraint InferOperator derived.
    node.runtime_checks = {};

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("runtime_checks does not match"),
              std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildRejectsRawAttrsForRegisteredOperator) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();
    TestAttrs attrs{.epsilon = 11};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(MakeRmsNormNodeSpec(attrs_bytes));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlanBuilder, BuildPlansWorkspaceOffsetsAcrossNodes) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    for (const auto& req: {WorkspaceRequirement{.bytes = 32, .alignment = 16, .offset = 999},
                           WorkspaceRequirement{.bytes = 8, .alignment = 64, .offset = 123}}) {
        ExecutionPlanNodeSpec node{
                .op_type = OpType::kRmsNorm,
                .device_type = DeviceType::kCPU,
                .act_dtype = DataType::Float32(),
                .weight_dtype = DataType::Float32(),
                .weight_format = WeightFormat::kPlain,
                .isa = IsaLevel::kScalar,
                .phase = ExecPhase::kBoth,
                .workspace_requirement = req,
        };
        node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
        node.input_specs = {
                TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
                TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
        };
        node.output_specs = analyzed->outputs;
        node.runtime_checks = analyzed->runtime_checks;
        nodes.push_back(std::move(node));
    }

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 2U);
    EXPECT_EQ(plan->steps()[0].workspace_requirement.offset, 0U);
    EXPECT_EQ(plan->steps()[1].workspace_requirement.offset, 64U);
}

TEST(ExecutionPlanBuilder, BuildBindsPackedWeightsFromModelInstanceSidecar) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<PackedTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
    ModelInstance model_instance;
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };

    ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<TestPackedWeights>(
                                                          OpType::kRmsNorm,
                                                          selector,
                                                          MakeTestBuffer(128)))
                        .ok());

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec node{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;
    nodes.push_back(std::move(node));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, model_instance, nodes);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_NE(plan->steps().front().packed_weights, nullptr);
    EXPECT_EQ(plan->steps().front().packed_weights,
              model_instance.FindPackedWeights(OpType::kRmsNorm, selector)->storage().data());
}

TEST(ExecutionPlanBuilder, BuildRejectsPackedWeightNodeWithoutModelInstanceSidecar) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec node{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;
    nodes.push_back(std::move(node));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kNotFound);
}

TEST(ExecutionPlanBuilder, ResolveKernelForNodeRejectsUnknownOpType) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::ResolveKernelForNode(backend,
                                                       ExecutionPlanNodeSpec{
                                                               .op_type = OpType::kUnknown,
                                                               .device_type = DeviceType::kCPU,
                                                               .act_dtype = DataType::Float32(),
                                                               .weight_dtype = DataType::Float32(),
                                                       });

    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlanBuilder, BuildFromLoweredGraphStoresRuntimeStateAliasPlan) {
    // Construct a LoweredGraph manually with a valid RmsNorm step and one
    // lowering-time alias record, avoiding any dependency on KVCacheUpdate kernels.
    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    LoweredGraph lowered;
    ExecutionPlanNodeSpec step = MakeRmsNormNodeSpec();
    step.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    step.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    step.output_specs = analyzed->outputs;
    step.runtime_checks = analyzed->runtime_checks;
    lowered.steps.push_back(std::move(step));
    lowered.step_bindings.push_back(LoweredStepBinding{
            .node = GraphNodeId{.index = 0},
            .input_values = {GraphValueId{.index = 0}, GraphValueId{.index = 1}},
            .output_values = {GraphValueId{.index = 2}},
    });
    lowered.state_aliases.push_back(LoweredStateAlias{
            .input = GraphValueId{.index = 0},
            .output = GraphValueId{.index = 2},
    });

    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(
            runtime, lowered);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_EQ(plan->state_alias_plan().size(), 1U);
    EXPECT_FALSE(plan->state_alias_plan().empty());
}

TEST(ExecutionPlanBuilder, BuildFromLoweredGraphPropagatesTrustedMetadata) {
    // Spy-based proof that the trusted path does NOT re-invoke InferOperator.
    //
    // Strategy: use ShapeSymbol::Create() to mint unique symbolic dims for
    // the input. InferRmsNorm echoes input[0] in its output_specs. We then
    // overwrite the lowered output_specs with a shape containing a fresh
    // "spy" symbol that is NOT present in input_specs. If the trusted path
    // carries lowered metadata verbatim, the spy symbol survives into the
    // plan. If the trusted path re-invoked InferOperator, the output would
    // echo input[0] (no spy symbol), and the assertion would fail.
    //
    // This satisfies the plan's requirement: "spy/fixture 证明 trusted path
    // 不调用 semantic analyzer", with no production-global mutable hook.
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const ShapeSymbol spy_symbol = ShapeSymbol::Create();
    const SymbolicShape act_shape(std::vector<ShapeSymbol>{seq_len, input_hidden});
    const SymbolicShape weight_shape(std::vector<ShapeSymbol>{weight_hidden});
    const SymbolicShape spy_shape(std::vector<ShapeSymbol>{seq_len, spy_symbol});

    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    // Sanity: InferRmsNorm echoes input[0] (act_shape), not spy_shape.
    ASSERT_EQ(analyzed->outputs[0].shape, act_shape);

    LoweredGraph lowered;
    ExecutionPlanNodeSpec step = MakeRmsNormNodeSpec();
    step.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    step.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    // Inject the spy shape into lowered output_specs. The trusted path must
    // carry this verbatim; a re-analysis would have produced act_shape.
    TensorSpec spy_output = analyzed->outputs[0];
    spy_output.shape = spy_shape;
    step.output_specs = {spy_output};
    step.runtime_checks = analyzed->runtime_checks;
    lowered.steps.push_back(std::move(step));
    lowered.step_bindings.push_back(LoweredStepBinding{
            .node = GraphNodeId{.index = 0},
            .input_values = {GraphValueId{.index = 0}, GraphValueId{.index = 1}},
            .output_values = {GraphValueId{.index = 2}},
    });

    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, lowered);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->steps().size(), 1U);
    ASSERT_EQ(plan->steps()[0].output_specs.size(), 1U);
    ASSERT_EQ(plan->steps()[0].output_specs[0].shape.rank(), 2U);

    // Spy assertion: the plan's output shape[1] is the spy symbol, NOT
    // input_hidden (which InferOperator would have echoed). This proves
    // the trusted path carried lowered metadata verbatim and did not
    // re-invoke the semantic analyzer.
    EXPECT_EQ(plan->steps()[0].output_specs[0].shape[1].value(), spy_symbol.value())
            << "Trusted path appears to have re-invoked InferOperator: "
               "output symbol matches input_hidden instead of the spy symbol";
    EXPECT_NE(plan->steps()[0].output_specs[0].shape[1].value(), input_hidden.value());

    // Trusted path: runtime_checks carried forward verbatim.
    EXPECT_EQ(plan->steps()[0].runtime_checks, lowered.steps[0].runtime_checks);
}

TEST(ExecutionPlanBuilder, BuildFromNodesAloneHasEmptyStateAliasPlan) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(11.0F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 11.0F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;
    nodes.push_back(node);

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(
            runtime, nodes);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_TRUE(plan->state_alias_plan().empty());
    EXPECT_EQ(plan->state_alias_plan().size(), 0U);
}

TEST(ExecutionPlanBuilder, BuildFromEmptyLoweredGraphHasEmptyStateAliasPlan) {
    const ModelGraph graph(
            HfModelConfig{.model_type = "llama",
                          .hidden_size = 8,
                          .num_hidden_layers = 1,
                          .num_attention_heads = 4,
                          .num_key_value_heads = 2,
                          .vocab_size = 32,
                          .max_position_embeddings = 128,
                          .head_dim = 2,
                          .rms_norm_eps = 1.0e-5,
                          .hidden_act = "silu"});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    EXPECT_TRUE(lowered->steps.empty());

    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(
            runtime, *lowered);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_TRUE(plan->state_alias_plan().empty());
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsMissingTypedParams) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    // Intentionally leave op_params as monostate.
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("monostate"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildFromLoweredGraphResolvesRawFallbackForUnregisteredOpType) {
    // kSoftmax has a schema but no registered Operator factory. The trusted
    // LoweredGraph path contract is "create Operator if registered, otherwise
    // resolve raw fallback"; an unregistered OpType must NOT be rejected with
    // FailedPrecondition. The lowered metadata (output_specs, runtime_checks)
    // is carried forward verbatim without re-invoking InferOperator.
    // Use SoftmaxTestBackend so the Softmax kernel can be resolved.
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<SoftmaxTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
    };
    const auto analyzed = InferOperator(OpType::kSoftmax,
                                        OpParams{SoftmaxParams{.axis = -1}},
                                        inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    LoweredGraph lowered;
    ExecutionPlanNodeSpec step{
            .op_type = OpType::kSoftmax,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    step.op_params = OpParams{SoftmaxParams{.axis = -1}};
    step.input_specs = inputs;
    step.output_specs = analyzed->outputs;
    step.runtime_checks = analyzed->runtime_checks;
    lowered.steps.push_back(std::move(step));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, lowered);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->steps().size(), 1U);
    // Trusted raw fallback: lowered metadata carried forward verbatim.
    EXPECT_EQ(plan->steps()[0].output_specs, lowered.steps[0].output_specs);
    EXPECT_EQ(plan->steps()[0].runtime_checks, lowered.steps[0].runtime_checks);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesPreservesFunctionOperatorMetadata) {
    // kSoftmax has a schema but no registered Operator factory, so Build
    // falls back to the FunctionOperator raw-kernel path. The untrusted path
    // still validates caller metadata via InferOperator (no no-op bypass).
    // Asserting step.output_specs is non-empty proves InferOperator was
    // called rather than FunctionOperator::InferOutputShapes (which returns
    // an empty InferenceResult).
    // Use SoftmaxTestBackend so the Softmax kernel can be resolved (CpuBackend
    // does not register a Softmax kernel).
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<SoftmaxTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
    };
    const auto analyzed = InferOperator(OpType::kSoftmax,
                                        OpParams{SoftmaxParams{.axis = -1}},
                                        inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kSoftmax,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    node.op_params = OpParams{SoftmaxParams{.axis = -1}};
    node.input_specs = inputs;
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    const auto& step = plan->steps().front();
    // output_specs is non-empty: InferSoftmax echoed the input spec.
    // FunctionOperator::InferOutputShapes would have returned empty outputs.
    ASSERT_EQ(step.output_specs.size(), 1U);
    EXPECT_EQ(step.output_specs[0], analyzed->outputs[0]);
    EXPECT_EQ(step.runtime_checks, analyzed->runtime_checks);
}

}// namespace
}// namespace aethermind
