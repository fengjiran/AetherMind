#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/graph/graph.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/runtime/runtime_builder.h"

#include <cstring>
#include <gtest/gtest.h>

#include <string>

namespace {

using namespace aethermind;

// Over-aligned scratch storage sized and aligned from a plan workspace layout.
struct AlignedScratch {
    explicit AlignedScratch(size_t bytes, size_t alignment)
        : bytes(bytes),
          alignment(alignment),
          data(static_cast<std::byte*>(
                  ::operator new(bytes, std::align_val_t{alignment}))) {}
    ~AlignedScratch() {
        ::operator delete(data, std::align_val_t{alignment});
    }
    AlignedScratch(const AlignedScratch&) = delete;
    AlignedScratch& operator=(const AlignedScratch&) = delete;

    size_t bytes = 0;
    size_t alignment = 0;
    std::byte* data = nullptr;
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
                      Buffer storage,
                      PackingRecipe recipe = {},
                      DataType logical_dtype = {},
                      std::vector<int64_t> logical_shape = {}) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)),
          recipe_(std::move(recipe)),
          logical_dtype_(logical_dtype),
          logical_shape_(std::move(logical_shape)) {}

    OpType op_type() const noexcept override {
        return op_type_;
    }

    const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    const Buffer& storage() const noexcept override {
        return storage_;
    }

    const PackingRecipe& recipe() const noexcept override {
        return recipe_;
    }

    DataType logical_dtype() const noexcept override {
        return logical_dtype_;
    }

    const std::vector<int64_t>& logical_shape() const noexcept override {
        return logical_shape_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
    PackingRecipe recipe_{};
    DataType logical_dtype_{};
    std::vector<int64_t> logical_shape_{};
};

Status PackedTestKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

// Recipe the PackedTestBackend declares it consumes; test stores must pack
// artifacts with the same recipe so exact-key resolution succeeds.
const PackingRecipe kTestPackedRecipe{.layout = "test_packed", .alignment = 64};

class PackedTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }

    StatusOr<ResolvedKernel> PrepareKernel(
            OpType op_type,
            const KernelSelector& selector,
            const OpParams&) const override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return Status::NotFound("Packed test backend only resolves packed selectors");
        }
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &PackedTestKernel,
                .attrs = {},
                .debug_name = "test::packed_kernel",
                .expected_packing_recipe = kTestPackedRecipe,
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

std::vector<WorkspaceBinding>* g_recorded_workspace_bindings = nullptr;

Status WorkspaceRecordingKernel(const KernelContext& context) noexcept {
    if (g_recorded_workspace_bindings != nullptr) {
        g_recorded_workspace_bindings->push_back(context.workspace_binding);
    }
    return Status::Ok();
}

class WorkspaceTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override { return capabilities_; }

    StatusOr<ResolvedKernel> PrepareKernel(OpType op_type,
                                           const KernelSelector&,
                                           const OpParams&) const override {
        switch (op_type) {
            case OpType::kEmbedding:
                return ResolvedKernel{
                        .op_type = op_type,
                        .fn = &WorkspaceRecordingKernel,
                        .attrs = {},
                        .debug_name = "test::embedding_workspace_kernel",
                        .workspace_requirement = {
                                .bytes = 24,
                                .alignment = 16,
                                .lifetime = WorkspaceLifetime::kPerOperator,
                        },
                };
            case OpType::kRmsNorm:
                return ResolvedKernel{
                        .op_type = op_type,
                        .fn = &WorkspaceRecordingKernel,
                        .attrs = {},
                        .debug_name = "test::rmsnorm_workspace_kernel",
                        .workspace_requirement = {
                                .bytes = 40,
                                .alignment = 64,
                                .lifetime = WorkspaceLifetime::kPerOperator,
                        },
                };
            default:
                return Status::NotFound("WorkspaceTestBackend does not prepare this op type");
        }
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities capabilities_{};
};

class WorkspaceTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<WorkspaceTestBackend>();
    }
};

class SoftmaxTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override { return caps_; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType op_type,
                                           const KernelSelector&,
                                           const OpParams&) const override {
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

class WrongOpTypeBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override { return capabilities_; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType,
                                           const KernelSelector&,
                                           const OpParams&) const override {
        return ResolvedKernel{.op_type = OpType::kSoftmax, .fn = &SoftmaxTestKernel};
    }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }

private:
    BackendCapabilities capabilities_{};
};

ExecutionPlanNodeSpec MakeRmsNormNodeSpec() {
    return ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
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

TEST(ExecutionPlanBuilder, PrepareKernelForNodeBuildsTypedMetadata) {
    CpuBackend backend;
    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 42.0F}};

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::PrepareKernelForNode(backend, node);

    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ(resolved->op_type, OpType::kRmsNorm);
    ASSERT_NE(resolved->fn, nullptr);
    ASSERT_EQ(resolved->attrs.size(), sizeof(float));
    float epsilon = 0.0F;
    std::memcpy(&epsilon, resolved->attrs.data(), sizeof(epsilon));
    EXPECT_FLOAT_EQ(epsilon, 42.0F);
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
    const ResolvedKernel& step_kernel = step.kernel;
    ASSERT_EQ(step_kernel.attrs.size(), sizeof(float));
    float stored_epsilon = 0.0F;
    std::memcpy(&stored_epsilon, step_kernel.attrs.data(), sizeof(stored_epsilon));
    EXPECT_EQ(step.kernel.op_type, OpType::kRmsNorm);
    EXPECT_EQ(step.selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(step.packed_weights, nullptr);
    EXPECT_EQ(step.workspace_requirement.bytes, 0U);
    EXPECT_EQ(step.workspace_requirement.alignment, 64U);
    EXPECT_EQ(step.workspace_requirement.offset, 0U);
    EXPECT_FLOAT_EQ(stored_epsilon, 11.0F);
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
    // Specs are owned once in plan.values; derive via inputs/outputs.
    ASSERT_EQ(step.inputs.size(), 2U);
    ASSERT_EQ(step.outputs.size(), 1U);
    EXPECT_EQ(plan->values()[step.inputs[0].index].spec, node.input_specs[0]);
    EXPECT_EQ(plan->values()[step.inputs[1].index].spec, node.input_specs[1]);
    {
        const TensorSpec& out_spec = plan->values()[step.outputs[0].index].spec;
        EXPECT_EQ(out_spec.dtype, DataType::Float32());
        ASSERT_EQ(out_spec.shape.rank(), 2U);
        // Spy: RmsNorm output shape echoes input[0] shape, so the inferred
        // ShapeSymbol IDs in the output spec match the input ShapeSymbol IDs.
        // This proves the untrusted path called InferOperator (which echoes
        // input symbols) rather than skipping inference.
        EXPECT_EQ(out_spec.shape[0], seq_len);
        EXPECT_EQ(out_spec.shape[1], input_hidden);
    }

    // Find the DimEqualConstraint without relying on constraint ordering.
    const DimEqualConstraint* equal = nullptr;
    for (const auto& check: step.runtime_checks) {
        if (std::holds_alternative<DimEqualConstraint>(check.condition)) {
            equal = &std::get<DimEqualConstraint>(check.condition);
            break;
        }
    }
    ASSERT_NE(equal, nullptr) << "expected a DimEqualConstraint";
    EXPECT_EQ(equal->lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal->lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(equal->lhs.dim_index, 1U);
    EXPECT_EQ(equal->rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal->rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(equal->rhs.dim_index, 0U);
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
    bool has_dim_equal = false;
    for (const auto& check: analyzed->runtime_checks) {
        if (std::holds_alternative<DimEqualConstraint>(check.condition)) {
            has_dim_equal = true;
            break;
        }
    }
    ASSERT_TRUE(has_dim_equal)
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

TEST(ExecutionPlanBuilder, BuildRejectsMissingTypedParams) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(MakeRmsNormNodeSpec());

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlanBuilder, BuildUsesPreparedKernelWorkspaceRequirementsForRawNodes) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<WorkspaceTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    for (size_t index = 0; index < 2; ++index) {
        ExecutionPlanNodeSpec node{
                .op_type = OpType::kRmsNorm,
                .selector = {
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .isa = IsaLevel::kScalar,
                        .phase = ExecPhase::kBoth,
                },
        };
        if (index == 0) {
            node.workspace_requirement = {
                    .bytes = 40,
                    .alignment = 64,
                    .lifetime = WorkspaceLifetime::kPerOperator,
                    .offset = 999,
            };
        }
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
    EXPECT_EQ(plan->steps()[0].workspace_requirement.bytes, 40U);
    EXPECT_EQ(plan->steps()[1].workspace_requirement.bytes, 40U);
}

TEST(ExecutionPlanBuilder, BuildRejectsRawWorkspaceRequirementThatDisagreesWithKernel) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<WorkspaceTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.workspace_requirement = {.bytes = 8, .alignment = 64};
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("must match"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildBindsPackedWeightsFromPackedWeightStore) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<PackedTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
    PackedWeightStore packed_weight_store;
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    // Untrusted nodes carry no WeightBinding, so the store key uses the empty
    // binding and must match the key derived by the builder for untrusted nodes.
    const WeightArtifactKey key{.binding = {},
                                .selector = selector,
                                .recipe = kTestPackedRecipe};

    ASSERT_TRUE(packed_weight_store
                        .Store(key, std::make_shared<TestPackedWeights>(
                                            OpType::kRmsNorm, selector,
                                            MakeTestBuffer(128),
                                            kTestPackedRecipe))
                        .ok());

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec node{
            .op_type = OpType::kRmsNorm,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPacked,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
    };
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;
    nodes.push_back(std::move(node));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, packed_weight_store, nodes);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_NE(plan->steps().front().packed_weights, nullptr);
    EXPECT_EQ(plan->steps().front().packed_weights->storage().data(),
              packed_weight_store.Find(key)->storage().data());
}

TEST(ExecutionPlanBuilder, BuildRejectsPackedWeightNodeWithoutPackedWeightStore) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec node{
            .op_type = OpType::kRmsNorm,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPacked,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
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

TEST(ExecutionPlanBuilder, PrepareKernelForNodeRejectsUnknownOpType) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::PrepareKernelForNode(backend,
                                                       ExecutionPlanNodeSpec{
                                                               .op_type = OpType::kUnknown,
                                                               .selector = {
                                                                       .device_type = DeviceType::kCPU,
                                                                       .act_dtype = DataType::Float32(),
                                                                       .weight_dtype = DataType::Float32(),
                                                               },
                                                       });

    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlanBuilder, PrepareKernelForNodeRejectsBackendOpTypeMismatch) {
    WrongOpTypeBackend backend;
    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    const auto kernel = ExecutionPlanBuilder::PrepareKernelForNode(backend, node);
    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kInternal);
}

TEST(ExecutionPlanBuilder, TrustedPathRejectsUnknownLoweredValuePayload) {
    LoweredGraph::Builder builder;
    builder.values.push_back({.spec = TensorSpec{.dtype = DataType::Float32(),
                                                 .shape = StaticShape({1})},
                              .payload = std::monostate{}});
    const auto lowered = std::move(builder).Build();
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    RuntimeContext runtime = RuntimeBuilder{}.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInternal);
}

TEST(ExecutionPlanBuilder, BuildFromLoweredGraphPropagatesPreparedKernelWorkspace) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId embedding_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();
    const GraphValueId norm_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kFinalNorm));
    const auto rms_norm = graph.AddNode(
            OpType::kRmsNorm, std::nullopt, {embedding->outputs[0], norm_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = 1.0e-5F});
    ASSERT_TRUE(rms_norm.ok()) << rms_norm.status().ToString();
    graph.MarkOutput(rms_norm->outputs[0]);

    const auto lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<WorkspaceTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->steps().size(), lowered->steps().size());
    for (size_t index = 0; index < plan->steps().size(); ++index) {
        const auto& step = plan->steps()[index];
        ASSERT_EQ(step.outputs.size(), lowered->steps()[index].spec.output_specs.size());
        for (size_t o = 0; o < step.outputs.size(); ++o) {
            EXPECT_EQ(plan->values()[step.outputs[o].index].spec,
                      lowered->steps()[index].spec.output_specs[o]);
        }
        EXPECT_EQ(step.runtime_checks,
                  lowered->steps()[index].spec.runtime_checks);
    }

    ASSERT_EQ(plan->steps()[0].workspace_requirement.bytes, 24U);
    EXPECT_EQ(plan->steps()[0].workspace_requirement.alignment, 16U);
    EXPECT_EQ(plan->steps()[0].workspace_requirement.offset, 0U);
    ASSERT_EQ(plan->steps()[1].workspace_requirement.bytes, 40U);
    EXPECT_EQ(plan->steps()[1].workspace_requirement.alignment, 64U);
    EXPECT_EQ(plan->steps()[1].workspace_requirement.offset, 64U);

    EXPECT_EQ(plan->total_workspace_bytes(), 104U);
    EXPECT_EQ(plan->workspace_alignment(), 64U);
}

TEST(ExecutionPlanBuilder, BuildFromLoweredGraphBindsDistinctPackedWeightsByBinding) {
    // Two RmsNorm steps consume two weights with identical dtypes and selectors
    // but distinct bindings (different layer indices). With packed weights
    // enabled, the lowered selectors must be kPacked and each step must resolve
    // to the artifact for its own binding — never a shared first-instance.
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId embedding_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt,
                                         TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();
    const GraphValueId norm0_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            MakeTransformerWeightBinding(0, TransformerWeightRole::kInputNorm));
    const auto norm0 = graph.AddNode(
            OpType::kRmsNorm, 0U, {embedding->outputs[0], norm0_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = 1.0e-5F});
    ASSERT_TRUE(norm0.ok()) << norm0.status().ToString();
    const GraphValueId norm1_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            MakeTransformerWeightBinding(1, TransformerWeightRole::kInputNorm));
    const auto norm1 = graph.AddNode(
            OpType::kRmsNorm, 1U, {norm0->outputs[0], norm1_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = 1.0e-5F});
    ASSERT_TRUE(norm1.ok()) << norm1.status().ToString();
    graph.MarkOutput(norm1->outputs[0]);

    GraphLoweringConfig config;
    config.enable_packed_weights = true;
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps().size(), 3U);
    for (const auto& step: lowered->steps()) {
        EXPECT_EQ(step.spec.selector.weight_format, WeightFormat::kPacked);
    }
    // Keys must carry the artifact identity and the exact (artifact-local)
    // value id of each weight so cross-model stores can never collide.
    const uint64_t source = lowered->artifact_id();

    const WeightArtifactKey embedding_key{
            .source_id = source,
            .value_index = embedding_weight.index,
            .binding = MakeTransformerWeightBinding(
                    std::nullopt, TransformerWeightRole::kTokenEmbedding),
            .selector = lowered->steps()[0].spec.selector,
            .recipe = kTestPackedRecipe};
    const WeightArtifactKey norm0_key{
            .source_id = source,
            .value_index = norm0_weight.index,
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kInputNorm),
            .selector = lowered->steps()[1].spec.selector,
            .recipe = kTestPackedRecipe};
    const WeightArtifactKey norm1_key{
            .source_id = source,
            .value_index = norm1_weight.index,
            .binding = MakeTransformerWeightBinding(1, TransformerWeightRole::kInputNorm),
            .selector = lowered->steps()[2].spec.selector,
            .recipe = kTestPackedRecipe};
    // The two RmsNorm steps share one selector (same dtypes/ISA/phase).
    EXPECT_EQ(lowered->steps()[1].spec.selector, lowered->steps()[2].spec.selector);

    PackedWeightStore packed_weight_store;
    ASSERT_TRUE(packed_weight_store.SetSourceId(lowered->artifact_id()).ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(embedding_key,
                               std::make_shared<TestPackedWeights>(
                                       OpType::kEmbedding,
                                       lowered->steps()[0].spec.selector,
                                       MakeTestBuffer(8 * 128),
                                       kTestPackedRecipe,
                                       DataType::Float32(),
                                       std::vector<int64_t>{32, 8}))
                        .ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(norm0_key,
                               std::make_shared<TestPackedWeights>(
                                       OpType::kRmsNorm,
                                       lowered->steps()[1].spec.selector,
                                       MakeTestBuffer(64),
                                       kTestPackedRecipe,
                                       DataType::Float32(),
                                       std::vector<int64_t>{8}))
                        .ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(norm1_key,
                               std::make_shared<TestPackedWeights>(
                                       OpType::kRmsNorm,
                                       lowered->steps()[2].spec.selector,
                                       MakeTestBuffer(64),
                                       kTestPackedRecipe,
                                       DataType::Float32(),
                                       std::vector<int64_t>{8}))
                        .ok());

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<PackedTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
    const auto plan =
            ExecutionPlanBuilder::Build(runtime, packed_weight_store, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 3U);

    // Each RmsNorm step resolves to the artifact of its own binding; the two
    // steps do not share a pointer even though their selectors are identical.
    EXPECT_EQ(plan->steps()[0].packed_weights,
              packed_weight_store.Find(embedding_key));
    ASSERT_NE(plan->steps()[1].packed_weights, nullptr);
    ASSERT_NE(plan->steps()[2].packed_weights, nullptr);
    EXPECT_EQ(plan->steps()[1].packed_weights, packed_weight_store.Find(norm0_key));
    EXPECT_EQ(plan->steps()[2].packed_weights, packed_weight_store.Find(norm1_key));
    EXPECT_NE(plan->steps()[1].packed_weights, plan->steps()[2].packed_weights);
}

TEST(ExecutionPlanBuilder, TrustedPathRejectsStoreFromAnotherArtifact) {
    // A store bound to a different model artifact must be refused even when
    // every packed key would otherwise match: identity includes the source.
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({1})});
    const GraphValueId embedding_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32, 8})},
            MakeTransformerWeightBinding(std::nullopt,
                                         TransformerWeightRole::kTokenEmbedding));
    const auto embedding = graph.AddNode(
            OpType::kEmbedding, std::nullopt, {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
    ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();
    const GraphValueId norm_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            MakeTransformerWeightBinding(0, TransformerWeightRole::kInputNorm));
    const auto rms_norm = graph.AddNode(
            OpType::kRmsNorm, 0U, {embedding->outputs[0], norm_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = 1.0e-5F});
    ASSERT_TRUE(rms_norm.ok()) << rms_norm.status().ToString();
    graph.MarkOutput(rms_norm->outputs[0]);

    GraphLoweringConfig config;
    config.enable_packed_weights = true;
    const auto lowered = LowerModelGraph(graph, config);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps().size(), 2U);
    for (const auto& step: lowered->steps()) {
        EXPECT_EQ(step.spec.selector.weight_format, WeightFormat::kPacked);
    }

    PackedWeightStore foreign_store;
    ASSERT_TRUE(foreign_store.SetSourceId(lowered->artifact_id() + 1000).ok());

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<PackedTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, foreign_store, *lowered);

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("different model artifact"), std::string::npos);
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
    const ModelGraph graph;

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    EXPECT_TRUE(lowered->steps().empty());

    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(
            runtime, *lowered);

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_TRUE(plan->state_alias_plan().empty());
}

TEST(ExecutionPlanBuilder, TrustedPathCopiesValueDataflowAfterLoweredGraphLifetimeEnds) {
    RuntimeContext runtime = RuntimeBuilder{}.Build();
    StatusOr<ExecutionPlan> plan = Status::Internal("not built");
    {
        ModelGraph graph;
        const GraphValueId tokens = graph.AddInput(
                TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({2})});
        const GraphValueId weight = graph.AddWeight(
                TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3})},
                MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding));
        const auto embedding = graph.AddNode(
                OpType::kEmbedding, std::nullopt, {tokens, weight},
                {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{});
        ASSERT_TRUE(embedding.ok()) << embedding.status().ToString();
        const auto add = graph.AddNode(
                OpType::kAdd, std::nullopt, {embedding->outputs[0], embedding->outputs[0]},
                {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
        ASSERT_TRUE(add.ok()) << add.status().ToString();
        graph.MarkOutput(add->outputs[0]);

        const auto lowered = LowerModelGraph(graph);
        ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
        plan = ExecutionPlanBuilder::Build(runtime, *lowered);
        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        ASSERT_EQ(plan->values().size(), lowered->values().size());
        ASSERT_EQ(plan->steps().size(), 2U);
        EXPECT_EQ(plan->steps()[1].inputs[0].index, embedding->outputs[0].index);
        EXPECT_EQ(plan->steps()[1].inputs[1].index, embedding->outputs[0].index);
        EXPECT_EQ(plan->model_inputs()[0].index, tokens.index);
        EXPECT_EQ(plan->model_outputs()[0].index, add->outputs[0].index);
    }

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_EQ(plan->steps()[1].inputs[0], plan->steps()[0].outputs[0]);
    EXPECT_EQ(plan->steps()[1].inputs[0], plan->steps()[1].inputs[1]);
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
    EXPECT_NE(plan.status().message().find("typed op_params"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesPreservesInferredMetadata) {
    // The untrusted path validates caller metadata via InferOperator before
    // resolving a schema-only OpType directly through the backend.
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
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
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
    // InferSoftmax echoed the input spec (owned in plan.values).
    ASSERT_EQ(step.outputs.size(), 1U);
    EXPECT_EQ(plan->values()[step.outputs[0].index].spec, analyzed->outputs[0]);
    EXPECT_EQ(step.runtime_checks, analyzed->runtime_checks);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsSelectorActDTypeMismatch) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    // Specs are semantically valid, but the selector claims Float16
    // activations; the backend would resolve a Float16 kernel for Float32 data.
    node.selector.act_dtype = DataType::Float(16);
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("act_dtype"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsSelectorWeightDTypeMismatch) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.selector.weight_dtype = DataType::Float(16);
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("weight_dtype"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsSelectorWeightDTypeWithoutWeightPort) {
    // Softmax has no weight port, so its selector weight_dtype must fall back
    // to the activation dtype; a caller-declared different weight dtype is
    // inconsistent with the operator.
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
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float(16),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
    };
    node.op_params = OpParams{SoftmaxParams{.axis = -1}};
    node.input_specs = inputs;
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("weight_dtype"), std::string::npos);
}

TEST(ExecutionPlanBuilder, BuildFromRawNodesRejectsUndefinedSelectorActDType) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape act_shape = StaticShape({4, 8});
    const SymbolicShape weight_shape = StaticShape({8});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node = MakeRmsNormNodeSpec();
    node.selector.act_dtype = {};
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(plan.status().message().find("act_dtype"), std::string::npos);
}

}// namespace
