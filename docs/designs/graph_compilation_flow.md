# 当前图编译流程实现梳理

本文按当前代码事实梳理 AetherMind 中图编译相关实现。结论先说：当前仓库已经具备 `ModelGraph -> LoweredGraph -> ExecutionPlanNodeSpec` 的语义图 lowering bridge，以及 `std::vector<ExecutionPlanNodeSpec> -> ExecutionPlan` 的计划构建流程；生产加载路径仍主要围绕手工/测试构造的 node spec，尚未形成完整的模型加载到执行计划自动编译入口。

## 1. 当前实际入口

```cpp
ExecutionPlanBuilder::Build(RuntimeContext& runtime,
                            const std::vector<ExecutionPlanNodeSpec>& nodes)

ExecutionPlanBuilder::Build(RuntimeContext& runtime,
                            const ModelInstance& model_instance,
                            const std::vector<ExecutionPlanNodeSpec>& nodes)
```

输入核心是：

```cpp
struct ExecutionPlanNodeSpec {
    OpType op_type;
    DeviceType device_type;
    DataType activation_dtype;
    DataType weight_dtype;
    WeightFormat weight_format;
    IsaLevel isa;
    ExecPhase phase;
    WorkspaceRequirement workspace_requirement;
    std::vector<TensorSpec> input_specs;
    std::span<const std::byte> attrs;
    OpParams op_params;
};
```

也就是说，`ExecutionPlanBuilder` 的直接输入是已经线性化好的 node spec 列表；`ModelGraph` 通过 `LowerModelGraph()` 先转换为 `LoweredGraph::steps`，再进入该构建流程。

## 2. 当前完整流程图

```text
调用方 / 测试 / ModelGraph Lowering
        │
        ▼
std::vector<ExecutionPlanNodeSpec>
        │
        ▼
ExecutionPlanBuilder::Build(...)
        │
        ▼
BuildExecutionPlan(runtime, model_instance?, nodes)
        │
        ├─ 收集 workspace_requirement
        │
        ├─ PlanWorkspaceRequirements(...)
        │
        ▼
for each node:
        │
        ├─ runtime.GetBackend(node.device_type)
        │
        ├─ CreateAndPrepareOperator(backend, node)
        │      ├─ OperatorRegistry::Create(...)
        │      ├─ op->ValidateParams()
        │      ├─ if input_specs non-empty:
        │      │      ├─ op->CheckInputSpecs(input_specs)
        │      │      └─ op->InferOutputShapes(input_specs)
        │      └─ op->Prepare(OperatorContext{backend, selector})
        │
        ├─ 如果没有注册 Operator:
        │      ├─ ResolveKernelForNode(backend, node)
        │      └─ FunctionOperator(...)
        │
        ├─ ResolvePackedWeightsForNode(model_instance?, node)
        │
        └─ plan.AddStep(ExecutionStep{...})
        │
        ▼
ExecutionPlan
```

## 3. 关键函数调用序列

### A. 顶层 Build

```cpp
ExecutionPlanBuilder::Build(runtime, nodes)
```

调用：

```cpp
BuildExecutionPlan(runtime, nullptr, nodes)
```

另一个重载：

```cpp
ExecutionPlanBuilder::Build(runtime, model_instance, nodes)
```

调用：

```cpp
BuildExecutionPlan(runtime, &model_instance, nodes)
```

第二个重载用于支持 packed weight 查找。

### B. `BuildExecutionPlan(...)`

位置：

```text
src/execution/execution_plan_builder.cpp
```

核心步骤：

1. 收集 workspace 需求：

```cpp
std::vector<WorkspaceRequirement> workspace_requirements;
for (const ExecutionPlanNodeSpec& node : nodes) {
    workspace_requirements.push_back(node.workspace_requirement);
}
```

2. 规划 workspace offset：

```cpp
PlanWorkspaceRequirements(std::span(workspace_requirements))
```

返回失败则直接返回 `Status`。

3. 遍历每个 node：

```cpp
for (size_t index = 0; index < nodes.size(); ++index)
```

每个 node 依次执行 backend 获取、operator 创建、kernel prepare、packed weight 绑定、step 写入。

### C. Backend 获取

```cpp
runtime.GetBackend(node.device_type)
```

输入：

- `DeviceType`

输出：

- `StatusOr<Backend*>` 或等价 backend 指针结果

失败时：

```cpp
return backend.status();
```

这个 backend 后续用于 kernel resolution。

### D. Selector 构造

每个 node 会通过 helper：

```cpp
MakeSelectorForNode(const ExecutionPlanNodeSpec& node)
```

生成：

```cpp
KernelSelector{
    .device_type = node.device_type,
    .activation_dtype = node.activation_dtype,
    .weight_dtype = node.weight_dtype,
    .weight_format = node.weight_format,
    .isa = node.isa,
    .phase = node.phase,
}
```

这个 selector 是 backend/kernel registry 选择内核的关键输入。

### E. Operator 创建与准备

```cpp
CreateAndPrepareOperator(Backend& backend,
                         const ExecutionPlanNodeSpec& node)
```

返回：

```cpp
StatusOr<PreparedOperator>
```

其中：

```cpp
struct PreparedOperator {
    OperatorPtr op;
    InferenceResult inference;
};
```

内部流程：

```text
CreateAndPrepareOperator
  ├─ OperatorRegistry::Create(node.op_type, MakeOperatorParamsForNode(node))
  ├─ 如果 NotFound:
  │     └─ 返回空 PreparedOperator，用 raw kernel fallback
  ├─ 如果 node.attrs 非空:
  │     └─ 对注册 Operator 报错
  ├─ op->ValidateParams()
  ├─ 如果 node.input_specs 非空:
  │     ├─ op->CheckInputSpecs(node.input_specs)
  │     └─ op->InferOutputShapes(node.input_specs)
  ├─ 构造 OperatorContext
  └─ op->Prepare(op_ctx)
```

这里的关键点是：`input_specs` 为空时，不做 shape 检查和 shape inference。这就是当前架构支持“shape 信息不完整/暂不可得”的方式之一。

### F. 参数来源

```cpp
MakeOperatorParamsForNode(const ExecutionPlanNodeSpec& node)
```

规则：

1. 如果 `node.op_params` 不是 `std::monostate`：

```cpp
return node.op_params;
```

2. 否则：

```cpp
OperatorRegistry::CreateDefaultParams(node.op_type)
```

3. 如果没有默认参数：

```cpp
return OpParams{};
```

因此注册算子一般从 `op_params` 或默认参数构造。

### G. 注册 Operator 路径

以 `RmsNormOp` 为例：

```cpp
AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)
```

注册后，builder 会走：

```text
OperatorRegistry::Create
  → RmsNormOp(params)
  → ValidateParams
  → CheckInputSpecs
  → InferOutputShapes
  → Prepare
```

`Prepare()` 内部会做 kernel resolution：

```cpp
ctx.backend->ResolveKernelInfo(OpType::kRmsNorm, ctx.selector)
```

成功后保存到 operator 内部的 `resolved_kernel_`。

### H. Raw Kernel fallback 路径

如果 `OperatorRegistry::Create(...)` 返回 `kNotFound`，builder 走 fallback：

```cpp
ResolveKernelForNode(*backend.value(), node)
```

内部：

```cpp
backend.ResolveKernelInfo(node.op_type, selector)
```

然后构造：

```cpp
FunctionOperator(
    resolved->op_type,
    resolved->fn,
    std::span<const std::byte>(resolved->attrs),
    resolved->debug_name)
```

这个路径不会做完整 operator-level shape inference，只是把 raw `KernelFunc` 包装进统一执行接口。

### I. Packed weight 绑定

```cpp
ResolvePackedWeightsForNode(const ModelInstance* model_instance,
                            const ExecutionPlanNodeSpec& node)
```

规则：

1. 如果：

```cpp
node.weight_format != WeightFormat::kPacked
```

返回：

```cpp
nullptr
```

2. 如果需要 packed weight 但没有 `ModelInstance`：

```cpp
Status::NotFound("Packed-weight node requires a ModelInstance sidecar")
```

3. 否则：

```cpp
model_instance->FindPackedWeights(node.op_type, selector)
```

成功后返回：

```cpp
packed_weights->storage().data()
```

这个指针最终写入：

```cpp
ExecutionStep::packed_weights
```

### J. 生成最终 step

每个 node 最终变成一个：

```cpp
ExecutionStep{
    .selector = MakeSelectorForNode(node),
    .op = std::move(op),
    .packed_weights = packed_weights.value(),
    .workspace_requirement = workspace_requirements[index],
    .output_specs = std::move(prepared.inference.outputs),
    .runtime_checks = std::move(prepared.inference.runtime_checks),
    .debug_name = nullptr,
}
```

然后：

```cpp
plan.AddStep(...)
```

`ExecutionPlan::AddStep` 会验证：

- `step.op != nullptr`
- `step.op->GetResolvedKernel().fn != nullptr`
- workspace alignment 合法

成功后 push 到内部：

```cpp
std::vector<ExecutionStep> steps_;
```

`AddStep` 是 private，`ExecutionPlanBuilder` 是 friend，所以 `ExecutionPlan` 构建后对外不可变。

## 4. Shape / constraint 在编译流程中的位置

当前 `TensorSpec` 已经不是简单 concrete shape，而是：

```cpp
struct TensorSpec {
    DataType dtype{};
    SymbolicShape shape{};
};
```

operator inference 返回：

```cpp
struct InferenceResult {
    std::vector<TensorSpec> outputs{};
    std::vector<ShapeConstraint> runtime_checks{};
};
```

例如 `RmsNormOp::InferOutputShapes(...)`：

- 输出 shape 直接继承 input activation shape；
- 如果 input hidden symbol 和 weight hidden symbol 不是同一个 symbol，就生成：

```cpp
DimEqualConstraint(input0.dim1, input1.dim0)
```

这个 constraint 被保存到：

```cpp
ExecutionStep::runtime_checks
```

执行时在 kernel 前校验：

```cpp
LayerRunner::RunStep(...)
  → ValidateShapeConstraints(step.runtime_checks, inputs, outputs)
```

所以当前设计是：

```text
编译期能证明的 shape 关系 → CheckInputSpecs / InferOutputShapes 处理
编译期无法证明但运行期可验证的关系 → runtime_checks
运行期真实 TensorView 出现后 → ValidateShapeConstraints
```

## 5. 执行阶段如何消费编译结果

虽然这不是“编译”本身，但它说明最终产物如何被使用。

```cpp
Executor::Execute(plan, bindings)
```

调用：

```cpp
LayerRunner::Run(plan, bindings)
```

`LayerRunner::Run` 遍历：

```cpp
for each ExecutionStep
```

每步：

```cpp
LayerRunner::RunStep(step_index, step, bindings)
```

顺序：

1. 绑定 workspace：

```cpp
bindings.BindWorkspace(step.workspace_requirement)
```

2. 构造 kernel context：

```cpp
BuildKernelContext(step, bindings)
```

写入：

```cpp
KernelContext{
    .device_type = step.selector.device_type,
    .workspace = bindings.GetWorkspaceArena(),
    .packed_weights = step.packed_weights,
    .attrs = resolved.attrs,
}
```

3. 校验 runtime shape constraints：

```cpp
ValidateShapeConstraints(...)
```

4. 调用 operator：

```cpp
step.op->Run(ctx, bindings, step_index)
```

例如 `RmsNormOp::Run` 会：

```cpp
bindings.GetStepTensorBinding(step_index)
```

然后构造：

```cpp
cpu::CpuRmsNormParams
```

设置：

```cpp
ctx.kernel_params = &params;
```

最后调用：

```cpp
resolved_kernel_.fn(ctx)
```

## 6. 当前 Graph Compile 阶段边界

当前代码已经落地的边界包括语义图优化（pass pipeline）和完整的优化+降低组合入口。

### 6.1 按阶段划分的编译流程

```text
                              ┌──────────────────────┐
                              │    ModelGraph         │
                              │  (语义 DAG)            │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  OptimizeModelGraph   │
                              │  O0/O1/O2+ pipeline   │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  optimized ModelGraph  │
                              │  (ID table owned by    │
                              │   CompiledModelGraph)  │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  LowerModelGraph      │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │     LoweredGraph      │
                              │  (ExecutionPlanNodeSpec│
                              │   + bindings)         │
                              └──────────────────────┘
```

### 6.2 规范组合 API

`CompileModelGraph` 将优化和降低合并为一步，返回拥有完整生命周期的编译产物：

```text
ModelGraph
  → CompileModelGraph(graph, GraphCompileConfig)
      ├─ OptimizeModelGraph(graph, config.optimization)
      │    └─ StatusOr<ModelGraph> optimized
      ├─ LowerModelGraph(optimized, config.lowering)
      │    └─ StatusOr<LoweredGraph> lowered
      └─ CompiledModelGraph { optimized_graph, lowered }
```

`GraphCompileConfig` 统一管理两个阶段的配置：

```text
GraphCompileConfig {
    PassContext optimization{};       // 默认 O2
    GraphLoweringConfig lowering{};  // 默认 CPU/scalar/plain/both
}
```

`OptimizeModelGraph` 按 `opt_level` 确定性地选择 pass pipeline（O0 无 pass，O1 ConstantFolding→DCE，O2+ ConstantFolding→SiluMulFusion→DCE）。特征 flag（`enable_constant_folding`、`enable_swiglu_fusion`、`enable_dce`）不参与 pass 注册，仅控制已注册 pass 的运行时行为。

### 6.3 当前的显式降低入口

语义图优化后，通过 lowering bridge 产生 `LoweredGraph`，其 `steps` 字段作为 `ExecutionPlanBuilder` 的输入：

```text
optimized ModelGraph
  → LowerModelGraph
  → LoweredGraph
       ├─ steps (std::vector<ExecutionPlanNodeSpec>)
       ├─ step_bindings (std::vector<LoweredStepBinding>)
       ├─ model_inputs
       ├─ model_outputs
       └─ state_aliases
  → ExecutionPlanBuilder::Build(steps)
  → ExecutionPlan
```

执行计划构建本身仍只消费 `steps`：

```text
std::vector<ExecutionPlanNodeSpec>
  → ExecutionPlanBuilder::Build
  → ExecutionPlan
```

### 6.4 仍待补齐的生产化部分

- 从 `ModelInstance` / HF config / resolved weights 自动生成图结构并经过优化、降低到可执行 node specs 的完整生产入口
- 更完整的 runtime tensor/state binding 接线
- 更完整的 Llama layer 执行覆盖与后续优化 pass

所以目前 `ExecutionPlanNodeSpec` 仍主要由测试或 lowering bridge 产出，完整的"模型加载后自动生成图结构、优化、lowering 并构建可执行计划"的端到端生产入口还未完成。

## 7. 当前最终编译产物

最终产物是：

```cpp
ExecutionPlan
```

它包含：

```cpp
std::vector<ExecutionStep>
```

每个 `ExecutionStep` 是一个已经准备好的可执行步骤，包含：

- `KernelSelector selector`
- `OperatorPtr op`
- `const void* packed_weights`
- `WorkspaceRequirement workspace_requirement`
- `std::vector<TensorSpec> output_specs`
- `std::vector<ShapeConstraint> runtime_checks`
- `const char* debug_name`

这就是当前“编译结果”。

## 8. 总结

当前代码库中的“图编译”已经涵盖了三个层次：

### 层次 A：显式的优化+编译组合入口（已实现）

```text
ModelGraph
  → CompileModelGraph(graph, GraphCompileConfig)
      ├─ OptimizeModelGraph(graph, config.optimization)
      │    └─ StatusOr<ModelGraph> optimized
      ├─ LowerModelGraph(optimized, config.lowering)
      │    └─ StatusOr<LoweredGraph> lowered
      └─ CompiledModelGraph { optimized_graph, lowered }
```

`CompileModelGraph` 是 Phase 1 中从语义图到可执行 artifact 的规范入口。优化 pipeline 由 `opt_level` 确定性地选择（O0 无 pass，O1 ConstantFolding→DCE，O2+ ConstantFolding→SiluMulFusion→DCE）。

### 层次 B：LoweredGraph 到 ExecutionPlan 的构建（已实现）

```text
LoweredGraph.steps (std::vector<ExecutionPlanNodeSpec>)
  → workspace planning
  → operator creation
  → shape inference / runtime constraint extraction
  → backend kernel resolution
  → packed weight sidecar binding
  → immutable ExecutionPlan
```

`ExecutionPlanBuilder` 不理解模型拓扑，它只消费已经准备好的 node specs。模型拓扑到 `ExecutionPlanNodeSpec` 的转换由 `LowerModelGraph` 负责。

### 层次 C：ModelInstance 到 ExecutionPlan 的生产管线（未完成）

`ModelInstance` / HF config / resolved weights → graph building → optimization → lowering → execution plan 的完整生产入口目前仍不完整。当前 `ExecutionPlanNodeSpec` 主要由 lowering bridge 或测试产出；从模型加载到自动执行计划生成的端到端流程还需要 `ModelInstance` 驱动的完整 graph builder → compile pipeline 衔接（见 §6.4）。
