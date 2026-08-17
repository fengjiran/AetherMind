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
    OpType op_type = OpType::kUnknown;
    DeviceType device_type = DeviceType::kCPU;
    DataType act_dtype{};                  // renamed from activation_dtype
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
    WorkspaceRequirement workspace_requirement{};
    // Schema-port-ordered input specs, including state ports that do not
    // contribute to runtime tensor bindings.
    std::vector<TensorSpec> input_specs{};
    std::vector<TensorSpec> output_specs{};
    // Deferred runtime shape constraints (derived during graph construction,
    // carried through lowering without re-inference).
    std::vector<ShapeConstraint> runtime_checks{};
    OpParams op_params{};
};
```

也就是说，`ExecutionPlanBuilder` 的直接输入可以是已经线性化好的 node spec 列表；同时也接受 `LoweredGraph`（包含 steps、step_bindings、inputs/outputs、state_aliases）。`ModelGraph` 通过 `LowerModelGraph()` 先转换为 `LoweredGraph`，再由 `ExecutionPlanBuilder::Build(RuntimeContext, LoweredGraph)` 执行 state alias resolution 并构建计划。

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
        ├─ PrepareNodeMetadata(node, trusted)
        │      ├─ schema -> MakeCompactInputSpecs(...)
        │      └─ raw node: InferOperator(...) 严格核对 metadata
        │
        ├─ backend.PrepareKernel(node.op_type, selector, node.op_params)
        │      └─ 返回按值持有的 ResolvedKernel
        │
        │   // NOTE: trusted LoweredGraph metadata is copied verbatim because
        │   // graph construction already called InferOperator. Untrusted raw
        │   // node specs are re-inferred and must match exactly.
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

每个 node 依次执行 backend 获取、语义 metadata 准备、kernel prepare、packed weight 绑定、step 写入。

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

### E. 语义 metadata 准备

`PrepareNodeMetadata(node, trusted)` 始终根据 `OperatorSchema` 调用
`MakeCompactInputSpecs`，因此 state/resource port 不会进入 runtime tensor
binding，也不会进入 `InferOperator` 的 compact 输入。

- `trusted == true`（`LoweredGraph`）：graph construction 已执行
  `InferOperator`，builder 直接保留 lowered 的 `output_specs` 与
  `runtime_checks`，不会创建第二个语义权威。
- `trusted == false`（直接 node spec）：`op_params` 必须不是
  `std::monostate`；builder 调用 `InferOperator`，并要求调用方提供的
  `output_specs` 和 `runtime_checks` 精确匹配推导结果。

### F. Kernel prepare

```cpp
backend.PrepareKernel(node.op_type, MakeSelectorForNode(node), node.op_params)
```

这是唯一的 plan-time kernel resolution API。backend 选择 descriptor，并将
由 `OpParams` 派生的不可变 metadata 写入返回的 `ResolvedKernel::attrs`。
例如 RmsNorm 的 epsilon 在此阶段冻结；运行期不再读取 `OpParams`，也不再
进行 backend 或 registry lookup。

### G. Packed weight 绑定

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

### H. 生成最终 step

每个 node 最终变成一个：

```cpp
ExecutionStep{
    .selector = MakeSelectorForNode(node),
    .kernel = std::move(kernel),
    .packed_weights = *packed_weights,
    .workspace_requirement = workspace_requirements[index],
    .input_specs = std::move(metadata.compact_input_specs),
    .output_specs = std::move(metadata.output_specs),
    .runtime_checks = std::move(metadata.runtime_checks),
}
```

然后：

```cpp
plan.AddStep(...)
```

`ExecutionPlan::AddStep` 会验证：

- `step.kernel.op_type != OpType::kUnknown`
- `step.kernel.fn != nullptr`
- kernel params builder/size 契约合法
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

例如 `InferRmsNorm(...)`：

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
编译期能证明的 shape 关系 → InferOperator 处理
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
    .attrs = step.kernel.attrs,
}
```

3. 校验 runtime shape constraints：

```cpp
ValidateShapeConstraints(...)
```

4. 从 `RuntimeBindingContext` 取得当前 step 的 tensor binding，校验 compact
   input/output arity。

5. 由通用 `InvokeKernel(step.kernel, ctx, inputs, outputs)` 执行：若 descriptor
   注册了 params builder，它在栈上构造 backend-specific params 并临时写入
   `ctx.kernel_params`，然后调用冻结的 `step.kernel.fn`。

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
                              │  (caller-owned，诊断/   │
                              │   验证场景自行持有)     │
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

### 6.2 规范组合调用

优化与降低是两个显式阶段，由调用方直接串联（不再有组合器函数或聚合配置类型）：

```text
ModelGraph
  → OptimizeModelGraph(graph, PassContext)      // 默认 opt_level = 2
  │    └─ StatusOr<ModelGraph> optimized
  → LowerModelGraph(optimized, GraphLoweringConfig)
       └─ StatusOr<LoweredGraph> lowered（优化图 caller-owned）
```

两个阶段的配置分别由 `PassContext` 与 `GraphLoweringConfig` 独立提供（默认 O2 优化、CPU/scalar/plain/both 低化）：

```text
PassContext {
    opt_level = 2;               // 默认 O2
    // + feature flags、checkpoint_every、const_eval_policy（逐字转发给每个 pass）
}
GraphLoweringConfig {
    device_type / isa / weight_format / phase  // 默认 CPU/scalar/plain/both
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
       ├─ state_aliases
  → ExecutionPlanBuilder::Build(runtime, lowered)   // consumes full LoweredGraph
  →   ├─ ResolveStateAliases(lowered)              // validates + converts KV cache aliases
  →   └─ BuildExecutionPlan(runtime, model?, steps, alias_plan, trusted=true)
  → ExecutionPlan
```

执行计划构建的主要入口接受完整 `LoweredGraph`（含 steps、step_bindings、inputs/outputs、state_aliases），而非仅消费 steps：

```text
LoweredGraph
  ├─ steps           (std::vector<ExecutionPlanNodeSpec>)
  ├─ step_bindings   (std::vector<LoweredStepBinding>)
  ├─ model_inputs    (std::vector<GraphInput>)
  ├─ model_outputs   (std::vector<GraphOutput>)
  └─ state_aliases   (std::vector<StateAlias>)
      │
      ▼
ExecutionPlanBuilder::Build(RuntimeContext&, LoweredGraph const&)
      │
      ├─ ResolveStateAliases(lowered)
      │     └─ validates alias pairs → returns StateAliasPlan
      │
      ▼
BuildExecutionPlan(runtime, model?, steps, alias_plan, trusted=true)
      │
      ▼
ExecutionPlan{... steps ..., state_alias_plan{...}}
```

保留的 `Build(runtime, vector<ExecutionPlanNodeSpec>)` 重载仅用于测试/手工构造场景，内部构造空 StateAliasPlan（无 KV cache alias 支持）。

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

### 层次 A：显式的优化 + 降低阶段入口（已实现）

```text
ModelGraph
  → OptimizeModelGraph(graph, PassContext)
  │    └─ StatusOr<ModelGraph> optimized
  → LowerModelGraph(optimized, GraphLoweringConfig)
       └─ StatusOr<LoweredGraph> lowered（优化图 caller-owned）
```

`OptimizeModelGraph` + `LowerModelGraph` 是 Phase 1 中从语义图到可执行 artifact 的两个规范入口。优化 pipeline 由 `opt_level` 确定性地选择（O0 无 pass，O1 ConstantFolding→DCE，O2+ ConstantFolding→SiluMulFusion→DCE）。

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
