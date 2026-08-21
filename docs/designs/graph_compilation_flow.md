# 当前图编译流程实现梳理

本文按当前代码事实梳理 AetherMind 中图编译相关实现。当前仓库已经具备 `LoadedModel -> ModelGraph -> optimized ModelGraph -> LoweredModelArtifact` 的模型编译主流程，以及 `LoweredGraph -> ExecutionPlan` 的计划构建流程；二者之间的 graph-driven weight materialization、完整 kernel 覆盖和生产 ExecutionPlan 串联仍未完成。

## 1. 当前实际入口

```cpp
ExecutionPlanBuilder::Build(RuntimeContext& runtime,
                            const std::vector<ExecutionPlanNodeSpec>& nodes)

ExecutionPlanBuilder::Build(RuntimeContext& runtime,
                            const PackedWeightStore& packed_weight_store,
                            const std::vector<ExecutionPlanNodeSpec>& nodes)
```

输入核心是：

```cpp
struct ExecutionPlanNodeSpec {
    OpType op_type = OpType::kUnknown;
    // Execution capabilities this step requires: lowering records the
    // configured prefix (device/ISA/weight format/phase) and fills the
    // activation/weight dtypes from the operator's inputs and outputs.
    KernelSelector selector{};
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

也就是说，`ExecutionPlanNodeSpec` 是给测试、手工构造和其他低层调用的 **untrusted execution request**；builder 会重跑 `InferOperator` 并严格比较其 metadata。compiler 的正式产物是 `LoweredStepSpec` 组成的 immutable `LoweredGraph`；`ModelGraph` 经 `LowerModelGraph()` 转换后，execution 内部先验证 artifact 并将 state aliases 转为 `StateAliasPlan`，再构建计划。`LoweredStepSpec` 不携带 `workspace_requirement`——workspace 大小是 kernel 实现细节，compiler 产物无法计算；只有 untrusted `ExecutionPlanNodeSpec` 由调用方显式预填。

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
BuildExecutionPlan(runtime, packed_weight_store?, nodes)
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
        │   // finalized LoweredGraph was validated by compiler; metadata is
        │   // copied without a second semantic inference. Untrusted raw node
        │   // specs are re-inferred and must match exactly.
        │
        ├─ ResolvePackedWeightsForNode(packed_weight_store?, node)
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
ExecutionPlanBuilder::Build(runtime, packed_weight_store, nodes)
```

调用：

```cpp
BuildExecutionPlan(runtime, &packed_weight_store, nodes)
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
for (const auto& node : nodes) {
    workspace_requirements.push_back(GetNodeWorkspaceRequirement(node));
}
```

untrusted `ExecutionPlanNodeSpec` 由调用方显式预填 workspace 需求；`LoweredStepSpec` 不携带该字段，lowered 步骤取默认零需求。

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
ResolvePackedWeightsForNode(const PackedWeightStore* packed_weight_store,
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

2. 如果需要 packed weight 但没有 `PackedWeightStore`：

```cpp
Status::NotFound("Packed-weight node requires a PackedWeightStore")
```

3. 否则：

```cpp
packed_weight_store->Find(node.op_type, selector)
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
                              │  (LoweredStepSpec     │
                              │   + bindings)         │
                              └──────────────────────┘
```

### 6.2 规范组合调用

优化与降低仍是 compiler-owned 的两个显式阶段；`ModelCompiler` 用 `ModelCompileOptions` 为生产模型编译提供唯一的组合入口，诊断/测试场景可继续直接串联：

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

`OptimizeModelGraph` 按 `opt_level` 确定性地选择 pass pipeline（O0 无 pass，O1 ConstantFolding→DCE，O2+ ConstantFolding→QkvLinearFusion→GateUpLinearFusion→SiluMulFusion→AddRmsNormFusion→DCE）。特征 flag 不参与 pass 注册，仅控制已注册 pass 的运行时行为。

### 6.3 当前的显式降低入口

语义图优化后，compiler lowering 产生 finalized `LoweredGraph`。其 storage 私有，只通过 const span accessor 公开；execution 是唯一把它适配成 plan 的模块：

```text
optimized ModelGraph
  → LowerModelGraph
  → LoweredGraph
       ├─ steps() (LoweredStep{LoweredStepSpec, binding}; 1:1 配对)
       ├─ values() (LoweredValueDesc, indexed by GraphValueId)
       ├─ model_inputs()
       ├─ model_outputs()
       ├─ state_aliases() (LoweredStateAlias, step/port coordinates
       │                 recorded at lowering time)
  → ExecutionPlanBuilder::Build(runtime, lowered)   // consumes full LoweredGraph
  →   ├─ ResolveStateAliasesForExecution(lowered)  // execution-private conversion
  →   └─ BuildExecutionPlan(runtime, model?, steps, alias_plan, trusted=true)
  → ExecutionPlan
```

执行计划构建的主要入口接受完整 `LoweredGraph`（含 steps、values、inputs/outputs、state_aliases），而非仅消费 steps：

```text
LoweredGraph (compiler-owned, immutable after finalization)
  ├─ steps()          (span<LoweredStep>, LoweredStepSpec + binding 1:1)
  ├─ values()         (span<LoweredValueDesc>)
  ├─ model_inputs()   (span<GraphValueId>)
  ├─ model_outputs()  (span<GraphValueId>)
  └─ state_aliases()  (span<LoweredStateAlias>, step/port coordinates
                     recorded at lowering time)
      │
      ▼
ExecutionPlanBuilder::Build(RuntimeContext&, LoweredGraph const&)
      │
      ├─ ValidateLoweredGraph(lowered)
      ├─ ResolveStateAliasesForExecution(lowered)
      │     └─ converts verified aliases → StateAliasPlan
      │
      ▼
BuildExecutionPlan(runtime, model?, steps, alias_plan, trusted=true)
      │
      ▼
ExecutionPlan{... steps ..., state_alias_plan{...}}
```

保留的 `Build(runtime, vector<ExecutionPlanNodeSpec>)` 重载仅用于测试/手工构造场景，内部构造空 StateAliasPlan（无 state alias 支持）。

### 6.4 state alias 的声明与解析机制

状态别名的声明、compiler validation 与 runtime conversion 分属三个清晰职责，均不依赖 lowering 对具体算子的认知：

- **声明**：算子在其 `OperatorSchema::state_alias_ports` 中声明（输入端口名, 输出端口名）对（如 KVCacheUpdate 声明 k_cache_in→k_cache_out、v_cache_in→v_cache_out）。新增有状态算子只需声明该字段，compiler lowering 无需 `OpType` 特判。
- **记录**：`LowerModelGraph` 按拓扑序逐节点消费 schema 声明，在发出每个节点时以当时已知的 step 索引与 schema 端口 index 生成 `LoweredStateAlias`（step_index/input_port/output_port + 值对），不扫描、不匹配。
- **验证**：compiler finalization 检查 schema declaration、state-port kind、binding/value ID、`StateValue` payload 与 state binding 一致性，以及 duplicate/conflicting aliases；失败为 trusted artifact 的 `Internal` 错误。
- **解析**：execution-private `ResolveStateAliasesForExecution` 在 trust boundary 重查 artifact 结构后，按 `(step_index, input_port, output_port)` 确定性排序并转换为 runtime `ResolvedStateAlias`。

### 6.5 仍待补齐的生产化部分

- graph-driven weight materialization/prepack，并以具体 `WeightBinding` 识别 artifact
- 更完整的 runtime tensor/state binding 接线
- 完整的 Llama layer kernel 覆盖与 `LoweredModelArtifact → ExecutionPlan` 编排

因此“模型加载后自动生成图结构、优化、lowering”已经可用；“模型加载后构建正确、可执行的 ExecutionPlan”仍未完成。

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

`OptimizeModelGraph` + `LowerModelGraph` 是 Phase 1 中从语义图到 lowering artifact 的两个规范入口。优化 pipeline 由 `opt_level` 确定性地选择（O0 无 pass，O1 ConstantFolding→DCE，O2+ ConstantFolding→QkvLinearFusion→GateUpLinearFusion→SiluMulFusion→AddRmsNormFusion→DCE）。

### 层次 B：LoweredGraph 到 ExecutionPlan 的构建（已实现）

```text
LoweredGraph::steps() (std::span<const LoweredStep>, carrying LoweredStepSpec)
  → workspace planning
  → operator creation
  → compact runtime tensor-spec derivation
  → backend kernel resolution
  → packed weight sidecar binding
  → immutable ExecutionPlan
```

`ExecutionPlanBuilder` 不理解模型拓扑；它消费 compiler artifact 并在 implementation 内部适配为 execution request metadata。模型拓扑到 `LoweredStepSpec` 的转换由 compiler `LowerModelGraph` 负责。

### 层次 C：LoadedModel 到 LoweredModelArtifact 的生产模型编译管线（已实现）

`ModelLoader::Load` 生成 `LoadedModel`，`ModelCompiler::Compile` 串联 `ModelGraphBuilder::BuildLlamaDense`、`OptimizeModelGraph` 与 `LowerModelGraph`，并以 `LoweredModelArtifact` 将 loaded raw-weight ownership 与 lowering artifact 一起返回。`ModelCompiler::LoadAndCompile` 是对应的一站式薄 facade。该阶段不调用 backend、prepack 或 `ExecutionPlanBuilder`。

### 层次 D：LoweredModelArtifact 到 ExecutionPlan 的生产管线（未完成）

正确的 production plan 仍需要从 optimized graph 的具体 `WeightBinding` 生成独立 weight artifact，并在完整 kernel 覆盖后执行 plan-time kernel resolution（见 §6.5）。
