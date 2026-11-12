# shape-inference：形状推导基础设施与执行期约束保障

`shape-inference` 提供符号维度、张量 spec、shape 约束与约束求值的通用基础设施。算子的 `Infer*` 语义函数与 `InferenceResult` 由 `operators` 定义并消费这些类型；graph、compiler、execution 只传递和校验这些纯数据对象。系统分层见 [architecture.md](../architecture.md)。

## 1. 职责与边界

### 负责

| 职责 | 位置 |
|---|---|
| `ShapeSymbol` / `SymbolicShape`：静态维度、命名符号、未知维度、`Join` / `Unify` | [shape_symbol.h](../../include/aethermind/shape_inference/shape_symbol.h)、[shape_symbol.cpp](../../src/shape_inference/shape_symbol.cpp) |
| `TensorSpec`：dtype + `SymbolicShape` | [tensor_spec.h](../../include/aethermind/shape_inference/tensor_spec.h) |
| `ShapeConstraint` 及六种约束条件、`TensorPort` / `DimLocator`、三态求值结果 | [shape_constraint.h](../../include/aethermind/shape_inference/shape_constraint.h) |
| 约束求值：符号态求值、concrete TensorView 求值、批量运行时校验 | [shape_constraint_evaluator.h](../../include/aethermind/shape_inference/shape_constraint_evaluator.h)、[shape_constraint_evaluator.cpp](../../src/shape_inference/shape_constraint_evaluator.cpp) |
| `SymbolConstraintSolver`：plan-time 等式的 union-find 事实库 | [symbol_constraint_solver.h](../../include/aethermind/shape_inference/symbol_constraint_solver.h)、[symbol_constraint_solver.cpp](../../src/shape_inference/symbol_constraint_solver.cpp) |
| broadcast 推导：concrete shape、stride 扩展、符号 broadcast + deferred axis | [broadcast.h](../../include/aethermind/shape_inference/broadcast.h)、[broadcast.cpp](../../src/shape_inference/broadcast.cpp) |

### 不得负责什么

| 不属于 shape-inference | 归属 |
|---|---|
| 算子端口 ABI、参数校验、`Infer*` 的算子语义 | `operators`（[operator_inference.h](../../include/aethermind/operators/operator_inference.h)） |
| 图节点/边的存储、producer 一致性、拓扑序、重写 | `graph`（[graph.md](graph.md)） |
| lowering、selector dtype、`LoweredStepSpec` | `compiler` |
| TensorView 绑定、activation 分配、workspace、kernel resolve | `execution` / `runtime` |
| 物理 stride/layout 合法性、packed weight 布局、kernel 参数构造 | `backend` / `execution` |
| 在热路径按 Execute 重新推导 shape | 当前不存在；执行期只消费已准备的 bindings |
| 全局图级 shape environment / 跨 step 等式传播 | 当前不存在（见 §4.5） |

依赖方向：`shape_inference` 只依赖 `base`、`dtypes` 与基础容器；`operators`、`graph`、`compiler`、`execution` 都可以直接依赖它，但 shape_inference 不得反向依赖它们中的任何一个。

## 2. 关键数据结构与所有权

### 2.1 `ShapeSymbol`

`ShapeSymbol` 是值类型，内部只有一个 `int64_t value_`，可安全复制并可并发读取。语义由取值区间决定：

| 取值 | 含义 | 运行期处理 |
|---|---|---|
| `>= 0` | 已知静态维度 | concrete dim 必须等于该值 |
| `-1`（`Unknown()`） | 无 identity 的未知维度 | 不绑定、不传播；只要求 concrete dim 非负 |
| `<= -2` | 由 `Create()` 生成的符号维度 | 同一 value 代表同一 identity；运行期必须绑定到同一个 concrete dim |

- `Create()` 通过进程级 `std::atomic<int64_t> next_symbol_` 递减分配 `-2, -3, ...`；符号不会复用。
- `CreateFromValue(v)` 要求 `v >= 0`，否则 `AM_CHECK` 终止。
- `operator<=>` 为默认比较；symbol identity 就是 value 相等。
- `GetStaticValue()` 在非静态时 `AM_CHECK` 终止。

`JoinShapeSymbol(a, b)`：相等返回 `a`，否则返回一个全新符号。用于控制流分支放宽。

`UnifyShapeSymbol(a, b)`：严格合一，返回“知识更多”的一方；static/static 冲突返回 `InvalidArgument`；symbolic/symbolic 不同且无 solver 参与时退化为 `Unknown()`。

### 2.2 `SymbolicShape`

`SymbolicShape` 内部是 `std::optional<std::vector<ShapeSymbol>>`：

- `nullopt`：unranked；
- 空 vector：rank-zero 标量；
- 非空 vector：ranked，每一维一个 `ShapeSymbol`。

| 接口 | 语义 |
|---|---|
| `IsRanked` / `IsUnranked` / `IsRankZero` | 形状状态查询 |
| `rank()` | ranked 返回维度数，unranked 返回 `nullopt` |
| `operator[]` | 越界或 unranked 时 `AM_CHECK` 终止 |
| `IsStatic()` | ranked 且所有维度均为 static |
| `Join(other)` | 分支合并：rank 不同或任一侧 unranked → unranked；逐维 `JoinShapeSymbol` |
| `Unify(other)` | 严格合一：unranked 视为另一侧；rank 不同报错；逐维 `UnifyShapeSymbol` |
| `GetSymbolicDims()` | 返回每一维是否为非静态；unranked 返回 `nullopt` |

`AreProvablyEqual(lhs, rhs)`：两侧都不是 `Unknown()` 且值相等。它是 `Infer*` 中“同一符号 identity 即已证明相等”的判定入口。

### 2.3 `TensorSpec`

`TensorSpec { DataType dtype; SymbolicShape shape; }`，相等比较为 dtype 与 shape 同时相等。它是 graph value、`InferenceResult`、`LoweredStepSpec` / `ExecutionPlan::values()` 之间传递的唯一 shape/dtype 表述，不携带 stride、address 或 layout。`ExecutionStep` 本身不保存 spec，只保存 value id 与 `runtime_checks`；spec 由 `ExecutionPlan::values()` 统一持有。

### 2.4 `ShapeConstraint`

`ShapeConstraint` 是纯数据对象：`condition`（`ConstraintVariant`）+ `error_context`（`std::string_view`）。

| 条件类型 | 语义 |
|---|---|
| `DimEqualConstraint` | 两个 `DimLocator` 指向的维度相等 |
| `DimBroadcastableConstraint` | `a == b || a == 1 || b == 1` |
| `VolumeEqualConstraint` | 两组维度的乘积相等；空 list 表示标量 volume 1 |
| `RankEqualConstraint` | 指定 port 的 rank 等于目标值 |
| `RankAtLeastConstraint` | 指定 port 的 rank 不小于目标值 |
| `DimPositiveConstraint` | 指定维度严格大于 0 |

`TensorPort { direction, tensor_idx }` 用输入/输出方向 + 张量下标定位端口；`DimLocator { tensor_port, dim_index }` 再定位维度。端口坐标是推理端口坐标，不是 kernel 端口坐标；execution plan builder 只把输出 port 投影到 compact kernel output port，输入 port 保持推理端口顺序。

`error_context` 借用发出约束的算子 `Infer*` 中的字符串字面量，必须是 static-duration 存储。`ShapeConstraint` 会被复制进 graph node、lowered step 和 execution step，但 `error_context` 始终只是借用。

`ShapeConstraint` 的 `operator==` / `operator<=>` 只比较 `condition`，不比较 `error_context`；因此以 `std::set` 等容器去重时按条件去重。graph 的 stale 检测会额外比较 `error_context`（见 [graph.md](graph.md) §4.1）。

### 2.5 `InferenceResult`

```cpp
struct InferenceResult {
    std::vector<TensorSpec> outputs{};
    std::vector<ShapeConstraint> runtime_checks{};
};
```

它由 `operators` 定义（[inference_result.h](../../include/aethermind/operators/inference_result.h)），不持有 solver，也不持有输入 spec。`runtime_checks` 表示“算子语义产生、plan-time 未能证明、需要执行期验证”的约束。

### 2.6 `SymbolConstraintSolver`

plan-time 等式事实库，使用 union-find：

| 接口 | 语义 |
|---|---|
| `AddEqual(lhs, rhs)` | 记录两个符号相等；static 冲突返回 `InvalidArgument`；`Unknown` 静默忽略；symbolic 按 rank 合并 |
| `AddEqual(symbol, static_value)` | 绑定符号到静态值；冲突返回 `InvalidArgument` |
| `EvaluateEqual(lhs, rhs)` | 返回 `kSatisfied` / `kViolated` / `kDeferred` |
| `AreEqual(lhs, rhs)` | `EvaluateEqual == kSatisfied` |
| `GetStaticBinding(symbol)` | 沿根查找静态绑定；static 直接返回自身；unknown/未绑定返回 `nullopt` |

存储是 dense vector，索引由符号值映射：`-2 → 0`、`-3 → 1`……因此 solver 适合长生命周期、单一持有者；每个算子创建独立 solver 会因全局符号值跨度而扩容到很大索引。当前生产代码没有 `SymbolConstraintSolver` 调用点，只有单元测试直接使用它；它不参与 `Infer*`、graph 校验或 execution plan 构建。

### 2.7 其他运行期辅助类型

| 类型 | 语义 |
|---|---|
| `ShapeConstraintEvaluationResult` | `kSatisfied` / `kViolated` / `kDeferred` |
| `SymbolValueMap` | `unordered_map<int64_t, int64_t>`，辅助函数内部的“符号值 → concrete dim”绑定表 |
| `DeferredBroadcastAxis` | 需要运行期解析的 broadcast 轴对（lhs/rhs 原始轴下标） |
| `SymbolicBroadcastResult` | `output_shape` + `deferred_axes`；deferred axis 按输出轴升序 |

### 2.8 所有权与生命周期

| 对象 | 所有权 | 生命周期 / 线程安全 |
|---|---|---|
| `ShapeSymbol` / `SymbolicShape` / `TensorSpec` / `ShapeConstraint` | 值类型，复制拥有 | 可自由复制；`ShapeSymbol` 读取线程安全 |
| `ShapeConstraint::error_context` | 借用 static-duration 字面量 | 必须比所有复制品活得久；不拥有、不分配 |
| `InferenceResult` | 值类型 | 由 `InferOperator` 按值返回；不含 solver |
| `SymbolConstraintSolver` | 调用方按值/引用拥有 | plan-time 使用；非线程安全；dense 存储随全局符号空间增长 |
| `SymbolValueMap` | 每次校验调用的局部变量 | 不跨调用、不进入 plan；当前不是 per-Execute 状态 |

## 3. 接口与代码位置

### 3.1 文件

| 头文件 | 实现 | 内容 |
|---|---|---|
| [shape_symbol.h](../../include/aethermind/shape_inference/shape_symbol.h) | [shape_symbol.cpp](../../src/shape_inference/shape_symbol.cpp) | `ShapeSymbol`、`SymbolicShape`、`Join` / `Unify`、`AreProvablyEqual` |
| [tensor_spec.h](../../include/aethermind/shape_inference/tensor_spec.h) | 无 | `TensorSpec` |
| [shape_constraint.h](../../include/aethermind/shape_inference/shape_constraint.h) | 无 | 约束类型与三态结果 |
| [shape_constraint_evaluator.h](../../include/aethermind/shape_inference/shape_constraint_evaluator.h) | [shape_constraint_evaluator.cpp](../../src/shape_inference/shape_constraint_evaluator.cpp) | 符号/运行期求值、spec 前提校验 |
| [symbol_constraint_solver.h](../../include/aethermind/shape_inference/symbol_constraint_solver.h) | [symbol_constraint_solver.cpp](../../src/shape_inference/symbol_constraint_solver.cpp) | union-find 事实库 |
| [broadcast.h](../../include/aethermind/shape_inference/broadcast.h) | [broadcast.cpp](../../src/shape_inference/broadcast.cpp) | broadcast shape/stride 推导 |

### 3.2 通用接口

| 接口 | 语义 |
|---|---|
| `ShapeSymbol::Create()` | 生成新的符号维度 |
| `ShapeSymbol::CreateFromValue(v)` | 生成静态维度；`v >= 0` |
| `ShapeSymbol::Unknown()` | 无 identity 的未知维度 |
| `JoinShapeSymbol(a, b)` | 分支放宽；不同则新建符号 |
| `UnifyShapeSymbol(a, b)` | strict 合一；冲突报错；symbolic 冲突退化为 `Unknown()` |
| `SymbolicShape::Join` / `Unify` | 形状级分支合并 / 严格合一 |
| `InferBroadcastShape(lhs, rhs)` | 符号 broadcast；不创建新符号；歧义维度退化为 `Unknown()` 并记录 deferred axis |
| `BroadcastShapes(lhs, rhs)` | concrete broadcast shape |
| `BroadcastInputStrides(input_shape, input_strides, output_shape)` | 为 broadcast 输入扩展 stride；leading/广播轴 stride 为 0 |

### 3.3 约束求值

| 接口 | 输入 | 行为 |
|---|---|---|
| `EvaluateShapeConstraint(constraint, span<const SymbolicShape>, span<const SymbolicShape>)` | 符号 shape | 纯函数，不读 solver；返回三态 |
| `EvaluateShapeConstraint(constraint, span<const TensorView>, span<const MutableTensorView>)` | concrete view | 返回 `StatusOr<result>`；越界 port/dim 返回 `InvalidArgument`，不 abort；结果不会 `kDeferred` |
| `ValidateShapeConstraints(constraints, inputs, outputs)` | concrete view | 逐条求值；violated → `InvalidArgument(error_context)`；deferred → `Internal`（运行期不应出现） |
| `ValidateConcreteShapeAgainstSpec(spec, dtype, shape, role, port_index, symbol_value_map)` | plan spec + concrete shape | 校验 dtype、rank、静态维度、符号 identity；`Unknown` 不绑定；首次绑定写入 map，冲突报错 |
| `ValidateTensorBindingPremises(input_specs, output_specs, inputs, outputs)` | 单 step 的 compact specs + 绑定 view | 对每个 port 调用上面的校验；无效 view（默认构造的测试桩）跳过；`SymbolValueMap` 在函数内部新建 |

符号态求值的关键规则（不涉及 solver）：

| 条件 | `kSatisfied` | `kViolated` | `kDeferred` |
|---|---|---|---|
| `DimEqual` | 两侧 `ShapeSymbol` 值相等 | 两侧都是 static 且不等 | 其余 |
| `DimBroadcastable` | 值相等，或任一侧 static 1 | 两侧 static 且既不等也都不是 1 | 其余 |
| `DimPositive` | static 且 > 0 | static 且 ≤ 0 | 非 static |
| `VolumeEqual` | 两侧维度列表逐位同值，或两侧可静态计算且乘积相等 | 两侧均可静态计算且乘积不等 | 存在非 static 维度或乘积溢出 |
| `RankEqual` / `RankAtLeast` | rank 已知且满足 | rank 已知且不满足 | unranked 或 rank 不可得 |

`DimEqual` 的“值相等”分支也覆盖两个 `Unknown()`，因为它们的内部值同为 `-1`；这一行为只影响符号态求值，不影响运行期 concrete 校验。

### 3.4 `Infer*` 语义层接口（operators）

`Infer*` 不属于 shape_inference 的实现，但它是 shape 约束的唯一生产者：

| 接口 | 语义 |
|---|---|
| `InferOperator(OpType, const OpParams&, std::span<const TensorSpec> inputs)` | 公共 dispatch 入口；输入按 operator schema port 顺序排列 |
| `detail::Infer*` | 各算子的参数校验、dtype/rank 校验、输出 spec 推导、deferred `ShapeConstraint` 生成 |
| `ValidateInferenceInputCount(OpType, inputs)` | 校验输入数量与 schema 一致 |
| `MakeCompactInputSpecs(schema, all_inputs)` | 过滤 `contributes_tensor_spec == false` 的端口，得到推理输入 |
| `DeriveSelectorDTypes(schema, input_specs, output_specs)` | 从 spec 推导 act/weight/state dtype；供 lowering 与 execution 交叉校验 |

算子契约与端口语义见 [operator_contract_design.md](../operators/operator_contract_design.md)；具体算子的端口/参数/数值契约以 `../operators/` 下对应文档为准，例如 [RoPE 算子契约](../operators/rope/RoPE算子契约.md)、[RMSNorm 算子契约](../operators/rmsnorm/RMSNorm算子契约.md)、[Attention](../operators/attention/README.md)。

当前接口形态是“输入 spec + 参数 → `InferenceResult`”，**没有** `ShapeInferenceContext`、`ProveShapeConstraints` 或 solver 参数。solver 与 `Infer*` 尚未在实现中耦合。

### 3.5 错误语义

- `EvaluateShapeConstraint` 的符号态重载不返回 `Status`，用三态表达“证明 / 证伪 / 未知”。
- concrete 重载对越界 port/dim 返回 `InvalidArgument`，热路径不会因格式错误的约束 abort；格式错误应在 plan 构建期被拒绝。
- `ValidateShapeConstraints` 只接受 concrete view；遇到 `kDeferred` 返回 `Internal`，表示运行期不应存在未解析约束。
- `ValidateConcreteShapeAgainstSpec` 的检查顺序为 dtype → rank → 逐维非负/静态值/符号绑定；第一个失败即返回 `InvalidArgument`。
- `ValidateTensorBindingPremises` 要求 spec 数量与 view 数量匹配（按索引直接访问）；无效 view 跳过。

## 4. 主流程 / 时序

### 4.1 构图期的形状推导

```text
ModelGraph::AddNode
  → 按 schema 顺序收集所有输入 value 的 TensorSpec
  → InferOperator(op_type, op_params, all_input_specs)
      → 参数校验（variant 类型、数值约束）
      → dtype/rank/static 维度校验
      → 输出 SymbolicShape / TensorSpec
      → 对无法静态证明的等式/广播/volume 生成 ShapeConstraint
  → 输出 spec 写入 NodeOutputDesc 派生的 GraphValue
  → runtime_checks 写入 GraphNode::runtime_checks
```

`Infer*` 中的“已证明”只基于局部信息：

- 同一 `ShapeSymbol`（`AreProvablyEqual`）→ 不生成等式约束；
- 两侧都是 static 且相等 → 不生成约束；
- 两侧都是 static 且冲突 → 直接返回 `InvalidArgument`；
- 其余情况 → 生成 `ShapeConstraint`，由执行期 concrete 求值兜底。

典型例子：`InferReshape` 构造 `VolumeEqualConstraint` 后立即调用符号态 `EvaluateShapeConstraint`：`kViolated` 直接失败，`kSatisfied` 不保留约束，`kDeferred` 才写入 `runtime_checks`（[reshape_op.cpp](../../src/operators/ops/reshape_op.cpp)）。`InferRmsNorm` 对 hidden/weight 相等使用 `AreProvablyEqual`；`InferAdd` 由 `InferBroadcastShape` 在局部消解 same-symbol / static-1，再只为歧义轴生成约束。

### 4.2 约束进入执行计划的路径

```text
GraphNode::runtime_checks
  → LowerModelGraph: LoweredStepSpec.runtime_checks（值拷贝）
  → ExecutionPlanBuilder: ExecutionStep.runtime_checks
      → RemapRuntimeChecks：输入 port 保持推理端口顺序；
        输出 port 从语义端口投影到 kernel output 端口
      → ExecutionPlan::Create: ValidateRuntimeCheckReferences
         （拒绝越界 port / dim 引用）
  → PrepareExecutionBindings / MaterializeExecutionBindings
```

`RemapRuntimeChecks` 在 [execution_plan_builder.cpp](../../src/execution/execution_plan_builder.cpp)。packed semantic weight 保留逻辑 shape 前提，但不作为 kernel TensorView；因此输入引用保持在推理端口坐标，输出引用才做 compact 投影。`ExecutionStep` 不保存 input/output spec；spec 由 `ExecutionPlan::values()` 持有，绑定准备阶段按 `step.inputs` / `step.outputs` 的 value id 取回。

### 4.3 绑定准备期的两道校验

执行阶段不在每次 `Execute()` 重新推导 shape；校验集中在 binding 准备的 cold path：

| 步骤 | 位置 | 检查 |
|---|---|---|
| `PlanExecutionBindings` | [execution_bindings.cpp](../../src/execution/execution_bindings.cpp) | 一个 plan-wide `SymbolValueMap`；校验全部 external read-only/writable view 的 dtype/rank/静态维度/符号 identity；校验 packed weight 的逻辑 dtype/shape；为每个 activation 维度解析 concrete 值：symbolic 未绑定或 unranked/Unknown 维度返回 `FailedPrecondition` |
| `MaterializeExecutionBindings` | 同上 | 对每个 step 的 kernel 输入/输出端口重新构造 specs 与 views；调用 `ValidateTensorBindingPremises`（每 step 新建 `SymbolValueMap`）；随后调用 `ValidateShapeConstraints(step.runtime_checks, ...)` |

`ValidateTensorBindingPremises` 的保障范围是：

- dtype 与 plan-time spec 一致；
- spec 的 rank 已知时，view rank 一致；
- static 维度一致；
- 同一步内多个端口引用同一 `ShapeSymbol` 时，concrete 值一致；
- `Unknown(-1)` 不建立任何前提；
- 无效 view 跳过（仅限该 per-step 校验路径；external binding 路径会先拒绝无效 view）。

`ValidateShapeConstraints` 的保障范围是：`runtime_checks` 中的每条约束（按构造均未被 plan-time 证明）在 concrete input/output view 上成立；违反时返回 `InvalidArgument`，并优先使用 `error_context` 作为错误消息。

### 4.4 执行期不再重复求值

`LayerRunner` 只消费已准备的 `StepTensorBinding` 与 params，不在每次 kernel 调用前重新执行 shape 推导。`LayerRunner` 的校验集中在 KV state 身份/alias（`ValidateKVCacheUpdateStateIdentity`、`ValidateAttentionStateIdentity`、`ValidateStateAliasesForStep`），不属于 shape_inference。

因此，`ExecutionPlan` / `PreparedExecutionBindings` 与具体 tensor 地址、shape、stride 是绑定关系：只要外部 binding 的 dtype/rank/静态维度/符号 identity 变化，就必须重新执行 `PrepareExecutionBindings`，不能复用旧 bindings。

### 4.5 “已证明约束”的保障边界

| 约束在 plan-time 的状态 | 进入执行计划的形式 | 执行期由谁保障 |
|---|---|---|
| 静态成立（`Infer*` 未生成约束） | 不存在 check | `TensorSpec` 前提校验：dtype/rank/static dim 不漂移 |
| 同一 `ShapeSymbol` identity（`AreProvablyEqual`） | 不存在 check | 同一步/外部绑定范围内的 symbol→concrete 一致性 |
| 静态冲突 | 构图失败 | 不进入计划 |
| 无法证明（`kDeferred`） | `ExecutionStep::runtime_checks` | `ValidateShapeConstraints` 在 concrete view 上求值 |
| `Unknown(-1)` | 相关约束通常为 `kDeferred`；identity 不传播 | 外部绑定只要求 concrete dim 非负；内部 activation 的 Unknown 维度无法解析 concrete 尺寸（`FailedPrecondition`） |
| rank 不可得 | rank 约束为 `kDeferred`；unranked spec 的前提校验只查 dtype | 运行期按 concrete rank 求值 |

明确的保障缺口与边界：

- **没有统一 plan-time proof 阶段**：当前不存在 `ProveShapeConstraints` 或等价的“solver 裁剪 runtime_checks”实现。所谓“已证明”仅指 `Infer*` 在局部不生成约束；不存在跨节点的统一证明与二次裁剪。
- **没有跨 step 的符号合一**：`ExecutionPlanBuilder` 不读取图边，不能证明 step 之间 output→input 的维度同一性。跨 step 一致性依赖：外部绑定在 `PlanExecutionBindings` 中按 plan-wide map 校验；内部 activation 由同一 plan spec 的符号解析值分配，属于构造性一致，不是独立校验。
- **per-step 校验的 map 不跨 step**：`ValidateTensorBindingPremises` 每次调用新建 `SymbolValueMap`；同一符号在不同 step 之间的 concrete 一致性不会被该函数单独检查。
- **不在 `Execute()` 热路径复验**：绑定准备完成后，kernel 调用前不再重新检查 spec 前提；计划复用必须配合重新准备 bindings。
- **packed weight 只校验逻辑形状**：`ValidatePackedWeightLogicalBindings` 使用 logical dtype/shape 代理视图；物理布局、stride、alignment 合法性由 backend/execution 的 packing 契约负责。
- **`error_context` 生命周期是外部契约**：约束复制品可以在 plan/step 中长期存在，但 `error_context` 指向的字面量必须保持有效；发出约束的算子必须使用 static-duration 字符串。

## 5. 关键取舍

| 选择 | 原因 | 被否方案 |
|---|---|---|
| 三态求值（satisfied / violated / deferred） | 必须区分“已证明”“已证伪”和“信息不足” | `bool`：无法表达 deferred，会把未知误判为失败或成功 |
| `ShapeConstraint` 为纯数据，求值由独立函数完成 | 约束可跨模块复制、进入 lowered/execution artifact；不引入虚函数与模块依赖 | 在 constraint 上放虚函数/堆分配：artifact 的复制与生命周期管理更复杂 |
| solver 只保存 proven fact，不保存普通 requirement | 把 requirement 写入 solver 会把“要求相等”误当“已经相等”，导致 runtime check 被错误裁剪 | 在 `Infer*` 中直接 `AddEqual(requirement)`：会产生假证明 |
| `Unknown` 表示“无 identity”，不参与绑定/传播 | 明确区分“未知但无同一性”和“动态但有同一性” | 把 unknown 当 fresh symbol：会凭空制造跨端口 identity |
| 符号 identity 使用全局递减整数 | 比较便宜，可作为 plan/execution artifact 的稳定键 | 字符串符号/进程内 interning：查找与复制成本更高；每图独立符号表：跨 artifact 无法比较 |
| `UnifyShapeSymbol` 在 symbolic 冲突时退化为 `Unknown`，不写 solver | 通用形状合一不应隐式产生 proof；proof 应由显式 solver 负责 | solver-aware `Unify`：会让所有调用点耦合 plan-time proof，并可能误裁剪 |
| `Join` 在分歧时创建 fresh symbol | 控制流分支合并应放宽约束，而不是记录等式 | 用等式合并分支：对分支不等值场景不可靠 |
| `Infer*` 只做局部证明，统一 proof 阶段未实现 | 当前没有图级 shape environment；局部证明足以避免大部分 obvious check | 在 plan builder 中引入全局 solver 并裁剪 checks：需要图边与统一的 solver 生命周期，当前未落地 |
| 校验集中在 binding 准备 cold path | 热路径无 shape 推导/分配；成本与 plan 生命周期对齐 | 每次 `Execute()` 重新校验：热路径成本高，且与“已准备 bindings 不可变”契约冲突 |
| external 绑定用 plan-wide `SymbolValueMap`，per-step 校验用独立 map | 外部符号必须在整个 plan 内一致；内部 activation 由分配值保证一致 | 所有校验共用一个跨 step map：internal activation 的符号解析与外部绑定混在一起，责任不清 |
| `error_context` 借用 `string_view` | 约束复制不产生分配，诊断文本仍可读 | `std::string`：每次复制分配；或丢弃 context：排障能力下降 |
| broadcast 歧义维度退化为 `Unknown` 并记录 deferred axis | 不凭空创建符号 identity；把不确定性显式留给运行期 | 创建 fresh symbol：会伪造图边不存在的同一性 |
| solver 使用 dense vector + union-by-rank | 查询路径简单、无哈希；符号值全局连续递减 | sparse map：常数因子更大；per-op solver：符号值跨度导致大索引 |
| 符号态求值不读 solver | 让通用求值器保持无状态、可并发 | 在 `EvaluateShapeConstraint` 内接 solver：求值器不再纯函数，调用方难以判断证明来源 |

## 6. 测试与验证入口

构建与运行：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter=ShapeConstraint*:SymbolConstraintSolver.*:BroadcastShapes.*:BroadcastInputStrides.*:InferBroadcastShape.*
```

主要套件：

| 测试文件 | 覆盖 |
|---|---|
| [test_shape_constraint.cpp](../../tests/unit/shape_inference/test_shape_constraint.cpp) | 六种约束的构造与定位、符号态 `DimEqual`/broadcast/volume/rank/positive、运行期求值、越界引用返回 `InvalidArgument`、`ValidateShapeConstraints` 传播 `error_context`、`error_context` 不参与相等/排序/去重 |
| [test_symbol_constraint_solver.cpp](../../tests/unit/shape_inference/test_symbol_constraint_solver.cpp) | 传递相等、静态绑定传播、静态冲突拒绝、`Unknown` 不传播、不同等价类但同静态绑定判等、无关符号 deferred |
| [test_broadcast.cpp](../../tests/unit/shape_inference/test_broadcast.cpp) | concrete broadcast、rank-zero、rank 扩展、零维、stride 扩展；符号 broadcast 的 same-symbol/static-1/冲突/unknown/不同符号、无 fresh symbol、deferred axis 顺序、确定性 |
| [test_shape_inference_includes.cpp](../../tests/unit/shape_inference/test_shape_inference_includes.cpp) | 公共类型可独立包含 |
| [test_operator_inference.cpp](../../tests/unit/operators/test_operator_inference.cpp) | 算子级校验分发、错误优先级、输入数量、`MakeCompactInputSpecs`、selector dtype 推导 |
| [operators/ops/](../../tests/unit/operators/ops/) | 各算子 `Infer*` 的参数、dtype/rank、输出 spec 与 deferred checks |
| [test_graph_op_builder.cpp](../../tests/unit/graph/test_graph_op_builder.cpp) | 构图 helper 对符号维度的接受/保留、失败不改图；间接覆盖 shape 约束生成 |
| [test_graph_lowering.cpp](../../tests/unit/compiler/lowering/test_graph_lowering.cpp) | `runtime_checks` 进入 lowered artifact、selector dtype 推导 |
| [test_execution_plan.cpp](../../tests/unit/execution/test_execution_plan.cpp) | `ExecuteRejectsBindingDTypeDrift` / `RankDrift` / `StaticDimDrift` / `InvalidViewsWhenCheckingPremises`、`RejectsExternalDtypeAndSymbolIdentityDrift` |
| [test_execution_binding_plan.cpp](../../tests/unit/execution/test_execution_binding_plan.cpp) | binding plan 的 external 校验、arena 规划与 materialize 失败保持可重试 |

验证重点：

- 每条约束条件至少覆盖 `kSatisfied`、`kViolated`、`kDeferred` 三种结果；
- 同一符号绑定到不同 concrete 值必须失败；
- `Unknown(-1)` 不产生 identity 约束；
- 运行期求值不得返回 `kDeferred`；
- 新增算子若产生约束，必须同时覆盖“静态可证不生成 check”“静态冲突构图失败”“symbolic 生成 check”三类路径。
