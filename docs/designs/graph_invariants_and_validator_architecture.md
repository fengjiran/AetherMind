# AetherMind 图不变量与统一校验器架构设计

> 状态：提案，待评审
> 范围：Phase 1（CPU 本地推理运行时，图编译与执行管线）
> 关联文档：[AGENTS.md](../../AGENTS.md)、[aethermind_prd.md](../products/aethermind_prd.md)、[model_graph_design.md](model_graph_design.md)、[graph_compilation_flow.md](graph_compilation_flow.md)、[kv_cache_design.md](kv_cache_design.md)

## 1. 文档定位

本文档是 AetherMind 图（ModelGraph）不变量与校验器架构的单一权威目录。它回答三个问题：

1. 图 IR 必须满足哪些不变量，这些不变量在哪个阶段成立。
2. 当前实现校验了什么、没有校验什么，证据落在哪些文件。
3. 若要走向"统一逻辑校验器"，架构应该是什么样，以及如何增量迁移。

文档刻意区分四类内容，避免把提案误读为现状：

- 已实现：当前代码已校验的条目，附精确路径。
- 部分覆盖：只覆盖子集或留有 TODO 的条目。
- 提案：本文建议的未来 API 与机制，全部明确标注，当前不存在。
- 非目标：明确不做的事，防止后续实现越界。

结论摘要：

- 图层的结构性与语义性不变量（§4、§5）已有扎实的现状覆盖，核心权威是 `ModelGraph::ValidateAndTopologicalOrder`。
- 校验目前是"沿流水线逐段分布"，没有统一目录、没有结构化 issue。本文不主张推翻现状，主张"逻辑集中、物理分布"：一份协议、一个编排入口，规则仍留在各模块本地。
- 三个明确缺口：activation 生命周期/缓冲复用校验未实现，全面临时内存规划校验器未实现，通用别名生命周期校验未实现（当前只有 KV 状态别名）。另有两条部分覆盖项：独立/外部 value 的 spec 合法性、常量 `inline_data` 字节数与 spec 匹配均无通用校验器（§5.1、§5.5）。
- 迁移是增量的，第一步只落地模块中立的 `ValidationIssue`/`ValidationReport` 协议，不改任何现有行为。

## 2. 背景与现状

校验逻辑目前沿流水线逐段分布，每段各自完成、各自报错：

- 图构造与全图校验：[graph.cpp](../../src/graph/graph.cpp:735) 的 `ModelGraph::ValidateAndTopologicalOrder`。
- 重写事务：[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:774) 的 `GraphRewriteSession::ValidateEdits` 与 [Commit](../../src/graph/optimization/graph_rewrite.cpp:1366)。
- Pass 管线：[graph_pass_manager.cpp](../../src/graph/optimization/graph_pass_manager.cpp:40) 的 `GraphPassManager::Run`。
- 图编译：[graph_lowering.cpp](../../src/compiler/graph_lowering.cpp) 的 `LowerModelGraph` / `ValidateLoweredGraph`，以及 execution-private [execution_plan_builder.cpp](../../src/execution/execution_plan_builder.cpp) 的 `ResolveStateAliasesForExecution`。
- 执行计划：[execution_plan_builder.cpp](../../src/execution/execution_plan_builder.cpp:291) 与 [execution_plan.cpp](../../src/execution/execution_plan.cpp:7) 的 `ExecutionPlan::Create`。
- 运行时：[layer_runner.cpp](../../src/execution/layer_runner.cpp:69) 的 `LayerRunner::ValidateStateAliasesForStep`，以及 [kv_cache_view.h](../../include/aethermind/execution/kv_cache_view.h:49) 的 `KVCacheView` 读写边界校验。

问题不在"校验少了"，而在"没有目录"。评审、修改、测试都缺少一张表回答：这条不变量归谁管、现在验没验、坏掉时谁先报。本文档补上这张表。

## 3. 基础约束与术语

### 3.1 ModelGraph 独立性

ModelGraph 是后端、设备、内存无关的纯图 IR。依据 [AGENTS.md](../../AGENTS.md) 模块所有权表：

- graph 模块不得出现 Backend、Kernel、Workspace、DeviceType 等执行层细节；`StateAlias` 仅以 must-alias 约束表达，物理存储归属执行层。
- 跨模块依赖规则：operators 到 shape_inference；graph 到 operators 与 shape_inference；execution 公共头只依赖 operators 与 shape_inference（唯一 adapter 边是 `execution_plan_builder.cpp` 内部 include graph 编译头）；model 到 graph、operators、execution；backend 到 operators。

校验器架构必须维持这些边界：graph 模块的校验器只推理 IR 坐标，execution 模块的校验器只推理 step/port 坐标，两者之间唯一的跨模块工件是模块中立的校验报告协议（§8.2）。

### 3.2 术语

- `GraphValueId`、`GraphNodeId`：图坐标。
- `GraphValuePayload`：`ActivationValue`、`WeightValue`、`ConstantValue`、`StateValue`、`ModelInputValue`。
- producer：产生某个 value 的节点，round-trip 语义见 §4.3。
- must-alias：输入与输出共享物理存储的语义契约，非拷贝语义。
- `ShapeConstraint`：随节点存储的运行时形状约束。
- `LoweredGraph`、`ExecutionStep`、`StateAliasPlan`、`KVCacheView`：编译与执行层的坐标与视图。

### 3.3 不变量分类

- 结构性不变量（§4）：只凭图结构即可静态判定，不依赖算子语义。
- 语义性不变量（§5）：依赖 `OperatorSchema` 与 `InferOperator` 重分析。
- 通用后置不变量（§6.1）：任何 pass 提交后、任何阶段边界上必须保持的集合。
- 阶段专属条件（§6.2）：只在特定阶段有意义，不是图不变量，但该阶段需要自己的本地校验。

## 4. 结构性不变量

### 4.1 生产者不变量（修正表述）

- activation 与 produced-state（由节点产出的 `StateValue`）：恰好一个生产者。
- model input、weight、constant、initial state（无 producer 的 `StateValue`）：零生产者。

现状实现于 [ValidateValueSelfConsistency](../../src/graph/graph.cpp:292)：

- `ActivationValue` 必须有 producer，producer 必须是合法节点，且 producer 节点必须把该 value 列入 outputs（[graph.cpp](../../src/graph/graph.cpp:299)）。
- `StateValue` 的 producer 可选；有 producer 时做同样的合法性与回环校验，无 producer 即为初始状态（[graph.cpp](../../src/graph/graph.cpp:312)）。
- `ConstantValue`、`WeightValue`、`ModelInputValue` 禁止携带 producer（[graph.cpp](../../src/graph/graph.cpp:329) 起）。

"恰好一个"由 producer 单值字段与双向回环校验共同构成：producer 唯一由构造保证，回环保证生产者与消费者视图一致。

### 4.2 无悬垂引用

- 图输入/输出引用的 value id 必须有效：[graph.cpp](../../src/graph/graph.cpp:737)。
- 节点输入引用的 value id 必须有效：[graph.cpp](../../src/graph/graph.cpp:801)。
- 节点输出引用的 value id 必须有效：[graph.cpp](../../src/graph/graph.cpp:847)。
- 拓扑序计算中 producer 必须指向合法节点：[graph.cpp](../../src/graph/graph.cpp:941)。
- 运行期对应物：`KVCacheView` 的指针三件套与 generation 存活检查（[kv_cache_view.h](../../include/aethermind/execution/kv_cache_view.h:95)）。

### 4.3 生产者-消费者一致性

producer 字段与节点输出列表必须双向一致：producer 指向的节点必须把该 value 列为输出（[graph.cpp](../../src/graph/graph.cpp:304)）；节点输出声明的 producer 必须是该节点自身（[graph.cpp](../../src/graph/graph.cpp:856)）。这一对检查覆盖"伪造 producer"与"声明了输出却没人认领"两类损坏。

### 4.4 DAG 与拓扑序

`TopologicalOrder` 用 Kahn 算法，只统计 activation 边与 produced-state 边，weight 与 model input 边不构成执行依赖，被显式排除（[graph.cpp](../../src/graph/graph.cpp:924)）。自环在入边阶段直接拒绝（[graph.cpp](../../src/graph/graph.cpp:946)），剩余环由"产出节点数不等于节点总数"判出（[graph.cpp](../../src/graph/graph.cpp:975)）。

### 4.5 支配关系由拓扑蕴含

当前 IR 无控制流：没有分支、循环、merge 点，节点间唯一的顺序约束来自数据依赖。因此任何消费者的生产者必然先执行，支配关系退化为平凡关系且由拓扑序蕴含，不需要单独计算。这是设计决策而非缺陷。未来若引入控制流（Phase 2 范围外），必须重新评估本不变量，届时图校验与执行序校验都要改。

### 4.6 Schema 端口 ABI

- 端口名与顺序是语义 ABI（[AGENTS.md](../../AGENTS.md) operators 模块）。节点输入/输出数量必须与 `OperatorSchema` 端口数量一致：[graph.cpp](../../src/graph/graph.cpp:788)。
- 每个端口的 payload kind 必须匹配：输入侧 [graph.cpp](../../src/graph/graph.cpp:809)，输出侧 [graph.cpp](../../src/graph/graph.cpp:862)。
- 注册算子必须使用 typed `op_params`，`attrs` 被拒绝：[graph.cpp](../../src/graph/graph.cpp:782)。

### 4.7 图 I/O 契约

- 图输入必须引用有效 value id：[graph.cpp](../../src/graph/graph.cpp:737)。
- 图输出必须是 activation 或 constant，且 activation 输出必须有 producer：[graph.cpp](../../src/graph/graph.cpp:743)。
- 节点输出不得复用输入 value id：[graph.cpp](../../src/graph/graph.cpp:893)。
- 输出声明走 `MarkOutput` 等显式 API，接口层保证输出集合的增删可见。

### 4.8 重写替换闭包与无环

`GraphRewriteSession::ValidateEdits`（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:774)）覆盖：

- 替换目标必须是非虚拟 value（`CheckNonVirtualValueId`）。
- 替换不得改变 dtype，shape 必须可 `Unify`（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:791)）。
- 替换节点与替换目标结构校验（`ValidateReplacementNode`、`ValidateReplacementTargets`）。
- 虚拟值不得跨 rewrite 边界（`ValidateVirtualValues`）。
- 每个 active rewrite 做 `InferOperator` 语义重放（`ValidateReplacementSemantics`）。
- 替换输入的发射序可用性（`ValidateReplacementInputAvailability`），保证 Commit 时不存在不可映射输入。

环的预防在注册点完成：`ReplaceValue` 在写入前沿 `new_value` 的替换链前向检查，若会闭合环则直接拒绝该编辑（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:314)）。`GetResolvedValue` 的路径压缩与 `value_replacements_.size()` 深度上限因此只是纵深防御（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:363)），正常路径下环不可能出现，也不存在"静默解析到环上某值"的行为。

Commit 的顺序是：`ValidateEdits`、计算保留集（含 DCE 语义）、按拓扑序发射、`committed.Validate()` 全量复验（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:1366)）。提交产物的正确性由"提交后全量校验"兜底，不依赖校验器之外的信任。

### 4.9 状态副作用

状态变更通过 must-alias 表达，而不是拷贝语义。`KVCacheUpdate` 的输入/输出共享物理存储，绑定规则由 [ValidateStateBindingsForNode](../../src/graph/graph.cpp:175) 在构造与复验两处统一执行：slot 匹配（k/v）、layer 与节点一致、输入输出同 family、K/V 同 collection。

副作用在编译期被物化：`AddLoweringTimeStateAliases` 在 lowering 期为每个声明 `state_alias_ports` 的算子生成别名（[graph_lowering.cpp](../../src/compiler/graph_lowering.cpp)）。`ValidateLoweredGraph` 验证 schema/state/value/conflict invariant；execution 的 `ResolveStateAliasesForExecution` 将它们映射到 step/port 坐标并按完整坐标确定性排序。

### 4.10 确定性语义

校验与优化必须是确定性的：同一输入图、同一管线、同一配置，产出相同校验结果与相同优化结果。现状支撑：Kahn 序稳定、别名按 step_index 排序、pass 无状态（[graph_pass_manager.h](../../include/aethermind/graph/optimization/graph_pass_manager.h:54)）、快照不可变（[model_graph_design.md](model_graph_design.md) §10.1 Immutable Snapshot Contract）。任何新校验器不得依赖未定义的遍历顺序。

## 5. 语义不变量

### 5.1 shape/dtype 一致性

- 节点输出 spec 必须与 `InferOperator` 派生 spec 严格相等：[graph.cpp](../../src/graph/graph.cpp:868)。
- 重写替换不得改变 dtype，shape 必须可统一：[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:791)。
- 部分覆盖：节点级输出 spec 的重推断是严格的，但不存在单一通用校验器证明每个独立/外部 value（如 model input、weight、constant 的 spec）的 dtype、rank、维度已初始化且合法。该检查目前分散在各 payload 绑定校验与 `InferOperator` 参数验证中，未以"全 value 遍历"的形式收敛。

### 5.2 InferOperator 重分析（trust-but-verify）

`ValidateAndTopologicalOrder` 对每个节点重放 `InferOperator`，派生 outputs 与 runtime_checks 必须与存储值严格相等，防止 stale 或伪造元数据扩散（[graph.cpp](../../src/graph/graph.cpp:833)）。同一原则作用于执行计划构建的未信任路径：调用方自带的 `ExecutionPlanNodeSpec` 必须通过 `ValidateCallerMetadata` 的重推断与严格相等检查（[execution_plan_builder.cpp](../../src/execution/execution_plan_builder.cpp:60)）。信任路径（来自 `LoweredGraph`）不重推断，元数据在构建期已经过单一语义权威。

### 5.3 运行时 ShapeConstraints

- 图层的等价比较覆盖 condition 与 error_context 两者（[graph.cpp](../../src/graph/graph.cpp:279)），避免只比条件不比诊断。
- 执行期按需求值：`LayerRunner::RunStep` 在 `runtime_checks` 非空时调用 `ValidateShapeConstraints`（[layer_runner.cpp](../../src/execution/layer_runner.cpp:56)）。

### 5.4 权重与状态绑定

- `WeightBinding` 自洽：slot 与 `semantic_role` 匹配，layer 需求与 `decoder_layer_index` 互斥匹配（[graph.cpp](../../src/graph/graph.cpp:145)）。
- 期望权重槽位按 op 表驱动：embedding 表、RMSNorm scale、Linear kernel（[graph.cpp](../../src/graph/graph.cpp:106)）。
- 状态绑定规则见 §4.9，覆盖 `kKVCacheUpdate` 与 `kAttention` 两类节点。

### 5.5 常量完整性

常量不得有 producer（§4.1）。lowering 后 `ConstantValue` 的绑定保留在 `LoweredGraph::values()` 的 payload 中，后端可按需经 value id 延迟解析内联数据或命名外部常量（[graph_lowering.cpp](../../src/compiler/graph_lowering.cpp)）。常量折叠产生的会话常量经 Commit 进入不可变快照，提交后同样受全量复验。

部分覆盖：不存在通用校验器核对每个 `ConstantValue` 的 `inline_data` 字节数与 `TensorSpec` 声称的元素数/dtype 是否匹配；内联常量的字节数与 spec 一致性依赖生产路径（常量折叠、`AddConstant`）各自保证。

### 5.6 别名语义

must-alias 是语义契约：别名输入与输出共享物理存储，不是拷贝。图层的表达是绑定约束（§4.9），lowering 物化为别名（§4.9），运行期以 `StateAliasPlan` 消费（[state_alias_plan.h](../../include/aethermind/execution/state_alias_plan.h:25)）。

现状边界：运行期只处理 KV 状态别名，`LayerRunner::ValidateStateAliasesForStep` 要求存在有效 `KVCacheView`（[layer_runner.cpp](../../src/execution/layer_runner.cpp:69)）。通用别名（非 KV 状态、activation 端口别名）的指针比较校验留有 TODO，未实现（[layer_runner.cpp](../../src/execution/layer_runner.cpp:87)）。本文不声称通用别名生命周期已被全面校验。

### 5.7 KV cache 池隔离

`KVCacheManager` 独占静态 KV 存储：按层数、KV 头数、最大 token 数、head_dim、dtype 与对齐初始化（[kv_cache_manager.h](../../include/aethermind/execution/kv_cache_manager.h:10)），会话视图带 generation 与槽位存活语义。布局合法性由 `KVCacheLayout::Validate` 校验（[kv_cache_view.h](../../include/aethermind/execution/kv_cache_view.h:24)），读写越界由 `KVCacheView::ValidateWrite`/`ValidateRead` 拦截（[kv_cache_view.h](../../include/aethermind/execution/kv_cache_view.h:65)）。

KV 池隔离是构造性的：`KVCacheManager` 独占静态存储，所有权与架构层面保证 KV 存储独立于工作区 arena 与临时池，静态分配模型见 [kv_cache_design.md](kv_cache_design.md) §5，`KVCacheView` 的边界检查在视图层兜底。但不存在显式的内存计划跨池重叠校验器，KV 池与临时池的重叠规划校验未实现（§7.2 行 17），该维度是部分覆盖而非校验器强制。

## 6. 通用后置不变量与阶段专属条件

### 6.1 通用后置不变量

以下集合在任何 pass 提交后、任何阶段边界上都必须保持，构成"提交后复验"的最小契约：

- §4.1 到 §4.4：生产者不变量、无悬垂引用、生产者-消费者一致、DAG 与拓扑序。
- §4.6 到 §4.8：schema 端口 ABI、图 I/O 契约、替换闭包与无环。
- §5.1 到 §5.4：shape/dtype 一致、`InferOperator` 重分析一致、runtime ShapeConstraints 等价、绑定自洽。

现状的落实方式是"提交后全量 `Validate()`"：`GraphRewriteSession::Commit` 末尾（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:1435)）与 `GraphPassManager` 的每个检查点及最终提交（§7.1 行 5）都执行它。

### 6.2 阶段专属条件（不是图不变量）

| 条件 | 所属阶段 | 现状 | 说明 |
| --- | --- | --- | --- |
| DCE 可达性 | DCE pass | [ComputeRetainedNodes](../../src/graph/optimization/graph_rewrite.cpp:906) 内部计算 | 图不变量不要求"无死节点"，裁剪是 pass 行为 |
| 静态形状解析 | 模型构建/规划 | `TensorSpec` + `ShapeConstraint` | 动态维度解析是阶段行为，非图校验职责 |
| kernel 解析/预打包 | 执行计划构建 | [ResolveKernelForNode](../../src/execution/execution_plan_builder.cpp:273)、`WeightPrepackPlanner` | backend 域，graph 层不得涉及 |
| workspace 对齐/偏移 | 执行计划构建 | [PlanWorkspaceRequirements](../../include/aethermind/runtime/workspace.h:178) | 顺序规划：对齐合法性 + 溢出检查，无生命周期复用 |
| 内存计划 | 规划阶段 | 无全面校验器 | activation 池、临时池、KV 池的分配与复用规划，见 §7.2 行 16、17 |

区分意义：这些条件不能作为 graph 校验的通用断言，否则会把 pass 内部策略误当图契约；但一旦进入对应阶段，各自必须有本地校验，不能因为"不是图不变量"就放任不验。

## 7. 当前实现覆盖矩阵

### 7.1 校验点清单

| 校验点 | 路径 | 状态 | 说明 |
| --- | --- | --- | --- |
| `ModelGraph::ValidateAndTopologicalOrder` | [graph.cpp](../../src/graph/graph.cpp:735) | 已实现 | 三段式：value 自洽、逐节点 schema 重放与 stale 检测、Kahn 拓扑 |
| `ModelGraph::Validate` | [graph.cpp](../../src/graph/graph.cpp:699) | 已实现 | 包装 `ValidateAndTopologicalOrder`，丢弃拓扑序 |
| `GraphRewriteSession::ValidateEdits` | [graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:774) | 已实现 | 替换链、dtype/shape 兼容、虚拟值、语义重放、发射序可用性 |
| `GraphRewriteSession::Commit` | [graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:1366) | 已实现 | `ValidateEdits` 后发射，末尾 `committed.Validate()` 复验 |
| `GraphPassManager::Run` | [graph_pass_manager.cpp](../../src/graph/optimization/graph_pass_manager.cpp:40) | 已实现 | 输入 `Validate()`，检查点与最终提交各一次完整 Commit；`checkpoint_every` 默认 0（[graph_pass_manager.h](../../include/aethermind/graph/optimization/graph_pass_manager.h:38)） |
| `LowerModelGraph` / `ValidateLoweredGraph` | [compiler/graph_lowering.cpp](../../src/compiler/graph_lowering.cpp) | 已实现 | 前置 `ValidateAndTopologicalOrder`，finalization 验证拓扑步骤、binding/spec、selector、model I/O 与声明式 state aliases |
| `ResolveStateAliasesForExecution` | [execution/execution_plan_builder.cpp](../../src/execution/execution_plan_builder.cpp) | 已实现 | execution trust boundary 重查 artifact 后，将 aliases 映射为 runtime step/port 并按完整坐标排序 |
| `ExecutionPlanBuilder::Build` | [execution_plan_builder.cpp](../../src/execution/execution_plan_builder.cpp:291) | 已实现 | 信任路径不重推断；未信任路径 `ValidateCallerMetadata` 严格相等；拒绝 monostate `op_params` |
| `ExecutionPlan::Create` | [execution_plan.cpp](../../src/execution/execution_plan.cpp:7) | 已实现 | 每步：op 非空、resolved kernel fn 非空、workspace 对齐为 2 的幂 |
| `PlanWorkspaceRequirements` | [workspace.h](../../include/aethermind/runtime/workspace.h:178) | 已实现 | 对齐校验、偏移溢出检查；顺序分配，不做跨步骤生命周期复用 |
| `LayerRunner::ValidateStateAliasesForStep` | [layer_runner.cpp](../../src/execution/layer_runner.cpp:69) | 部分覆盖 | 仅要求 KV 别名存在 `KVCacheView`；通用别名指针比较留 TODO（[layer_runner.cpp](../../src/execution/layer_runner.cpp:87)） |
| `KVCacheManager` / `KVCacheView` | [kv_cache_manager.h](../../include/aethermind/execution/kv_cache_manager.h:10)、[kv_cache_view.h](../../include/aethermind/execution/kv_cache_view.h:49) | 已实现 | 静态存储独占、布局校验、读写边界、会话 generation；池隔离为构造性，见 §5.7 |
| 独立/外部 value spec 合法性（dtype/rank/维度初始化） | 无独立入口 | 部分覆盖 | 节点输出经重推断严格校验；外部 value 依赖各绑定校验，无全 value 遍历的通用校验器（§5.1） |
| 常量 `inline_data` 字节数与 spec 匹配 | 无独立入口 | 部分覆盖 | 无通用校验器，依赖生产路径各自保证（§5.5） |
| activation 生命周期/缓冲复用校验 | 无 | 未实现 | 任何 activation 池复用的引入必须先落地本校验器（§12 风险 3） |
| 全面临时内存规划校验器 | 无 | 未实现 | 现状只有顺序偏移规划；跨步骤复用、池间重叠校验缺失 |

### 7.2 不变量 × 阶段矩阵

阶段列：A 图构造与全图校验；B 重写提交；C PassManager 节律；D Lowering；E 执行计划构建；F 运行时。

| 不变量 | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| 1. 生产者不变量 | 已实现 | 提交后复验 | 输入+检查点+最终 | 前置 | 不适用 | 不适用 |
| 2. 无悬垂引用 | 已实现 | 已实现 | 同上 | 前置 | 每步非空检查 | View 有效性 |
| 3. 生产者-消费者一致 | 已实现 | 提交后复验 | 同上 | 前置 | 不适用 | 不适用 |
| 4. DAG 与拓扑序 | 已实现 | 提交后复验 | 同上 | 前置，步骤序 | 不适用 | 顺序执行 |
| 5. Schema 端口 ABI | 已实现 | 已实现 | 同上 | 前置 | 不适用 | 不适用 |
| 6. 图 I/O 契约 | 已实现 | 提交后复验 | 同上 | 透传 | 不适用 | 不适用 |
| 7. 替换闭包与无环 | 不适用 | 已实现 | 检查点提交 | 不适用 | 不适用 | 不适用 |
| 8. shape/dtype | 部分覆盖（节点输出重推断严格；独立/外部 value 无通用合法性校验） | dtype 相等 + Unify | 同上 | 透传 | 未信任路径重推断 | 形状约束求值 |
| 9. InferOperator 重分析 | 已实现 | 逐 rewrite 重放 | 同上 | 前置 | 仅未信任路径 | 不适用 |
| 10. 运行时 ShapeConstraints | 等价比较 | 重放 | 同上 | 透传 | 未信任路径相等 | 运行时求值 |
| 11. 权重绑定 | 已实现 | 提交后复验 | 同上 | 透传 | 打包解析 | 不适用 |
| 12. 状态绑定 | 已实现 | 提交后复验 | 同上 | KV 别名生成 | 别名映射 | KVCacheView 存在性 |
| 13. 常量完整性 | 部分覆盖（无 producer；`inline_data` 字节数与 spec 匹配无通用校验） | 提交后复验 | 同上 | 常量绑定记录 | 不适用 | 不适用 |
| 14. 别名语义（must-alias） | 约束表达 | 提交后复验 | 同上 | 别名生成 | StateAliasPlan | 部分，仅 KV |
| 15. KV 池隔离 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 构造性隔离/部分覆盖 |
| 16. activation 生命周期/复用 | 未实现 | 未实现 | 未实现 | 未实现 | 未实现 | 未实现 |
| 17. 临时内存规划校验 | 不适用 | 不适用 | 不适用 | 不适用 | 部分（对齐/溢出） | Bind 检查 |

读法：行 1 到 15 的 A/B/C 列以"提交后全量复验"为兜底，行 8、13 标注的语义缺口以 A 列详述为准；行 15 是构造性隔离而非校验器强制；行 16、17 是明确缺口。任何声称"已全面校验"的说法都以本表为准。阶段 B（重写提交）与 C（PassManager 节律）的 issue 在 §8.2 协议中统一归入 `kPassCommit`：单次 Commit 对应 B 列，PassManager 检查点触发的 Commit 对应 C 列，二者复用同一 Commit 校验路径。

## 8. 统一逻辑校验器设计（提案）

本节全部内容均为提案，当前代码中不存在这些 API。评审通过前不得按"已存在接口"引用。

### 8.1 设计原则：逻辑集中，物理分布

- 逻辑集中：一份不变量目录（本文档）、一份模块中立报告协议、一个编排入口（收集、排序、去重、标注阶段）。
- 物理分布：每条规则的实现留在所属模块本地，由该模块的既有权威函数承载。graph 模块的规则不能搬进 execution，execution 的规则不能搬进 graph。
- 明确反对 god-object：单一 `Validator` 类持有全部规则，会强迫模块互相可见，直接破坏 §3.1 的依赖方向。

编排者只做聚合，不做判定。判定永远发生在模块本地，这就是"统一"与"集中"的边界。

### 8.2 模块中立协议（提案）

协议头建议放 `include/aethermind/base/validation_report.h`（若评审倾向控制 base 规模，可改放 graph 模块，但执行层不得反向依赖 graph，故 base 是唯一两全位置）。它只依赖基础类型，不引用任何模块类型：

```cpp
// 提案草案，当前不存在
namespace aethermind::validation {

enum class Severity { kError, kWarning, kInfo };

enum class Phase {
    kGraphBuild,   // 图构造与全图校验
    kPassCommit,   // 重写提交（覆盖 §7.2 阶段 B 与 C）
    kLowering,     // 图编译 lowering
    kPlanBuild,    // 执行计划构建
    kRuntime,      // 运行时绑定与执行
};

// 模块中立的位置：全部用原始索引，语义由 phase 决定
struct GraphLocation {
    uint32_t node_index = 0;
    std::optional<uint32_t> value_index;
};
struct StepLocation {
    size_t step_index = 0;
    std::optional<uint32_t> port_index;
};
using Location = std::variant<std::monostate, GraphLocation, StepLocation>;

struct ValidationIssue {
    std::string_view code;    // 稳定标识，如 "graph.producer.missing"
    Severity severity = Severity::kError;
    Phase phase = Phase::kGraphBuild;
    Location location{};
    std::string message;      // 人类可读诊断，保留现有节点上下文格式
};

struct ValidationReport {
    std::vector<ValidationIssue> issues{};
    Phase phase = Phase::kGraphBuild;
    AM_NODISCARD bool ok() const noexcept { return issues.empty(); }
};

}  // namespace aethermind::validation
```

### 8.3 模块本地 API（提案）

每个模块暴露一个薄封装，内部调用既有权威函数，产出报告而非直接抛错：

- graph 模块：`GraphValidator::Validate(const ModelGraph&)`，包装 `ValidateAndTopologicalOrder`，逐条不变量映射为 issue code；同时保留 `ValidateAndOrder` 返回拓扑序的语义。
- optimization 模块：`RewriteEditValidator`，包装 `ValidateEdits` 与提交后复验。
- compilation 模块：`LoweringValidator`，包装 lowering 前置校验与 `ResolveStateAliases` 的未命中诊断。
- execution 模块：`ExecutionPlanValidator` 与 `RuntimeBindingValidator`。后者是 `LayerRunner::ValidateStateAliasesForStep` 的扩展点，通用别名场景在此补 `StepTensorBinding` 指针比较。
- 编排：`ValidationReportCollector::Add(module, report)` 与 `Merge()`，只做排序、去重、阶段标注，不做判定。

这些封装与现有 `Validate*` 函数双轨并存，见 §10 迁移顺序。

### 8.4 依赖与头文件约束

- graph 公共头不得依赖 execution、backend、kernel、workspace、device 或 memory planner 类型，本条对校验器提案同样成立。
- 协议头只依赖 `base`，`Location` 用原始索引而非模块类型，保证任意两个模块都能互相读懂对方的报告。
- execution 模块的校验器可以消费 `ValidationReport`，但不得引入 graph 头；graph 模块的校验器同理，不得引入 execution 头。

## 9. 校验节律（cadence）

### 9.1 三档节律

| 档位 | 配置 | 校验范围 | 定位目标 |
| --- | --- | --- | --- |
| release | `checkpoint_every=0`（默认） | 输入图 `Validate()` + 最终 Commit 复验 + lowering/plan 构建校验 | 吞吐优先，一次最终判定 |
| developer | `SetCheckpointEvery(1)` | 每 pass 后完整提交校验 | 快速定位失败 pass，复现成本最低 |
| CI | `SetCheckpointEvery(1)` + 全量负面测试 + sanitizer | 每 pass 后完整校验 + 破坏性用例 | 门禁，覆盖 §7.2 矩阵 |

### 9.2 checkpoint_every=1 的现状含义

当前校验只挂在 `Validate()` 与 `Commit()` 上，没有 per-pass 中间检查点钩子。因此"每 pass 后完整提交校验"的唯一现成机制就是 `checkpoint_every=1`：每个 pass 边界触发一次完整 `Commit`（`ValidateEdits` + `committed.Validate()` + 快照拷贝），[graph_pass_manager.cpp](../../src/graph/optimization/graph_pass_manager.cpp:74)。已有测试证明 `checkpoint_every=1` 与无检查点的优化结果一致（[test_semantic_optimization_pipeline.cpp](../../tests/unit/compiler/optimization/test_semantic_optimization_pipeline.cpp)）。

成本是每 pass 一次深拷贝加全图重分析，量级 O(passes × |graph|)。developer/CI 是否默认启用应以编译流水线基准为依据；release 不应在缺少收益数据时默认承担该成本。

提案（§8 配套）：未来在 `PassContext` 增加轻量 per-pass 校验钩子（只跑 §6.1 通用后置不变量，不做快照拷贝），把 developer/CI 从对 `checkpoint_every=1` 的依赖中解放出来。在此之前，`checkpoint_every=1` 就是 developer/CI 的标准配置。

## 10. 增量迁移步骤

| 步骤 | 内容 | 验收 |
| --- | --- | --- |
| M1 | 评审冻结本文档：不变量目录、分类、矩阵、缺口清单 | 评审意见闭环，缺口清单无异议 |
| M2 | 落地 §8.2 协议头（纯新增，无行为变化） | 编译通过，协议单测通过 |
| M3 | graph 模块 `GraphValidator` 薄封装，内部仍走 `ValidateAndTopologicalOrder` | 行为等价测试：新老入口对同一批图产出相同判定 |
| M4 | optimization/compilation/execution 本地校验器 issue 化 + `ValidationReportCollector` | 诊断一致性测试：结构化 issue 与现有 Status 字符串语义等价 |
| M5 | 补齐两个缺口：activation 生命周期/缓冲复用校验器（IR 级 liveness 分析，不绑定 allocator）；临时内存规划校验器（跨步骤重叠与生命周期） | 每条缺口对应负面测试（构造必然失败的规划并断言拒绝） |
| M5.5 | 通用别名生命周期校验：`RuntimeBindingValidator` 补 `StepTensorBinding` 指针比较，关闭 [layer_runner.cpp:87](../../src/execution/layer_runner.cpp:87) TODO | 通用别名场景的负面测试（构造非 KV 状态别名指针不一致并断言拒绝） |
| M5.6 | 独立/外部 value spec 合法性校验器（全 value 遍历，校验 dtype/rank/维度初始化）；常量 `inline_data` 字节数与 `TensorSpec` 匹配校验器 | 两条部分覆盖项各对应负面测试（构造非法 spec/字节数不匹配并断言拒绝） |
| M6 | 节律接入：developer/CI 默认 `checkpoint_every=1`；release 保持轻量 | CI 全绿，release 路径无新增校验开销 |
| M7 | 落地 `PassContext` 轻量 per-pass 校验钩子（只跑 §6.1 通用后置不变量，不做快照拷贝）；developer/CI 从 `checkpoint_every=1` 迁移至钩子 | 钩子路径与 `checkpoint_every=1` 诊断等价，且无快照拷贝开销；`checkpoint_every=1` 保留为兜底 |

每步独立可合并、可回退，不要求一次性完成。M2 到 M4 期间新旧 API 双轨并存，行为等价测试是防漂移的唯一手段。

## 11. 测试与验证策略

- 逐不变量负面测试：§4、§5 每条不变量至少一个"破坏后必须失败"用例。现有套件已覆盖一部分（[test_graph_rewrite.cpp](../../tests/unit/graph/optimization/test_graph_rewrite.cpp)、[test_graph_pass_manager.cpp](../../tests/unit/graph/optimization/test_graph_pass_manager.cpp)、[test_commit_pruning.cpp](../../tests/unit/graph/optimization/test_commit_pruning.cpp)、[test_graph_lowering.cpp](../../tests/unit/compiler/lowering/test_graph_lowering.cpp)），M3 之后按 §7.2 矩阵逐格核对补缺。
- 矩阵驱动：§7.2 的每个"已实现"格至少一个正面用例，每个"未实现"格在 M5 前挂 TODO 测试，M5 后转为正式负面测试。
- 节律回归：`checkpoint_every=1` 的等价性测试保持常绿（[test_semantic_optimization_pipeline.cpp](../../tests/unit/compiler/optimization/test_semantic_optimization_pipeline.cpp)），作为"每 pass 后完整校验"不破坏结果的守门员。
- 属性测试（提案）：随机合法图生成器 + 注入式破坏器，合法图必须通过全量复验，注入破坏必须被拒绝。这是 §7.2 矩阵的机械补充。
- sanitizer：TSAN 下跑完整管线（[AGENTS.md](../../AGENTS.md) §4），重点覆盖 LayerRunner 别名路径与 KV 视图生命周期。
- 新增规则必须带测试进入，禁止"规则先行、测试后补"。

## 12. 风险

1. 双轨漂移：M2 到 M4 期间新旧 API 并存，行为等价测试若缺失，结构化 issue 会与 Status 诊断逐渐分叉。缓解：每步迁移都锁定等价测试。
2. 校验成本：`checkpoint_every=1` 的重复重分析在模型规模增长后不可忽视。缓解：§9.2 的轻量 per-pass 钩子提案；在落地前，节律只影响 developer/CI，不影响 release。
3. 未实现缺口：activation 生命周期/缓冲复用校验缺失，现状不构成现实漏洞，因为执行层还没有 activation 池复用；但任何引入复用或池重叠的改动，必须先落地对应校验器再合入。临时内存规划同理。
4. 替换环检测的覆盖范围：环在 `ReplaceValue` 注册点被显式拒绝（[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp:314)），`GetResolvedValue` 的深度上限只是纵深防御。残余风险是未来新增替换写入入口时若绕过 `ReplaceValue` 会失去环检测；规则是任何修改 `value_replacements_` 的路径都必须复用该检测。
5. 诊断一致性：issue 化迁移必须保留 `FormatNodeContext` 式的节点上下文（[graph.cpp](../../src/graph/graph.cpp:283)），否则排障能力倒退。
6. KV 池与临时池隔离：现状靠所有权分离成立，无显式规划校验；统一内存规划落地时，必须先补池间重叠校验（§7.2 行 17），否则隔离不变量会从"构造性成立"退化为"约定性成立"。

## 13. 非目标

- 不改动任何现有代码或文档，本文档是纯提案，落地必须走评审后的增量步骤（§10）。
- 不引入 Phase 2 分布式执行或 paged KV 的校验需求；本目录只覆盖 Phase 1 静态 KV 模型。
- 不声称通用别名生命周期、activation 池重叠、KV 池与临时池规划已被全面校验。§7.2 行 16、17 与 §5.6、§5.7 是权威表述。
- 不采用单一 god-object `Validator`。
- 不新增 graph 公共头对 execution、backend、kernel、workspace、device、memory planner 类型的依赖。

## 14. 参考文档

- [AGENTS.md](../../AGENTS.md)：模块所有权、跨模块依赖规则、构建测试规范。
- [aethermind_prd.md](../products/aethermind_prd.md)：Phase 1 产品范围与验收标准。
- [model_graph_design.md](model_graph_design.md)：ModelGraph 设计、§10 快照与 checkpoint 契约、§13 Validation 规则、§16 pass 管线、§17.1 状态缓冲别名。
- [graph_compilation_flow.md](graph_compilation_flow.md)：按当前代码事实梳理的图编译 lowering 与计划构建流程。评审版（图编译功能完整实现流程，含已实现/待生产化边界）见 [graph_compilation_flow.md](../reviews/graph_compilation_flow.md)。
- [kv_cache_design.md](kv_cache_design.md)：KV cache 静态分配模型、Manager/View 职责与布局契约。
