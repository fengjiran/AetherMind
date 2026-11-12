# graph：语义图 IR 与 semantic passes

`graph` 提供 backend-independent 的 `ModelGraph` 语义 DAG、图重写框架与 semantic passes。它只回答“模型在语义上由哪些计算组成、哪些 value 在节点之间流动”，不回答某次部署使用哪个设备、哪个 kernel 或哪块内存。系统分层见 [architecture.md](../architecture.md)。

## 1. 职责与边界

`graph` 的公共头在 `include/aethermind/graph/`，实现在 `src/graph/`；模块所有权以 [AGENTS.md](../../AGENTS.md) §2.1 为准。

### 负责

| 职责 | 位置 |
|---|---|
| `ModelGraph` 容器、immutable snapshot 语义、node/value/input/output 存储 | [graph.h](../../include/aethermind/graph/graph.h)、[graph.cpp](../../src/graph/graph.cpp) |
| `GraphNodeId` / `GraphValueId`、value payload、`TensorSpec`、weight/constant/state 逻辑绑定 | [graph_types.h](../../include/aethermind/graph/graph_types.h) |
| 常用算子的构图 helper（`AddLinear`、`AddRmsNorm`、`AddAttention` 等） | [graph_op_builder.h](../../include/aethermind/graph/graph_op_builder.h)、[graph_op_builder.cpp](../../src/graph/graph_op_builder.cpp) |
| 重写事务：`GraphRewriteSession`、`GraphMutation`、`SubgraphBuilder` | [graph_rewrite.h](../../include/aethermind/graph/optimization/graph_rewrite.h)、[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp) |
| pass 框架：`GraphPass`、`PassContext`、`GraphPassManager`、checkpoint | [graph_pass_manager.h](../../include/aethermind/graph/optimization/graph_pass_manager.h)、[graph_pass_manager.cpp](../../src/graph/optimization/graph_pass_manager.cpp) |
| backend-independent semantic passes：常量折叠、DCE、QKV/Gate-Up/SiluMul/Add-RMSNorm 融合 | `src/graph/optimization/`、同名头文件 |
| 结构校验、图不变量、节点/值/消费者 liveness 查询 | [graph.cpp](../../src/graph/graph.cpp)、[graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp) |
| 诊断 dump | [graph_dump.h](../../include/aethermind/graph/graph_dump.h)、[graph_dump.cpp](../../src/graph/graph_dump.cpp) |

### 不得负责什么

| 不属于 graph | 归属 |
|---|---|
| HF config 解析、模型家族判定、`BuildModelGraph` / `BuildLlamaDense` | `model`（[llama_dense_graph_builder.cpp](../../src/model/llama_dense_graph_builder.cpp)） |
| 按 opt level 组装 target-aware pipeline、`LowerModelGraph`、`LoweredGraph` | `compiler`（[optimize_graph.h](../../include/aethermind/compiler/optimize_graph.h)） |
| kernel resolve、packed weight 物化、workspace、device/ISA 选择 | `compiler` / `execution` / `backend` |
| `ExecutionPlan`、`ExecutionContext`、TensorView 绑定、运行时 shape 校验 | `execution` |
| 算子端口 ABI、`Infer*` 参数/ dtype / shape 推导、`OpParams` 语义 | `operators` |
| 通用 shape 符号与约束求值基础设施 | `shape_inference`（见 [shape-inference.md](shape-inference.md)） |
| 整图序列化与 round-trip | 当前无；`DumpGraph` 仅用于诊断，不是稳定格式 |
| 控制流、支配关系、跨迭代调度 | 当前 IR 无控制流，不引入 |

依赖方向固定为 `graph → operators + shape_inference + base`。graph 不得 include `compiler`、`execution`、`backend` 或 `model`，也不得出现 Backend、Kernel、Workspace、DeviceType 等执行层类型。

## 2. 关键数据结构与所有权

### 2.1 标识与 value

| 类型 | 语义 |
|---|---|
| `GraphNodeId` / `GraphValueId` | `uint32_t index` 的轻量值类型，snapshot-local；只支持相等比较，identity 由容器下标决定 |
| `GraphValuePayload` | `std::variant<std::monostate, ModelInputValue, ActivationValue, WeightValue, ConstantValue, StateValue>`；variant alternative 同时充当 kind tag |
| `GraphValue` | `payload`、`spec`（`TensorSpec`）、`producer`（`std::optional<GraphNodeId>`）、`quantization`、`name` |
| `GraphNode` | `op_type`、`decoder_layer_index`、schema-ordered `inputs` / `outputs`、`attrs`、`op_params`、`runtime_checks`、`name` |
| `GraphInput` / `GraphOutput` | 只保存 `GraphValueId`；端口名来自被引用 `GraphValue::name` |

payload 与 producer 的规则：

| payload | 允许的 producer |
|---|---|
| `std::monostate` | 非法哨兵，`Validate()` 拒绝 |
| `ModelInputValue` | 不允许 |
| `ActivationValue` | 必须恰好由节点产生，且 producer 必须把该 value 列入 `outputs` |
| `WeightValue` | 不允许 |
| `ConstantValue` | 不允许 |
| `StateValue` | 允许无 producer 的初始状态，也允许节点产生更新后的新版本 |

`GraphValue` 不保存 `consumers`。use-def 的权威只有两处：`GraphNode::inputs` 表达 use，`GraphValue::producer` 表达 def；消费者索引由 `BuildConsumerIndex()` 按需派生。

`GraphNode::decoder_layer_index` 只是 semantic/debug metadata，不构成数据依赖；真实依赖必须经过 value edge。

### 2.2 逻辑绑定

| 类型 | 语义 |
|---|---|
| `WeightBinding` | `decoder_layer_index` + `WeightBindingSpec`；逻辑引用，不拥有权重存储 |
| `WeightBindingSpec` | `std::variant<DirectWeightBinding, QkvWeightBinding, GateUpWeightBinding>` |
| `DirectWeightBinding` | `ParameterSlot` + 可选 `TransformerWeightRole`；`SlotForTransformerRole()` 是 slot↔role 的唯一映射 |
| `QkvWeightBinding` / `GateUpWeightBinding` | 固定 concat 配方；Q/K/V 与 Gate/Up 的顺序由 recipe 固定，split 边界在消费算子参数中 |
| `ConstantBinding` | `shared_ptr<const std::vector<std::byte>> inline_data` + `name`；inline 字节通过共享指针 intern，Commit 复制图时不深拷贝 |
| `StateBinding` | `std::variant<KVCacheStateBinding, DecodeStateBinding, StreamingStateBinding>`；`KVCacheStateBinding{decoder_layer_index, KVCacheSlot}` 是 KV state 的结构化逻辑身份 |
| `QuantizationSpec` | `kind` / `group_size` / `scale_dtype` / `has_zero_point`；只描述模型语义量化，不描述 packed 格式 |

`ModelGraphAttrs` 是字节容器，但注册算子必须使用 typed `op_params`；`AddNode` 与 `Validate()` 对非空 `attrs.bytes` 直接报错。

### 2.3 构图与重写类型

| 类型 | 用途 |
|---|---|
| `NodeOutputDesc` | `AddNode` 的输出来源：payload tag、quantization、name；**不包含 spec**，spec 由 `InferOperator` 派生 |
| `GraphValueDesc` | 已存在 value 的完整描述（含 `spec`）；用于重写与常量折叠读取 |
| `AddedNode` | `AddNode` 返回值：`node` id + 输出 value ids |
| `GraphMutation` | `std::variant<SubgraphReplacement, NodeRemoval, InputRedirection, ValueReplacement>` |
| `ReplacementNode` / `RewriteOutputBinding` | 替换节点及其输出接管关系 |
| `GraphNodeView` | session 内的 live node 快照；inputs 经 `GetResolvedValue` 解析，outputs 保持原图 id |

### 2.4 所有权与生命周期

| 对象 | 持有者 | 生命周期 / 可变性 |
|---|---|---|
| `ModelGraph` 的 `nodes_` / `values_` / `inputs_` / `outputs_` | `ModelGraph` 按值拥有 | 构造或 `Commit()` 后视为 immutable snapshot；公共查询只返回 const span |
| `GraphValue::spec`、payload、`name` | `GraphValue` 按值拥有 | 随图生命周期；`ConstantBinding::inline_data` 为共享只读字节 |
| `WeightBinding` / `StateBinding` | 值类型逻辑引用 | 不拥有权重或 KV 存储 |
| `GraphRewriteSession` | 借用 `const ModelGraph&` | 源图必须比 session 活得久；session 拥有 rewrite overlay、替换链与缓存；非线程安全 |
| `SubgraphBuilder` | 借用 `GraphRewriteSession&` | pending `ReplacementNode` 与 alias map；`Commit()` 后清空并可复用 |
| `GraphPass` | `GraphPassManager` 通过 `unique_ptr` 拥有 | 无状态；`Run()` 为 `const noexcept` |
| `GraphPassManager` | 调用方按值拥有 | 构建完成后只读；对各自不同的输入图可并发 `Run()` |
| `Commit()` / `GraphPassManager::Run()` 返回值 | 调用方 | 新 `ModelGraph`，不依赖源图存活；常量字节可能共享 intern 存储 |

线程安全：`ModelGraph` 的 const 查询可并发；任何构图/重写 API 都不是线程安全的。`GraphRewriteSession` 的 const 查询内部使用 `mutable` 路径压缩缓存，也不能并发调用。`GraphPassManager` 在完成 `Add` / `SetCheckpointEvery` 后不再修改，`Run()` 之间不共享可变状态。

## 3. 接口与代码位置

### 3.1 文件

| 头文件 | 实现 | 内容 |
|---|---|---|
| [graph_types.h](../../include/aethermind/graph/graph_types.h) | 无 | ID、payload、`GraphValue` / `GraphNode`、绑定与量化类型 |
| [graph.h](../../include/aethermind/graph/graph.h) | [graph.cpp](../../src/graph/graph.cpp) | `ModelGraph`、校验、拓扑序、consumer index |
| [graph_op_builder.h](../../include/aethermind/graph/graph_op_builder.h) | [graph_op_builder.cpp](../../src/graph/graph_op_builder.cpp) | 常用构图 helper |
| [optimization/graph_rewrite.h](../../include/aethermind/graph/optimization/graph_rewrite.h) | [optimization/graph_rewrite.cpp](../../src/graph/optimization/graph_rewrite.cpp) | rewrite session 与 `SubgraphBuilder` |
| [optimization/graph_pass_manager.h](../../include/aethermind/graph/optimization/graph_pass_manager.h) | [optimization/graph_pass_manager.cpp](../../src/graph/optimization/graph_pass_manager.cpp) | `GraphPass`、`PassContext`、`GraphPassManager` |
| [optimization/const_evaluator.h](../../include/aethermind/graph/optimization/const_evaluator.h) | `src/graph/optimization/const_evaluator.cpp` 等 | 常量求值 `Plan` / `Evaluate` 契约与预算 |
| [graph_dump.h](../../include/aethermind/graph/graph_dump.h) | [graph_dump.cpp](../../src/graph/graph_dump.cpp) | 文本 dump 与枚举字符串 |

### 3.2 `ModelGraph`

| 接口 | 语义 |
|---|---|
| `AddInput(TensorSpec, std::string) -> GraphValueId` | 创建无 producer 的 `ModelInputValue`，同时登记 graph input |
| `AddWeight(TensorSpec, WeightBinding, std::string) -> GraphValueId` | 创建 `WeightValue`；不登记 graph input/output |
| `AddConstant(TensorSpec, ConstantBinding, std::string) -> GraphValueId` | 创建 `ConstantValue` |
| `AddState(TensorSpec, StateBinding, std::string) -> GraphValueId` | 创建无 producer 的 `StateValue` |
| `AddNode(...) -> StatusOr<AddedNode>` | 创建节点及其输出 value 的唯一原子入口；失败时图不变 |
| `MarkOutput(GraphValueId)` | 追加 graph output；合法性由 `Validate()` 检查 |
| `SetQuantization(GraphValueId, QuantizationSpec)` | 给任意 value 附加语义量化；越界 id 抛 `std::out_of_range` |
| `GetNodes()` / `GetValues()` / `GetInputs()` / `GetOutputs()` | const span 查询 |
| `GetNode(GraphNodeId)` / `GetValue(GraphValueId)` | 越界时 `AM_CHECK` 终止，不做 Status 返回 |
| `FindNodesByOpType(OpType)` | 线性扫描，按 node index 升序返回 |
| `Validate()` / `ValidateAndTopologicalOrder()` | 语义校验；后者同时返回拓扑序 |
| `TopologicalOrder()` | Kahn 拓扑排序；有环返回 `InvalidArgument` |
| `BuildConsumerIndex(const ModelGraph&)` / `GetConsumers(index, value)` | 自由函数，按需派生消费者索引 |

`AddNode` 的构造顺序：

1. 拒绝非空 `attrs`，查找 `OperatorSchema`；
2. 校验输入/输出 arity、输入 value id、payload kind 与 schema port kind；
3. 校验 weight binding 自洽/recipe/slot/layer，state binding 自洽，以及 KVCacheUpdate/Attention 的 slot/layer/family/collection 规则；
4. 按 schema 顺序收集输入 spec，调用 `InferOperator` 得到输出 spec 与 `runtime_checks`；
5. 在不触碰 `values_` / `nodes_` 的前提下暂存输出 value 与 id；
6. 预构造 `GraphNode` 与 `AddedNode`，以 `nothrow` move 提交到存储。

输出 payload 为 `std::monostate` 时归一化为 `ActivationValue`。调用方不能提供输出 `TensorSpec`；`NodeOutputDesc` 只描述类别与元数据。

### 3.3 构图 helper

| helper | 关键行为 |
|---|---|
| `AddInput` / `AddState` | 直接转调 `ModelGraph` |
| `AddLinear` | 要求 `out_features > 0`、输入 ranked 且 rank ≥ 1、最后一维 static 时为正；weight spec 为 `[out_features, in_features]` 并复用输入最后一维的 `ShapeSymbol`；节点 layer 取自 binding |
| `AddRmsNorm` | 输入 ranked 且 rank ≥ 1；weight spec 为 `[in_features]`；生成 `RmsNormParams{eps}` |
| `AddEmbedding` | `vocab_size` / `embedding_dim` 必须为正；weight spec 为 `[vocab, dim]`；节点不带 layer |
| `AddRoPE` | 接收 q、k、position_ids，返回 `RoPEOutputs{q, k}` |
| `AddKVCacheUpdate` | 接收新 K/V 与旧 K/V state，返回更新后的 `KVCachePair` |
| `AddAttention` | 接收 q、k、v，返回单个 activation |
| `AddElementwiseAdd` | 要求 lhs/rhs dtype 相同；广播不兼容由 `InferAdd` 返回错误 |
| `AddSiluMul` / `AddElementwiseMul` | 要求两个输入 `TensorSpec` 完全相等 |
| `AddSilu` / `AddArgmax` | 输出 spec 由语义推导；`AddArgmax` 不接受调用方 spec |
| `AddReshape` / `AddPermute` / `AddReorder` | 输出 spec 由 `InferReshape` / `InferPermute` / `InferReorder` 推导；输入 `QuantizationSpec` 复制到输出 |

helper 在修改图之前快照输入 spec，避免 `AddWeight` / `AddNode` 扩容导致引用失效。失败返回 `Status`，图保持不变。

### 3.4 重写接口

| 接口 | 语义 |
|---|---|
| `AllocateVirtualValue()` | 分配 rewrite-internal virtual value；只能在同一 replacement 子图内作为边 |
| `AddSessionConstant` / `AddSessionWeight` | 创建 session-local 外部 value；只有被保留消费者或 resolved graph output 引用时才 materialize |
| `Apply(std::span<const GraphMutation>)` | 顺序应用 mutation；**非原子**，失败前已应用的 mutation 保留 |
| `RemoveNode(GraphNodeId)` | 等价于 `ReplaceSubgraph({node}, {})` |
| `ReplaceSubgraph(old_nodes, replacement_nodes)` | 替换旧节点；空 replacement 表示删除；覆盖重叠 rewrite 时使旧 entry 失效 |
| `RedirectInput(node, input_index, new_value)` | 仅对 live 节点有效；`new_value` 必须是 source/session external（不得为 virtual），且 dtype 必须与旧输入一致；通过 mirror replacement 暴露修改后的 node view |
| `ReplaceValue(old_value, new_value)` | `old_value` 必须是 source value，`new_value` 必须是 source/session external；消费者重定向语义；拒绝形成替换环；不改变 value liveness |
| `GetResolvedValue(value)` | 沿替换链解析到 terminal，带路径压缩；越界 id 返回自身 |
| `GetNodeView(node)` | 返回 inputs 已解析的 live node view；非 live 返回 `NotFound` |
| `IsNodeLive` / `IsValueLive` | liveness 的单一真理源 |
| `GetTopologicalOrder` / `FindNodesByOpType` | live 节点枚举，基于源图拓扑序 |
| `GetLiveValues` | 升序返回 live value，排除 virtual 与无人接管的已删除产物 |
| `FindConsumers` | 返回 live **原图节点**，输入经解析；不含 replacement 节点 |
| `HasLiveConsumers` | 同时扫描原图节点与 active replacement；DCE 的正确判定入口 |
| `IsConstant` / `AreAllInputsConstant` | 经替换链解析后判断 constant/weight/session external |
| `GetValueOutputMetadata` | 返回含 spec 的 `GraphValueDesc`；virtual/越界返回 `InvalidArgument` |
| `IsGraphOutput` / `GetResolvedGraphOutputs` | 直接输出标记 / 经替换链解析后的输出 terminal |
| `ValidateEdits()` | 不 materialize 的 session 校验 |
| `RequestCommitPruning()` | 请求 Commit 时裁剪不可达残留；被动请求，重复 Commit 稳定 |
| `Commit(CommitOptions)` | 生成新的 immutable `ModelGraph`；`force_prune_unreachable` 可强制裁剪 |

`SubgraphBuilder` 封装 virtual value 分配、输出绑定与 `ReplaceSubgraph` 提交：`Emit` 创建 replacement 节点并返回 virtual value，`Yield` 把 virtual value 接管为源 value，`Commit` 以单个 mutation 提交；`Reset` 丢弃 pending 状态。

### 3.5 pass 框架

| 接口 / 字段 | 语义 |
|---|---|
| `PassContext::opt_level` | 默认 `2`；`0` 不注册 pass，`1` 注册 CF→DCE，`2+` 注册完整 pipeline |
| `PassContext::checkpoint_every` | 每 N 个 pass 提交快照；`0` 只在末尾提交 |
| `enable_constant_folding` / `enable_dce` | pass 内部检查；不决定注册 |
| `enable_qkv_fusion` / `enable_gate_up_fusion` / `enable_swiglu_fusion` / `enable_fused_add_rms_norm` | 同上 |
| `ConstEvalPolicy` | 单节点折叠的 `max_output_bytes` / `max_compute_ops` |
| `GraphPass::Name()` | 稳定 pass 名 |
| `GraphPass::Run(GraphRewriteSession&, const PassContext&) const noexcept` | 只通过 session 读写；返回 `Status` |
| `GraphPassManager::Add` / `AddSequential` | 追加 pass，拥有 `unique_ptr`；空指针 `AM_CHECK` 终止 |
| `GraphPassManager::SetCheckpointEvery` | 覆盖 `ctx_.checkpoint_every` |
| `GraphPassManager::Run(const ModelGraph&)` | 校验输入图，运行 pipeline，返回新图；源图不改 |

### 3.6 诊断

| 接口 | 输出 |
|---|---|
| `DumpGraph(const ModelGraph&, std::ostream&)` | `ModelGraph`、inputs、outputs、values、nodes 的逐行文本 |
| `DumpOpParams(const OpParams&, std::ostream&)` | `ParamsName{field=value, ...}` |
| `GraphValuePayloadKindName` | payload alternative 的小写标签 |
| `ToString(ParameterSlot / TransformerWeightRole / ModelSemanticRole / KVCacheSlot)` | 稳定诊断字符串 |
| `SerializeOpParams` / `ParseOpParams` | 位于 `operators/op_params_serde.h`，不属于 graph |

### 3.7 错误语义

- 构图与校验通过 `Status` / `StatusOr` 报错，不使用异常表达非法图。
- `AddNode` 失败时 `values_` / `nodes_` 未修改；成功提交由 `nothrow` move 保证不会半提交。
- `Apply` 非原子；`ValidateEdits` / `Commit` 是全有或全无，源图始终不被修改。
- `GraphPassManager::Run` 在第一个失败 pass 处终止，不返回部分结果；空 pipeline 通过 `Commit()` 返回输入图的有效副本。
- `GetNode` / `GetValue` 的越界是编程错误，`AM_CHECK` 终止；session 查询 API 对越界返回 `InvalidArgument` / `false` / 空结果。

## 4. 主流程 / 时序

### 4.1 构图与校验

```text
LoadedModel
  → model::BuildModelGraph / BuildLlamaDense
  → ModelGraph::AddInput / AddWeight / AddConstant / AddState
  → GraphOpBuilder::Add*（多数 helper 内部 AddWeight + AddNode）
  → ModelGraph::MarkOutput
  → ModelGraph::ValidateAndTopologicalOrder
```

`ValidateAndTopologicalOrder` 分三段：

1. **value 自洽**：graph input/output 引用有效；graph output 只能是 activation/constant，activation output 必须有 producer；逐 value 检查 payload↔producer、weight binding 自洽、state KV slot 已知。
2. **逐节点重放**：schema 存在、attrs 为空、arity 匹配、输入 value 有效且 payload kind 匹配 port、weight slot/recipe/layer 合法；再次调用 `InferOperator`，派生 outputs 与 `runtime_checks` 必须与存储值严格相等（包括 `error_context`）；输出不得复用输入 value；执行 state binding 规则。
3. **无环**：Kahn 算法只统计 activation 边与 produced-state 边；weight、model input、constant 边不构成执行依赖；自环直接拒绝，剩余环由节点覆盖数不足判出。

图不变量的最小集合：

| 不变量 | 权威检查 |
|---|---|
| producer 规则：activation 恰好一个 producer，model input/weight/constant 零 producer，state 零或一个 | `ValidateValueSelfConsistency` + `AddNode` |
| 无悬垂引用：graph input/output 与每个 node input/output 都引用有效 id | `ValidateAndTopologicalOrder` |
| producer–consumer 一致：producer 节点必须把 value 列入 outputs；node output 的 producer 必须是该 node | 双向回环检查 |
| schema 端口 ABI：arity、端口顺序、payload kind 与 schema 一致 | `AddNode` 与逐节点重放 |
| graph I/O 契约：output 只能是 activation/constant，activation output 必须有 producer | `ValidateAndTopologicalOrder` |
| DAG：activation 与 produced-state 边无环 | `TopologicalOrder` 的 Kahn 覆盖检查 |
| state 绑定：KV slot 正确、layer 与节点一致、K/V 同 collection、输入输出同 family | `ValidateStateBindingsForNode` |
| 替换闭包与无环：替换目标合法、dtype 不变、shape 可 Unify、virtual value 不越界、替换链不成环 | `ValidateEdits` + `ReplaceValue` |
| 确定性：同一输入图与配置产出同一拓扑序/同一优化结果 | 稳定 Kahn、固定 pass 顺序、按源图拓扑序发射、pass 无状态、snapshot 不可变 |

### 4.2 semantic pass pipeline

pipeline 的组装在 compiler 模块的 `OptimizeModelGraph`，graph 只提供 pass 与框架。按 `opt_level` 的选择：

| opt_level | pass 顺序 |
|---|---|
| `0` | 无 |
| `1` | `ConstantFoldingPass` → `DeadCodeEliminationPass` |
| `2+`（默认） | `ConstantFoldingPass` → `QkvLinearFusionPass` → `GateUpLinearFusionPass` → `SiluMulFusionPass` → `AddRmsNormFusionPass` → `DeadCodeEliminationPass` |

`checkpoint_every != 0` 时，每 N 个 pass 后 `Commit()` 生成快照，后续 pass 在最新快照上继续；最后一个 pass 落在边界上时直接返回该快照，避免重复 Commit。`checkpoint_every=0` 只在末尾 Commit。

pass 的匹配与跳过规则：

| pass | 匹配 / 变换 | 安全跳过条件 |
|---|---|---|
| `ConstantFoldingPass` | 拓扑序遍历；`IsCompileTimeEvaluable(schema)` 为真且注册了 `ConstEvaluator`；所有输入都是带 inline 字节且字节数与 spec 匹配的 constant；`Plan` → 预算 → `Evaluate`；用 `AddSessionConstant` + `ReplaceValue` 替换输出 | 无 evaluator、输入非常量、`Plan`/`Evaluate` 返回 `Unimplemented`/`Overflow`、超预算；输出数量/spec 与存储不一致属于 `Internal` 错误，不是跳过 |
| `DeadCodeEliminationPass` | 反向拓扑固定点；节点的全部输出都 dead（未被替换、不是 graph output terminal、`HasLiveConsumers == false`）且 `IsDceRemovableOp` 为真时删除；结束后 `RequestCommitPruning()` | flag 关闭；op 有 side effect 或有 stateful output；schema 未注册 |
| `QkvLinearFusionPass` | 按共享输入 + 同一 `decoder_layer_index` 聚合三个 `Linear`，以 `TransformerWeightRole::kAttentionQ/K/V` 锚定；要求输出 live、未被替换、有 live consumer 或为 graph output、dtype/quantization 一致、`in_features` 可证相等、`out_features` 静态为正且求和无溢出；创建 `QkvWeightBinding`，替换为单节点 `kQkvLinear`（三输出） | 组不完整、同角色重复（ambiguous）、权重不是 direct kernel role、输入 feature 不可证相等、输出已替换或 dead |
| `GateUpLinearFusionPass` | 同构；以 `kMlpGate` / `kMlpUp` 聚合，创建 `GateUpWeightBinding`，替换为单节点 `kGateUpLinear`（两输出） | 同上 |
| `SiluMulFusionPass` | `kSilu` 的唯一消费者是 `kElementwiseMul`，layer 一致，`silu_out` 非 graph output 且未被替换，`mul_out` 有 live consumer 或为 graph output；替换为 `kSiluMul` | 多消费者、`silu_out` 为 graph output、链完全 dead、输入被替换、两个输入都指向 silu 输出 |
| `AddRmsNormFusionPass` | 以 `kRmsNorm` 为锚，`inputs[0]` 由 `kAdd` 产生且是该 Add 的唯一 RMSNorm consumer；Add 输出未被替换；三个输入 payload 分别为 activation/activation/weight；用 `InferOperator(kAddRmsNorm)` 验证得到两个输出、无 `runtime_checks`、spec 分别等于原 RMSNorm/Add 输出；替换为 `kAddRmsNorm`，双输出分别接管原值 | Add 为 broadcast 或形状不确定、输出已替换、dead chain、同一 Add 被多个 RMSNorm 消费 |

Attention 不参与 semantic fusion：`kAttention` 在语义层保持单节点，是否使用融合 kernel 由 lowering/backend 决定。

### 4.3 重写与 liveness

`GraphRewriteSession` 以 immutable 源图为底，维护：

- `rewrites_`：每条 `RewriteEntry` 记录 `old_nodes`、`replacement_nodes`、`active`、`exposes_node_view`；
- `node_to_rewrite_`：旧 node → 当前 active rewrite；
- `session_value_metadata_`：session external value 元数据；`nullopt` 表示 virtual value；
- `value_replacements_` 与 `resolved_value_cache_`：消费者重定向与路径压缩；
- consumer cache：按 mutation generation 失效/重建。

liveness 区分两个层次：

- `IsNodeLive(node)`：未被 rewrite 触及，或仅由 `RedirectInput` 的 mirror replacement 暴露；`RemoveNode` / `ReplaceSubgraph` 产生的节点不 live。
- `IsValueLive(value)`：外部 value（无 producer）、producer 为 live 节点、或 active rewrite 的 replacement 输出通过 `replaces` 接管该 value。它不等价于 DCE-reachable；`ReplaceValue` 不影响它。

`FindConsumers` 用于 rewiring 场景，只返回 live 原图节点；`HasLiveConsumers` 是 DCE 的正确判定入口，因为它同时扫描 replacement node 的输入。virtual value 与越界 id 在两者中都返回空/false。

### 4.4 Commit

```text
Commit(options)
  → ValidateEdits()
      ├─ 替换目标合法、dtype 不变、shape 可 Unify
      ├─ replacement 节点结构与 replaces 目标合法
      ├─ virtual value 不跨 rewrite 边界、生产先于消费
      ├─ 每个 active rewrite 重放 InferOperator
      └─ replacement 输入在发射点可用
  → 计算 retained masks（显式请求 pruning 时做反向可达闭包）
  → CopyExternalValues（丢弃未被引用的 session external）
  → 按源图拓扑序发射：遇到 rewrite 的首个 old_node 时发射整组 replacement
  → MarkCommittedOutputs（resolved terminal）
  → committed.Validate()
```

默认 Commit 忠实物化所有 active rewrite 与 live 源节点；只有 `RequestCommitPruning()` 或 `force_prune_unreachable` 才裁剪不可达残留。裁剪种子包括 graph output、`RedirectInput` mirror、非 DCE-removable 的 replacement，以及 live 的非 DCE-removable 源节点。

`RedirectInput` 可以把输入改接到拓扑序更晚的节点，`GetTopologicalOrder()` 不会检测这种 back-edge，最终在 Commit 阶段以“producer 尚未发射”失败。

## 5. 关键取舍

| 选择 | 原因 | 被否方案 |
|---|---|---|
| immutable snapshot + rewrite transaction + phase checkpoint | pass 只能观察到完整合法图；批量 mutation 避免每个 pass 全量重建 | 每个 pass 直接改图：中间非法状态可能外泄；每个 pass 深拷贝：长 pipeline 成本高 |
| `GraphNodeId` / `GraphValueId` 使用容器下标 | ID 轻量、缓存友好；构造期 append-only 足以保证稳定 | 指针/对象引用：随重分配失效；字符串 ID：每次查找有哈希/分配成本 |
| payload 用 `std::variant`，不并行保存 kind enum | kind 与 binding 的一致性由类型系统保证 | 独立 `GraphValueKind` + `void*`：运行期才发现不一致 |
| 不持久化 `consumers` | use-def 只有两个权威来源，避免双向缓存漂移 | 持久 consumer 列表：任何 input 修改都必须同步两份数据 |
| `AddNode` 是创建 node 与输出的唯一原子入口，输出 spec 由 `InferOperator` 派生 | 防止调用方伪造 spec、避免 producer/value 构造循环 | 调用方提供输出 `TensorSpec` 或先构造 `GraphNode`：会产生 stale/forged metadata |
| 注册算子必须使用 typed `op_params`，拒绝 `attrs` | 参数校验、dump、pass 重写都有确定类型 | `attrs` 作为无类型参数袋：错误推迟到执行期 |
| `GraphPass::Run` 返回 `Status` 而非 `bool` | 能区分“未修改”和“非法图结构” | `bool`：无法传播失败原因 |
| `ReplaceValue` 采用消费者重定向，不先解析 old_value 的 terminal | 连续替换时不会误改其他 value 的消费者 | 先解析 terminal 再写 map：会把 terminal 的消费者错误重定向 |
| 区分 `FindConsumers` 与 `HasLiveConsumers` | replacement 节点没有 `GraphNodeId`，单个 API 无法同时服务 rewiring 与 DCE | 合并为一个消费者查询：DCE 会漏掉 replacement 消费者 |
| 只做拓扑序，不计算支配关系 | 当前 IR 无控制流，数据依赖已蕴含执行顺序 | 引入支配树：当前无收益，且增加 IR 复杂度 |
| state 用 SSA 版本表达，物理 alias 留给 lowering | 语义图保持清晰 use-def；in-place 更新属于执行计划 | 在图内表达原地更新：引入隐式副作用与环 |
| pipeline composition 归 compiler，graph 只提供 pass | target 感知编排与 lowering 不属于语义层 | graph 内组装完整优化 pipeline：破坏模块边界 |
| Commit pruning 是显式请求/兜底，不默认裁剪 | 默认行为可预测，避免 pass 未运行时静默删除节点 | 默认全图反向可达裁剪：会改变未声明裁剪时的语义 |
| DCE 固定点 + 反向拓扑 | 一次删除可解锁上游 dead 节点 | 单遍正向删除：需要多轮外部调用才能收敛 |
| fusion 保守跳过而非报错 | 语义融合是可选的优化；匹配不成立时保留原图 | 匹配失败即失败：会让正常模型因非规范拓扑无法编译 |
| 不实现整图序列化 | 当前无 round-trip/跨进程序列化用例；dump 仅诊断 | 以文本 dump 当稳定格式：格式与调试信息耦合 |

## 6. 测试与验证入口

构建与运行：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter=ModelGraph.*:ModelGraphSemanticValidation.*
```

主要套件：

| 测试文件 | 覆盖 |
|---|---|
| [test_model_graph.cpp](../../tests/unit/graph/test_model_graph.cpp) | 构图 API、producer 规则、schema arity/port kind、`op_params` 匹配、weight slot/layer、state slot/layer/family、graph output 规则、拓扑序、consumer index、stale output spec/runtime check 检测 |
| [test_graph_op_builder.cpp](../../tests/unit/graph/test_graph_op_builder.cpp) | 各 helper 的 spec 推导、broadcast、symbolic hidden dim、量化保留、失败不改图 |
| [test_graph_dump.cpp](../../tests/unit/graph/test_graph_dump.cpp) | 最小图、constant、量化、空图、Llama dense 图、全部 `OpParams` variant 的 dump |
| [test_graph_rewrite.cpp](../../tests/unit/graph/optimization/test_graph_rewrite.cpp) | 替换链解析、环拒绝、dtype/shape 保持、virtual value 作用域与顺序、session external、node/value liveness、消费者查询、`ValidateEdits`、Commit 物化与裁剪 |
| [test_graph_pass_manager.cpp](../../tests/unit/graph/optimization/test_graph_pass_manager.cpp) | 空 pipeline、顺序执行、首个错误终止、checkpoint 语义、flag/context 传递、输入图预校验 |
| [test_commit_pruning.cpp](../../tests/unit/graph/optimization/test_commit_pruning.cpp) | pruning 触发条件、可达性闭包、mirror 保留、常量对齐、重复 Commit 稳定性 |
| [test_constant_folding_pass.cpp](../../tests/unit/graph/optimization/test_constant_folding_pass.cpp) / `test_const_eval_*.cpp` | 折叠可行性、预算、inline 字节校验、链式折叠、溢出跳过、各 dtype 的 evaluator |
| [test_dead_code_elimination_pass.cpp](../../tests/unit/graph/optimization/test_dead_code_elimination_pass.cpp) | 死链删除、graph output/state/side-effect 保留、replacement 消费者、固定点与幂等 |
| [test_qkv_linear_fusion_pass.cpp](../../tests/unit/graph/optimization/test_qkv_linear_fusion_pass.cpp) / [test_gate_up_linear_fusion_pass.cpp](../../tests/unit/graph/optimization/test_gate_up_linear_fusion_pass.cpp) | 分组、MHA/GQA、split 边界、溢出/ dtype/量化/权重角色/输出替换跳过、flag 与 opt level |
| [test_silu_mul_fusion_pass.cpp](../../tests/unit/graph/optimization/test_silu_mul_fusion_pass.cpp) | 正反输入顺序、多消费者/graph output/dead chain/layer 不一致跳过、CF 后不重新引入 |
| [test_add_rmsnorm_fusion_pass.cpp](../../tests/unit/graph/optimization/test_add_rmsnorm_fusion_pass.cpp) | 双输出、层内/跨层/末层模式、broadcast/replaced/dead/多消费者跳过、幂等 |
| [test_semantic_optimization_pipeline.cpp](../../tests/unit/compiler/optimization/test_semantic_optimization_pipeline.cpp) | opt level 选择、flag 与注册分离、checkpoint 等价性、输入图不变、完整 Llama 图 |
| [test_graph_lowering.cpp](../../tests/unit/compiler/lowering/test_graph_lowering.cpp) | 优化后的图可 lower、schema/state alias/selector 校验 |

验证重点：任何新增或修改的 pass 至少覆盖“匹配成功、匹配失败、安全跳过、非法输入”四类；被融合/折叠后的图必须通过 `ValidateAndTopologicalOrder()`，并保持可 lowering。
