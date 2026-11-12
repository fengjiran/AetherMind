# compiler 模块设计

compiler 位于语义图之后、执行计划之前：它把 `LoadedModel` 编译为不可变的 `LoweredModelArtifact`，不解析 kernel，也不分配运行时资源。分层与依赖方向见 [系统架构](../architecture.md)，算子端口 ABI 见 [算子契约设计](../operators/operator_contract_design.md)。

## 1. 职责与边界

### 1.1 负责

| 职责 | 具体内容 |
|---|---|
| 编译阶段编排 | `ModelCompiler::Compile` 串联 `BuildModelGraph` → `OptimizeModelGraph` → `LowerModelGraph`，返回 `LoweredModelArtifact` |
| 默认语义优化 pipeline 组合 | `OptimizeModelGraph` 按 `PassContext::opt_level` 确定性地注册 pass 顺序；pass 实现本身归 graph 模块 |
| Lowering | `LowerModelGraph` 按拓扑序把每个语义节点直沉为一个 `LoweredStep`，派生 selector dtype，记录 state alias |
| 不可变 artifact | `LoweredGraph` 仅通过 const span 暴露 steps/values/model I/O/state aliases，`Build()` 后不可变 |
| 结构校验 | `ValidateLoweredGraph` 校验 schema arity、绑定、值表、selector dtype、state alias 与 artifact 内部一致性 |
| artifact 身份与打包请求映射 | `Build()` 分配 `artifact_id`；`BuildWeightPackingRequests` 从 artifact 的 kWeight value 纯数据映射出 `WeightPackingRequest` |

### 1.2 不得负责

- **不实现 graph pass**：常量折叠、Qkv/GateUp/SiluMul/AddRmsNorm 融合、DCE 的实现与框架归 graph；compiler 只决定默认顺序。
- **不做模型前端 I/O 与家族判定**：HF 配置校验、权重解析、`LoadedModel` 构造归 model；家族识别与 HF → 语义图分发由 model 的 `BuildModelGraph` 承担。
- **不做 kernel 选择**：不查询 `KernelRegistry`，不调用 `Backend::PrepareKernel`，不保存 `KernelId`/kernel 函数指针。
- **不做执行资源规划**：不计算 workspace，不决定物理 layout、buffer 分配、线程/stream 或执行调度。
- **不做权重物化**：不 mmap/加载权重，不选择 packing recipe，不执行 `Backend::PackWeights`；`WeightPackingRequest.recipe` 在编译期为空。
- **不做 in-place / may-alias 决策**：artifact 只记录 mandatory state alias；kernel 的 optional in-place 属 execution/backend。
- **不构造执行契约**：不产生 `ExecutionPlan`、`StateAliasPlan` 或 runtime tensor/state binding，也不调用 `ExecutionPlanBuilder`。
- **不重跑语义推导**：`ValidateLoweredGraph` 不调用 `InferOperator`；受信 artifact 的语义结果由 graph 构建阶段产生，未受信路径才由 execution 重新推导。
- **不感知执行层类型**：public header 不依赖 execution/backend/runtime；只携带 base 层 `KernelSelector` 纯数据契约。

### 1.3 依赖边界

依赖方向：`compiler → model + graph + operators + shape_inference + base`。compiler 可读取 `OperatorSchema`、`TensorSpec`、`ShapeConstraint`、`KernelSelector`，但不得出现 `Backend`、`KernelRegistry`、`WorkspaceArena`、`ExecutionPlan` 等执行层实体。

## 2. 关键数据结构与所有权

### 2.1 配置与产物

| 类型 | 关键字段 | 所有权与生命周期 |
|---|---|---|
| `ModelCompileOptions` | `PassContext optimization`、`GraphLoweringConfig lowering`、`ModelGraphBuildOptions graph_build` | 调用方按值传入的配置；compiler 只读，不保存 |
| `ModelGraphBuildOptions` | `activation_dtype`、`kv_dtype` | 调用方按值提供，compiler 原样转交 `BuildModelGraph` |
| `LoweredModelArtifact` | `std::unique_ptr<LoadedModel> loaded_model`、`LoweredGraph graph` | compiler 产出；`Compile` 把入参 `LoadedModel` 所有权移入 artifact；可移动 |
| `LoweredGraph` | 私有 `steps_`、`values_`、`model_inputs_`、`model_outputs_`、`state_aliases_`、`artifact_id_` | 按值拥有全部 vectors；构造后不可变；所有 accessor 返回借用该对象的 span，不得短于对象生命周期 |
| `GraphLoweringConfig` | `KernelSelector selector`、`bool enable_packed_weights` | 调用方配置；不保存 |

### 2.2 步骤、值与 state alias

| 类型 | 关键字段 | 契约 |
|---|---|---|
| `LoweredStep` | `LoweredStepSpec spec`、`LoweredStepBinding binding` | spec/binding 1:1 配对，随步骤一起移动与校验 |
| `LoweredStepSpec` | `op_type`、`selector`、`input_specs`、`output_specs`、`runtime_checks`、`op_params` | 完整 schema 端口序；`input_specs` 保留不参与 compact 推导的 state 端口 |
| `LoweredStepBinding` | `node`、`input_values`、`output_values` | `GraphValueId` 与 schema 端口位置一一对应；用于溯源与值表校验 |
| `LoweredStateAlias` | `step_index`、`input_port`、`output_port`、`input`、`output` | lowering 时以已知 step/port 坐标记录的 must-alias，不依赖执行层类型 |
| `LoweredValueDesc` | `spec`、`payload`、`quantization`、`name` | 按 `GraphValueId.index` 稠密索引；payload 可携带 weight/constant/state 绑定，深拷贝后不依赖源 `ModelGraph` |

### 2.3 跨模块数据契约

| 类型 | 关键字段 | 语义 |
|---|---|---|
| `KernelSelector` | `device_type`、`act_dtype`、`weight_dtype`、`state_dtype`、`weight_format`、`phase` | base 层纯数据；不含 ISA。`state_dtype` 对无状态步骤为 Undefined，匹配时严格相等，不作通配 |
| `WeightPackingRequest` | `op_type`、`source_id`、`value_index`、`binding`、`components`、`selector`、`recipe` | `source_id` 为 artifact_id；`components` 按逻辑绑定顺序保留 backing storage；编译期 `recipe` 为空 |

### 2.4 不变量

1. **1 节点 → 1 步骤**：不拆分、不展开、不折叠语义算子；步骤类型就是 `OpType`。
2. **步骤不含执行决策**：无 kernel、workspace、物理 layout、packed weight handle、buffer/arena offset、in-place 记录。
3. **端口序即 schema 序**：`input_specs`/`output_values` 等向量的位置就是 schema 端口 index；消费方通过 `MakeCompactInputSpecs` 派生 compact 视图。
4. **state alias 为 mandatory alias**：仅表达 state 输入必须复用同一物理 state 输出；当前唯一载体是 KV cache。
5. **artifact 自持**：值元数据、specs、runtime checks、名字均为 artifact 自持副本；源图销毁后 artifact 仍可用。
6. **不可变与并发读**：`Build()` 后不再修改；并发只读安全，span 不延长生命周期。
7. **selector 身份完整**：dtype 与 `weight_format`/`phase` 参与 kernel 匹配；`kPacked` 只允许恰好一个 kWeight 输入的步骤。

## 3. 接口与代码位置

| 入口 | 声明 | 实现 | 行为与错误语义 |
|---|---|---|---|
| `OptimizeModelGraph` | [optimize_graph.h](../../include/aethermind/compiler/optimize_graph.h) | [optimize_graph.cpp](../../src/compiler/optimize_graph.cpp) | 返回优化后的新 `ModelGraph`；源图不变；源图语义非法时在 pass 运行前返回 `InvalidArgument` |
| `LowerModelGraph` | [graph_lowering.h](../../include/aethermind/compiler/graph_lowering.h) | [graph_lowering.cpp](../../src/compiler/graph_lowering.cpp) | 自行 `ValidateAndTopologicalOrder`，构建 `LoweredGraph::Builder` 并冻结；无效图传播图校验错误 |
| `LoweredGraph` / `LoweredGraph::Builder` | [lowered_graph.h](../../include/aethermind/compiler/lowered_graph.h) | [lowered_graph.cpp](../../src/compiler/lowered_graph.cpp) | `Builder::Validate()` 不消费 builder；`Build() &&` 校验后移动并分配 `artifact_id` |
| `ValidateLoweredGraph` | [lowered_graph.h](../../include/aethermind/compiler/lowered_graph.h) | [lowered_graph.cpp](../../src/compiler/lowered_graph.cpp) | 只校验结构与溯源；失败返回 `Internal`，消息以 `ValidateLoweredGraph:` 开头 |
| `ModelCompiler::Compile` | [model_compiler.h](../../include/aethermind/compiler/model_compiler.h) | [model_compiler.cpp](../../src/compiler/model_compiler.cpp) | `nullptr` 模型返回 `InvalidArgument`；阶段失败保留错误码并加阶段前缀 |
| `ModelCompiler::LoadAndCompile` | [model_compiler.h](../../include/aethermind/compiler/model_compiler.h) | [model_compiler.cpp](../../src/compiler/model_compiler.cpp) | `ModelLoader::Load` 后转调 `Compile`；失败加 `Model loading failed:` 前缀 |
| `BuildWeightPackingRequests` | [weight_packing_request_builder.h](../../include/aethermind/compiler/weight_packing_request_builder.h) | [weight_packing_request_builder.cpp](../../src/compiler/weight_packing_request_builder.cpp) | 不触碰 backend；缺失/非法 raw weight 分别返回 `Internal`/`InvalidArgument` |

下游契约：`ExecutionPlanBuilder::Build(Runtime&, const LoweredGraph&)` 位于 execution（[execution_plan_builder.h](../../include/aethermind/execution/execution_plan_builder.h)），在信任边界重查 artifact、把 state aliases 转成 runtime `StateAliasPlan`，再 resolve kernel。compiler 只负责产出 artifact，不参与该流程。

## 4. 主流程/时序

```text
LoadedModel
  → BuildModelGraph(config, resolved_weights, graph_build)      // model
  → ModelGraph
  → OptimizeModelGraph(graph, PassContext)                       // compiler: pass composition
  → optimized ModelGraph
  → LowerModelGraph(optimized, GraphLoweringConfig)              // compiler: lowering
  → LoweredGraph (validated, artifact_id assigned)
  → LoweredModelArtifact{loaded_model, graph}
```

### 4.1 `ModelCompiler::Compile`

1. 校验 `model != nullptr`，否则 `InvalidArgument`。
2. `BuildModelGraph(model->GetConfig(), model->GetResolvedWeights(), options.graph_build)`；失败加 `Model graph construction failed:`。
3. `OptimizeModelGraph(*graph, options.optimization)`；失败加 `Model graph optimization failed:`。
4. `LowerModelGraph(*optimized, options.lowering)`；失败加 `Model graph lowering failed:`。
5. 把 `model` 移入 `LoweredModelArtifact.loaded_model`，把 lowered graph 移入 `graph` 后返回。

`LoadAndCompile` 只额外执行 `ModelLoader::Load`，其余相同。compiler 不在该阶段调用 backend、prepack 或 plan builder。

### 4.2 默认优化 pipeline

`OptimizeModelGraph` 通过 `GraphPassManager::Run` 返回新的 `ModelGraph`；`PassContext` 默认 `opt_level = 2`。

| `opt_level` | 注册顺序 |
|---|---|
| `0` | 无 pass |
| `1` | `ConstantFoldingPass` → `DeadCodeEliminationPass` |
| `2+`（默认） | `ConstantFoldingPass` → `QkvLinearFusionPass` → `GateUpLinearFusionPass` → `SiluMulFusionPass` → `AddRmsNormFusionPass` → `DeadCodeEliminationPass` |

`opt_level` 决定注册哪些 pass；`enable_*`/`const_eval_policy`/`checkpoint_every` 只作为 `PassContext` 转交，由已注册 pass 自身解释，不参与注册决策。

### 4.3 `LowerModelGraph`

1. `graph.ValidateAndTopologicalOrder()`；失败直接返回图校验错误。
2. 复制所有 `GraphValue` 的 `spec`/`payload`/`quantization`/`name` 到 `LoweredValueDesc`，复制 `GetInputs()`/`GetOutputs()` 的 value id。
3. 按拓扑序遍历节点，逐节点读取 `GetOperatorSchema(node.op_type)`；已验证图仍无 schema 时返回 `Internal`。
4. 以 schema 端口序填充 `spec.input_specs`/`output_specs` 与 `binding.input_values`/`output_values`；spec 直接取被绑定 value 的 `TensorSpec`。
5. 调用共享函数 `DeriveSelectorDTypes(schema, input_specs, output_specs)`：
   - `act_dtype`：首个 contributes 的 activation 输入；无则回退首个 activation 输出；都没有则失败；
   - `weight_dtype`：首个 contributes 的 weight 输入；无则回退 `act_dtype`；
   - `state_dtype`：所有 kState 输入/输出端口必须共享同一 dtype；无 state 端口时为 Undefined。
6. 设置 `weight_format`：
   - 含 kWeight 输入且 `enable_packed_weights == true` → `kPacked`；
   - 无 kWeight 输入 → 强制 `kPlain`；
   - 其余情况保留 `config.selector.weight_format`（默认 `kPlain`）。
7. 原样复制 `node.runtime_checks`，不在 lowering 重新推导。
8. 对 `schema.state_alias_ports` 的每个声明对，解析端口名得到 index，记录 `LoweredStateAlias{step_index = 当前 steps 数, input_port, output_port, node.inputs[port], node.outputs[port]}`；端口名解析失败即返回错误。
9. push `LoweredStep`；循环结束后 `std::move(builder).Build()`：先 `ValidateLoweredGraph`，再分配单调递增的 `artifact_id`（从 1 开始）。

### 4.4 artifact 校验范围

`ValidateLoweredGraph` 对每个步骤检查：

- `GraphNodeId` 不重复；`OpType` 有 schema；input/output binding 与 spec 的 arity 均等于 schema 端口数。
- `selector.act_dtype`、`selector.weight_dtype` 非 Undefined；`selector.state_dtype` 等于所有 kState 端口共享的 dtype（无 state 端口则为 Undefined）。
- `kPacked` 步骤恰好有一个 kWeight 输入。
- 每个绑定 value id 在范围内，且 step spec 与 `LoweredValueDesc.spec` 逐位相等。

图级检查：model inputs 均携带 `ModelInputValue`；model outputs 的 value id 合法；每条 state alias 的 step/port 在范围内、两端均为 kState、由 schema 声明、与绑定的 value id 一致、payload 均为 `StateValue` 且 `binding` 相等、无重复三元组且同一 input/output 端口不被多条 alias 复用。

明确不检查：View 合法性、resource 合法性（artifact 无此概念）、DAG 环检测（拓扑序由上游保证）。

### 4.5 打包请求映射

`BuildWeightPackingRequests` 只遍历 `selector.weight_format == kPacked` 的步骤：

1. 找到该步骤唯一 kWeight 输入端口（命中后 `break`，保持每个 packed consumer 一个请求）。
2. 要求该 value 的 payload 为 `WeightValue`，否则 `Internal`。
3. 生成请求：`source_id = artifact_id`、`value_index = GraphValueId.index`、`binding`、`selector`，`recipe` 留空。
4. 由 `ResolveWeightComponents` 解析 components：直接绑定取单一 role；`QkvWeightBinding` 取 Q/K/V；`GateUpWeightBinding` 取 Gate/Up。复合绑定必须有 `decoder_layer_index`。

该函数不选择 recipe、不调用 backend；具体 recipe 与 prepack 由准备阶段完成。

## 5. 关键取舍

| 决策 | 选择 | 原因 | 被否方案 |
|---|---|---|---|
| Lowering 粒度 | 语义 `OpType` 直沉，1 图节点 = 1 `LoweredStep` | 保持 artifact 与算子语义同构，避免引入第二套执行原语；phase 差异由 selector 表达 | 按 kernel 或 phase 拆分步骤、引入独立 primitive 层 |
| artifact 内容 | kernel-unbound、resource-unbound | kernel 选择、workspace、物理 layout、packed format 属 execution planning；提前绑定会让 compiler 依赖 backend | 在 `LoweredStepSpec` 保存 `KernelId`、`WorkspaceRequirement`、`PackedWeightHandle` 或 in-place 决策 |
| 执行属性表达 | `KernelSelector` 纯数据契约，ISA 不进入 selector | 让 graph/compiler/execution public header 不依赖 backend；ISA 由 backend 按能力快照过滤 | 把 AVX2/AVX-512/AMX 等具体指令集写入 selector 或步骤类型 |
| state alias | 算子 schema 声明端口对，lowering 记录 step/port 坐标，execution 转 runtime plan | 新增有状态算子无需 lowering 特判；compiler 不引入 execution 类型 | lowering 直接产出 `StateAliasPlan`；或 execution 扫描步骤重新发现别名 |
| artifact 校验 | 只校验结构与溯源，不重跑 `InferOperator` | 避免 compiler 成为第二语义权威；受信路径可直接复用已推导 metadata | 在 `ValidateLoweredGraph` 中重新做完整 shape/dtype 推导 |
| packed 权重 | `enable_packed_weights` 显式 opt-in；`kPacked` 仅允许一个 kWeight 输入；请求只描述逻辑绑定 | packing format 是 backend/准备阶段决策；compiler 只表达“该权重需要 packed 存储” | 在 lowering 选择 tile/panel/layout，或由 compiler 执行 prepack |
| 阶段划分 | `OptimizeModelGraph` 与 `LowerModelGraph` 保持独立入口，`ModelCompiler` 仅作无状态 facade | 诊断与测试可持有中间优化图；compiler 只拥有顺序与编排 | 把优化隐藏进 lowering，或让 `ModelCompiler` 持有可变编译状态 |
| 错误模型 | 统一 `StatusOr`，阶段失败保留原错误码并加阶段前缀 | 无异常、可定位失败阶段，调用方可按错误码分支 | 抛异常或返回 `bool` 丢失错误语义 |
| 构造入口 | `LowerModelGraph` 是唯一生产路径；`LoweredGraph::Builder` 同时是畸形 artifact 测试 seam | 集中校验与冻结；测试可注入非法结构验证拒绝路径 | 暴露可变 artifact 或允许调用方绕过校验构造 |

## 6. 测试与验证入口

构建与运行：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests --gtest_filter='ModelCompiler.*:GraphLowering.*'
```

| 测试套件 / 过滤 | 文件 | 覆盖点 |
|---|---|---|
| `ModelCompiler.*` | [test_model_compiler.cpp](../../tests/unit/compiler/test_model_compiler.cpp) | tiny Llama 的 `LoadAndCompile`、O0/O2 步骤数差异、lowering selector 透传、`nullptr` 模型拒绝、RoPE 配置可编译性 |
| `ModelCompilerGraphBuildOptions.*` | 同上 | `activation_dtype`/`kv_dtype` 透传、源 config hint 保留、非法 dtype 构图失败的阶段前缀 |
| `GraphLowering.*` | [test_graph_lowering.cpp](../../tests/unit/compiler/lowering/test_graph_lowering.cpp) | 单节点 lowering、拓扑序与 op params、完整 schema 端口序、compact 视图、state 端口、runtime checks 原样透传、源图销毁后 artifact 可用、自定义 selector 前缀 |
| `LoweredGraphBuilder.*` | 同上 | 未声明 state alias、重复 state alias 的拒绝 |
| `DefaultGraphPassPipeline.*`、`OptimizeModelGraph.*` | [test_semantic_optimization_pipeline.cpp](../../tests/unit/compiler/optimization/test_semantic_optimization_pipeline.cpp) | O0/O1/O2/O3/99 的 pipeline 行为、默认 O2、源图不变、feature flag 与 const-eval policy、checkpoint 等价性、CF 先于融合 |
| `GraphCompilerPrecondition/OptLevelPreconditionTest.*` | 同上 | 每个 opt level 都在 pass 运行前拒绝非法源图 |
| `GraphCompilerIntegration.*` | 同上 | `runtime_checks` 经 `OptimizeModelGraph → LowerModelGraph → ExecutionPlanBuilder` 原样传递，并在冷路径绑定阶段拒绝违约 shape |
| `ExecutionStateAliasAdapter.*` | 同上 | compiler 记录的 alias 能被 execution 适配为 `StateAliasPlan`（跨边界验证） |
| `WeightPacking.LoweredDriven*` | [test_weight_packing.cpp](../../tests/unit/model/weight/test_weight_packing.cpp) | 从真实 lowered artifact 派生请求、复合绑定 Q/K/V 与 Gate/Up 展开、prepack 后可解析 |

约束与缺口：

- `ModelCompiler` 测试使用 `AETHERMIND_TEST_MODELS_DIR/tiny-random-LlamaForCausalLM` fixture。
- 畸形 artifact 目前通过 `LoweredGraphBuilder` 覆盖未声明/重复 alias；其他校验分支没有独立负例，修改 `ValidateLoweredGraph` 时应补对应 seam 测试。
- `ValidateLoweredGraph` 不验证 View/resource/DAG 环；这些依赖上游图校验与 execution 的 trust-boundary 检查。
