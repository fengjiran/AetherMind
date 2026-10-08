# ExecutableModel 模块设计

- **状态**: Current（描述已验证实现；只写仓库事实）
- **版本**: 1.3
- **日期**: 2026-09-23
- **最近更新**: 2026-10-05
- **来源提案**: [07 号提案：ExecutableModel 生产准备入口方案](../../improvement-plan/07-executable-model-preparation.md)（Implemented）
- **关联代码**: [include/aethermind/inference/](../../../include/aethermind/inference/)（`executable_model.h`、`weight_binding_storage.h`）/[src/inference/](../../../src/inference/)
- **上游依赖**: compiler（`LoweredModelArtifact`、`BuildWeightPackingRequests`）、model（`LoadedModel`/`ResolvedModelWeights`、`ResolveWeightBinding`、`PrepackWeightRequests`、`PackedWeightCollection`）、execution（`ExecutionPlanBuilder`、`ComputeExternalReadRequirements`、`ExternalTensorBindings`）、runtime（`Runtime` 提供 backends/allocator）、graph/operators 纯数据 payload 契约（`WeightValue`/`ConstantValue`）
- **下游消费者**: [`InferenceSession`](02-inference-session.md)（同步 greedy `Generate`，已落地；[01 号计划](../../improvement-plan/01-inference-session-generate-readiness.md) M5）
- **关联测试**: [tests/unit/inference/test_executable_model.cpp](../../../tests/unit/inference/test_executable_model.cpp)（20 例）、[test_weight_binding_storage.cpp](../../../tests/unit/inference/test_weight_binding_storage.cpp)（8 例）、[test_load_prepare_executable_model.cpp](../../../tests/unit/inference/test_load_prepare_executable_model.cpp)（3 例）；权重解析权威测试见 [tests/unit/model/weight/test_weight_binding_resolver.cpp](../../../tests/unit/model/weight/test_weight_binding_resolver.cpp)（`WeightBindingResolver` 套件），需求查询测试见 [test_execution_bindings.cpp](../../../tests/unit/execution/test_execution_bindings.cpp)

## 1. 背景与目标

`PrepareExecutableModel` 是编译与执行之间唯一的 artifact 准备入口：`LoweredModelArtifact → ExecutableModel`，负责把 packed 权重、外部绑定与 plan 组装起来。`LoadAndPrepareExecutableModel(runtime, model_dir, options)` 是面向 HF 模型目录的便捷编排入口，依次调用 `ModelCompiler::LoadAndCompile` 与 `PrepareExecutableModel`，不复制其内部准备逻辑。

目标：

- **单一入口**：调用方（如 `InferenceSession`）只持有 `ExecutableModel`，不读取 `LoweredGraph`/`LoweredModelArtifact`，不解析 `TransformerWeightRole`，不解释 packing 决策。
- **准备期失败优于执行期错误数值**：权重无法解析、常量无 inline 数据、phase 自相矛盾等在 prepare 期以可定位的 `Status` 失败，不推进到执行期。
- **绑定与需求同源**：不可变权重绑定表由 execution 层的唯一需求查询 `ComputeExternalReadRequirements` 推导，二者不可能分叉。

## 2. 职责与边界

- **提供**：`LoadAndPrepareExecutableModel(runtime, model_dir, options)` 文件到可执行模型的编排、`PrepareExecutableModel(runtime, artifact)` artifact 准备、`ExecutableModel::plan(phase)`、`ExecutableModel::immutable_weight_bindings(phase)`、`artifact_id()`、`phase()`。
- **请求**：`Runtime`（backend 解析与 workspace 规划）、compiler 的 request builder、model 的 resolver/prepack planner/store、execution 的 plan builder 与需求查询。
- **所有权**：`ExecutableModel` 按值持有 artifact、`PackedWeightCollection`、`WeightBindingStorage`、`ExternalTensorBindings`、`ExecutionPlan`；绑定表只借用其中数据，不复制权重。
- **明确不做**：不承担算子语义、kernel resolve 算法、weight materialization 算法（packing recipe）或 KV 物理存储；不提供 model inputs 绑定（由 Session 按 phase 追加）；不暴露 packed store 访问器。
- **生命周期**：`LoadAndPrepareExecutableModel` 借用调用方的 `Runtime`，不创建或拥有 Runtime/Session；Runtime 必须保持原地址并长于返回的 `ExecutableModel`。整体顺序为 `Runtime` > `ExecutableModel` > `InferenceSession` > `ExecutionContext`；`PreparedExecutionBindings` 借用模型数据指针，必须先于 `ExecutableModel` 释放。

## 3. 关键数据结构

### 3.1 ExecutableModel 成员与销毁契约

| 成员（声明序） | 含义 | 备注 |
|---|---|---|
| `artifact_` | `LoweredModelArtifact`（间接持有 `LoadedModel`/`ResolvedModelWeights`/`RawStorage`） | 最先声明、最后销毁，保证绑定借用的权重 backing 与常量 `inline_data` 全程有效 |
| `packed_weights_` | `PackedWeightCollection` | 与 plan 之间通过 `shared_ptr` 共享所有权；保留它是为了维持 `source_id` 与 artifact 的对账能力 |
| `binding_storage_` | `WeightBindingStorage` | 被绑定表借用的 shape/stride 元数据 |
| `bindings_` | `ExternalTensorBindings`（只含 weight/constant） | `TensorView` 借用 `binding_storage_` 的元数据与 artifact 的数据 |
| `plan_` | `ExecutionPlan` | 构建后自持，不借用 `LoweredGraph`；packed artifact 以 `shared_ptr` 持有 |
| `phase_` | artifact 的唯一 `ExecPhase` | 由 plan 各 step 的 `selector.phase` 推导 |

成员声明顺序即销毁顺序的逆序，是头文件写明的 teardown contract。

### 3.2 WeightBindingStorage

`std::vector<Entry>`，`Entry{std::vector<int64_t> shape, strides}`：

- 元数据指针稳定性由两个独立机制保证——(a) 外层 vector 扩容时 `Entry` 被 move 构造，但内层 vector 的 `data()` 跨 move 不变；(b) `ExecutableModel`/storage 整体移动时外层缓冲指针直接移交。
- 因此**禁止**把 shape/stride 换成内联存储（`std::array`、small-buffer）——任何一次扩容或移动都会让已发出的 `IntArrayView` 悬垂。
- stride 按行主序紧凑推导（权重连续性由 HF 校验器与 prepack 路径保证）；`alignment` 统一以 0（未指定）交付，因为 `RawWeightView` 不携带对齐信息。

## 4. 并发模型

准备是冷路径操作；`ExecutableModel` 构造后只读，多线程并发读取 plan/绑定表安全，但**构造与访问不并发**（头文件将并发序列化的责任交给调用方）。

## 5. 接口定义

```cpp
AM_NODISCARD StatusOr<ExecutableModel> LoadAndPrepareExecutableModel(
        Runtime& runtime, const std::filesystem::path& model_dir,
        const ModelCompileOptions& compile_options);

AM_NODISCARD StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime, LoweredModelArtifact artifact);

class ExecutableModel {
    AM_NODISCARD StatusOr<const ExecutionPlan*> plan(ExecPhase phase) const noexcept;
    AM_NODISCARD StatusOr<const ExternalTensorBindings*> immutable_weight_bindings(
            ExecPhase phase) const noexcept;
    AM_NODISCARD uint64_t artifact_id() const noexcept;
    AM_NODISCARD ExecPhase phase() const noexcept;
};
```

- 两个 phase 访问器返回 `StatusOr<const T*>`（借用指针），使 phase 不匹配可表达为失败而非静默复用。
- 只可移动构造、不可赋值：就地替换一个已 prepare 的模型会作废此前交出的 `plan()`/`immutable_weight_bindings()` 借用指针与派生的 `PreparedExecutionBindings`，且逐成员赋值会先释放 artifact 再替换借用它的绑定表（违反 §3.1 的逆序销毁契约）；换模型应构造新对象。
- 返回路径只依赖 `StatusOr` 的 nothrow 移动构造前提（`StatusOr` 不再要求 `T` 可赋值，其赋值运算符对这类 `T` 变为 deleted）。
- 不暴露 packed store 与 artifact 访问器。

## 6. 算法与流程

`PrepareExecutableModel` 八步：

1. 取 `artifact.loaded_model->GetResolvedWeights()`（缺失 `loaded_model` 即拒绝）。
2. `BuildWeightPackingRequests(artifact.graph, resolved)`：graph-driven，跳过非 `kPacked` step，按 `(value_index, selector)` 去重。
3. 取 backend（`runtime.GetBackend`，按首个 request 的 device；空请求批默认 CPU）；请求先经 `ResolveWeightPackingRequests` 注入 descriptor recipe、收敛共享消费者，再由 `PrepackWeightRequests(backend, requests)` 创建并返回绑定批内 source 的 store（`AM_ASSIGN_OR_RETURN`；失败全有或全无，空批为未绑定空 store）；composite 物化、对齐与分配由 backend 的 `Backend::PackWeights` 落实，产物 recipe 须等于 request recipe。
4. `ExecutionPlanBuilder::Build(runtime, store, artifact.graph)`。
5. `ComputeExternalReadRequirements(plan)`：execution 层的唯一需求权威（packed 裁剪掉权重端口后自然不进入需求集合）。
6. 对每个"被需求且非 `kModelInput`"的值按下标 i 物化绑定：
   - 结构化身份只能取自 `artifact.graph.values()[i].payload`（`ExecutionValueDesc` 不含 payload），下标同一性由 `PrepareTrustedGraph` 1:1 push 保证；
   - `kWeight` → 取 `WeightValue` 的 `binding` 字段交给 `ResolveWeightBinding` 解析（model 层单一权威，含 tied lm-head 回退）；
   - `kConstant` → `ConstantValue::inline_data` 按值 spec 包成 `TensorView`，并校验 inline 字节数与形状推导一致；
   - shape/stride 数组写入 `binding_storage_` 并被 `TensorView` 借用。
7. 双向对账：`{required 且非 kModelInput}` 与 `{已生成绑定}` 必须完全相等，任一方向不匹配即 `kInternal`。
8. 组装 `ExecutableModel`（成员按 §3.1 顺序，失败路径不泄漏半成品对象）。

### 6.1 phase 推导与查询

- phase 的权威来源是 **plan step 的 `selector.phase`**（lowering 期的 `GraphLoweringConfig.selector` 只是默认值，完成后不再可查）。
- step 间 phase 不一致的 artifact 在 prepare 期直接拒绝（`kFailedPrecondition`，"mixes execution phases"）。
- 内部只构建一个 plan；`plan(phase)` 复用 base 层 `PhaseMatches(candidate, request)`（`candidate == request || candidate == kBoth`）：
  - artifact 为 `kBoth` → `kPrefill`/`kDecode`/`kBoth` 查询一律返回同一 plan；
  - artifact 为单 phase → 不匹配查询返回 `kFailedPrecondition`，不静默复用。

### 6.2 目录加载编排与验证范围

- `LoadAndPrepareExecutableModel` 先调用 `ModelCompiler::LoadAndCompile(model_dir, compile_options)`，再把产物移动给 `PrepareExecutableModel(runtime, artifact)`；加载/编译和准备失败都会保留原 `StatusCode`，并在消息中标明失败阶段。
- `ModelCompileOptions` 必须显式传入。当前默认 O2 优化会形成 fused `QkvLinear`/`GateUpLinear`，而默认 plain weights 与 CPU 对应 kernel 的 packed 要求不兼容；已验证的 HF FP32 文件路径配置为 O1 + plain weights。该入口不创建 Runtime 或 InferenceSession，也不推导 KV Cache 配置。
- 调用方负责按模型配置创建 Runtime，并保证 Runtime 地址稳定且长于返回模型及其 session；返回的 `ExecutableModel` 持有编译 artifact 和权重 backing，但借用 Runtime 的 backend。
- 文件到生成的证据见 `LoadAndPrepareExecutableModel.LoadsPreparesAndGeneratesFromHfDirectory`：HF safetensors tiny Llama 使用 O1 + plain weights，入口完成加载、编译、准备及 session 创建，并成功生成 3 个 token（Prefill 后执行两步 Decode）；测试只检查 token 数量与词表范围，不构成标量数值对照或性能证据。
- O2 + packed 的现有证据是 `ExecutableModel.PackedLoweringPreparesAllWeightConsumers`：合成 tiny Llama artifact 能完成 kernel resolve 与模型准备，并检查 packed step/artifact；它不覆盖 HF 目录入口、`Generate` 或生成数值正确性。

## 7. 边界条件与错误处理

| 情形 | 错误码 | 备注 |
|---|---|---|
| artifact 无 `loaded_model` | `kInvalidArgument` | 首个检查 |
| 权重解析返回 `nullptr`（`kMoERouter`、越界/缺失 layer index、composite/roleless） | `kFailedPrecondition` | 消息含 value index 与 `role=<label>, layer=<n>` |
| 权重视图无效/不连续 | `kFailedPrecondition` | 经 `ValidateRawWeightView` + 连续性复查 |
| 常量无 `inline_data`、非静态形状、负维度、形状乘积溢出、字节数与形状不符 | `kFailedPrecondition` | 每条均含 value index |
| step 间 phase 不一致 | `kFailedPrecondition` | prepare 期拒绝 |
| phase 查询不匹配 | `kFailedPrecondition` | 两个访问器一致 |
| plan 值下标越出 artifact 值表 / payload 类别不符 | `kInternal` | 值索引同一性被破坏的内部错误 |
| 第 7 步对账不相等 | `kInternal` | 不进入执行期 |

model inputs 不进绑定表：token/position 由 Session 按 phase 追加，重复 id 由 `PrepareExecutionBindings` 的 `InvalidArgument` 校验兜底。

## 8. 风险与权衡

| 风险 | 现状与处理 |
|---|---|
| 值索引同一性是隐式约定 | 已由 plan builder 的 1:1 push 与专项测试固化；prepare 对越界索引返回 `kInternal` |
| shape/stride 借用被后续改动破坏 | §3.2 明确禁止内联缓冲；移动与扩容后的有效性有回归测试 |
| 单 plan 阻碍后续 phase 拆分 | 对外保持按 phase 查询的形状，未来拆分双 plan 不改调用方 |
| 完整 packed Llama 的可解析性依赖描述符覆盖 | `enable_packed_weights=true` 会把所有含权重的 step 标为 packed，因此每个含权重算子都必须有 packed 描述符。`kEmbedding`/`kRmsNorm`/`kLinear` 的 packed identity 变体已补齐，完整 Llama 的 packed 配置现可 prepare（`ExecutableModel.PackedLoweringPreparesAllWeightConsumers`）；新增含权重算子时若漏掉 packed 变体，会在 kernel resolve 期以 `kNotFound` 失败 |
| packed recipe 选举 | `cpu_bpanel_f32_v1_avx2` 候选与 identity 同 priority，当前选举仍落 identity；升为默认需 benchmark 证据，见 [GEMM packed weight 提案](../../operators/gemm/cpu-gemm-packed-weight.md) |
| phase 匹配语义 | 单 phase artifact 对 `kBoth` 查询返回错误（不声称覆盖两个 phase），与 `PhaseMatches` 一致 |

## 9. 测试要点

- artifact 准备：真实 `ModelCompiler` artifact（O1 未融合 tiny GQA Llama）→ `PrepareExecutableModel`，权重自动绑定并与需求集合双向对账。文件编排：HF safetensors tiny Llama（O1 + plain）→ `LoadAndPrepareExecutableModel` → `InferenceSession::Generate`，请求 3 个新 token 以覆盖 Prefill 与 Decode。
- tied/untied lm-head：绑定 `data()` 共享/独立。
- 多层（≥2 layer）：每个权重绑定到自己的 backing，不串层。
- 常量：物化成功、无 inline 数据、字节数不符三条路径。
- packed：`AddRmsNorm` 子图权重不进绑定表且 `step.packed_weights` 非空；完整 Llama 在 `enable_packed_weights=true` 下所有含权重 step 均可解析（`PackedLoweringPreparesAllWeightConsumers`）。
- phase：`kBoth` artifact 三查询同 plan；单 phase artifact 拒绝不匹配查询；混合 phase artifact prepare 期拒绝。
- 元数据稳定性：移动后、扩容后已发出的 `TensorView` 仍有效；绑定表先于模型释放的 teardown 顺序在 ASAN/TSAN 下验证。
- 契约钉住：`IsMoveConstructibleButNotAssignable` 断言只可移动构造、不可赋值/复制。

## 10. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-23 | 1.0 | 从 [07 号提案](../../improvement-plan/07-executable-model-preparation.md) 落地实现承接：入口与 API、所有权与销毁契约、八步准备流程、phase 合同、错误码表、测试要点 |
| 2026-09-24 | 1.1 | 修正两处过期断言：下游消费者 `InferenceSession` 已落地（改指 [02-inference-session.md](02-inference-session.md)）；`kEmbedding`/`kRmsNorm`/`kLinear` 的 packed identity 变体已补齐，完整 Llama 的 packed 配置现可 prepare，原"完整 packed Llama 不可解析"风险行改为描述符覆盖依赖，并新增 packed recipe 选举一行 |
| 2026-09-26 | 1.2 | 记录 `LoadAndPrepareExecutableModel` 的职责、显式编译配置与 Runtime 生命周期；区分 O1+plain 文件到 Generate 的覆盖范围和 O2+packed 的 prepare 证据 |
| 2026-10-05 | 1.3 | `PrepackWeightRequests` 签名改为返回 `StatusOr<PackedWeightStore>`：store 由函数创建并绑定批内 source，失败全有或全无（不再存在部分打包的调用方 store）；生产准备不再显式 `SetSourceId`，§6 第 3 步同步 |
