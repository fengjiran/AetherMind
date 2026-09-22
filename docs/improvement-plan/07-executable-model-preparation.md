# ExecutableModel 生产准备入口方案

- **状态**: Draft
- **版本**: 1.1
- **日期**: 2026-09-23
- **最近更新**: 2026-09-23
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **关联计划**: [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md)（本提案是其 M2 的细化）
- **关联模块**: model / compiler / execution / runtime / inference（新增）

## 1. 结论与范围

`LoweredModelArtifact → ExecutionPlan → PreparedExecutionBindings` 的每个构件都已单独实现并有测试，但**没有任何生产代码把它们串起来**：`BuildWeightPackingRequests` 与 `WeightPrepackPlanner::PrepackAndStore` 至今只有测试调用者，`ExecutionPlanBuilder::Build` 无 `LoweredModelArtifact` 重载（现有 4 个重载只接受 `LoweredGraph` 或 untrusted `ExecutionPlanNodeSpec` 列表），`ExternalTensorBindings` 全部由测试 helper 手工拼装。本提案定义唯一的生产准备入口 `PrepareExecutableModel`，并给出它所需的三处公共合同补齐。

拆分依据见 [系统能力演进路线图](06-system-capability-evolution-roadmap.md) §20：本提案引入新的 artifact/owner/lifetime（`ExecutableModel`），并修改 model / compiler / execution 三个模块的公共合同。

### 1.1 本计划包含

- `ExecutableModel` 类型：所有权、生命周期与 phase plan 查询接口；
- `PrepareExecutableModel`：artifact → packing request → prepack → plan build → immutable external binding map → 完整性验证；
- model 层权重身份解析单一权威（收敛 tied lm-head 的两处重复实现）；
- execution 层 external binding 需求集合的公开查询接口；
- 新增 `inference/` 模块与根 `AGENTS.md` ownership 表更新；
- 从真实 `LoweredModelArtifact` 出发的完整 Llama plan 构建测试。

### 1.2 本计划不包含

- `InferenceSession` / `Generate` 编排（属 [01](01-inference-session-generate-readiness.md) M5）；
- Prefill→Decode 端到端数值验证（属 01 M4，依赖本提案交付的入口）；
- 真实 tile/block packing recipe（当前 `cpu_identity` 逻辑行主序拷贝，见 [CPU GEMM 优化方案](../operators/gemm/cpu-gemm-optimization.md)）；
- phase-specific 双 plan 拆分（见 §4.5，baseline 共享单个 `kBoth` plan）；
- tokenizer、sampling、服务化、GPU 执行。

## 2. 已验证的当前状态

### 2.1 已具备的构件

| 构件 | 位置 | 现状 |
|---|---|---|
| `LoweredModelArtifact{unique_ptr<LoadedModel>, LoweredGraph}` | [`model_compiler.h:25-28`](../../include/aethermind/compiler/model_compiler.h) | 已实现；artifact 自带 `LoadedModel`，可经 `GetResolvedWeights()`（[`loaded_model.h:25`](../../include/aethermind/model/loaded_model.h)）取到真实权重 |
| `ResolvedModelWeights` | [`resolved_model_weights.h:11-42`](../../include/aethermind/model/resolved_model_weights.h) | 嵌套结构体（非 role map）；`lm_head` 为 `std::optional<RawWeightView>` |
| `RawWeightView` | [`raw_weight.h:21-37`](../../include/aethermind/model/raw_weight.h) | `data/bytes/dtype/shape/shared_ptr<const RawStorage>/is_contiguous` + `IsValid()`/`IsAligned()`；backing 为引用计数存储 |
| `BuildWeightPackingRequests` | [`packing_request_builder.h:28-30`](../../include/aethermind/compiler/packing_request_builder.h) | 已实现且 graph-driven：跳过 `weight_format != kPacked` 的 step，按 `(value_index, selector)` 去重 |
| `WeightPrepackPlanner::PrepackAndStore` | [`weight_prepack_planner.h:57-59`](../../include/aethermind/model/weight_prepack_planner.h) | 已实现；composite 权重按 recipe 序 axis-0 拼接进自有对齐存储 |
| `PackedWeightStore` | [`packed_weight_store.h:39-58`](../../include/aethermind/model/packed_weight_store.h) | 已实现；与 plan 共享 `shared_ptr` 所有权，`source_id` 首次 Store 后冻结 |
| `ExecutionPlanBuilder::Build` | [`execution_plan_builder.h:67-81`](../../include/aethermind/execution/execution_plan_builder.h) | 已实现 `LoweredGraph` 与 `(PackedWeightStore, LoweredGraph)` 重载 |
| packed/plain 端口裁剪 | [`execution_plan_builder.cpp:61-73`](../../src/execution/execution_plan_builder.cpp) | `weight_format == kPacked` 时丢弃 `kWeight` 语义端口 |
| `PrepareExecutionBindings` | [`execution_bindings.h:109-112`](../../include/aethermind/execution/execution_bindings.h) | 已实现；缺必需绑定报 `FailedPrecondition`，重复 id 报 `InvalidArgument` |

### 2.2 缺失的构件与已存在的重复实现

| 缺口 | 事实依据 | 影响 |
|---|---|---|
| 无 artifact 级准备入口 | `PrepareExecutableModel`/`ExecutableModel` 在仓库中只出现于 docs；`ExecutionPlanBuilder::Build` 无 `LoweredModelArtifact` 重载 | 调用方必须自己按正确顺序拼装 5 个组件 |
| packing 链路无生产调用者 | 调用者全部在测试且均为单算子图：`test_weight_prepack_planner.cpp`、`test_cpu_qkv_linear_kernel.cpp:624`、`test_cpu_add_rmsnorm_kernel.cpp:729`、`test_cpu_gate_up_linear_kernel.cpp:642` | packed 路径从未在真实模型上走通 |
| 无真实权重 → external binding 的生产 API | 生产侧 `ResolvedModelWeights` 只由 `ModelLoader` 解析、`LoadedModel` 持有，无任何代码把它转成 `ExternalTensorBindings`；该转换只存在于测试（`ResolvedModelWeights` 出现在 11 个测试文件 84 处，`ExternalTensorBindings` 由 `tests/unit/execution/test_execution_binding_helpers.h` 手工构造） | 01 §3.3 未闭环 |
| role → 原始权重的解析是私有的且重复两份 | `FindRawWeightByRole` 位于 [`packing_request_builder.cpp:83-135`](../../src/compiler/packing_request_builder.cpp) 匿名命名空间；tied lm-head 三元式在 `:97-101` 与 [`model_graph_builder.cpp:509-511`](../../src/model/model_graph_builder.cpp) 各写一遍 | 第三份实现（plain binding 映射）会再复制一次 tied 语义 |
| external binding 需求集合无公开查询 | `ComputeExternalReadRequirements`（[`execution_bindings.cpp:238-264`](../../src/execution/execution_bindings.cpp)）在匿名命名空间内，仅 `:399` 自用 | 准备入口若不复制该逻辑，就无法保证"不重复不遗漏" |
| `ConstantValue.inline_data` 未被执行层消费 | payload 携带 `shared_ptr<const vector<byte>>`（[`graph_types.h:226-237`](../../include/aethermind/graph/graph_types.h)），但 `PrepareExecutionBindings` 无条件要求每个 `kConstant` 提供 external 绑定 | 常量折叠产生的常量目前无人物化 |
| 无完整 Llama plan 构建证据 | `BuildLlamaDense` 只出现在 model/graph/compiler 测试；`OptimizeModelGraph.LowersFullLlamaDenseGraph` 止于 lowering | 01 §9 "baseline pipeline 可通过真实 CpuBackend 构建完整 plan" 未勾选 |

### 2.3 可直接复用的既有不变量

以下事实已由代码或测试确立，本方案在其上设计，不重复论证：

- **值索引同一性**：`PrepareTrustedGraph` 按 lowered 顺序 1:1 push 值（[`execution_plan_builder.cpp:484-505`](../../src/execution/execution_plan_builder.cpp)），因此 `ExecutionValueId{index}` 与 `LoweredGraph` 的 `GraphValueId{index}` 同索引；`packed_key->value_index` 直接用于索引 `graph.values`（`:648`）即为佐证。
- **plan 构建后自持**：`ExecutionPlanBuilder.TrustedPathCopiesValueDataflowAfterLoweredGraphLifetimeEnds`（[`test_execution_plan_builder.cpp:1035`](../../tests/unit/execution/test_execution_plan_builder.cpp)）证明 plan 不借用 `LoweredGraph`。
- **packed artifact 生命周期解耦**：plan step 持 `shared_ptr<const PackedWeights>`，store 销毁后 plan 仍可执行（`packed_weight_store.h:39-45`）。
- **权重连续性**：HF 校验器拒绝非连续视图（[`hf_model_validator.cpp:83`](../../src/model/formats/hf/hf_model_validator.cpp)）；packing 路径另行复查（`weight_prepack_planner.cpp:68`）。
- **tied lm-head 语义**：解析结果不是标志位，而是复用同一 `RawWeightView`（共享 `storage`），由 `BuildRequestsFallBackToEmbedTokensForTiedLmHead`（`test_weight_prepack_planner.cpp:733-784`）覆盖。

## 3. 目标架构

### 3.1 模块归属与依赖

新增 `inference/` 模块（`include/aethermind/inference/` + `src/inference/`），作为 execution 之上的编排层，同时是 01 M5 `InferenceSession` 的落地位置。

```text
inference → execution + compiler + model + runtime
          + graph/operators 的纯数据 payload 契约（WeightValue / ConstantValue / WeightBinding）
```

不得放入 `runtime`（runtime 禁止依赖 execution/compiler/graph/model，见根 `AGENTS.md` §2.1）。落地前须在 `AGENTS.md` 模块 ownership 表新增一行，规则见 01 §4.2。

### 3.2 对象与所有权

```text
ExecutableModel                            模型生命周期（move-only）
├── owns LoweredModelArtifact              → 间接 owns LoadedModel / ResolvedModelWeights / RawStorage
├── owns PackedWeightStore                 → 与 plan 共享 shared_ptr 所有权
├── owns WeightBindingStorage              → 每个绑定的 shape/stride 数组（堆稳定）
├── owns ExternalTensorBindings            → 仅 weight/constant 只读子集，数据指针借用上面两项
└── owns ExecutionPlan                     → 构建后自持，不借用 LoweredGraph
```

成员声明顺序即销毁顺序的逆序：`artifact` 最先声明、最后销毁，保证 bindings 借用的 `RawStorage` 与常量 `inline_data` 全程有效。

### 3.3 接口轮廓

```cpp
// include/aethermind/inference/executable_model.h
class ExecutableModel {
public:
    ExecutableModel(ExecutableModel&&) noexcept;
    ExecutableModel(const ExecutableModel&) = delete;

    /// 按 phase 取 plan；baseline 下 prefill/decode 共享同一不可变 plan。
    AM_NODISCARD StatusOr<std::reference_wrapper<const ExecutionPlan>>
    plan(ExecPhase phase) const noexcept;

    /// 只含 weight/constant 的不可变只读绑定；model inputs 由 Session 在
    /// prepare 时追加（见 §4.4）。
    AM_NODISCARD const ExternalTensorBindings& immutable_weight_bindings(
            ExecPhase phase) const noexcept;

    AM_NODISCARD uint64_t artifact_id() const noexcept;
    AM_NODISCARD const PackedWeightStore& packed_weights() const noexcept;
};

/// 唯一生产准备入口。artifact 按值移入并被 ExecutableModel 拥有。
StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime,
        LoweredModelArtifact artifact);
```

相对 01 §4.2 的示意签名，本提案**去掉 `ExecutableModelOptions`**：`enable_packed_weights` 与 `selector.phase` 已在编译期固化进 artifact 的 step selector，准备阶段无可配置项。01 §4.2 明确该轮廓"不是已冻结 public API"。

### 3.4 准备流程

```text
PrepareExecutableModel(runtime, artifact)
  1. resolved = artifact.loaded_model->GetResolvedWeights()
  2. requests = BuildWeightPackingRequests(artifact.graph, resolved)     // 已有
  3. WeightPrepackPlanner::PrepackAndStore(store, requests)              // 已有；store.SetSourceId 一致性
  4. plan = ExecutionPlanBuilder::Build(runtime, store, artifact.graph)  // 已有
  5. required = CollectExternalReadRequirements(plan)                    // §4.2 新公开接口
  6. 对每个 required[i] 为真、且 plan.values()[i].kind != kModelInput 的 i：
       payload = artifact.graph.values()[i].payload    // 取值来源见下方说明
       kWeight    → ResolveWeightBinding(get<WeightValue>(payload).binding, resolved)
       kConstant  → get<ConstantValue>(payload).inline_data
     二者包成 TensorView；shape/stride 数组写入 WeightBindingStorage 并被 TensorView 借用
  7. 完整性对账：{required 且非 kModelInput} 与 {已生成绑定} 两个集合必须完全相等，
     缺失或多余即 FailedPrecondition（错误信息含 value index）
  8. 组装 ExecutableModel（成员按 §3.2 顺序）
```

step 6 的 payload 来源必须写死：`ExecutionValueDesc` 只有 `spec/kind/state_binding/name`，**不含** `WeightBinding` 或 `ConstantBinding`（[`execution_plan.h:75-80`](../../include/aethermind/execution/execution_plan.h)），从 plan 侧取 binding 会落空。结构化身份只能从 artifact 持有的 `LoweredGraph.values()[i].payload` 读取，其正确性完全依赖 §2.3 的值索引同一性（plan 值数组按 lowered 顺序 1:1 构建）。`kind` 取自 plan 侧即可，与 lowered payload 分类一致。

step 7 的对账集合必须排除 `kModelInput`：token/position 输入由 Session 在 prepare 期追加（§4.4），准备阶段无权重可绑，若按 `required` 字面全集对账会把它们误报为缺失。

## 4. 关键设计裁决

### 4.1 权重身份解析单一权威（model 层）

把 `FindRawWeightByRole` 从 `packing_request_builder.cpp` 匿名命名空间提升为 model 层公共 API：

```cpp
// include/aethermind/model/weight_binding_resolver.h
/// 按结构化身份解析原始权重；不依赖字符串或 debug name。
AM_NODISCARD const RawWeightView* ResolveWeightBinding(
        const WeightBinding& binding,
        const ResolvedModelWeights& resolved) noexcept;
```

放 model 层的理由：它映射 `WeightBinding`（graph 纯数据类型）+ `ResolvedModelWeights` → `RawWeightView`，而 `AGENTS.md` 已允许 model → graph。改造后三处共用同一实现：`packing_request_builder.cpp`、`model_graph_builder.cpp:509-511`、以及本提案的 plain binding 映射，tied lm-head 语义只有一份。

`QkvWeightBinding`/`GateUpWeightBinding` 为 packed-only（`kQkvLinear`/`kGateUpLinear` 无 plain 描述符），因此 plain 路径只会遇到 `DirectWeightBinding`；解析器仍覆盖 composite 以保持单一权威，composite 的组件序由既有 recipe 定义。

`nullptr` 契约必须随提升一起写明：解析失败（越界 layer index、`kMoERouter`，见 [`packing_request_builder.cpp:131-132`](../../src/compiler/packing_request_builder.cpp)）返回 `nullptr` 而非 `Status`，与既有私有实现保持一致。plain binding 映射处遇 `nullptr` 必须转成 `FailedPrecondition`（错误信息含 value index 与 role），不得静默跳过、也不得回退到任何默认权重——缺一个权重的 plan 在执行期才暴露会难定位得多。

### 4.2 external binding 需求集合的唯一来源（execution 层）

把 `ComputeExternalReadRequirements` 提升为公开 API（`execution_bindings.h`），签名保持 `StatusOr<std::vector<bool>>`，并在 `PrepareExecutionBindings` 内部继续使用同一函数。这样"哪些值需要 external 绑定"只有 execution 一个权威，packed step 裁剪掉的权重端口自然不进入需求集合，**结构性地**满足 01 M2 验收项"packed/plain 路径不会重复或遗漏 binding"，而不是靠准备入口复制一份判断。

### 4.3 常量物化

`PrepareExecutionBindings` 对每个 `kConstant` 无条件要求 external 只读绑定，而 `ConstantBinding.inline_data` 已随 payload 进入 artifact。准备入口按值 spec（dtype/shape）把 `inline_data` 包成 `TensorView`，并校验 `inline_data->size()` 与 spec 推导字节数一致。

当前 `ModelGraphBuilder` 不产生常量（RoPE 表是 `RoPEParams` 内核参数），但 O2 常量折叠会（`constant_folding_pass.cpp:38-56`），因此该路径必须实现而非假设集合为空。

### 4.4 shape/stride 元数据所有权

`TensorView(data, dtype, IntArrayView shape, IntArrayView strides, alignment)` **借用** shape/stride 数组（`tensor_view.h:49-61`），而 `RawWeightView` 只有 shape、没有 stride。因此 `ExecutableModel` 必须自有每个绑定的元数据：

- `WeightBindingStorage` 为 `std::vector<Entry>`，`Entry` 内含 `std::vector<int64_t> shape, strides`；
- 元数据指针稳定性由两个**独立**机制保证，二者缺一即悬垂：(a) `ExecutableModel` 整体移动时，外层 vector 的堆缓冲指针直接移交，`Entry` 不重定位；(b) 构建期 `push_back` 触发外层扩容时 `Entry` 被 move 构造，但 `Entry` 内层 `std::vector<int64_t>` 的 `data()` 跨 move 不变（移动只转移缓冲所有权）；
- 因此禁止把 shape/stride 换成内联存储（`std::array`、small-buffer 优化、`absl::InlinedVector` 之类）：那会同时破坏 (a) 与 (b)，任何一次扩容或移动都会让已发出的 `IntArrayView` 悬垂；
- stride 按行主序紧凑推导（连续性由 §2.3 保证），`alignment` 透传实际对齐。

model inputs 不进入 `immutable_weight_bindings()`：token/position 输入由 Session 按 phase 提供。Session 复制 weight/constant 只读条目后追加自身输入条目，重复 id 由 `PrepareExecutionBindings` 的 `InvalidArgument` 校验兜底（`execution_bindings.cpp:360-363`）。

### 4.5 phase plan 合同

`ExecPhase` 当前只作为 `KernelSelector` 字段存在，全部 CPU 描述符为 `kBoth`。phase 的**权威来源是 artifact 内 per-step 的 `selector.phase`**，不是 `GraphLoweringConfig.selector`：后者只是 lowering 期的默认值（[`graph_lowering.h:17-28`](../../include/aethermind/compiler/graph_lowering.h)），编译完成后不再可查，且 lowering 会按算子覆写 `weight_format`，未来同样可能覆写 phase。因此：

- prepare 期扫描全部 step 的 `selector.phase` 得出 artifact 的唯一 phase 并记录进 `ExecutableModel`（step 间不一致即拒绝，见下）；
- 内部只构建并持有一个 plan；
- `plan(phase)` 的匹配复用既有 `PhaseMatches(candidate, request)`（[`kernel_attrs.h:40-43`](../../include/aethermind/base/kernel_attrs.h)，`candidate == request || candidate == kBoth`），不新写匹配逻辑：candidate 为记录的 artifact phase，request 为调用方查询的 phase；
- artifact phase 为 `kBoth` 时对 `kPrefill`/`kDecode`/`kBoth` 三种查询一律返回同一 plan；
- 若 artifact 以 `kPrefill` 或 `kDecode` 编译，查询不匹配的 phase 返回错误，不静默复用；
- **step 间 phase 不一致的 artifact 在 prepare 期直接拒绝**：当前 lowering 不产生该形态，但类型上允许；静默接受会让"共享单 plan"的语义不可判定，也会把矛盾推迟到执行期；
- 对外保持按 phase 查询的形状，未来拆分双 plan 不改调用方。

### 4.6 生命周期与销毁顺序

```text
Runtime > ExecutableModel(artifact, store, 元数据, bindings, plan) > InferenceSession > ExecutionContext
```

- bindings 借用 `RawStorage` 与 `inline_data` → artifact 必须与 `ExecutableModel` 同生命周期（由所有权直接保证）；
- plan 持 packed artifact 的 `shared_ptr` → 即使 store 先销毁也可执行（以 `packed_weight_store.h:39-45` 与成员类型为准；`execution_plan.h:84-85` 的 "borrowed pointer / store must outlive this plan" 注释与之矛盾，属失实注释，M2.4 修正）；`ExecutableModel` 仍持有 store 以维持 `source_id` 与 artifact 的对账能力；
- `PreparedExecutionBindings` 借用 external 数据指针（`execution_bindings.h:59-64`）→ 由 Session 保证其先于 `ExecutableModel` 销毁，该约束在 M5 落地并测试，本提案只在头文件契约中写明。

## 5. 方案与备选

### 5.1 推荐：新增 inference 模块 + 三处公共合同补齐

即 §3–§4。优点：职责与 01 §4.1 的对象图一一对应；tied lm-head 与 binding 需求各只有一个权威；M5 有现成落点；完整 Llama plan 首次获得真实 CpuBackend 证据。

### 5.2 否决：放入 execution 模块

`AGENTS.md` 允许 execution 实现依赖 compiler artifact 与 model 的打包契约，故技术上可行。但 `ExecutableModel` 要在公共头拥有 `LoweredModelArtifact`（间接拥有 `LoadedModel`），会把 model 的所有权语义抬进"执行数据契约层"，与 execution 现有定位（plan/bindings/context 的窄契约）冲突；且 M5 的 Session 仍需另找归属，导致 `ExecutableModel` 与 `InferenceSession` 分家。

### 5.3 否决：放入 model 模块

model 禁止依赖 execution/runtime，而准备入口必须调用 `ExecutionPlanBuilder::Build(runtime, ...)`。不可行。

### 5.4 否决：给 ExecutionPlanBuilder 增加 artifact 重载

只解决 plan 构建一步，binding 映射、常量物化与元数据所有权仍无归属，调用方依旧要理解 `TransformerWeightRole`，违反 01 §3.2 的职责划分。

### 5.5 否决：Session 直接组装

即 01 §5.2 已否决的路线在 M2 粒度上的重现：Session 将承担 weight mapping 与 phase plan 解释，形成错误边界。

## 6. 实施步骤

### M2.1 model：权重身份解析单一权威

提取 `ResolveWeightBinding` 到 `include/aethermind/model/weight_binding_resolver.h`，改造 `packing_request_builder.cpp` 与 `model_graph_builder.cpp` 复用之；补 tied lm-head / 越界 layer / `kMoERouter` 的单测，含 §4.1 `nullptr` 错误路径（调用方必须转 `FailedPrecondition`）。

退出条件：仓库内 tied lm-head 三元式只剩一处；`nullptr` 返回路径与调用方错误转换各有专项测试；既有测试全绿。

### M2.2 execution：公开 external binding 需求查询

提升 `ComputeExternalReadRequirements` 为公共 API 并补文档注释；新增 packed/plain 混合图的查询测试（同一权重同时被 packed step 与 plain step 消费的情形必须可判定）。

退出条件：`PrepareExecutionBindings` 与新公开接口共用同一实现，无逻辑副本。

### M2.3 inference：模块骨架与 ownership 登记

新增 `include/aethermind/inference/` 与 `src/inference/`；定义 `ExecutableModel` 与 `WeightBindingStorage`，头文件写明 §4.6 生命周期契约；更新根 `AGENTS.md` §2.1 ownership 表与跨模块依赖规则。

**CMake 无需改动**：`src/CMakeLists.txt:1-4` 与 `tests/unit/CMakeLists.txt:4-6` 均为 `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)`，新目录自动收录；`include/` 已在 `AetherMind` 的 PUBLIC 头路径上。因此本步唯一的仓库级登记动作是 `AGENTS.md`。

退出条件：`AGENTS.md` §2.1 新增 inference 行且依赖规则与 §3.1 一致；`--target AetherMind` 与 `--target aethermind_unit_tests` 均编译通过，且新目录下的源文件确实进入构建（以一个占位测试的符号可链接为判据，避免 glob 未命中却静默通过）。

### M2.4 inference：`PrepareExecutableModel`

按 §3.4 实现 8 步流程，含 §4.3 常量物化与第 7 步完整性对账；错误路径不泄漏半成品对象。

同批修正三处与实现不符的既有注释：

- [`packed_weight_store.h:53`](../../include/aethermind/model/packed_weight_store.h) 与 [`weight_prepack_planner.h:23`](../../include/aethermind/model/weight_prepack_planner.h) 把 `artifact_id()` 归给 `LoweredModelArtifact`，实际只定义在 `LoweredGraph`（[`lowered_graph.h:156`](../../include/aethermind/compiler/lowered_graph.h)）；`ExecutableModel::artifact_id()` 直接委托 `artifact.graph.artifact_id()`；
- [`execution_plan.h:84-85`](../../include/aethermind/execution/execution_plan.h) 称 `packed_weights` 是 "borrowed pointer into a PackedWeightStore's storage; the store must outlive this plan"，与同文件 `:91-92`（plan 自持引用，store 销毁后仍可执行）直接矛盾；实际成员类型是 `std::shared_ptr<const PackedWeights>`，应删除失实的前者。

退出条件：从真实 `LoweredModelArtifact` 构建成功，无手工拼 plan 路径；上述注释与实现一致。

### M2.5 测试：完整 Llama 真实后端证据

`BuildLlamaDense` → `ModelCompiler::Compile` → `PrepareExecutableModel(真实 CpuBackend Runtime)`；覆盖 tied lm-head、GQA、`enable_packed_weights` 真/假两种 lowering、常量折叠产生的常量、销毁顺序。

退出条件：01 §9 "baseline pipeline 可以通过真实 CpuBackend 构建完整 plan" 与 "`PrepareExecutableModel` 可从真实 `LoweredModelArtifact` 构建"、"real weights 可自动生成完整 external bindings" 三项可勾选。

## 7. 验收标准

对齐 01 M2 验收项，并给出可执行判据：

| 01 M2 验收项 | 本提案判据 |
|---|---|
| 从真实 artifact 构建，不手工拼 plan | 测试只经 `ModelCompiler` 产出 artifact，不出现 `ExecutionPlanNodeSpec` 手工构造 |
| Session 不读取 compiler artifact | `ExecutableModel` 不暴露 `LoweredGraph`/`LoweredModelArtifact` 访问器 |
| tied lm-head 正确共享 backing | 断言 lm-head 与 embed_tokens 绑定 `data()` 相同，且二者 value id 不同 |
| packed/plain 不重复不遗漏 | 绑定集合与 §4.2 需求集合逐一对账；缺任一方向即测试失败 |
| 销毁顺序有明确测试或 contract | 头文件写明成员顺序契约 + ASAN/TSAN 下的 teardown 测试 |

补充判据：

- 完整性验证在第 7 步失败时返回可定位的 `FailedPrecondition`（含缺失 value index），不进入执行期，且不留下半成品 `ExecutableModel`；
- 权重解析返回 `nullptr` 时转 `FailedPrecondition`（含 value index 与 role），覆盖 `kMoERouter` 与越界 layer；
- `plan(phase)` 对 phase 不匹配的 artifact 返回错误而非静默复用；step 间 phase 不一致的 artifact 在 prepare 期即被拒绝；
- 元数据堆稳定性有回归测试（`ExecutableModel` 移动后、以及绑定数量增长触发外层扩容后，已发出的 `TensorView.shape()`/`strides()` 仍有效）。

## 8. 风险与依赖

| 风险 | 影响 | 处理原则 |
|---|---|---|
| 值索引同一性是隐式约定 | 绑定错位导致数值错误 | §2.3 已验证；在 `PrepareExecutableModel` 内以 `AM_DCHECK` + 专项测试固化 |
| shape/stride 借用被后续改动破坏 | 悬垂指针 | §4.4 明确禁止内联缓冲；移动后有效性测试 |
| 常量路径无真实模型覆盖 | 折叠常量在准备期才暴露 | M2.5 显式构造含常量的 artifact |
| 新模块引入依赖环 | 构建失败 | inference 单向依赖 execution/compiler/model/runtime；`AGENTS.md` 登记后评审 |
| 提升 `ComputeExternalReadRequirements` 扩大 execution 公共合同 | 后续演进受限 | 只暴露只读查询，语义仍由 execution 独占 |
| packed 路径首次跑真实模型 | 可能暴露 `cpu_identity` recipe 与 selector 对账缺陷 | 属预期收益；缺陷按 [算子开发工作流](../guides/operator-development-workflow.md) 记录 |

依赖：01 M1（已闭环）、[KVCache Manager 演进方案](05-kv-cache-manager-evolution.md) 的 binding 语义（已落地部分）。

## 9. 与现有文档的关系

- [01 号提案](01-inference-session-generate-readiness.md)：定义 Generate 整体门禁；本提案是其 M2 的细化，§3.2–§3.3 缺口由本提案闭环，§3.4 phase 合同按本提案 §4.5 落地。01 §4.2 的示意签名以本提案 §3.3 为准（去掉 `ExecutableModelOptions`）。
- [05 号提案](05-kv-cache-manager-evolution.md)：KV/state 子路径细化，与本提案不重叠；`ExecutableModel` 不持有 KV 资源，KV reservation 属 Session。
- [06 号路线图](06-system-capability-evolution-roadmap.md)：§9.1 P0 "ExecutableModel preparation" 的详细设计即本提案。
- 根 `AGENTS.md` §2.1：M2.3 新增 `inference` 行，规则为"inference → execution + compiler + model + runtime；不得被上述模块反向依赖"。
- 落地后由 `docs/designs/` 承接已验证实现描述，本提案转为 Implemented。

## 10. 关联代码

- [`include/aethermind/compiler/model_compiler.h`](../../include/aethermind/compiler/model_compiler.h)
- [`include/aethermind/model/resolved_model_weights.h`](../../include/aethermind/model/resolved_model_weights.h)
- [`include/aethermind/model/raw_weight.h`](../../include/aethermind/model/raw_weight.h)
- [`include/aethermind/model/packed_weight_store.h`](../../include/aethermind/model/packed_weight_store.h)
- [`include/aethermind/model/weight_prepack_planner.h`](../../include/aethermind/model/weight_prepack_planner.h)
- [`include/aethermind/compiler/packing_request_builder.h`](../../include/aethermind/compiler/packing_request_builder.h)
- [`src/compiler/packing_request_builder.cpp`](../../src/compiler/packing_request_builder.cpp)
- [`include/aethermind/execution/execution_plan_builder.h`](../../include/aethermind/execution/execution_plan_builder.h)
- [`include/aethermind/execution/execution_bindings.h`](../../include/aethermind/execution/execution_bindings.h)
- [`src/execution/execution_bindings.cpp`](../../src/execution/execution_bindings.cpp)
- [`include/aethermind/base/tensor_view.h`](../../include/aethermind/base/tensor_view.h)

## 11. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-23 | 1.0 | 基于仓库实测事实建立 M2 细化提案：确认 packing 链路无生产调用者、role 解析私有且 tied lm-head 重复两份、binding 需求集合无公开查询、常量未被物化；给出 inference 模块归属、`PrepareExecutableModel` 流程与 M2.1–M2.5 实施步骤 |
| 2026-09-23 | 1.1 | 评审修正事实精度与规格缺口：§1 改为"无 `LoweredModelArtifact` 重载"（`Build` 另有 untrusted node-spec 重载）；§2.2 修正 packing 调用者为 4 个测试文件，并把权重绑定缺口重述为"生产侧无 `ResolvedModelWeights` → `ExternalTensorBindings` 转换"；§3.4 step 6 写明 payload 只能取自 `LoweredGraph.values()[i]`（`ExecutionValueDesc` 不含 payload）、step 7 对账排除 `kModelInput`；§4.1 补 `nullptr` 契约与 `FailedPrecondition` 转换；§4.4 补内层 vector 跨 move 稳定性论证 (b)；§4.5 裁决 phase 权威为 per-step selector、复用 `PhaseMatches`、混合 phase artifact prepare 期拒绝；§4.6 与 M2.4 记录 `execution_plan.h:84-85` 及 `artifact_id()` 归属的失实注释；M2.3 澄清 CMake 为 glob 无需改动并给出可判退出条件 |
