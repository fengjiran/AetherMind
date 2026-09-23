# ExecutableModel 生产准备入口方案

- **状态**: Implemented
- **版本**: 1.11
- **日期**: 2026-09-23
- **最近更新**: 2026-09-23
- **实现承接**: [ExecutableModel 模块设计](../designs/inference/01-executable-model.md)
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **关联计划**: [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md)（本提案是其 M2 的细化）
- **关联模块**: model / compiler / execution / runtime / inference（新增）

## 1. 结论与范围

`LoweredModelArtifact → ExecutionPlan → PreparedExecutionBindings` 的每个构件都已单独实现并有测试，但**没有任何生产代码把它们串起来**：`BuildWeightPackingRequests` 与 `WeightPrepackPlanner::PrepackAndStore` 至今只有测试调用者（本节记录提案建立时的现状，闭环见 §6 M2.4/M2.5），`ExecutionPlanBuilder::Build` 无 `LoweredModelArtifact` 重载（现有 4 个重载只接受 `LoweredGraph` 或 untrusted `ExecutionPlanNodeSpec` 列表），`ExternalTensorBindings` 全部由测试 helper 手工拼装。本提案定义唯一的生产准备入口 `PrepareExecutableModel`，并给出它所需的三处公共合同补齐。

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
| `PrepackWeightRequests` | [`weight_packing.h:86-88`](../../include/aethermind/model/weight/weight_packing.h) | 已实现；经 `Backend::PackWeights` 执行，composite 物化与对齐归 backend（原 `WeightPrepackPlanner::PrepackAndStore`，1.8 起为自由函数） |
| `PackedWeightStore` | [`weight_packing.h:112-162`](../../include/aethermind/model/weight/weight_packing.h) | 已实现；与 plan 共享 `shared_ptr` 所有权，`source_id` 首次 Store 后冻结 |
| `ExecutionPlanBuilder::Build` | [`execution_plan_builder.h:67-81`](../../include/aethermind/execution/execution_plan_builder.h) | 已实现 `LoweredGraph` 与 `(PackedWeightStore, LoweredGraph)` 重载 |
| packed/plain 端口裁剪 | [`execution_plan_builder.cpp:61-73`](../../src/execution/execution_plan_builder.cpp) | `weight_format == kPacked` 时丢弃 `kWeight` 语义端口 |
| `PrepareExecutionBindings` | [`execution_bindings.h:109-112`](../../include/aethermind/execution/execution_bindings.h) | 已实现；缺必需绑定报 `FailedPrecondition`，重复 id 报 `InvalidArgument` |

### 2.2 缺失的构件与已存在的重复实现

| 缺口 | 事实依据 | 影响 |
|---|---|---|
| 无 artifact 级准备入口（M2.4 已闭环） | `PrepareExecutableModel`/`ExecutableModel` 在仓库中只出现于 docs；`ExecutionPlanBuilder::Build` 无 `LoweredModelArtifact` 重载 | 调用方必须自己按正确顺序拼装 5 个组件 |
| packing 链路无生产调用者（M2.4 已闭环；完整模型的 packed 覆盖仍缺，见末行） | 调用者全部在测试且均为单算子图：`tests/unit/model/weight/test_weight_packing.cpp`、`test_cpu_qkv_linear_kernel.cpp:624`、`test_cpu_add_rmsnorm_kernel.cpp:729`、`test_cpu_gate_up_linear_kernel.cpp:642` | packed 路径从未在真实模型上走通 |
| 无真实权重 → external binding 的生产 API（M2.4 已闭环） | 生产侧 `ResolvedModelWeights` 只由 `ModelLoader` 解析、`LoadedModel` 持有，无任何代码把它转成 `ExternalTensorBindings`；该转换只存在于测试（`ResolvedModelWeights` 出现在 11 个测试文件 84 处，`ExternalTensorBindings` 由 `tests/unit/execution/test_execution_binding_helpers.h` 手工构造） | 01 §3.3 未闭环 |
| role → 原始权重的解析是私有的且重复两份（M2.1 已闭环） | `FindRawWeightByRole` 曾位于 [`packing_request_builder.cpp:83-135`](../../src/compiler/packing_request_builder.cpp) 匿名命名空间；tied lm-head 三元式曾在该文件与 [`llama_dense_graph_builder.cpp:301`](../../src/model/llama_dense_graph_builder.cpp)（原 `model_graph_builder.cpp`）各写一遍 | 第三份实现（plain binding 映射）会再复制一次 tied 语义 |
| external binding 需求集合无公开查询（M2.2 已闭环） | `ComputeExternalReadRequirements`（[`execution_bindings.cpp:238-264`](../../src/execution/execution_bindings.cpp)）曾在匿名命名空间内，仅 `:399` 自用 | 准备入口若不复制该逻辑，就无法保证"不重复不遗漏" |
| `ConstantValue.inline_data` 未被执行层消费（M2.4 已闭环，M2.5 补物化测试） | payload 携带 `shared_ptr<const vector<byte>>`（[`graph_types.h:226-237`](../../include/aethermind/graph/graph_types.h)），但 `PrepareExecutionBindings` 无条件要求每个 `kConstant` 提供 external 绑定 | 常量折叠产生的常量目前无人物化 |
| 无完整 Llama plan 构建证据（M2.4 已闭环） | `BuildLlamaDense` 只出现在 model/graph/compiler 测试；`OptimizeModelGraph.LowersFullLlamaDenseGraph` 止于 lowering | 01 §9 "baseline pipeline 可通过真实 CpuBackend 构建完整 plan" 未勾选 |
| packed lowering 对完整模型不可解析（M2.4 实测发现；已闭环） | `enable_packed_weights=true` 使**所有**含 `kWeight` 输入的 step 变 packed（[`graph_lowering.cpp:116`](../../src/compiler/graph_lowering.cpp)），当时只有 `QkvLinear`/`GateUpLinear`/`AddRmsNorm` 注册 packed 描述符，`Embedding` 与 `Linear` 均无 | 完整 Llama 的 packed 配置曾在 kernel resolve 即失败；`Embedding`/`RmsNorm`/`Linear` 的 packed identity descriptor 落地后已可解析，由 `ExecutableModel.PackedLoweringPreparesAllWeightConsumers` 正向覆盖（见 01 §2.3） |

### 2.3 可直接复用的既有不变量

以下事实已由代码或测试确立，本方案在其上设计，不重复论证：

- **值索引同一性**：`PrepareTrustedGraph` 按 lowered 顺序 1:1 push 值（[`execution_plan_builder.cpp:484-505`](../../src/execution/execution_plan_builder.cpp)），因此 `ExecutionValueId{index}` 与 `LoweredGraph` 的 `GraphValueId{index}` 同索引；`packed_key->value_index` 直接用于索引 `graph.values`（`:648`）即为佐证。
- **plan 构建后自持**：`ExecutionPlanBuilder.TrustedPathCopiesValueDataflowAfterLoweredGraphLifetimeEnds`（[`test_execution_plan_builder.cpp:1035`](../../tests/unit/execution/test_execution_plan_builder.cpp)）证明 plan 不借用 `LoweredGraph`。
- **packed artifact 生命周期解耦**：plan step 持 `shared_ptr<const PackedWeights>`，store 销毁后 plan 仍可执行（`weight_packing.h:112-116`）。
- **权重连续性**：HF 校验器拒绝非连续视图（[`hf_model_validator.cpp:83`](../../src/model/formats/hf/hf_model_validator.cpp)）；packing 路径另行复查（`weight_packing.cpp:128`）。
- **tied lm-head 语义**：解析结果不是标志位，而是复用同一 `RawWeightView`（共享 `storage`），由 `BuildWeightPackingRequestsFallsBackToEmbedTokensForTiedLmHead`（`test_weight_packing.cpp:708-759`）覆盖。

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
// include/aethermind/inference/executable_model.h（已实现，以此为准）
class ExecutableModel {
public:
    ExecutableModel(ExecutableModel&&) noexcept = default;
    ExecutableModel& operator=(ExecutableModel&&) = delete;
    ExecutableModel(const ExecutableModel&) = delete;
    ExecutableModel& operator=(const ExecutableModel&) = delete;

    /// 按 phase 取 plan；baseline 下三个 phase 查询返回同一不可变 plan。
    AM_NODISCARD StatusOr<const ExecutionPlan*> plan(ExecPhase phase) const noexcept;

    /// 只含 weight/constant 的不可变只读绑定；model inputs 由 Session 在
    /// prepare 时追加（见 §4.4）。
    AM_NODISCARD StatusOr<const ExternalTensorBindings*> immutable_weight_bindings(
            ExecPhase phase) const noexcept;

    AM_NODISCARD uint64_t artifact_id() const noexcept;
    AM_NODISCARD ExecPhase phase() const noexcept;
};

/// 唯一生产准备入口。artifact 按值移入并被 ExecutableModel 拥有。
AM_NODISCARD StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime,
        LoweredModelArtifact artifact);
```

相对 01 §4.2 与本提案初版轮廓的四处落地偏差：

1. **去掉 `ExecutableModelOptions`**：`enable_packed_weights` 与 `selector.phase` 已在编译期固化进 artifact 的 step selector，准备阶段无可配置项。
2. **两个 phase 访问器改为可失败**（`StatusOr<const T*>`，与 `Runtime::GetBackend`、`KVCacheView::KeyData` 同风格）：§4.5 要求 phase 不匹配时报错而非静默复用，返回引用的签名无法表达该失败。仓库无 `std::reference_wrapper` 先例。
3. **不暴露 `packed_weights()`**：Session 不需要它，packed/plain 正确性可由绑定表与 plan step 的 `packed_weights` 指针验证，保留只读访问器只会扩大公共表面。
4. **删除移动赋值**（只可移动构造）：就地替换一个已 prepare 的模型会作废此前交出的 `plan()`/`immutable_weight_bindings()` 借用指针与派生的 `PreparedExecutionBindings`；且默认的逐成员赋值按声明顺序执行，会先释放 `artifact_` 再替换借用它的 `bindings_`，与 §3.2/§4.6 的逆序销毁契约相反。换模型应构造新对象。该改动同时把 base 层 `StatusOr` 的 `is_nothrow_move_assignable_v<T>` 断言移除（保留 move-constructible 断言）：赋值运算符保持 `= default`，对不可赋值的 `T` 变为 deleted，`StatusOr<ExecutableModel>` 的返回路径只依赖 nothrow 移动构造。

### 3.4 准备流程

```text
PrepareExecutableModel(runtime, artifact)
  1. resolved = artifact.loaded_model->GetResolvedWeights()
  2. requests = BuildWeightPackingRequests(artifact.graph, resolved)     // 已有
  3. PrepackWeightRequests(*runtime.GetBackend(device), store, requests)  // 已有；经 Backend::PackWeights
  4. plan = ExecutionPlanBuilder::Build(runtime, store, artifact.graph)  // 已有
  5. required = ComputeExternalReadRequirements(plan)                    // §4.2 已公开
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
// include/aethermind/model/weight/weight_packing.h
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

一个实施期核实、并修正了本提案初版假设的不变量：`LowerModelGraph` 对**所有**含 `kWeight` 输入的 step 统一赋 `weight_format`（[`graph_lowering.cpp:110-123`](../../src/compiler/graph_lowering.cpp)），而 untrusted 路径的 `ExecutionPlanNodeSpec` 不携带输入 value id、每个节点获得独立值（[`execution_node_spec.h:24-47`](../../include/aethermind/execution/execution_node_spec.h)）。因此"同一权重同时被一个 packed step 和一个 plain step 消费"当前**不可构造**，不能作为验收前提。需求计算按 step 的 `kernel_input_ports` 逐端口判定，将来若引入 per-op packing，无需改动即正确。测试因此覆盖 plain 图、packed 图与查询—prepare 一致性三个**可达**形态。

### 4.3 常量物化

`PrepareExecutionBindings` 对每个 `kConstant` 无条件要求 external 只读绑定，而 `ConstantBinding.inline_data` 已随 payload 进入 artifact。准备入口按值 spec（dtype/shape）把 `inline_data` 包成 `TensorView`，并校验 `inline_data->size()` 与 spec 推导字节数一致。

当前 per-family graph builders 不产生常量（RoPE 表是 `RoPEParams` 内核参数），但 O2 常量折叠会（`constant_folding_pass.cpp:38-56`），因此该路径必须实现而非假设集合为空。

### 4.4 shape/stride 元数据所有权

`TensorView(data, dtype, IntArrayView shape, IntArrayView strides, alignment)` **借用** shape/stride 数组（`tensor_view.h:49-61`），而 `RawWeightView` 只有 shape、没有 stride。因此 `ExecutableModel` 必须自有每个绑定的元数据：

- `WeightBindingStorage` 为 `std::vector<Entry>`，`Entry` 内含 `std::vector<int64_t> shape, strides`；
- 元数据指针稳定性由两个**独立**机制保证，二者缺一即悬垂：(a) `ExecutableModel` 整体移动时，外层 vector 的堆缓冲指针直接移交，`Entry` 不重定位；(b) 构建期 `push_back` 触发外层扩容时 `Entry` 被 move 构造，但 `Entry` 内层 `std::vector<int64_t>` 的 `data()` 跨 move 不变（移动只转移缓冲所有权）；
- 因此禁止把 shape/stride 换成内联存储（`std::array`、small-buffer 优化、`absl::InlinedVector` 之类）：那会同时破坏 (a) 与 (b)，任何一次扩容或移动都会让已发出的 `IntArrayView` 悬垂；
- stride 按行主序紧凑推导（连续性由 §2.3 保证），`alignment` 透传实际对齐。

model inputs 不进入 `immutable_weight_bindings()`：token/position 输入由 Session 按 phase 提供。Session 复制 weight/constant 只读条目后追加自身输入条目，重复 id 由 `PrepareExecutionBindings` 的 `InvalidArgument` 校验兜底（`execution_bindings.cpp:360-363`）。

实施期核实的一处精度修正：`PrepareExecutionBindings` **不借用**外部视图的元数据——它经 `SnapshotMetadata`（[`execution_bindings.cpp:49-54`](../../src/execution/execution_bindings.cpp)，应用于 `:419-424`）把 shape/stride 深拷贝进自有的 `ConcreteTensorMetadata`，只借用 `data()`。因此堆稳定性的真实理由是：`immutable_weight_bindings()` 是一张会被反复交付的长寿命表（prefill 与 decode 各 prepare 一次），表内 `TensorView` 的元数据必须在整个模型生命周期内有效；而不是"prepared bindings 借用元数据"。另外 `alignment` 被原值透传且不做校验，而 `RawWeightView` 不携带对齐信息，故绑定统一以 0（未指定）交付；出现真实对齐需求时再按证据补，不预先推测。

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
- plan 持 packed artifact 的 `shared_ptr` → 即使 store 先销毁也可执行（以 `weight_packing.h:112-116` 与成员类型为准；`execution_plan.h:84-85` 的 "borrowed pointer / store must outlive this plan" 注释与之矛盾，属失实注释，M2.4 修正）；`ExecutableModel` 仍持有 store 以维持 `source_id` 与 artifact 的对账能力；
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

**状态（2026-09-23）**：已落地。`ResolveWeightBinding` 位于 [`weight_packing.h`](../../include/aethermind/model/weight/weight_packing.h) / [`weight_packing.cpp`](../../src/model/weight/weight_packing.cpp)（1.9 起与 request/prepack/store 合并为一个单元），`packing_request_builder.cpp` 与 `llama_dense_graph_builder.cpp`（原 `model_graph_builder.cpp`）均改为复用，tied lm-head 回退在仓库内只剩 `weight_packing.cpp` 一处。新增 [`test_weight_packing.cpp`](../../tests/unit/model/weight/test_weight_packing.cpp) 的 `WeightBindingResolver` 套件（12 例，1.10 起与其他 weight 测试合并同文件）覆盖全角色解析、tied/untied lm-head、越界与缺失 layer index、`kMoERouter`、composite 与 roleless 的 `nullptr` 路径；全量单元测试 3499 例通过。

一处刻意的语义收紧：原 `FindRawWeightByRole` 对 attention/MLP 角色使用 `layer.value_or(0)`（缺失 layer index 时静默解析到 layer 0），而 norm 角色返回 `nullptr`，两者不一致。现统一为 layer-scoped 角色一律要求 in-range index，缺失即 `nullptr`。依据是 `ModelGraph::Validate` 已拒绝缺失 layer index 的 per-layer 角色（[`graph.cpp:161-163`](../../src/graph/graph.cpp)），故该回退对任何经验证的图不可达；收紧后全量测试全绿，实测为无操作。

遗留观察（M2.5 已闭环）：`WeightPrepackPlanner::BuildRequests`（legacy，生产不调用）对 tied lm-head 是**跳过**而非回退，语义与单一权威不同。该入口已于 M2.5 连同 `MakePackedSelector` 一并删除，其旧用例改写为直接构造 `WeightPackingRequest`；原 `BuildRequests*` 三例 graph-driven 测试更名为 `BuildWeightPackingRequests*`，避免读者误认存在第三份 request 权威。`WeightPrepackPlanner` 类壳体本身已于 1.8 函数化（见 §10 `weight_packing.h`）。

提取 `ResolveWeightBinding` 到 model/weight 单元（现 `weight_packing.h`，合并前为独立文件 `weight_binding_resolver.h`），改造 `packing_request_builder.cpp` 与 `llama_dense_graph_builder.cpp`（原 `model_graph_builder.cpp`）复用之；补 tied lm-head / 越界 layer / `kMoERouter` 的单测，含 §4.1 `nullptr` 错误路径（调用方必须转 `FailedPrecondition`）。

退出条件：仓库内 tied lm-head 三元式只剩一处；`nullptr` 返回路径与调用方错误转换各有专项测试；既有测试全绿。

### M2.2 execution：公开 external binding 需求查询

**状态（2026-09-23）**：已落地。`ComputeExternalReadRequirements` 已从 `execution_bindings.cpp` 匿名命名空间移出，在 [`execution_bindings.h`](../../include/aethermind/execution/execution_bindings.h) 声明并补齐索引空间与各值类别的契约注释；`PrepareExecutionBindings` 继续调用同一实现，无逻辑副本。新增 [`test_execution_bindings.cpp`](../../tests/unit/execution/test_execution_bindings.cpp)（4 例，真实 CpuBackend + 真实 lowering）：plain 图（model input、constant、两个 plain weight 需要，activation 不需要）、两个 step 共享同一 weight（仍只贡献一条）、packed `AddRmsNorm` 图（`kernel_input_ports` 由 3 投影为 2、weight 不需要而 constant 仍需要）、以及查询与 `PrepareExecutionBindings` 的双向一致性（按查询结果精确供给则成功，任缺一条即 `FailedPrecondition`）。全量 3503 测试通过。该测试同时实证了 §2.3 的值索引同一性（lowered value id 直接用作 execution value id）。

提升 `ComputeExternalReadRequirements` 为公共 API 并补文档注释；新增查询测试覆盖 plain 图、packed 图与查询—`PrepareExecutionBindings` 一致性（混合 packed/plain 形态当前不可构造，见 §4.2）。

退出条件：`PrepareExecutionBindings` 与新公开接口共用同一实现，无逻辑副本。

### M2.3 inference：模块骨架与 ownership 登记

**状态（2026-09-23）**：已落地，范围有一处调整。新增 [`weight_binding_storage.h`](../../include/aethermind/inference/weight_binding_storage.h) / [`weight_binding_storage.cpp`](../../src/inference/weight_binding_storage.cpp)；根 `AGENTS.md` §2.1 已新增 inference 行，跨模块依赖规则已新增 `inference → execution + compiler + model + runtime`。CMake 确认无需改动：两处 `GLOB_RECURSE ... CONFIGURE_DEPENDS` 实测自动收录新目录，`AetherMind.dir/inference/weight_binding_storage.cpp.o` 与 `aethermind_unit_tests.dir/inference/test_weight_binding_storage.cpp.o` 均进入构建。新增 [`test_weight_binding_storage.cpp`](../../tests/unit/inference/test_weight_binding_storage.cpp)（8 例）覆盖视图内容、rank-1/rank-0 形状、以及扩容后与移动构造/移动赋值后已发出视图仍有效；全量 3511 测试通过。

范围调整：`ExecutableModel` 本体不在本步交付，改与 `PrepareExecutableModel` 一起在 M2.4 落地。先定义一个没有任何构造入口的类型会留下无法使用的半成品；而 `WeightBindingStorage` 契约完整、可独立测试，又正是 §4.4 悬垂风险最高的一环，适合作为本步的完整交付物。

新增 `include/aethermind/inference/` 与 `src/inference/`；定义 `WeightBindingStorage` 并写明 §4.4/§4.6 契约；`ExecutableModel` 随 M2.4 落地；更新根 `AGENTS.md` §2.1 ownership 表与跨模块依赖规则。

**CMake 无需改动**：`src/CMakeLists.txt:1-4` 与 `tests/unit/CMakeLists.txt:4-6` 均为 `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)`，新目录自动收录；`include/` 已在 `AetherMind` 的 PUBLIC 头路径上。因此本步唯一的仓库级登记动作是 `AGENTS.md`。

退出条件（已满足）：`AGENTS.md` §2.1 新增 inference 行且依赖规则与 §3.1 一致；`--target AetherMind` 与 `--target aethermind_unit_tests` 均编译通过，且新目录下的源文件确实进入构建（以生成的 `.o` 为准）。

### M2.4 inference：`PrepareExecutableModel`

**状态（2026-09-23）**：已落地。[`executable_model.h`](../../include/aethermind/inference/executable_model.h) / [`executable_model.cpp`](../../src/inference/executable_model.cpp) 按 §3.4 八步实现，成员声明顺序即销毁契约；三处失实注释已修正（`execution_plan.h` 的 `ExecutionStep` brief 与 `Create` 的 `steps` 参数说明、`weight_packing.h`（原 `packed_weight_store.h:24/53` 与 `weight_prepack_planner.h:23`））。新增 [`test_executable_model.cpp`](../../tests/unit/inference/test_executable_model.cpp)（8 例）与共享 fixture [`test_llama_checkpoint_helpers.h`](../../tests/unit/model/test_llama_checkpoint_helpers.h)（字节后备的 tiny GQA Llama，形状占位权重会被 `ValidateRawWeightView` 拒绝）。全量 3519 测试通过。

**这同时是仓库首次通过生产路径构建出完整 Llama plan**：`ModelCompiler::Compile`（O1 未融合 + 真实 CpuBackend）→ `PrepareExecutableModel`，1 层、GQA 4/2 头，12 个权重值全部自动绑定、无手工拼 plan。01 §9 的三项门禁据此可勾选。

实施期发现的 packed 缺口比 01 §2.3 描述的更宽：`enable_packed_weights=true` 会把**所有**含 `kWeight` 输入的 step 标为 packed，而当时只有 `QkvLinear`/`GateUpLinear`/`AddRmsNorm` 注册了 packed 描述符——`Embedding` 同样没有（01 §2.3 只提到 kLinear）。因此完整 Llama 的 packed 配置在 kernel resolve 阶段即以 `NOT_FOUND: op_type=Embedding, weight_format=Packed` 失败，测试 `PackedLoweringIsUnresolvableForOpsWithoutPackedKernels` 把它固化为可执行记录（断言 `kNotFound`，即失败在 kernel 解析而非权重解析）。该缺口已闭环：`Embedding`/`RmsNorm`/`Linear` 的 packed identity descriptor 落地，缺口测试被正向的 `PackedLoweringPreparesAllWeightConsumers` 取代。

按 §3.4 实现 8 步流程，含 §4.3 常量物化与第 7 步完整性对账；错误路径不泄漏半成品对象。

同批修正三处与实现不符的既有注释：

- [`weight_packing.h`](../../include/aethermind/model/weight/weight_packing.h)（原 `packed_weight_store.h:53` 与 `weight_prepack_planner.h:23`）把 `artifact_id()` 归给 `LoweredModelArtifact`，实际只定义在 `LoweredGraph`（[`lowered_graph.h:156`](../../include/aethermind/compiler/lowered_graph.h)）；`ExecutableModel::artifact_id()` 直接委托 `artifact.graph.artifact_id()`；
- [`execution_plan.h:84-85`](../../include/aethermind/execution/execution_plan.h) 称 `packed_weights` 是 "borrowed pointer into a PackedWeightStore's storage; the store must outlive this plan"，与同文件 `:91-92`（plan 自持引用，store 销毁后仍可执行）直接矛盾；实际成员类型是 `std::shared_ptr<const PackedWeights>`，应删除失实的前者。

退出条件（已满足）：从真实 `LoweredModelArtifact` 构建成功，无手工拼 plan 路径；上述注释与实现一致。

### M2.5 测试：完整 Llama 真实后端证据

**状态（2026-09-23）**：已落地。M2.4 交付了主体证据（真实 `ModelCompiler` artifact → `PrepareExecutableModel`、GQA、tied/untied lm-head 的 backing 共享、12 个权重值双向对账、移动后绑定表仍有效、缺 `loaded_model` 的错误路径）；本步补齐其余分支与两处验收缺口，同批删除 legacy request 入口：

- 多层（2 decoder layer，21 个权重 backing）不串层：`MultiLayerLlamaBindsEveryWeightToItsOwnBacking`；
- 常量物化：`MaterializesConstantFromInlineData` 走 §4.3 路径并被绑定，另有 `RejectsConstantWithoutInlineData`、`RejectsConstantWhoseInlineSizeDisagreesWithShape` 两条拒绝路径；含常量的图手工构造后经真实 `LowerModelGraph`，未经过 constant folding pass（折叠产物在 lowered 图中的形态与之一致）；
- packed 子图：`PackedSubgraphKeepsWeightOutOfBindingTable`（`AddRmsNorm`，packed 权重不进绑定表且 `step.packed_weights` 非空）；完整模型的 packed 曾不可解析，现由 `PackedLoweringPreparesAllWeightConsumers` 正向覆盖；
- teardown：`PreparedBindingsAreReleasedBeforeTheModel` 验证 `PreparedExecutionBindings` 先于 `ExecutableModel` 释放、模型不反向引用 Session 侧状态，并在 ASAN/TSAN 下运行；
- phase 合同（§7 补充判据）：`PhaseSpecificArtifactRejectsUnmatchedPhaseQueries`、`RejectsArtifactWhoseStepsMixPhases`（后者经 `LoweredGraph::Builder` 测试缝构造 lowering 今天产生不了的 mixed-phase 形态）；
- 权重解析错误路径：`RejectsWeightWithNoDenseStorageRole`（`kMoERouter`）、`RejectsWeightWhoseLayerIndexHasNoStorage`（越界 layer），消息含 value index 与 role；
- legacy 删除：`WeightPrepackPlanner::BuildRequests` 与 `MakePackedSelector` 移除，三个仅测 legacy 的用例删除，其旧夹具（`MakeTestLayer`/`MakeLlamaConfig`）一并清理；三个 graph-driven 用例更名为 `BuildWeightPackingRequests*`。

测试实测：`ExecutableModel` 20 例、`WeightPacking` 16 例、ASAN/TSAN 下 `ExecutableModel.*` 全绿；全量单元测试 3533 例通过。

退出条件（已满足）：01 §9 的 "baseline pipeline 可以通过真实 CpuBackend 构建完整 plan"、"`PrepareExecutableModel` 可从真实 `LoweredModelArtifact` 构建"、"real weights 可自动生成完整 external bindings" 三项可勾选（M2.4 已满足）；多层、常量、teardown 覆盖由本步补齐。

## 7. 验收标准

对齐 01 M2 验收项，并给出可执行判据：

| 01 M2 验收项 | 本提案判据 |
|---|---|
| 从真实 artifact 构建，不手工拼 plan | 测试只经 `ModelCompiler` 产出 artifact，不出现 `ExecutionPlanNodeSpec` 手工构造 |
| Session 不读取 compiler artifact | `ExecutableModel` 不暴露 `LoweredGraph`/`LoweredModelArtifact` 访问器 |
| tied lm-head 正确共享 backing | 断言 lm-head 与 embed_tokens 绑定 `data()` 相同，且二者 value id 不同 |
| packed/plain 不重复不遗漏 | 绑定集合与 §4.2 需求集合逐一对账；缺任一方向即测试失败 |
| 销毁顺序有明确测试或 contract | 头文件写明成员顺序契约 + ASAN/TSAN 下的 teardown 测试 |

五项均已满足：前四项由 M2.4 交付（测试只经 `ModelCompiler` 产出 artifact；`ExecutableModel` 只暴露 `plan`/`immutable_weight_bindings`/`artifact_id`/`phase`，无 `LoweredGraph` 或 `LoweredModelArtifact` 访问器；tied 与 untied 两种 checkpoint 分别断言 backing 共享与独立；绑定集合与 §4.2 需求集合双向对账）；第五项由 M2.5 补齐——头文件写明成员顺序契约，teardown 测试 `PreparedBindingsAreReleasedBeforeTheModel` 在 ASAN/TSAN 下验证绑定表先于模型释放。

补充判据（均已有实现与测试）：

- 完整性/物化失败返回可定位的 `FailedPrecondition`（含 value index），不进入执行期，且不留下半成品 `ExecutableModel`（`RejectsConstantWithoutInlineData`、`RejectsConstantWhoseInlineSizeDisagreesWithShape`）；
- 权重解析返回 `nullptr` 时转 `FailedPrecondition`，消息含 value index 与 `role=<label>, layer=<n>`（`RejectsWeightWithNoDenseStorageRole` 覆盖 `kMoERouter`，`RejectsWeightWhoseLayerIndexHasNoStorage` 覆盖越界 layer）；
- `plan(phase)` 对 phase 不匹配的 artifact 返回错误而非静默复用（`PhaseSpecificArtifactRejectsUnmatchedPhaseQueries`）；step 间 phase 不一致的 artifact 在 prepare 期即被拒绝（`RejectsArtifactWhoseStepsMixPhases`，经 `LoweredGraph::Builder` 测试缝构造 mixed 形态）；
- 元数据堆稳定性有回归测试（`SurvivesBeingMoved` 与 `test_weight_binding_storage.cpp`：移动后、以及绑定数量增长触发外层扩容后，已发出的 `TensorView.shape()`/`strides()` 仍有效）。

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
- 已验证实现描述已由 [`docs/designs/inference/01-executable-model.md`](../designs/inference/01-executable-model.md) 承接（Current），本提案转为 Implemented。

## 10. 关联代码

- [`include/aethermind/inference/executable_model.h`](../../include/aethermind/inference/executable_model.h)
- [`src/inference/executable_model.cpp`](../../src/inference/executable_model.cpp)
- [`include/aethermind/inference/weight_binding_storage.h`](../../include/aethermind/inference/weight_binding_storage.h)
- [`include/aethermind/model/weight/weight_packing.h`](../../include/aethermind/model/weight/weight_packing.h)
- [`src/model/weight/weight_packing.cpp`](../../src/model/weight/weight_packing.cpp)
- [`include/aethermind/compiler/model_compiler.h`](../../include/aethermind/compiler/model_compiler.h)
- [`include/aethermind/model/resolved_model_weights.h`](../../include/aethermind/model/resolved_model_weights.h)
- [`include/aethermind/model/raw_weight.h`](../../include/aethermind/model/raw_weight.h)
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
| 2026-09-23 | 1.2 | M2.1 落地并转为 In Progress：新增 `weight_binding_resolver.h/.cpp` 与 12 例单测，`packing_request_builder.cpp`、`model_graph_builder.cpp` 改为复用，tied lm-head 回退收敛为一处；记录 layer-scoped 角色 `value_or(0)` 回退的刻意收紧及其不可达依据；记录 legacy `WeightPrepackPlanner::BuildRequests` 的 tied 语义差异与删除计划；全量 3499 测试通过 |
| 2026-09-23 | 1.3 | M2.2 落地：`ComputeExternalReadRequirements` 提升为 execution 公共 API，`PrepareExecutionBindings` 共用同一实现；新增 `test_execution_bindings.cpp`（4 例，真实 CpuBackend），全量 3503 测试通过。修正初版的不可达验收前提——`LowerModelGraph` 对所有含权重 step 统一赋 `weight_format`、`ExecutionPlanNodeSpec` 不携带输入 value id，故"同一权重同时被 packed 与 plain step 消费"当前不可构造，测试改为覆盖 plain/packed/一致性三个可达形态（§4.2、M2.2）；§3.4 step 5 函数名与实际 API 对齐；§2.2 标注 M2.1/M2.2 已闭环 |
| 2026-09-23 | 1.4 | M2.3 落地：新增 `inference/` 模块与 `WeightBindingStorage`（含 8 例稳定性测试），根 `AGENTS.md` §2.1 新增 inference 行与依赖规则，实测确认 CMake 的 `GLOB_RECURSE` 自动收录新目录；全量 3511 测试通过。范围调整：`ExecutableModel` 本体移至 M2.4 与 `PrepareExecutableModel` 同批交付，避免留下无构造入口的半成品类型。§4.4 精度修正：`PrepareExecutionBindings` 经 `SnapshotMetadata` 深拷贝 shape/stride、只借用 `data()`，故堆稳定性的真实理由是绑定表被反复交付；`alignment` 统一以 0（未指定）交付并记录依据 |
| 2026-09-23 | 1.5 | M2.4 落地：`ExecutableModel` 与 `PrepareExecutableModel` 按 §3.4 八步实现，修正四处失实注释，新增 8 例真实 artifact 测试与字节后备 tiny GQA Llama fixture；全量 3519 测试通过，仓库首次经生产路径构建出完整 Llama plan，01 §9 三项门禁可勾选。§3.3 更新为已实现签名并记录三处落地偏差（去 options、phase 访问器改为可失败、不暴露 packed store）；§2.2 新增实测发现——packed lowering 对含 `Embedding`/`Linear` 的完整模型不可解析（比 01 §2.3 记录的更宽）；M2.5 范围据此重划 |
| 2026-09-23 | 1.6 | M2.5 落地并转 Implemented：补多层（2 层 21 权重不串层）、常量物化与两条拒绝路径、packed 子图（`AddRmsNorm`）三个 M2.5 覆盖面；补 teardown（`PreparedBindingsAreReleasedBeforeTheModel`，ASAN/TSAN 验证）、phase 合同（`PhaseSpecificArtifactRejectsUnmatchedPhaseQueries`、`RejectsArtifactWhoseStepsMixPhases` 经 `LoweredGraph::Builder` 测试缝）与权重解析错误路径（`kMoERouter`、越界 layer，消息含 value index 与 role）三处验收缺口；`ResolveWeightBinding` 失败消息在 inference 侧补 role/layer 标签。删除 legacy `WeightPrepackPlanner::BuildRequests` 与 `MakePackedSelector` 及其 3 例测试，三个 graph-driven 用例更名为 `BuildWeightPackingRequests*`。`ExecutableModel` 删除移动赋值（只可移动构造，契约由 `IsMoveConstructibleButNotAssignable` 钉住），并相应移除 base 层 `StatusOr` 的 nothrow move-assignable 断言（见 §3.3 第 4 条）。实现描述由 [docs/designs/inference/01-executable-model.md](../designs/inference/01-executable-model.md) 承接；01 §9 "Prefill/Decode phase-plan 合同已验证" 可勾选。`ExecutableModel` 20 例、`WeightPrepackPlanner` 16 例、全量 3533 例通过 |
| 2026-09-23 | 1.7 | 结构整理（与 M2 实现无关的路径维护）：权重组装三件套迁入 `include/aethermind/model/weight/` 与 `src/model/weight/`（`packed_weight_store`、`weight_binding_resolver`、`weight_prepack_planner`），与 `model/formats/hf/` 的子目录风格一致；全仓 include 与本文档 §10 关联路径同步。CMake 的 `GLOB_RECURSE ... CONFIGURE_DEPENDS` 自动收录新目录（实测新路径 `.o` 生成），全量 3533 例通过 |
| 2026-09-23 | 1.8 | 形态整理：`WeightPrepackPlanner` 静态工具类函数化——`PrepackAndStore` → 自由函数 `PrepackWeightRequests`，嵌套 `Request` → 顶层 `WeightPackingRequest`，文件 `weight_prepack_planner.{h,cpp}` → `weight_packing.{h,cpp}`，与 compiler 侧 `BuildWeightPackingRequests` 命名对称；测试文件与套件更名 `test_weight_packing.cpp` / `WeightPacking`（`PrepackWeightRequests*` 用例名同步）。类零状态单静态方法、且 `BuildRequests` 删除后名实不符，仓库同类形态已统一为自由函数（`BuildModelGraph`/`BuildLlamaDense`）。全量 3533 例通过 |
| 2026-09-23 | 1.9 | 与并行重构同步（非本提案实施）：model/weight 三件套合并为 `weight_packing.{h,cpp}`——`ResolveWeightBinding`、`WeightPackingRequest`、`PrepackWeightRequests`、`WeightArtifactKey`/`PackedWeightStore` 同址；打包执行改经 `Backend::PackWeights`（composite 物化/对齐/分配归 backend，key 的 recipe 由产物回读）；`hf_weight_resolver` 更名为 `hf_tensor_resolver` 以区分两个 resolver。本文档 §2.1/§2.2/§3.4/§4.1/§4.6/§10 的链接与行号已同步；命名与依赖红线见 [02-weight-data-concepts.md](../designs/model/02-weight-data-concepts.md)。全量 3539 例通过 |
| 2026-09-23 | 1.10 | 测试文件向库单元对齐：`test_packed_weight_store_ownership.cpp`（原在 `tests/unit/backend/`，与所测类型不同层）与 `test_weight_binding_resolver.cpp` 并入 `tests/unit/model/weight/test_weight_packing.cpp`，与单一库单元同址同层；三个套件（`WeightPacking` 17 例、`WeightBindingResolver` 12 例、`PackedWeightStoreOwnership` 5 例）共 34 例。全量 3539 例通过 |
| 2026-09-23 | 1.11 | packed 缺口闭环同步：§2.2 末行与 M2.4/M2.5 两处不再把"完整模型 packed 不可解析"记为现状——`Embedding`/`RmsNorm`/`Linear` 的 packed identity descriptor 已落地，缺口测试 `PackedLoweringIsUnresolvableForOpsWithoutPackedKernels` 被正向的 `PackedLoweringPreparesAllWeightConsumers` 取代；同时 recipe 传递链（`KernelDef::packing_recipe` → `Backend::GetPackingRecipe` → `WeightPackingRequest::recipe` → `PackWeights(..., recipe)`）与 bpanel 打包/消费链已落地，详见 [GEMM 提案](../operators/gemm/cpu-gemm-packed-weight.md) 与 01 §2.3 |
