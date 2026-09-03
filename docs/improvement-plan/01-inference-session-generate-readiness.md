# InferenceSession / Generate 前置闭环计划

- **状态**: Draft
- **版本**: 1.0
- **日期**: 2026-09-03
- **产品边界**: [AetherMind Phase 1 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **关联模块**: compiler / execution / runtime / backend / model / API orchestration

## 1. 结论与范围

当前 `Runtime → ExecutionPlan → PreparedExecutionBindings → ExecutionContext → Executor` 生命周期和执行边界已经足够稳定，可以开始闭环 `InferenceSession` 的前置模块；但尚不具备直接实现并对外宣称完整 `InferenceSession::Generate` 的条件。

本计划采用以下原则：

1. 先证明真实 Llama Prefill→Decode 执行链，再增加 public Session facade。
2. 不以 fake backend、空 kernel 或只改变状态字段的 placeholder 冒充 Generate 实现。
3. 先完成 FP32 reference baseline，再增加 fusion、packing、quantization 和 SIMD 优化。
4. Session 只负责编排，不承担 compiler、weight materialization、kernel resolve 或算子语义。
5. Phase 1 保持同步、单请求、Token IDs 边界，不引入 scheduler、continuous batching 或 paged KV cache。

### 1.1 本计划包含

- execution-native KV/state binding；
- `LoweredModelArtifact → ExecutableModel` 的生产准备入口；
- Prefill/Decode plan 选择合同；
- 最小 FP32 Llama reference kernel 链；
- tiny Llama 端到端数值验证；
- `InferenceSession::Generate` 同步编排；
- Decode 稳态零分配验证。

### 1.2 本计划不包含

- HTTP/gRPC 服务；
- tokenizer 或字符串输入输出；
- request scheduler、batching、continuous batching；
- PagedAttention、动态 KV 扩容；
- temperature/top-k/top-p sampling；
- GPU/CUDA/CANN execution；
- 为尚未证明的性能需求预先引入 Primitive IR。

## 2. 已验证的当前状态

### 2.1 已闭环基础设施

| 能力 | 当前状态 | 事实依据 |
|---|---|---|
| Runtime 资源所有权 | 已实现 | `Runtime` 持有 allocator/backend/KVCacheManager |
| 不可变执行计划 | 已实现 | `ExecutionPlanBuilder` resolve kernel、绑定 packed artifact、规划 workspace |
| tensor specialization | 已实现 | `PrepareExecutionBindings` 校验 concrete binding、分配 activation、准备 kernel params |
| 窄执行上下文 | 已实现 | `ExecutionContext` 拥有 prepared bindings，借用 workspace，保存 KV view |
| 执行热路径 | 已实现 | `Executor → LayerRunner → InvokePreparedKernel`，无 registry lookup 和 params rebuild |
| KV storage/session view | 部分实现 | `KVCacheManager` 支持 reserve/reset/release；`KVCacheView` 支持 generation 检查和 commit watermark |
| semantic Llama graph | 已实现 | `ModelGraphBuilder::BuildLlamaDense` 生成完整 decoder-only semantic graph |
| compiler/lowering | 已实现 | `ModelCompiler` 生成结构验证过的 `LoweredModelArtifact` |

### 2.2 当前 CPU kernel 覆盖

真实 CPU registry 当前只有：

| OpType | Reference kernel | Optimized kernel | Generate baseline 状态 |
|---|---:|---:|---|
| Embedding | FP32 scalar | 无 | 可用 |
| RMSNorm | FP32 scalar | AVX2+FMA | 可用 |
| Add | FP32/FP64/BF16/I32/I64 scalar | 无 | FP32 可用 |
| ElementwiseMul | FP32 scalar | 无 | semantic Llama baseline 不直接依赖 |
| Linear | 无 | 无 | 阻塞 |
| RoPE | 无 | 无 | 阻塞 |
| KVCacheUpdate | 无 | 无 | 阻塞 |
| Attention | 无 | 无 | 阻塞 |
| SiluMul | 无 | 无 | 阻塞 |
| Argmax | 无 | 无 | 阻塞 |
| QkvLinear / GateUpLinear / AddRmsNorm | 无 | 无 | O2 fused path 阻塞 |

当前 O2 默认 semantic pipeline 会产生 `QkvLinear`、`GateUpLinear` 和 `AddRmsNorm`，但 execution lowering 仍是一个 semantic node 对应一个 kernel step，且不存在 kernel-sequence fallback。因此 semantic compilation 成功不等于真实 CPU plan 可构建。

### 2.3 当前 packed-weight 能力

已经具备：

- binding-aware `WeightArtifactKey`；
- graph-driven packing request；
- direct/QKV/Gate-Up composite weight materialization；
- `RawWeightView` byte-size 验证；
- tied lm-head fallback；
- plain-step filtering和 exact recipe lookup。

仍未具备：

- 真实 packed/quantized Linear kernel；
- 实际 tile/block packing recipe；
- packed Linear 数值端到端验证。

当前 `CpuWeightPrepacker` 是 `cpu_identity` copy。它可验证 artifact identity/lifetime，但不能作为生产 packed compute 已就绪的证据。

## 3. 必须先闭环的阻塞项

### 3.1 KV/state identity 没有到达 kernel

当前 `ExecutionContext` 保存 `KVCacheView`，`LayerRunner` 只验证 state alias 的 presence、dtype 和静态 geometry。`KernelContext` 不携带 KV storage、layer、K/V slot 或 commit position。

同时，`LoweredGraph` 中 `StateValue::binding` 所携带的：

```text
KVCacheStateBinding {
    decoder_layer_index,
    KVCacheSlot
}
```

在转换为 `ExecutionValueDesc` 时只保留了 `kind == kState`，资源 identity 没有进入 `ExecutionPlan`。因此真实 `KVCacheUpdate`/`Attention` kernel 无法确定物理 KV 位置。

必须建立以下数据链：

```text
LoweredGraph StateBinding
    → execution-native resolved state identity
    → ExecutionPlan / ExecutionStep
    → LayerRunner narrow binding
    → KernelContext
    → KVCacheUpdate / Attention kernel
```

硬性约束：

- 不得把 `Runtime*`、`ExecutionContext*` 或整个 `KVCacheManager*` 传给 kernel；
- backend 不得依赖 graph/compiler；
- kernel 只获得当前 step 必需的窄 KV binding；
- layer/slot/position/capacity 必须有唯一权威来源；
- KV commit watermark 只在完整 plan 成功后推进。

### 3.2 缺少 ExecutableModel 准备入口

当前以下组件彼此独立：

```text
ModelCompiler
BuildWeightPackingRequests
WeightPrepackPlanner
ExecutionPlanBuilder
PrepareExecutionBindings
```

缺少一个生产入口统一完成：

```text
LoweredModelArtifact
    → graph-driven weight materialization/prepack
    → plan build
    → immutable weight/constant external binding map
    → phase plan selection
    → ExecutableModel
```

`InferenceSession` 不得直接扫描 `LoweredGraph`、根据 debug name 查找权重或理解 `TransformerWeightRole`。这些职责属于模型准备阶段。

### 3.3 缺少真实 external weight binding 构造

`PrepareExecutionBindings` 要求 model inputs、plain weights 和 constants 提供 `ExternalTensorBindings`。目前测试手工构造这些 views，但没有从 `LoadedModel::resolved_weights` 按 `ExecutionValueId` 生成 immutable binding map 的生产 API。

该映射必须：

- 基于结构化 weight identity，不依赖字符串；
- 保证 backing storage 比 prepared bindings 活得久；
- 正确处理 tied lm-head；
- 区分 plain weight TensorView 和 packed artifact；
- 在 model preparation 阶段完成完整性验证。

### 3.4 Prefill/Decode plan 合同未冻结

当前 lowering 支持 `ExecPhase::{kPrefill, kDecode, kBoth}`，但一次 `ModelCompiler::Compile` 只返回一个 `LoweredGraph`。

Phase 1 reference baseline 允许先使用一个 `kBoth` plan：

- semantic topology 相同；
- prefill/decode 分别构建不同的 `PreparedExecutionBindings`；
- reference kernel 根据 concrete binding/token count 执行正确语义；
- 两阶段不能同时执行，符合同步单请求边界。

只有出现以下真实需求时才拆成独立 plan：

- prefill/decode 选择不同 kernel；
- workspace requirement 不同；
- physical topology 或 state/resource use 不同；
- phase-specific layout/packing 产生可验证收益。

对外 `ExecutableModel` 应提供按 phase 获取 plan 的接口，并允许 prefill/decode 在内部共享同一不可变 plan，避免把当前 baseline 实现固化为长期限制。

### 3.5 缺少端到端数值与稳态证据

在实现 public Session 前必须存在真实 CPU backend 测试，而不是 fake kernel call count：

- tiny one-layer Llama Prefill；
- 至少两个 Decode step；
- logits/token 与可信 reference 对比；
- KV key/value 内容与 commit position 验证；
- tied lm-head；
- GQA；
- 相同输入重复执行的确定性；
- Decode loop malloc/free 次数为零。

## 4. 目标架构

### 4.1 对象与生命周期

```text
Runtime                                      最长生命周期
├── AllocatorRegistry
├── BackendRegistry
└── KVCacheManager

ExecutableModel                              模型生命周期
├── owns LoweredModelArtifact / weight backing
├── owns packed artifacts
├── owns immutable external weight bindings
└── owns/shares phase ExecutionPlans

InferenceSession                             Generate/session 生命周期
├── borrows Runtime
├── owns/shares ExecutableModel
├── owns workspace backing + arena adapter
├── owns KV reservation view
├── owns GenerationState
└── owns one active ExecutionContext

ExecutionContext                             plan specialization 生命周期
├── owns PreparedExecutionBindings
├── borrows WorkspaceArena
└── holds KVCacheView
```

生命周期 invariant：

```text
Runtime
  > ExecutableModel / ExecutionPlan
  > InferenceSession
  > ExecutionContext
  > single Executor::Execute
```

### 4.2 推荐接口轮廓

以下为目标职责示意，不是已冻结 public API：

```cpp
class ExecutableModel {
public:
    const ExecutionPlan& plan(ExecPhase phase) const;
    const ExternalTensorBindings& immutable_weight_bindings(ExecPhase phase) const;
};

StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime,
        LoweredModelArtifact artifact,
        const ExecutableModelOptions& options);

class InferenceSession {
public:
    static StatusOr<InferenceSession> Create(
            Runtime& runtime,
            std::shared_ptr<const ExecutableModel> model);

    StatusOr<std::vector<uint32_t>> Generate(
            std::span<const uint32_t> prompt_tokens,
            const GenerationConfig& config);
};
```

`ExecutableModel` 是否放入新 `inference/` 模块、API 层或现有 model/execution 之上的 orchestration 层，需要在落地 M2 前更新根 `AGENTS.md` 模块 ownership 表。不得将它放入 `runtime`，因为 runtime 禁止依赖 compiler/execution/model。

## 5. 方案与备选

### 5.1 推荐：先执行闭环，后 Session facade

```text
state binding
  → ExecutableModel
  → reference kernels
  → direct end-to-end Executor test
  → InferenceSession
```

优点：

- Session API 建立在已验证合同上；
- kernel/state/weight 问题不会泄漏进 orchestration；
- 每个里程碑都能独立提供 correctness 证据；
- 避免后续重写 public lifecycle。

### 5.2 否决：先实现 Session shell

先实现 `Session::Generate`，内部使用 fake backend、空 kernel 或固定 token，可以快速得到 API 形状，但无法证明任何 Llama execution 语义。它还会迫使 Session 临时承担 weight mapping、phase plan 和 KV state 解释，形成错误边界。

结论：不采用。

### 5.3 否决：先完成所有 fused/quantized kernel

等待 QKV/Gate-Up/AddRmsNorm、INT8/INT4 和所有 SIMD 路径完成后再闭环 Generate，会把 correctness baseline 与性能优化绑在一起，显著扩大调试空间。

结论：不采用；先使用 O1/unfused FP32 reference baseline。

## 6. 实施步骤

### M1：execution-native KV/state binding

#### 交付内容

- execution-native state identity；
- lowering artifact 到 execution state identity 的受控转换；
- per-step narrow KV binding；
- `KernelContext` 可访问本 step 所需的 KV slice/position；
- `KVCacheUpdate` reference kernel；
- state binding、过期 view、越界 position、错误 layer/slot 测试。

#### 验收标准

- 不通过 graph name 或 step index 猜测 layer/slot；
- kernel 不接收宽 Runtime/Session 对象；
- KV 写入在 commit 前不可读；
- plan 失败不推进 commit watermark；
- TSAN 聚焦测试通过。

### M2：ExecutableModel preparation

#### 交付内容

- `PrepareExecutableModel`；
- graph-driven raw/packed weight materialization；
- immutable `ExecutionValueId → TensorView` weight/constant binding；
- phase plan 获取接口；
- artifact/runtime/plan/binding lifetime 文档和测试。

#### 验收标准

- 从真实 `LoweredModelArtifact` 构建，不手工拼 plan；
- Session 不读取 compiler artifact；
- tied lm-head 正确共享 backing；
- packed/plain 路径不会重复或遗漏 binding；
- model/runtime 销毁顺序有明确测试或 contract。

### M3：最小 FP32 reference kernel 链

建议顺序：

1. Linear；
2. RoPE；
3. KVCacheUpdate（与 M1 联合）；
4. causal GQA Attention；
5. SiluMul；
6. Argmax。

Baseline 使用 O1/unfused graph。每个 kernel 必须走唯一生产路径：

```text
CpuBackend::PrepareKernel
  → KernelRegistry
  → PrepareExecutionBindings params builder
  → InvokePreparedKernel
  → typed reference compute
```

#### 验收标准

- 每个 kernel 有 shape/layout/alias/overflow 测试；
- 至少一个 vector-tail 或非整块边界；
- 与独立 scalar/reference 公式比较；
- 不通过测试 helper 绕过 registry/entry 路径。

### M4：direct Prefill→Decode execution proof

不经过 Session facade，直接使用 `ExecutableModel`、`ExecutionContext` 和 `Executor`：

```text
prepare prefill bindings
  → Execute
  → commit prompt KV
  → read first token
  → replace with decode bindings once
  → Execute decode #1
  → commit
  → Execute decode #2
  → commit
```

#### 必测配置

- 1 decoder layer；
- prompt length > 1；
- GQA；
- tied lm-head；
- 两个 decode token；
- deterministic repeat；
- logits、tokens、KV content、commit position 与 reference 一致。

### M5：InferenceSession / Generate

在 M4 通过后实现同步 Session orchestration：

1. validate prompt/config；
2. reserve KV session；
3. allocate/reuse workspace；
4. prepare and execute prefill；
5. commit KV；
6. read Argmax token；
7. prepare decode bindings once；
8. run decode loop；
9. EOS/max-token stop；
10. clear context before releasing KV reservation；
11. return token IDs。

Session 必须使用 RAII 管理 KV reservation 和失败路径。`ExecutionContext::Clear()` 不代替 `KVCacheManager::ReleaseSession()`，也不 reset borrowed workspace。

## 7. Generate 执行顺序与 invariant

### 7.1 Prefill

```text
prompt token IDs + position IDs
  → PrepareExecutionBindings(prefill specialization)
  → ExecutionContext::Create
  → workspace reset by Session owner
  → Executor::Execute
  → success: KVCacheView::CommitUntil(prompt_len)
  → read first output token
```

### 7.2 Decode steady state

```text
PrepareExecutionBindings(decode specialization)   // exactly once

loop:
  write previous token into stable input buffer
  write/update position ID
  reset workspace through Session owner
  Executor::Execute
  commit one KV position on success
  read Argmax output
  evaluate EOS/max_tokens
```

Decode 循环中允许变化：

- input/output buffer 内容；
- KV backing 内容与 commit position；
- generated token count。

Decode 循环中不得变化：

- plan identity；
- input/output addresses；
- tensor shape/stride/dtype；
- prepared kernel params；
- workspace backing address；
- packed-weight artifact identity。

## 8. 风险与依赖

| 风险 | 影响 | 处理原则 |
|---|---|---|
| state identity 过早下沉为 runtime 类型 | backend/runtime 形成反向依赖 | 使用 execution/base 层窄纯数据合同 |
| Session 承担 weight mapping | 生命周期与职责混乱 | 收敛到 ExecutableModel preparation |
| 默认 O2 产生无 kernel fused op | plan build 失败 | baseline 明确使用 O1；fusion 在真实 kernel/fallback 完成后启用 |
| prefill/decode 过早拆双 plan | artifact/packing identity 复杂化 | reference baseline 允许共享 kBoth plan |
| 单 plan 阻碍后续 phase optimization | 性能演进受限 | ExecutableModel 对外按 phase 查询，内部可共享或分离 |
| KV 写入失败后状态不一致 | correctness | 完整 plan 成功后才推进 commit watermark |
| fake backend 测试被误认为生产证据 | 错误成熟度判断 | readiness gate 要求真实 CPU 数值测试 |
| Decode 隐式分配 | 延迟抖动 | malloc hook + 重复执行验收 |

## 9. Public Session 实现门禁

本提案在 M1 开始实施时从 Draft 转为 In Progress。以下条件全部满足后，才允许进入 M5 并开始实现 public `InferenceSession::Generate`：

- [ ] state binding identity 从 LoweredGraph 到达 kernel；
- [ ] kernel 获得窄 KV binding，不依赖 Runtime/Session 宽对象；
- [ ] baseline pipeline 可以通过真实 CpuBackend 构建完整 plan；
- [ ] Linear/RoPE/KVCacheUpdate/Attention/SiluMul/Argmax reference kernel 可用；
- [ ] `PrepareExecutableModel` 可从真实 `LoweredModelArtifact` 构建；
- [ ] real weights 可自动生成完整 external bindings；
- [ ] Prefill/Decode phase-plan 合同已验证；
- [ ] tiny Llama Prefill + 2 Decode 数值测试通过；
- [ ] KV content 与 commit position 测试通过；
- [ ] Decode 重复执行不重新调用 `PrepareExecutionBindings`；
- [ ] Decode malloc-hook 稳态零分配测试通过；
- [ ] 错误路径释放 KV reservation，borrowed resource teardown 顺序正确。

## 10. 关联代码

- [`include/aethermind/runtime/runtime.h`](../../include/aethermind/runtime/runtime.h)
- [`include/aethermind/execution/execution_plan.h`](../../include/aethermind/execution/execution_plan.h)
- [`include/aethermind/execution/execution_bindings.h`](../../include/aethermind/execution/execution_bindings.h)
- [`include/aethermind/execution/execution_context.h`](../../include/aethermind/execution/execution_context.h)
- [`include/aethermind/backend/kernel_context.h`](../../include/aethermind/backend/kernel_context.h)
- [`src/execution/layer_runner.cpp`](../../src/execution/layer_runner.cpp)
- [`src/model/model_graph_builder.cpp`](../../src/model/model_graph_builder.cpp)
- [`src/compiler/optimize_graph.cpp`](../../src/compiler/optimize_graph.cpp)
- [`src/backend/cpu/kernels/`](../../src/backend/cpu/kernels/)

## 11. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-03 | 1.0 | 基于当前 Runtime/Execution 生命周期与真实 CPU kernel 覆盖建立前置闭环计划 |
