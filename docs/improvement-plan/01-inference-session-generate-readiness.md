# InferenceSession / Generate 前置闭环计划

- **状态**: In Progress
- **版本**: 1.8
- **日期**: 2026-09-03
- **最近更新**: 2026-09-23
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **关联模块**: compiler / execution / runtime / backend / model / API orchestration

## 1. 结论与范围

当前 `Runtime → ExecutionPlan → PreparedExecutionBindings → ExecutionContext → Executor` 生命周期和执行边界已经足够稳定，可以开始闭环 `InferenceSession` 的前置模块；但尚不具备直接实现并对外宣称完整 `InferenceSession::Generate` 的条件。

本计划采用以下原则：

1. 先证明真实 Llama Prefill→Decode 执行链，再增加 public Session facade。
2. 不以 fake backend、空 kernel 或只改变状态字段的 placeholder 冒充 Generate 实现。
3. 先完成 FP32 reference baseline，再增加 fusion、packing、quantization 和 SIMD 优化。
4. Session 只负责编排，不承担 compiler、weight materialization、kernel resolve 或算子语义。
5. 当前产品保持同步、单请求、Token IDs 边界，不引入 scheduler、continuous batching 或 paged KV cache。

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

真实 CPU registry 当前有（截至 2026-09-17，共 15 类、21 个描述符；reference 命名统一为 `cpu::<op>_f32_reference`）：

| OpType | Reference kernel | Optimized kernel | Generate baseline 状态 |
|---|---:|---:|---|
| Embedding | FP32 reference | 无 | 可用 |
| RMSNorm | FP32 reference | AVX2+FMA | 可用 |
| Add | FP32/FP64/BF16/I32/I64 reference | 无 | FP32 可用 |
| ElementwiseMul | FP32 reference | 无 | semantic Llama baseline 不直接依赖 |
| Linear | FP32 reference | 无 | 可用 |
| RoPE | FP32 reference | 无 | 可用 |
| KVCacheUpdate | FP32 reference（窄 KV binding） | 无 | 可用 |
| Attention | FP32 reference（kv_read 读绑定） | 无 | 可用 |
| Silu | FP32 reference | 无 | semantic Llama baseline 不直接依赖（SiluMul 未融合对偶） |
| SiluMul | FP32 reference | 无 | 可用 |
| Argmax | FP32 reference | 无 | 可用 |
| QkvLinear | FP32 reference（packed-only） | 无 | 可用（需 O2 融合 + `enable_packed_weights=true`） |
| GateUpLinear | FP32 reference（packed-only） | 无 | 可用（需 O2 融合 + `enable_packed_weights=true`） |
| AddRmsNorm | FP32 reference（plain + packed identity） | 无 | 可用（O2 fused path；packed 需 `enable_packed_weights=true`） |

当前 O2 默认 semantic pipeline 会产生 `QkvLinear`、`GateUpLinear` 和 `AddRmsNorm`。三者的 packed kernel 与 execution packed 绑定链路（`ExecutionStep.packed_weights` → packing request → `WeightPrepackPlanner` → plan build → execute）均已落地并走通全链路测试；AddRmsNorm 另外保留 plain FP32 reference descriptor。execution lowering 仍是一个 semantic node 对应一个 kernel step，且不存在 kernel-sequence fallback；**baseline 全链路 kernel 已全部齐备（15 类 21 描述符全部可用）**，剩余准入项为 ExecutableModel 入口、真实权重绑定与端到端证据（见 §3.2–§3.5）。

### 2.3 当前 packed-weight 能力

已经具备：

- binding-aware `WeightArtifactKey`；
- graph-driven packing request；
- direct/QKV/Gate-Up composite weight materialization；
- QkvLinear/GateUpLinear packed-only reference kernel（cpu_identity 契约：logical shape/recipe/alignment 校验、行切分、与输出的 disjoint 校验）；
- `RawWeightView` byte-size 验证；
- tied lm-head fallback；
- plain-step filtering和 exact recipe lookup。

仍未具备：

- kLinear 的 kPacked 变体（unfused packed 路径）；
- kEmbedding 的 kPacked 变体。由于 lowering 会把**所有**含 `kWeight` 输入的 step 标为 packed（`graph_lowering.cpp:116-122`），缺这两者意味着 `enable_packed_weights=true` 的完整 Llama 在 kernel resolve 阶段即失败（`NOT_FOUND: op_type=Embedding, weight_format=Packed`）；该结论由 `ExecutableModel.PackedLoweringIsUnresolvableForOpsWithoutPackedKernels` 固化，细节见 [07 号提案](07-executable-model-preparation.md) §2.2；
- 实际 tile/block packing recipe（当前 `cpu_identity` 是逻辑行主序拷贝）；
- `enable_packed_weights=true` 的 unfused e2e 数值验证。

当前 `CpuWeightPrepacker` 是 `cpu_identity` copy。QkvLinear/GateUpLinear 的 packed 契约已被全链路数值测试覆盖，但它仍不能作为生产 packed compute（tile/block 重排）已就绪的证据。

## 3. 必须先闭环的阻塞项

### 3.1 KV/state identity 到达 kernel ✅（2026-09-16 闭环）

KVCache Manager 的 correctness 修复、lease/append transaction、execution binding 与长期 Paged KV 边界由 [KVCache Manager 演进方案](05-kv-cache-manager-evolution.md) 详细定义；本节只保留 Generate 闭环所需的集成门禁。

该数据链已落地（commits 66df0256..24c60799；query interval 扩展 5afa877c/299fb93d）：

```text
LoweredGraph StateBinding
    → ExecutionKVCacheStateIdentity（execution 自有的 layer/slot identity，进入 ExecutionPlan / ExecutionStep）
    → LayerRunner 组装窄绑定（KVCacheAppendBinding / KVCacheReadBinding）
    → KernelContext::kv_append / kv_read（仅单次同步调用有效）
    → KVCacheUpdate / Attention kernel（均已消费：`kv_append` / `kv_read`）
```

- [kv_cache_binding.h](../../include/aethermind/base/kv_cache_binding.h)：`KVCacheLayerStorageBinding`（key/value 指针、dtype、layer、kv_heads、head_dim、capacity、strides）+ `KVCacheAppendBinding`（本 plan 写入窗口 `[begin, end)`，`end` 为本地可见前沿）+ `KVCacheReadBinding`（`[0, committed_end)` 跨 plan 可见，`[committed_end, visible_end)` 仅同一 plan 内后续已验证 step 可读；`query_begin/query_end` 为 attention query 行的绝对位置区间，由 LayerRunner 填充）。
- commit 事务：`LayerRunner` 只在完整 plan 成功后推进 commit watermark；plan 级 append/commit 语义由 `KVCacheUpdateKernel.*` 覆盖、read 侧 query interval 与几何由 `CPUKernelAttention.*` 覆盖；manager 级前沿与越界由 `test_kv_cache_manager.cpp` 覆盖。
- kernel 侧只接收窄绑定：KV 存储指针与窗口经 `KernelContext` 逐调用传入（`KVCacheUpdateF32KernelArgs` 刻意不携带 cache 指针与位置），不进 prepared params。

硬性约束（已落实）：

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

当前产品的 reference baseline 允许先使用一个 `kBoth` plan：

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

**状态（2026-09-16）**：交付内容已全部落地（见 §3.1 闭环记录）；plan 级 binding/commit 语义由 `KVCacheUpdateKernel.*` 覆盖，manager 级 view/越界由 `test_kv_cache_manager.cpp` 覆盖。

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

模块归属、接口轮廓、准备流程与实施子步骤由 [ExecutableModel 生产准备入口方案](07-executable-model-preparation.md) 详细定义；本节只保留 Generate 闭环所需的交付边界与验收口径。

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

建议顺序（截至 2026-09-17 已完成 6/6；fused 变体 kQkvLinear/kGateUpLinear（packed-only）与 kAddRmsNorm（plain + packed identity）亦已提前落地，见 §2.2）：

1. Linear；✅ 已完成（`cpu::linear_f32_reference`）
2. RoPE；✅ 已完成（`cpu::rope_f32_reference`，含参数化 HF golden 对拍）
3. KVCacheUpdate（与 M1 联合）；✅ 已完成（`cpu::kvcache_update_f32_reference` + 窄 KV binding 链路）
4. causal GQA Attention；✅ 已完成（`cpu::attention_f32_reference`；`KVCacheReadBinding` query interval 语义）
5. SiluMul；✅ 已完成（`cpu::silu_mul_f32_reference`；kSilu 对偶 `cpu::silu_f32_reference` 同步落地）
6. Argmax。✅ 已完成（`cpu::argmax_f32_reference`）

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

- [x] state binding identity 从 LoweredGraph 到达 kernel（`ExecutionKVCacheStateIdentity` 进入 plan，窄绑定经 `KernelContext` 到达 kernel，见 §3.1）；
- [x] kernel 获得窄 KV binding，不依赖 Runtime/Session 宽对象（`KVCacheAppendBinding`/`KVCacheReadBinding` 逐调用传入）；
- [x] baseline pipeline 可以通过真实 CpuBackend 构建完整 plan（O1 未融合 tiny GQA Llama 经 `ModelCompiler` → `PrepareExecutableModel`，见 [07 号提案](07-executable-model-preparation.md) M2.4）；
- [x] Linear/RoPE/KVCacheUpdate/Attention/SiluMul/Argmax reference kernel 可用（6/6 全部可用）；fused QkvLinear/GateUpLinear/AddRmsNorm 亦已落地；
- [x] `PrepareExecutableModel` 可从真实 `LoweredModelArtifact` 构建（`inference/executable_model.h`，07 号提案 M2.4）；
- [x] real weights 可自动生成完整 external bindings（12 个权重值自动绑定并与需求集合双向对账；packed 路径受 §2.3 缺口限制，只能以子图取证）；
- [x] Prefill/Decode phase-plan 合同已验证（07 号提案 §4.5：`kBoth` artifact 三种 phase 查询共享同一 plan、单 phase artifact 拒绝不匹配查询、step 间 phase 不一致在 prepare 期拒绝，由 M2.5 测试覆盖）；
- [ ] tiny Llama Prefill + 2 Decode 数值测试通过；
- [ ] KV content 与 commit position 测试通过；
- [ ] Decode 重复执行不重新调用 `PrepareExecutionBindings`；
- [ ] Decode malloc-hook 稳态零分配测试通过；
- [ ] 错误路径释放 KV reservation，borrowed resource teardown 顺序正确。

## 10. 关联代码

- [`include/aethermind/runtime/runtime.h`](../../include/aethermind/runtime/runtime.h)
- [`include/aethermind/execution/execution_plan.h`](../../include/aethermind/execution/execution_plan.h)
- [`include/aethermind/base/kv_cache_binding.h`](../../include/aethermind/base/kv_cache_binding.h)
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
| 2026-09-15 | 1.1 | 同步 §2.2 kernel 覆盖表：SiluMul 升级为 FP32 reference/可用并新增 Silu 行，统一 reference 命名；M3 标注完成 4/6；门禁清单同步进度 |
| 2026-09-16 | 1.2 | 同步 §2.2/§2.3：新增 QkvLinear/GateUpLinear（packed-only, cpu_identity）行与 packed 能力说明，O2 fused 阻塞项收敛为 AddRmsNorm；M3 与门禁清单同步 |
| 2026-09-16 | 1.3 | AddRmsNorm（plain + packed identity）落表；PRD 链接与术语（当前产品口径）更新 |
| 2026-09-16 | 1.4 | 同步 KVCacheUpdate 与窄 KV binding 链路：§2.2 覆盖表（14 类 20 描述符）、§3.1 闭环重写、M1 状态、M3 5/6、门禁前两项勾选 |
| 2026-09-17 | 1.5 | 同步 Attention kernel 与 read binding query interval：§2.2（15 类 21 描述符）、§3.1、M3 6/6、门禁 kernel 项勾选 |
| 2026-09-23 | 1.6 | 按 §9 流转规则（M1 已闭环）将状态由 Draft 转为 In Progress；复核确认 §2.2 描述符计数（15 类 21 个）与 §3.2–§3.5 缺口描述仍与仓库一致：`ExecutableModel`/`PrepareExecutableModel`、`InferenceSession`、真实权重 external binding 生产 API、完整 Llama plan 构建与 Prefill→Decode 端到端测试均未落地，§9 其余 9 项门禁保持未勾选；M2 细化拆出为 [07 号提案](07-executable-model-preparation.md) |
| 2026-09-23 | 1.7 | 07 号提案 M2.4 落地后同步：§9 勾选 "baseline pipeline 可通过真实 CpuBackend 构建完整 plan"、"`PrepareExecutableModel` 可从真实 artifact 构建"、"real weights 可自动生成完整 external bindings" 三项；§2.3 补记 kEmbedding 亦无 kPacked 变体，并写明其后果——`enable_packed_weights=true` 的完整 Llama 在 kernel resolve 即失败，packed 取证只能走子图 |
| 2026-09-23 | 1.8 | 07 号提案 M2.5 落地后同步：§9 勾选 "Prefill/Decode phase-plan 合同已验证"（共享单 plan、phase 不匹配报错、混合 phase prepare 期拒绝三项由 M2.5 测试覆盖，见 07 §4.5/§7）；07 转 Implemented，实现描述由 [designs/inference/01-executable-model.md](../designs/inference/01-executable-model.md) 承接。剩余五项门禁（Prefill/Decode 数值、KV content/commit、重复 decode、malloc-hook、KV reservation teardown）属 M4/M5，未勾选 |
