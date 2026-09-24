# InferenceSession / Generate 前置闭环计划

- **状态**: Implemented
- **版本**: 1.12
- **日期**: 2026-09-03
- **最近更新**: 2026-09-24
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **关联模块**: compiler / execution / runtime / backend / model / API orchestration

## 1. 结论与范围

M1–M4 已闭环，M5 已提供基于真实 `CpuBackend` 的同步 `InferenceSession::Generate`。当前范围为 Llama FP32 greedy Argmax、token IDs 输入输出和静态 KV reservation；C ABI、sampling 与 scheduler 不属于本提案交付。

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
| KV storage/session view | 已实现（静态 contiguous baseline） | `KVCacheManager` 支持 reserve/reset/release；`KVCacheView` 支持 generation 检查和 commit watermark，watermark 由 `Executor` 在完整 plan 成功后自动推进；M4/M5 端到端复用该路径。KV owner/epoch 修复与 Paged KV 长期准入见 [08 号演进提案](08-kv-cache-and-attention-capability-evolution.md)（Draft，Paged KV 非当前承诺） |
| semantic Llama graph | 已实现 | `BuildModelGraph` 家族分发到 `BuildLlamaDense`，生成完整 decoder-only semantic graph |
| compiler/lowering | 已实现 | `ModelCompiler` 生成结构验证过的 `LoweredModelArtifact` |

### 2.2 当前 CPU kernel 覆盖

真实 CPU registry 当前有（截至 2026-09-24，共 15 类、27 个描述符，其中 9 个声明 `weight_format = kPacked`——含 QkvLinear/GateUpLinear 的 packed-only reference、Embedding/RMSNorm/Linear/AddRmsNorm 的 packed identity，以及 Linear/QkvLinear/GateUpLinear 的 `cpu_bpanel_f32_v1_avx2` 候选；reference 命名统一为 `cpu::<op>_f32_reference`）：

| OpType | Reference kernel | Optimized kernel | Generate baseline 状态 |
|---|---:|---:|---|
| Embedding | FP32 reference（+ packed identity） | 无 | 可用（packed 需 `enable_packed_weights=true`） |
| RMSNorm | FP32 reference（+ packed identity） | AVX2+FMA | 可用 |
| Add | FP32/FP64/BF16/I32/I64 reference | 无 | FP32 可用 |
| ElementwiseMul | FP32 reference | 无 | semantic Llama baseline 不直接依赖 |
| Linear | FP32 reference（+ packed identity） | packed bpanel candidate（AVX2+FMA） | 可用 |
| RoPE | FP32 reference | 无 | 可用 |
| KVCacheUpdate | FP32 reference（窄 KV binding） | 无 | 可用 |
| Attention | FP32 reference（kv_read 读绑定） | 无 | 可用 |
| Silu | FP32 reference | 无 | semantic Llama baseline 不直接依赖（SiluMul 未融合对偶） |
| SiluMul | FP32 reference | 无 | 可用 |
| Argmax | FP32 reference | 无 | 可用 |
| QkvLinear | packed identity reference（packed-only） | packed bpanel candidate（AVX2+FMA） | 可用（需 O2 融合 + `enable_packed_weights=true`） |
| GateUpLinear | packed identity reference（packed-only） | packed bpanel candidate（AVX2+FMA） | 可用（需 O2 融合 + `enable_packed_weights=true`） |
| AddRmsNorm | FP32 reference（plain + packed identity） | 无 | 可用（O2 fused path；packed 需 `enable_packed_weights=true`） |

当前 O2 默认 semantic pipeline 会产生 `QkvLinear`、`GateUpLinear` 和 `AddRmsNorm`。三者的 packed kernel 与 execution packed 绑定链路（`ExecutionStep.packed_weights` → packing request → `PrepackWeightRequests` → plan build → execute）均已落地并走通全链路测试；AddRmsNorm 另外保留 plain FP32 reference descriptor。execution lowering 仍是一个 semantic node 对应一个 kernel step，且不存在 kernel-sequence fallback；**baseline 全链路 kernel 已全部齐备（15 类 27 描述符全部可用）**，剩余准入项为 Prefill→Decode 端到端数值证据（见 §3.3–§3.5）。

### 2.3 当前 packed-weight 能力

已经具备：

- binding-aware `WeightArtifactKey`；
- graph-driven packing request（compiler）+ 编排期 recipe 注入（`Backend::GetPackingRecipe` → `WeightPackingRequest::recipe`）；
- direct/QKV/Gate-Up composite weight materialization（backend 经 `Backend::PackWeights` 落实，含对齐与分配）；
- QkvLinear/GateUpLinear packed-only reference kernel（cpu_identity 与 cpu_bpanel_f32_v1_avx2 双 recipe 契约：logical shape/recipe/alignment 校验、行切分、与输出的 disjoint 校验）；
- `RawWeightView` byte-size 验证；
- tied lm-head fallback；
- plain-step filtering和 exact recipe lookup。

已补齐（2026-09-23）：

- kLinear/kEmbedding/kRmsNorm 的 kPacked identity 变体（unfused packed 路径可解析）；`enable_packed_weights=true` 的完整 Llama 已可 prepare，由 `ExecutableModel.PackedLoweringPreparesAllWeightConsumers` 正向覆盖（此前的 `PackedLoweringIsUnresolvableForOpsWithoutPackedKernels` 缺口测试已移除）；
- QkvLinear/GateUpLinear/Linear 的 `cpu_bpanel_f32_v1_avx2` 候选 descriptor（AVX2+FMA 特化；与 identity 同 priority，选举当前仍落 identity，测试 `CPUKernelLinear.PreparesPackedIdentityFallback` 固化）。

仍未具备：

- bpanel 成为默认选择（需 benchmark 证据后调整 priority/eligibility，见 [GEMM 提案](../operators/gemm/cpu-gemm-packed-weight.md) §5 M5）；
- `enable_packed_weights=true` 的 unfused e2e 数值验证。

`CpuWeightPrepacker::Pack(..., recipe)` 已按显式 recipe 分派 identity 与 `cpu_bpanel_f32_v1_avx2` 两种 layout；`RecipeFor(selector)` 仅作兼容保留（无生产调用者）。QkvLinear/GateUpLinear/Linear 的 packed 契约已有全链路数值测试覆盖；bpanel 是否升为默认仍取决于 benchmark 结论。

## 3. 必须先闭环的阻塞项

### 3.1 KV/state identity 到达 kernel ✅（2026-09-16 闭环）

KVCache Manager 当前尚存的 owner/epoch correctness 缺口、CPU Attention 优化准入与长期 Paged KV 边界由 [08 号演进提案](08-kv-cache-and-attention-capability-evolution.md) 定义；本节只保留 Generate 闭环所需的集成门禁。

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

### 3.2 ExecutableModel 准备入口已落地

`PrepareExecutableModel(Runtime&, LoweredModelArtifact)` 是从真实 compiler artifact 到可执行模型的生产准备入口，负责按 lowered graph 物化权重、构建 immutable external binding map 和 execution plan，并让 `ExecutableModel` 持有 artifact、weight backing、packed artifacts、binding metadata 与 plan。Session 不扫描 `LoweredGraph`，也不按 debug name 解析权重角色。

证据包括 `ExecutableModel.*` 测试，以及 M4 直接执行测试 `DirectPrefillDecode.TinyTiedGqaLlamaMatchesScalarOracleAndReusesDecodeBindings`：O1 FP32 plain tiny Llama 从 `ModelCompiler::Compile` 进入 `PrepareExecutableModel`，plan 由真实 `CpuBackendFactory` 构建。

### 3.3 immutable external weight binding 已落地

模型准备阶段通过结构化 `WeightBinding` identity 将权重和常量绑定到 `ExecutionValueId`。`WeightBindingStorage` 保有 view 所借用的 shape/stride metadata，artifact 保有原始 backing；plain tensors 与 packed artifacts 按 plan requirements 区分。Tied lm-head 由准备阶段解析为与 embedding 相同的 backing。

已有 `BindingsMatchExternalReadRequirementsExactly`、`TiedLmHeadSharesEmbeddingBacking` 和 packed preparation 测试验证 binding 完整性与 ownership。M4 配置额外断言 immutable table 中 embedding backing 恰好被 embedding 和 tied lm-head 两个 value 引用。

### 3.4 Prefill/Decode phase-plan 合同已验证

reference baseline 编译为一个 `kBoth` lowered graph；`ExecutableModel::plan(kPrefill)` 和 `plan(kDecode)` 可共享同一个 immutable `ExecutionPlan`。两阶段仍分别用 concrete token/position shape 创建 `PreparedExecutionBindings` 和 `ExecutionContext`，Decode loop 复用其单个 specialization。

07 号提案 M2.5 已验证共享 plan identity、phase mismatch 拒绝和混合 phase prepare 拒绝。M4 进一步执行该共享 plan：同一 Decode context 连续执行两次，验证所有 step 输入/输出地址和 prepared kernel params 地址保持不变。

### 3.5 M4 direct Prefill→Decode 证据已闭环

2026-09-24，`DirectPrefillDecode.TinyTiedGqaLlamaMatchesScalarOracleAndReusesDecodeBindings` 通过真实 `CpuBackend` 执行 tiny Llama，配置为 1 layer、hidden size 8、4 query heads / 2 KV heads、head_dim 2、tied lm-head、3-token prompt。权重由确定性 fixture 填为非零值。独立 scalar oracle 覆盖 Embedding、RMSNorm、Linear、RoPE、causal GQA Attention、SwiGLU MLP、tied lm-head 与 Argmax。

测试逐 stage 比较 logits、tokens、完整已提交 key/value 内容和 commit position；重复运行新的 Prefill→Decode 链并逐项比较结果。Prefill 后和每次 Decode 成功后由 `Executor` 自动推进 commit watermark，测试不手动调用 `CommitUntil`。同一 Decode bindings 执行两步；输入/输出地址与 prepared kernel params 保持稳定。

错误路径将 Decode `position_ids` 设为 -1，验证执行失败且 watermark 不前进；先 `ExecutionContext::Clear()`，再 `ReleaseSession()`，随后成功重新 reserve，证明资源释放顺序可用。

零分配计数覆盖每个 steady-state Decode body：写入稳定 token/position buffer、存在 workspace 时调用 `WorkspaceArena::Reset()`、执行 `Executor::Execute` 并读取输出 token。scalar oracle、断言、KV snapshot 和计数结果检查都在窗口外。glibc/Linux interposer 对窗口内的 `malloc/calloc/realloc/free/aligned_alloc/posix_memalign/memalign` 计数；C++ `operator new` 经 malloc-family 入口计入。`MallocInterposerObservesCpuAllocatorCalls` 通过共享库内的 `CPUAllocator::Allocate` 校准 `posix_memalign/free` 符号拦截。Debug 和 Release 均观察到两次 Decode 各 0 次分配与释放。

该测试还发现并修复两个成功路径分配：KVCacheUpdate 和 Attention 的 KV footprint 验证原先每次为诊断参数动态拼接 `std::string`，现在传入固定 `string_view` 错误标签。执行层也已接受模型图中符号化的 cache_len 轴，同时继续要求静态 KV heads/head_dim 与 `KVCacheView` 匹配；显式静态容量仍须等于 runtime capacity，并有错误几何回归覆盖。

Allocation interposer 目前为 glibc/Linux 专用；其他平台仍运行数值、KV、确定性和失败清理证明，跳过仅有的 malloc-family 计数证据。

### 3.6 M5 public InferenceSession / Generate 已闭环

2026-09-24，public C++ `InferenceSession::Create` 和 `Generate` 已落地。Session 借用必须长于 Session 的 `Runtime`，并以 `shared_ptr<const ExecutableModel>` 保活模型；每次 `Generate` 从新的 Prefill 开始。`Generate` 返回本次新 token，包含 Prefill 预测出的第一个 token，也包含触发停止条件的 EOS。空 prompt 返回错误；`max_new_tokens == 0` 验证 prompt/config 后返回空结果，不要求 KV manager，也不执行模型。

`ExecutableModel` 在准备期从已验证的 HF config 冻结 vocabulary size 与 `max_position_embeddings`，Session 不访问 compiler artifact。后者作为最大 context token 数使用，是保守上限；即使 RoPE 配置支持外推，本接口也不超过该值。Session 按当前 Llama 有序 `model_inputs`（token IDs、position IDs）分别校验 Prefill/Decode plan 的 I/O 合同；Decode plan 的 value ID 可独立于 Prefill。

对于生成上限 N，Prefill 已给出 token #1，因此 KV reservation 精确为 `prompt_len + (N - 1)`。`ReserveForSession` 的第二个参数现明确表示 prompt 后最多追加的 KV positions。请求级 RAII 持有输入 buffer、对齐 workspace、KV view 和当前 context，所有错误路径都先清理 context 再调用 `ReleaseSession`。EOS 命中或 N=1 时不创建 Decode context；否则只准备一次 Decode context，并复用固定 buffers、workspace 与 prepared kernel params。

验证在 `tests/unit/inference/test_direct_prefill_decode.cpp`：

- `InferenceSession.TinyGenerateMatchesScalarOracleAndStartsFreshEachCall`：真实 CpuBackend 的 tiny tied-GQA Llama，完整生成 token 序列与独立 scalar oracle 一致，连续两次调用相同且从新 Prefill 开始；
- `InferenceSession.HandlesZeroOneEosAndExactKvCapacity`：N=1、EOS 首 token/中途 token、恰好用满物理 KV capacity 和多一个 append 的拒绝；
- `InferenceSession.RejectsContextLimitAndArithmeticOverflowBeforeReservation`：保守模型 context limit 与 checked-add overflow 在 reservation 前拒绝；
- `InferenceSession.ZeroLimitDoesNotRequireKvAndInputsAreValidated`：无 KV manager 的零上限、missing manager、空 prompt、越界 prompt/EOS；
- `InferenceSession.HandlesPromptSizedCapacityAndReservationCleanup`：prompt-only reservation、请求竞争、KV 几何执行失败后的 reservation 清理；
- `InferenceSession.MissingCpuAllocatorReturnsStatusBeforeReservation`：CPU allocator 未启用时返回状态，且不占用 KV reservation；
- `InferenceSession.RejectsRuntimeDifferentFromPreparationRuntime`：拒绝与 ExecutableModel 准备时不同的 Runtime；
- `InferenceSession.SharedDecodeLoopHasNoMallocFamilyCallsAfterPreparation`：对 Generate 共用的 Decode loop helper 做 glibc/Linux malloc-family 计数，bindings/context/workspace 与 output capacity 均在计数窗口前准备，循环内计数为零。其他平台跳过该 interposer 证据。

M5 证明的是当前 C++ greedy Generate orchestration，不包含 C ABI、采样、会话跨调用 KV 复用、并发调用或 HTTP 服务。

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

### 4.2 已落地接口

实现已落地，签名以头文件为准；职责与错误码细节见 [ExecutableModel 模块设计](../designs/inference/01-executable-model.md) §5 与 [InferenceSession 模块设计](../designs/inference/02-inference-session.md) §5：

```cpp
StatusOr<ExecutableModel> PrepareExecutableModel(Runtime& runtime,
                                                LoweredModelArtifact artifact);

class ExecutableModel {
    StatusOr<const ExecutionPlan*> plan(ExecPhase phase) const noexcept;
    StatusOr<const ExternalTensorBindings*> immutable_weight_bindings(
            ExecPhase phase) const noexcept;
    size_t context_limit() const noexcept;  // 冻结自已验证的 HF max_position_embeddings
    size_t vocab_size() const noexcept;
    bool IsPreparedFor(const Runtime& runtime) const noexcept;
};

struct GenerationConfig {
    size_t max_new_tokens = 0;
    std::optional<uint32_t> eos_token_id{};
};

class InferenceSession {
    static StatusOr<InferenceSession> Create(
            Runtime& runtime, std::shared_ptr<const ExecutableModel> model);
    StatusOr<std::vector<uint32_t>> Generate(std::span<const uint32_t> prompt_tokens,
                                             const GenerationConfig& config);
};
```

与本节初版示意的三处差异：无 `ExecutableModelOptions`（packing 与 phase 已在编译期固化进 artifact 的 step selector）；两个 phase 访问器返回 `StatusOr<const T*>`，使 phase 不匹配可表达为失败而非静默复用；`Generate` 以 `std::span` 接收 prompt 并经 `StatusOr` 返回。

`ExecutableModel` 与 `InferenceSession` 均落在新 `inference/` 模块，根 `AGENTS.md` §2.1 ownership 表与跨模块依赖规则已登记（`inference → execution + compiler + model + runtime`，且不得被下层模块反向依赖）；未放入 `runtime`，因为 runtime 禁止依赖 compiler/execution/model。

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

**状态（2026-09-24）**：已完成，详见 §3.5 与 §9。测试不经过 Session facade，直接使用 `ExecutableModel`、`ExecutionContext` 和 `Executor`；完整 plan 成功后由 Executor 自动推进 KV watermark，不额外手动 commit。

```text
prepare prefill bindings
  → Execute Prefill
  → verify committed prompt KV and read last token
  → prepare one Decode context
  → Execute Decode #1 and verify commit
  → update stable input buffer and reset workspace
  → Execute Decode #2 and verify commit
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

**状态（2026-09-24）：已完成**，实现与证据见 §3.6。

在 M4 通过后实现同步 Session orchestration：

1. validate prompt/config；
2. reserve KV session；
3. allocate/reuse workspace；
4. prepare and execute prefill；
5. 完整 plan 成功时由 `Executor` 提交 KV append transaction，Session 读取 commit watermark；
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
  → successful full plan automatically commits the KV append transaction
  → verify current_pos and read the last output token
```

### 7.2 Decode steady state

```text
PrepareExecutionBindings(decode specialization)   // exactly once

loop:
  write previous token into stable input buffer
  write/update position ID
  reset workspace through Session owner
  Executor::Execute
  successful full plan automatically commits one KV position
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

本提案在 M1 开始实施时从 Draft 转为 In Progress。原 M1–M4 准入门禁已全部满足；以下新增 M5 交付门禁均已完成，构成本提案的最终验收：

- [x] state binding identity 从 LoweredGraph 到达 kernel（`ExecutionKVCacheStateIdentity` 进入 plan，窄绑定经 `KernelContext` 到达 kernel，见 §3.1）；
- [x] kernel 获得窄 KV binding，不依赖 Runtime/Session 宽对象（`KVCacheAppendBinding`/`KVCacheReadBinding` 逐调用传入）；
- [x] baseline pipeline 可以通过真实 CpuBackend 构建完整 plan（O1 未融合 tiny GQA Llama 经 `ModelCompiler` → `PrepareExecutableModel`，见 [07 号提案](07-executable-model-preparation.md) M2.4）；
- [x] Linear/RoPE/KVCacheUpdate/Attention/SiluMul/Argmax reference kernel 可用（6/6 全部可用）；fused QkvLinear/GateUpLinear/AddRmsNorm 亦已落地；
- [x] `PrepareExecutableModel` 可从真实 `LoweredModelArtifact` 构建（`inference/executable_model.h`，07 号提案 M2.4）；
- [x] real weights 可自动生成完整 external bindings（12 个权重值自动绑定并与需求集合双向对账；packed 全模型路径已可解析，见 §2.3）；
- [x] Prefill/Decode phase-plan 合同已验证（07 号提案 §4.5：`kBoth` artifact 三种 phase 查询共享同一 plan、单 phase artifact 拒绝不匹配查询、step 间 phase 不一致在 prepare 期拒绝，由 M2.5 测试覆盖）；
- [x] tiny Llama Prefill + 2 Decode logits/tokens 与独立 scalar oracle 对齐（`DirectPrefillDecode.TinyTiedGqaLlamaMatchesScalarOracleAndReusesDecodeBindings`）；
- [x] 所有已写 KV key/value 与 commit position 对齐 oracle，并由 Executor 在完整 plan 成功后自动提交；
- [x] 同一 Decode `ExecutionContext` 连续执行两步，不重新调用 `PrepareExecutionBindings`，plan / tensor / params 地址稳定；
- [x] glibc/Linux Decode steady-state body 的 malloc-family 与 C++ allocation 入口计数为零；窗口包括输入内容更新、workspace reset、Execute 和读取输出 token；
- [x] 非法 Decode position 失败后 watermark 不前进，清理 context 后释放 reservation 并可重新 reserve。
- [x] public C++ `InferenceSession::Generate` 使用真实 `ExecutableModel` phase plans 和 CpuBackend，不解析 artifact 或自行解析权重；
- [x] prompt/config、Int64→uint32 token 输出、vocabulary 与保守 context limit 校验完成；
- [x] Prefill 返回第一个新 token；N 个新 token 的 KV reservation 为 prompt 长度加 N−1；
- [x] N=0 不要求 KV、不执行模型；N=1 和首 token EOS 不建立 Decode context；
- [x] 正常、EOS、capacity 与执行失败路径均先清理 context 再释放 KV reservation，重复 Generate 从新 Prefill 开始；
- [x] Session 返回完整 token 序列与独立 scalar oracle 对齐，覆盖 EOS、精确容量、竞争及失败清理；
- [x] Generate 共用 Decode loop 的 glibc/Linux malloc-family 计数为零，准备工作与结果 capacity reserve 均在计数窗口外；
- [x] Session 拒绝与 ExecutableModel 准备 Runtime 不同的 Runtime，并拒绝非 CPU execution plan；
- [x] Runtime 未注册 CPU allocator 时，Generate 在 KV reservation 前返回 FAILED_PRECONDITION。

## 10. 关联代码

- [`include/aethermind/runtime/runtime.h`](../../include/aethermind/runtime/runtime.h)
- [`include/aethermind/execution/execution_plan.h`](../../include/aethermind/execution/execution_plan.h)
- [`include/aethermind/base/kv_cache_binding.h`](../../include/aethermind/base/kv_cache_binding.h)
- [`include/aethermind/execution/execution_bindings.h`](../../include/aethermind/execution/execution_bindings.h)
- [`include/aethermind/execution/execution_context.h`](../../include/aethermind/execution/execution_context.h)
- [`include/aethermind/backend/kernel_context.h`](../../include/aethermind/backend/kernel_context.h)
- [`src/execution/layer_runner.cpp`](../../src/execution/layer_runner.cpp)
- [`src/model/llama_dense_graph_builder.cpp`](../../src/model/llama_dense_graph_builder.cpp)
- [`src/compiler/optimize_graph.cpp`](../../src/compiler/optimize_graph.cpp)
- [`src/backend/cpu/kernels/`](../../src/backend/cpu/kernels/)
- [`src/backend/cpu/kernels/kvcache_update/kvcache_update_entry.cpp`](../../src/backend/cpu/kernels/kvcache_update/kvcache_update_entry.cpp)
- [`src/backend/cpu/kernels/attention/attention_entry.cpp`](../../src/backend/cpu/kernels/attention/attention_entry.cpp)
- [`tests/unit/inference/test_direct_prefill_decode.cpp`](../../tests/unit/inference/test_direct_prefill_decode.cpp)
- [`tests/unit/execution/test_kvcache_update_kernel.cpp`](../../tests/unit/execution/test_kvcache_update_kernel.cpp)

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
| 2026-09-23 | 1.9 | 按当前代码状态同步 packed-weight 能力：§2.2 描述符计数 21→27（新增 kLinear/kEmbedding/kRmsNorm 的 packed identity 与 Qkv/GateUp/Linear 的 `cpu_bpanel_f32_v1_avx2` 候选），表格 packed 相关行更新，段末"剩余准入项"改为端到端数值证据；§2.3 "仍未具备" 重写——kLinear/kEmbedding kPacked 变体与 tile/block recipe 已补齐（`PackedLoweringIsUnresolvableForOpsWithoutPackedKernels` 缺口测试已由 `PackedLoweringPreparesAllWeightConsumers` 取代），仅剩"bpanel 升为默认（待 benchmark）"与 unfused e2e 数值验证；§9 括注同步。另注：§3.2–§3.5 的缺口叙述（如"缺少 ExecutableModel 准备入口"）早于本次同步即已过期，待该文件自身维护时重写 |
| 2026-09-24 | 1.10 | 更正 §3.2–§3.4 中过期的 M2 缺口描述；记录 M4 真实 CpuBackend tiny Llama Prefill + 两步 Decode 的数值、KV、确定性、失败清理与 glibc/Linux steady-state allocation 证据。M4 修复符号化 cache_len 与 KVCacheView geometry 的执行校验冲突，并将 KVCacheUpdate/Attention 热路径的 eager diagnostic string 改为固定 string_view；§7 同步说明 commit 由 Executor 在完整 plan 成功后自动推进，§9 五项 M4 门禁全部勾选。M5 public Generate 仍未实现。|
| 2026-09-24 | 1.11 | 落地 M5 public C++ `InferenceSession::Generate`：冻结 ExecutableModel 的词表/context 元数据，精确按 prompt + N−1 预约 KV，增加请求级 RAII、Decode loop 复用和 Session oracle/边界/清理/分配测试，并校验 preparation Runtime 身份与 CPU allocator；状态更新为 Implemented。|
| 2026-09-24 | 1.12 | 修正三处过期表述：§2.2 packed 计数 6 → 实测 9 个描述符声明 `weight_format = kPacked`（并列明其构成）；§2.1 KV 行由"部分实现"改为"已实现（静态 contiguous baseline）"并指向 05 号提案的演进范围；§4.2 由"推荐接口轮廓"改为"已落地接口"，替换为实际签名、删除"落地 M2 前需更新 AGENTS.md"的过期句并记录与初版示意的三处差异。M5 实现描述另由 [InferenceSession 模块设计](../designs/inference/02-inference-session.md) 承接 |
