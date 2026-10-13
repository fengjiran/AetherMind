# InferenceSession 模块设计

- **状态**: Current（描述已验证实现；只写仓库事实）
- **版本**: 1.3
- **日期**: 2026-10-12
- **来源提案**: [01 号计划：InferenceSession / Generate 前置闭环计划](../../improvement-plan/01-inference-session-generate-readiness.md)（Implemented，§3.6）
- **关联代码**: [include/aethermind/inference/inference_session.h](../../../include/aethermind/inference/inference_session.h) / [src/inference/inference_session.cpp](../../../src/inference/inference_session.cpp)、[src/inference/inference_internal.h](../../../src/inference/inference_internal.h)
- **上游依赖**: inference（`ExecutableModel` 提供 phase plan、immutable 绑定表、`vocab_size()`/`context_limit()`/`IsPreparedFor()`）、execution（`PrepareExecutionBindings`、`ExecutionContext`、`Executor`、`ExecutionPlan` workspace 需求）、runtime（Runtime 的 `GetKVCacheManager`、`HasAllocatorProvider`/`GetAllocator`，`KVCacheManager::ReserveForSession`/`ReleaseSession`）、base（`WorkspaceArena`、`TensorView`、`CheckOverflowAdd`）、operators（`OpType::kArgmax` 合同校验）
- **下游消费者**: 当前只有单元测试；C ABI（`am_session_*`）仍为目标草案，未实现
- **关联测试**: [tests/unit/inference/test_direct_prefill_decode.cpp](../../../tests/unit/inference/test_direct_prefill_decode.cpp)（`TinyLlamaExecution` 的 `O1Plain` / `O2Packed` 两种生产路径与 `DirectPrefillDecode`）
- **架构总览**: [architecture_overview.md](../architecture/architecture_overview.md)

## 1. 背景与目标

`InferenceSession` 是 `ExecutableModel` 之上的同步单请求编排层，提供 greedy Argmax 生成：token IDs 进、token IDs 出。它存在的理由是让调用方不必理解 phase plan 选择、KV 预约算术、workspace 归属与失败清理顺序。

目标：

- **只编排**：Session 不读取 `LoweredGraph`/`LoweredModelArtifact`，不解析 `TransformerWeightRole`，不解释 packing 决策。实现中这些符号零引用。
- **每次调用自洽**：`Generate` 从新的 Prefill 开始，返回前释放自己的 KV reservation，失败路径同样释放。
- **Decode 稳态零分配**：绑定、context、workspace 与输出 capacity 全部在循环前准备完毕，循环内只改输入内容与 KV commit 位置。
- **准备期/调用期失败优于执行期错误数值**：词表、context 上限、算术溢出、plan I/O 合同都在预约 KV 之前校验。

当前范围不含：采样、C ABI、跨调用 KV 复用、并发 `Generate`、服务化调度。

## 2. 职责与边界

- **提供**：`InferenceSession::Create(runtime, shared_ptr<const ExecutableModel>)`、`Generate(prompt_tokens, GenerationConfig)`、`GenerationConfig{max_new_tokens, eos_token_id}`。
- **请求**：模型侧的 phase plan 与 immutable 绑定表；Runtime 侧的 KVCacheManager 与 CPU allocator；execution 的 specialize/create/execute 三段。
- **所有权**：借用 `Runtime`（裸指针，不延长其生命周期）；以 `shared_ptr<const ExecutableModel>` 共享模型所有权；**每个请求**独占输入 buffer、对齐 workspace、KV view 与当前唯一的 `ExecutionContext`（见 §3.3）。
- **明确不做**：不承担算子语义、kernel resolve、weight materialization、KV 物理存储；不提供 model inputs 之外的绑定；不缓存任何跨请求状态（Session 成员只有 Runtime 指针、模型指针与两份 phase 合同）。
- **生命周期**：`Runtime` > `ExecutableModel` > `InferenceSession` > 请求级 `RequestResources`（含 `ExecutionContext`）。`Runtime` 必须保持地址不变并长于模型，因为 resolved kernel 函数指针借用 backend 自有状态，`ExecutableModel::IsPreparedFor` 按地址比对。

## 3. 关键数据结构

### 3.1 InferenceSession 成员

| 成员 | 含义 | 备注 |
|---|---|---|
| `runtime_` | 借用的 `Runtime*` | 非拥有；地址必须稳定 |
| `model_` | `shared_ptr<const ExecutableModel>` | 保活模型，从而保活权重 backing 与绑定表借用的元数据 |
| `prefill_contract_` / `decode_contract_` | `PhaseContract{token_input, position_input, token_output}` | 两份独立的 value id 合同，Decode 的 id 可不同于 Prefill |

只可移动、不可复制（移动为 defaulted，复制被删除）。`Generate` 首行检查 `runtime_ == nullptr || model_ == nullptr` 并返回 `kFailedPrecondition`；被移动过的 Session 由 `model_` 为空触发该分支（裸指针 `runtime_` 在默认移动下不会置空，故守卫实际由 `model_` 承担）。

### 3.2 PhaseContract 与 plan I/O 合同

`ValidatePlanContract(plan)` 在 `Create` 时对 Prefill 与 Decode plan **各校验一次**：

- `model_inputs().size() == 2` 且 `model_outputs().size() == 1`，否则 `kUnimplemented`；
- 两个输入 id 必须互异、在值表范围内、`kind == kModelInput`、dtype `Int64`、rank 1；
- 唯一输出必须 `kind == kActivation`、`Int64`、rank 1；
- 每个 step 的 `selector.device_type` 必须为 `kCPU`；
- 全 plan 恰好一个 `kArgmax` step，且其唯一输出就是模型输出。

形状类不匹配返回 `kUnimplemented`（Session 只支持当前 Llama I/O 合同），结构性不可能（id 越界、输入重复）返回 `kInvalidArgument`。

输入角色依赖生产构建路径的有序契约：Llama graph builder 依次追加 token_ids、position_ids，lowering 保持 `model_inputs` 顺序，因此 Session 将 `[0]` 绑定为 token_ids、`[1]` 绑定为 position_ids，**不依赖 debug name**。两个输入均为 rank-one Int64，`ValidatePlanContract` 无法识别两者互换；扩展模型家族或改变输入顺序时，必须同步维护该角色契约。

### 3.3 RequestResources（请求级 RAII）

`Generate` 内的局部对象，析构即清理，因此正常返回、`AM_RETURN_IF_ERROR` 提前返回和 allocation exception 展开都会先清 context 再释放 reservation：

| 成员 | 含义 |
|---|---|
| `prefill_tokens` / `prefill_positions` | Prefill 输入，`std::vector<int64_t>`，地址在整个请求内稳定 |
| `decode_tokens` / `decode_positions` | 长度 1 的 Decode 稳定输入 buffer，循环内只写内容 |
| `generated_tokens` | 结果序列，循环前 `reserve(max_new_tokens)` |
| `storage_` | `unique_ptr<std::byte[]>`，拥有工作区分配；为空时没有 workspace |
| `workspace_arena_` | `SessionWorkspaceArena`，保存工作区对齐基址与逻辑字节数；基址仅在初始化时计算，不在 RequestResources 中重复保存 |
| `cache_view_` / `has_reservation_` | KV 预约句柄与标志 |
| `context_` | `std::optional<ExecutionContext>`，同一时刻至多一个 |

清理契约：`~RequestResources` → `Close()` → `ClearContext()`（`context_.reset()` 销毁阶段 context，由 RAII 释放其 owned bindings）→ 若持有预约则 `ReleaseSession(cache_view_)`。`Finish()` 先 `Close()`，成功后移出 `generated_tokens`。`ClearContext()` 也在 Prefill 结束后显式调用，使 Decode context 成为当时唯一的 context。请求级 KV 句柄与预约标志独立于 context，在准备失败及两个阶段之间仍负责预约清理。

### 3.4 SessionWorkspaceArena

把调用方自有的一块对齐 buffer 适配成 `WorkspaceArena`：`Bind` 校验 `requirement.offset/bytes` 落在 buffer 内、且 `base + offset` 满足 `requirement.alignment`，越界或未对齐返回空绑定；`Reset()` 是 no-op（arena 不拥有分配，内容由使用者负责）。workspace 尺寸与对齐取 **Prefill 与 Decode 两个 plan 的最大值**，使一块 buffer 服务两阶段。

## 4. 并发模型

- 模块内无锁：Session 成员在 `Create` 后只读，请求 buffer、workspace 与 context 归请求所有；Runtime 中的 KV slot 和 allocator registry 仍是共享可变状态。
- `Generate` 非线程安全；调用方必须串行化同一 Runtime 上的所有请求，包括来自不同 Session 的调用，并避免在请求期间修改 Runtime 资源。
- 多个 Session 可共享同一个 `shared_ptr<const ExecutableModel>`（模型只读）；`KVCacheManager` 的单 reservation 检查没有同步，只能在串行使用时拒绝已占用的 reservation。它不提供并发仲裁，不能保证并发请求安全失败。
- Decode 循环不重新调用 `PrepareExecutionBindings`，也不重新分配：`RunDecodeLoop` 的前置条件断言 `generated_tokens.capacity() >= max_new_tokens`。

## 5. 接口定义

| 接口 | 签名 | 语义要点 | Hot path |
|---|---|---|---|
| `InferenceSession::Create` | `static StatusOr<InferenceSession> Create(Runtime&, std::shared_ptr<const ExecutableModel>)` | 拒绝空模型（`kInvalidArgument`）、准备 Runtime 不匹配（`kFailedPrecondition`）、模型无已验证 token 上限（`kFailedPrecondition`）；随后取两个 phase plan 并各校验一次 I/O 合同。允许 Runtime 无 KVCacheManager——该缺失只在请求正数上限时报告 | ❌ |
| `InferenceSession::Generate` | `StatusOr<std::vector<uint32_t>> Generate(std::span<const uint32_t>, const GenerationConfig&)` | 返回本次新生成的 token，**含** Prefill 预测的首 token，也**含**触发停止的 EOS；空 prompt 非法；`max_new_tokens == 0` 在校验 prompt/config 后返回空结果，不要求 KV manager、不执行模型 | ✅（内含 Decode 循环） |
| `GenerationConfig` | `struct { size_t max_new_tokens = 0; std::optional<uint32_t> eos_token_id; }` | 只有生成上限与停止 token，无采样参数 | — |
| `inference::internal::RunDecodeLoop` | 见 [inference_internal.h](../../../src/inference/inference_internal.h) | 内部函数，非公共 API；单独成函数是为了让 malloc-family 计数窗口只覆盖 Decode 循环本体 | ✅ |
| `inference::internal::ValidatePlanContract` | `StatusOr<ValidatedPhaseContract>(const ExecutionPlan&)` | 内部函数，非公共 API；位于 internal 头是为了让聚焦测试用合成 ExecutionPlan 直接覆盖全部 kUnimplemented 失败分支（`ExecutionPlan::Create` 已在生产路径拒绝重复/越界输入 id 与 kModelInput kind 不符等结构性不可能，故 kInvalidArgument 守卫经公共路径不可达） | ❌ |

## 6. 算法与流程

### 6.1 Generate 顺序

```text
校验 moved-from / 空 prompt / 词表（prompt 与 EOS）/ prompt ≤ context_limit
  → max_new_tokens == 0：返回空结果（不取 KV manager、不执行）
  → future_kv_appends = N - 1；checked add 求 prompt + appends，再比 context_limit
  → 取 KVCacheManager 与 CPU allocator provider（缺失即 kFailedPrecondition）
  → 取 prefill/decode plan 与两份 immutable 绑定表
  → RequestResources::Reserve(prompt_len, future_kv_appends)
  → 填 prefill token/position buffer（position = 0..prompt_len-1），reserve 输出 capacity
  → InitializeWorkspace(max(两 phase bytes), max(两 phase alignment))
  → PrepareContext(Prefill)：复制 immutable 表 + 追加两个输入绑定 → PrepareExecutionBindings → ExecutionContext::Create
  → workspace->Reset()（存在时）→ Executor::Execute
  → 校验 kv_cache_session_view().current_pos() == prompt_len，否则 kInternal
  → ReadOutputToken(最后一个位置) → push 进 generated_tokens → ClearContext()
  → 若已达 N 或首 token 即 EOS：Finish() 返回（不创建 Decode context）
  → 写 decode buffer（token = 首 token，position = prompt_len）
  → PrepareContext(Decode) 一次
  → RunDecodeLoop(...)
  → Finish()
```

### 6.2 Decode 稳态循环

```text
while generated_tokens.size() < max_new_tokens:
  若上一个 token 是 EOS → 返回 Ok（不再执行）
  *input_token = 上一个 token                       // 只写内容，地址不变
  workspace->Reset()（存在时）
  Executor::Execute(decode_plan, context)
  checked add 求 expected = *next_position + 1
  校验 current_pos() == expected，否则 kInternal     // 恰好提交一个 KV position
  ReadOutputToken(长度 1) → push
  若该 token 是 EOS → 返回 Ok
  若还会继续生成：*next_position 自增（Int64 上限保护）
```

循环内不得变化：plan identity、输入/输出地址、shape/stride/dtype、prepared kernel params、workspace 基址、packed artifact identity。`next_position` 只在还会继续生成时自增，因此最后一次迭代不会把 position 推到未使用的位置。

### 6.3 KV 预约算术

Prefill 预测出的首 token 尚未写入 KV，因此生成上限 N 只需要 `N - 1` 次 Decode append：预约量为 `prompt_len + (N - 1)`。`ReserveForSession` 的第二个参数即"prompt 之后最多追加的 KV positions"（该参数由 `max_new_tokens` 改名为 `future_kv_appends`，`prompt_len + 参数` 的算术未变）。所有加法经 `CheckOverflowAdd`，溢出返回 `kOverflow` 且发生在预约之前。

编排成本：请求准备包含 O(prompt_len) 的输入填充；对于生成上限 N，调用一次 Prefill 和至多 N−1 次 Decode，每次 Decode 的 Session 控制逻辑为 O(1)，不分配、不重建绑定。模型执行成本另计：固定模型维度下，dense causal Attention 的 Prefill 计算量随 prompt 长度二次增长，单步 Decode 的 Attention 计算量随当前 context length 线性增长。

## 7. 边界条件与错误处理

| 情形 | 错误码 |
|---|---|
| Session 被移动过 / 无 Runtime | `kFailedPrecondition` |
| 空 prompt | `kInvalidArgument` |
| prompt token 或 EOS 超出词表 | `kOutOfRange` |
| prompt 超 `context_limit()`；prompt+appends 超 `context_limit()` | `kOutOfRange` |
| `max_new_tokens == 0` | Ok（空结果，不预约、不执行） |
| prompt + appends 溢出 `size_t` | `kOverflow` |
| Runtime 无 KVCacheManager；无 CPU allocator provider | `kFailedPrecondition` |
| plan I/O 合同不符（输入/输出个数、dtype、rank、非 CPU step、Argmax 不是唯一输出） | `kUnimplemented` |
| plan 值 id 越界或两个输入 id 相同 | `kInvalidArgument` |
| workspace alignment 非法 / 尺寸溢出 / `new(std::nothrow)` 失败 | `kInvalidArgument` / `kOverflow` / `kResourceExhausted` |
| 容器、bindings/context 或 CPU activation 分配的 `std::bad_alloc` 传播至 `Generate` 函数边界 | `kResourceExhausted`（错误 message 为空，避免 OOM 处理再次分配） |
| tokens 与 positions 长度不等或为空 | `kInvalidArgument` |
| Prefill 未提交完整 prompt；Decode 未恰好提交一个 position | `kInternal` |
| 输出 tensor 运行时 shape/dtype 与合同不符 | `kInternal` |
| Argmax 结果为负、≥ vocab、或超 `uint32` 范围 | `kOutOfRange`（收窄失败即报错，不截断） |
| Decode 循环收到无效预备状态（空结果、capacity 不足、position 为负等） | `kInvalidArgument` |
| Decode position 触及 `int64_t` 上限 | `kOverflow` |

CPU allocator 探测的存在理由：Runtime 的 `GetAllocator` 在缺 provider 时通过 `AM_CHECK` 终止进程，因此 Session 使用不抛的 `HasProvider`/`HasAllocatorProvider` 探针，在预约 KV 之前返回 `kFailedPrecondition`。这项探测防止资源配置错误。CPU allocator 将 `posix_memalign` 的 `ENOMEM` 转为 `std::bad_alloc`；`Generate` 捕获传播至其函数边界的 `std::bad_alloc`，返回不需分配错误 message 的 `kResourceExhausted`。

失败清理：请求建立后的 Status 错误和 allocation exception 都经由 `RequestResources` 析构（或显式 `Finish()`）执行 `ClearContext()` → `ReleaseSession()`；`ReleaseSession` 成功后才清除 `has_reservation_`。正常完成时，`Finish()` 会传播 `Close()` 的释放错误；析构时忽略 `Close()` 返回值，因此无法保证向调用方报告提前返回或异常展开期间的释放错误。测试验证非法 Decode position 失败后 watermark 不前进，且清理后可重新预约。

## 8. 风险与权衡

| 风险 / 权衡 | 现状与处理 |
|---|---|
| 依赖 `model_inputs` 的**顺序**契约 | 顺序由生产 Llama graph builder 与 lowering 保真；校验独立于 debug name，但无法识别两个同 dtype/rank 输入互换。新增家族或改变输入顺序须同步维护角色契约 |
| `context_limit` 取 `max_position_embeddings` | 保守上限；即使 RoPE 配置支持外推，本接口也不超过该值。该值与 `vocab_size` 在准备期从已验证 HF config 冻结进 `ExecutableModel`，Session 不回读 artifact |
| 不支持跨调用 KV 复用 | 每次 `Generate` 重新 Prefill。这是当前产品边界（同步、单请求），不是实现限制 |
| workspace 取两 phase 最大值 | 可能大于单 phase 需求，换取一块 buffer 服务两阶段与稳态零分配 |
| `RunDecodeLoop` 位于 internal 头 | 为让 malloc 计数窗口只包住循环本体而拆出；不是公共 API，签名可随实现调整 |
| `ValidatePlanContract` 位于 internal 头 | 为让聚焦测试用合成 plan 覆盖失败分支而暴露；其 kInvalidArgument 守卫（重复/越界输入 id、输入 kind 不符）在 `ExecutionPlan::Create` 处已被拒绝，属生产路径防御性冗余，不经公共路径可测 |
| Prefill 与 Decode 分别 prepare 一次 | 两次 `PrepareExecutionBindings` 是冷路径成本；Decode 只 prepare 一次后复用，符合 01 号计划 §7.2 |
| token 收窄 Int64 → uint32 | 越界报 `kOutOfRange` 而非截断，避免把模型错误当成合法 token |

## 9. 测试要点

`TinyLlamaExecution` 参数化套件（[test_direct_prefill_decode.cpp](../../../tests/unit/inference/test_direct_prefill_decode.cpp)），分别覆盖 `O1Plain` 与 `O2Packed`：

- `TinyGenerateMatchesScalarOracleAndStartsFreshEachCall`：真实 CpuBackend 的 tiny tied-GQA Llama，完整生成序列与独立 scalar oracle（`TinyLlamaScalarReference`）一致；连续两次调用结果相同且从新 Prefill 开始。
- `HandlesZeroOneEosAndExactKvCapacity`：N=1、首 token 即 EOS、中途 EOS、恰好用满物理 KV capacity、以及再多一个 append 被拒绝。
- `RejectsContextLimitAndArithmeticOverflowBeforeReservation`：保守 context limit 与 checked-add 溢出都在预约前拒绝。
- `ZeroLimitDoesNotRequireKvAndInputsAreValidated`：无 KV manager 的零上限、缺 manager、空 prompt、越界 prompt/EOS。
- `HandlesPromptSizedCapacityAndReservationCleanup`：prompt-only 预约、预约竞争、KV 几何执行失败后的预约清理。
- `MissingCpuAllocatorReturnsStatusBeforeReservation`：未注册 CPU allocator 时返回状态且不占用预约。
- `GenerateReturnsResourceExhaustedForRequestBufferAllocationFailure`：注入 Prefill token/position buffer 与输出 capacity 分配失败。
- `GenerateReturnsResourceExhaustedForBindingAllocationFailure`：注入 external bindings 扩容失败。
- `GenerateReturnsResourceExhaustedForActivationMetadataFailure`：注入 `BufferImpl` 分配失败，验证已分配的 activation payload 被释放。
- `GenerateReturnsResourceExhaustedForPrefillAndDecodeActivationFailure`：分别在 Prefill、Decode activation 分配注入 `posix_memalign` 的 `ENOMEM`。
- `RejectsRuntimeDifferentFromPreparationRuntime`：拒绝与准备期不同的 Runtime。
- `SharedDecodeLoopHasNoMallocFamilyCallsAfterPreparation`：对 `RunDecodeLoop` 做 glibc/Linux malloc-family 计数，绑定/context/workspace 与输出 capacity 均在窗口外准备，循环内计数为零；其他平台跳过该项证据。

四项分配失败测试要求 glibc/Linux interposer；共同验证返回 `kResourceExhausted` 且 message 为空、KV 可重新预约，以及取消故障注入后同一 Session 重试与 scalar oracle 一致。

同一参数化套件的 `TinyTiedGqaLlamaMatchesScalarOracleAndReusesDecodeBindings` 覆盖不经 Session 的直接执行链；`DirectPrefillDecode.MallocInterposerObservesCpuAllocatorCalls` 校准 interposer。

`CPUAllocator.SystemAllocationFailureThrowsBadAllocAndCanRetry`（[test_cpu_allocator.cpp](../../../tests/unit/memory/test_cpu_allocator.cpp)）直接验证系统分配的 `ENOMEM` 转为 `std::bad_alloc`，以及取消故障注入后 allocator 可重试。

`PlanContractValidation` 套件（同文件）用合成 `ExecutionPlan` 直接覆盖 `ValidatePlanContract` 的全部 kUnimplemented 失败分支：I/O 个数（1 输入/3 输入/2 输出）、输入 dtype/rank、输出 kind/dtype/rank、非 CPU step、无 Argmax、双 Argmax、Argmax 非模型输出，以及规范 Llama 合同的接受路径。

## 10. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-12 | 1.3 | 简化 RequestResources：移除重复 workspace 基址/大小和未使用的 view getter，以 optional reset 销毁 context，保留请求级 KV 清理责任 |
| 2026-10-11 | 1.2 | `ValidatePlanContract` 移入 internal 头，新增 `PlanContractValidation` 合成 plan 测试覆盖全部 kUnimplemented 失败分支 |
| 2026-10-11 | 1.1 | 修复 Generate OOM Status 通路；明确共享 Runtime 的请求串行化、输入角色顺序与析构清理错误传播边界；区分编排成本和 Attention 计算量；更新参数化测试说明 |
| 2026-09-24 | 1.0 | 从 [01 号计划](../../improvement-plan/01-inference-session-generate-readiness.md) §3.6 落地实现承接：Create/Generate 契约、plan I/O 合同校验、请求级 RAII 与清理顺序、KV 预约算术（prompt + N−1）、workspace 适配、错误码表、并发边界与测试要点 |
