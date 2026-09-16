# KVCache Manager 演进方案

- **状态**: Draft
- **版本**: 1.0
- **日期**: 2026-09-16
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **当前实现说明**: [KV Cache 设计](../designs/kv_cache_design.md)
- **关联计划**: [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md)
- **关联模块**: runtime / execution / backend / model orchestration

## 1. 结论与范围

AetherMind 当前产品应继续采用 **Contiguous Static KV Cache**，不应直接引入面向多请求 serving 的 Paged KV Cache。当前目标是把已有静态内存池修正为具有明确生命周期、事务提交语义和 kernel binding 的 production baseline：单 `Runtime`、单 active session、初始化时一次性预分配、Decode 稳态零分配。

推荐的长期结构为：

```text
Runtime
  └── KVCacheManager                 资源预算、pool 与 lease 生命周期
        └── ContiguousKVPool         当前产品的物理 storage policy

InferenceSession
  └── KVCacheLease                  move-only Session reservation
        └── KVAppendTransaction     单次 Prefill / Decode append
              └── KVCacheExecutionBinding
                    └── KernelContext / KVCacheUpdate / Attention
```

未来进入多请求 serving 后，再增加并列的 `PagedKVPool` policy：fixed-size token block、logical block table、refcount、copy-on-write、prefix reuse 和 eviction。Paged KV 是 runtime/backend 的物理存储策略，不进入 Graph IR 或 OperatorSchema 语义。

### 1.1 当前计划包含

- 修复现有 reserve/commit/generation/owner identity 的 correctness 问题；
- 将模型 KV geometry、物理 layout、storage、Session ownership 和执行期 binding 分离；
- 定义 move-only `KVCacheLease` 与 append transaction；
- 将窄 `KVCacheExecutionBinding` 传入 `KernelContext`；
- 保持当前同步、单请求、静态预分配和 Decode 稳态零分配；
- 为将来的 Paged KV 保留 storage policy 与 layout capability 边界；
- 给出分阶段实施、测试和验收门禁。

### 1.2 当前计划不包含

- 在本提案中直接实现代码；
- request scheduler、continuous batching 或 chunked prefill；
- 多 Session 并发或请求级资源仲裁；
- prefix cache、eviction、preemption 或 KV offload；
- GPU/CUDA/CANN storage；
- 在没有 benchmark 的情况下固定 block size、V transpose 或具体 SIMD layout；
- 修改 Graph IR 以表达 page、block id 或物理地址。

## 2. 已验证的当前状态

当前实现已经具备一个可用但尚未闭环 production execution 的静态基线：

| 能力 | 当前实现 | 结论 |
|---|---|---|
| 物理所有权 | `Runtime` 持有 `KVCacheManager` | 生命周期方向正确 |
| storage | K/V 两块独立 CPU `Buffer`，`posix_memalign` 分配 | 连续静态存储已存在，但绕过 Runtime allocator/provider |
| layout | `[layer][kv_head][token][head_dim_padded]`，显式 stride/alignment | 适合作为 reference baseline |
| Session | 单个 `SessionKVSlot`，同一时刻只允许一个 reservation | 符合当前单请求产品边界 |
| stale view | `generation` 检查 | Release 后可失效，但 re-init 存在 epoch 回退风险 |
| bounds | layer/head/token/committed watermark 验证 | 基础边界检查已存在 |
| execution | `ExecutionContext` 保存 `KVCacheView`；`LayerRunner` 验证 state alias geometry | 只完成 presence/geometry validation，KV binding 尚未进入 kernel |
| kernel | `KVCacheUpdate`、`Attention` 尚无完整 CPU execution chain | 尚不能证明真实 Prefill→Decode |

关联代码：

- [`KVCacheManager`](../../include/aethermind/runtime/kv_cache_manager.h)
- [`KVCacheView` / `KVCacheLayout`](../../include/aethermind/runtime/kv_cache_view.h)
- [`kv_cache_manager.cpp`](../../src/runtime/kv_cache_manager.cpp)
- [`kv_cache_view.cpp`](../../src/runtime/kv_cache_view.cpp)
- [`ExecutionContext`](../../include/aethermind/execution/execution_context.h)
- [`LayerRunner`](../../src/execution/layer_runner.cpp)
- [KV Cache 单元测试](../../tests/unit/execution/test_kv_cache_manager.cpp)

## 3. 当前 correctness 阻塞项

以下问题必须在扩展功能之前修复。它们不是性能优化或未来 serving 能力，而是当前静态实现的正确性边界。

### 3.1 Reserve 提前发布未写入的 prompt

当前 `ReserveForSession(prompt_len, max_new_tokens)` 将：

```text
prompt_len  = prompt_len
current_pos = prompt_len
```

但 reserve 时 Prefill 尚未产生 `[0, prompt_len)` 的 K/V。`ValidateRead()` 又把 `current_pos` 解释为 committed watermark，因此新 Session 可以读取未初始化数据或上一 Session 留下的旧字节。

目标 invariant：

```text
committed_tokens == 已由所有 decoder layers 成功产生并对 Attention 发布的 token 数
```

新 reservation 必须从 `committed_tokens = 0` 开始。`prompt_len` 只能表示本次 Prefill 计划写入的 token 数，不能等价于已提交状态。

### 3.2 Commit 与写入缺少事务顺序

当前 `CommitUntil(new_pos)` 只推进 watermark，不知道 `[old_pos, new_pos)` 是否已经完成所有 layer 的 K/V 写入。API 甚至允许先 commit、后调用 mutable pointer 写入。

目标顺序必须是：

```text
BeginAppend -> 执行所有 layer 的 KV write/Attention -> Execute 成功 -> Commit
                                                   -> Execute 失败 -> Abort
```

失败后允许 pending range 留有脏字节，但不得提升 committed watermark；下一次 retry 必须覆盖整个 pending range。

### 3.3 Re-init 存在 generation ABA

当前 `Init()` 会将 `SessionKVSlot` 清零。如果旧 view 的 generation 与重新初始化、再次 reserve 后的 generation 相同，旧 view 可能错误地重新变为 valid，并指向新的 storage。

目标 invariant：

- Manager lifetime epoch 单调变化，re-init 不得回退；
- `{owner_id, slot_id, generation}` 共同标识一次 reservation；
- 已失效的 lease/view 永远不能重新有效。

### 3.4 Reset/Release 缺少 owner identity 验证

当前 `ResetSession(view)` 和 `ReleaseSession(view)` 只验证 `view.valid()`，没有证明 view 属于当前 Manager。来自另一 Manager 的有效 view 可能修改错误的 slot。

所有 lifecycle operation 必须验证：

```text
view.owner_id == manager.owner_id
view.slot_id   == expected_slot
view.generation == slot.generation
```

### 3.5 Manager 固定 CPU allocation 与 K/V 同构布局

当前 Manager 直接调用 `posix_memalign`，并假定 K/V 具有相同 dtype、head dimension 和 layout。这会阻碍：

- Runtime allocator/provider 统一预算与统计；
- NUMA placement、huge page、pinned/device memory；
- K/V 独立 dtype；
- V transpose 或其他 backend-specific layout；
- layout capability 驱动的 kernel selection。

### 3.6 KV state identity 尚未进入 kernel

当前 `LayerRunner` 只验证 state alias 与 `KVCacheView` 的 dtype/shape 一致。`KernelContext` 尚不携带 layer、K/V slot、read range、append range、base pointer/stride 或 block table。

因此当前机制证明的是 execution-side state validation，不是 `KVCacheUpdate → Attention` 的完整执行数据流。

## 4. 业界模式与适用性

### 4.1 llama.cpp：本地推理的直接参考

llama.cpp 的 KV cache 更接近当前 AetherMind 产品形态：预分配 backend buffer、区分 K/V 类型与布局、维护 sequence/cell 元数据，并允许 V 使用 transpose layout。其 sequence 管理能力明显宽于当前单 Session 产品，但“逻辑 sequence 状态与物理 buffer/layout 分离”的原则可直接借鉴。

参考：

- [llama.cpp KV cache header](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.h)
- [llama.cpp KV cache implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)

### 4.2 vLLM：block pool 与 hash-based prefix cache

vLLM 使用 fixed-size KV block、logical block table、free queue、refcount 和 writable/COW 判断。prefix cache 以 parent hash、当前 block token 和额外 identity 组成链式 hash；只有满足缓存语义的 block 才能共享和参与 eviction。

可借鉴的长期机制：

- fixed-size token block；
- logical-to-physical indirection；
- full immutable block sharing；
- partial tail exclusive ownership；
- refcount/COW；
- prefix identity 与多租户 salt；
- free/eviction queue 与缓存索引分离。

参考：

- [vLLM BlockPool](https://docs.vllm.ai/en/latest/api/vllm/v1/core/block_pool/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

### 4.3 TensorRT-LLM：按 attention geometry 分 pool

TensorRT-LLM 的 KV cache 同样由 fixed-size block pool 构成；一个 block 可以包含多个 layer 的 KV。对不同 attention window/head geometry 创建独立 pool，并进一步支持 prefix reuse、prioritized LRU 和 secondary-memory offload。

可借鉴的长期机制：

- 以同构 layer group 为分配单位；
- pool geometry 是物理合同，不是 semantic operator 属性；
- 多种 attention/window 类型不强行塞入单一 layout；
- eviction/offload 属于 block manager，不属于通用 allocator。

参考：[TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)

### 4.4 对当前产品的裁决

| 机制 | 当前采用 | 原因 |
|---|---:|---|
| 连续预分配 K/V plane | 是 | 单请求、静态容量、最低复杂度、零分配 Decode |
| move-only Session lease | 是 | 明确 ownership，避免遗漏 release |
| append transaction | 是 | 正确表达跨 layer 写入与发布边界 |
| kernel execution binding | 是 | 完成真实 KV dataflow，避免 kernel 依赖 Manager |
| K/V 独立 layout capability | 是 | 避免 reference layout 固化 backend 优化空间 |
| fixed-size block pool | 否 | 当前没有多请求、碎片或 prefix reuse 需求 |
| prefix cache / radix/hash index | 否 | 当前 PRD 明确不承诺 |
| refcount/COW/eviction | 否 | 单 Session 静态 pool 不需要 |
| offload / tiered storage | 否 | 当前 CPU-only 单层内存边界不需要 |

## 5. 约束与 invariant

### 5.1 模块边界

- `runtime/` 拥有物理 KV storage、内存预算、pool 和 lease 生命周期；不依赖 model/compiler/graph。
- `execution/` 将 state alias 与具体 KV execution binding 连接，构造 `KernelContext`；不拥有 KV storage。
- `backend/` 消费已解析的具体 layout/binding；不查询 `KVCacheManager`，不理解 Graph IR。
- `model`/Session orchestration 从已验证模型信息生成纯数据 `KVCacheSpec`；runtime 只消费纯数据合同。
- `operators/` 定义 KVCacheUpdate/Attention 语义和 shape/dtype contract，不出现 page、block id、allocator 或 device pointer。

### 5.2 生命周期

- `Runtime` 必须长于 `KVCacheManager`、pool 和所有 Session lease。
- `KVCacheLease` 独占当前单 slot，是 move-only owning handle。
- `ExecutionContext` 只借用当前 append transaction 产生的 execution binding。
- active lease/view 存在期间，Manager/pool 不得移动或重新初始化。
- release/re-init 后的 stale lease/view 永远失效，不允许 ABA resurrection。

### 5.3 状态与发布

- `reserved_capacity` 是本 Session 可写上限，不代表已提交 token 数。
- `committed_tokens` 初始为 0，只能在完整 execution 成功后单调增加。
- 同一 lease 同时最多存在一个 active append transaction。
- read 默认只能访问 `[0, committed_tokens)`。
- 当前 transaction 可向经过执行顺序证明的同层 Attention 暴露 pending append range；普通 read API 不能读取 pending range。
- abort 不改变 committed watermark。
- rewind 只能显式回退到已建立 checkpoint，不以未执行的 `prompt_len` 作为隐式 checkpoint。

### 5.4 热路径

- Decode steady-state 不分配、不扩容、不重新 resolve kernel。
- kernel 不查询 Manager、不锁全局 mutex、不做 registry lookup。
- pointer、stride、layout kind、read/append range 在 execution binding 阶段解析。
- current product 不为未来 multi-session 引入 atomic refcount、block table 或 eviction bookkeeping。

### 5.5 容量与整数安全

- 所有 shape × stride × element-size 计算使用 checked arithmetic。
- allocation 前必须生成完整 memory plan 并执行 budget check。
- logical capacity、physical capacity 和 model max context 分别验证。
- K/V plane 可以具有不同 dtype、head dimension、stride 和 total bytes。

## 6. 目标数据结构与接口

本节接口均为目标设计草案，不代表当前公共 API。具体命名可在实现评审中调整，但职责和 invariant 不应退化。

### 6.1 `KVCacheSpec`

```cpp
enum class KVLayoutKind : uint8_t {
    kHeadMajorContiguous,
    kValueTransposed,
    kPaged,
};

struct KVPlaneSpec {
    DataType dtype{};
    uint32_t head_dim = 0;
    KVLayoutKind layout = KVLayoutKind::kHeadMajorContiguous;
};

struct KVCacheSpec {
    uint32_t num_layers = 0;
    uint32_t num_kv_heads = 0;
    uint32_t max_context_tokens = 0;
    KVPlaneSpec key{};
    KVPlaneSpec value{};
    Device device = Device::CPU();
    size_t alignment = 64;
};
```

设计要点：

- K/V plane 独立表达，不能用一个 `kv_dtype/head_dim` 永久绑定二者；
- `KVLayoutKind` 是 backend/storage capability，不是 graph semantic attribute；
- spec 是纯数据合同，runtime 不需要 include model/compiler 类型；
- model/session preparation 负责从模型配置生成并交叉验证 spec。

### 6.2 `KVCacheMemoryPlan`

```cpp
struct KVPlaneMemoryPlan {
    size_t token_stride = 0;
    size_t head_stride = 0;
    size_t layer_stride = 0;
    size_t total_bytes = 0;
};

struct KVCacheMemoryPlan {
    KVCacheSpec spec{};
    KVPlaneMemoryPlan key{};
    KVPlaneMemoryPlan value{};
    size_t metadata_bytes = 0;
    size_t total_bytes = 0;
};
```

`PlanKVCache(spec)` 必须是 allocation-free 的纯计算，可在实际申请前用于错误报告、预算决策和测试。

### 6.3 `KVCacheManager`

Manager 只负责资源与 lease，不参与每个 element 的 offset 访问：

```cpp
class KVCacheManager {
public:
    Status Configure(const KVCacheMemoryPlan& plan,
                     Allocator& allocator);

    StatusOr<KVCacheLease> Acquire(
            size_t token_capacity) noexcept;

    const KVCacheMemoryPlan& memory_plan() const noexcept;
    KVCacheStats stats() const noexcept;
};
```

当前产品只允许一个 active lease。未来多 pool/multi-session 不通过扩大当前 slot 状态机预实现，而是在明确 serving 需求后增加 storage policy。

### 6.4 `KVCacheLease`

```cpp
class KVCacheLease {
public:
    KVCacheLease(KVCacheLease&&) noexcept;
    KVCacheLease& operator=(KVCacheLease&&) noexcept;
    ~KVCacheLease();

    KVCacheLease(const KVCacheLease&) = delete;
    KVCacheLease& operator=(const KVCacheLease&) = delete;

    size_t capacity_tokens() const noexcept;
    size_t committed_tokens() const noexcept;

    StatusOr<KVAppendTransaction> BeginAppend(
            size_t token_count) noexcept;
    Status RewindTo(KVCheckpoint checkpoint) noexcept;
    KVCheckpoint CreateCheckpoint() const noexcept;
};
```

lease 内部身份至少包含：

```text
owner_id + pool_epoch + slot_id + generation
```

析构自动 release。显式 `Release()` 可以作为错误可观察入口，但不能要求调用者必须手工释放才能保证资源安全。

### 6.5 `KVAppendTransaction`

```cpp
class KVAppendTransaction {
public:
    KVAppendTransaction(KVAppendTransaction&&) noexcept;
    ~KVAppendTransaction();

    size_t begin_position() const noexcept;
    size_t token_count() const noexcept;

    StatusOr<KVCacheExecutionBinding> BindForExecution() const noexcept;
    Status Commit() noexcept;
    void Abort() noexcept;
};
```

约束：

- transaction 默认析构为 abort；
- `Commit()` 只能执行一次；
- commit 只能由 Session/Executor orchestration 在完整 plan 成功后调用；
- kernel 不能直接推进 committed watermark；
- retry 必须重写完整 pending range。

### 6.6 `KVCacheExecutionBinding`

Contiguous baseline 的 per-layer binding 可表达为：

```cpp
struct ContiguousKVLayerBinding {
    std::byte* key_base = nullptr;
    std::byte* value_base = nullptr;

    size_t key_token_stride = 0;
    size_t key_head_stride = 0;
    size_t value_token_stride = 0;
    size_t value_head_stride = 0;

    uint32_t num_kv_heads = 0;
    uint32_t key_head_dim = 0;
    uint32_t value_head_dim = 0;

    size_t committed_tokens = 0;
    size_t append_begin = 0;
    size_t append_count = 0;
};
```

`KernelContext` 只携带指向 immutable execution binding 的窄指针。`KVCacheUpdate`/`Attention` 通过 state binding 中的 decoder layer identity 取得相应 per-layer binding，不接触 `Runtime` 或 `KVCacheManager`。

未来 Paged KV 应提供不同 concrete binding，例如 block table + block geometry。kernel descriptor 必须声明它支持的 layout kind；不得让 contiguous kernel 在热路径模拟 page translation。

## 7. Contiguous Static KV 物理布局

### 7.1 baseline layout

当前 CPU reference baseline 保持：

```text
K[layer][kv_head][token][key_dim_padded]
V[layer][kv_head][token][value_dim_padded]
```

即 layer-major、KV-head-major、token-major、dim-minor，K/V split。

该布局便于单个 Attention head 顺序扫描历史 token，同时保持 offset 和越界验证简单。它是 baseline capability，不是所有 backend 的唯一合法布局。

### 7.2 内存公式

定义：

```text
L  = num_layers
H  = num_kv_heads
T  = max_context_tokens
DK = padded key head dimension
DV = padded value head dimension
BK = key element bytes
BV = value element bytes
```

则：

```text
key_bytes   = L * H * T * DK * BK
value_bytes = L * H * T * DV * BV
total_bytes = key_bytes + value_bytes + metadata_bytes
```

所有乘加都必须 checked。alignment padding 必须体现在 plane memory plan 中，不能在 allocation 后隐式改变。

示例（不含 padding/metadata）：

| Geometry | FP16 4096-token KV bytes |
|---|---:|
| 32 layers × 32 KV heads × head_dim 128 | 2 GiB |
| 32 layers × 8 KV heads × head_dim 128 | 512 MiB |

GQA 对 KV 容量影响显著，因此预算必须使用 `num_kv_heads`，不能误用 query head 数。

### 7.3 allocation provider

物理 allocation 必须由 Runtime allocator/provider 完成，而不是 Manager 内部硬编码 `posix_memalign`。provider 至少表达：

- device/memory kind；
- alignment；
- requested bytes；
- ownership/deleter；
- allocation failure；
- committed/reserved byte statistics。

当前 CPU provider 可以继续使用 aligned pageable memory。NUMA、huge page、pinned memory 和 device memory 只有在独立需求与 benchmark 证明后增加。

## 8. Prefill / Decode 事务流程

### 8.1 Session 创建

```text
ExecutableModel/session preparation
    -> derive and validate KVCacheSpec
    -> PlanKVCache
    -> KVCacheManager::Configure (cold path)
    -> Acquire(requested token capacity)
    -> lease.committed_tokens == 0
```

`requested token capacity = prompt_tokens + max_new_tokens`，并同时受 model context limit 和 physical pool capacity 限制。

### 8.2 Prefill

```text
BeginAppend(prompt_len)                         committed = 0
    -> bind [append_begin=0, append_count=prompt_len]
    -> execute every decoder layer
         KVCacheUpdate writes layer K/V
         same-layer Attention reads committed + current pending range
    -> complete plan success
    -> Commit                                      committed = prompt_len
    -> optionally CreateCheckpoint("prompt")
```

Attention 在 transaction 内看到 pending range 是 execution-order capability，不等于该 range 已全局 committed。若后续 layer 失败，transaction abort，普通 read 仍只能看到旧 watermark。

### 8.3 Decode

```text
BeginAppend(1)                                  committed = N
    -> append range [N, N+1)
    -> execute all decoder layers
    -> success: Commit                          committed = N+1
    -> failure: Abort                           committed = N
```

每个 Decode step 复用预构造 metadata/binding storage，不允许新建 vector、block table 或 owning buffer。

### 8.4 Rewind/Reset

- `CreateCheckpoint()` 只能捕获 committed position；
- `RewindTo(checkpoint)` 只能回退到同 lease、同 generation 的有效 checkpoint；
- rewind 不清零物理内存；
- rewind 后超出 watermark 的数据不可由普通 read 访问；
- “回到 prompt”由成功 Prefill 后创建的 checkpoint 表达，不由 reserve 参数隐式表达。

## 9. 并发模型

当前产品保持单请求同步执行：

- Manager configure/acquire/release 是 cold path；
- 一个 lease 同时最多一个 active transaction；
- 同一 Session 的 Prefill/Decode 串行；
- kernel 内部可以按 head/token 使用 Runtime 统一线程设施并行，但不得并发修改 lease metadata；
- commit/abort 由调用 execution 的控制线程执行；
- hot path 不引入 Manager mutex、atomic refcount 或全局 block lock。

未来 serving 模式推荐由 Scheduler 单线程拥有 block table mutation，worker/kernel 只消费 immutable execution binding。该模型需要独立设计，不属于当前实现的“预留字段”。

## 10. Paged KV 长期演进边界

### 10.1 目标对象

```cpp
struct KVBlockId {
    uint32_t value = 0;
};

struct SessionBlockTable {
    std::span<const KVBlockId> blocks;
    size_t logical_tokens = 0;
};

struct KVPhysicalBlockMetadata {
    uint32_t ref_count = 0;
    KVBlockState state{};
    PrefixKey prefix_key{};
};
```

### 10.2 必须成立的 invariant

1. logical token block 通过 block table 映射到 physical block；
2. full、immutable block 才能无条件共享；
3. partial tail 默认由单 Session 独占；
4. shared block 再次写入前必须 copy-on-write；
5. active request 引用的 block 不得 eviction；
6. `ref_count == 0` 才能进入 free/evictable queue；
7. prefix identity 至少包含 model/weight revision、token chain、position/RoPE semantics、K/V dtype/layout、adapter identity 和 tenant/cache salt；
8. block allocation/eviction 是 manager/scheduler 行为，kernel 只消费 resolved block table。

### 10.3 physical block grouping

推荐按同构 layer group 打包：

```text
Block[group][layer-in-group][K/V][token-in-block][head][dim]
```

不同 attention window、head geometry 或 state type 使用不同 pool/group。这样避免一个全局 block layout 被异构模型状态绑死。

### 10.4 block size 决策

不预先冻结 16/32 token。block size 必须通过以下维度 benchmark：

- Attention kernel 的历史扫描和 gather 代价；
- block-table metadata 与 translation 成本；
- partial-tail internal fragmentation；
- prefix match granularity；
- allocation/free latency；
- page/TLB/NUMA 行为；
- per-block all-layer bytes 与 huge-page fit。

## 11. 备选方案与 trade-off

### 11.1 方案 A：保持当前 View + CommitUntil

**优点**：改动最小。

**拒绝原因**：不能证明跨所有 layer 的写入完成；commit 可先于写入；reserve 提前发布 prompt；owner/epoch 不完整。继续扩展会把错误状态机固化到 Attention kernel。

### 11.2 方案 B：当前产品立即改为 Paged KV

**优点**：形式上接近 vLLM/TensorRT-LLM，未来多请求能力更直接。

**拒绝原因**：当前没有 scheduler、continuous batching、prefix reuse 或多 Session 需求；会立即引入 block table、refcount/COW、fragmentation policy 和 paged Attention kernel，增加 correctness 与性能验证面，并破坏最短 Generate baseline 路径。

### 11.3 方案 C：Contiguous storage + lease/transaction/binding

**优点**：满足当前产品边界；修复 publish/lifetime 问题；Decode 零分配；同时为 layout capability 和未来 storage policy 建立正确切点。

**缺点**：未来 Paged KV 仍需新的 kernel binding 和 block manager，不能仅替换 offset 函数。

**裁决**：采用方案 C。允许未来新增并列 policy，不承诺两种 layout 共用同一个热路径 kernel。

### 11.4 虚接口 vs variant/static binding

当前产品不需要 runtime-polymorphic virtual storage interface。推荐在 cold path 使用 concrete type/`std::variant` 选择 storage policy，并给 kernel 传递已解析的 concrete binding。这样避免在每个 token/head 访问中产生虚调用，也不要求 backend 理解 Manager 类型。

## 12. 实施步骤

### M0：正确性修复

- reservation 的 committed watermark 从 0 开始；
- 增加 Manager owner identity；
- generation/epoch 在 re-init 后仍单调；
- Reset/Release 验证 owner/slot/generation；
- 明确禁止 active view 时 re-init/move；
- 增加 stale view、cross-manager 和 ABA 测试。

退出条件：当前静态 API 不再发布未写入 prompt，旧 view 不能访问新 reservation/storage。

### M1：append transaction

- 引入 move-only lease；
- 引入 `BeginAppend/Commit/Abort`；
- transaction 析构默认 abort；
- Session orchestration 在完整 plan 成功后 commit；
- 增加执行失败注入与 retry 测试。

退出条件：任意 layer/step 失败均不提升 committed watermark。

### M2：spec、memory plan 与 allocator

- 从散落 Runtime geometry 迁移到纯数据 `KVCacheSpec`；
- 分离 K/V plane spec/layout；
- 增加 checked `PlanKVCache`；
- 接入 Runtime allocator/provider；
- allocation 前执行 budget check；
- 增加 capacity/peak/failure stats。

退出条件：Manager 不再直接硬编码 CPU allocation，K/V memory bytes 可在 allocation 前精确报告。

### M3：execution-native KV binding

- state binding 保留 decoder layer 与 K/V identity；
- transaction 构造 immutable `KVCacheExecutionBinding`；
- `KernelContext` 携带窄 KV binding；
- `KVCacheUpdate` 和 `Attention` 只消费 binding；
- kernel descriptor/layout capability 与 resolved binding 一致。

退出条件：真实 CPU `KVCacheUpdate → Attention` 能通过 production `Executor` 路径访问正确 layer/range，不查询 Manager。

### M4：Prefill→Decode 闭环

- tiny Llama 完整 Prefill；
- 连续至少两个 Decode step；
- 每步 commit 与输出数值验证；
- prompt checkpoint/rewind 验证；
- Decode 无 re-prepare、无 heap allocation。

退出条件：满足 [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md) 的 KV/state 与直接 Prefill→Decode 门禁。

### M5：Paged KV 专项立项（非当前承诺）

仅在产品明确进入多请求 serving 后启动：

- scheduler ownership；
- block geometry benchmark；
- block pool/table/refcount/COW；
- paged Attention kernel；
- prefix cache identity/security；
- eviction/preemption/quota；
- continuous batching 端到端验证。

## 13. 验收标准

### 13.1 单元测试

- layout/stride/alignment/byte-size checked arithmetic；
- K/V plane 不同 dtype/head dimension/layout 的 plan contract；
- zero/overflow/exceed-model-limit/exceed-budget 拒绝；
- reservation 初始 committed 为 0；
- append begin/count 边界；
- commit once、double commit、abort、RAII abort；
- cross-manager Reset/Release 拒绝；
- release/re-init 后 stale lease/view 永不复活；
- checkpoint 必须属于同 lease/generation；
- rewind 后越界 token 不可读。

### 13.2 execution 集成测试

- state identity 正确到达每个 decoder layer kernel；
- Prefill transaction 内同层 Attention 可看到本层刚写入的 pending KV；
- plan 中途失败后 committed watermark 不变；
- retry 覆盖完整 pending range并得到正确结果；
- workspace 与 KV persistent state 不混用；
- binding/layout 与 resolved kernel capability 不匹配时在执行前拒绝。

### 13.3 数值测试

- tiny Llama Prefill 与无缓存 reference 一致；
- 至少两个连续 Decode token 一致；
- GQA：`num_kv_heads != num_attention_heads`；
- prompt checkpoint → decode → rewind → decode 可重复；
- Reset/reuse 不读取上一 Session 的未提交数据。

### 13.4 性能与资源测试

- Decode steady-state allocation count 为 0；
- hot path 无 Manager lookup、mutex、registry lookup；
- memory plan 与实际 allocation bytes 一致；
- benchmark 分离 layout/access、Attention kernel 和 Session lifecycle 成本；
- 任何 layout 优化结论必须基于真实 Attention kernel 与实际 build flags，不以 Manager microbenchmark 代替。

## 14. 风险与依赖

| 风险/依赖 | 影响 | 缓解方式 |
|---|---|---|
| `InferenceSession` 尚未实现 | transaction 缺少最终 orchestration owner | M1 可先由内部 execution fixture 驱动；public facade 延后 |
| KVCacheUpdate/Attention kernel 尚未闭环 | 无法证明 binding 端到端正确 | M3 与 reference kernel 同步推进，先做 CPU FP32 baseline |
| execution plan 对 state alias 只做 geometry 校验 | layer/K/V identity 不能到 kernel | 在 execution 内完成 state binding → concrete binding conversion |
| K/V layout 过早冻结 | 阻碍 V transpose、量化 KV 或 paged kernel | spec 分离 semantic geometry 与 physical layout capability |
| transaction 粒度不当 | 过细无法原子发布，过粗影响同层 Attention 读取 pending KV | 全 plan commit + transaction 内受控 pending visibility |
| 未来 Paged KV 复用 contiguous API 过度 | 热路径出现 variant/virtual/page translation 开销 | Manager facade 可统一，kernel binding/descriptor 保持 concrete |

## 15. 与现有文档的关系

- [当前产品 PRD](../products/aethermind_prd.md) 决定“静态 KV、单请求、无 PagedAttention”的当前交付边界；本提案不能扩大该范围。
- [KV Cache 设计](../designs/kv_cache_design.md) 描述存量静态设计与实现背景；其中与当前代码不一致或属于未来 API 的内容，应在本提案实施时逐步校正，不能把本提案当成已实现事实。
- [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md) 定义公开 Generate 的整体门禁；本提案的 M0–M4 是其中 KV/state 子路径的细化。
- Paged KV、prefix cache、continuous batching 仍属于长期方向。除非 PRD 和专项提案明确更新，不进入当前 implementation scope。

## 16. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-16 | 1.0 | 基于当前静态实现与业界 fixed-block/prefix-cache 方案，建立 contiguous baseline、lease/transaction/binding 目标架构及 Paged KV 演进边界 |
