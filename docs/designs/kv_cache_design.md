# AetherMind Phase 1 KV Cache 设计文档

> **文档关系**：本文保留存量静态 KV Cache 的设计背景；其中历史缺口叙述不代表当前代码状态。当前源码归属为 `include/aethermind/kv_cache/` 与 `src/kv_cache/`：`kv_cache_layout.h` 公开独立的布局值类型 `KVCacheLayout`，`kv_cache_session_view.h` 公开 `KVCacheSessionView` 并只前置声明其借用的 storage/session state；`KVCacheStorage` 与 `KVCacheSessionState` 位于 `detail/kv_cache_state.h`，由 `KVCacheManager` 按值持有，storage 按值拥有 layout 与 K/V Buffer。`Runtime` 按值聚合 Manager，并不拥有本模块的实现职责。KV owner/epoch 缺陷、CPU Attention 优化准入与 Paged KV 长期边界以 [08 号演进提案](../improvement-plan/08-kv-cache-and-attention-capability-evolution.md) 为当前计划入口；[05 号草案](../improvement-plan/05-kv-cache-manager-evolution.md) 已被取代。

**版本**: v1.1  
**日期**: 2026-03-25  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [职责与范围](#2-职责与范围)
3. [设计原则](#3-设计原则)
4. [核心概念与对象](#4-核心概念与对象)
5. [静态分配模型](#5-静态分配模型)
6. [布局与访问契约](#6-布局与访问契约)
7. [生命周期与状态变化](#7-生命周期与状态变化)
8. [与执行器和算子的交互](#8-与执行器和算子的交互)
9. [错误处理与验证](#9-错误处理与验证)
10. [Phase 2 演进边界](#10-phase-2-演进边界)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind Phase 1 的静态 KV Cache 设计。KV Cache 是 decoder-only Transformer 推理路径中的核心状态组件，负责在 Prefill 与 Decode 阶段保存历史 Key / Value，并向 Execution Layer 与 Attention 算子提供稳定访问边界。

本文档的目标不是引入复杂的分页式 KV 子系统，而是在 Phase 1 已冻结的边界内，给出一套**可直接指导 C++20 实现**的静态 KV 方案。

### 1.2 Phase 1 定位

Phase 1 的 KV Cache 采用：

- 单 Session 独占
- 静态预分配
- 连续地址空间
- 逻辑视图访问
- Decode 稳态零分配

它不承担以下职责：

- 分页管理
- 块回收
- 多请求共享
- prefix cache 复用
- 运行时动态扩容
- 调度器感知资源仲裁

### 1.3 在整体架构中的位置

KV Cache 是独立的 `kv_cache` 源模块；`Runtime` 仅聚合并按值拥有 `KVCacheManager`。它在运行时数据流中位于 Execution Layer 与 Operator Layer 之间：

```text
Executor / PrefillPath / DecodePath
                │
                ▼
           KVCacheSessionView
                │
                ▼
         KVCacheManager
          ├─ KVCacheStorage
          │   ├─ KVCacheLayout
          │   ├─ key_buffer
          │   └─ value_buffer
          └─ KVCacheSessionState
```

其中：

- `KVCacheManager` 负责 Session 级生命周期与逻辑绑定
- `KVCacheSessionView` 是 Execution / Attention 的稳定访问边界
- `KVCacheStorage` 将布局与对应 K/V Buffer 成套持有，View 借用 storage 与 session state
- `KVCacheLayout` 冻结布局、stride、alignment 与 offset 规则；`Validate()` 强制 `token_stride` 为元素大小倍数且对齐 `alignment`，并要求 `head_stride`/`layer_stride` 覆盖全部 token/head（防重叠），`KVCacheSessionView` 校验 layout 不超过 backing buffer

---

## 2. 职责与范围

### 2.1 核心职责

KVCacheManager 在 Phase 1 中承担以下核心职责：

- **静态持有**：由 `KVCacheManager` 在初始化阶段一次性预留 K/V 存储；`Runtime` 负责聚合其生命周期
- **逻辑绑定**：将当前 Session 与底层存储建立逻辑绑定
- **逻辑索引**：为执行层提供按 `layer / kv_head / position / dim` 的稳定访问方式
- **追加写入**：支持 Prefill 与 Decode 阶段按 token position 单调递增写入
- **只读回看**：支持 Attention 在当前步读取历史 K/V
- **边界保护**：检查层范围、head 范围、位置范围与提交上限合法性
- **视图构造**：通过 `KVCacheSessionView` 向上层暴露轻量 borrowed view
- **会话复用**：Session Reset 后复用底层存储，不重新申请物理内存

### 2.2 非目标

Phase 1 明确不做：

- **Paged KV / PagedAttention**
- **多 Session 共享 KV**
- **Prefix Sharing / Prefix Cache**
- **动态扩容**
- **多请求并发下的资源竞争控制**
- **热路径动态 block 分配**
- **CPU/GPU 分层 offload**

### 2.3 不承担的职责

KVCacheManager 不负责：

- 请求调度
- tokenization
- 模型加载与权重重排
- Attention 的数值计算
- Executor 的控制流决策
- Workspace 的生命周期管理

这些职责分别归属于 Runtime、Model、Execution 与 Operator 子系统。

---

## 3. 设计原则

### 3.1 静态存储优先

Phase 1 的 KV 物理内存在初始化或 Session Preparation 阶段一次性准备好。进入 Decode 稳态后：

- 不允许扩容
- 不允许重新分配
- 不允许引入新的 owning buffer

### 3.2 K / V 分离存储

K 与 V 使用两块独立的线性连续区域，而不是交错存储。这样做的原因是：

- 布局更直观
- offset 计算更简单
- 对 reference kernel 更友好
- 更利于后续局部替换为优化实现

### 3.3 View 边界稳定

Execution Layer 与 Operator Layer 一律通过 `KVCacheSessionView` 访问 KV 数据，而不是直接依赖底层物理布局。
`Attention` 必须通过 `KVCacheSessionView` 读取历史 KV，并通过写视图写入当前步 KV。

### 3.4 逻辑位置与物理地址分离

`seq_pos` 在 Phase 1 中表示**逻辑 token 位置**。虽然当前实现中它可以映射到连续线性偏移，但语义上不与未来分页式实现的物理地址绑定。

### 3.5 Manager 轻控制，View 重访问

KVCacheManager 负责：

- Init
- Reserve
- Reset
- Release
- 构造 `KVCacheSessionView`
- 合法性检查

真正的热路径访问尽量通过 `KVCacheSessionView` 直接完成，而不是在 Decode 每步反复回调 manager。

### 3.6 单请求场景下避免无意义并发设施

Phase 1 是 `single-request + synchronous execution`。  
因此 KVCacheManager 默认按**单控制流使用**设计：

- 不引入复杂锁
- 不引入无必要的原子同步
- 不为多请求共享预埋复杂状态机

同一 Manager 的 `Init`、`ReserveForSession`、`ResetSession` 与 `ReleaseSession` 不得在一次 `Execute()` 内插入，包括 kernel 重入；append transaction 也不得逃离执行栈。当前因此不引入 reset revision/token。该限制不是并发安全保证；若未来需要重入或并发，必须另行定义 transaction identity 与同步协议。

---

## 4. 核心概念与对象

### 4.1 对象总览

| 对象 | 职责 | 生命周期 |
|------|------|----------|
| **KVCacheManager** | 对外统一管理 KV 生命周期、Reserve / Reset / Release 与 View 构造 | Runtime / Engine 生命周期 |
| **KVCacheStorage** | Manager-owned 的布局与 K/V 底层线性存储（`detail/`） | Runtime / Engine 生命周期 |
| **KVCacheLayout** | 独立布局值类型，定义 dtype、stride、alignment 与 offset 计算规则；物理存储对应实例由 storage 按值拥有 | 与所属 storage 一致；也可在分配前独立构造验证 |
| **KVCacheSessionState** | Manager-owned 的 Session 逻辑占用状态记录（`detail/`） | Session 生命周期 |
| **KVCacheSessionView** | 面向 Executor / Attention 的逻辑访问视图 | Session 执行期 |

### 4.2 KVCacheManager

Phase 1 中的统一入口，职责包括：

- 根据模型配置构建 KV Layout
- 申请并持有底层 K / V 存储
- 为 Session 执行前准备逻辑绑定信息
- 构造 `KVCacheSessionView`
- 处理 Reset / Release
- 返回容量与内存统计信息

当前实现接口（与 `include/aethermind/kv_cache/kv_cache_manager.h` 同步）：

```cpp
class KVCacheManager {
public:
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) noexcept = default;
    KVCacheManager& operator=(KVCacheManager&&) noexcept = default;

    Status Init(Allocator& allocator,      // 只借用：构造期间使用，不保存引用
                size_t num_layers,
                size_t num_kv_heads,
                size_t max_tokens,
                size_t head_dim,
                DataType kv_dtype,
                size_t alignment = 64);

    StatusOr<KVCacheSessionView> ReserveForSession(size_t prompt_len,
                                                   size_t future_kv_appends) noexcept;

    Status ResetSession(const KVCacheSessionView& view) noexcept;
    Status ReleaseSession(KVCacheSessionView& view) noexcept;

    const KVCacheLayout& layout() const noexcept;
    size_t capacity_tokens() const noexcept;
    size_t total_bytes() const noexcept;
    bool is_initialized() const noexcept;
};
```

与历史草图的差异：`Init` 接收几何参数与 `Allocator&`（而非 `LoadedModel`/`Runtime`）；预约按 `prompt_len + future_kv_appends` 精确预留；Re-init 会释放旧 reservation 并保留 generation 使旧 view 永不复活。

Manager 按值持有 `storage_` 与 `session_state_`，不单独保存 layout；`layout()` 返回 `storage_.layout` 的 const 引用。`Init` 在局部构造完整 candidate storage，布局验证与两个 plane 分配均成功后，以不抛异常的移动整体发布。失败时 Manager 保持默认空 layout、零容量与零已分配字节，旧 view 持续失效；诊断分配 OOM 返回 `ResourceExhausted`。

### 4.3 KVCacheStorage

底层物理存储持有者，按值拥有一份布局及其对应的 K/V Buffer，不承载 Session 生命周期或提交权限。

当前字段：

```cpp
struct KVCacheStorage {
    KVCacheLayout layout{};
    Buffer key_buffer{};
    Buffer value_buffer{};
};
```

字节数和分配对齐信息由 Buffer 提供；dtype 以 `layout.kv_dtype` 为唯一权威，供 view 投影到 kernel binding，storage 不另存一份。组合将 layout 与 backing buffers 的所有权收束在一起；View 仍需校验布局、容量和 CPU plane 实际地址对齐。

### 4.4 KVCacheLayout

`KVCacheLayout` 保持独立的公开值类型，不依赖 Buffer，可在物理分配前构造、验证和计算字节数。它作为 storage 的成员负责冻结以下信息：

- `num_layers`
- `num_kv_heads`
- `max_tokens`
- `head_dim`
- `padded_head_dim`
- `token_stride`
- `head_stride`
- `layer_stride`
- `dtype`
- `alignment`

推荐接口：

```cpp
struct KVCacheLayout {
    std::size_t num_layers{0};
    std::size_t num_kv_heads{0};
    std::size_t max_tokens{0};

    std::size_t head_dim{0};
    std::size_t padded_head_dim{0};
    std::size_t token_stride{0};
    std::size_t head_stride{0};
    std::size_t layer_stride{0};

    DataType kv_dtype{DataType::kF16};
    std::size_t alignment{64};

    std::size_t ElementBytes() const noexcept;
    std::size_t Offset(std::size_t layer_idx,
                       std::size_t kv_head_idx,
                       std::size_t seq_pos,
                       std::size_t dim_idx) const noexcept;
};
```

### 4.5 KVCacheSessionState（内部状态）

`KVCacheSessionState` 是 `KVCacheManager` 按值拥有的 `detail/` 类型，不是公开 handle。当前状态以生命周期枚举表达互斥的 reservation 与执行阶段：

```cpp
struct KVCacheSessionState {
    enum class Lifecycle : std::uint8_t {
        kIdle,
        kAwaitingPrefill,
        kPrefillCommitted,
    };

    std::uint64_t generation{0};
    std::size_t capacity_tokens{0};
    std::size_t prompt_len{0};
    std::size_t committed_tokens{0};
    Lifecycle lifecycle{Lifecycle::kIdle};
};
```

`generation` 是 manager-owned epoch：`Init` 保留它，Reserve/Release 推进它，从而令旧 view 永不复活。`ResetSession` 保留 epoch、reservation 和 view 有效性，只将 `committed_tokens` 归零并回到 `AwaitingPrefill`。`capacity_tokens`、`prompt_len` 和 `committed_tokens` 分别表达 reservation 上限、首次 Prefill 长度和跨 plan 已发布 watermark，不能互相推导。

### 4.6 KVCacheSessionView

Execution / Operator 看到的是轻量 borrowed view，而不是 manager 本体。

当前实现能力（与 `include/aethermind/kv_cache/kv_cache_session_view.h` 同步，省略 `AM_NODISCARD` 等细节）：

```cpp
class KVCacheSessionView {
public:
    KVCacheSessionView() = default;
    KVCacheSessionView(KVCacheStorage* storage,
                       KVCacheSessionState* session_state) noexcept;

    bool valid() const noexcept;

    size_t max_tokens() const noexcept;
    size_t current_pos() const noexcept;
    size_t num_layers() const noexcept;
    size_t num_kv_heads() const noexcept;
    size_t head_dim() const noexcept;
    DataType kv_dtype() const noexcept;
    size_t token_capacity() const noexcept;
    size_t committed_tokens() const noexcept;
    size_t prompt_len() const noexcept;
    bool awaiting_prefill() const noexcept;

    Status ValidateWrite(size_t layer_idx, size_t kv_head_idx,
                         size_t seq_pos, size_t token_count) const noexcept;

    Status ValidateRead(size_t layer_idx, size_t kv_head_idx,
                        size_t seq_begin, size_t seq_end) const noexcept;

    StatusOr<KVCacheAppendBinding> BindLayerForAppend(
            size_t layer_idx, size_t begin, size_t end) const noexcept;

    StatusOr<KVCacheReadBinding> BindLayerForRead(
            size_t layer_idx, size_t committed_end, size_t visible_end) const noexcept;

    StatusOr<const void*> KeyData(size_t layer_idx, size_t kv_head_idx,
                                  size_t seq_pos, size_t dim_idx = 0) const noexcept;
    StatusOr<const void*> ValueData(size_t layer_idx, size_t kv_head_idx,
                                    size_t seq_pos, size_t dim_idx = 0) const noexcept;

    void Invalidate() noexcept;
};
```

与历史草图的差异：历史 `KeyWriteView / ValueWriteView / KeyReadView / ValueReadView` 未落地；写路径统一为 `BindLayerForAppend`（强制 `begin == committed` 前沿），元素访问器只读，`ValidateWrite` 也仅做容量/范围校验。CPU plane 还会校验 base 地址满足 layout alignment（见 §6.5）。

View 仅保存 borrowed storage/session-state 指针及捕获的 generation，从 `storage.layout` 读取几何信息，不独立接收或保存 layout 指针，也不在公开头文件暴露 Buffer。Manager 必须长于全部 view（含 stale 副本）且保持地址稳定；同一 Manager 与 view 的访问仍由调用方串行化。

View 的 `current_pos()` 返回 canonical `committed_tokens`；`CommitAppend(begin, end) const noexcept` 是公开提交接口。调用方必须在完整 append plan 成功且所有 layer 写完后调用它；方法只验证 view generation、lifecycle、`begin/end` 和 reservation/physical capacity，不证明 layer 写入完成。`LayerRunner` 仍负责整计划 transaction，在所有 layer 成功后发布同一 append range。View 只借用 manager-owned storage/session state，`const` 不表示提交不会修改被借用的 session state。

### 4.7 Phase 1 简化原则

虽然保留 `KVCacheStorage / KVCacheLayout / KVCacheSessionState / KVCacheSessionView` 这些边界，但 Phase 1 实现要求仍然克制：

- 不要求设计复杂 allocator
- 不要求 block table
- 不要求分页式 slot 管理
- 可以只有单个 `KVCacheSessionState`
- 只要能稳定暴露视图、正确维护逻辑位置并满足零分配，即满足 Phase 1 目标

---

## 5. 静态分配模型

### 5.1 基本策略

KV Cache 采用 **Session 执行前一次性预分配** 模式：

- 容量由 `max_tokens`、层数、KV 头数、`head_dim`、`dtype` 决定
- K 与 V 使用独立连续区域
- Session 生命周期内不扩容
- Reset 仅清空逻辑状态，不重新申请物理内存

### 5.2 推荐容量模型

定义：

- `L = num_layers`
- `Hkv = num_kv_heads`
- `T = max_tokens`
- `D = padded_head_dim`
- `B = sizeof(kv_dtype)`

则：

```text
bytes_key   = L * Hkv * T * D * B
bytes_value = L * Hkv * T * D * B
total_bytes = bytes_key + bytes_value
```

其中：

```text
max_tokens = prompt_len_limit + max_new_tokens_limit
```

这里的 `max_tokens` 表示 **KV 物理容量上限**。在产品与配置语义上，它对应于上下文/生成上限组合后的总 token 容量；具体默认上限与约束应与 PRD 和 runtime config 保持一致。

Phase 1 推荐在 Runtime / Model 初始化阶段固定 `max_tokens` 上限，而不是为不同 Session 动态申请不同大小的 KV 池。

### 5.3 padded_head_dim

引入 `padded_head_dim`；初版即按 `alignment` 对 token 行做对齐：`padded_head_dim = AlignUp(head_dim, alignment / element_bytes)`（`element_bytes > alignment` 时保持 `head_dim`），使 `token_stride` 为 alignment 倍数、每个 token 行起始对齐，`head_stride`/`layer_stride` 因其整数倍关系自动对齐。这样做的原因是：

- 为 SIMD / cache line 对齐留空间
- 为后续优化 kernel 保留 padding 能力
- 不需要改动上层视图语义

### 5.4 预留时机

推荐流程：

1. Runtime / LoadedModel 初始化
2. KVCacheManager 根据模型维度构建局部 candidate storage 的 `KVCacheLayout`
3. 一次性申请 K / V 线性存储，成功后整体发布 candidate storage
4. InferenceSession 在会话准备阶段调用 `ReserveForSession(prompt_len, future_kv_appends)`
5. 返回 `KVCacheSessionView`
6. Session 进入 Prefill / Decode

### 5.5 Decode 稳态零分配

Phase 1 的 KV 设计必须满足：

- Prefill 前允许一次性预分配
- Decode 循环中不允许新增堆分配
- `KVCacheSessionView` 的获取与使用不触发新的 owning allocation
- Reset 仅重置逻辑状态，不重新申请物理内存

---

## 6. 布局与访问契约

### 6.1 逻辑索引维度

Phase 1 至少需要支持以下逻辑维度：

- `layer_idx`
- `kv_head_idx`
- `seq_pos`
- `head_dim_idx`

### 6.2 Phase 1 默认实现建议

Phase 1 的默认实现可优先采用如下物理布局：

```text
K: [layer][kv_head][seq_pos][head_dim_padded]
V: [layer][kv_head][seq_pos][head_dim_padded]
```

即：

- `Layer-major`
- `KV-head-major`
- `Token-major`
- `Dim-minor`
- K/V split

这是 Phase 1 最简单、最稳定且最利于 reference kernel 的布局之一，但本设计冻结的重点是**逻辑访问维度与可计算性**，而非唯一物理排布。

### 6.3 Offset 公式（默认实现示意）

若采用上述默认布局，则逻辑 offset 可定义为：

```text
offset =
(((layer_idx * num_kv_heads + kv_head_idx) * max_tokens + seq_pos)
 * padded_head_dim + dim_idx)
```

该公式可由 `KVCacheLayout::Offset(...)` 统一提供。执行层与算子层不得散落硬编码的魔法常量，但这并不要求 Phase 1 将某一个物理 offset 公式写死为唯一合法实现。

### 6.4 布局要求

Phase 1 必须满足以下要求：

- K 与 V 分开存储
- 写入顺序按 `seq_pos` 单调递增
- 相同 Session 的历史范围可连续读取
- 布局可支撑按层访问与按历史区间读取
- 允许通过 `padded_head_dim` 实现对齐与 padding

### 6.5 KVCacheSessionView 访问原则

- 执行层与算子层通过 `KVCacheSessionView` 访问 KV 数据
- View 对 CPU plane 校验 base 地址满足 layout alignment，使 token 行相对进程地址
  空间绝对对齐；layout 本身只保证行相对 plane base 的 stride 对齐
- 算子签名中不直接暴露底层物理偏移常量
- `seq_pos` 表示逻辑 token 位置
- `BindLayerForAppend` 返回当前 append 写窗口
- `BindLayerForRead` 返回历史区间读窗口
- `LayerRunner` 只在 Prefill / Decode 的完整 plan 成功后，经 `KVCacheSessionView::CommitAppend` 推进逻辑位置；调用前必须证明所有 layer 已写完
- `KVCacheAppendBinding` / `KVCacheReadBinding` 是 kernel invocation-only 的 borrowed
  contract；其物理指针不能缓存进 prepared params，也不能跨 Execute() 保存

### 6.6 历史读取与当前写入语义

- Prefill：批量写入 `[0, prompt_len)`
- Decode：每步写入 `[committed_tokens, committed_tokens + 1)`
- 历史读取区间统一采用左闭右开表示 `[seq_begin, seq_end)`

### 6.7 对 Future Paged KV 的兼容边界

Phase 1 当前采用静态线性布局，但算子和执行层看到的永远是逻辑视图。  
后续如果切换到分页式实现，应优先替换底层存储与 offset 解析逻辑，并尽量保持 `KVCacheSessionView` 上层语义稳定。

---

## 7. 生命周期与状态变化

### 7.1 生命周期绑定

KV Cache 与单次生成 Session 逻辑绑定：

- Runtime / Engine 生命周期内持有底层物理存储
- Session 创建后在 `PrepareSession()` 阶段完成逻辑 Reserve
- Prefill / Decode 期间通过 `KVCacheSessionView` 读写
- Session Reset 时清空已提交位置、保留 reservation 与现有 view 的有效性
- Session Release 时解除逻辑占用并推进 generation，使旧 view 失效

### 7.2 状态阶段

推荐状态流如下：

```text
Uninitialized
    -> Init
Initialized
    -> ReserveForSession
AwaitingPrefill
    -> Prefill transaction succeeds
PrefillCommitted
    -> Decode transaction succeeds
PrefillCommitted
    -> ResetSession
AwaitingPrefill
    -> ReleaseSession
Initialized
```

### 7.3 ReserveForSession

`ReserveForSession(prompt_len, future_kv_appends)` 需要完成：

- 校验 manager 已初始化
- 校验 `prompt_len + future_kv_appends <= max_tokens`
- Phase 1 拒绝空 `prompt_len`；BOS/空上下文策略属于更高层输入语义
- 校验 lifecycle 为 `Idle`
- 将 lifecycle 设为 `AwaitingPrefill`
- 记录 `prompt_len`
- 设置 `committed_tokens = 0`
- 记录 `AwaitingPrefill`；`prompt_len` 和 reservation capacity 是 immutable metadata，
  不能由 `committed_tokens` 代替
- 返回 `KVCacheSessionView`

### 7.4 Append

#### Prefill

- 以批量方式写入 `0 .. prompt_len-1`
- 一个 execution plan 的所有 decoder layer 必须恰好写入一次相同区间
  `[0, prompt_len)`；仅当整个 plan 成功且所有 layer 已写完后由 `LayerRunner` 调用
  `KVCacheSessionView::CommitAppend` 发布该区间

#### Decode

- 每步写入当前 `committed_tokens`
- Phase 1 每个 Decode transaction 只接受一个 token；仅当整个 plan 成功且所有 layer 已写完后由 `LayerRunner` 调用 `KVCacheSessionView::CommitAppend` 发布 `[committed_tokens, committed_tokens + 1)`

### 7.5 Read

- `[0, committed_end)` 是跨 plan 已提交范围，可由普通 `KVCacheSessionView` read API 读取
- `[committed_end, visible_end)` 是当前同步 plan 已写但尚未提交的范围；只允许同一
  transaction 内、且本 layer 的 KVCacheUpdate 已成功之后的 Attention 读取
- `[visible_end, capacity)` 不可读取。Attention-before-update 不能通过扩大
  `visible_end` 读取未初始化内容

### 7.6 Reset

`ResetSession()` 的语义：

- 清空逻辑位置
- 将 `committed_tokens` 设为 0 并回到 `AwaitingPrefill`
- 保留 `prompt_len` 和 reservation capacity；下一次执行必须重新 Prefill，
  Reset 不表达“保留 prompt KV 后重跑 Decode”
- 不推进 generation；同一 session 的既有 view 副本仍有效
- 保留已分配的底层 K/V 存储
- 不触发新的堆分配
- 不强制对底层存储做 `memset`

### 7.7 Release

`ReleaseSession()` 的语义：

- 解除当前 Session 的逻辑占用
- 递增 `generation`
- 使已有 `KVCacheSessionView` 自动失效
- 回到可再次 Reserve 的状态

### 7.8 为什么不强制清零物理内存

Phase 1 的主要目标是：

- 正确性
- 零分配
- 可复用

不是安全擦除。如果未来需要安全擦除策略，可由更高层配置控制。

---

## 8. 与执行器和算子的交互

### 8.1 与 Executor 的交互

Executor 不应直接操作底层物理内存。推荐交互方式如下：

- `PrepareSession()` 阶段：
  - 调用 `KVCacheManager::ReserveForSession(...)`
  - 获取 `KVCacheSessionView`
  - 将 `KVCacheSessionView` 放入 `SessionExecutionState`
- `Prefill()` 阶段：
  - 通过 `BindLayerForAppend` 的写窗口写入整段 Prompt 的 KV
  - 完整 plan 成功且所有 layer 写完后由 `LayerRunner` 调用 `KVCacheSessionView::CommitAppend` 发布 `[0, prompt_len)`
- `DecodeStep()` 阶段：
  - 通过 `BindLayerForAppend` 写入当前 step 的 KV
  - 通过 `BindLayerForRead` 读取历史 KV
  - 完整 plan 成功且所有 layer 写完后由 `LayerRunner` 调用 `KVCacheSessionView::CommitAppend` 发布单 token append range

### 8.2 与 PrefillPath / DecodePath 的边界

- `PrefillPath` 关心批量写入区间
- `DecodePath` 关心当前单步位置与历史读取区间
- 二者都不应自行计算底层 offset
- 二者都只通过 `KVCacheSessionView` 读写 KV

### 8.3 与 Attention 的交互

Attention 算子需要：

- 读取历史 K/V
- 写入当前 step 的 K/V

因此 KV 接口必须同时支持：

- **历史区间视图**
- **当前写入位置视图**

并且应遵守以下约束：

- 算子签名不暴露底层物理地址公式
- 算子看到的是 borrowed view，而不是 owning buffer
- `seq_pos` 是逻辑位置，而不是页地址或 block 偏移

### 8.4 与 Workspace 的边界

- KV Cache 持有持久态历史状态
- Workspace 持有算子执行期临时中间结果
- 二者必须清晰分离，避免临时缓冲污染 KV 生命周期
- Q / K / V 临时 scratch 不属于 KVCacheManager 管理范围

### 8.5 与 Layout Contract 的配合

Execution / Operator 不应自行维护 stride 常量。所有布局相关信息均从 `KVCacheLayout` 获取：

- 层步长
- 头步长
- token 步长
- dim padding
- element bytes

---

## 9. 错误处理与验证

### 9.1 错误处理原则

推荐通过 `Status` / `StatusOr` / `Expected` 风格返回错误，而不是在热路径抛异常。

### 9.2 推荐错误类型

KV Cache 至少需要处理以下错误：

- `NotInitialized`
- `AlreadyInUse`
- `CapacityExceeded`
- `InvalidArgument`
- `InvalidView`
- `GenerationMismatch`
- `OutOfRange`
- `InternalError`

示意枚举：

```cpp
enum class KVErrorCode : std::uint8_t {
    kOk,
    kNotInitialized,
    kAlreadyInUse,
    kCapacityExceeded,
    kInvalidArgument,
    kInvalidView,
    kGenerationMismatch,
    kOutOfRange,
    kInternalError,
};
```

### 9.3 Preparation 阶段错误

典型包括：

- `prompt_len + future_kv_appends > max_tokens`
- manager 未初始化
- 当前 slot 已占用
- 参数为 0 或非法上限

### 9.4 执行阶段错误

典型包括：

- `layer_idx` 越界
- `kv_head_idx` 越界
- `seq_pos` 越界
- 读取超出已提交范围
- `CommitAppend` 的 generation、begin/end、容量或 lifecycle 校验失败
- stale view 被继续使用

### 9.5 测试与验证

建议至少覆盖以下验证：

#### 1. 布局正确性

- 验证 `layer / kv_head / pos / dim` 的 offset 计算正确
- 验证 K / V 的独立区域不重叠
- 验证 `padded_head_dim` / padding 生效

#### 2. Prefill / Decode 语义正确性

- Prefill 后提交位置为 `prompt_len`
- Decode 第一步写入 `committed_tokens = prompt_len`
- 历史读取范围与已提交位置一致

#### 3. 越界保护

- 非法层号 / head / pos 输入应返回错误
- 读取未提交区间应返回错误

#### 4. Reset / Release 行为

- Reset 后逻辑状态清空
- Reset 不触发重新分配
- Release 后旧 view 因 `generation mismatch` 失效

#### 5. 零分配验证

- Decode 稳态路径中无新增堆分配
- View 获取不触发 owning allocation
- 热路径无 `std::vector` 扩容

#### 6. 与 Attention 的集成验证

- 验证 Attention 对历史 KV 的读取与当前步写入一致
- 验证通过 `KVCacheSessionView` 读写的控制流与 reference kernel 预期一致

---

## 10. Phase 2 演进边界

Phase 1 的 KV 设计应为以下能力保留兼容边界：

- `Paged KV Cache`
- `PagedAttention`
- `Prefix Cache`
- `Continuous Batching`

但 Phase 1 不要求现在就实现这些能力。

### 10.1 推荐演进方式

更合理的演进方式是：

1. 保持 `KVCacheSessionView` 上层访问语义稳定
2. 保持 `KVCacheLayout` 作为布局契约入口
3. 在 Phase 2 替换底层 `KVCacheStorage` 实现
4. 再逐步引入 block / page 管理与共享策略
5. 最后扩展到多 Session / batching 场景

### 10.2 不建议的演进方式

不建议直接让：

- Executor 自行计算物理 offset
- Attention 直接依赖线性地址公式
- Session 控制流直接持有底层裸指针

否则一旦切换到分页式 KV，Execution / Operator 层会整体返工。

### 10.3 Phase 1 的冻结结论

因此，Phase 1 的 KV 设计冻结为：

- **静态线性底层存储**
- **单 Session 逻辑独占**
- **K/V split**
- **通过 KVCacheSessionView 暴露访问边界**
- **通过 KVCacheLayout 冻结逻辑布局契约，而非唯一物理布局实现**
- **Decode 稳态零分配**
- **后续优先替换底层存储实现，而不是改动上层执行语义**

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: M1 里程碑完成后补充实践经验
