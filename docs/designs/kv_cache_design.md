# AetherMind Phase 1 KV Cache 设计文档

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

KV Cache 子系统在整体架构中处于 Execution Layer 与 Operator Layer 之间：

```text
Executor / PrefillPath / DecodePath
                │
                ▼
           KVCacheView
                │
                ▼
         KVCacheManager
          ├─ KVCacheStorage
          ├─ KVLayoutContract
          └─ (optional) SessionKVSlot
```

其中：

- `KVCacheManager` 负责 Session 级生命周期与逻辑绑定
- `KVCacheView` 是 Execution / Attention 的稳定访问边界
- `KVCacheStorage` 持有底层物理存储
- `KVLayoutContract` 冻结布局、stride、alignment 与 offset 规则

---

## 2. 职责与范围

### 2.1 核心职责

KVCacheManager 在 Phase 1 中承担以下核心职责：

- **静态持有**：在 Runtime / Session Preparation 阶段一次性预留 K/V 存储
- **逻辑绑定**：将当前 Session 与底层存储建立逻辑绑定
- **逻辑索引**：为执行层提供按 `layer / kv_head / position / dim` 的稳定访问方式
- **追加写入**：支持 Prefill 与 Decode 阶段按 token position 单调递增写入
- **只读回看**：支持 Attention 在当前步读取历史 K/V
- **边界保护**：检查层范围、head 范围、位置范围与提交上限合法性
- **视图构造**：通过 `KVCacheView` 向上层暴露轻量 borrowed view
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

Execution Layer 与 Operator Layer 一律通过 `KVCacheView` 访问 KV 数据，而不是直接依赖底层物理布局。  
`Attention` 必须通过 `KVCacheView` 读取历史 KV，并通过写视图写入当前步 KV。

### 3.4 逻辑位置与物理地址分离

`seq_pos` 在 Phase 1 中表示**逻辑 token 位置**。虽然当前实现中它可以映射到连续线性偏移，但语义上不与未来分页式实现的物理地址绑定。

### 3.5 Manager 轻控制，View 重访问

KVCacheManager 负责：

- Init
- Reserve
- Reset
- Release
- 构造 `KVCacheView`
- 合法性检查

真正的热路径访问尽量通过 `KVCacheView` 直接完成，而不是在 Decode 每步反复回调 manager。

### 3.6 单请求场景下避免无意义并发设施

Phase 1 是 `single-request + synchronous execution`。  
因此 KVCacheManager 默认按**单控制流使用**设计：

- 不引入复杂锁
- 不引入无必要的原子同步
- 不为多请求共享预埋复杂状态机

---

## 4. 核心概念与对象

### 4.1 对象总览

| 对象 | 职责 | 生命周期 |
|------|------|----------|
| **KVCacheManager** | 对外统一管理 KV 生命周期、Reserve / Reset / Release 与 View 构造 | Runtime / Engine 生命周期 |
| **KVCacheStorage** | 持有 K/V 底层线性存储 | Runtime / Engine 生命周期 |
| **KVLayoutContract** | 定义布局、stride、alignment 与 offset 计算规则 | Runtime / Engine 生命周期 |
| **SessionKVSlot** | 可选的 Session 逻辑占用状态记录 | Session 生命周期 |
| **SessionKVHandle** | 可选的 Session 逻辑持有句柄 | Session 生命周期 |
| **KVCacheView** | 面向 Executor / Attention 的逻辑访问视图 | Session 执行期 |

### 4.2 KVCacheManager

Phase 1 中的统一入口，职责包括：

- 根据模型配置构建 KV Layout
- 申请并持有底层 K / V 存储
- 为 Session 执行前准备逻辑绑定信息
- 构造 `KVCacheView`
- 处理 Reset / Release
- 返回容量与内存统计信息

推荐接口形态：

```cpp
class KVCacheManager {
public:
    Status Init(const LoadedModel& model,
                const RuntimeContext& runtime,
                std::size_t max_tokens,
                DataType kv_dtype,
                std::size_t alignment = 64);

    StatusOr<KVCacheView> ReserveForSession(std::size_t prompt_len,
                                            std::size_t max_new_tokens);

    Status ResetSession(KVCacheView& view) noexcept;
    Status ReleaseSession(KVCacheView& view) noexcept;

    const KVLayoutContract& layout() const noexcept;
    std::size_t capacity_tokens() const noexcept;
    std::size_t total_bytes() const noexcept;
};
```

### 4.3 KVCacheStorage

底层物理存储持有者，仅负责内存，不表达高层语义。

建议字段：

```cpp
struct KVCacheStorage {
    void* key_base{nullptr};
    void* value_base{nullptr};

    std::size_t bytes_per_plane{0};
    std::size_t total_bytes{0};

    DataType kv_dtype{DataType::kF16};
    std::size_t alignment{64};
};
```

> 注：KV 元素格式以 PRD 为准，Phase 1 支持 `FP16 / BF16` 作为 KV 存储格式；此处仅以 `kF16` 作为默认示意值。

### 4.4 KVLayoutContract

布局契约负责冻结以下信息：

- `num_layers`
- `num_kv_heads`
- `max_tokens`
- `head_dim`
- `head_dim_stride`
- `token_stride`
- `head_stride`
- `layer_stride`
- `dtype`
- `alignment`

推荐接口：

```cpp
struct KVLayoutContract {
    std::size_t num_layers{0};
    std::size_t num_kv_heads{0};
    std::size_t max_tokens{0};

    std::size_t head_dim{0};
    std::size_t head_dim_stride{0};
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

### 4.5 SessionKVSlot / SessionKVHandle（可选内部机制）

若实现需要防止 stale view 在 Reset / Release 后继续被误用，可引入轻量 slot / handle 概念。它们是 **Phase 1 的可选内部安全增强机制**，而不是必须先冻结的公开对象体系。

推荐结构：

```cpp
struct SessionKVSlot {
    std::uint64_t generation{0};
    bool in_use{false};

    std::size_t capacity_tokens{0};
    std::size_t prompt_len{0};
    std::size_t current_pos{0};
};

struct SessionKVHandle {
    SessionKVSlot* slot{nullptr};
    std::uint64_t generation{0};
};
```

`generation` 的目标是防止 stale view 在 Reset / Release 后继续被误用；若初版实现希望保持最小复杂度，也可先不引入该机制。

### 4.6 KVCacheView

Execution / Operator 看到的是轻量 borrowed view，而不是 manager 本体。

推荐能力：

```cpp
class KVCacheView {
public:
    bool valid() const noexcept;

    std::size_t max_tokens() const noexcept;
    std::size_t current_pos() const noexcept;
    std::size_t num_layers() const noexcept;
    std::size_t num_kv_heads() const noexcept;
    std::size_t head_dim() const noexcept;

    Status ValidateWrite(std::size_t layer_idx,
                         std::size_t kv_head_idx,
                         std::size_t seq_pos,
                         std::size_t token_count) const noexcept;

    StatusOr<MutableTensorView> KeyWriteView(std::size_t layer_idx,
                                             std::size_t kv_head_idx,
                                             std::size_t seq_pos,
                                             std::size_t token_count) noexcept;

    StatusOr<MutableTensorView> ValueWriteView(std::size_t layer_idx,
                                               std::size_t kv_head_idx,
                                               std::size_t seq_pos,
                                               std::size_t token_count) noexcept;

    StatusOr<TensorView> KeyReadView(std::size_t layer_idx,
                                     std::size_t kv_head_idx,
                                     std::size_t seq_begin,
                                     std::size_t seq_end) const noexcept;

    StatusOr<TensorView> ValueReadView(std::size_t layer_idx,
                                       std::size_t kv_head_idx,
                                       std::size_t seq_begin,
                                       std::size_t seq_end) const noexcept;

    Status CommitUntil(std::size_t new_pos) noexcept;
};
```

### 4.7 Phase 1 简化原则

虽然保留 `KVCacheStorage / KVLayoutContract / SessionKVSlot / KVCacheView` 这些边界，但 Phase 1 实现要求仍然克制：

- 不要求设计复杂 allocator
- 不要求 block table
- 不要求分页式 slot 管理
- 可以只有单个 `SessionKVSlot`
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
- `D = head_dim_stride`
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

### 5.3 head_dim_stride

引入 `head_dim_stride`，即便初版令其等于 `head_dim`。这样做的原因是：

- 为 SIMD / cache line 对齐留空间
- 为后续优化 kernel 保留 padding 能力
- 不需要改动上层视图语义

### 5.4 预留时机

推荐流程：

1. Runtime / LoadedModel 初始化
2. KVCacheManager 根据模型维度构建 `KVLayoutContract`
3. 一次性申请 K / V 线性存储
4. Executor 在 `PrepareSession()` 中调用 `ReserveForSession(prompt_len, max_new_tokens)`
5. 返回 `KVCacheView`
6. Session 进入 Prefill / Decode

### 5.5 Decode 稳态零分配

Phase 1 的 KV 设计必须满足：

- Prefill 前允许一次性预分配
- Decode 循环中不允许新增堆分配
- `KVCacheView` 的获取与使用不触发新的 owning allocation
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
 * head_dim_stride + dim_idx)
```

该公式可由 `KVLayoutContract::Offset(...)` 统一提供。执行层与算子层不得散落硬编码的魔法常量，但这并不要求 Phase 1 将某一个物理 offset 公式写死为唯一合法实现。

### 6.4 布局要求

Phase 1 必须满足以下要求：

- K 与 V 分开存储
- 写入顺序按 `seq_pos` 单调递增
- 相同 Session 的历史范围可连续读取
- 布局可支撑按层访问与按历史区间读取
- 允许通过 `head_dim_stride` 实现对齐与 padding

### 6.5 KVCacheView 访问原则

- 执行层与算子层通过 `KVCacheView` 访问 KV 数据
- 算子签名中不直接暴露底层物理偏移常量
- `seq_pos` 表示逻辑 token 位置
- `KeyWriteView` / `ValueWriteView` 返回当前写窗口
- `KeyReadView` / `ValueReadView` 返回历史区间视图
- `CommitUntil(new_pos)` 只在 Prefill / Decode 成功后推进逻辑位置

### 6.6 历史读取与当前写入语义

- Prefill：批量写入 `[0, prompt_len)`
- Decode：每步写入 `[current_pos, current_pos + 1)`
- 历史读取区间统一采用左闭右开表示 `[seq_begin, seq_end)`

### 6.7 对 Future Paged KV 的兼容边界

Phase 1 当前采用静态线性布局，但算子和执行层看到的永远是逻辑视图。  
后续如果切换到分页式实现，应优先替换底层存储与 offset 解析逻辑，并尽量保持 `KVCacheView` 上层语义稳定。

---

## 7. 生命周期与状态变化

### 7.1 生命周期绑定

KV Cache 与单次生成 Session 逻辑绑定：

- Runtime / Engine 生命周期内持有底层物理存储
- Session 创建后在 `PrepareSession()` 阶段完成逻辑 Reserve
- Prefill / Decode 期间通过 `KVCacheView` 读写
- Session Reset 时清空逻辑状态
- Session Release 时解除逻辑占用；若启用了 generation 机制，则使旧 view 失效

### 7.2 状态阶段

推荐状态流如下：

```text
Uninitialized
    -> Init
Initialized
    -> ReserveForSession
Reserved
    -> Prefill / Decode
Active
    -> ResetSession
Reserved
    -> ReleaseSession
Initialized
```

### 7.3 ReserveForSession

`ReserveForSession(prompt_len, max_new_tokens)` 需要完成：

- 校验 manager 已初始化
- 校验 `prompt_len + max_new_tokens <= max_tokens`
- 校验当前 slot 未被占用
- 标记 slot `in_use = true`
- 记录 `prompt_len`
- 设置 `current_pos = 0`
- 返回 `KVCacheView`

### 7.4 Append

#### Prefill

- 以批量方式写入 `0 .. prompt_len-1`
- 完成后调用 `CommitUntil(prompt_len)`

#### Decode

- 每步写入当前 `current_pos`
- 成功后调用 `CommitUntil(current_pos + 1)`

### 7.5 Read

- Attention 读取 `[0, current_pos)` 的历史范围
- 读取必须服从当前已提交位置上限
- 不允许读取未提交写入区间

### 7.6 Reset

`ResetSession()` 的语义：

- 清空逻辑位置
- 清空 prompt_len / current_pos 等逻辑状态
- 保留已分配的底层 K/V 存储
- 不触发新的堆分配
- 不强制对底层存储做 `memset`

### 7.7 Release

`ReleaseSession()` 的语义：

- 解除当前 Session 的逻辑占用
- 若实现启用了 generation 机制，则递增 `generation`
- 若实现启用了 generation 机制，则使已有 `KVCacheView` 自动失效
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
  - 获取 `KVCacheView`
  - 将 `KVCacheView` 放入 `SessionExecutionState`
- `Prefill()` 阶段：
  - 通过 `KeyWriteView / ValueWriteView` 写入整段 Prompt 的 KV
  - 成功后 `CommitUntil(prompt_len)`
- `DecodeStep()` 阶段：
  - 通过写视图写入当前 step 的 KV
  - 通过读视图读取历史 KV
  - 成功后 `CommitUntil(current_pos + 1)`

### 8.2 与 PrefillPath / DecodePath 的边界

- `PrefillPath` 关心批量写入区间
- `DecodePath` 关心当前单步位置与历史读取区间
- 二者都不应自行计算底层 offset
- 二者都只通过 `KVCacheView` 读写 KV

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

Execution / Operator 不应自行维护 stride 常量。所有布局相关信息均从 `KVLayoutContract` 获取：

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

- `prompt_len + max_new_tokens > max_tokens`
- manager 未初始化
- 当前 slot 已占用
- 参数为 0 或非法上限

### 9.4 执行阶段错误

典型包括：

- `layer_idx` 越界
- `kv_head_idx` 越界
- `seq_pos` 越界
- 读取超出已提交范围
- `CommitUntil()` 回退
- stale view 被继续使用

### 9.5 测试与验证

建议至少覆盖以下验证：

#### 1. 布局正确性

- 验证 `layer / kv_head / pos / dim` 的 offset 计算正确
- 验证 K / V 的独立区域不重叠
- 验证 `head_dim_stride` / padding 生效

#### 2. Prefill / Decode 语义正确性

- Prefill 后提交位置为 `prompt_len`
- Decode 第一步写入 `current_pos = prompt_len`
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
- 验证通过 `KVCacheView` 读写的控制流与 reference kernel 预期一致

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

1. 保持 `KVCacheView` 上层访问语义稳定
2. 保持 `KVLayoutContract` 作为布局契约入口
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
- **通过 KVCacheView 暴露访问边界**
- **通过 KVLayoutContract 冻结逻辑布局契约，而非唯一物理布局实现**
- **Decode 稳态零分配**
- **后续优先替换底层存储实现，而不是改动上层执行语义**

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: M1 里程碑完成后补充实践经验
