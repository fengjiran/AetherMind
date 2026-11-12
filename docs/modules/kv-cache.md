# kv-cache 模块设计

> 本文描述已验证实现。系统分层与依赖方向见 [architecture.md](../architecture.md)，Runtime 对 Manager 的聚合见 [runtime.md](runtime.md)，执行期事务与 kernel 调用见 [execution.md](execution.md)，产品边界见 [product.md](../product.md)。

## 1. 职责与边界

kv-cache 是**静态 KV 物理状态域**：管理一份连续 K/V plane 的布局、分配、单 session 预约、提交水位和窄读写视图。它不感知模型语义、请求调度或 Attention 数值计算。

| 负责 | 说明 |
|---|---|
| 静态存储 | 按 `num_layers / num_kv_heads / max_tokens / head_dim / kv_dtype / alignment` 规划布局，一次性分配 K/V 两个 plane |
| 布局契约 | `KVCacheLayout` 冻结 stride、padding 与 offset 计算规则，并在分配前验证几何和字节可表示性 |
| 单 session 预约 | `ReserveForSession` 在物理容量内记录 `prompt_len` 与 `capacity_tokens`，同一时刻只允许一个活动预约 |
| 提交水位 | `CommitAppend` 推进 `committed_tokens`，把当前 plan 已完成的 append range 发布给后续 plan |
| 窄访问边界 | `KVCacheSessionView` 借出按层 append/read binding 与只读元素指针；执行层不接触裸 `Buffer` 或物理 offset 常量 |
| 生命周期复用 | `ResetSession` 保留预约与物理存储；`ReleaseSession` 解除预约、推进 generation 并使旧 view 失效 |

**不得负责什么**：

| 不负责 | 实际归属 |
|---|---|
| Paged KV、block table、block 回收、动态扩容、CPU/GPU offload | 当前产品明确不做，见 [architecture.md](../architecture.md) 产品边界 |
| 多 session 共享、prefix cache、跨请求 KV 复用、请求排队 | 当前产品边界外；本模块只有单 session slot |
| whole-plan append transaction、layer 完成判定、失败回滚 | execution 的 `LayerRunner`；本模块只提供单次 append/commit 原语 |
| Attention 数值计算、K/V narrow/widen、kernel dispatch | operators / backend |
| 模型加载、权重转换、图构建、kernel resolve | model / compiler / execution / backend |
| workspace 与 activation 的分配、生命周期 | execution / `WorkspaceArena` |
| KV dtype 的产品准入（KV16 是否允许） | inference 的 `ExecutableModel::Load` 与 C ABI/CLI 准入；本模块只校验物理布局 |
| 内部锁、跨线程同步、重入协议 | 无；调用方串行化 |

依赖与所有权规则：

- 只依赖 base、dtypes、memory；不得依赖 runtime/execution/inference/model/backend。
- `KVCacheManager` 按值持有 `KVCacheStorage` 与 `KVCacheSessionState`；不保存 `Allocator` 引用，也不持有 `Runtime`。
- Manager 必须比所有借用其 state 的 `KVCacheSessionView`（含 stale 副本）活得久，并在 view 存在期间保持地址稳定；只能在没有 view 借用时 move/assign。
- `KVCacheSessionView` 只借用 Manager-owned storage/session state；其 `const` 不表示 `CommitAppend` 不修改被借用的 session state。

## 2. 关键数据结构与所有权

### 2.1 KVCacheLayout

独立的公开值类型，不依赖 `Buffer`，可在分配前构造、验证和计算字节数。`KVCacheStorage` 按值持有物理实例。

| 字段 | 含义 |
|---|---|
| `num_layers` | decoder layer 数 |
| `num_kv_heads` | 每层 KV head 数 |
| `max_tokens` | 物理 token 容量上限 |
| `head_dim` | 逻辑 head 宽度（元素数） |
| `padded_head_dim` | 对齐后的 head 宽度（元素数），可为 `head_dim` 或更大 |
| `token_stride` | 相邻 token 的字节步长 |
| `head_stride` | 相邻 KV head 的字节步长 |
| `layer_stride` | 相邻 layer 的字节步长 |
| `kv_dtype` | 存储元素类型；默认未定义，`Init`/`PlanKVCacheLayout` 拒绝未定义或非固定长度类型 |
| `alignment` | 行与 plane 的字节对齐，默认 64；必须是零以外的 2 的幂 |

`PlanKVCacheLayout` 是无分配的纯规划入口，几何关系为：

```text
E = ElementBytes()
padded_head_dim = AlignOffsetUp(head_dim, alignment / gcd(alignment, E))
token_stride    = padded_head_dim * E
head_stride     = max_tokens * token_stride
layer_stride    = num_kv_heads * head_stride
bytes_per_plane = num_layers * layer_stride
total_bytes     = 2 * bytes_per_plane
offset          = layer_idx * layer_stride + kv_head_idx * head_stride
                + seq_pos * token_stride + dim_idx * E
```

- `ElementBytes()` 只接受固定长度、非 scalable 的元素编码；否则返回 0，`Validate` 拒绝。
- 行 padding 以元素为单位，保证 `token_stride` 是 `alignment` 的倍数且不拆分元素；`head_stride`/`layer_stride` 由整数倍关系自动对齐。
- `Validate` 检查：维度非零、`alignment` 为 2 的幂、元素大小固定非零、`padded_head_dim >= head_dim`、`token_stride` 是元素大小与 `alignment` 的倍数且覆盖整行、`head_stride` 覆盖全部 token、`layer_stride` 覆盖全部 head。
- `Offset` 先 `Validate` 再检查索引与乘加溢出；`OffsetValidated` 由 View 在 `ValidateBaseState` 成功后调用，保留索引与溢出检查，但不重复 `Validate`。
- `BytesPerPlane`/`TotalBytes` 按 stride 计算；手工构造的 layout 若声明了 stride gap，字节数会计入 gap，而不是仅按 `head_dim` 紧排计算。
- 物理 dtype 合法不等于对应 CPU kernel 可执行；kernel 准入不在本模块判定。

### 2.2 KVCacheStorage 与 KVCacheSessionState

两者均定义在 `detail/kv_cache_state.h`，是 Manager-owned 内部类型，不是公开 handle。

| 类型 | 字段 | 语义 |
|---|---|---|
| `KVCacheStorage` | `KVCacheLayout layout` | 按值拥有布局；`layout()` 返回它的 const 引用 |
| | `Buffer key_buffer` / `Buffer value_buffer` | 按值拥有 K/V 两个线性 plane；`is_initialized()` 要求两者均已初始化 |
| `KVCacheSessionState` | `uint64_t generation` | Manager-owned epoch；view 绑定时捕获 |
| | `size_t capacity_tokens` | 本次预约的 token 上限，等于 `prompt_len + future_kv_appends` |
| | `size_t prompt_len` | 首次 Prefill 长度；不可由 `committed_tokens` 推导 |
| | `size_t committed_tokens` | 跨 plan 已发布的提交水位 |
| | `Lifecycle lifecycle` | `kIdle` / `kAwaitingPrefill` / `kPrefillCommitted` |

`KVCacheStorage` 依赖不抛异常的移动赋值来整体发布；Manager 只在 K/V 两个 plane 都分配成功后发布 candidate storage。

生命周期约定：`kIdle` 的所有计数字段为零；活动状态由 `ValidateBaseState` 检查下述不变量，而所有 View 访问都要求活动 lifecycle，因此 `kIdle` 会被访问入口拒绝。

| Lifecycle | 不变量 |
|---|---|
| `kIdle` | `capacity_tokens == 0`，`prompt_len == 0`，`committed_tokens == 0` |
| `kAwaitingPrefill` | `0 < prompt_len <= capacity_tokens <= layout.max_tokens`；`committed_tokens == 0` |
| `kPrefillCommitted` | `prompt_len <= committed_tokens <= capacity_tokens` |

`generation` 规则：`Init` 保留原值；`ReserveForSession` 和成功的 `ReleaseSession` 各推进一次；`ResetSession` 不推进。达到 `uint64_t` 最大值时 release 仍成功但不递增，后续 reservation 以 `Overflow` 拒绝，因此旧 view 不会复活。

### 2.3 KVCacheManager

| 成员 | 类型 | 所有权/语义 |
|---|---|---|
| `storage_` | `KVCacheStorage` | 按值拥有 layout 与 K/V Buffer |
| `session_state_` | `KVCacheSessionState` | 按值拥有单 session 状态 |
| `initialized_` | `bool` | 成功 `Init` 后为 true；失败路径保持 false |

- 禁止复制，允许 noexcept move；move 只允许发生在 view 借用之前。
- 默认构造为空：空 layout、容量 0、`total_bytes() == 0`、`is_initialized() == false`。
- `Init(Allocator&, ...)` 只在该调用期间借用 allocator，不保存引用；分配出的 `Buffer` 在 allocator 销毁后仍有效。
- `layout()` 返回 `storage_.layout` 的 const 引用；未初始化时返回默认空布局。

### 2.4 KVCacheSessionView

| 成员 | 类型 | 语义 |
|---|---|---|
| `storage_` | `KVCacheStorage*` | 借用；绑定时与 `session_state_` 同时置位，`Invalidate` 同时清空 |
| `session_state_` | `KVCacheSessionState*` | 借用；不拥有状态 |
| `generation_` | `uint64_t` | 绑定时捕获的 session generation |

- View 可复制；所有副本共享同一份 Manager-owned state。
- 不保存独立的 layout 指针，几何信息一律读 `storage_->layout`；公开头不暴露 `Buffer`。
- `valid()` 只判断绑定非空、storage 已初始化、generation 匹配且 lifecycle 处于活动状态；读写与提交入口另行执行完整的 `ValidateBaseState`。
- `current_pos()`/`committed_tokens()` 返回 canonical `committed_tokens`，view 无效时返回 0。
- `ValidateBaseState` 还检查：reservation 不变量、layout 有效、`bytes_per_plane` 不超过两个 backing Buffer、CPU plane 的 base 地址满足 layout alignment。
- View 不提供元素写指针；写路径只有 append binding。`Invalidate()` 只解除本 view 的绑定，不修改 storage。

### 2.5 窄 binding 契约

定义在 `base/kv_cache_binding.h`，是 kernel invocation-only 的借用数据：

| 类型 | 内容 | 约束 |
|---|---|---|
| `KVCacheLayerStorageBinding` | `key_data`/`value_data` 为该层 plane base，另含 `dtype`、`layer_index`、`num_kv_heads`、`head_dim`、`token_capacity`、`token_stride_bytes`、`head_stride_bytes` | 指针只在下一次同步 kernel 调用期间有效；不得缓存进 prepared params 或跨 `Execute()` 保存 |
| `KVCacheAppendBinding` | 上述 storage + `begin`/`end` | update kernel 精确写 `[begin, end)`；`end` 是 plan-local 可见前沿，plan 成功前不提交 |
| `KVCacheReadBinding` | 上述 storage + `committed_end`/`visible_end` + `query_begin`/`query_end` | `[0, committed_end)` 跨 plan 可见；`[committed_end, visible_end)` 只允许同一同步 plan 内、该层 append 成功后的 Attention 读取；query 区间由 `LayerRunner` 填写 |

## 3. 接口与代码位置

### 3.1 布局与规划

| 接口 | 语义要点 | 代码位置 |
|---|---|---|
| `PlanKVCacheLayout(num_layers, num_kv_heads, capacity_tokens, head_dim, dtype, alignment)` | 纯规划、零分配；返回已验证且两 plane 字节可表示的 layout | [kv_cache_layout.h](../../include/aethermind/kv_cache/kv_cache_layout.h) / [kv_cache_layout.cpp](../../src/kv_cache/kv_cache_layout.cpp) |
| `KVCacheLayout::Validate()` | 校验几何、对齐、stride 一致性；`padded_head_dim`/stride 不足返回 `InvalidArgument`，乘加溢出返回 `Overflow` | 同上 |
| `KVCacheLayout::Offset(layer, head, pos, dim)` | 先 `Validate` 再计算字节 offset；索引越界 `OutOfRange` | 同上 |
| `KVCacheLayout::BytesPerPlane()` / `TotalBytes()` | 按 `num_layers * layer_stride` 与两 plane 计算；成功不分配 | 同上 |

### 3.2 Manager

| 接口 | 语义要点 | 代码位置 |
|---|---|---|
| `Init(Allocator&, num_layers, num_kv_heads, max_tokens, head_dim, kv_dtype, alignment = 64)` | 清理旧 storage 与预约（保留 generation），规划 layout，按 `AllocationRequest{nbytes, alignment}` 分配两个 plane；两者成功后再整体发布 | [kv_cache_manager.h](../../include/aethermind/kv_cache/kv_cache_manager.h) / [kv_cache_manager.cpp](../../src/kv_cache/kv_cache_manager.cpp) |
| `ReserveForSession(prompt_len, future_kv_appends)` | 校验已初始化、无活动预约、prompt 非空、预约容量不超物理容量、generation 未耗尽；推进 generation 并返回 view | 同上 |
| `ValidateReservation(prompt_len, future_kv_appends)` | 复用 reservation 的几何/lifecycle/capacity/epoch 检查，不修改状态、不分配 KV 存储 | 同上 |
| `ResetSession(const KVCacheSessionView&)` | 要求 view 属于本 Manager 且仍有效；清零 `committed_tokens`，回到 `kAwaitingPrefill`，保留预约与 generation | 同上 |
| `ReleaseSession(KVCacheSessionView&)` | 要求 view 属于本 Manager 且仍有效；清空预约字段、回到 `kIdle`、推进 generation（饱和时除外）并 `Invalidate` 该 view | 同上 |
| `layout()` / `capacity_tokens()` / `total_bytes()` / `is_initialized()` | 只读查询；`total_bytes()` 返回两个 Buffer 的 `nbytes()` 之和 | 同上 |

`future_kv_appends` 是 prompt 之后最多还能 append 的 KV 位置数。Prefill 产生的首 token 尚未进入 KV，因此 Generate 上限为 N 时传 N - 1；`InferenceSession` 在创建请求前用 `ValidateReservation` 做准入，再实际 `ReserveForSession`。

### 3.3 View

| 接口 | 语义要点 | 代码位置 |
|---|---|---|
| `valid()` / `max_tokens()` / `current_pos()` / `num_layers()` / `num_kv_heads()` / `head_dim()` / `kv_dtype()` / `token_capacity()` / `committed_tokens()` / `prompt_len()` / `awaiting_prefill()` | 只读视图查询；invalid view 返回 0 或未定义 dtype | [kv_cache_session_view.h](../../include/aethermind/kv_cache/kv_cache_session_view.h) / [kv_cache_session_view.cpp](../../src/kv_cache/kv_cache_session_view.cpp) |
| `ValidateWrite(layer, head, seq_pos, token_count)` | `token_count` 非零；layer/head 在范围内；`seq_pos + token_count` 不超过 reservation 与物理容量；不要求起点等于提交前沿 | 同上 |
| `ValidateRead(layer, head, seq_begin, seq_end)` | `seq_begin <= seq_end`；`seq_end` 不超过当前 `committed_tokens` | 同上 |
| `BindLayerForAppend(layer, begin, end)` | `begin == committed_tokens`；`end > begin`；不超过 reservation 与物理容量；返回该层 append binding | 同上 |
| `BindLayerForRead(layer, committed_end, visible_end)` | `committed_end == committed_tokens`；`visible_end >= committed_end`；不超过 reservation 与物理容量；允许同 plan 已写未提交的可见范围 | 同上 |
| `KeyData` / `ValueData` | 只读元素指针；先做已提交范围校验，再按 layout offset 计算；越界返回 `OutOfRange` | 同上 |
| `CommitAppend(begin, end) const noexcept` | 公开提交入口；只验证 view 状态与范围，不证明 layer 写入完成 | 同上 |
| `Invalidate()` | 清空 borrowed 指针与捕获的 generation | 同上 |

`BindLayerStorageValidated` 由 append/read binding 共用：先把 layer 索引限制在 `uint32_t` 范围，再用 `OffsetValidated(layer, 0, 0, 0)` 得到该层 plane 起点；返回的 storage binding 不包含 per-head/per-token 指针，kernel 通过 stride 自行定位。

### 3.4 错误语义

| 条件 | `Status` |
|---|---|
| Manager 未初始化；存在活动预约时再次 reserve；view 不属于该 Manager 或已 stale | `FailedPrecondition` |
| `prompt_len == 0`；`token_count == 0`；`end <= begin`；`seq_begin > seq_end`；append 起点不等于提交前沿；commit 范围为空 | `InvalidArgument` |
| layer/head/position/dim 越界；写入或提交超过 reservation 或物理容量；读取超过已提交范围 | `OutOfRange` |
| 几何、stride、字节数乘加溢出；generation 耗尽 | `Overflow` |
| layout 维度为零、对齐非法、元素大小非固定、stride 不足 | `InvalidArgument` |
| 诊断消息或内部 `std::bad_alloc` | 空消息 `ResourceExhausted` |

所有公开入口都用 `Status`/`StatusOr` 报错；除 `KVCacheManager::Init` 外均声明 `noexcept`，`Init` 捕获 `std::bad_alloc` 并返回 `ResourceExhausted`。错误路径可能为诊断消息分配内存；分配失败时返回空消息 `ResourceExhausted`，且不发布半完成状态。

### 3.5 模块外集成位置

| 集成点 | 事实 | 代码位置 |
|---|---|---|
| Runtime 组装 | `RuntimeBuilder::TryBuildKVCacheManager` 先 `PlanKVCacheLayout`/`TotalBytes` 检查 `max_bytes`，再取 CPU allocator 并调用 `Init` | [runtime_builder.cpp](../../src/runtime/runtime_builder.cpp) |
| Inference 预约 | `InferenceSession` 用 `prompt_len + (max_new_tokens - 1)` 计算预约容量；先 `ValidateReservation`，后 `ReserveForSession` | [inference_session.cpp](../../src/inference/inference_session.cpp) |
| 执行期提交 | `LayerRunner` 为完整 plan 建立 `KVAppendTransaction`，所有 layer 写完后只调用一次 `CommitAppend` | [layer_runner.cpp](../../src/execution/layer_runner.cpp) |
| KV16 准入 | `ExecutableModel::Load` 校验 `kv_dtype` 与 Runtime Manager 的物理 dtype | [executable_model.cpp](../../src/inference/executable_model.cpp) |

## 4. 主流程/时序

### 4.1 Runtime 组装与 Init

```text
RuntimeOptions.kv_cache
  → PlanKVCacheLayout → TotalBytes
  → max_bytes != 0 && required > max_bytes ? ResourceExhausted : 继续
  → AllocatorRegistry.TryGetAllocator(Device::CPU())
  → KVCacheManager::Init
       → storage_ = {}; session_state_ = {generation 保留}; initialized_ = false
       → PlanKVCacheLayout → BytesPerPlane
       → TryAllocate(key), TryAllocate(value)
       → 两者成功后 std::move 发布 storage_，initialized_ = true
```

- `enable_manager == false` 时返回默认空 Manager，不规划、不分配。
- 任一阶段失败，Manager 保持空 layout、零容量、零已分配字节；旧 view 因 lifecycle 回到 `kIdle` 而失效，后续 reservation 推进 generation 后也不会复活。
- 第二个 plane 分配失败时，局部 candidate 析构释放第一个 plane；Manager 可重试 `Init`。

### 4.2 预约

```text
ValidateReservation(prompt_len, future)
  → initialized? lifecycle == kIdle? prompt_len > 0?
  → prompt_len + future 不溢出且 > 0
  → requested <= layout.max_tokens
  → generation != UINT64_MAX
ReserveForSession
  → ++generation
  → capacity_tokens = requested
  → prompt_len = prompt_len; committed_tokens = 0
  → lifecycle = kAwaitingPrefill
  → 返回 KVCacheSessionView(&storage_, &session_state_)
```

`ValidateReservation` 只做准入，不改变 reservation；实际 Reserve 必须再次通过完整检查。

### 4.3 Prefill / Decode append 与 commit

| 阶段 | 写入范围 | 提交条件 |
|---|---|---|
| Prefill | 所有 layer 各写一次 `[0, prompt_len)` | `LayerRunner` 确认所有 layer 已写后调用 `CommitAppend(0, prompt_len)`，lifecycle → `kPrefillCommitted` |
| Decode | 所有 layer 各写一次 `[committed_tokens, committed_tokens + 1)` | 所有 layer 已写后调用 `CommitAppend(begin, begin + 1)`，lifecycle 保持 `kPrefillCommitted` |

`CommitAppend` 的验证顺序与语义：

1. `ValidateBaseState`：绑定、storage 初始化、generation、reservation 不变量、layout、Buffer 大小、CPU base alignment。
2. `begin == committed_tokens`；`end > begin`；`end <= capacity_tokens`；`end <= layout.max_tokens`。
3. `kAwaitingPrefill` 要求 `begin == 0 && end == prompt_len`，否则 `FailedPrecondition`。
4. `kPrefillCommitted` 要求 `end - begin == 1`，否则 `InvalidArgument`。
5. 通过后写 `committed_tokens = end` 并设置下一 lifecycle；失败不改变任何已提交状态。

**调用方证明责任**：`CommitAppend` 不检查 K/V 字节是否已写入，也不检查每个 layer 是否执行。调用方必须先证明完整 append plan 成功且所有 layer 已写完 `[begin, end)`；生产中由 `LayerRunner` 在 plan 末尾统一发布。`const` 只描述 view 描述符，成功提交仍会修改 Manager-owned session state。

### 4.4 读取可见性

- `[0, committed_end)`：跨 plan 已提交，可被普通 read API 读取。
- `[committed_end, visible_end)`：当前同步 plan 已写但尚未提交的范围；只允许同一 transaction 内、且该 layer 的 `KVCacheUpdate` 已成功后的 Attention 读取。
- `[visible_end, capacity_tokens)`：不可读取；Attention-before-update 不能通过扩大 `visible_end` 读取未初始化内容。

`BindLayerForRead` 只表达可见性前沿；`query_begin`/`query_end` 属于执行事务语义，由 `LayerRunner` 按 plan 填写。

### 4.5 Reset / Release

```text
(Manager 未初始化) --Init 成功--> lifecycle kIdle
kIdle --ReserveForSession--> kAwaitingPrefill
kAwaitingPrefill --CommitAppend(0, prompt_len)--> kPrefillCommitted
kPrefillCommitted --CommitAppend(c, c + 1)--> kPrefillCommitted
kAwaitingPrefill | kPrefillCommitted --ResetSession--> kAwaitingPrefill
kAwaitingPrefill | kPrefillCommitted --ReleaseSession--> kIdle
任意已初始化状态 --Init--> kIdle（storage 替换，generation 保留）
```

- `ResetSession` 保留 `prompt_len`、`capacity_tokens`、generation 与物理存储，只把 `committed_tokens` 清零；下一次执行必须重新 Prefill，不表达“保留 prompt KV 后重跑 Decode”。同一 reservation 的所有 view 副本同时观察到 reset。
- `ResetSession`/`ReleaseSession` 会拒绝不属于该 Manager 的 view，也不会改变另一个 reservation 的状态。
- `ReleaseSession` 成功后该 view 被 `Invalidate`，其他副本靠 generation 不匹配失效；物理存储不释放，可被后续 reservation 复用。
- Reset/Release 都不强制 `memset` 底层存储；成功路径不触发新的堆分配。

### 4.6 线程与重入约束

- Manager/View 均非线程安全；同一 Manager 及其 view 的访问由调用方串行化。
- `Init`、`ReserveForSession`、`ResetSession`、`ReleaseSession` 不得与借用该 Manager view 的 `Execute()` 交错，包括 kernel 重入；append transaction 不得逃离执行栈。
- 当前没有 reset revision/token，也不为并发重入预埋状态机；该限制不是并发安全保证。若未来需要重入或并发，必须另行定义 transaction identity 与同步协议。

## 5. 关键取舍

| 选择 | 原因 | 被否方案 |
|---|---|---|
| 静态预分配、不扩容 | Decode 稳态不需要新 owner 分配；布局与地址在 Runtime 组装后稳定 | 动态扩容/按 session 申请不同大小 KV 池；被否因为热路径分配、地址不稳定且超出当前产品边界 |
| 单 session slot | 当前产品是单请求同步执行；单状态足以表达 reservation 与提交水位 | 多 slot/多请求共享；被否因为需要调度、共享与回收协议，当前不承诺 |
| K/V 分成两个连续 plane | offset 计算简单、reference kernel 友好、便于局部替换 | K/V 交错存储；被否因为 stride/offset 更复杂，且无当前收益 |
| 布局用独立值类型 + stride 字段 | 执行层只依赖布局契约，不硬编码物理公式；可先规划后分配 | 把唯一 offset 公式写死在执行层/Attention；被否因为切换底层存储会迫使上层返工 |
| 行 padding 以元素为粒度对齐 | 保持 `token_stride` 对齐且不拆分元素，兼容非 2 的幂元素大小 | 按字节强行切分元素或只对 4 字节类型对齐；前者破坏元素完整性，后者不通用 |
| View 只借用、不拥有 Buffer | 存储所有权集中在 Manager/Runtime；View 可廉价复制 | View 持有 owning buffer 或公开物理 Buffer；被否因为所有权与生命周期扩散到 execution |
| 写路径只有 append binding，元素访问器只读 | 明确“谁能写、写到哪”，避免绕过提交水位直接改 KV | 公开 `MutableKeyData`/`KeyWriteView` 等可写视图；未落地，且会让写入边界不可审计 |
| `CommitAppend` 为 `const noexcept`，由调用方证明写入完成 | 提交只做状态与范围校验，避免在 KV 模块复制 whole-plan transaction 逻辑 | 在 `CommitAppend` 内检查 layer 完成情况或执行回滚；被否因为需要跨越 execution 的 plan 状态，破坏模块边界 |
| Decode 每次 commit 恰好一个 token | 与同步单 token Decode 循环一致，提交水位语义最简单 | 允许任意跨度的 Decode commit；被否因为会模糊 plan 与 token 的对应关系，当前无需求 |
| Reset 保留预约但清零水位，且不清物理内存 | 复用存储与容量，避免重新分配；正确性不依赖清零 | Reset 释放预约或重新分配；被否因为破坏复用与零分配。强制 memset；被否因为当前目标不是安全擦除 |
| 无内部锁与原子同步 | 单请求同步模型下由调用方串行化，避免热路径同步成本 | 内部互斥/原子 epoch；被否因为无法覆盖 `Execute()` 与 Manager 变更的跨对象顺序，且当前无并发承诺 |
| generation 只增不减，饱和时安全失败 | 旧 view 永不因回绕而复活；最后一个 reservation 仍可正常 release | 用 `reset revision`/token 或允许 generation 回绕；被否因为需要额外事务身份协议 |
| 物理 dtype 与 kernel 支持分离 | kv_cache 只管布局与字节，F32/F16/BF16 的执行准入归 inference/backend | 在 layout 中硬编码仅允许某一种 dtype；被否因为会把产品准入塞进物理存储层 |
| 不提供旧 View 类型/头兼容 alias | 避免同一语义出现两套公开边界 | 保留 `KeyWriteView`/`KeyReadView` 等兼容层；被否因为增加维护面且易绕过提交合同 |
| 不实现 PagedAttention/prefix cache/多 session | 当前产品边界只承诺静态单 session KV | 预埋 block table/页管理；被否因为会引入未验证的所有权、回收与调度复杂度 |

KV16 准入边界不属于本模块的策略判断，但与本模块的物理 dtype 直接相连：

- `PlanKVCacheLayout` 接受任意固定长度元素类型；`KVCacheManager::Init` 只保证物理布局合法。
- inference 的 `MakeKVCacheOptions` 接受 scalar `Float32`/`Float(16)`/`BFloat(16)`，拒绝 undefined、scalable、非 scalar 或其他类型；`MakeFp32KVCacheOptions` 是 FP32 wrapper。
- `ExecutableModel::Load` 要求 model 的 `kv_dtype` 为 scalar F32/F16/BF16；KV16 必须有 retained-W16 policy（`kKeepFp16WeightsFp32Activation` 或 `kKeepBf16WeightsFp32Activation`），且请求 dtype 必须与既有 Manager 的 `layout().kv_dtype` 一致；KV16 无 Manager 返回 `FailedPrecondition`，dtype 不一致返回 `InvalidArgument`，均在转换/packing 前拒绝。
- C ABI v1 与 CLI 保持 strict FP32；不会自动扩大为 KV16。数值合同与验收见 [w16-fp32-inference.md](../guides/w16-fp32-inference.md)。
- 两字节元素不保证物理 KV 或进程内存减半：padded layout 下小 `head_dim` 仍可能占用相同的对齐行。

## 6. 测试与验证入口

| 测试文件 | 覆盖重点 |
|---|---|
| [test_kv_cache_layout_planning.cpp](../../tests/unit/kv_cache/test_kv_cache_layout_planning.cpp) | 多种 dtype/对齐下的 padding、stride、offset、plane/total 字节；两 plane 可表示边界；各阶段 overflow；规划零分配；Manager 精确分配两个 plane；第二 plane 失败可重试；规划失败先清旧 storage |
| [test_kv_cache_manager.cpp](../../tests/unit/kv_cache/test_kv_cache_manager.cpp) | Init 参数/分配失败矩阵；Init 后空状态与重试；reserve/commit/读写边界；Reset/Release 与 generation；re-init 永久失效旧 view；foreign view 拒绝；行对齐与低于指针大小的 alignment；move 前借用安全 |
| [test_kv_cache_commit.cpp](../../tests/unit/kv_cache/test_kv_cache_commit.cpp) | `CommitAppend` 的 const/noexcept 签名；不存在旧 `CommitUntil`/可写元素访问；Prefill 部分提交、跨 Decode 位置、非连续 begin、重复 commit、空 append、超预约均不改状态；错误 epoch 与饱和 generation；诊断失败保持完整状态 |
| [test_kv_cache_binding_validation.cpp](../../tests/unit/kv_cache/test_kv_cache_binding_validation.cpp) | binding 保留 padded 几何；元素读取 offset；每次调用重新验证 layout/backing/committed watermark；stale generation/lifecycle 失效；替换 backing 后拒绝 |
| [test_kv_cache_session_view_oom.cpp](../../tests/unit/kv_cache/test_kv_cache_session_view_oom.cpp) | 每个诊断路径在任意一次分配失败时返回空消息 `ResourceExhausted`，状态与底层字节不变且 view 仍可用 |
| [test_kvcache_update_kernel.cpp](../../tests/unit/execution/test_kvcache_update_kernel.cpp) | `LayerRunner` 单 plan 事务；所有 layer 写完后只提交一次；缺 layer/alias 失败不提交；Attention 读取已提交范围与同 plan 可见范围；后续 step 失败不提交 |
| [test_kvcache_update_kv16_transaction.cpp](../../tests/unit/execution/test_kvcache_update_kv16_transaction.cpp) | KV16 下 Prefill/Decode 每次 plan 推进一个位置；多层共享同一 append range；update 后 Attention 可见范围；缺失 update 或后续 Attention 失败不提交 |
| [test_kv16_load_chain.cpp](../../tests/unit/inference/test_kv16_load_chain.cpp) | KV16 配置链：Runtime/Manager dtype 一致、KV16 需要 retained-W16 policy、拒绝 dtype 不匹配/无 Manager/非法 dtype |
| [test_kv16_model_acceptance.cpp](../../tests/unit/inference/test_kv16_model_acceptance.cpp) | KV16 与 KV32 的 F16/BF16 四组合质量、近容量长上下文、GQA/tied、绑定后 Decode 无 malloc-family 调用 |

```bash
./build/tests/unit/aethermind_unit_tests \
  --gtest_filter=KVCacheBindingValidationTest.*:KVCacheSessionView.*:KVCacheCommitTest.*:KVCacheLayoutPlanning.*:KVCacheLayoutPlanningNoAllocation.*:KVCacheLayoutPlanningDiagnosticFailure.*:KVCacheManager.*:KVLayoutContract.*:Validation/KVCacheManagerInitFailure.*:Allocation/KVCacheManagerInitAllocationFailure.*:InvalidReservations/KVCacheInvalidStateTest.*:DiagnosticPaths/KVCacheSessionViewOomTest.*
```

执行期提交与 KV16 集成可用以下聚焦入口（需已构建对应测试目标）：

```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter=KVCacheUpdateKernel.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=KVCacheUpdateKV16Transaction.*
```

预算与产品准入的交叉说明见 [resource-budgeting.md](../guides/resource-budgeting.md)；KV 在系统中的位置见 [architecture.md](../architecture.md) 第五节。
