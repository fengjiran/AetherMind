---
scope: submodule
module: central_cache
parent: ammalloc
depends_on: [page_cache, size_class, spin_lock]
adr_refs:
  - ../adrs/ADR-001.md
  - ../adrs/ADR-005.md
last_verified: 2026-03-10
owner: team
status: active
---

# central_cache 子模块记忆

> 保存路径：`docs/agent/memory/modules/ammalloc/submodules/central_cache.md`
> 父模块：`docs/agent/memory/modules/ammalloc/module.md`
> 用途：记录 `central_cache` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：按 SizeClass 管理全局 `Bucket`；在 `TransferCache` 中完成 O(1) 批量指针流转；在 `SpanList` 中切分 `Span`、扫描位图，并与 `PageCache` 交换页级资源。
- 不负责：TLS 实例生命周期、请求尺寸映射策略的定义、直接系统页分配。

### 边界
- 输入来源：`ThreadCache` 通过 `FetchRange()` 请求一批对象，或通过 `ReleaseListToSpans()` 归还一批对象。
- 输出去向：向 `ThreadCache` 交付对象链表；向 `PageCache` 请求新 `Span` 或归还完全空闲的 `Span`。
- 不直接管理：线程本地缓存实例、`PageAllocator` 的系统调用策略、OS 级虚拟内存区域。
- 未涉及：无

## 已确认事实
- 已验证约束：`CentralCache` 是单例；`Bucket` 使用 `alignas(SystemConfig::CACHE_LINE_SIZE)`；每个 bucket 同时持有 `SpinLock transfer_cache_lock`、`std::mutex span_list_lock`、`SpanList span_list` 与指针数组型 `TransferCache`。
- 已验证限制：`FetchRange()` 和 `ReleaseListToSpans()` 按 `SizeClass::Index(obj_size)` 选桶；`FetchRange()` 通过 `AM_DCHECK` 要求 `batch_num <= SizeClass::kMaxBatchSize`。
- ADR 关联：[ADR-001: TransferCache 设计](../adrs/ADR-001.md)、[ADR-005: 无 STL 约束](../adrs/ADR-005.md)
- 非阻塞注意事项：初始化阶段通过一次 `PageAllocator::SystemAlloc()` 申请所有 bucket 的 TransferCache 底座，并按 `capacity = 8 * batch_size` 切分；当前 `FetchRange()` 的 `prefetch_target` 固定为一个 `batch_num`。
- 未涉及：无

## 核心抽象
### 关键抽象
- `Bucket`：单个 SizeClass 的全局资源容器；把快路径 `TransferCache` 与慢路径 `SpanList` 拆分到不同锁域。
- `TransferCache`：连续指针数组；用 `transfer_cache_count` / `transfer_cache_capacity` 管理占用，服务批量取放对象的快路径。
- `SpanList`：持有可切分或部分已用的 `Span` 双向链表；慢路径在锁内做位图扫描、满 Span 调整和空 Span 下沉。

### 数据流
- 输入：来自 `ThreadCache` 的批量获取或归还请求。
- 处理：优先从 `TransferCache` 读写；不足或溢出时转入 `SpanList`；必要时临时释放 bucket 锁向 `PageCache` 申请新 `Span`，并在慢路径结束后把额外对象预取回 `TransferCache`。
- 输出：构造成 `FreeList` 链表返回给 `ThreadCache`，或把空闲 `Span` 回退到 `PageCache`。
- 未涉及：无

## 对外接口
- 对主模块暴露：`CentralCache::GetInstance()`、`FetchRange(FreeList&, size_t, size_t)`、`ReleaseListToSpans(void*, size_t)`、`Reset()`。
- 调用约束：`obj_size` 必须与目标 SizeClass 一致；批量链表中的对象必须都属于同一大小类别；`GetOneSpan()` 只在内部使用，并要求调用方持有 `span_list_lock` 后再调用。
- 头文件位置：`ammalloc/include/ammalloc/central_cache.h`
- 未涉及：无

## 不变量
- 每个 bucket 的 `TransferCache` 和 `SpanList` 分别由不同锁保护，不能在无锁条件下跨域修改。
- `TransferCache` 的底层指针数组在初始化后按 bucket 分段固定，不在运行期重新分配。
- 调用 `PageCache::AllocSpan()` / `ReleaseSpan()` 前必须先释放 `span_list_lock`，避免与全局页锁形成死锁。
- 未涉及：无

## 所有权与生命周期
- 所有者：`CentralCache` 自身与各个 `Bucket` 为单例成员；TransferCache 底层连续内存由 `CentralCache` 初始化并在 `Reset()` 中通过 `PageAllocator::SystemFree()` 释放。
- 借用关系：`SpanList` 中的 `Span` 元数据由 `PageCache` 的 `span_pool_` 拥有；`CentralCache` 只借用并重新挂回 `PageCache`。
- 生命周期边界：单例在首次访问时构造；`Reset()` 会清空 `TransferCache`、把对象和 `Span` 归还后端，并释放 TransferCache 底层内存。
- 未涉及：无

## 并发约束
- 并发模型：多线程共享全局 `CentralCache`，并按 SizeClass bucket 隔离竞争。
- 同步要求：`TransferCache` 只能在 `SpinLock` 保护下读写；`SpanList` 和位图扫描只能在 `span_list_lock` 保护下执行。
- 禁止事项：禁止持有 `span_list_lock` 直接进入 `PageCache` 长临界区；禁止在未同步的情况下同时修改同一 bucket 的数组计数和 `SpanList`。
- 未涉及：无

## 性能约束
- 热路径：`TransferCache` 批量取出与回填，以及 `FetchRange()` 中的桶级快路径。
- 约束：快路径保持 O(1) 数组操作；慢路径才允许位图扫描；额外预取对象优先填回 `TransferCache` 以摊还后续互斥锁成本。
- 监控点：关注 `TransferCache` 命中率、`span_list_lock` 竞争、预取后回退对象比例，以及 `capacity = 8 * batch_size` 布局对缓存局部性的影响。
- 未涉及：无

## 已否决方案
- 方案：仅依赖 `SpanList` 处理所有跨线程对象流转。
  - 原因：设计文档明确指出在互斥锁下扫描位图对高并发批量操作过慢，因此引入 `TransferCache` 作为 O(1) 指针数组快路径。
- 无

## 未决问题
- `CentralCache::FetchRange()` 源码中保留 TODO：`prefetch_target` 未来可能根据 `TransferCache` 剩余容量动态调节，而不是固定等于一个 `batch_num`。
- 无

## 待办事项
- [ ] 评估 `prefetch_target` 的动态调节策略，避免高并发下把过量预取对象再次回退到慢路径。
- 无
