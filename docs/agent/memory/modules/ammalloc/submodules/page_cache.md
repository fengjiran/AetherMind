---
scope: submodule
module: page_cache
parent: ammalloc
depends_on: [page_allocator]
adr_refs:
  - ../adrs/ADR-002.md
  - ../adrs/ADR-004.md
  - ../adrs/ADR-005.md
last_verified: 2026-03-10
owner: team
status: active
---

# page_cache 子模块记忆

> 保存路径：`docs/agent/memory/modules/ammalloc/submodules/page_cache.md`
> 父模块：`docs/agent/memory/modules/ammalloc/module.md`
> 用途：记录 `page_cache` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：以页为单位管理 `Span`；实现 `AllocSpan()` 的精确命中、切分和系统补充；在 `ReleaseSpan()` 中做相邻空闲 `Span` 合并；维护 `PageMap` 的写路径。
- 不负责：线程局部对象缓存、SizeClass 计算、直接暴露给最终用户的分配接口。

### 边界
- 输入来源：`CentralCache` 和大对象慢路径调用 `AllocSpan(page_num, obj_size)`；`CentralCache` 或大对象释放路径调用 `ReleaseSpan(span)`。
- 输出去向：向上层返回可用 `Span`；向 `PageAllocator` 请求或归还页；向 `PageMap` 写入或清理页到 `Span*` 的映射。
- 不直接管理：TLS `ThreadCache` 状态、`TransferCache` 指针数组、用户对象的尺寸分级策略。
- 未涉及：无

## 已确认事实
- 已验证约束：`PageCache` 是单例，内部以单个 `std::mutex mutex_` 保护 `span_lists_`；`span_lists_` 范围为 `0..PageConfig::MAX_PAGE_NUM`；`span_pool_` 负责 `Span` 元数据复用。
- 已验证限制：`AllocSpanLocked()` 对 `page_num > PageConfig::MAX_PAGE_NUM` 的请求直接走 `PageAllocator::SystemAlloc()`；普通缓存路径只管理 `<= 128` 页的 `Span`。
- ADR 关联：[ADR-002: RadixTree PageMap](../adrs/ADR-002.md)、[ADR-004: MADV_DONTNEED 策略](../adrs/ADR-004.md)、[ADR-005: 无 STL 约束](../adrs/ADR-005.md)
- 非阻塞注意事项：`PageMap` 采用 `RadixRootNode` + 3 层 `RadixNode` 的胖根布局；`GetSpan()` 无锁，`SetSpan()` / `ClearRange()` / `Reset()` 依赖 `PageCache::mutex_` 保护树结构写入。
- 未涉及：无

## 核心抽象
### 关键抽象
- `SpanList[1..128]`：按页数分类的空闲 `Span` 桶；命中时直接弹出，未命中时可从更大桶切分。
- `PageMap`：`PageID -> Span*` 的全局基数树映射；读路径用 acquire 原子加载，写路径懒创建节点并批量写叶子。
- `Span`：连续页区间的元数据对象；记录页范围、对象大小、位图、`use_count`、最近使用时间和 `is_committed`。

### 数据流
- 输入：目标页数和对象大小，或待归还的 `Span`。
- 处理：优先尝试精确命中；其次从更大桶头部分割；再不足时向系统申请 `128` 页补货；释放时先处理超大 `Span` 直接归还系统，再做左右邻居合并并重写 `PageMap`。
- 输出：返回分配好的 `Span`，或把空闲 `Span` 挂回对应页桶并更新 `PageMap`。
- 未涉及：无

## 对外接口
- 对主模块暴露：`PageCache::GetInstance()`、`AllocSpan(size_t, size_t)`、`ReleaseSpan(Span*)`、`Reset()`、`GetMutex()`；内部协作接口包括 `PageMap::GetSpan()`、`SetSpan()`、`ClearRange()`、`Reset()`。
- 调用约束：`AllocSpan()` 由线程安全包装获取全局锁；`PageMap::SetSpan()` / `ClearRange()` / `Reset()` 必须在持有 `PageCache::mutex_` 时调用。
- 头文件位置：`ammalloc/include/ammalloc/page_cache.h`
- 未涉及：无

## 不变量
- 空闲 `Span` 只能位于与自身 `page_num` 匹配的 `span_lists_[page_num]` 中。
- 合并完成后，`PageMap` 必须把新 `Span` 覆盖的所有页重新映射到同一个 `Span*`。
- `PageMap` 树节点只增不减，单个 `RadixNode` 不会在正常运行中单独释放。
- 未涉及：无

## 所有权与生命周期
- 所有者：`PageCache` 通过 `span_pool_` 持有全部 `Span` 元数据；`PageMap` 通过各自 `ObjectPool` 持有 `RadixRootNode` 和 `RadixNode`。
- 借用关系：`CentralCache` 只借用 `Span` 并在空闲时归还；`PageAllocator` 只负责底层页映射，不拥有 `Span` 元数据。
- 生命周期边界：系统补货或分割时创建新 `Span` 元数据；超过 `128` 页的 `Span` 在 `ReleaseSpan()` 中直接 `SystemFree()` 并删除元数据；`Reset()` 会清空所有空闲桶并释放对象池内存。
- 未涉及：无

## 并发约束
- 并发模型：全局共享后端，核心页操作串行化到 `mutex_`。
- 同步要求：`AllocSpan()` / `ReleaseSpan()` / `Reset()` 在全局锁内维护 `span_lists_` 和 `PageMap` 写路径；`PageMap::GetSpan()` 读路径依赖 acquire/release 原子发布语义而不加锁。
- 禁止事项：禁止在未持有 `PageCache::mutex_` 时修改 radix 树结构；禁止由上层直接释放 `Span` 元数据。
- 未涉及：无

## 性能约束
- 热路径：`PageMap::GetSpan()`、`AllocSpanLocked()` 的精确命中与切分逻辑、`ReleaseSpan()` 的邻居查找与合并。
- 约束：尽量命中缓存页桶并避免系统调用；`PageMap::GetSpan()` 保持固定层数原子加载；系统补货统一按 `128` 页进行以提升复用率。
- 监控点：关注空闲页桶命中率、合并成功率、`PageMap` 读延迟，以及全局页锁在大对象与慢路径混合场景下的竞争。
- 未涉及：无

## 已否决方案
- 方案：使用覆盖整个虚拟地址空间的扁平页映射数组。
  - 原因：设计文档明确选择多层基数树来适配稀疏、高位地址空间，并避免巨大的平面数组带来的内存浪费或 OOM。
- 无

## 未决问题
- 无

## 待办事项
- 无
