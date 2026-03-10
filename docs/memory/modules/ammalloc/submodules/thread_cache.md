---
scope: submodule
module: thread_cache
parent: ammalloc
depends_on: [central_cache, size_class]
adr_refs:
  - ../adrs/ADR-005.md
last_verified: 2026-03-10
owner: team
status: active
---

# thread_cache 子模块记忆

> 保存路径：`docs/memory/modules/ammalloc/submodules/thread_cache.md`
> 父模块：`docs/memory/modules/ammalloc/module.md`
> 用途：记录 `thread_cache` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：为 `size <= SizeConfig::MAX_TC_SIZE` 的小对象维护线程局部 `FreeList` 阵列；在本线程内执行无锁 `Allocate()` / `Deallocate()`；在局部缓存空或达到水位线时批量与 `CentralCache` 交互。
- 不负责：跨线程共享缓存、`PageMap` 查询、页级 `Span` 合并，以及 `mmap` / `munmap` / `madvise`。

### 边界
- 输入来源：`ammalloc/src/ammalloc.cpp` 在小对象路径调用 `ThreadCache::Allocate(size)` / `ThreadCache::Deallocate(ptr, size)`；线程退出时调用 `ReleaseAll()`。
- 输出去向：向调用方返回对象指针；或把批量对象经 `CentralCache::FetchRange()` / `ReleaseListToSpans()` 转交全局中端缓存。
- 不直接管理：共享 bucket 状态、`Span` 元数据所有权、系统页映射。
- 未涉及：无

## 已确认事实
- 已验证约束：`ThreadCache` 使用 `alignas(SystemConfig::CACHE_LINE_SIZE)`；内部固定持有 `std::array<FreeList, SizeClass::kNumSizeClasses>`；复制和移动均被禁用。
- 已验证限制：`Allocate()` / `Deallocate()` 通过 `AM_DCHECK` 要求 `size <= SizeConfig::MAX_TC_SIZE`；分配慢路径会把 `SizeClass::RoundUp(size)` 的对齐尺寸传给后端。
- ADR 关联：[ADR-005: 无 STL 约束](../adrs/ADR-005.md)
- 非阻塞注意事项：`FreeList::max_size_` 初始为 `1`，并在慢路径中按 `batch_num / 4`（至少为 `1`）递增；当水位条件满足时会批量回收而不是逐个回收。
- 未涉及：无

## 核心抽象
### 关键抽象
- `FreeList`：嵌入式 LIFO 单链表；节点复用对象自身内存保存 `next` 指针；记录当前元素数和动态水位线。
- `ThreadCache`：线程私有前端缓存；按 SizeClass 索引访问 `FreeList`，快路径只做 `push()` / `pop()`，慢路径才进入 `CentralCache`。

### 数据流
- 输入：用户请求大小、待释放对象指针，以及线程退出事件。
- 处理：通过 `SizeClass::Index(size)` 选桶；命中时直接操作本地 `FreeList`；未命中或缓存过大时批量向 `CentralCache` 取回或归还对象。
- 输出：返回单个对象指针，或构造嵌入式链表后批量交给 `CentralCache`。
- 未涉及：无

## 对外接口
- 对主模块暴露：`ThreadCache::Allocate(size_t)`、`ThreadCache::Deallocate(void*, size_t)`、`ThreadCache::ReleaseAll()`。
- 调用约束：调用方必须保证对象大小属于 ThreadCache 管理范围；`Deallocate()` 要求 `ptr != nullptr`；`ReleaseAll()` 会把所有局部对象回退到 `CentralCache`。
- 头文件位置：`ammalloc/include/ammalloc/thread_cache.h`
- 未涉及：无

## 不变量
- 每次访问 `free_lists_` 都通过 `SizeClass::Index(size)` 选定唯一桶。
- 快路径只读写当前线程的 `FreeList`，不接触共享锁或共享 `TransferCache`。
- `ReleaseAll()` 逐桶弹出对象并按对应 `SizeClass::Size(i)` 归还，不直接释放 `Span` 元数据。
- 未涉及：无

## 所有权与生命周期
- 所有者：TLS `ThreadCache` 实例由 `ammalloc/src/ammalloc.cpp` 使用 `PageAllocator::SystemAlloc()` 分配并 placement new；`FreeList` 节点对应的小对象内存由上层分配体系拥有。
- 借用关系：`ThreadCache` 只短暂借用对象指针；不拥有 `Span`、`TransferCache` 或 `PageMap`。
- 生命周期边界：首次慢路径请求时按需创建；线程退出时 `ThreadCacheCleaner` 先 `ReleaseAll()` 再析构并归还承载 `ThreadCache` 的页。
- 未涉及：无

## 并发约束
- 并发模型：严格线程亲和；每个实例只服务所属线程。
- 同步要求：快路径无锁；跨线程资源交换只能通过 `CentralCache` 完成。
- 禁止事项：禁止跨线程共享、移动或复用某线程的 `ThreadCache` 指针。
- 未涉及：无

## 性能约束
- 热路径：`ThreadCache::Allocate()`、`ThreadCache::Deallocate()`、`FreeList::pop()`、`FreeList::push()`。
- 约束：快路径保持 O(1) 且完全无锁；热路径依赖 `AM_ALWAYS_INLINE`、`AM_LIKELY` / `AM_UNLIKELY`；避免进入 `CentralCache`、`PageMap` 或系统调用。
- 监控点：设计基线要求 TLS 前端快路径约 `~3.8 ns`；应关注命中率、水位线增长后的局部内存占用，以及批量回退频率。
- 未涉及：无

## 已否决方案
- 无

## 未决问题
- 无

## 待办事项
- 无
