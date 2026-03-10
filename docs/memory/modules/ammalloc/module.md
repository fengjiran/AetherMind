---
scope: module
module: ammalloc
parent: none
depends_on: []
adr_refs:
  - ./adrs/ADR-001.md
  - ./adrs/ADR-002.md
  - ./adrs/ADR-003.md
  - ./adrs/ADR-004.md
  - ./adrs/ADR-005.md
last_verified: 2026-03-10
owner: team
status: active
---

# ammalloc 主模块记忆

> 保存路径：`docs/memory/modules/ammalloc/module.md`
> 用途：记录 `ammalloc` 主模块的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 负责：提供高性能、多线程友好的内存分配/释放；以 `ThreadCache -> CentralCache -> PageCache -> PageAllocator` 分层处理对象与页；在核心路径内保持无递归 `malloc` 约束。
- 不负责：业务逻辑、模型推理、tokenizer 管理，以及调用方对象的业务语义。

### 边界
- 上游输入：调用方通过 `aethermind::am_malloc(size)` / `aethermind::am_free(ptr)` 发起请求；小对象先经 `SizeClass` 做尺寸映射，大对象按页数进入后端路径。
- 下游输出：向调用方返回对象指针或 `nullptr`；在释放、回收和系统补充时更新 `FreeList`、`Span`、`PageMap`，并调用 `mmap` / `munmap` / `madvise`。
- 不直接管理：推理张量、模型权重、tokenizer 资源，以及上层模块的生命周期状态。

### 子模块划分
- `thread_cache`：线程局部前端缓存，维护各 SizeClass 的 `FreeList`，提供无锁快路径；文件：`docs/memory/modules/ammalloc/submodules/thread_cache.md`
- `central_cache`：全局中端缓存，按 SizeClass 分桶，`TransferCache` 走 `SpinLock`，`SpanList` 走 `std::mutex`；文件：`docs/memory/modules/ammalloc/submodules/central_cache.md`
- `page_cache`：全局后端页缓存，负责 `Span` 分配、切分、合并，并维护 `PageMap` 写路径；文件：`docs/memory/modules/ammalloc/submodules/page_cache.md`
- `page_allocator`：OS 交互层，封装 `mmap` / `munmap` / `madvise` 与大页缓存；文件：`docs/memory/modules/ammalloc/submodules/page_allocator.md`
- `size_class`：请求大小到桶索引、对齐尺寸和批量搬运参数的 O(1) 映射；文件：`docs/memory/modules/ammalloc/submodules/size_class.md`
- `spin_lock`：TTAS 自旋锁，服务 `CentralCache` 快路径的短临界区；文件：`docs/memory/modules/ammalloc/submodules/spin_lock.md`
- `page_heap_scavenger`：后台清理线程，周期性对空闲且长时间未使用的 `Span` 执行 `MADV_DONTNEED`；文件：`docs/memory/modules/ammalloc/submodules/page_heap_scavenger.md`
- 未涉及：无

## 已确认事实
- 已验证约束：`ammalloc` 采用 `ThreadCache -> CentralCache -> PageCache -> PageAllocator` 分层架构；Phase 1 设计目标包含单线程 `< 5ns`、多核近似线性扩展、平均内部碎片约 `< 12.5%`。
- 已验证限制：核心分配/释放路径和元数据管理禁止使用会触发堆分配的 STL 容器及原生 `new` / `delete`；`ThreadCache` 与 `CentralCache::Bucket` 必须按缓存行对齐；公开 API 目前只有 `am_malloc()` / `am_free()`。
- ADR 关联：[ADR-001: TransferCache 设计](./adrs/ADR-001.md)、[ADR-002: RadixTree PageMap](./adrs/ADR-002.md)、[ADR-003: 乐观大页策略](./adrs/ADR-003.md)、[ADR-004: MADV_DONTNEED 策略](./adrs/ADR-004.md)、[ADR-005: 核心路径禁用 STL](./adrs/ADR-005.md)
- 非阻塞注意事项：`PageMap` 在架构约束中定义为 4 层基数树；当前实现采用 `RadixRootNode` + 3 层 `RadixNode` 的胖根布局，读路径无锁，`SetSpan()` / `ClearRange()` 写路径受 `PageCache::mutex_` 保护。
- 未涉及：无

## 核心抽象
### 关键抽象
- `FreeList`：嵌入式 LIFO 单链表，维护 `head_`、`size_`、`max_size_`，服务 TLS 缓存与批量对象搬运。
- `Span`：连续页区间及对象切分元数据，记录页范围、对象大小、位图、扫描游标、`use_count`、提交状态。
- `TransferCache`：`CentralCache::Bucket` 内的 O(1) 指针数组快缓存，容量按 `8 * batch_size` 规划，用于跨线程批量流转。
- `PageMap`：`PageID -> Span*` 的基数树索引，根节点惰性初始化，叶子保存 `Span*`。

### 数据流
- 输入：`am_malloc(size)` / `am_free(ptr)` 进入模块；尺寸先通过 `SizeClass` 选择桶和对齐策略。
- 转换：小对象优先走 `ThreadCache`；未命中时经 `CentralCache` 批量取放对象；仍不足则由 `PageCache` 分配或切分 `Span`，最终通过 `PageAllocator` 触达 OS；释放路径反向更新 `FreeList`、位图、`PageMap` 和空闲页链表。
- 输出：返回用户可用指针，或把对象/页归还到上层缓存、页缓存或 OS。
- 未涉及：无

## 对外接口
- 头文件 / API：公开接口位于 `ammalloc/include/ammalloc/ammalloc.h`；主要内部头文件位于 `ammalloc/include/ammalloc/`；实现位于 `ammalloc/src/`。
- 入口函数：`void* aethermind::am_malloc(size_t size)`；`void aethermind::am_free(void* ptr)`。`am_malloc` 在 TLS 未初始化或请求大于 `SizeConfig::MAX_TC_SIZE` 时进入 slow path；`am_free` 先通过 `PageMap::GetSpan(ptr)` 确认归属。
- 错误语义：`am_malloc` 在系统分配失败或 `ThreadCache` 创建失败时返回 `nullptr`；`am_free(nullptr)` 直接返回，未被 `PageMap` 识别的指针也直接返回；当前未公开 `am_realloc()`。
- 未涉及：无

## 不变量
- `ThreadCache` 快路径只访问线程本地 `FreeList`，不引入共享锁。
- `Span` 的位图分配状态必须与 `use_count` 保持一致；空闲小对象 `Span` 会从 `CentralCache` 退回 `PageCache`。
- `Span` 生命周期由 `PageCache` 的 `span_pool_` 管理；`CentralCache` 和 `ThreadCache` 只借用，不直接释放 `Span` 元数据。
- `PageMap` 读取保持无锁；树结构创建、映射写入和范围清理由 `PageCache` 锁保护。
- 未涉及：无

## 所有权与生命周期
- 所有者：TLS `ThreadCache` 由 `ammalloc/src/ammalloc.cpp` 在首次慢路径分配时创建，并由线程局部 `ThreadCacheCleaner` 回收；`Span`、`RadixRootNode`、`RadixNode` 分别由 `PageCache` / `PageMap` 内部 `ObjectPool` 持有。
- 借用关系：`CentralCache` 的 `SpanList` 暂持可分配 `Span`；`ThreadCache` 只持有对象指针，不持有 `Span` 元数据所有权；`PageMap` 为释放路径提供只读索引。
- 销毁时机：线程结束时 `ThreadCacheCleaner` 先 `ReleaseAll()` 再释放 TLS 缓存；超大 `Span` 在 `PageCache::ReleaseSpan()` 中直接归还 OS；Radix 节点正常运行中不单独释放，仅在 `PageMap::Reset()` 或进程结束时由对象池统一回收。
- 未涉及：无

## 并发约束
- 线程角色：调用线程直接访问 TLS `ThreadCache`；跨线程共享状态集中在 `CentralCache`、`PageCache`、`PageMap` 和 `PageHeapScavenger`。
- 同步要求：`ThreadCache` 无锁；`CentralCache::Bucket` 以 `SpinLock` 保护 `TransferCache`、以 `std::mutex` 保护 `SpanList`；`PageCache` 以全局 `std::mutex` 保护 `span_lists_` 与 `PageMap` 写路径；原子操作显式使用 `relaxed` / `acquire` / `release`。
- 禁止事项：不要跨线程共享 TLS `ThreadCache`；不要依赖原子默认 `seq_cst`；不要在持有 bucket `span_list_lock` 时直接进入 `PageCache` 长临界区，需先解锁再调用后端。
- 未涉及：无

## 性能约束
- 热路径：`ThreadCache::Allocate()` / `Deallocate()`、`FreeList::pop()` / `push()`、`SizeClass::Index()` / `RoundUp()`、`PageMap::GetSpan()`、`CentralCache` 的 `TransferCache` 批量搬运。
- 时间复杂度 / 常数项要求：快路径保持 O(1) 且无锁；`SizeClass` 依赖编译期查表与位运算；热路径避免隐藏 O(N^2) 扫描、额外系统调用和递归分配。
- 内存约束：性能基线为单线程极速路径约 `~3.8 ns`、随机尺寸分配约 `~26.0 ns`、16 线程高压 `64B` 场景约 `~8.9 us` / `100+ GiB/s`；内部碎片目标约 `< 12.5%`；`TransferCache` 采用一次性连续内存布局，核心结构保持缓存行对齐。
- 未涉及：无

## 已否决方案
- 无

## 未决问题
- `CentralCache::FetchRange()` 当前把预取量固定为一个 `batch_num`；源码已标注后续可能按 `TransferCache` 剩余容量动态调节。
- 无

## 待办事项
- [ ] 评估并实现 `CentralCache::FetchRange()` 中关于 `prefetch_target` 的动态调节 TODO，避免在高并发下把多余预取对象再退回慢路径。
- 无
