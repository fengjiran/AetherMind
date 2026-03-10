---
scope: submodule
module: page_allocator
parent: ammalloc
depends_on: []
adr_refs:
  - ../adrs/ADR-003.md
last_verified: 2026-03-10
owner: team
status: active
---

# page_allocator 子模块记忆

> 保存路径：`docs/memory/modules/ammalloc/submodules/page_allocator.md`
> 父模块：`docs/memory/modules/ammalloc/module.md`
> 用途：记录 `page_allocator` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：封装 `mmap` / `munmap` / `madvise`；按页数提供 `SystemAlloc()` / `SystemFree()`；实现普通页、大页和标准 `2MB` 大页缓存路径；维护原子统计信息。
- 不负责：`Span` 合并、对象尺寸映射、线程局部缓存或 `PageMap` 元数据。

### 边界
- 输入来源：`PageCache`、`CentralCache` 初始化、`ObjectPool` 等内部路径通过页数调用 `SystemAlloc()` / `SystemFree()`。
- 输出去向：返回页对齐的虚拟内存地址或 `nullptr`；把释放的映射归还 OS，或把标准大页缓存到 `HugePageCache`。
- 不直接管理：`Span` 生命周期、桶级锁策略、用户对象指针链表。
- 未涉及：无

## 已确认事实
- 已验证约束：`SystemAlloc(page_num)` 以 `page_num << SystemConfig::PAGE_SHIFT` 计算字节数；`page_num == 0` 直接返回 `nullptr` 并告警；统计字段全部使用显式原子更新。
- 已验证限制：小于 `SystemConfig::HUGE_PAGE_SIZE / 2` 的请求优先走普通页路径；仅当大小恰好等于 `SystemConfig::HUGE_PAGE_SIZE` 时才尝试命中 `HugePageCache`。
- ADR 关联：[ADR-003: 乐观大页策略](../adrs/ADR-003.md)
- 非阻塞注意事项：`HugePageCache` 是内部单例，最大容量固定为 `16`；`AllocWithRetry()` 最多重试 `PageConfig::MAX_ALLOC_RETRIES` 次；若大页路径失败会统计一次 `huge_fallback_to_normal_count` 后退化到普通页分配。
- 未涉及：无

## 核心抽象
### 关键抽象
- `PageAllocator`：页级系统接口门面；根据请求大小在普通页、大页和缓存路径之间做选择。
- `HugePageCache`：仅缓存标准 `2MB` 大页地址的内部数组缓存；`Get()` / `Put()` / `ReleaseAll()` 均由内部 `std::mutex` 保护。
- `PageAllocatorStats`：记录普通页、大页、缓存命中、失败与释放等统计计数。

### 数据流
- 输入：页数请求、待释放地址和对应页数。
- 处理：小请求直接 `mmap` 普通页；大请求优先尝试乐观大页分配，必要时裁剪到 `2MB` 对齐；释放标准大页时先 `madvise(MADV_DONTNEED)` 再尝试放入 `HugePageCache`，否则 `munmap`。
- 输出：返回可用虚拟地址，或把映射释放/缓存并更新统计。
- 未涉及：无

## 对外接口
- 对主模块暴露：`PageAllocator::SystemAlloc(size_t)`、`SystemFree(void*, size_t)`、`GetStats()`、`ResetStats()`、`ReleaseHugePageCache()`；同一头文件还定义复用该接口的 `ObjectPool<T>` 模板。
- 调用约束：调用方必须以页数为单位传入长度；`SystemFree()` 在 `ptr == nullptr` 或 `page_num == 0` 时直接返回；`ObjectPool<T>` 在底层页分配失败时会抛出 `std::bad_alloc`。
- 头文件位置：`ammalloc/include/ammalloc/page_allocator.h`
- 未涉及：无

## 不变量
- `HugePageCache` 只缓存标准 `2MB` 大页，其他大小不会进入缓存。
- 大页释放时如果成功放入缓存，则保留 VMA 而不立即 `munmap`。
- 所有统计更新都显式指定内存序，不依赖默认 `seq_cst`。
- 未涉及：无

## 所有权与生命周期
- 所有者：调用方拥有 `SystemAlloc()` 返回的映射，直到显式调用 `SystemFree()`；`HugePageCache` 拥有暂存的标准大页地址。
- 借用关系：`PageAllocator` 不拥有 `Span` 或用户对象元数据；`ObjectPool<T>` 只借助 `PageAllocator` 获取页块。
- 生命周期边界：缓存中的大页由 `HugePageCache` 保留，直到缓存溢出、`ReleaseHugePageCache()` 调用或进程结束时统一 `munmap`。
- 未涉及：无

## 并发约束
- 并发模型：可被多线程并发调用；标准大页缓存通过内部 `std::mutex` 串行化访问。
- 同步要求：统计使用原子变量；`HugePageCache::Get()` / `Put()` / `ReleaseAll()` 必须在其内部锁保护下执行。
- 禁止事项：禁止把非标准 `2MB` 块放入 `HugePageCache`；禁止假设释放缓存块后其虚拟地址已经被 `munmap`。
- 未涉及：无

## 性能约束
- 热路径：`SystemAlloc()` 的普通页分支、标准大页缓存命中，以及乐观大页分配的直接命中分支。
- 约束：优先复用缓存的大页或直接命中对齐的大页地址，避免额外 `munmap`；仅在必要时进入 over-allocate-and-trim 分支。
- 监控点：关注 `huge_cache_hit_count`、`huge_fallback_to_normal_count`、`madvise_failed_count`、`mmap_enomem_count` 和对齐裁剪浪费字节数。
- 未涉及：无

## 已否决方案
- 方案：大页分配始终先多申请一段内存再裁剪对齐。
  - 原因：工程说明明确采用“乐观大页”策略，先尝试直接申请目标大小；若 OS 恰好返回 `2MB` 对齐地址，可避免额外 VMA 碎片和额外 `munmap`。
- 无

## 未决问题
- 无

## 待办事项
- 无
