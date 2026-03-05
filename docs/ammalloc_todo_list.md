本文档记录了 AetherMind 内存池项目的已知缺陷修复计划、性能优化点以及未来向工业级功能对齐的演进路线。

> **🔗 相关资源**: [📅 查看开发进度日志 (Development Log)](DEVELOPMENT_LOG.md)

### 🔴 P0: 紧急修复 (Critical Fixes)

*涉及内存安全和核心正确性，必须立即修复。*

- [ ] #### **解决“自举（Bootstrapping）”与“递归死锁”问题 [Bug/Safety]**

- 背景：当 `ammalloc` 接管系统 `malloc` 后，分配器内部使用的 STL 容器、`std::jthread` 或日志库（如 `spdlog`）可能会反过来调用 `malloc`。这会导致无限递归和死锁。

- 方案：
  1. 在 `am_malloc` 入口处实现 `thread_local` 递归守卫 (`in_malloc` flag)。
     -  如果 `in_malloc == true`，说明是分配器内部（例如 std::jthread 或 spdlog）触发了二次分配
     - 对策：此时直接走“最底层路径”，绕过 `PageCache`，直接通过 `PageAllocator::SystemAlloc (mmap) `返回内存，或者返回一个预先分配好的小静态缓冲区。
  2. 如果 `in_malloc == false`，将其设为 `true`，执行正常逻辑，结束后设为 `false`。
  3. 在递归路径（`in_malloc == true`）下，由于 `mmap` 分配的内存无法通过 `PageMap::GetSpan` 正常释放（因为我们还没把它封装成 `Span` 注册到树里），这会导致分配器内部泄露（虽然很少）。为了完美解决，可以在递归路径下使用一个简单的“静态紧急缓冲区”或者也将其注册为普通的 `SystemAlloc` 并在 `am_free` 中通过检查 `PageMap` 是否为空来处理。
   
- 优先级：最高。这是实现 `PageHeapScavenger` (基于 `jthread`) 的先决条件。

- [ ] #### **HugePageCache 使用 std::vector 违反自举约束 [Bug/Safety]**

- 背景：`PageAllocator` 底层 `HugePageCache` 使用 `std::vector` 作为缓存容器。若 `am_malloc` 已通过 `LD_PRELOAD` 替换系统 `malloc`，`std::vector` 的 `reserve/push_back` 会触发系统 `malloc`，导致无限递归 → 栈溢出。

- 方案：将 `std::vector<void*>` 替换为定长栈数组：
  ```cpp
  void* cache_[kMaxCacheSize];
  size_t count_{0};
  ```
- 文件位置：`ammalloc/src/page_allocator.cpp:66`

- [ ] #### **Span::FreeObject 缺少 double-free 防护 [Bug/Safety]**

- 背景：当前实现直接设置 bitmap 位并递减 `use_count`，无重复释放检查。若用户 double-free，`use_count` 会下溢为 `SIZE_MAX`，此后 `AllocObject()` 永远返回 `nullptr`。

- 方案：在 `FreeObject` 中检查 bitmap 位状态，若已释放则触发 `AM_DCHECK`：
  ```cpp
  uint64_t bit = 1ULL << bit_pos;
  AM_DCHECK(!(bitmap[bitmap_idx] & bit), "double free detected");
  bitmap[bitmap_idx] |= bit;
  --use_count;
  ```
- 文件位置：`ammalloc/src/span.cpp:98-106`
- 相关测试：`test_span.cpp:DoubleFreeCorruption` 已证实此 bug。

- [ ] #### **GetOneSpan Release 模式空指针解引用 [Bug/Safety]**

- 背景：`AM_DCHECK(span != nullptr)` 在 Release 构建（`NDEBUG`）下编译为空。若 `AllocSpan` 返回 `nullptr`（OOM），下一行 `span->Init(size)` 直接崩溃。

- 方案：将 `AM_DCHECK` 替换为运行时检查：
  ```cpp
  auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
  if (!span) return nullptr;
  span->Init(size);
  ```
- 文件位置：`ammalloc/src/central_cache.cpp:303-305`

- [x] #### **修复 ObjectPool 内存对齐失效问题 [Bug]**

- 背景：当前 ObjectPool::New 直接在 ChunkHeader (16 Bytes) 之后分配对象。对于 Span (alignas 64) 和 RadixNode (alignas 4096)，这会导致严重的未对齐访问。

- 后果：
  1. RadixNode 跨页导致基数树物理映射错乱。
  2. Span 跨缓存行导致伪共享和原子操作性能下降。
  
- 方案：
  1. 在 ObjectPool::New 中，计算 ChunkHeader 之后的地址。
  2. 使用 alignof(T) 计算需要的 Padding。
  3. 调整 data_ 到对齐后的地址，并确保剩余空间计算扣除了 Padding。
  
- 状态：已完成，见 `ammallloc/page_allocator.h`。

- [ ] ####   **修复 Radix Tree 地址空间限制 (48-bit vs 57-bit) [Bug]** 

- 背景：当前 4 层基数树仅覆盖 48 位虚拟地址空间 (9 bits * 4 levels + 12 bits offset)。在支持 5-level paging (57-bit) 的现代 CPU (如 Intel Ice Lake+) 上，mmap 可能返回高位地址，导致 `PageMap::GetSpan` 数组越界崩溃。

- 方案：在 `GetSpan` 和 `SetSpan` 中增加地址范围检查，拒绝超出 48-bit 的地址，或者实现动态层级调整。

### 🟠 P1: 性能微调 (Performance Tuning)
*基于 Benchmark 数据和 Review 意见的针对性优化，性价比极高。*

- [x] #### **实现 CentralCache 预取机制 (Prefetching) [Perf]**
- 背景：当前 FetchRange 的慢速路径（查 SpanList Bitmap）只获取 batch_num 个对象返回给 ThreadCache，此时 TransferCache 依然是空的。下一次请求大概率又要走慢速路径。
- 方案：
  1. 在 Slow Path 中，一次性从 Span 申请 batch_num + N 个对象（或填满 tc_capacity）。
  2. 将一部分返回给 ThreadCache，剩下的直接填入 TransferCache 数组。
- 收益：大幅减少 Span Bitmap 的扫描频率和 span_lock (Mutex) 的争抢。Benchmark 显示多线程小对象分配吞吐量提升 3-5 倍。
- 状态：已完成，逻辑包含在 `FetchRange` 中。

- [x] #### **优化 SizeClass 边界测试 [Test]**
- 背景：确保索引计算逻辑在跨组边界（如 128B, 129B, 256B）时的绝对正确性。
- 方案：增加针对 SizeClass::Index 和 SizeClass::Size 的互逆性单元测试，覆盖所有 Size Class 的边界值。
- 状态：已完成，见 `tests/unit/test_size_class.cpp`。

- [ ] #### **优化 CentralCache 锁粒度与 Bitmap 扫描 [Perf]** 

- 背景：`FetchRange` 持有 `span_list_lock` 调用 `AllocObject`。对于大 Span (128页)，`AllocObject` 内部的 Bitmap 扫描可能耗时较长，阻塞其他线程。
- 方案：在 `Span` 中维护 `scan_cursor` 提示，记录上次扫描位置，避免每次从头扫描 Bitmap。

#### **优化 ThreadCache 对象分配内存占用 [Mem]**

- 背景：`ThreadCache` 对象本身约 1KB，当前直接使用 `SystemAlloc` 分配一个 4KB 页，每个线程浪费约 3KB 内存。在数千线程的高并发场景下浪费显著。
- 方案：使用全局 `ObjectPool<ThreadCache>` 替代裸页分配。

### 🟡 P2: 功能对齐 (Feature Parity)
*对齐 TCMalloc/Jemalloc 的核心功能，使其具备在生产环境长期运行的能力。*

- [x] #### **实现 PageHeapScavenger (后台清理线程) [Feature]**
- 背景：目前内存只进不出（除非合并出超大 Span）。长期运行会导致 RSS 虚高。
- 方案：
  1. 启动一个后台守护线程 (`std::jthread` + `stop_token`)。
  2. 定期扫描 PageCache 中闲置超过 `kIdleThresholdMs` (如 5s) 的 Span。
  3. 调用 `madvise(MADV_DONTNEED)` 释放物理内存（保留虚拟地址）。
  4. 维护 Span 的 `is_committed` 状态。
- 状态：**已实现并修复所有 P0 问题**（启动集成、并发安全、析构顺序、时间戳语义）。详见下方关联条目。

- [x] #### **修复 PageHeapScavenger 并发安全风险 [Bug]**
- 背景：`ScavengeOnePass()` 将 Span 从 `span_lists_` 摘下后释放锁执行 `madvise`，此时 `ReleaseSpan()` 可能通过 `PageMap` 找到该 Span 并尝试 `erase`，导致链表损坏。
- 方案：摘下 Span 前临时设置 `span->is_used = true` 使其对合并逻辑不可见，挂回后恢复 `false`。
- 严重级：P0
- 状态：已修复（见 `page_heap_scavenger.cpp:86, 123`）

- [x] #### **修复 PageHeapScavenger 析构顺序 UAF 风险 [Bug]**
- 背景：`PageHeapScavenger` 和 `PageCache` 均为函数内静态单例，析构顺序跨 TU 不确定。
- 方案：改为显式生命周期管理（`ammalloc_init`/`ammalloc_shutdown` 钩子）或采用 leaky singleton 策略。
- 严重级：P0
- 状态：已修复（采用 placement new + 静态存储的 leaky singleton 模式，见 `page_cache.h:127-132`、`page_heap_scavenger.h:16-19`）

- [x] #### **修复 `last_used_time_ms` 语义不一致 [Bug]**
- 背景：新通过系统补货进入 PageCache 的 Span 未初始化 `last_used_time_ms`（默认 0），会被立即误判为 idle。
- 方案：在所有 Span 进入 free list 的路径统一写入当前时间戳，或特判 0 为“不参与清理”。
- 严重级：P2
- 状态：已修复（在 `ReleaseSpan`、`AllocSpanLocked` 的切分路径和系统补货路径统一初始化，见 `page_cache.cpp:227, 256, 332`）

- [ ] #### **实现 ThreadCache 垃圾回收 (GC) [Feature]**
- 背景：线程如果从忙碌转为闲置，其 ThreadCache 中囤积的内存无法被其他线程复用。
- 方案：
  1. 建立全局 ThreadCacheRegistry 链表。
  2. 配合 Scavenger 线程，定期检查 ThreadCache 的使用活跃度。
  3. 强制回收闲置 ThreadCache 的 FreeList 到 CentralCache。

- [ ] #### **实现统计监控模块 (Statistics) [Observability]**
- 背景：缺乏运行时观测手段，难以排查内存泄漏或碎片化问题。
- 方案：
  1. 完善 PageAllocatorStats。
  2. 实现 GetStats() 接口，输出类似 TCMalloc 的文本报表（包含各层级缓存占用、碎片率、系统申请量等）。

### 🔵 P3: 架构演进 (Architecture Evolution)
*面向超大规模 NUMA 架构或特殊场景的长期规划。*

- [ ] **PageCache 分片 (Sharding) [Scalability]**
- 背景：在超多核（>64 Core）场景下，PageCache 的全局大锁可能成为瓶颈。
- 方案：将 PageCache 拆分为多个独立的 PageHeap（例如按 CPU ID 取模），降低锁粒度。

- [ ] **NUMA 感知 (NUMA Awareness) [Perf]**

- 背景：跨 NUMA 节点访问内存延迟较高。
- 方案：为每个 NUMA 节点维护独立的 CentralCache 和 PageCache 实例，优先分配本地内存。

- [ ] **Malloc Hooks [Feature]**

- 背景：用户需要自定义内存分配行为（如统计、故障注入）。
- 方案：提供 AddMallocHook / RemoveMallocHook 接口。

### 📝 变更日志 (Changelog)

- 2026-3-5
  - 完成 PageHeapScavenger 所有 P0 问题修复：
    - Off-list Span 并发安全：`is_used` 标记保护（`page_heap_scavenger.cpp:86, 123`）
    - 静态析构顺序 UAF：placement new + BSS 段静态存储的 leaky singleton（`page_cache.h`、`page_heap_scavenger.h`）
    - `last_used_time_ms` 语义：在切分、系统补货、释放三处统一初始化（`page_cache.cpp:227, 256, 332`）
  - 启动集成完成：`EnsureScavengerStarted()` 在 `am_malloc_slow_path` 中调用，支持 `AM_ENABLE_SCAVENGER` 环境变量控制
  - 更新开发日志记录完整修复过程。
  - **PageHeapScavenger 功能现在完整可用**。

- 2026-3-4
  - 完成 `PageHeapScavenger` 核心实现，标记为完成（待修复并发安全问题后可用）。
  - 同步 `docs/review.md` 中的 Code Review 问题到本 TODO 列表。
  - 新增 P0 级别问题：HugePageCache std::vector 自举违反、Span::FreeObject double-free 漏洞、GetOneSpan Release 空指针解引用。
  - 新增 P1 级别问题：ThreadCache 初始化锁、CentralCache unlock-relock 窗口、TLS 析构顺序。
  - 新增 P2/P3 级别功能需求：am_realloc/calloc/memalign、data 区域验证、ObjectPool 归属检查等。
  - 更新开发日志 (`DEVELOPMENT_LOG.md`) 记录 PageHeapScavenger 架构决策。

- 2026-3-1
  -  增加 Code Review 发现的 5 个新优化项 (Radix Tree Fix, CentralCache Lock, ThreadCache Pool, Safety)。
- 2026-2-27: 
  - 初始化 TODO 列表。
  - 修复 ObjectPool 内存对齐 Bug。
  - 增加 SizeClass 边界单元测试。
  - 实现 CentralCache 预取机制，性能大幅提升 (Benchmark: +350% ~ +500% in multi-threaded small object allocs)。
