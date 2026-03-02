本文档记录了 AetherMind 内存池项目的已知缺陷修复计划、性能优化点以及未来向工业级功能对齐的演进路线。

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

- [ ] #### **实现 PageHeapScavenger (后台清理线程) [Feature]**
- 背景：目前内存只进不出（除非合并出超大 Span）。长期运行会导致 RSS 虚高。
- 方案：
  1. 启动一个后台守护线程。
  2. 定期扫描 PageCache 中闲置超过 kIdleThresholdMs (如 10s) 的 Span。
  3. 调用 madvise(MADV_DONTNEED) 释放物理内存（保留虚拟地址）。
  4. 维护 Span 的 is_unmapped 状态。
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

- 2026-3-1
  -  增加 Code Review 发现的 5 个新优化项 (Radix Tree Fix, CentralCache Lock, ThreadCache Pool, Safety)。
- 2026-2-27: 
  - 初始化 TODO 列表。
  - 修复 ObjectPool 内存对齐 Bug。
  - 增加 SizeClass 边界单元测试。
  - 实现 CentralCache 预取机制，性能大幅提升 (Benchmark: +350% ~ +500% in multi-threaded small object allocs)。
