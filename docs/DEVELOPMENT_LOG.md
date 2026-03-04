# 🌌 AetherMind 开发进度与技术决策日志 (Development Log)

> **[AI 助手指令]**: 本文档是 AetherMind 项目的“开发黑匣子”。
> 1. 每当完成重大 Commits、解决复杂 Bug 或进行架构重构时，**必须**在此同步更新。
> 2. 任务引用请严格遵循 `[TODO: 任务简短描述]` 格式，并确保与 `docs/ammalloc_todo_list.md` 中的项对应。
> 3. 记录应侧重于“为什么这么做”而非“做了什么”。

---

## 📅 2026-03-04 (Wednesday)
### 🚀 今日概要
完成 `PageHeapScavenger` 后台清理线程的核心实现，通过独立后台线程周期性扫描 PageCache 的空闲 Span 列表，将长期闲置的物理内存通过 `MADV_DONTNEED` 归还给操作系统，同时保留虚拟地址映射以避免后续的 `mmap` 开销。

### 🧩 任务关联 (Task Linkage)
- [x] **[TODO: PageHeapScavenger]** - 核心功能实现完成，进入 Code Review 和缺陷修复阶段。
- [ ] **[TODO: PageHeapScavenger Integration]** - 尚未接入 `am_malloc` 启动路径，当前为未激活状态。

### ⚠️ 遇到的问题与解决方案 (Troubleshooting)
1. **[Issue] Off-list Span 的并发安全风险 (P0)**
    - **描述**: `ScavengeOnePass()` 将 Span 从 `PageCache::span_lists_` 摘下后释放锁执行 `madvise`，此时 `ReleaseSpan()` 的合并逻辑仍可能通过 `PageMap` 找到该 Span 并尝试 `erase`，导致链表损坏或崩溃。
    - **解决方案**: 摘下 Span 前临时设置 `span->is_used = true` 使其对合并逻辑不可见，挂回后恢复 `false`；或退而求其次全程持锁执行 `madvise`（牺牲并发性换取安全）。

2. **[Issue] 静态析构顺序导致的 UAF 风险 (P0)**
    - **描述**: `PageHeapScavenger` 和 `PageCache` 均为函数内静态单例，析构顺序跨 TU 不确定。后台线程可能在 `PageCache` 析构后仍尝试访问它。
    - **解决方案**: 改为显式生命周期管理（`ammalloc_init`/`ammalloc_shutdown` 钩子）或采用 leaky singleton 策略，避免依赖静态析构顺序。

3. **[Issue] `last_used_time_ms` 语义不一致 (P2)**
    - **描述**: 新通过系统补货进入 PageCache 的 Span 未初始化 `last_used_time_ms`（默认 0），会被立即误判为 idle 并触发不必要的 `madvise`。
    - **解决方案**: 在所有 Span 进入 free list 的路径统一写入当前时间戳，或特判 0 为“不参与清理”。

### 💡 架构思考 (Architectural Insights)
- **后台清理线程的并发模型**: 采用 `std::jthread` + `std::stop_token` + `std::condition_variable_any` 实现可中断的周期性扫描，符合 C++20 最佳实践。`madvise` 放在锁外的设计正确，但需配合 "in-flight span 标记" 机制避免并发合并冲突。
- **锁策略权衡**: 当前 `PageCache` 的全局大锁在 scavenger 频繁遍历场景下成为瓶颈。P2 后期可考虑将 `PageCache` 分片 (Sharding) 以降低锁竞争。
- **启动时机设计**: 经过对比 4 种启动方案（构造函数属性、首次 malloc、首次慢路径、阈值触发），确定采用**首次慢路径启动**作为主要策略。该方案避开热路径、延迟适中、无递归风险，且可通过环境变量 `AM_ENABLE_SCAVENGER` 控制开关。详见设计方案文档 [docs/page_heap_scavenger_start.md]。

---

## 📅 2026-03-02 (Monday)
### 🚀 今日概要
初始化 AetherMind 开发日志系统，同步 `ammalloc` 模块的遗留技术债并建立任务追踪机制。

### 🧩 任务关联 (Task Linkage)
- [ ] **[TODO: Radix Tree 57-bit Fix]** - 正在进行风险评估。
- [x] **[TODO: PageHeapScavenger]** - 核心功能实现完成，进入 Code Review 和缺陷修复阶段（详见 2026-03-04 条目）。

### ⚠️ 遇到的问题与解决方案 (Troubleshooting)
1. **[Issue] 跨平台虚拟地址空间差异**
    - **描述**: 在 macOS (Apple Silicon) 和现代 Linux (5-level paging) 上，虚拟地址位宽不一致导致基数树索引溢出。
    - **风险**: 4 层基数树硬编码 48-bit 假设在生产环境（尤其是高性能服务器）中极其危险。
    - **预研策略**: 考虑引入 `StaticConfig::MAX_VIRTUAL_BITS`，根据编译目标动态调整基数树层数，或在 `GetSpan` 快速路径加入 `AM_UNLIKELY` 的范围断言。

### 💡 架构思考 (Architectural Insights)
- **关于 PageCache 锁竞争**: 考虑到 P2 阶段的 `PageHeapScavenger` 会频繁遍历 PageCache，目前的全局大锁必然成为瓶颈。建议提前将 `PageCache 分片 (Sharding)` 的优先级从 P3 提升至 P2 后期，以配合后台清理线程的引入。

---

## 📅 2026-03-01 (Sunday)
### 🚀 今日概要
完成 `ammalloc` 核心组件的 Code Review，确立了 P0/P1 阶段的关键修复路径。

### 🧩 任务关联
- [x] **[TODO: ObjectPool 内存对齐]** - 修复完成。
- [x] **[TODO: CentralCache 预取]** - 性能验证通过，吞吐量提升显著。

### ⚠️ 遇到的问题与解决方案
1. **[Issue] ObjectPool 内存对齐导致基数树错乱**
    - **解决方案**: 在 `ObjectPool::New` 中引入 `alignof(T)` 计算 Padding，确保 `RadixNode` 严格 4KB 对齐，避免跨页。

---
[ 查看完整 TODO List ](ammalloc_todo_list.md)
