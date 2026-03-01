# 🌌 AetherMind 开发进度与技术决策日志 (Development Log)

> **[AI 助手指令]**: 本文档是 AetherMind 项目的“开发黑匣子”。
> 1. 每当完成重大 Commits、解决复杂 Bug 或进行架构重构时，**必须**在此同步更新。
> 2. 任务引用请严格遵循 `[TODO: 任务简短描述]` 格式，并确保与 `docs/ammalloc_todo_list.md` 中的项对应。
> 3. 记录应侧重于“为什么这么做”而非“做了什么”。

---

## 📅 2026-03-02 (Monday)
### 🚀 今日概要
初始化 AetherMind 开发日志系统，同步 `ammalloc` 模块的遗留技术债并建立任务追踪机制。

### 🧩 任务关联 (Task Linkage)
- [ ] **[TODO: Radix Tree 57-bit Fix]** - 正在进行风险评估。
- [ ] **[TODO: PageHeapScavenger]** - 处于架构设计阶段。

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
    - **验证**: 通过 `tests/unit/test_alignment.cpp` 验证。

---
[ 查看完整 TODO List ](ammalloc_todo_list.md)
