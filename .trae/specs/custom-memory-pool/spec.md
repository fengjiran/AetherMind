# 自定义内存池模块 - 产品需求文档

## Overview
- **Summary**: 基于现有 ammalloc 实现，构建完整的自定义内存池模块，将其集成到 AetherMind 的 Allocator 体系中，为大模型推理引擎提供高效的底层内存分配能力。
- **Purpose**: 解决现有 CPUAllocator 使用标准 posix_memalign 分配器性能不足的问题，提供具有三级缓存（ThreadCache → CentralCache → PageCache）的高性能内存分配器，支持小对象快速分配、内存复用和高效的大对象分配。
- **Target Users**: 大模型推理引擎开发者、需要高性能内存分配的 AI 应用开发者。

## Goals
- 完善现有 ammalloc 实现，确保功能完整和稳定
- 实现 Transfer Cache，优化跨线程内存流转
- 实现 PageHeapScavenger，定期归还未使用内存给操作系统
- 实现完整的 Profiler & Statistics，提供内存可观测性
- 实现 Thread Cache GC，治理线程缓存
- 实现 System Allocator (VMA)，支持大页和 NUMA
- 实现 Malloc Hooks，提供扩展性
- 实现 AmmallocAllocator，继承自 Allocator，但暂不注册为默认 CPU 分配器
- 提供内存统计和监控能力
- 支持内存池的重置和清理，用于测试隔离
- 保持与现有 Allocator API 的兼容性

## Non-Goals (Out of Scope)
- 不实现跨设备（CUDA/CANN）的内存池（第一阶段仅 CPU）
- 不实现分布式内存池
- 不实现内存压缩功能

## Background & Context
AetherMind 项目当前具备：
- **现有内存体系**：Allocator 基类、AllocatorTable、CPUAllocator（使用 posix_memalign）
- **自定义内存分配器（ammalloc）**：
  - 完整的三级缓存架构：ThreadCache（无锁）→ CentralCache（带锁）→ PageCache
  - 支持小对象（≤ 256KB）和大对象分配
  - PageMap 用于快速查找 Span
  - 高效的内存复用机制
- **问题**：
  - 当前 CPUAllocator 未使用 ammalloc，性能有优化空间
  - 缺少工业级内存分配器的关键治理和观测模块（参考 TCMalloc）

### ammalloc 架构现状与目标
当前架构：`ThreadCache <-> CentralCache <-> PageCache`
目标架构（参考 TCMalloc）：
```
ThreadCache
    ↓↑
Transfer Cache (新增)
    ↓↑
CentralCache
    ↓↑
PageCache
    ↓↑
System Allocator (VMA, 新增)
    ↓↑
OS (mmap/madvise)
```

## Functional Requirements

### 核心功能
- **FR-1**: 实现 AmmallocAllocator，继承自 Allocator，内部使用 am_malloc/am_free
- **FR-2**: 暂不将 AmmallocAllocator 注册为默认 CPU 分配器，保持独立可测试
- **FR-3**: 提供内存统计接口，包括已分配内存大小、预留内存大小、分配次数等
- **FR-4**: 支持内存池重置，释放所有未使用的内存并清理内部状态（用于测试隔离）
- **FR-5**: 支持配置选项，如 PageSize、MaxThreadCacheSize 等
- **FR-6**: 保持与现有 DataPtr 机制的兼容性

### 治理模块 (Governance)
- **FR-7**: 实现 Transfer Cache，在 ThreadCache 和 CentralCache 之间插入无锁中转层
  - 优化跨线程内存流转，减少 CentralCache 锁竞争
  - 实现无锁 stack 或 array 结构
- **FR-8**: 实现 PageHeapScavenger，后台清理线程
  - 定期扫描 PageCache 中的空闲 Span
  - 对长时间未使用的 Span 调用 madvise(MADV_DONTNEED) 归还 OS
- **FR-9**: 实现 ThreadCacheRegistry + GCThread
  - 全局注册所有 ThreadCache
  - 后台定期搜刮闲置线程的缓存归还给 CentralCache

### 观测模块 (Observability)
- **FR-10**: 实现完整的 Statistics 接口
  - GetStats() 输出详细统计（ThreadCache、CentralCache、PageCache、系统占用）
  - 兼容 Google 文本格式
- **FR-11**: 实现 Heap Profiler (pprof)
  - 采样机制：每分配一定量内存记录堆栈
  - 生成 pprof 格式 profile 文件
- **FR-12**: 实现 Heap Checker
  - 检测内存泄漏

### 系统抽象与扩展
- **FR-13**: 实现 System Allocator (Virtual Memory Allocator)
  - 支持 Huge Page (THP) 对齐分配
  - 支持 NUMA 感知的内存分配
- **FR-14**: 实现 Malloc Hooks
  - AddMallocHook() API
  - 支持在分配/释放前后插入自定义回调

## Non-Functional Requirements
- **NFR-1**: 性能：小对象分配（≤ 4KB）延迟比 posix_memalign 降低 50% 以上
- **NFR-2**: 内存效率：内存碎片率低于 10%
- **NFR-3**: 可扩展性：支持多线程环境，无明显锁竞争
- **NFR-4**: 正确性：所有内存分配/释放操作安全，无内存泄漏或 use-after-free

## Constraints
- **Technical**: 基于现有 ammalloc 实现，保持代码一致性
- **Business**: 优先支持 CPU 平台
- **Dependencies**: 依赖现有 ammalloc、Allocator、DataPtr 等基础设施

## Assumptions
- 大模型推理引擎中存在大量小对象分配
- 多线程环境下的内存分配是常见场景
- 测试环境需要内存池可重置以实现测试隔离

## Acceptance Criteria

### AC-1: AmmallocAllocator 实现
- **Given**: 现有 ammalloc 和 Allocator 框架已就绪
- **When**: 实现 AmmallocAllocator 类
- **Then**: AmmallocAllocator 可以独立使用，不影响现有 CPUAllocator
- **Verification**: `programmatic`

### AC-2: 内存分配/释放功能
- **Given**: AmmallocAllocator 已实现
- **When**: 执行 allocate/deallocate 操作
- **Then**: 内存可以正确分配和释放，DataPtr 机制正常工作
- **Verification**: `programmatic`
- **Notes**: 测试各种大小的内存分配（小对象、大对象）

### AC-3: 内存统计功能
- **Given**: AmmallocAllocator 已实现
- **When**: 查询内存统计信息
- **Then**: 可以获取到正确的已分配内存大小、预留内存大小等统计数据
- **Verification**: `programmatic`

### AC-4: 内存池重置功能
- **Given**: AmmallocAllocator 已实现
- **When**: 调用 Reset 接口
- **Then**: 所有未使用的内存被释放，内部状态清理干净，可以用于下一次测试
- **Verification**: `programmatic`
- **Notes**: 用于解决测试隔离问题

### AC-5: Transfer Cache 实现
- **Given**: 现有 ThreadCache 和 CentralCache 已就绪
- **When**: 实现 Transfer Cache
- **Then**: 跨线程内存流转性能提升，CentralCache 锁竞争减少
- **Verification**: `programmatic`
- **Notes**: 多线程测试验证性能提升

### AC-6: PageHeapScavenger 实现
- **Given**: PageCache 已就绪
- **When**: 实现 PageHeapScavenger 后台线程
- **Then**: 长时间未使用的内存通过 madvise 归还给 OS，RSS 可以下降
- **Verification**: `programmatic`
- **Notes**: 测试峰值内存后 RSS 下降

### AC-7: Thread Cache GC 实现
- **Given**: ThreadCacheRegistry 已实现
- **When**: GCThread 定期运行
- **Then**: 闲置线程的缓存被搜刮归还给 CentralCache
- **Verification**: `programmatic`

### AC-8: Statistics & Profiler 实现
- **Given**: 内存池运行中
- **When**: 调用 GetStats() 或启用 profiler
- **Then**: 可以获取详细的内存统计和 profile 文件
- **Verification**: `programmatic`
- **Notes**: 统计格式兼容 Google 格式

### AC-9: System Allocator (VMA) 实现
- **Given**: PageAllocator 已就绪
- **When**: 实现 System Allocator
- **Then**: 支持 Huge Page 和 NUMA 感知分配
- **Verification**: `programmatic`

### AC-10: Malloc Hooks 实现
- **Given**: 内存分配器运行中
- **When**: 注册自定义 hook
- **Then**: hook 在分配/释放前后被正确调用
- **Verification**: `programmatic`

### AC-11: 性能对比
- **Given**: AmmallocAllocator 和现有 CPUAllocator 都可用
- **When**: 运行性能基准测试
- **Then**: AmmallocAllocator 的小对象分配性能显著优于 CPUAllocator
- **Verification**: `programmatic`
- **Notes**: 小对象分配延迟降低 ≥ 50%

### AC-12: API 兼容性
- **Given**: AmmallocAllocator 已实现
- **When**: 使用现有 Allocator API 调用 AmmallocAllocator
- **Then**: 所有现有代码无需修改即可正常工作
- **Verification**: `programmatic`

## Open Questions
- [ ] 是否需要完全替换 CPUAllocator，还是让两者共存？
- [ ] 内存统计信息需要暴露到什么粒度？
- [ ] 是否需要支持内存池的动态配置（运行时调整参数）？
