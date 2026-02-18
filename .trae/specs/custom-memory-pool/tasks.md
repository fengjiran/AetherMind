# 自定义内存池模块 - 实现计划

## [ ] Task 1: 完善 ammalloc 的 Reset 接口
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 确保 ammalloc 的 PageCache::Reset() 接口完整可用
  - 确保 ThreadCache、CentralCache 的清理和重置功能
  - 验证内存池可以完全重置到初始状态
- **Acceptance Criteria Addressed**: [AC-4]
- **Test Requirements**:
  - `programmatic` TR-1.1: 调用 Reset 后内存池回到初始状态
  - `programmatic` TR-1.2: Reset 后可以正常进行新的内存分配
- **Notes**: 参考之前的 Reset 接口实现

## [ ] Task 2: 实现内存统计接口
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 为 ammalloc 添加基础内存统计功能
  - 跟踪已分配内存大小、预留内存大小、分配次数等
  - 在 PageCache、CentralCache 中集成统计逻辑
- **Acceptance Criteria Addressed**: [AC-3]
- **Test Requirements**:
  - `programmatic` TR-2.1: 分配内存后统计数据正确更新
  - `programmatic` TR-2.2: 释放内存后统计数据正确更新
  - `programmatic` TR-2.3: Reset 后统计数据清零

## [ ] Task 3: 实现 AmmallocAllocator 类
- **Priority**: P0
- **Depends On**: [Task 1, Task 2]
- **Description**: 
  - 创建 AmmallocAllocator，继承自 Allocator
  - 实现 allocate() 方法，内部调用 am_malloc()
  - 实现 deallocate() 方法，内部调用 am_free()
  - 实现 GetAllocatedSize() 和 GetReservedSize() 方法
  - 添加 Reset() 静态方法
- **Acceptance Criteria Addressed**: [AC-1, AC-2, AC-3, AC-4, AC-12]
- **Test Requirements**:
  - `programmatic` TR-3.1: AmmallocAllocator 可以成功分配内存
  - `programmatic` TR-3.2: AmmallocAllocator 可以成功释放内存
  - `programmatic` TR-3.3: DataPtr 机制正常工作
  - `programmatic` TR-3.4: Reset() 方法正常工作
  - `programmatic` TR-3.5: 内存统计方法返回正确数据

## [ ] Task 4: 提供 AmmallocAllocator 的独立访问接口
- **Priority**: P1
- **Depends On**: [Task 3]
- **Description**: 
  - 暂不使用 REGISTER_ALLOCATOR 宏注册
  - 提供独立的访问接口，如 AmmallocAllocator::GetInstance()
  - 确保可以独立测试，不影响现有 CPUAllocator
- **Acceptance Criteria Addressed**: [AC-1, AC-12]
- **Test Requirements**:
  - `programmatic` TR-4.1: AmmallocAllocator 可以独立获取和使用
  - `programmatic` TR-4.2: 现有 CPUAllocator 不受影响，继续正常工作

## [ ] Task 5: 实现 Transfer Cache
- **Priority**: P0
- **Depends On**: [Task 1]
- **Description**: 
  - 在 ThreadCache 和 CentralCache 之间插入 Transfer Cache 层
  - 实现无锁 stack 或 array 结构
  - 优化跨线程内存流转，减少 CentralCache 锁竞争
- **Acceptance Criteria Addressed**: [AC-5]
- **Test Requirements**:
  - `programmatic` TR-5.1: Transfer Cache 无锁数据结构工作正常
  - `programmatic` TR-5.2: 跨线程内存可以正确流转
  - `programmatic` TR-5.3: 多线程性能测试显示锁竞争减少
- **Notes**: 参考 TCMalloc 的 Transfer Cache 设计

## [ ] Task 6: 实现 PageHeapScavenger
- **Priority**: P1
- **Depends On**: [Task 1]
- **Description**: 
  - 实现后台清理线程 PageHeapScavenger
  - 定期扫描 PageCache 中的空闲 Span
  - 对长时间未使用的 Span 调用 madvise(MADV_DONTNEED)
- **Acceptance Criteria Addressed**: [AC-6]
- **Test Requirements**:
  - `programmatic` TR-6.1: Scavenger 线程可以正常启动和停止
  - `programmatic` TR-6.2: 空闲 Span 被正确标记和归还
  - `programmatic` TR-6.3: 峰值内存后 RSS 可以下降
- **Notes**: 测试时注意 madvise 的行为

## [ ] Task 7: 实现 ThreadCacheRegistry + GCThread
- **Priority**: P1
- **Depends On**: [Task 1]
- **Description**: 
  - 实现 ThreadCacheRegistry，全局注册所有 ThreadCache
  - 实现 GCThread 后台线程
  - 定期搜刮闲置线程的缓存归还给 CentralCache
- **Acceptance Criteria Addressed**: [AC-7]
- **Test Requirements**:
  - `programmatic` TR-7.1: ThreadCache 正确注册到全局链表
  - `programmatic` TR-7.2: GCThread 定期遍历并检查 ThreadCache
  - `programmatic` TR-7.3: 闲置线程的缓存被正确搜刮

## [ ] Task 8: 实现完整的 Statistics 接口
- **Priority**: P1
- **Depends On**: [Task 2]
- **Description**: 
  - 完善 GetStats()，输出详细统计
  - 统计信息包括：ThreadCache、CentralCache、PageCache、系统占用
  - 输出格式兼容 Google 文本格式
- **Acceptance Criteria Addressed**: [AC-8]
- **Test Requirements**:
  - `programmatic` TR-8.1: GetStats() 返回完整的统计信息
  - `programmatic` TR-8.2: 统计数据准确无误
  - `programmatic` TR-8.3: 输出格式符合预期

## [ ] Task 9: 实现 Heap Profiler
- **Priority**: P2
- **Depends On**: [Task 8]
- **Description**: 
  - 实现采样机制：每分配一定量内存记录堆栈
  - 生成 pprof 格式 profile 文件
  - 支持 profile 的启停控制
- **Acceptance Criteria Addressed**: [AC-8]
- **Test Requirements**:
  - `programmatic` TR-9.1: 采样机制正常工作
  - `programmatic` TR-9.2: 可以生成有效的 pprof 文件
  - `programmatic` TR-9.3: profile 文件可以被 pprof 工具解析
- **Notes**: 可以用 gperftools 作为参考

## [ ] Task 10: 实现 Heap Checker
- **Priority**: P2
- **Depends On**: [Task 8]
- **Description**: 
  - 实现内存泄漏检测功能
  - 跟踪所有未释放的分配
  - 输出泄漏报告
- **Acceptance Criteria Addressed**: [AC-8]
- **Test Requirements**:
  - `programmatic` TR-10.1: 可以检测到简单的内存泄漏
  - `programmatic` TR-10.2: 泄漏报告包含有用信息

## [ ] Task 11: 实现 System Allocator (VMA)
- **Priority**: P2
- **Depends On**: [Task 1]
- **Description**: 
  - 抽象 System Allocator 层
  - 支持 Huge Page (THP) 对齐分配
  - 支持 NUMA 感知的内存分配
- **Acceptance Criteria Addressed**: [AC-9]
- **Test Requirements**:
  - `programmatic` TR-11.1: System Allocator 抽象层工作正常
  - `programmatic` TR-11.2: 支持 Huge Page 分配（如可用）
  - `programmatic` TR-11.3: NUMA 感知分配正常（如可用）

## [ ] Task 12: 实现 Malloc Hooks
- **Priority**: P2
- **Depends On**: [Task 1]
- **Description**: 
  - 实现 AddMallocHook() API
  - 支持在分配前、分配后、释放前、释放后插入回调
  - 支持多个 hook 的链式调用
- **Acceptance Criteria Addressed**: [AC-10]
- **Test Requirements**:
  - `programmatic` TR-12.1: 可以成功注册和注销 hook
  - `programmatic` TR-12.2: hook 在正确的时机被调用
  - `programmatic` TR-12.3: 多个 hook 可以链式工作

## [ ] Task 13: 编写单元测试
- **Priority**: P0
- **Depends On**: [Task 3, Task 4, Task 5]
- **Description**: 
  - 为 AmmallocAllocator 编写完整的单元测试
  - 为 Transfer Cache 编写单元测试
  - 测试各种大小的内存分配（小对象、大对象）
  - 测试多线程环境下的内存分配
  - 测试 Reset 功能
  - 测试内存统计功能
- **Acceptance Criteria Addressed**: [AC-1, AC-2, AC-3, AC-4, AC-5]
- **Test Requirements**:
  - `programmatic` TR-13.1: 所有单元测试通过
  - `programmatic` TR-13.2: 多线程测试无数据竞争
  - `programmatic` TR-13.3: 测试覆盖各种边界情况

## [ ] Task 14: 性能基准测试
- **Priority**: P1
- **Depends On**: [Task 4, Task 5]
- **Description**: 
  - 实现性能基准测试框架
  - 对比 AmmallocAllocator 和 CPUAllocator 的性能
  - 测试小对象分配延迟
  - 测试多线程环境下的吞吐量
  - 测试内存碎片率
  - 测试 Transfer Cache 的性能提升
- **Acceptance Criteria Addressed**: [AC-11]
- **Test Requirements**:
  - `programmatic` TR-14.1: 小对象分配延迟降低 ≥ 50%
  - `programmatic` TR-14.2: 多线程环境下性能良好
  - `human-judgement` TR-14.3: 性能报告清晰易读

## [ ] Task 15: 集成测试
- **Priority**: P1
- **Depends On**: [Task 4]
- **Description**: 
  - 运行现有的 AetherMind 测试套件
  - 确保所有现有测试通过
  - 验证与 Tensor 系统的集成
  - 验证与其他模块的兼容性
- **Acceptance Criteria Addressed**: [AC-12]
- **Test Requirements**:
  - `programmatic` TR-15.1: 所有现有单元测试通过
  - `programmatic` TR-15.2: 集成测试无内存泄漏
