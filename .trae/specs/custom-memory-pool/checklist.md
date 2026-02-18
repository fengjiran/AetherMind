# 自定义内存池模块 - 验证检查清单

## 设计评审
- [ ] PRD 中的所有目标都清晰明确
- [ ] 非目标（Out of Scope）已明确界定，避免范围蔓延
- [ ] 所有验收标准（Acceptance Criteria）都有明确的验证方式
- [ ] 开放问题已列出，以便后续决策

## 实现计划评审
- [ ] 所有任务都有明确的优先级（P0/P1/P2）
- [ ] 任务依赖关系正确，没有循环依赖
- [ ] 每个任务都有可验证的测试要求
- [ ] 任务粒度适中，没有过大或过小的任务
- [ ] 所有验收标准都被至少一个任务覆盖

## Task 1 验证：完善 ammalloc 的 Reset 接口
- [ ] PageCache::Reset() 接口完整可用
- [ ] ThreadCache 可以正确清理和重置
- [ ] CentralCache 可以正确清理和重置
- [ ] 调用 Reset 后内存池回到初始状态
- [ ] Reset 后可以正常进行新的内存分配
- [ ] 相关单元测试全部通过

## Task 2 验证：实现内存统计接口
- [ ] 已分配内存大小统计正确
- [ ] 预留内存大小统计正确
- [ ] 分配次数统计正确
- [ ] 分配内存后统计数据正确更新
- [ ] 释放内存后统计数据正确更新
- [ ] Reset 后统计数据清零
- [ ] 相关单元测试全部通过

## Task 3 验证：实现 AmmallocAllocator 类
- [ ] AmmallocAllocator 继承自 Allocator
- [ ] allocate() 方法正确调用 am_malloc()
- [ ] deallocate() 方法正确调用 am_free()
- [ ] GetAllocatedSize() 方法返回正确数据
- [ ] GetReservedSize() 方法返回正确数据
- [ ] Reset() 静态方法正常工作
- [ ] DataPtr 机制正常工作
- [ ] 相关单元测试全部通过

## Task 4 验证：提供 AmmallocAllocator 的独立访问接口
- [ ] 暂不使用 REGISTER_ALLOCATOR 宏注册
- [ ] AmmallocAllocator 提供独立的访问接口
- [ ] 可以独立获取和使用 AmmallocAllocator
- [ ] 现有 CPUAllocator 不受影响，继续正常工作
- [ ] 相关单元测试全部通过

## Task 5 验证：实现 Transfer Cache
- [ ] Transfer Cache 插入到 ThreadCache 和 CentralCache 之间
- [ ] 无锁 stack 或 array 数据结构工作正常
- [ ] 跨线程内存可以正确流转
- [ ] 多线程性能测试显示锁竞争减少
- [ ] 相关单元测试全部通过

## Task 6 验证：实现 PageHeapScavenger
- [ ] PageHeapScavenger 后台线程可以正常启动和停止
- [ ] 定期扫描 PageCache 中的空闲 Span
- [ ] 长时间未使用的 Span 被正确标记
- [ ] madvise(MADV_DONTNEED) 被正确调用
- [ ] 峰值内存后 RSS 可以下降
- [ ] 相关单元测试全部通过

## Task 7 验证：实现 ThreadCacheRegistry + GCThread
- [ ] ThreadCacheRegistry 正确实现
- [ ] 所有 ThreadCache 在创建时注册到全局链表
- [ ] GCThread 后台线程可以正常启动和停止
- [ ] GCThread 定期遍历并检查 ThreadCache
- [ ] 闲置线程的缓存被正确搜刮归还给 CentralCache
- [ ] 相关单元测试全部通过

## Task 8 验证：实现完整的 Statistics 接口
- [ ] GetStats() 接口完整实现
- [ ] 统计信息包含 ThreadCache 占用
- [ ] 统计信息包含 CentralCache 占用
- [ ] 统计信息包含 PageCache 闲置
- [ ] 统计信息包含向系统申请的内存
- [ ] 输出格式兼容 Google 文本格式
- [ ] 统计数据准确无误
- [ ] 相关单元测试全部通过

## Task 9 验证：实现 Heap Profiler
- [ ] 采样机制正常工作
- [ ] 每分配一定量内存记录堆栈调用
- [ ] 可以生成有效的 pprof 格式 profile 文件
- [ ] 支持 profile 的启停控制
- [ ] profile 文件可以被 pprof 工具解析
- [ ] 相关单元测试全部通过

## Task 10 验证：实现 Heap Checker
- [ ] Heap Checker 功能完整
- [ ] 可以检测到简单的内存泄漏
- [ ] 跟踪所有未释放的分配
- [ ] 泄漏报告包含有用信息
- [ ] 相关单元测试全部通过

## Task 11 验证：实现 System Allocator (VMA)
- [ ] System Allocator 抽象层完整实现
- [ ] 支持 Huge Page (THP) 对齐分配（如可用）
- [ ] 支持 NUMA 感知的内存分配（如可用）
- [ ] 相关单元测试全部通过

## Task 12 验证：实现 Malloc Hooks
- [ ] AddMallocHook() API 完整实现
- [ ] 支持在分配前插入回调
- [ ] 支持在分配后插入回调
- [ ] 支持在释放前插入回调
- [ ] 支持在释放后插入回调
- [ ] 支持多个 hook 的链式调用
- [ ] 可以成功注册和注销 hook
- [ ] hook 在正确的时机被调用
- [ ] 相关单元测试全部通过

## Task 13 验证：编写单元测试
- [ ] 为 AmmallocAllocator 编写了完整的单元测试
- [ ] 为 Transfer Cache 编写了单元测试
- [ ] 测试覆盖各种大小的内存分配（小对象、大对象）
- [ ] 测试覆盖多线程环境下的内存分配
- [ ] 测试覆盖 Reset 功能
- [ ] 测试覆盖内存统计功能
- [ ] 所有单元测试通过
- [ ] 多线程测试无数据竞争

## Task 14 验证：性能基准测试
- [ ] 性能基准测试框架可用
- [ ] 对比了 AmmallocAllocator 和 CPUAllocator 的性能
- [ ] 测试了小对象分配延迟
- [ ] 测试了多线程环境下的吞吐量
- [ ] 测试了内存碎片率
- [ ] 测试了 Transfer Cache 的性能提升
- [ ] 小对象分配延迟降低 ≥ 50%
- [ ] 多线程环境下性能良好
- [ ] 性能报告清晰易读

## Task 15 验证：集成测试
- [ ] 运行了现有的 AetherMind 测试套件
- [ ] 所有现有单元测试通过
- [ ] 验证了与 Tensor 系统的集成
- [ ] 验证了与其他模块的兼容性
- [ ] 集成测试无内存泄漏

## 整体验证
- [ ] 所有 P0 任务已完成
- [ ] 所有 P0 和 P1 任务的验收标准已满足
- [ ] AmmallocAllocator 性能显著优于 CPUAllocator
- [ ] Transfer Cache 工作正常，锁竞争减少
- [ ] PageHeapScavenger 工作正常，RSS 可以下降
- [ ] Thread Cache GC 工作正常
- [ ] Statistics & Profiler 工作正常
- [ ] 代码符合项目现有风格和约定
- [ ] 没有引入明显的性能回退
- [ ] 文档更新到位
