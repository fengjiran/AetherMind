---
scope: submodule
module: page_heap_scavenger
parent: ammalloc
depends_on: [page_cache]
adr_refs:
  - ../adrs/ADR-004.md
last_verified: 2026-03-10
owner: team
status: active
---

# page_heap_scavenger 子模块记忆

> 保存路径：`docs/agent/memory/modules/ammalloc/submodules/page_heap_scavenger.md`
> 父模块：`docs/agent/memory/modules/ammalloc/module.md`
> 用途：记录 `page_heap_scavenger` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：启动并停止后台 `std::jthread`；周期性扫描 `PageCache` 空闲 `Span`；对长时间空闲且仍处于 committed 状态的页调用 `madvise(MADV_DONTNEED)` 以释放物理内存。
- 不负责：分配新 `Span`、执行 `Span` 合并、管理 ThreadCache 或改变 SizeClass 策略。

### 边界
- 输入来源：`ammalloc/src/ammalloc.cpp` 通过 `EnsureScavengerStarted()` 在运行时按需启动；停止由 `Stop()` 发起。
- 输出去向：与 `PageCache` 共享空闲 `Span` 链表；对符合条件的页执行 `madvise`；更新 `Span::is_used`、`is_committed`、`last_used_time_ms`。
- 不直接管理：`PageCache` 的 `Span` 所有权、用户对象、OS 映射创建与销毁。
- 未涉及：无

## 已确认事实
- 已验证约束：`PageHeapScavenger` 是单例；公开接口只有 `Start()` 和 `Stop()`；后台线程函数为 `ScavengeLoop(std::stop_token)`。
- 已验证限制：扫描周期固定为 `kScavengeIntervalMs = 1000`；空闲阈值固定为 `kIdleThresholdMs = 10000`；只处理 `!is_used && is_committed` 的空闲 `Span`。
- ADR 关联：[ADR-004: MADV_DONTNEED 策略](../adrs/ADR-004.md)
- 非阻塞注意事项：`Start()` 会避免重复启动；`Stop()` 通过 `request_stop()` + `join()` 收敛线程；`ammalloc/src/ammalloc.cpp` 仅在 `RuntimeConfig::EnableScavenger()` 为真时尝试启动，并在进程生命周期内用原子布尔值保证只触发一次启动逻辑。
- 未涉及：无

## 核心抽象
### 关键抽象
- `PageHeapScavenger`：后台清理器；持有 `std::jthread`、`condition_variable_any` 和内部互斥量。
- `ScavengeLoop`：可中断等待循环；利用 `stop_token` 与 `cv_.wait_for()` 组合实现可提前唤醒的睡眠。
- `ScavengeOnePass`：一次完整清理；遍历 `PageCache::span_lists_`，摘链、解锁后 `madvise`、再重新挂回。

### 数据流
- 输入：停止信号、当前时间戳，以及 `PageCache` 的空闲 `Span` 链表。
- 处理：从大页桶到小页桶扫描；在 `PageCache` 锁内摘出空闲且超过阈值的 `Span`，临时标记 `is_used = true`；解锁后执行 `madvise`，成功时设 `is_committed = false`；再加锁恢复为空闲并回插链表。
- 输出：释放部分物理内存并记录调试日志，不改变 `Span` 的虚拟地址区间。
- 未涉及：无

## 对外接口
- 对主模块暴露：`PageHeapScavenger::GetInstance()`、`Start()`、`Stop()`。
- 调用约束：`Start()` 只在 `scavenge_thread_` 尚未 `joinable()` 时启动新线程；`Stop()` 只对已启动线程生效。
- 头文件位置：`ammalloc/include/ammalloc/page_heap_scavenger.h`
- 未涉及：无

## 不变量
- 进入 `madvise` 前，目标 `Span` 会先从 `PageCache` 空闲链表摘下并暂时标记为 `is_used = true`，避免与合并逻辑并发冲突。
- 只有 `madvise` 成功时才把 `Span::is_committed` 置为 `false`。
- 重新挂回 `PageCache` 前必须把 `is_used` 恢复为 `false`，并刷新 `last_used_time_ms`。
- 未涉及：无

## 所有权与生命周期
- 所有者：`PageHeapScavenger` 自身拥有后台 `std::jthread` 与等待原语；`Span` 仍由 `PageCache` 持有。
- 借用关系：清理线程只暂时摘下 `PageCache` 的空闲 `Span`，不接管其元数据所有权。
- 生命周期边界：首次启动后线程常驻，直到 `Stop()` 请求停止并 join；单例本身使用静态存储 placement new。
- 未涉及：无

## 并发约束
- 并发模型：独立后台线程与前台分配/释放线程并发运行。
- 同步要求：遍历和摘链必须持有 `PageCache::GetMutex()`；耗时的 `madvise` 明确放在解锁区间执行；睡眠与停止通过 `stop_token` 和 `condition_variable_any` 协调。
- 禁止事项：禁止在持有 `PageCache` 全局锁时直接执行 `madvise`；禁止把仍在使用或已被 `MADV_DONTNEED` 的 `Span` 重复加入本轮清理链表。
- 未涉及：无

## 性能约束
- 热路径：`ScavengeOnePass()` 中的空闲 `Span` 扫描与锁外 `madvise`。
- 约束：清理线程必须把昂贵系统调用移到 `PageCache` 锁外，避免阻塞正常分配/释放；扫描顺序固定为从 `MAX_PAGE_NUM` 递减到 `1`。
- 监控点：关注每轮释放的物理内存字节数、空闲阈值命中率，以及后台线程对 `PageCache` 全局锁持有时间的影响。
- 未涉及：无

## 已否决方案
- 无

## 未决问题
- 无

## 待办事项
- 无
