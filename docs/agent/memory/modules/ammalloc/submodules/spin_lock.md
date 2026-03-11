---
scope: submodule
module: spin_lock
parent: ammalloc
depends_on: []
adr_refs: []
last_verified: 2026-03-10
owner: team
status: active
---

# spin_lock 子模块记忆

> 保存路径：`docs/agent/memory/modules/ammalloc/submodules/spin_lock.md`
> 父模块：`docs/agent/memory/modules/ammalloc/module.md`
> 用途：记录 `spin_lock` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：提供 `SpinLock` 互斥原语；以 TTAS（Test-and-Test-and-Set）算法保护 `CentralCache` 的短临界区 `TransferCache` 操作。
- 不负责：长临界区阻塞等待、公平调度、条件同步或页级全局锁管理。

### 边界
- 输入来源：`CentralCache::Bucket` 在访问 `transfer_cache` 指针数组前后调用 `lock()` / `unlock()` 或 `try_lock()`。
- 输出去向：为调用方建立短临界区互斥边界，并通过 acquire/release 语义发布可见性。
- 不直接管理：`SpanList`、`PageCache`、线程生命周期或 OS 调度策略。
- 未涉及：无

## 已确认事实
- 已验证约束：`SpinLock` 内部唯一状态为 `std::atomic<bool> locked_{false}`；复制与赋值均被禁用。
- 已验证限制：实现明确使用 `std::memory_order_relaxed`、`std::memory_order_acquire`、`std::memory_order_release`，不依赖默认 `seq_cst`。
- ADR 关联：无
- 非阻塞注意事项：`lock()` 采用四阶段流程：relaxed 读测试、acquire `exchange` 抢占、`details::CPUPause()` 退避、超过 `2000` 次自旋后 `std::this_thread::yield()`。
- 未涉及：无

## 核心抽象
### 关键抽象
- `SpinLock`：面向极短临界区的用户态自旋锁；以 TTAS 减少缓存行抖动和总线风暴。
- `locked_`：原子布尔锁位；`false -> true` 表示成功获取锁，`unlock()` 用 release 存储恢复为空闲状态。

### 数据流
- 输入：调用方的加锁、尝试加锁和解锁请求。
- 处理：先做乐观读取，只有观察到空闲时才执行带 acquire 语义的原子交换；竞争失败时暂停并必要时让出时间片。
- 输出：返回加锁是否成功，或释放锁让下一个线程可见临界区更新。
- 未涉及：无

## 对外接口
- 对主模块暴露：`SpinLock::lock()`、`try_lock()`、`unlock()`。
- 调用约束：适用于极短临界区；调用方必须成对调用 `lock()` / `unlock()`；`try_lock()` 失败时不会阻塞。
- 头文件位置：`ammalloc/include/ammalloc/spin_lock.h`
- 未涉及：无

## 不变量
- 只有成功的 `exchange(true, std::memory_order_acquire)` 才表示获得锁。
- `unlock()` 始终以 release 语义写回 `false`，建立临界区可见性边界。
- 失败等待阶段只做 relaxed 读取和退避，不在竞争期间反复触发重型同步序。
- 未涉及：无

## 所有权与生命周期
- 所有者：持有该成员的上层结构（当前主要是 `CentralCache::Bucket`）拥有 `SpinLock` 实例。
- 借用关系：调用线程只临时借用锁，不拥有被保护的数据结构。
- 生命周期边界：随所属对象构造和销毁，无独立资源分配。
- 未涉及：无

## 并发约束
- 并发模型：多线程互斥；适合保护极短、不可阻塞的共享状态更新。
- 同步要求：调用方必须把所有受保护的共享写入放在 `lock()` / `unlock()` 之间。
- 禁止事项：禁止把 `SpinLock` 用于长临界区或可能睡眠/阻塞的代码段。
- 未涉及：无

## 性能约束
- 热路径：`lock()` 的成功抢锁分支与 `try_lock()`。
- 约束：在短临界区内用忙等待替代上下文切换；竞争失败时先 `CPUPause()` 再深度退避，减少缓存一致性流量。
- 监控点：关注 `TransferCache` 临界区长度、自旋次数与 `yield` 触发频率。
- 未涉及：无

## 已否决方案
- 方案：在 `TransferCache` 快路径上直接使用 `std::mutex`。
  - 原因：设计文档明确将 `SpinLock` 用于短临界区，以避免内核调度和上下文切换开销；`std::mutex` 保留给 `SpanList` 与 `PageCache` 这类长临界区。
- 无

## 未决问题
- 无

## 待办事项
- 无
