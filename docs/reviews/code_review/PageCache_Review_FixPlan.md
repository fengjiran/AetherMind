# PageCache 评审与修订方案（Review + Fix Plan）

## 1. 结论

当前这版 `PageCache` **可以作为第一版可运行骨架**，但从 `ammalloc` 的项目目标来看（纳秒级延迟、线性并发扩展、内存安全），**尚不能作为稳定基线直接继续叠加复杂功能**。

核心原因不是“性能还不够”，而是：

1. **PageMap 的 lock-free 读语义与 Span 元数据回收语义没有闭合**
2. **Span 生命周期状态机定义不足**
3. **个别接口缺乏 release 模式下的硬性防御**
4. **Reset / 回收路径的并发契约不够清晰**

因此，这一层当前更适合被定义为：

> 结构方向正确，但仍处于“正确性协议需要收紧”的阶段。

---

## 2. 本次评审范围

本次评审基于以下实现：

- `page_cache.h`
- `page_cache.cpp`
- `span.h`
- `span.cpp`

重点关注四个维度：

- 并发正确性
- PageMap 一致性
- Span 生命周期管理
- 可回收性与后续可演进性

---

## 3. 主要问题分级

---

### P0：阻塞级问题（必须优先修复）

#### P0-1. `GetSpan()` 为 lock-free 读，但合并路径会立即销毁邻居 `Span`

**问题描述**

`PageMap::GetSpan()` 被设计为热路径 lock-free 读取；而 `ReleaseSpan()` 在左右合并时，会：

1. 通过 `GetSpan()` 找到邻居 `Span`
2. 从 free list 摘除邻居
3. 直接 `span_pool_.Delete(left_span/right_span)`
4. 最后再 `PageMap::SetSpan(span)` 重写合并后区间映射

这意味着：

- PageMap 中旧映射所指向的 `Span*`，在某个时间窗口内**可能已被销毁**
- 如果并发读线程在该窗口读取到旧指针，就会形成**悬空指针访问**

**本质**

这是一个典型的：

> lock-free 发布/回收协议未闭合

只要 `GetSpan()` 真正允许并发读，就不能在写路径里立即回收旧 `Span` 元数据。

**风险**

- Use-after-free
- 隐蔽竞态
- 极难复现的堆损坏

**结论**

这是当前实现中最关键的架构级问题。

---

#### P0-2. oversized span 归还 OS 时也存在同类生命周期风险

**问题描述**

对于 `page_num > MAX_PAGE_NUM` 的 span，`ReleaseSpan()` 的路径是：

1. `page_map_.ClearRange(...)`
2. `allocator_->SystemFree(...)`
3. `span_pool_.Delete(span)`

如果系统允许其他线程 lock-free 地并发 `GetSpan()`，那么这里同样存在：

- PageMap 清理与元数据销毁之间没有 reader-safe reclamation 协议
- 读线程可能观察到旧状态或刚被销毁的 `Span`

**结论**

这与 P0-1 是同一类问题，只是发生在 direct-to-OS 回收路径上。

---

#### P0-3. `AllocSpanLocked()` 没有在 release 语义上拒绝 `page_num == 0`

**问题描述**

当前实现只检查了过大页数，但没有显式拒绝 `page_num == 0`。

虽然有 `DCHECK(page_num >= 1)` 一类断言，但它只在 debug 路径有效，不能代替 release 防御。

**潜在后果**

- 可能构造出 `page_num == 0` 的 `Span`
- 切分逻辑失真
- 上层获得非法 span 元数据
- 后续释放/合并逻辑进入未定义行为区域

**结论**

应直接在入口做硬性防御。

---

### P1：高优先级问题（应尽快修复）

#### P1-1. `Reset()` 不是并发安全 reset，只能用于“全局静止点”

**问题描述**

`Reset()` 当前会：

- 回收 free lists 中的 spans
- 释放 span pool
- Reset page map

但它：

- 没有处理已借给上层、仍在使用中的 span
- 没有与 lock-free `GetSpan()` 形成并发安全协议
- 释放 radix tree 节点时默认没有并发读者

**风险**

如果有人把它当作“通用运行时 reset 接口”使用，会非常危险。

**结论**

该接口必须被明确收紧为：

> 仅测试使用，且调用前系统必须处于全局 quiescent 状态。

---

#### P1-2. 合并逻辑过度依赖“所有 free span 必然在 free list 中”

**问题描述**

当前左右合并的前提判断基本是：

- 邻居存在
- `!neighbor->IsUsed()`

随后就直接执行：

- `span_lists_[neighbor->page_num].erase(neighbor)`

这隐含了一个强不变量：

> 只要 span 不是 used，它就一定在对应 free list 中

这个假设在当前版本也许成立，但未来一旦引入：

- scavenger
- decommit 队列
- background reclaim
- lazy coalescing

这个不变量就很容易失效。

**风险**

- 对未入链节点执行 `erase`
- 双重摘链
- Span 状态错乱

---

### P2：中优先级问题（影响可维护性和后续演进）

#### P2-1. Span 状态机定义不足

当前 `Span` 更像只区分两种状态：

- used
- free

但 PageCache 未来如果要支持更成熟的后端能力，通常至少要区分：

- in use
- free & linked
- free & scavenged
- releasing to OS
- metadata-retired / waiting reclamation

如果不提前把状态机定义清楚，后面功能一叠加，PageCache 会越来越脆弱。

---

#### P2-2. PageMap 接口契约容易误导调用者

从接口命名和注释看，调用者容易自然理解为：

- `GetSpan()` 是线程安全 lock-free 读
- `SetSpan/ClearRange()` 在持锁下更新即可

但实际上，这组接口是否**整体并发安全**，取决于旧 `Span` 的回收时机。

也就是说，当前问题并不只是实现层面，而是：

> 接口契约表达得比真实安全边界更“乐观”

---

## 4. 修订总原则

接下来修复 PageCache，我建议遵循四条原则：

### 原则 1：先封 correctness，再谈进一步并发优化

在 allocator 里，后端元数据生命周期协议一旦不严密，后面无论怎样做 NUMA 分片、per-node cache、background reclaim，都会放大问题。

### 原则 2：明确 PageMap 的并发模型

必须二选一：

#### 路线 A：PageMap 真正支持 lock-free 并发读

那就必须配套：

- 延迟回收（epoch / QSBR / hazard pointer / retire list）
- 不能立即 `Delete(span)`
- reset / unmap / merge 都要纳入同一套回收协议

#### 路线 B：PageMap 只在“受控条件下”无锁读

例如：

- 读者保证不会与 span 元数据销毁并发
- 或更高层保证查表发生在对象已稳定归属期间

那就必须把这个限制写进接口契约，不能继续模糊表达为“天然线程安全”。

### 原则 3：引入明确的 Span 状态机

不要再只靠 `is_used_` 一个 bit 支撑整个 PageCache 后端语义。

### 原则 4：把测试态接口与运行时接口分开

`Reset()` 这类接口必须明确是测试专用，不应混淆为生产路径能力。

---

## 5. 建议的修订方案

---

### 方案 A：短期保守修复版（最适合作为下一步）

这是我最推荐的近期方案。

#### 目标

- 先让 PageCache 成为**严格正确的稳定基线**
- 暂时不追求复杂 lock-free 生命周期回收
- 为后续 NUMA 分片和 scavenger 演进打地基

#### 核心做法

##### 1. 收紧 PageMap 的并发语义

把当前语义明确改成：

> `GetSpan()` 可以无锁读取 PageMap，但调用方必须保证不会与 `Span` 元数据销毁并发。

换言之：

- 无锁 ≠ 完整并发安全回收
- 读路径只是“不加锁访问映射”，不是 RCU/epoch 读者

##### 2. 合并路径禁止立即销毁旧 span，先转入 retired list

在 `PageCache` 内部增加一个简单的 retired list：

- 被吞并的 `Span` 不立即 `Delete`
- 先挂到 `retired_spans_`
- 只在确定没有并发读者的时机统一回收

如果当前系统还没有统一 quiescent 协议，那么保守做法甚至可以是：

- 当前阶段先不真正回收被合并吞并的小 `Span` 元数据
- 暂时接受少量 metadata 滞留
- 等后续统一接入 epoch 再做回收闭合

这会牺牲一点 metadata 占用，但能显著降低 correctness 风险。

##### 3. oversized span 同样采用延后元数据回收

不要在 `SystemFree()` 后立刻 `Delete(span)`；改为：

- 清映射
- 释放实际内存
- span 元数据进入 retired list

##### 4. `AllocSpanLocked()` 入口显式拒绝 0

建议直接：

```cpp
if (page_num == 0) return nullptr;
```

不要依赖 DCHECK。

##### 5. `Reset()` 明确标为 test-only / quiescent-only

建议接口文档明确写出：

- only for tests
- caller must guarantee no checked-out spans
- caller must guarantee no concurrent readers/writers

---

### 方案 B：中期正式版（推荐后续演进目标）

当你准备把 PageCache 作为高并发正式后端时，应走这条路线。

#### 目标

让 `PageMap::GetSpan()` 真正成为：

> 合法支持并发读取的 lock-free 热路径

#### 核心机制

引入 **epoch-based reclamation**（推荐）或 QSBR。

##### 推荐原因

相对 allocator 场景，epoch/QSBR 通常比 hazard pointer 更合适：

- 读路径更轻
- Span 查表是高频路径
- 被 retire 的对象类型相对单一
- 更容易和线程缓存/线程注册体系结合

#### 需要做的事

1. 每个参与 `GetSpan()` 的线程进入 reader epoch
2. merge / clear / system free 后仅 retire span
3. 只有在所有读者都越过对应 epoch 后才真正回收元数据
4. `Reset()` 也必须走同一回收协议或要求全局静止

---

## 6. 建议新增的 Span 状态机

建议把 `Span` 状态从单 bit 升级为枚举，例如：

```cpp
enum class SpanState : uint8_t {
    InUse,
    FreeLinked,
    FreeScavenged,
    ReleasingToOS,
    Retired,
};
```

### 状态含义

- `InUse`：已分配给上层，不在 free list
- `FreeLinked`：空闲且挂在 PageCache free list 中
- `FreeScavenged`：空闲但已脱链，处于后台回收/去提交化状态
- `ReleasingToOS`：正在归还系统，防止其他路径错误合并
- `Retired`：逻辑上失效，但元数据尚未安全回收

### 好处

- 合并条件更清晰
- scavenger 更容易接入
- 不再依赖“`!IsUsed()` 就等于可 merge 且必在链上”这种脆弱假设

---

## 7. 建议修改的接口契约

---

### `PageMap::GetSpan()`

建议注释收紧为类似：

> Lock-free lookup of page index to span metadata.
> Caller must ensure returned span metadata is not concurrently reclaimed,
> unless a global reclamation protocol (e.g. epoch/QSBR) is active.

这样能避免误导后续维护者。

---

### `PageCache::Reset()`

建议明确：

> Test-only API.
> Requires global quiescence:
> no checked-out spans, no concurrent readers, no concurrent mutations.

---

### `ReleaseSpan()`

建议在接口层面注明：

- only accepts valid, non-null span
- span must not already be in PageCache free structure
- caller must transfer ownership back exactly once

这样能更清楚地定义“归还”的所有权语义。

---

## 8. 具体修改清单（行动项）

### 第一阶段：立即修改

1. `AllocSpanLocked()` 显式拒绝 `page_num == 0`
2. 为 `Reset()` 增补 test-only / quiescent-only 注释
3. 补充文档，收紧 `GetSpan()` 并发语义说明
4. 给 `ReleaseSpan()` / merge 路径加上更强的断言与注释

### 第二阶段：修复生命周期漏洞

5. 增加 `retired_spans_` 机制，禁止 merge 后立即 `Delete(neighbor)`
6. oversized span 归还 OS 后不立即回收 `Span` 元数据
7. 统一设计 retired span 的回收时机

### 第三阶段：演进为稳定后端

8. 引入正式的 epoch/QSBR 回收协议
9. 将 `Span` 状态升级为枚举状态机
10. 为 scavenger / decommit 预留状态与链路
11. 后续再考虑 NUMA 分片 PageCache

---

## 9. 我建议的下一步落地顺序

如果你现在准备继续改代码，我建议按下面顺序推进：

### 第一步

先做**保守正确性修复版**：

- 修 `page_num == 0`
- 明确接口契约
- merge / OS release 后不立即删除元数据

### 第二步

把这版作为“稳定基线”跑测试：

- 单线程功能测试
- Span split/coalesce 测试
- 重复 alloc/release 压测
- 边界页数测试
- reset 前置条件测试

### 第三步

再决定是否引入正式 epoch 回收协议。

这一步不建议和当前 correctness 修复混在一起，否则调试成本会很高。

---

## 10. 架构判断

从 allocator 架构角度看，这版 `PageCache`：

- **方向是对的**
- **分层职责是合理的**
- **数据结构选择在当前阶段是够用的**

但它目前最大的问题是：

> “映射可见性” 与 “元数据回收安全性” 之间还没有形成完整协议。

这类问题一旦不先收紧，后面继续叠加：

- NUMA 分片
- 后台 scavenger
- decommit/recommit
- 高并发 central/page cache 协作

都会把问题复杂度成倍放大。

所以正确策略不是立刻追求更花哨的数据结构，而是：

> 先把 PageCache 做成一个严格正确、生命周期清晰、可验证的后端基线。

---

## 11. 最终结论

这版实现可以概括为：

> **结构合理，但并发回收语义尚未封口。**

在 `ammalloc` 这样的高性能分配器项目里，这不是“小瑕疵”，而是决定后端是否能稳定演进的关键分水岭。

因此，建议你下一步优先做的不是继续叠加功能，而是：

1. 收紧 PageMap 并发契约
2. 修复 Span 元数据立即回收的问题
3. 建立更明确的 Span 状态机
4. 将 Reset 明确降级为测试态接口

把这四件事做好之后，PageCache 才适合作为后续 NUMA 分片和后端回收机制的可靠基础。
