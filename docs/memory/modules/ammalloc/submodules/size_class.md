---
scope: submodule
module: size_class
parent: ammalloc
depends_on: []
adr_refs: []
last_verified: 2026-03-10
owner: team
status: active
---

# size_class 子模块记忆

> 保存路径：`docs/memory/modules/ammalloc/submodules/size_class.md`
> 父模块：`docs/memory/modules/ammalloc/module.md`
> 用途：记录 `size_class` 的长期稳定信息。只写已验证事实；缺失信息写 `无` 或 `未涉及`。

## 模块范围
### 职责
- 隶属主模块：`ammalloc`
- 负责：把请求大小映射到 SizeClass 索引；返回桶大小与对齐结果；计算批量搬运数量和 `CentralCache` 向 `PageCache` 申请的页数。
- 不负责：保存任何运行期缓存状态、管理对象内存、执行锁同步或系统页分配。

### 边界
- 输入来源：`ThreadCache`、`CentralCache` 和其他内部路径传入对象大小或桶索引。
- 输出去向：返回 `Index()`、`Size()`、`RoundUp()`、`CalculateBatchSize()`、`GetMovePageNum()` 的纯计算结果，供前中后端缓存使用。
- 不直接管理：`FreeList`、`Span`、TLS 或 OS 虚拟内存。
- 未涉及：无

## 已确认事实
- 已验证约束：`SizeClass` 是纯静态工具类，构造函数被删除；`kNumSizeClasses` 在编译期由 `details::CalculateIndex(SizeConfig::MAX_TC_SIZE) + 1` 计算。
- 已验证限制：`Index(size)` 对 `size > SizeConfig::MAX_TC_SIZE` 返回 `std::numeric_limits<size_t>::max()`；`RoundUp(size)` 在收到无效索引时直接返回原始 `size`。
- ADR 关联：无
- 非阻塞注意事项：`small_index_table_` 在编译期覆盖 `0..SizeConfig::kSmallSizeThreshold`；`size_table_` 在编译期覆盖全部 SizeClass；源码包含多组 `static_assert` 保证双向映射正确性。
- 未涉及：无

## 核心抽象
### 关键抽象
- `details::CalculateIndex()`：尺寸到桶索引的核心数学模型；`<= 128B` 走 `8B` 步进，之后按幂次区间的 `4` 档步进计算。
- `small_index_table_` / `size_table_`：编译期生成的查找表；分别优化小对象 `Size -> Index` 与全部 `Index -> Size` 查询。
- `SizeClass`：统一暴露尺寸映射、批量策略与页数策略的静态接口。

### 数据流
- 输入：用户请求大小或桶索引。
- 处理：小对象优先查表；大对象用 `std::bit_width` 和位运算计算组号与步长；再基于目标尺寸推导批量数和页数。
- 输出：返回对齐后大小、桶索引、批量传输数量或页数。
- 未涉及：无

## 对外接口
- 对主模块暴露：`Index(size_t)`、`Size(size_t)`、`SafeSize(size_t)`、`RoundUp(size_t)`、`CalculateBatchSize(size_t)`、`GetMovePageNum(size_t)`。
- 调用约束：`Size(idx)` 要求调用方保证 `idx < kNumSizeClasses`；`SafeSize(idx)` 在越界时会触发检查并返回 `0`；`GetMovePageNum()` 仅对有效对象大小有意义。
- 头文件位置：`ammalloc/include/ammalloc/size_class.h`
- 未涉及：无

## 不变量
- 对任意有效 `size`，`Size(Index(size)) >= size`。
- 小对象查表与大对象数学计算共同覆盖 `1..SizeConfig::MAX_TC_SIZE`。
- `CalculateBatchSize()` 的结果始终位于 `[2, 512]`（`size == 0` 时返回 `0`）。
- 未涉及：无

## 所有权与生命周期
- 所有者：无运行期拥有者；全部核心数据为编译期常量。
- 借用关系：调用方只读取静态结果，不持有额外资源。
- 生命周期边界：随程序映像存在，无独立初始化或销毁流程。
- 未涉及：无

## 并发约束
- 并发模型：纯函数，可被任意线程并发调用。
- 同步要求：无共享可变状态，无需锁或原子同步。
- 禁止事项：禁止假设 `Index()` 可处理大于 `MAX_TC_SIZE` 的 ThreadCache 对象尺寸。
- 未涉及：无

## 性能约束
- 热路径：`Index()`、`RoundUp()`、`CalculateBatchSize()`。
- 约束：查询保持 O(1)；小对象优先使用编译期表；大对象仅允许位运算和常数级计算。
- 监控点：关注随机尺寸分配场景的映射延迟，以及对数步进类间距对平均内部碎片率（设计目标约 `< 12.5%`）的影响。
- 未涉及：无

## 已否决方案
- 无

## 未决问题
- 无

## 待办事项
- 无
