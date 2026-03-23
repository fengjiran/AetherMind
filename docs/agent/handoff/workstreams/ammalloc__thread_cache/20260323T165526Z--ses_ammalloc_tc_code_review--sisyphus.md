---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-23T16:55:26Z
session_id: ses_ammalloc_tc_code_review
task_id: task_ammalloc_tc_003
module: ammalloc
submodule: thread_cache
slug: null
agent: sisyphus
status: active
bootstrap_ready: true
memory_status: not_needed
supersedes: 20260323T125544Z--ses_ammalloc_tc_20260323--sisyphus.md
closed_at: null
closed_reason: null
---

# SizeClass Code Review 文档契约修复

## 目标
根据 `docs/reviews/code_review/size_class_code_review2.md` 的 Code Review 结论，修复 SizeClass 模块的文档契约和静态校验问题。

## 当前状态

### 已完成
- ✅ `CalculateBatchSize` 参数语义：入参改为 `norm_size`，注释已说明是经过 `RoundUp()` 归一化后的值
- ✅ fragmentation 百分比描述：`< 12.5%` → `~12.5%-25% depending on kStepShift`（统一三处不一致）
- ✅ `Size()` inverse 描述：`exact inverse` → 明确说明是 left-inverse，非严格双射
- ✅ size 范围描述：统一为 `[1, 128] bytes`
- ✅ `RoundUp()` oversize 语义：明确 passthrough 行为（超出 MAX_TC_SIZE 原样返回）
- ✅ `Index()` size=0 语义：明确映射层兼容 0、策略层拒绝 0 的设计契约
- ✅ MAX_TC_SIZE 边界校验：添加 `static_assert(std::has_single_bit(MAX_TC_SIZE))`
- ✅ 静态校验点：添加 5 个编译期采样验证函数

### 未完成（原目标）
- ⏳ 动态水位线算法设计
- ⏳ 高并发场景下的性能测试
- ⏳ 边界条件处理（TC 容量极低时）

## 设计契约总结

### size=0 处理策略

| 接口类型 | `size=0` 行为 | 原因 |
|----------|---------------|------|
| 映射接口 | 返回最小 class (index=0, size=8) | 兼容 `malloc(0)` C 标准 |
| 策略接口 | 返回 0（无效） | batch/page 计算对零尺寸无意义 |

### RoundUp oversize 语义

- `size <= MAX_TC_SIZE`：返回最小满足条件的 size class
- `size > MAX_TC_SIZE`：原样返回（passthrough）
- 调用方需单独处理 oversize 分配（如直接页分配）

### MAX_TC_SIZE 约束

- 必须是 2 的次幂（`std::has_single_bit` 校验）
- 当 `kStepsPerGroup=4` 时，2 的次幂一定是 size class 边界

## 静态校验点

```cpp
// 5 个编译期验证函数（采样策略，避免 constexpr 步数限制）
ValidateIndexInRangeSampled()      // Index(s) 映射正确
ValidateSizeNotLessThanInputSampled()  // Size(Index(s)) >= s
ValidateIndexIdempotentSampled()   // Index(Size(Index(s))) == Index(s)
ValidateSizeMonotonic()            // Size(idx) 严格递增
ValidateRoundUpMonotonicSampled()  // RoundUp(s) 非递减
```

## 涉及文件
- `ammalloc/include/ammalloc/size_class.h`：主要修改文件
- `docs/reviews/code_review/size_class_code_review2.md`：Code Review 文档

## 验证方式
- 构建：`cmake --build build --target ammalloc -j`
- 单测：`./build/tests/unit/aethermind_unit_tests --gtest_filter="SizeClassTest.*:SizeClassInvalidInput.*"`
- 状态：✅ 全部通过（14/14）

## 推荐下一步
1. **继续动态水位线实现**（原目标）
   - 修改 `thread_cache.cpp:FetchRange()`，添加动态水位线逻辑
   - 添加 `CentralCache::GetTransferCacheCapacity()` 查询接口
   - 单测验证

## 风险评估
- 正确性：低风险（仅修改注释和静态校验）
- 并发：无影响
- 内存：无影响
- 性能：无影响（编译期校验）