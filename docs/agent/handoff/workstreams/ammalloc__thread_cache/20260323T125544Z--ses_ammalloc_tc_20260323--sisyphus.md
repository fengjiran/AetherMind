---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-23T12:55:44Z
session_id: ses_ammalloc_tc_20260323
task_id: task_ammalloc_tc_002
module: ammalloc
submodule: thread_cache
slug: null
agent: sisyphus
status: superseded
bootstrap_ready: true
memory_status: not_needed
supersedes: 20260311T103000Z--ses_example_001--sisyphus.md
closed_at: null
closed_reason: null
---

# ThreadCache 动态水位线实现 - Size=0 语义分析

## 目标
1. 实现 ThreadCache::FetchRange() 的动态水位线调节机制
2. （本会话）分析 am_malloc(0) 行为和 SizeClass size=0 语义冲突

## 当前状态

### 已完成
- ✅ am_malloc(0) 行为分析：返回有效指针（8 字节块），非 UB
- ✅ SizeClass size=0 语义冲突确认：Code Review `docs/reviews/code_review/size_class_code_review2.md` 指出的问题确实存在
- ✅ 调用链分析：所有生产代码路径通过 `RoundUp()` 归一化，不会触发 `CalculateBatchSize(0)`

### 未完成
- ⏳ 动态水位线算法设计
- ⏳ 高并发场景下的性能测试
- ⏳ 边界条件处理（TC 容量极低时）

### 语义冲突详情

| 方法 | size=0 返回 | 语义 |
|------|-------------|------|
| `Index(0)` | `0` | 最小 class |
| `RoundUp(0)` | `8` | 最小 class |
| `CalculateBatchSize(0)` | `0` | 非法输入 |
| `GetMovePageNum(0)` | `8` | 混乱（32KB 最小保护） |

**结论**：运行时安全（调用链已归一化），但设计语义不一致。

## 涉及文件
- `ammalloc/src/ammalloc.cpp`：am_malloc 主入口
- `ammalloc/include/ammalloc/size_class.h`：SizeClass 映射
- `ammalloc/include/ammalloc/thread_cache.h`：ThreadCache 快路径
- `ammalloc/src/thread_cache.cpp`：ThreadCache 慢路径
- `tests/unit/test_thread_cache.cpp:74-77`：size=0 测试用例
- `docs/reviews/code_review/size_class_code_review2.md`：Code Review 文档

## 已确认接口与不变量
- `ThreadCache::Allocate(0)` 返回有效 8 字节指针
- 所有 `CalculateBatchSize` 调用点传入的是 `RoundUp()` 后的值，不会为 0
- Code Review 建议：统一 size=0 语义（方案 A：视为非法输入）

## 阻塞点
- Size=0 语义冲突是否需要立即修复？（当前运行时安全，可记录为技术债务）

## 推荐下一步
1. **优先**：继续动态水位线实现（原目标）
   - 修改 `thread_cache.cpp:FetchRange()`，添加动态水位线逻辑
   - 添加 `CentralCache::GetTransferCacheCapacity()` 查询接口
   - 单测验证：`./aethermind_unit_tests --gtest_filter=ThreadCache.DynamicWatermark`

2. **可选**：修复 size=0 语义冲突
   - 在 `GetMovePageNum` 中对 size=0 返回 0（明确非法）
   - 更新文档契约，说明 size=0 由上层 `am_malloc` 处理

## 验证方式
- 构建：`cmake --build build --target aethermind_unit_tests -j`
- 单测：`./build/tests/unit/aethermind_unit_tests --gtest_filter=ThreadCache.*`
- 状态：未执行（本会话仅分析）