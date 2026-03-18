---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-18T09:00:00Z
session_id: ses_3001f7ddcffe764XhsH3T9LQqc
task_id: ammalloc__page_allocator
module: ammalloc
submodule: page_allocator
slug: null
agent: sisyphus
status: superseded
memory_status: pending
supersedes: 20260314T222700Z--ses_318bd5c17ffeP3CBYxVYXTJuAj--sisyphus.md
closed_at: null
closed_reason: null
---

# PageAllocator Robustness Hardening - Handoff

## 目标

基于 `docs/reviews/code_review/PageAllocator_HugePageCache_Review.md` 的审查结论，对 `PageAllocator` 和 `HugePageCache` 进行稳健性收口，重点修复：
- P0: Trim 失败一致性（避免返回部分 trim 后的"半残映射"）
- P1: 失败语义统一（`AllocWithRetry` 返回 `nullptr` 而非 `MAP_FAILED`）
- P1: syscall errno 纪律化（第一时间捕获 errno）
- P2: 失败路径日志降噪与 backoff 优化

## 当前状态

**✅ 已完成**

### P0-1: Trim 失败一致性修复
- `AllocHugePageWithTrim()` 现在跟踪 `head_trimmed` 和 `tail_trimmed` 状态
- 任一 trim 失败时返回 `nullptr`，不再返回中段指针
- 失败时增加 `huge_alloc_failed_count`，触发上层 fallback 路径

### P1-1: 失败语义统一
- `AllocWithRetry()` 失败统一返回 `nullptr`（原返回 `MAP_FAILED`）
- 所有调用点从 `ptr == MAP_FAILED` 改为 `if (!ptr)`

### P1-2: errno 纪律化
- `AllocWithRetry()` 中 `const int err = errno` 第一时间捕获
- `SafeMunmap()` 中 `const int err = errno` 捕获后用于日志
- 避免后续调用污染 errno 的二次读取问题

### P2: 日志与 backoff 优化（附带）
- ENOMEM retry 日志从 warn 降级（移除）
- backoff 从 1ms 改为 50us
- `ApplyHugePageHint()` 中 MADV_HUGEPAGE 失败不再打 debug 日志
- MADV_WILLNEED 失败从 warn 降级为 debug

### 增强的 DCHECK
- `IsPageAligned()` / `IsHugePageAligned()` 辅助函数
- `AllocWithRetry()` 输入校验（size > 0, page aligned）
- `SafeMunmap()` 输入校验（ptr page aligned, size page aligned）
- `ApplyHugePageHint()` 输入校验
- `AllocNormalPage()` / `AllocHugePageWithTrim()` 输入校验

**🔄 进行中 / 待完成**

### P0-1 残余：精细回收方案
已设计但未实现：
- `CleanupTrimFailure()` helper（按状态选择非重叠回收区间）
- 新增统计字段：`huge_trim_fail_count`, `huge_trim_cleanup_fail_count`
- 4 个失败注入测试用例

## 涉及文件

| 文件路径 | 变更类型 | 状态 |
|---------|---------|------|
| `ammalloc/src/page_allocator.cpp` | 修改 | ✅ 已更新（P0-1, P1-1, P1-2, P2, DCHECK） |
| `ammalloc/include/ammalloc/page_allocator.h` | 未修改 | 待添加统计字段声明 |
| `tests/unit/test_page_allocator.cpp` | 未修改 | 待添加 trim 失败注入测试 |

## 已确认接口与不变量

### AllocWithRetry 契约
```cpp
static void* AllocWithRetry(size_t size, int flags);
// Returns: 成功返回 page-aligned ptr，失败返回 nullptr
// Preconditions: size > 0 && page_aligned(size)
```

### AllocHugePageWithTrim 契约（更新后）
```cpp
static void* AllocHugePageWithTrim(size_t size);
// Returns: 
//   - 成功返回 huge-page-aligned ptr
//   - trim 任一失败返回 nullptr（上层 fallback 到 normal page）
// Preconditions: size > 0 && page_aligned(size)
```

### SafeMunmap 契约（更新后）
```cpp
static bool SafeMunmap(void* ptr, size_t size);
// Returns: munmap 成功返回 true，失败返回 false
// Post-failure: errno 已捕获，计数器已更新，低噪日志已输出
// Preconditions: ptr page_aligned, size page_aligned
```

## 阻塞点

**无阻塞点，但有残余风险**

当前实现的风险：
- trim 失败后尝试 `SafeMunmap(raw_ptr, alloc_size)` 回收整块，但如果 head/tail 已部分 unmap，这次调用可能返回 `EINVAL`
- 这会导致残留 VMA（未完全回收），但不影响"不给调用方返回坏指针"的 P0 目标

建议下一步实现精细回收方案消除此残余风险。

## 推荐下一步

### 立即执行（P0-1 收尾）

1. **实现 `CleanupTrimFailure()` helper**
   - 位置：`ammalloc/src/page_allocator.cpp`，作为 `PageAllocator` 的 private static 方法
   - 逻辑：根据 `head_trimmed`/`tail_trimmed` 状态选择非重叠回收区间
   - 验收：无 diagnostics 错误，构建通过

2. **更新 `AllocHugePageWithTrim()` 调用 `CleanupTrimFailure()`**
   - 替换当前的 `SafeMunmap(raw_ptr, alloc_size)` 简单回收
   - 验收：逻辑与伪代码流程图一致

3. **添加统计字段**
   - `ammalloc/include/ammalloc/page_allocator.h`: `PageAllocatorStats` 新增 `huge_trim_fail_count`, `huge_trim_cleanup_fail_count`
   - `ammalloc/src/page_allocator.cpp`: `ResetStats()` 初始化新字段
   - `AllocHugePageWithTrim()`: 失败时更新新字段
   - 验收：编译通过，新计数器可被测试读取

4. **实现失败注入测试**
   - `tests/unit/test_page_allocator.cpp`: 添加 4 个测试用例
     - `TrimFail_HeadFail_TailOk_CleanupRemaining`
     - `TrimFail_HeadOk_TailFail_CleanupRemaining`
     - `TrimFail_BothFail_FullCleanupAttempt`
     - `TrimFail_CleanupFail_StillReturnNullAndCounted`
   - 需要设计 munmap 失败注入机制（如 mock 或预置失败点）
   - 验收：4 个测试通过

### 验证命令

```bash
# 构建
cmake --build build --target ammalloc -j
cmake --build build --target aethermind_unit_tests -j

# 运行所有 PageAllocator 测试
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.*:PageAllocatorThreadSafeTest.*

# 运行新增测试（假设命名）
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.TrimFail*
```

### 后续可选（P1-3 统计拆分）

如果本轮还有余力，可考虑：
- 拆分 `huge_alloc_failed_count` 语义（exact mmap fail vs trim fail）
- 新增 `huge_cache_put_reject_count`, `huge_cache_get_miss_count`
- 区分 `madvise_dontneed_fail_count` 与 hint 失败

但这属于 P1-3，优先级低于 P0-1 收尾。

## 验证方式（已完成项）

### 单元测试（15/15 通过）
```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.*:PageAllocatorThreadSafeTest.*
# 结果：15/15 通过
```

### 构建验证
```bash
cmake --build build --target ammalloc -j
cmake --build build --target aethermind_unit_tests -j
# 结果：无 warnings，无 errors
```

### LSP Diagnostics
- `ammalloc/src/page_allocator.cpp`: 无诊断问题

## 关键提交

- 当前工作区包含 P0-1/P1-1/P1-2/P2 改动，尚未提交
- 建议提交信息：`fix(ammalloc): harden PageAllocator trim failure handling and unify error semantics`

## 设计文档参考

- 精细回收方案伪代码：`docs/agent/handoff/workstreams/ammalloc__page_allocator/` 当前 handoff 正文
- Review 原文：`docs/reviews/code_review/PageAllocator_HugePageCache_Review.md`
- ADR-006：`docs/agent/memory/modules/ammalloc/adrs/ADR-006.md` (HugePageCache 无锁架构)
