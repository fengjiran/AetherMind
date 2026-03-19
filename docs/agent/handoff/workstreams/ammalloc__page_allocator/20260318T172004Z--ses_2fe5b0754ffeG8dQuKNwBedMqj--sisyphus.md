---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-18T17:20:04Z
session_id: ses_2fe5b0754ffeG8dQuKNwBedMqj
task_id: ammalloc__page_allocator
module: ammalloc
submodule: page_allocator
slug: null
agent: sisyphus
status: superseded
memory_status: pending
supersedes: 20260318T090000Z--ses_3001f7ddcffe764XhsH3T9LQqc--sisyphus.md
closed_at: null
closed_reason: null
---

# PageAllocator AllocHugePageWithTrim Robustness Hardening - Handoff

## 目标

基于 code review 结论，对 `PageAllocator::AllocHugePageWithTrim` 进行稳健性收口：
- P0: Trim 失败一致性（避免返回部分 trim 后的"半残映射"）
- P1: 失败语义统一（返回 `nullptr` 而非 `MAP_FAILED`）
- P1: syscall errno 纪律化（第一时间捕获 errno）
- P2: 失败路径日志降噪与 cleanup 严谨化

## 当前状态

**✅ 已完成**

### P0-1: Trim 失败一致性修复
- `AllocHugePageWithTrim()` 现在使用 `head_mapped`/`body_mapped`/`tail_mapped` 状态跟踪
- 任一 trim 失败时返回 `nullptr`，不再返回中段指针
- 失败时增加 `huge_alloc_failed_count`，触发上层 fallback 路径
- Cleanup  helper `cleanup_remaining_mappings()` 只清理已知仍映射的子区间，避免重复 munmap 重试

### P1-1: 失败语义统一
- `AllocWithRetry()` 失败统一返回 `nullptr`（原返回 `MAP_FAILED`）
- 所有调用点从 `ptr == MAP_FAILED` 改为 `if (!ptr)`

### P1-2: errno 纪律化
- `AllocWithRetry()` 中 `const int err = errno` 第一时间捕获
- `SafeMunmap()` 中 `const int err = errno` 捕获后用于日志
- 避免后续调用污染 errno 的二次读取问题

### P2: Cleanup 严谨化（关键修复）
- 移除了 `cleanup_remaining_mappings()` 中的整块 `SafeMunmap(raw_ptr, alloc_size)` 重试
- 现在严格按分段清理：body → tail → head，配合 `skip_head`/`skip_tail` 避免重试已失败区间
- 若仍有残留映射，仅记录错误日志（`cleanup incomplete`），不再尝试不严谨的整块回收

### 其他修复
- **统计一致性**: overflow 保护分支现在会累加 `huge_alloc_failed_count`
- **文档修正**: `page_allocator.h` 中 HugePageCache 线程安全描述从 "mutex" 改为 "lock-free"
- **代码清理**: 删除了遗留的注释掉旧逻辑

## 涉及文件

| 文件路径 | 变更类型 | 状态 |
|---------|---------|------|
| `ammalloc/src/page_allocator.cpp` | 修改 | ✅ 已更新（trim 失败处理、cleanup 严谨化、统计修复） |
| `ammalloc/include/ammalloc/page_allocator.h` | 修改 | ✅ 已更新（文档修正） |

## 关键修复点

### Cleanup 严谨化（P0-1 核心）
```cpp
// cleanup_remaining_mappings 现在只执行分段清理，不整块重试
auto cleanup_remaining_mappings = [&](bool skip_head, bool skip_tail) {
    if (body_mapped && SafeMunmap(body_ptr, size)) {
        body_mapped = false;
    }
    if (!skip_tail && tail_mapped && SafeMunmap(tail_ptr, tail_gap)) {
        tail_mapped = false;
    }
    if (!skip_head && head_mapped && SafeMunmap(head_ptr, head_gap)) {
        head_mapped = false;
    }
    // 仅记录残留状态，不再 SafeMunmap(raw_ptr, alloc_size)
    if (head_mapped || body_mapped || tail_mapped) {
        spdlog::error("AllocHugePageWithTrim cleanup incomplete: ...");
    }
};
```

### Overflow 统计修复
```cpp
if (size > (std::numeric_limits<size_t>::max() - SystemConfig::HUGE_PAGE_SIZE)) {
    stats_.huge_alloc_failed_count.fetch_add(1, std::memory_order_relaxed);  // 新增
    spdlog::error("AllocHugePageWithTrim size overflow: {}", size);
    return nullptr;
}
```

## 验证结果

### LSP Diagnostics
- `ammalloc/src/page_allocator.cpp`: 无诊断问题
- `ammalloc/include/ammalloc/page_allocator.h`: 无诊断问题

### 构建验证
```bash
cmake --build build --target ammalloc -j
# 结果：无 warnings，无 errors
```

### 单元测试（15/15 通过）
```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.*:PageAllocatorThreadSafeTest.*
# 结果：15/15 通过
```

## 已确认接口与不变量

### AllocHugePageWithTrim 契约（更新后）
```cpp
static void* AllocHugePageWithTrim(size_t size);
// Returns: 
//   - 成功返回 huge-page-aligned ptr
//   - trim 任一失败返回 nullptr（上层 fallback 到 normal page）
// Preconditions: size > 0 && page_aligned(size)
// Cleanup: 只清理已知仍映射的子区间，不整块重试
```

## 阻塞点

**无阻塞点**

当前实现的风险：
- 若 cleanup 中的 `SafeMunmap` 调用失败，仍可能残留 VMA 映射
- 但这已通过错误日志暴露（`cleanup incomplete`），不影响"不给调用方返回坏指针"的 P0 目标

## 推荐下一步

### 可选：新增统计字段（P1-3）
可考虑拆分失败统计语义，但优先级较低：
- `huge_trim_fail_count`: head/tail trim 失败次数
- `huge_trim_cleanup_fail_count`: cleanup 阶段 munmap 失败次数
- 区分 `huge_alloc_failed_count` 的来源（mmap fail vs trim fail）

### 可选：失败注入测试（P2）
增加 trim-failure 注入测试：
- `TrimFail_HeadFail_CleanupRemaining`
- `TrimFail_TailFail_CleanupRemaining`
- `TrimFail_CleanupFail_Logged`

但这需要设计 munmap 失败注入机制（如 mock 或预置失败点）。

## 关键提交

- 当前工作区包含完整修复，建议提交信息：
  ```
  fix(ammalloc): harden AllocHugePageWithTrim cleanup rigor
  
  - Remove whole-range munmap retry from cleanup_remaining_mappings
  - Track head/body/tail mapped state explicitly
  - Add overflow branch failure counting
  - Fix HugePageCache documentation (lock-free, not mutex)
  ```

## 设计文档参考

- Review 原文：`docs/reviews/code_review/PageAllocator_HugePageCache_Review.md`
- ADR-003：`docs/agent/memory/modules/ammalloc/adrs/ADR-003.md` (乐观大页策略)
- ADR-006：`docs/agent/memory/modules/ammalloc/adrs/ADR-006.md` (HugePageCache 无锁架构)
