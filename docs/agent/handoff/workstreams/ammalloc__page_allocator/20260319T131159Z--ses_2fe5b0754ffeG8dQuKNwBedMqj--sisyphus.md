---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-19T13:11:59Z
session_id: ses_2fe5b0754ffeG8dQuKNwBedMqj
task_id: ammalloc__page_allocator
module: ammalloc
submodule: page_allocator
slug: null
agent: sisyphus
status: active
bootstrap_ready: true
memory_status: not_needed
supersedes: 20260319T064417Z--ses_2fe5b0754ffeG8dQuKNwBedMqj--sisyphus.md
closed_at: null
closed_reason: null
---

# PageAllocator SystemFree 审查与 HugePageCache 审核 - Handoff

## 目标

1. 审查 `SystemFree` 实现并评估 Code Review 建议
2. 深入审核 `HugePageCache` 实现正确性
3. 规划大页策略长期演进路线

## 当前状态

### ✅ 已完成

#### 1. SystemFree 审查报告

| 问题 | 优先级 | 状态 |
|------|--------|------|
| madvise 失败后仍尝试缓存 | P1 | 待修复 |
| madvise 失败未捕获 errno | P1 | 待修复 |
| 缺少 `huge_cache_put_*` 统计 | P2 | 待实现 |
| overflow guard 无统计 | P2 | 待实现 |
| huge cache 资格判断脆弱 | P3 | 长期演进 |

**修改方案已设计**，见前序 handoff。

#### 2. Code Review 建议评估

**P2-2（乐观对齐命中率统计）**：
- ✅ 建议部分合理
- 推荐增加统计，不推荐直接改 over-allocate + trim

**P2-3（MAP_POPULATE 与 MADV_WILLNEED 拆分）**：
- ✅ 建议部分合理
- 优先级 P2，当前共用配置够用

#### 3. MAP_HUGETLB 研究

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| THP (MADV_HUGEPAGE) | 零配置、异步合并 | **默认策略** |
| MAP_HUGETLB | 严格延迟保证、需预留 | 可选后端（HFT/DPDK） |

**结论**：当前 THP 策略正确，MAP_HUGETLB 作为长期演进项。

#### 4. TODO List 更新

新增 3 个条目到 `docs/designs/ammalloc/ammalloc_todo_list.md`：

| 条目 | 章节 | 优先级 |
|------|------|--------|
| Page 来源标记机制 | P3 架构演进 | P3 |
| 大页策略重构 | P3 架构演进 | P3 |
| 拆分 MAP_POPULATE 与 MADV_WILLNEED 配置 | P1 性能微调 | P2 |

#### 5. HugePageCache 审核报告

| 方面 | 状态 | 说明 |
|------|------|------|
| 数据结构设计 | ✅ 正确 | 零分配、cache-line 对齐 |
| 双栈模型 | ✅ 正确 | 逻辑清晰 |
| ABA 防护 | ✅ 正确 | 48-bit tag 充足 |
| 发布-获取语义 | ✅ 正确 | Get 能读到 Put 的数据 |
| Pop 内存序 | ⚠️ 可优化 | `acquire` → `acq_rel` |
| errno 捕获 | ❌ 缺失 | ReleaseAllForTesting |
| debug 校验 | ❌ 缺失 | Put 参数校验 |

## 涉及文件

| 文件路径 | 变更类型 | 状态 |
|---------|---------|------|
| `docs/designs/ammalloc/ammalloc_todo_list.md` | 修改 | ✅ 已更新 |
| `ammalloc/include/ammalloc/page_allocator.h` | 修改 | ✅ 已更新（SystemFree 注释） |
| `ammalloc/src/page_allocator.cpp` | 未修改 | 待实施 |

## 已确认结论

### SystemFree 设计契约

```cpp
// 目标行为：
// 1. madvise 失败时捕获 errno 并跳过缓存，直接 SafeMunmap
// 2. 新增统计：huge_cache_put_success_count / reject_count
// 3. overflow guard 添加 system_free_overflow_count
```

### HugePageCache 内存序

```cpp
// Pop CAS 建议修改：
head.compare_exchange_weak(old_val, new_val,
                           std::memory_order_acq_rel,  // success: acquire → acq_rel
                           std::memory_order_acquire); // failure
```

### 大页策略选择

- **默认**：THP (`MADV_HUGEPAGE`) - 零配置
- **可选**：`MAP_HUGETLB` - 严格延迟保证

## 阻塞点

**无阻塞点**

## 推荐下一步

### 优先级 1：修复 HugePageCache P1 问题

修改 `ammalloc/src/page_allocator.cpp`：
```cpp
// 1. Pop CAS success 改为 acq_rel
if (head.compare_exchange_weak(old_val, new_val,
                               std::memory_order_acq_rel,  // 修改
                               std::memory_order_acquire)) {

// 2. ReleaseAllForTesting 保存 errno
if (munmap(ptr, ...) != 0) {
    const int err = errno;  // 新增
    ...
}

// 3. Put 添加 debug 校验
bool Put(void* ptr) noexcept {
    AM_DCHECK(ptr != nullptr);
    AM_DCHECK(IsHugePageAligned(ptr));
    ...
}
```

### 优先级 2：实施 SystemFree P1 修改

见前序 handoff 的修改方案。

### 验证命令

```bash
# 构建
cmake --build build --target ammalloc -j

# 单元测试
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.*:PageAllocatorThreadSafeTest.*

# TSAN 验证（并发正确性）
cmake -S . -B build-tsan -DENABLE_TSAN=ON
cmake --build build-tsan --target aethermind_unit_tests -j
./build-tsan/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorThreadSafeTest.*
```

## 相关文档

- Code Review: `docs/reviews/code_review/PageAllocator_HugePageCache_Review.md`
- TODO List: `docs/designs/ammalloc/ammalloc_todo_list.md`
- Memory: `docs/agent/memory/modules/ammalloc/submodules/page_allocator.md`
- ADR-003: 乐观大页策略
- ADR-006: HugePageCache 无锁架构