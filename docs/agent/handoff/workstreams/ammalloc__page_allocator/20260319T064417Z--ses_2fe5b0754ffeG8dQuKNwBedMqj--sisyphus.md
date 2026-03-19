---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-19T06:44:17Z
session_id: ses_2fe5b0754ffeG8dQuKNwBedMqj
task_id: ammalloc__page_allocator
module: ammalloc
submodule: page_allocator
slug: null
agent: sisyphus
status: superseded
bootstrap_ready: true
memory_status: not_needed
supersedes: 20260318T172004Z--ses_2fe5b0754ffeG8dQuKNwBedMqj--sisyphus.md
closed_at: null
closed_reason: null
---

# PageAllocator SystemFree Review 与大页策略演进 - Handoff

## 目标

基于 Code Review 对 `PageAllocator::SystemFree` 进行审查，并规划大页策略长期演进路线。

## 当前状态

### ✅ 已完成

#### 1. SystemFree 审查报告
- **P1-1**: madvise 失败后仍尝试缓存 → 需修复（失败时跳过缓存，直接 SafeMunmap）
- **P1-2**: madvise 失败未捕获 errno → 需修复
- **P2-1**: 缺少 `huge_cache_put_*` 统计 → 需新增
- **P2-2**: overflow guard 无统计 → 需新增
- **P3-1**: huge cache 资格判断脆弱（继承自 Code Review P1-5）→ 长期演进

#### 2. 修改方案设计（未实施）
详细修改方案已设计，见本次会话记录。核心修改：
- `madvise(MADV_DONTNEED)` 失败时跳过缓存
- 新增统计字段：`huge_cache_put_success_count`、`huge_cache_put_reject_count`、`system_free_overflow_count`

#### 3. Code Review P2-2 评估（乐观对齐命中率）
- **结论**：建议部分合理
- **推荐**：增加命中率统计（`huge_exact_align_hit_count`、`huge_exact_align_miss_count`）
- **不推荐**：直接改成 over-allocate + trim（破坏命中场景的价值）

#### 4. Code Review P2-3 评估（MAP_POPULATE 与 MADV_WILLNEED 拆分）
- **结论**：建议部分合理，优先级 P2
- **推荐**：当前保持共用配置，未来按需拆分

#### 5. MAP_HUGETLB 方案研究
- **THP (MADV_HUGEPAGE)**：零配置、灵活，但异步合并、延迟不可控 → **默认策略**
- **MAP_HUGETLB**：严格延迟保证，但需预留、不可换出、SIGBUS 风险 → **可选后端**
- **结论**：当前 THP 策略正确，MAP_HUGETLB 作为长期演进项

#### 6. TODO List 更新
新增 3 个演进条目到 `docs/designs/ammalloc/ammalloc_todo_list.md`：

| 条目 | 章节 | 优先级 |
|------|------|--------|
| Page 来源标记机制 | P3 架构演进 | P3 |
| 大页策略重构 | P3 架构演进 | P3 |
| 拆分 MAP_POPULATE 与 MADV_WILLNEED 配置 | P1 性能微调 | P2 |

#### 7. 头文件注释更新
`ammalloc/include/ammalloc/page_allocator.h` 的 `SystemFree` 接口新增文档注释，说明 HugePageCache 资格判断是临时启发式策略。

## 涉及文件

| 文件路径 | 变更类型 | 状态 |
|---------|---------|------|
| `docs/designs/ammalloc/ammalloc_todo_list.md` | 修改 | ✅ 已更新（新增 3 个演进条目） |
| `ammalloc/include/ammalloc/page_allocator.h` | 修改 | ✅ 已更新（SystemFree 注释） |
| `ammalloc/src/page_allocator.cpp` | 未修改 | 待实施 P1 修改方案 |

## 已确认结论

### SystemFree 设计契约
```cpp
// 当前行为（待优化）：
// 1. madvise 失败后仍尝试 Put()，可能缓存无效 VMA
// 2. 缺少 errno 捕获和细化统计

// 目标行为：
// 1. madvise 失败时跳过缓存，直接 SafeMunmap
// 2. 第一时间捕获 errno
// 3. 记录 huge_cache_put_success_count / reject_count
```

### 大页策略选择原则
- **默认**: THP (`MADV_HUGEPAGE`) - 零配置，适合通用场景
- **可选**: `MAP_HUGETLB` - 严格延迟保证，需系统配置，适合 HFT/DPDK

## 阻塞点

**无阻塞点**

SystemFree P1 修改方案已设计完成，可直接实施。

## 推荐下一步

### 优先级 1：实施 SystemFree P1 修改
修改 `ammalloc/src/page_allocator.cpp`：
1. madvise 失败时捕获 errno 并跳过缓存
2. 新增 `huge_cache_put_success_count`、`huge_cache_put_reject_count` 统计
3. overflow guard 添加 `system_free_overflow_count` 统计
4. 新增对应单元测试

### 优先级 2：增加对齐命中率统计（P2-2）
在 `AllocHugePage` 中增加 `huge_exact_align_hit_count` / `miss_count` 统计，为后续策略决策提供数据支撑。

### 验证命令
```bash
# 构建
cmake --build build --target ammalloc -j

# 单元测试
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.*:PageAllocatorThreadSafeTest.*

# 新增测试（实施后）
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageAllocatorTest.HugeCachePutStats
```

## 相关文档

- Code Review: `docs/reviews/code_review/PageAllocator_HugePageCache_Review.md`
- TODO List: `docs/designs/ammalloc/ammalloc_todo_list.md`
- Memory: `docs/agent/memory/modules/ammalloc/submodules/page_allocator.md`
- ADR-003: `docs/agent/memory/modules/ammalloc/adrs/ADR-003.md` (乐观大页策略)