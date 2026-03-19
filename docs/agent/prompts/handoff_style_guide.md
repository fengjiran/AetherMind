# Handoff 精简指南

> **用途**：定义精简 handoff 的最佳实践，确保启动加载量控制在合理范围。
> **目标**：单个 handoff ≤ 150 行或 ≤ 6,000 tokens。
> **更新时间**：2026-03-19

---

## 1. 核心原则

### Handoff 是什么
- **会话交接文档**：记录当前状态增量，不替代长期稳定事实
- **恢复起点**：新会话可快速接手，无需从零开始
- **生命周期有限**：工作完成后应关闭，不应永久保留

### Handoff 不是什么
- ❌ 详细设计文档 → 应写入 ADR
- ❌ 代码实现细节 → 应写入代码注释或 memory
- ❌ 历史决策记录 → 应写入 memory 或 ADR
- ❌ 测试用例清单 → 应写入测试文档

---

## 2. 必须包含的内容

### Frontmatter（必须）
```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-19T10:30:00Z
session_id: ses_xxx
task_id: task_xxx
module: <module>
submodule: <submodule> or null
slug: <slug> or null
agent: sisyphus
status: active
bootstrap_ready: false  # 仅当 handoff 足以支撑低上下文恢复时才写 true
memory_status: not_needed
supersedes: null
closed_at: null
closed_reason: null
---
```

### 正文结构（必须）

```markdown
# <任务标题>

## 目标
- 一句话说明任务目标

## 当前状态
- **阶段**：[设计 | 实施 | 测试 | 完成]
- **阻塞点**：[无 | 具体阻塞项]

## 已完成工作
- 简要列表，每项一行

## 涉及文件
- 文件路径列表，格式：`path/to/file` (修改类型)

## 未完成事项
- [ ] 任务项1
- [ ] 任务项2

## 下一步
- 具体可执行的操作
```

---

## 3. 禁止包含的内容

### ❌ 详细代码片段
**错误示例**：
```markdown
## 已完成工作
修改了 Span 结构体：
\`\`\`cpp
struct alignas(64) Span {
    Span* next{nullptr};
    Span* prev{nullptr};
    uint64_t start_page_idx{0};
    // ... 50+ 行代码
};
\`\`\`
```

**正确示例**：
```markdown
## 已完成工作
- 修改 `ammalloc/include/ammalloc/span.h`: Span 结构体 112B→64B
- 添加 GetBitmap()/GetDataBasePtr()/GetBitmapNum() 接口
- 详见 `ammalloc/include/ammalloc/span.h:68-90`
```

### ❌ 完整设计决策
**错误示例**：
```markdown
## 架构分析决策
1. HugePageCache 不加 CPUPause
   - 原因：测试显示 CPUPause 在该场景下无性能提升
   - 数据：benchmark 结果显示延迟无变化
   - 结论：不添加
2. In-band bitmap 性能中性
   - ...（详细分析）
```

**正确示例**：
```markdown
## 架构分析决策
- HugePageCache 不加 CPUPause → 详见 ADR-005
- In-band bitmap 性能中性 → 详见 ADR-006
- 拒绝 bitfield+union 方案 → 详见 ADR-007
```

### ❌ 完整文件清单+修改详情
**错误示例**：
```markdown
## 涉及文件

### 需修改的源文件

| 文件路径 | 变更类型 | 关键修改点 |
|----------|----------|-----------|
| `ammalloc/include/ammalloc/span.h` | 重写 | Span 结构体 112B→64B；添加 GetBitmap()/GetDataBasePtr()/GetBitmapNum()；flags 位域替代 is_used/is_committed |
| `ammalloc/src/span.cpp` | 修改 | Init(): bitmap→GetBitmap()；计算并设置 obj_offset；AllocObject()/FreeObject(): data_base_ptr→GetDataBasePtr() |
| ...（10+ 行表格）|
```

**正确示例**：
```markdown
## 涉及文件
- `ammalloc/include/ammalloc/span.h` (重写)
- `ammalloc/src/span.cpp` (修改)
- `ammalloc/src/central_cache.cpp` (修改)
- `ammalloc/src/page_cache.cpp` (修改)
- `ammalloc/src/page_heap_scavenger.cpp` (修改)
```

---

## 4. 精简技巧

### 技巧1：引用而非复制
- ✅ `详见 ADR-005`
- ✅ `详见 src/page_cache.cpp:120-150`
- ❌ 复制粘贴完整内容

### 技巧2：回写而非保留
- 已稳定的设计决策 → 回写到 memory 或 ADR
- 已验证的实现细节 → 回写到代码注释
- Handoff 只保留"未稳定"和"未完成"的部分

### 技巧3：状态而非历史
- ✅ "当前阶段：设计审核通过"
- ❌ "3月15日完成代码审查，3月16日完成测试..."

### 技巧4：具体而非模糊
- ✅ "阻塞点：等待用户批准 Span v2 方案"
- ❌ "阻塞点：需要讨论"

---

## 5. Token 预算指南

### 单个 Handoff 建议限制
- **行数**：≤ 150 行
- **Token数**：≤ 6,000 tokens（约 2,400 字符）

### 超出预算时的处理
1. **识别可回写内容**：设计决策、已验证事实
2. **识别可删除内容**：详细代码、完整历史
3. **识别可引用内容**：已有文档、代码文件

### 示例对比

**精简前（242 行，~11,000 tokens）**：
```markdown
## 已完成工作
1. 代码审查 (docs/reviews/code_review/20260315_page_cache_code_review.md)
   - 修复 P0 问题：PageMap::GetSpan 缺少 i0 边界检查
   - 问题详情：在 xxx 情况下会导致 yyy
   - 修复方案：添加 zzz 检查
   - 测试验证：添加了 aaa 测试用例
   ...（详细展开）
```

**精简后（80 行，~3,500 tokens）**：
```markdown
## 已完成工作
- 代码审查：修复 P0 问题（PageMap::GetSpan 边界检查）→ 详见 docs/reviews/code_review/20260315_page_cache_code_review.md
- 测试用例补充：13/13 通过
- 性能基准测试：benchmark_page_cache.cpp 已修复
```

---

## 6. 检查清单

创建 handoff 前，确认：

- [ ] Frontmatter 完整且正确
- [ ] 目标明确（一句话）
- [ ] 当前状态清晰（阶段 + 阻塞点）
- [ ] 已完成工作简洁（列表形式）
- [ ] 涉及文件列明路径和类型
- [ ] 未完成事项具体可执行
- [ ] 无详细代码片段
- [ ] 无完整设计决策（已稳定部分已回写）
- [ ] 行数 ≤ 150 行
- [ ] Token数 ≤ 6,000

---

## 7. 相关文档

- Handoff 模板：`docs/agent/prompts/handoff_template.md`
- Memory 规范：`docs/agent/memory/README.md`
- 启动契约：`AGENTS.md` 第10节