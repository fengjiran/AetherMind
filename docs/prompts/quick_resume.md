# 快速恢复工作流（Quick Resume）

> **默认入口**：一句话快速接续之前的工作  
> **详细入口**：需要显式填写目标/ADR/回写项时，使用 [`new_session_template.md`](./new_session_template.md)

---

## 触发语句

Agent 应识别以下自然语言意图：

```
继续/接续/恢复/加载 ... [模块名] [子模块名] ... 的完整记忆和最新 handoff
```

**示例：**
- "继续 ammalloc thread_cache 的工作"
- "加载 tensor 的完整记忆和最新 handoff"
- "恢复昨天的工作，模块 ammalloc"
- "读取 central_cache 的 handoff，继续开发"

---

## 解析规则（精确优先，模糊最少）

### 1. 精确匹配（优先）
直接匹配 `docs/memory/modules/<module>/` 目录：
```
用户: "继续 ammalloc thread_cache"
解析:
  - 模块: ammalloc
  - 子模块: thread_cache
  - 检查: docs/memory/modules/ammalloc/module.md
  - 检查: docs/memory/modules/ammalloc/submodules/thread_cache.md
  - 检查: docs/handoff/workstreams/ammalloc__thread_cache/
```

### 2. 模糊匹配（仅模块级）
如果只提供模块名，未指定子模块：
```
用户: "继续 ammalloc"
解析:
  - 模块: ammalloc
  - 子模块: 无
  - 检查: docs/memory/modules/ammalloc/module.md
  - 检查: docs/handoff/workstreams/ammalloc__/ 或 ammalloc__none/
```

### 3. 歧义处理
如果模块名/子模块名无法唯一命中：
```
用户: "继续 cache 的工作"
歧义:
  - 可能: ammalloc/central_cache
  - 可能: ammalloc/page_cache
  - 可能: tensor/cache_manager

Agent 响应:
  "找到多个匹配：
   1. ammalloc/central_cache
   2. ammalloc/page_cache
   3. tensor/cache_manager
   请明确指定模块和子模块。"
```

---

## 加载顺序（按此固定顺序）

按 `docs/memory/README.md` 的规范读取：

```
1. AGENTS.md                          → 项目执行指南
2. docs/memory/README.md              → 操作规范（必须先读，明确读取顺序）
3. docs/memory/project.md             → 全局层
4. docs/memory/modules/<module>/module.md        → 模块层
5. docs/memory/modules/<module>/submodules/<submodule>.md  → 子模块层（如指定）
6. docs/handoff/workstreams/<workstream_key>/    → 最新 active handoff（临时状态）
```

**Workstream 键规则**（与 `docs/memory/README.md` 一致）：
- **On-disk 键（目录名）**：`<module>__<submodule-or-none>`（例如：`ammalloc__thread_cache`）
- **逻辑键（frontmatter）**：`task_id` 作为元数据记录，不用于目录

**Handoff 状态过滤**：
- **只读取** `status: active` 的 handoff
- 忽略 `superseded` 和 `closed`（保留用于审计，但不用于恢复）
- 如果没有 `active` handoff，视为"无可恢复临时状态"，直接从 memory 开始

---

## 重要约束提醒

⚠️ **必须先加载记忆，后执行操作** —— 详见 `AGENTS.md` 第10节。

**快速检查清单**：
- ✅ 已按顺序加载：AGENTS.md → README.md → project.md → module.md → submodule.md → handoff
- ✅ 已输出"已加载文件"清单
- ✅ 根据 memory/handoff 的"推荐下一步"执行操作

**禁止**：先扫描代码/编译/测试，再加载记忆。

---

## 精简输出格式（5要点）

Agent 确认上下文已加载后，输出：

```markdown
## 已解析范围
- 模块：`[module]`
- 子模块：`[submodule | 无]`
- Workstream：`[module]__[submodule-or-none]`

## 已加载文件
- ✅ `docs/memory/project.md`
- ✅ `docs/memory/modules/[module]/module.md`
- ✅ `docs/memory/modules/[module]/submodules/[submodule].md`（如适用）
- ✅ `docs/handoff/workstreams/[key]/[latest].md`（如存在）

## 当前接续目标
[从 handoff 读取的目标，或从 memory 推断的默认目标]

## 下一步动作
- [具体的下一步操作]

## 阻塞点（如有）
- [从 handoff 读取的阻塞点，或 `无`]
```

---

## 与详细模式的关系

| 场景 | 推荐入口 | 原因 |
|------|----------|------|
| 日常接续昨天工作 | **quick_resume.md**（本文件） | 一句话启动，快速恢复 |
| 首次建档新模块 | new_session_template.md | 需要明确填写目标和约束 |
| 跨模块协调任务 | new_session_template.md | 需要显式指定多个模块 |
| 需要标记 ADR/回写 | new_session_template.md | 需要预填元数据 |
| 对话上下文已丢失 | quick_resume.md | 从 git 同步的 handoff 恢复 |

---

## 边界约束

1. **不跳过检查**：简化的是交互，不是校验。`AGENTS.md` 和 `docs/memory/README.md` 仍必须读取。
2. **不模糊匹配到多文件**：`继续 ammalloc` 只加载模块级，不自动扫所有子模块。
3. **输出必须明确**：必须写出"实际加载了哪些文件"，避免用户误以为上下文已完整恢复。
4. **handoff 不是必须**：如果 `docs/handoff/` 目录为空，说明没有未完成的临时状态，从 memory 重新开始即可。

---

## 失败回退

如果解析失败或歧义：
1. 列出所有可能的匹配项
2. 询问用户明确指定
3. 提供备选："或使用 new_session_template.md 显式启动"

### 模块不存在时的智能回退

如果解析出的模块 `docs/memory/modules/<module>/` 不存在：

**Agent 响应：**
```markdown
模块 `[module]` 不存在。

检测到的可能意图：
1. **创建新模块** - 首次开发 `[module]`，需要创建 module.md
2. **指定现有模块** - 可能是拼写错误，想指定其他已有模块
3. **检查拼写** - 确认模块名是否正确

请选择（输入 1/2/3）：

---
如果选择 **1. 创建新模块**：
- Agent 将切换到 new_session_template.md 流程
- 提示填写：模块名、目标、初始约束等
- 自动创建 docs/memory/modules/[module]/module.md

如果选择 **2. 指定现有模块**：
- Agent 列出所有已有模块：
  - ammalloc
  - tensor
  - runtime
- 请明确指定

如果选择 **3. 检查拼写**：
- 重新输入正确的模块名
```

**默认行为：**
- 如果用户说"开始...开发"（含"开始"关键词）→ 默认倾向选项 1（创建新模块）
- 如果用户说"继续..." → 默认倾向选项 2（指定现有模块）
- 无论哪种，都先询问确认，不自动执行
