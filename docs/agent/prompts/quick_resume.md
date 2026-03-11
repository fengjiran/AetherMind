# 快速恢复工作流（Quick Resume）

> **默认入口**：一句话快速接续之前的工作  
> **详细入口**：需要显式填写目标/ADR/回写项时，使用 [`new_session_template.md`](./new_session_template.md)  
> **跨模块任务**：请直接使用 [`new_session_template.md`](./new_session_template.md)，当前 quick resume 只面向单一 workstream

---

## ⚠️ 强制约束（HARD CONSTRAINT）

**加载记忆后，必须等待用户明确确认，禁止自动执行任何操作。**

```
用户: "继续 ammalloc thread_cache 的工作"
    ↓
Agent: 加载记忆 (AGENTS.md → README.md → project.md → module.md → submodule.md → handoff)
    ↓
Agent: 输出 "记忆已加载，推荐下一步是..."
    ↓
⚠️ 检查点: 必须等待用户说"继续"、"执行"或"是"
    ↓
用户: "继续"  ← 必须显式确认
    ↓
Agent: 才能执行工具操作（扫描代码、编译、测试等）
```

**禁止行为（违反将导致错误）**：
- ❌ 加载记忆后立即扫描代码/编译/测试
- ❌ 假设"继续"等于自动执行
- ❌ 未经确认就修改文件

**正确流程**：
- ✅ 加载记忆 → 输出状态 → **显式询问用户** → 等待确认 → 执行操作

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
直接匹配 `docs/agent/memory/modules/<module>/` 目录：
```
用户: "继续 ammalloc thread_cache"
解析:
  - 模块: ammalloc
  - 子模块: thread_cache
  - 检查: docs/agent/memory/modules/ammalloc/module.md
  - 检查: docs/agent/memory/modules/ammalloc/submodules/thread_cache.md
  - 检查: docs/agent/handoff/workstreams/ammalloc__thread_cache/
```

### 2. 模糊匹配（仅模块级）
如果只提供模块名，未指定子模块：
```
用户: "继续 ammalloc"
解析:
  - 模块: ammalloc
  - 子模块: 无
  - 检查: docs/agent/memory/modules/ammalloc/module.md
   - 检查: docs/agent/handoff/workstreams/ammalloc__none/
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

按 `docs/agent/memory/README.md` 的规范读取：

```
1. AGENTS.md                          → 项目执行指南
2. docs/agent/memory/README.md              → 操作规范（必须先读，明确读取顺序）
3. docs/agent/memory/project.md             → 全局层
4. docs/agent/memory/modules/<module>/module.md        → 模块层
5. docs/agent/memory/modules/<module>/submodules/<submodule>.md  → 子模块层（如指定）
6. docs/agent/handoff/workstreams/<workstream_key>/    → 最新 active handoff（临时状态）
```

**Workstream 键与读取规则**：详见 `docs/agent/memory/README.md` "Handoff 存储规范"章节

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

## Resume Gate（执行任何操作前的必经检查）

在调用任何工具（扫描代码、编译、测试等）之前，必须完成以下检查并输出：

```markdown
## Resume Gate
- [x] 解析范围：模块=[module], 子模块=[submodule]
- [x] 命中文件：project.md ✅, module.md ✅, submodule.md ⚠️（如缺失）
- [x] 缺失项：[列表或"无"]
- [x] 选中 handoff：[文件名] ✅ 或 无 active handoff
- [x] resume_status: complete | partial | blocked
```

**resume_status 定义**：
- `complete`：所有文件加载成功，包括 handoff
- `partial`：部分文件缺失（如缺子模块或 handoff），但可按 module 层恢复
- `blocked`：模块不存在，需用户决策（创建新模块/指定其他模块）

**规则**：
- ⚠️ resume_status 为 `blocked` 时，禁止执行任何工具操作
- ⚠️ 未完成 Resume Gate 输出，禁止执行任何工具操作

Agent 确认上下文已加载后，输出：

```markdown
## 已解析范围
- 模块：`[module]`
- 子模块：`[submodule | 无]`
- Workstream：`[module]__[submodule-or-none]`

## 已加载文件
- ✅ `docs/agent/memory/project.md`
- ✅ `docs/agent/memory/modules/[module]/module.md`
- ✅ `docs/agent/memory/modules/[module]/submodules/[submodule].md`（如适用）
- ✅ `docs/agent/handoff/workstreams/[key]/[latest].md`（如存在）

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

1. **不跳过检查**：简化的是交互，不是校验。`AGENTS.md` 和 `docs/agent/memory/README.md` 仍必须读取。
2. **不模糊匹配到多文件**：`继续 ammalloc` 只加载模块级，不自动扫所有子模块。
3. **输出必须明确**：必须写出"实际加载了哪些文件"，避免用户误以为上下文已完整恢复。
4. **handoff 不是必须**：如果 `docs/agent/handoff/` 目录为空，说明没有未完成的临时状态，从 memory 重新开始即可。
5. **加载后需确认**：使用 quick_resume.md 时，Agent 完成记忆加载后，**必须等待用户确认**是否执行下一步操作，**禁止**在加载记忆后立即自动执行任何工具操作（如扫描代码、编译、测试等）。只有在用户明确说"继续"、"执行"或类似指令后，才能执行操作。

---

## 边界情况处理

### 1. 模块存在但子模块 memory 不存在

**场景**：用户说"继续 ammalloc size_class"，但 `docs/agent/memory/modules/ammalloc/submodules/size_class.md` 不存在。

**Agent 处理**：
```markdown
## 加载结果
- ✅ docs/agent/memory/modules/ammalloc/module.md（已加载）
- ❌ docs/agent/memory/modules/ammalloc/submodules/size_class.md（不存在）
- resume_status: partial

## 决策选项
1. **创建子模块 memory** - 首次开发 size_class，需要创建 submodule.md
2. **仅按模块层继续** - 从 module.md 的"子模块划分"推断信息，继续工作
3. **检查拼写** - 确认子模块名是否正确

默认选项：2（仅按模块层继续，同时提示用户考虑创建子模块 memory）
```

**处理规则**：
- 加载 module.md 中的相关信息（子模块划分、待办事项）
- 标记 resume_status: partial（非完整恢复）
- 不阻塞工作，但提示用户子模块 memory 缺失

### 2. 无 handoff（从 memory 重新开始）

**场景**：workstream 目录为空或没有 `status: active` 的 handoff。

**Agent 处理**：
- 从 module.md/submodule.md 的"待办事项"开始
- 输出："无可恢复临时状态，从 memory 的待办事项开始"
- resume_status: partial

### 3. 多个 active handoff（异常）

**场景**：同一 workstream 存在多个 `status: active` 的 handoff（如 git merge 导致）。

**Agent 处理**：
- 选择 `created_at` 最新的一个
- ⚠️ 显式警告："发现多个 active handoff，选择最新一个，请检查并收敛状态"
- 建议用户将旧的标记为 `superseded`

---

## 失败回退

如果解析失败或歧义：
1. 列出所有可能的匹配项
2. 询问用户明确指定
3. 提供备选："或使用 new_session_template.md 显式启动"

### 模块不存在时的智能回退

如果解析出的模块 `docs/agent/memory/modules/<module>/` 不存在：

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
- 自动创建 docs/agent/memory/modules/[module]/module.md

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
