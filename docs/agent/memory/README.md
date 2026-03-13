# 记忆系统操作规范

> **本文档为记忆系统的操作规范**。  
> 架构设计说明见: [`docs/agent/memory_system.md`](../memory_system.md)
>
> 用途：定义 `docs/agent/memory/` 的唯一结构规则。稳定事实写入记忆文档；会话级临时状态保留在 handoff。

## 本文档范围
- 本文档是 `docs/agent/memory/` 的操作规范，用于实际创建、读取、更新和维护 memory 文件；不承担架构总览说明。
- 架构设计说明：[`docs/agent/memory_system.md`](../memory_system.md)
- Prompt 入口与衔接：[`docs/agent/prompts/README.md`](../prompts/README.md)、[`docs/agent/prompts/new_session_template.md`](../prompts/new_session_template.md)
- 需要理解分层模型、工作流全貌和示例时，优先阅读架构文档；需要执行路径、元数据、命名和冲突规则时，优先阅读本文档。
- 实际使用或回写 memory 时，以本文档为最终操作依据。

## 记忆层级模型
- 全局层：`docs/agent/memory/project.md`。记录跨模块稳定事实、目录约定和默认规则。
- 模块层：`docs/agent/memory/modules/<module>/module.md`。记录主模块职责、边界、接口、不变量和长期约束。
- 子模块层：`docs/agent/memory/modules/<module>/submodules/<submodule>.md`。只在父模块内存在独立边界时拆分；与主模块同属领域层。
- handoff 层：按 `docs/agent/prompts/handoff_template.md` 输出当前会话摘要，只服务交接，不替代稳定记忆；输出存储在 `docs/agent/handoff/` 目录，通过 git 同步实现跨机器恢复。
- 读取顺序：`AGENTS.md` -> `docs/agent/memory/README.md` -> `docs/agent/memory/project.md` -> `docs/agent/memory/modules/<module>/module.md`（模块工作时）-> `docs/agent/memory/modules/<module>/submodules/<submodule>.md`（子模块工作时） -> `docs/agent/handoff/workstreams/<workstream_key>/`（从任务系统或实际存储目录）。
  - **模块工作**：加载 `docs/agent/memory/modules/<module>/module.md` -> `docs/agent/memory/modules/<module>/submodules/<submodule>.md`（如存在）
  - **项目级工作**：跳过 `docs/agent/memory/modules/<module>/module.md` 和 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`，直接加载 `docs/agent/memory/project.md` -> `docs/agent/handoff/workstreams/project__<slug>/`
- 冲突优先级：用户显式指令与已验证代码/测试事实 > `AGENTS.md` > `docs/products/aethermind_prd.md` > ADR > 模块/子模块记忆 > `docs/agent/memory/project.md` > handoff。
- handoff 与稳定记忆冲突时，先回到代码、测试或用户指令验证；未验证前不要直接覆盖 memory 文件。

## Handoff 存储规范

### 存储位置
- **路径**：`docs/agent/handoff/workstreams/<workstream_key>/`
- **位置**：`docs/agent/handoff/` 目录在 git 管理下，随仓库同步
- **目的**：
  - 本地 fallback（任务系统不可用时）
  - 跨机器恢复（通过 git pull/push 同步）
- **当前限制**：handoff 按 workstream 单独存储；跨模块协调暂不支持通过单个 handoff 统一恢复，需改用 `docs/agent/prompts/new_session_template.md` 明确启动。

### Workstream 键

#### 类型 1：模块工作（标准）
**On-disk 键（目录名）**：`<module>__<submodule-or-none>`
- 格式：`ammalloc__thread_cache` 或 `ammalloc__none`（无子模块时）
- 作用：确定 handoff 文件存储的目录路径
- 加载顺序：`docs/agent/memory/project.md` → `docs/agent/memory/modules/<module>/module.md` → `docs/agent/memory/modules/<module>/submodules/<submodule>.md` → `docs/agent/handoff/workstreams/<module>__<submodule>/`

**示例**：
- `ammalloc__thread_cache` → ammalloc 模块的 thread_cache 子模块
- `ammalloc__none` → ammalloc 模块（无特定子模块）

#### 类型 2：项目级工作（新增）
**On-disk 键（目录名）**：`project__<slug>`
- 格式：`project__docs-reorg`、`project__ci-setup`
- 作用：处理不归属特定模块的仓库级工作
- 加载顺序：`docs/agent/memory/project.md` → `docs/agent/handoff/workstreams/project__<slug>/`（跳过 `docs/agent/memory/modules/<module>/module.md` 和 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`）

**何时使用**：
- ✅ 仓库级、跨模块的工作（如目录重构、CI 配置）
- ✅ 多步骤、有明确起止的任务
- ✅ 没有自然模块归属者

**何时不用**：
- ❌ 已有明确模块归属（即使涉及配置文件）
- ❌ 已 settled 的事实（直接写 project.md）
- ❌ 一次性简单任务

**示例**：
- `project__docs-reorg` → docs/ 目录结构重构
- `project__ci-setup` → CI/CD 配置迁移
- `project__arch-migration` → 跨模块架构迁移

**沉淀规则**：
- **进行中**：handoff 保存临时状态（目标、进度、阻塞点）
- **完成后**：稳定结论写入 `project.md`，handoff 标记为 `closed`

### 文件命名
```
YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md
```
- 时间戳：UTC，ISO 8601 格式
- 示例：`20260311T103000Z--ses_abc123--sisyphus.md`

### 文件格式（v1.1）

#### 模块工作 handoff
```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_abc123
task_id: task_456
module: ammalloc
submodule: thread_cache
slug: null                        # 模块工作：必须为 null
agent: sisyphus
status: active
memory_status: not_needed         # 默认 not_needed
supersedes: null                  # 被本 handoff 取代的旧文件
closed_at: null                   # 关闭时间（仅 status=closed）
closed_reason: null               # 关闭原因（仅 status=closed）
---
```

#### 项目级工作 handoff
```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_abc123
task_id: task_456
module: project                   # 固定为 project
submodule: null                   # 项目级工作：必须为 null
slug: docs-reorg                  # workstream 标识（如 docs-reorg、ci-setup）
agent: sisyphus
status: active
memory_status: not_needed         # 默认 not_needed
supersedes: null
closed_at: null
closed_reason: null
---
```

#### 决策表：何时用模块 vs 项目级

| 场景 | 推荐类型 | Workstream 键示例 | 理由 |
|------|---------|-------------------|------|
| 开发 ammalloc 的 thread_cache | 模块 | `ammalloc__thread_cache` | 有明确模块归属 |
| 重构 docs/ 目录结构 | 项目级 | `project__docs-reorg` | 跨模块、无单一归属 |
| 配置 CI/CD | 项目级 | `project__ci-setup` | 影响整个仓库 |
| 修复 ammalloc 的 bug（需改配置） | 模块 | `ammalloc__size_class` | 虽有配置改动，但属模块内 |
| 添加项目级日志规范 | 项目级 | `project__logging-std` | 跨模块规范 |
| 设计 review | 项目级 | `project__arch-review-2026q1` | 跨模块评审 |

**核心原则**：
- 如果工作可以问"这是哪个模块的事？"且有明确答案 → 用模块类型
- 如果工作影响整个仓库或多个模块，无法归属单一模块 → 用项目级类型
- 如果犹豫，先用项目级，后续可归一时再迁移

**字段说明**：
- `schema_version`: "1.1"（当前版本）
- `status`: 生命周期状态（active 可恢复，superseded/closed 终态）
- `memory_status`: memory 回写进度（not_needed | pending | applied）
- `supersedes`: 如果是取代旧 handoff，填写旧文件名
- `closed_at`/`closed_reason`: 工作完成时填写

### 向后兼容

- Reader 必须接受 `schema_version: "1.0"` 和 `"1.1"`
- 缺失字段使用默认值：
  - `memory_status: not_needed`
  - `status: active`
  - `supersedes: null`
  - `closed_at: null`
  - `closed_reason: null`

### 生成策略

**主要方式：用户主动触发（默认）**

用户明确告知 Agent 生成 handoff：
- 显式指令："生成 handoff"、"保存当前进度"
- 结束语："今天先到这里"、"结束工作"、"我去吃饭了"

**辅助方式：Agent 提示（可选）**

Agent 在以下情况**提示**用户是否生成 handoff：
- 长时间无操作（如 30 分钟）
- 检测到重大进展（完成了阻塞点）
- 对话即将结束或超时

**原则**：Agent **不自动**提交 git，需要用户确认后手动 `git commit/push`

### 跨机器同步流程

**在公司电脑结束工作：**
```bash
# 1. 用户说"生成 handoff"（Agent 按 handoff_template.md 生成文件）
# 2. 用户确认后提交到 git
git add docs/agent/handoff/
git commit -m "handoff: ammalloc thread_cache progress"
git push
```

**在家庭电脑继续工作：**
```bash
# 1. 拉取最新 handoff
git pull
# 2. Agent 自动读取 docs/agent/handoff/ 最新文件恢复上下文
```

### 文件排序与状态过滤规则
1. **只读取** `status: active` 的 handoff（忽略 `superseded` 和 `closed`）
2. 在 active handoff 中，按 `created_at` 降序排序
3. 若 `created_at` 相同，按文件名字典序 tie-break
4. 忽略 frontmatter 缺失或格式损坏的文件

### 状态转换
```
创建 handoff
    ↓
status: active, memory_status: pending/not_needed
    ↓
├─[生成新 handoff] → 新: active + supersedes=旧, 旧: superseded
├─[回写 memory] → memory_status: applied
└─[工作完成] → status: closed, closed_at, closed_reason
```

**规则**：
- 同一 workstream 同时只能有一个 `active` handoff
- `superseded` 和 `closed` 是终态，不可回到 `active`
- 新 handoff 取代旧 handoff 时，必须更新旧文件为 `status: superseded`

### 并发处理（git 冲突）
- **场景**：两台电脑同时生成 handoff，git push 时冲突
- **策略**：git 会标记冲突，用户手动选择保留哪一个
- **简化**：通常保留最新时间戳的文件即可
- **注意**：Agent 读取时只取 `status: active`，不自动合并

### 生命周期管理
- **写入**：直接写入目标路径（无需临时文件，git 管理版本）
- **状态更新**：修改旧 handoff 的 `status` 时，同步更新 git
- **清理策略**：
  - 本地自动清理策略（建议手动执行）：
    - 删除超过 7 天的 closed/superseded handoff
    - 始终保留最近 3 个文件（无论状态）
    - 永远不要删除 `status: active` 的 handoff
  - **远端（git）**：保留所有历史（包括 superseded 和 closed），供审计追溯
- **读取顺序**：
  1. 任务系统/对话中的 handoff（优先）
  2. `docs/agent/handoff/` 目录中 `status: active` 的最新文件（本地 + git 同步）
  3. 如果没有 active，视为"无可恢复状态"，直接从 memory 开始

### 边界情况处理

#### 模块存在但子模块 memory 不存在
**场景**：`docs/agent/memory/modules/<module>/module.md` 存在，但 `submodules/<submodule>.md` 不存在。

**处理**：
- 加载 module.md 中的相关信息（子模块划分、待办事项）
- 从 module.md 推断子模块职责和边界
- 标记为 partial 恢复（非完整）
- 提示用户考虑创建子模块 memory

#### 无 handoff（从 memory 重新开始）
**场景**：workstream 目录为空或没有 `status: active` 的 handoff。

**处理**：
- 从 module.md/submodule.md 的"待办事项"开始
- 标记为 partial 恢复
- 输出："无可恢复临时状态，从 memory 的待办事项开始"

#### 多个 active handoff（异常）
**场景**：同一 workstream 存在多个 `status: active` 的 handoff。

**处理**：
- 选择 `created_at` 最新的一个
- 显式警告用户检查并收敛状态
- 建议将旧的标记为 `superseded`

### 约束
- handoff 保持**语义临时**：不是真相源，只是会话上下文缓存
- 稳定结论必须在会话结束前回写到 `docs/agent/memory/` 或 ADR
- 不要把 `docs/agent/handoff/` 当作长期历史记录（虽然 git 会保留历史）

## 快捷恢复规则

使用 [`docs/agent/prompts/quick_resume.md`](../prompts/quick_resume.md) 一句话接续工作。

### 触发示例
```
"继续 ammalloc thread_cache 的工作"
"加载 tensor 的完整记忆和最新 handoff"
```

### 解析规则
1. **精确匹配优先**：直接对应 `docs/agent/memory/modules/<module>/`
2. **模糊匹配仅模块级**：`继续 ammalloc` 只加载模块，不自动扫所有子模块
3. **歧义时询问**：无法唯一命中时列出所有候选，要求明确指定

### 加载顺序（与详细模式一致）

**模块工作**：
```
AGENTS.md -> docs/agent/memory/README.md -> docs/agent/memory/project.md -> docs/agent/memory/modules/<module>/module.md -> docs/agent/memory/modules/<module>/submodules/<submodule>.md -> docs/agent/handoff/workstreams/<module>__<submodule>/
```

**项目级工作**：
```
AGENTS.md -> docs/agent/memory/README.md -> docs/agent/memory/project.md -> docs/agent/handoff/workstreams/project__<slug>/
```

### 精简输出（5要点）
- 已解析范围（模块/子模块/workstream）
- 已加载文件（✅ 标记实际加载的文件）
- 当前接续目标
- 下一步动作
- 阻塞点（如有）

## 文件路径规范
```text
docs/agent/memory/
├── README.md
├── project.md
└── modules/
    └── <module>/
        ├── module.md
        ├── submodules/
        │   └── <submodule>.md
        └── adrs/
            └── ADR-XXX.md
```
- `project.md` 只存跨模块稳定事实，不重复模块内部细节。
- 每个主模块目录只保留一个 `module.md`，子模块统一放入 `submodules/`。
- 模块级 ADR 默认放在对应模块的 `adrs/` 目录；模板来源见 `docs/agent/decisions/template.md`。

## 何时创建记忆文档
- 创建主模块记忆：某一能力域已经形成稳定边界，并且会跨多个文件、目标或会话反复维护。
- 继续只用 `project.md`：信息仍然是跨模块共识，或不足以支撑单独的模块边界。
- 创建子模块记忆：该部分拥有独立接口、不变量、所有权、并发或性能约束，且这些信息继续堆在 `module.md` 会显著降低可读性。
- 不创建子模块记忆：内容只是父模块的局部实现细节，或当前只有一次性任务信息。
- 同一事实只保留一处主描述；公共约束留在更上层，局部特例留在更下层并显式指出覆盖范围。

## ADR 链接规范
- 只有对架构、接口契约、并发模型或性能特征产生长期影响的决定才创建 ADR。
- 主模块记忆通过 frontmatter `adr_refs` 维护索引；正文引用使用相对路径，例如 `[ADR-001](./modules/ammalloc/adrs/ADR-001.md)`。
- 子模块记忆优先复用父模块 `adrs/` 目录；正文引用使用相对路径，例如 `[ADR-001](./modules/ammalloc/adrs/ADR-001.md)`。
- 项目级记忆只链接跨模块相关的 ADR，例如 `./modules/<module>/adrs/ADR-001.md`；不要把 ADR 原因大段复制回 memory。
- ADR 若被替代或废弃，必须同步更新引用它的 `adr_refs` 与正文链接。

## 记忆更新触发条件
- 任务完成后形成了新的稳定事实、接口契约或不变量。
- 设计里程碑落定，且结果已被代码、测试、评审结论或用户显式指令确认。
- 新增、替换或废弃 ADR，导致模块边界、并发模型、性能策略或所有权规则发生变化。
- handoff 中出现了会影响后续开发的稳定结论，需要在结束会话前回写为长期记忆。
- 纯试验、调试日志、临时 workaround 和未定案讨论不进入 memory。

## 命名约定
- `<module>` 使用仓库内稳定领域名，统一采用 ASCII、小写、`snake_case`。
- `<submodule>` 使用父模块内可独立检索的单一概念名，统一采用 ASCII、小写、`snake_case`。
- 目录名与文件内 frontmatter 保持一致；不要引入 `temp`、`final`、`v2` 之类临时命名。
- ADR 文件名使用零填充编号：`ADR-001.md`、`ADR-002.md`。

## 冲突处理
- 同一 memory 文件同一时间只允许一个活跃编辑者提交最终版本。
- 若并发工作触及同一文件，后写入方必须基于最新文件和已验证代码事实做显式合并评审，不能静默覆盖。
- 合并时先保留共同事实，再按作用范围下沉或上提内容；无法确认的冲突项先留在 handoff 或任务记录，不要写成稳定结论。

## 元数据规范
- 除本控制文档外，所有 memory 文件必须包含 YAML frontmatter。
- 必填字段：`scope`、`module`、`parent`、`depends_on`、`adr_refs`、`last_verified`、`owner`、`status`。
- `scope` 取值：`global`、`module`、`submodule`。
- `module` 表示当前记忆节点名称；项目级固定写 `project`。
- `parent`：项目级与主模块写 `none`；子模块写父模块名。
- `depends_on`：列出直接依赖的其他模块或子模块名；没有时写 `[]`。
- `adr_refs`：列出当前文件直接引用的 ADR 标识或相对路径；没有时写 `[]`。
- `last_verified` 使用 `YYYY-MM-DD`。
- `owner` 写 `agent`、`team` 或明确责任主体。
- `status` 取值建议：`active`、`draft`、`deprecated`。
- 推荐模板：

```yaml
---
scope: module
module: example_module
parent: none
depends_on: []
adr_refs:
  - ./adrs/ADR-001.md
last_verified: 2026-03-10
owner: team
status: active
---
```
