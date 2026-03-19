---
scope: global
module: agent_memory
parent: none
depends_on: []
adr_refs: []
last_verified: 2026-03-18
owner: team
status: active
---

# 记忆系统操作规范
> 本文件是 Agent Memory System 的**按需升级读取操作手册**。
> 默认启动契约、最小加载路径、Resume Gate 与确认闸门，统一以根目录 `AGENTS.md` 为准。
> 新对话默认**不要求**完整读取本文件。

## 1. 本文件负责什么
- 负责：memory/handoff 的目录规则、命名、frontmatter 契约、兼容默认值、写回规则、冲突处理。
- 不负责：重新定义默认启动顺序、重复 Resume Gate 模板、代替根目录 `AGENTS.md` 充当启动入口。

### 何时读取本文件
仅在以下情况按需升级读取：
- 需要生成、更新或校验 handoff frontmatter
- 需要回写 stable memory、整理长期结论或处理 ADR 链接
- 需要处理 workstream 键、命名、兼容默认值或冲突合并
- 需要判断 handoff 存储规则、状态流转或清理策略

## 2. 记忆层级模型

- 全局层：`docs/agent/memory/project.md`
  - 存放项目级跨模块稳定事实、目录约定、默认规则
- 模块层：`docs/agent/memory/modules/<module>/module.md`
  - 存放主模块职责、边界、接口、不变量和长期约束
- 子模块层：`docs/agent/memory/modules/<module>/submodules/<submodule>.md`
  - 存放子模块级约束、实现边界、性能/并发细节
- handoff 层：`docs/agent/handoff/workstreams/<workstream_key>/`
  - 只保存**当前状态增量**，不替代长期稳定事实

## 3. Workstream 键与存储路径
### 模块级工作
- on-disk key：`<module>__<submodule-or-none>`
- 示例：`ammalloc__thread_cache`、`ammalloc__none`
- handoff 目录：`docs/agent/handoff/workstreams/<module>__<submodule-or-none>/`

### 项目级工作
- on-disk key：`project__<slug>`
- 示例：`project__docs-reorg`、`project__agent-memory-v1.1`
- handoff 目录：`docs/agent/handoff/workstreams/project__<slug>/`

### 选择原则
- 能明确回答“这是哪个模块的事？”并且答案唯一 → 模块工作
- 影响整个仓库或跨多个模块，无法归属单一模块 → 项目级工作


## 4. Handoff frontmatter 契约
### 模块级工作示例

```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_xxx
task_id: task_xxx
module: ammalloc
submodule: thread_cache
slug: null
agent: sisyphus
status: active
bootstrap_ready: false
memory_status: not_needed
supersedes: null
closed_at: null
closed_reason: null
---
```

### 项目级工作示例

```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_xxx
task_id: task_xxx
module: project
submodule: null
slug: docs-reorg
agent: sisyphus
status: active
bootstrap_ready: false
memory_status: not_needed
supersedes: null
closed_at: null
closed_reason: null
---
```

### 字段语义

- `kind`: 固定为 `handoff`
- `schema_version`: 当前接受 `"1.0"` 与 `"1.1"`
- `module`: 项目级固定为 `project`；模块级写模块名
- `submodule`: 项目级固定 `null`；模块级写子模块名或 `none` 语义对应的实际空值规则
- `slug`: 项目级必须为非空字符串；模块级必须为 `null`
- `status`: `active | superseded | closed`
- `bootstrap_ready`: 是否足以支撑低上下文恢复；缺失时必须按 `false` 处理
- `memory_status`: `not_needed | pending | applied`
- `supersedes`: 取代旧 handoff 时写旧文件名，否则 `null`
- `closed_at` / `closed_reason`: 仅 `status: closed` 时填写

## 5. 兼容与默认值
- Reader 必须接受 `schema_version: "1.0"` 和 `"1.1"`
- 缺失字段默认值：
  - `bootstrap_ready: false`
  - `memory_status: not_needed`
  - `status: active`
  - `supersedes: null`
  - `closed_at: null`
  - `closed_reason: null`
- `bootstrap_ready` 为 `false` 或缺失时，只能作为 handoff 候选；不能单独替代更深层 memory

## 6. Handoff 读取与状态流转

### 读取规则

恢复工作时：
1. 只读取 `status: active` 的 handoff
2. 在 active handoff 中按 `created_at` 降序排序
3. `created_at` 相同时按文件名字典序 tie-break
4. frontmatter 缺失或损坏的文件不参与恢复
5. 若无 active handoff，则退化为“从 memory 开始”

### 状态流转

```text
创建 handoff
    ↓
status: active, memory_status: pending/not_needed
    ↓
├─[生成新 handoff] → 新: active + supersedes=旧, 旧: superseded
├─[回写 memory] → memory_status: applied
└─[工作完成] → status: closed, closed_at, closed_reason
```

### 约束
- 同一 workstream 同时只能有一个 `active` handoff
- `superseded` 和 `closed` 是终态，不可回到 `active`
- `supersedes` 非空时，目标文件必须存在于同一 workstream

## 7. Stable memory 与 handoff 的边界
- stable memory 记录长期有效的事实、边界和不变量
- handoff 只记录当前状态、阻塞点、下一步和未完成事项
- 不要把长期启动规则、完整加载顺序或模板性政策复制进 handoff
- handoff 中形成的稳定结论，应在合适时机回写到 `project.md`、`module.md`、`submodule.md` 或 ADR

## 8. 命名约定
- `<module>` 与 `<submodule>` 统一采用 ASCII、小写、`snake_case`
- 项目级 slug 使用稳定、可复用、可检索的 ASCII 标识
- ADR 文件名使用零填充编号：`ADR-001.md`、`ADR-002.md`
- 不要引入 `temp`、`final`、`v2` 之类临时命名

## 9. 记忆更新触发条件
- 任务完成后形成了新的稳定事实、接口契约或不变量
- 设计里程碑落定，且结果已被代码、测试、评审结论或用户显式指令确认
- 新增、替换或废弃 ADR，导致模块边界、并发模型、性能策略或所有权规则变化
- handoff 中出现影响后续开发的稳定结论，需要沉淀为长期记忆
- 纯试验、调试日志、临时 workaround、未定案讨论不进入 stable memory

## 10. 冲突处理
- 同一 memory 文件同一时间只允许一个活跃编辑者提交最终版本
- 并发修改同一 memory 文件时，后写入方必须基于最新文件和已验证事实做显式合并评审
- 无法确认的冲突项先留在 handoff 或任务记录，不要直接写成稳定结论
- handoff 与 stable memory 冲突时，优先回到代码、测试或用户指令验证

## 11. 元数据规范

- 除本控制文档外，所有 memory 文件必须包含 YAML frontmatter
- 必填字段：`scope`、`module`、`parent`、`depends_on`、`adr_refs`、`last_verified`、`owner`、`status`
- `scope` 取值：`global | module | submodule`
- `module`：当前记忆节点名称；项目级固定为 `project`
- `parent`：项目级与主模块写 `none`；子模块写父模块名
- `depends_on`：列出直接依赖的其他模块或子模块；没有时写 `[]`
- `adr_refs`：列出当前文件直接引用的 ADR 标识或相对路径；没有时写 `[]`
- `last_verified`：使用 `YYYY-MM-DD`
- `owner`：写 `agent`、`team` 或明确责任主体
- `status`：建议值为 `active | draft | deprecated`

推荐模板：

```yaml
---
scope: module
module: example_module
parent: none
depends_on: []
adr_refs:
  - ./adrs/ADR-001.md
last_verified: 2026-03-18
owner: team
status: active
---
```

## 12. 相关文档

- 默认启动契约：`AGENTS.md`
- 项目级稳定事实：`docs/agent/memory/project.md`
- handoff 模板：`docs/agent/prompts/handoff_template.md`
- handoff 存储说明：`docs/agent/handoff/README.md`
- 架构说明：`docs/agent/memory_system.md`
