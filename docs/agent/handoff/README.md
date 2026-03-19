# Handoff 目录说明

本目录存储 Agent Memory System 的 handoff（会话交接）文件。

## 目录结构

```
docs/agent/handoff/
└── workstreams/
    ├── <module>__<submodule-or-none>/
    │   └── YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md
    └── project__<slug>/
        └── YYYYMMDDTHHMMSSZ--<session_id>--<agent_id>.md
```

## 文件格式（v1.1）

每个 handoff 文件必须包含 YAML frontmatter：

**模块级示例**：

```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_xxx
task_id: task_xxx
module: ammalloc
submodule: thread_cache
slug: null                        # 模块工作：必须为 null
agent: sisyphus
status: active                    # active | superseded | closed
bootstrap_ready: false            # 默认 false；仅当 handoff 足以支撑低上下文恢复时才写 true
memory_status: not_needed         # not_needed | pending | applied（默认 not_needed）
supersedes: null                  # 被本 handoff 取代的旧文件（可选）
closed_at: null                   # 关闭时间（仅 status=closed）
closed_reason: null               # 关闭原因（仅 status=closed）
---
```

**项目级示例**：

```yaml
---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-11T10:30:00Z
session_id: ses_xxx
task_id: task_xxx
module: project                   # 固定值
submodule: null                   # 项目级工作：必须为 null
slug: docs-reorg                  # 项目级工作：填写 slug
agent: sisyphus
status: active
bootstrap_ready: false            # 默认 false；仅当 handoff 足以支撑低上下文恢复时才写 true
memory_status: not_needed
supersedes: null
closed_at: null
closed_reason: null
---
```

### 状态说明

**status（生命周期）**：
- `active`：当前可继续接手的唯一 handoff（非终态）
- `superseded`：已被新 handoff 取代（终态）
- `closed`：工作完成，不再需要恢复（终态）

**memory_status（回写进度）**：
- `not_needed`：无需回写 stable memory（默认）
- `pending`：有稳定结论待回写
- `applied`：已回写 stable memory 或 ADR

**bootstrap_ready（低上下文恢复资格）**：
- `false`：默认值；恢复时只可作为 handoff 候选，不能单独替代更深层 memory
- `true`：该 handoff 已包含继续工作所需的最小恢复信息，可优先用于低上下文恢复
- 缺失字段时，Reader **必须**按 `false` 处理，确保旧 handoff 走安全降级路径

### 向后兼容

- Reader 必须接受 `schema_version: "1.0"` 和 `"1.1"`
- 缺失字段使用默认值：
  - `bootstrap_ready: false`
  - `memory_status: not_needed`
  - `status: active`
  - `supersedes: null`
  - `closed_at: null`
  - `closed_reason: null`

## 状态转换

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
- 新 handoff 取代旧 handoff 时，**必须**更新旧文件的 `status: superseded`

## 读取策略

恢复工作时：
1. **只读取** `status: active` 的 handoff
2. **排序**：按 `created_at` 降序，文件名字典序 tie-break
3. **忽略**：`superseded` 和 `closed`（保留用于审计和排查）
4. **空结果**：如果没有 `active`，从 memory 重新开始
5. **低上下文恢复**：只有 `bootstrap_ready: true` 的 handoff 才可直接支撑最小启动路径；否则必须按 Agent 规则升级读取 `module.md`、`submodule.md` 或 `docs/agent/memory/README.md`

## 同步机制

handoff 文件通过 git 同步实现跨机器恢复：

```bash
# 结束工作时提交（Agent 辅助生成，用户确认后提交）
git add docs/agent/handoff/
git commit -m "handoff: <workstream_key> progress"
git push

# 新机器恢复时拉取
git pull
```

## 规范参考

- 完整规范见 `docs/agent/memory/README.md` 的 "Handoff 存储规范" 章节
- 快捷恢复流程见 `docs/agent/prompts/quick_resume.md`

## 设计约束

- handoff 只保存**当前状态增量**，不重复长期稳定规则
- 默认启动时优先使用最小可行上下文；但缺失 `bootstrap_ready` 或值为 `false` 时，必须安全降级为更完整的记忆加载路径
