# Prompt 说明

> 用途：说明 `docs/agent/prompts/` 下提示词的职责、衔接关系和启动流程。
>
> workstream 键、handoff frontmatter 和加载顺序以 `docs/agent/memory/README.md` 为最终依据。

## Prompt 列表

### 会话启动（二选一）
- **`docs/agent/prompts/quick_resume.md`（默认）**：一句话快速接续工作。  
  适合日常恢复："继续 ammalloc thread_cache 的工作"、"继续 project__docs-reorg 的工作"
  
- **`docs/agent/prompts/new_session_template.md`（显式）**：结构化正式启动。  
  适合首次建档、跨模块任务、需要预填 ADR/回写项的场景；支持 `<module>__<submodule-or-none>` 与 `project__<slug>` 两类 workstream。

### 会话流转
- `docs/agent/prompts/handoff_template.md`：会话交接摘要模板。输出给下一位 agent 的最小必要上下文。
- `docs/agent/prompts/generate_module_memory.md`：生成完整主模块或子模块 memory 草案，适合首次建档或大规模重写。
- `docs/agent/prompts/memory_update_and_adr.md`：生成 memory 增量和 ADR 草案，适合任务结束后的稳定结论回写。

## 工作流

### 快速恢复（默认）
```text
旧会话结束
    -> handoff_template.md 生成摘要
    -> git commit/push docs/agent/handoff/
    
新会话启动（家庭电脑）
    -> git pull
    -> 用户说："继续 [模块] [子模块] 的工作" 或 "继续 project__<slug> 的工作"
    -> Agent 按 quick_resume.md 加载 project.md + （module.md/submodule.md，如适用）+ 最新 handoff
    -> Agent 输出"记忆已加载，推荐下一步是..."
    -> **用户确认**："继续" / "执行"
    -> 执行任务
    -> memory_update_and_adr.md 回写稳定结论
    -> 新一轮 handoff
```

### 显式启动（备选）
```text
新会话启动
    -> 用户填充 new_session_template.md 变量（workstream = <module>__<submodule-or-none> | project__<slug>）
    -> Agent 按模板加载 memory 与 handoff
    -> 模块工作：docs/agent/memory/project.md -> docs/agent/memory/modules/<module>/module.md -> docs/agent/memory/modules/<module>/submodules/<submodule>.md -> docs/agent/handoff/workstreams/<module>__<submodule>/
    -> 项目级工作：docs/agent/memory/project.md -> docs/agent/handoff/workstreams/project__<slug>/
    -> 输出"记忆已加载，推荐下一步是..."
    -> ⚠️ 等待用户说"继续"、"执行"或"是"（强制确认）
    -> 执行任务
    -> generate_module_memory.md 或 memory_update_and_adr.md
    -> handoff_template.md
```

## 关系说明
- `new_session_template.md` 依赖 `handoff_template.md` 的输出结构，并要求按 `docs/agent/memory/README.md` 的路径规则加载 memory。
- `generate_module_memory.md` 的输出必须匹配 `docs/agent/memory/mainmodule_memory_template.md` 或 `docs/agent/memory/submodule_memory_template.md`。
- `memory_update_and_adr.md` 负责把本轮稳定结论整理成 memory 增量，并按 `docs/agent/decisions/template.md` 的字段生成 ADR 草案。
- `handoff_template.md` 只描述当前进度；真正长期保留的信息要回写到 memory 或 ADR。

## Agent 快速启动
1. 先读 `AGENTS.md`，再读 `docs/agent/memory/README.md`。
2. 加载 `docs/agent/memory/project.md`。
3. 若 workstream 为模块类型，加载 `docs/agent/memory/modules/<module>/module.md`；若命中子模块，再加载 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`。
4. 若 workstream 为 `project__<slug>`，跳过 `docs/agent/memory/modules/<module>/module.md` 和 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`，直接定位 `docs/agent/handoff/workstreams/project__<slug>/`。
5. 如果存在上一轮交接摘要，按 `docs/agent/prompts/handoff_template.md` 的结构读取并吸收当前状态。
6. 任务结束后，优先用 `docs/agent/prompts/memory_update_and_adr.md` 回写稳定结论；仅在首次建档或需要整体重写时使用 `docs/agent/prompts/generate_module_memory.md`。
