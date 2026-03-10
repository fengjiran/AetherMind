# Prompt 说明

> 用途：说明 `docs/prompts/` 下四个提示词的职责、衔接关系和最小启动流程。

## Prompt 列表
- `docs/prompts/new_session_template.md`：新会话入口。先收敛目标，再决定需要加载哪些 memory 与 handoff。
- `docs/prompts/handoff.md`：会话交接摘要模板。输出给下一位 agent 的最小必要上下文。
- `docs/prompts/generate_module_memory.md`：生成完整主模块或子模块 memory 草案，适合首次建档或大规模重写。
- `docs/prompts/memory_update_and_adr.md`：生成 memory 增量和 ADR 草案，适合任务结束后的稳定结论回写。

## 工作流
```text
旧会话结束
    -> handoff.md
    -> 新会话读取 new_session_template.md
    -> 加载 project/module/submodule memory
    -> 执行任务
    -> 若需要完整重建 memory：generate_module_memory.md
    -> 若只需增量回写或新增决策：memory_update_and_adr.md
```

## 关系说明
- `new_session_template.md` 依赖 `handoff.md` 的输出结构，并要求按 `docs/memory/README.md` 的路径规则加载 memory。
- `generate_module_memory.md` 的输出必须匹配 `docs/memory/mainmodule_memory_template.md` 或 `docs/memory/submodule_memory_template.md`。
- `memory_update_and_adr.md` 负责把本轮稳定结论整理成 memory 增量，并按 `docs/decisions/template.md` 的字段生成 ADR 草案。
- `handoff.md` 只描述当前进度；真正长期保留的信息要回写到 memory 或 ADR。

## Agent 快速启动
1. 先读 `AGENTS.md`，再读 `docs/memory/README.md`。
2. 加载 `docs/memory/project.md`。
3. 根据任务定位主模块 `docs/memory/modules/<module>/module.md`；若命中子模块，再加载 `docs/memory/modules/<module>/submodules/<submodule>.md`。
4. 如果存在上一轮交接摘要，按 `docs/prompts/handoff.md` 的结构读取并吸收当前状态。
5. 任务结束后，优先用 `docs/prompts/memory_update_and_adr.md` 回写稳定结论；仅在首次建档或需要整体重写时使用 `docs/prompts/generate_module_memory.md`。
