# Prompt 说明
> 用途：说明 `docs/agent/prompts/` 下各提示词的职责边界。
> 默认启动契约、最小加载路径、确认闸门与 Resume Gate，统一以根目录 `AGENTS.md` 为准。

## Prompt 列表

### 会话启动

- `docs/agent/prompts/quick_resume.md`
  - 默认入口
  - 适合恢复单一 workstream
  - 不重新定义启动顺序，只说明何时用它

- `docs/agent/prompts/new_session_template.md`
  - 结构化正式入口
  - 适合首次建档、跨模块任务、需要显式输入目标/回写项/ADR 增量的场景

### 会话流转

- `docs/agent/prompts/handoff_template.md`
  - handoff 写作模板
  - 只定义交接摘要结构，不是运行时状态源

- `docs/agent/prompts/generate_module_memory.md`
  - 生成完整模块或子模块 memory 草案

- `docs/agent/prompts/memory_update_and_adr.md`
  - 生成 memory 增量和 ADR 草案

## 使用关系
- 启动类 prompt 负责“如何进入工作”，不负责“重新定义启动契约”
- 真正的默认启动规则在根目录 `AGENTS.md`
- 需要 schema、兼容、命名或写回规则时，查 `docs/agent/memory/README.md`
- 需要 handoff 存储与状态规则时，查 `docs/agent/handoff/README.md`

## 选择建议

- 恢复单一 workstream → `quick_resume.md`
- 正式启动或跨模块任务 → `new_session_template.md`
- 生成交接摘要 → `handoff_template.md`
- 回写稳定结论 → `memory_update_and_adr.md`
- 首次建档或整体重写模块 memory → `generate_module_memory.md`

## 边界约束

- 不要把 `handoff_template.md` 当作真实 handoff
- 不要在 prompt 文档中重复完整加载顺序或 Resume Gate 模板
- prompt 文档可以引用 `AGENTS.md`，但不应与其形成第二套规范
