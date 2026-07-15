# 快速恢复工作流（Quick Resume）

> **默认入口**：一句话恢复单一 workstream 的当前状态
> **正式入口**：需要显式填写目标、回写项或跨模块信息时，使用 `new_session_template.md`
> **启动契约**：默认启动路径、确认闸门与 Resume Gate 统一以根目录 `AGENTS.md` 第12节为准

## 适用范围

- 继续单一项目级 workstream：`project__<slug>`
- 继续单一模块 workstream：`<module>__<submodule-or-none>`
- 不适用于跨模块协调、首次建档、复杂 ADR 回写

## 触发语句示例

- "继续 project__docs-reorg 的工作"
- "继续 ammalloc thread_cache 的工作"
- "继续 ammalloc"
- "恢复昨天的 docs 重构工作"

## 解析原则

- 显式给出 `project__<slug>` 时，按项目级 workstream 处理
- 显式给出模块/子模块时，按模块 workstream 处理
- 只给模块名时，默认解析为 `<module>__none`
- 无法唯一确定时，必须要求用户澄清
- 不得发明 slug，不得把模板文件当作 handoff

## 恢复流程

1. 按根目录 `AGENTS.md` 第12节解析工作流类型
2. 执行根目录 `AGENTS.md` 第12节定义的默认启动与只读引导操作（读取 AGENTS.md、project.md，扫描候选 frontmatter，读取唯一选中 handoff 正文，按需升级读取 module.md/submodule.md）
3. 若 handoff 不足或任务触及更深约束，再按需升级读取 `README.md` / `module.md` / `submodule.md`
4. 输出 Resume Gate：已加载文件、`resume_status`、下一步建议、阻塞点
5. **必须通过 Resume Gate 并等待用户明确确认后**，才能执行非恢复/业务工具操作
6. 第一轮“继续”表示选择 workstream，不授权执行业务操作；只有在 Resume Gate 询问后用户说“继续”/“执行”/“是”时，才授权执行推荐操作

## 输出要求

至少包含以下内容：

- 工作流类型与 `resolved_workstream_key`
- 实际加载的文件路径
- 是否发生按需升级读取
- `resume_status: complete | partial | blocked`
- 当前接续目标
- 推荐下一步
- 阻塞点（若无则写 `无`）

## handoff 规则提醒

- handoff 位于 `docs/agent/handoff/workstreams/<workstream_key>/`
- 扫描候选 frontmatter 后，只读取唯一选中的 `status: active` handoff 正文
- `bootstrap_ready: true` 可用于低上下文恢复；缺失按 `false` 处理
- `docs/agent/prompts/handoff_template.md` 只是模板，不是运行时状态

## 何时改用 `new_session_template.md`

- 需要显式填写目标、回写项、ADR 增量
- 需要跨模块协调
- 需要正式启动一项新工作
- 当前请求本身就需要更多结构化上下文
