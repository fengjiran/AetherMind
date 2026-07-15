# Agent Memory System 快速提示

> 这是**极简操作提示**，不是完整教程。
> 默认启动契约以根目录 `AGENTS.md` 为准；需要 schema、命名、兼容与写回细则时，再查 `docs/agent/memory/README.md`。

## 1. 启动前先判断工作流类型

- 项目级：`project__<slug>`
- 模块级：`<module>__<submodule-or-none>`
- 无法唯一确定时：先澄清，不要猜测

## 2. 默认启动只做三件事

1. 读取根目录 `AGENTS.md` 的启动契约（第12节）
2. 读取 `docs/agent/memory/project.md`
3. 扫描目标 workstream 下所有候选 handoff 文件的 frontmatter（仅元数据），筛选 `active` 后排序，读取唯一选中 handoff 的正文

> 项目级工作默认跳过 `module.md` 和 `submodule.md`；仅当触及模块边界、所有权、线程安全或性能约束时才按需读取，并在 Resume Gate 中记录。

## 3. 何时升级读取

出现以下任一情况时，再读取更深层文档：

- 没有可用的 active handoff
- handoff 信息不足，无法继续工作
- 任务触及模块边界、生命周期、线程安全或性能约束
- 需要处理 handoff schema、兼容、命名、写回或冲突

## 4. 升级读取顺序

- 先查 `docs/agent/memory/README.md`（操作与 schema）
- 模块工作再查 `docs/agent/memory/modules/<module>/module.md`
- 需要更细约束时再查 `docs/agent/memory/modules/<module>/submodules/<submodule>.md`

## 5. Handoff 记忆

- handoff 存在于 `docs/agent/handoff/workstreams/<workstream_key>/`
- 扫描候选 frontmatter 后，只读取唯一选中的 `status: active` handoff 正文
- `docs/agent/prompts/handoff_template.md` 只是模板，不是运行时状态
- `bootstrap_ready: true` 表示该 handoff 可用于低上下文恢复；缺失按 `false` 处理

## 6. 继续执行前必须做的事

- 输出已加载文件（含 Resume Gate 清单）
- 输出 `resume_status`
- 询问用户“记忆已加载，是否执行[推荐操作]？”
- 第一轮“继续”表示选择 workstream，不授权执行业务操作
- 只有在 Resume Gate 中明确询问后，用户说“继续”/“执行”/“是”时，才授权执行推荐操作
- 确认前不执行非恢复/业务工具操作（如代码扫描、构建、测试、编辑、写入等外部副作用操作）

## 7. 进一步查阅

- 启动契约：根目录 `AGENTS.md`
- 操作手册：`docs/agent/memory/README.md`
- 架构说明：`docs/agent/memory_system.md`
