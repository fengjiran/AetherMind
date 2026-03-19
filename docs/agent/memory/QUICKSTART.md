# Agent Memory System 快速提示

> 这是**极简操作提示**，不是完整教程。
> 默认启动契约以根目录 `AGENTS.md` 为准；需要 schema、命名、兼容与写回细则时，再查 `docs/agent/memory/README.md`。

## 1. 启动前先判断工作流类型

- 项目级：`project__<slug>`
- 模块级：`<module>__<submodule-or-none>`
- 无法唯一确定时：先澄清，不要猜测

## 2. 默认启动只做三件事

1. 读取根目录 `AGENTS.md` 的启动契约
2. 读取 `docs/agent/memory/project.md`
3. 读取目标 workstream 的最新 `active` handoff

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
- 只读取 `status: active` 的 handoff
- `docs/agent/prompts/handoff_template.md` 只是模板，不是运行时状态
- `bootstrap_ready: true` 表示该 handoff 可用于低上下文恢复；缺失按 `false` 处理

## 6. 继续执行前必须做的事

- 输出已加载文件
- 输出 `resume_status`
- 询问用户是否继续
- 在用户明确说“继续”/“执行”/“是”之前，不执行工具操作

## 7. 进一步查阅

- 启动契约：根目录 `AGENTS.md`
- 操作手册：`docs/agent/memory/README.md`
- 架构说明：`docs/agent/memory_system.md`
