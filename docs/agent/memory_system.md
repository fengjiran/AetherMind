# AetherMind Agent Memory System Architecture（架构说明）

> 本文档解释 Agent Memory System 的设计目标、分层职责和架构取舍。
> 它是**架构参考文档**，不是默认启动入口。
> 默认启动契约与最小加载路径，统一以根目录 `AGENTS.md` 为准。

## 1. 设计目标
Agent Memory System 解决以下长期问题：
- 多会话间上下文漂移
- 重要决策散落在对话中，难以追溯
- 大型仓库中约束与边界不易发现
- handoff 与稳定事实混写导致冲突面扩大
- 新对话默认加载过多文档，启动成本过高

## 2. 分层模型
系统采用“三层 stable memory + 一层 handoff”的结构：

- 全局层：`docs/agent/memory/project.md`
- 模块层：`docs/agent/memory/modules/<module>/module.md`
- 子模块层：`docs/agent/memory/modules/<module>/submodules/<submodule>.md`
- handoff 层：`docs/agent/handoff/workstreams/<workstream_key>/`

### 角色边界
- stable memory：长期有效的事实、接口、不变量、约束
- handoff：当前状态增量、阻塞点、下一步、未完成事项
- 根目录 `AGENTS.md`：启动契约与执行硬约束

## 3. 核心原则

- 单点事实：同一稳定事实只保留一处主描述
- 分层存储：长期事实与临时状态严格分离
- 显式引用：依赖、ADR 和路径规则都应可追踪
- 渐进加载：默认只加载最小必要信息，更深规则按需升级读取
- 兼容优先：旧 handoff 缺少新字段时走安全降级路径


## 4. Summary vs Reference 拆分
为降低新对话启动成本，文档职责拆分如下：

- 根目录`AGENTS.md`：唯一规范性启动契约
- `docs/agent/memory/README.md`：按需操作参考
- `docs/agent/memory_system.md`：架构说明与设计意图
- `docs/agent/memory/QUICKSTART.md`：极简操作提示，不再承载完整教程

这一拆分的目标不是减少规则，而是把“默认必读”和“遇事再查”分开。

## 5. Handoff 设计要点

- handoff 必须带 frontmatter，支持 `v1.0` 与 `v1.1`
- `status: active` 表示当前可恢复状态
- `bootstrap_ready` 用于标记该 handoff 是否足以支撑低上下文恢复
- 缺失 `bootstrap_ready` 时，读取方必须按 `false` 处理
- handoff 不得复制长期启动规则或稳定 memory 内容

## 6. 为什么默认启动必须变轻

旧设计把启动规则、教程、操作规范和架构说明混在一起，导致：

- 默认新对话需要加载过多文档
- 多个文件重复定义加载顺序与 Resume Gate
- 修改任一入口后容易产生规则漂移

轻量化后的目标是：

- 默认只加载最小安全集合
- 复杂规则只在需要时升级读取
- 通过 validator 防止重复契约回流

## 7. 维护建议

- 修改启动契约时，优先更新 `AGENTS.md`
- 修改 schema、兼容、命名或写回规则时，更新 `docs/agent/memory/README.md`
- 修改架构目标或分层设计时，更新本文档
- 修改 handoff 写法时，同时检查 `docs/agent/prompts/handoff_template.md` 与 `docs/agent/handoff/README.md`

## 8. 相关文档

- 启动契约：根目录`AGENTS.md`
- 操作参考：`docs/agent/memory/README.md`
- 项目级稳定事实：`docs/agent/memory/project.md`
- handoff 模板：`docs/agent/prompts/handoff_template.md`
- handoff 存储说明：`docs/agent/handoff/README.md`
