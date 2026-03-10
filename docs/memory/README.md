# 记忆系统操作规范

> **本文档为记忆系统的操作规范**。  
> 架构设计说明见: [`docs/agent_memory_system.md`](../agent_memory_system.md)
>
> 用途：定义 `docs/memory/` 的唯一结构规则。稳定事实写入记忆文档；会话级临时状态保留在 handoff。

## 本文档范围
- 本文档是 `docs/memory/` 的操作规范，用于实际创建、读取、更新和维护 memory 文件；不承担架构总览说明。
- 架构设计说明：[`docs/agent_memory_system.md`](../agent_memory_system.md)
- Prompt 入口与衔接：[`docs/prompts/README.md`](../prompts/README.md)、[`docs/prompts/new_session_template.md`](../prompts/new_session_template.md)
- 需要理解分层模型、工作流全貌和示例时，优先阅读架构文档；需要执行路径、元数据、命名和冲突规则时，优先阅读本文档。
- 实际使用或回写 memory 时，以本文档为最终操作依据。

## 记忆层级模型
- 全局层：`docs/memory/project.md`。记录跨模块稳定事实、目录约定和默认规则。
- 模块层：`docs/memory/modules/<module>/module.md`。记录主模块职责、边界、接口、不变量和长期约束。
- 子模块层：`docs/memory/modules/<module>/submodules/<submodule>.md`。只在父模块内存在独立边界时拆分；与主模块同属领域层。
- handoff 层：按 `docs/prompts/handoff.md` 输出当前会话摘要，只服务交接，不替代稳定记忆；输出存储在任务记录/对话中，不作为长期文件保存。
- 读取顺序：`AGENTS.md` -> `docs/memory/README.md` -> `project.md` -> `module.md` -> `submodule.md`（如存在）-> handoff。
- 冲突优先级：用户显式指令与已验证代码/测试事实 > `AGENTS.md` > `docs/aethermind_prd.md` > ADR > 模块/子模块记忆 > `docs/memory/project.md` > handoff > `GEMINI.md`。
- handoff 与稳定记忆冲突时，先回到代码、测试或用户指令验证；未验证前不要直接覆盖 memory 文件。

## 文件路径规范
```text
docs/memory/
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
- 模块级 ADR 默认放在对应模块的 `adrs/` 目录；模板来源见 `docs/decisions/template.md`。

## 何时创建记忆文档
- 创建主模块记忆：某一能力域已经形成稳定边界，并且会跨多个文件、目标或会话反复维护。
- 继续只用 `project.md`：信息仍然是跨模块共识，或不足以支撑单独的模块边界。
- 创建子模块记忆：该部分拥有独立接口、不变量、所有权、并发或性能约束，且这些信息继续堆在 `module.md` 会显著降低可读性。
- 不创建子模块记忆：内容只是父模块的局部实现细节，或当前只有一次性任务信息。
- 同一事实只保留一处主描述；公共约束留在更上层，局部特例留在更下层并显式指出覆盖范围。

## ADR 链接规范
- 只有对架构、接口契约、并发模型或性能特征产生长期影响的决定才创建 ADR。
- 主模块记忆通过 frontmatter `adr_refs` 维护索引；正文引用使用相对路径，例如 `[ADR-001](./adrs/ADR-001.md)`。
- 子模块记忆优先复用父模块 `adrs/` 目录；正文引用使用相对路径，例如 `[ADR-001](../adrs/ADR-001.md)`。
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
