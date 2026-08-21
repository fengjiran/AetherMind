# AetherMind 文档系统规范

> 本文件定义 AetherMind 文档系统的分类、命名、交叉引用、质量标准与维护流程。所有文档贡献者必须遵循。
>
> 优先级：与 `AGENTS.md` 或已验证的仓库事实冲突时，以 `AGENTS.md` 与仓库事实为准。

## 1. 分类与职责边界

| 类型 | 目录 | 状态字段 |
|---|---|---|
| 架构总览 | `docs/designs/architecture/architecture_overview.md` | Current |
| 模块设计 | `docs/designs/<module>/NN-*.md` | Current / Deprecated |
| 调研备忘 | `docs/designs/research/` | 无状态，头部标注日期与结论可信度 |
| 演进提案 | `docs/improvement-plan/` | Draft / In Progress / Implemented / Superseded |
| 开发指南 | `docs/guides/` | Current / Deprecated |
| API 参考 | `docs/api/` | Current |
| 决策记录 | `docs/decisions/` | Proposed / Accepted / Deprecated / Superseded |
| 评审报告 | `docs/reviews/` | Current / 历史快照 |
| 验证报告 | `docs/tests/` | Current |
| 问题跟踪 | `docs/issues.md` | 每条目 `[x]`/`[ ]` |
| 变更记录 | `CHANGELOG.md`（仓库根） | 无 |
| 开发日志 | `docs/logs/` | 无（追加式过程记录） |
| 产品需求 | `docs/products/` | Current |

**职责边界判定**：描述"代码里现在是什么" → `designs/`；"将来要做什么" → `improvement-plan/`；"曾经怎么决策的" → `decisions/`；"怎么干活" → `guides/`；"API 怎么用" → `api/`；"已知缺陷/待办" → `issues.md`。

**准入条件**：

- `designs/` 下所有文档只描述**已验证实现**；草稿/提案不得进入，统一放 `improvement-plan/` 或标记为调研。
- `decisions/` 每篇 ADR 必须包含至少 1 个被否定的备选方案。
- 调研备忘不承诺实现，禁止被后续文档当作事实引用（可作参考链接）。
- `docs/agent/` 是独立子系统，本规范不适用于其内部文档；唯一交叉点是 ADR 互链（见 §3.3）。

## 2. 命名规则

| 对象 | 规则 | 示例 |
|---|---|---|
| 目录 | kebab-case | `improvement-plan/`、`designs/graph/` |
| 模块设计 | `<module>/NN-<kebab-name>.md`，NN=两位，模块内编号（推荐阅读顺序，沿数据流方向） | `model/01-model-loader.md` |
| 架构总览 | 存量保留原名 `architecture/architecture_overview.md`（避免破坏引用，在 `docs/README.md` 索引标注） | — |
| ADR | `NNNN-<kebab-title>.md`，四位全局递增 | `0001-kernel-registry-freeze.md` |
| 演进提案 | `NN-<kebab-name>.md`，两位 | `02-session-api.md` |
| 指南/其他文档 | `<kebab-name>.md` | `documentation-guide.md` |
| 评审/验证报告 | `<topic>_<scope>_review|validation_<YYYY-MM-DD>.md` | `operator_kernel_architecture_review_2026-07-18.md` |
| 章节编号 | 阿拉伯数字层级，作为稳定标识 | `architecture_overview.md` 第 4 章 |

存量例外：`architecture_overview.md`、`guides/` 现有指南、`products/aethermind_prd.md`、`docs/agent/` 保留原名。

**编号管理**：

- 模块设计编号模块内全局唯一；新增时取当前最大编号 + 1（除非数据流顺序要求插入，插入时一并重排后续编号并更新引用）。
- ADR 编号全局只增不减；被否决或废弃的 ADR 改为 Deprecated/Superseded，不回收编号。
- 重命名或删除文档前，必须用 `grep` 校验并更新全部引用（含 `docs/README.md`、`architecture_overview.md`、`AGENTS.md`、其他设计文档）。

## 3. 交叉引用规则

| 方向 | 规则 | 示例 |
|---|---|---|
| 文档 → 文档 | 具名相对链接；跨文档禁止"见上文/见前文"，必须给出文件链接 | `[ModelLoader 模块设计](../designs/model/01-model-loader.md)` |
| 同文档内 | 锚点引用 `#章节编号-锚点` | `[并发模型](#4-并发模型)` |
| 章节引用 | `<file>.md#<章节编号-锚点>`，编号是跨 PR/评审/ADR 引用的稳定标识 | `architecture_overview.md#4-核心模块职责` |
| 文档 → 代码 | 符号名反引号包裹 + 可点击路径链接；接口表与头文件符号完全同名 | ``ModelCompiler::Compile``、[src/compiler/model_compiler.cpp](../../src/compiler/model_compiler.cpp) |
| 代码 → 文档 | 头文件 file-level 注释追加 `/// @see docs/designs/<module>/NN-*.md`；实现文件可用 `// See docs/...` | `/// @see docs/designs/compiler/01-lowering.md` |

**单一事实源**：同一信息只在一个文档详述，其余文档引用链接。已确立的事实源：

| 事实 | 唯一来源 |
|---|---|
| 分层架构/硬性约束/性能基线 | `docs/designs/architecture/architecture_overview.md` |
| 模块边界与依赖方向 | `AGENTS.md` §2.1（设计文档引用不复制） |
| API 语义 | 头文件 Doxygen 注释 与 `docs/api/public-api.md` 同源同步 |
| 构建选项与运行配置 | 根 `README.md` |
| 产品范围与验收标准 | `docs/products/aethermind_prd.md` |
| 术语 | `docs/README.md` 术语表 |

### 3.3 与 docs/agent 互链

- `docs/decisions/NNNN-*.md` 的"关联"节注明对应 agent 记忆 ADR（`docs/agent/memory/modules/<module>/adrs/ADR-XXX.md`，如有）。
- agent 记忆 ADR 更新时同样反向引用人工 ADR。
- 两套编号体系各自独立、互不合并。

## 4. 质量标准

| 维度 | 标准 | 检查方式 |
|---|---|---|
| 完整性 | 模板章节齐全；设计文档覆盖该模块全部公共头文件符号；ADR 含被否定方案；issues 条目含背景/方案/状态 | 对照 `docs/templates/` + grep 头文件符号 |
| 准确性 | 只写已验证事实；状态字段正确（不把计划当已实现）；无过期数据；接口表与头文件签名一致 | 抽查代码 + 状态字段检查 |
| 可读性 | 单文档 ≤ 1000 行（超出拆分子节或子文档）；跨层流程用图；表格优先于长段落 | 行数检查 + 人工审读 |
| 一致性 | 术语与术语表一致；命名符合 §2；模板遵循度；链接全部有效 | `tools/verify_docs.py` + 人工审读 |

## 5. 评审流程

**文档变更 PR 检查清单**（新增或修改文档必过）：

- [ ] 遵循对应模板（`docs/templates/`）
- [ ] 头部状态与元数据（状态/日期/关联代码）正确
- [ ] `docs/README.md` 索引已同步（新增文档必须登记）
- [ ] 全部相对链接有效（可运行 `tools/verify_docs.py`）
- [ ] 接口符号与头文件一致（grep 验证）
- [ ] 命名符合 §2 规则；术语符合术语表

**代码变更 PR 检查清单**：按 §6.1 触发矩阵核对是否命中，命中项必须在本 PR 或显式标注的跟进 PR 中更新对应文档（code_review_guide.md 快速门禁执行）。

**定期审计**：每季度或每次大版本发布前，对照触发矩阵执行漂移检查；过期文档立即更新或标记 Deprecated 迁入 `docs/archive/`（不删除，保留历史与链接）。

## 6. 维护与更新流程

### 6.1 变更触发矩阵

| 变更类型 | 必须同步更新的文档 |
|---|---|
| 公共 API 变更（签名/语义） | 头文件 Doxygen、`api/public-api.md`、README.md API 章节、CHANGELOG；重大决策另建 ADR |
| 模块内部实现变更 | 对应 `designs/<module>/NN-*.md`、`architecture_overview.md` 相关章节 |
| 新增/删除模块 | `docs/README.md` 索引、`architecture_overview.md` 模块地图、新建/删除模块设计文档 |
| 配置变更（构建选项/环境变量） | README.md、相关模块设计文档配置章节、CHANGELOG |
| 并发契约/内存序/invariant 变更 | `architecture_overview.md` 硬性约束节、模块设计并发模型节、ADR（按需） |
| 性能基线变化 | `architecture_overview.md` 性能章节、CHANGELOG |
| 编码/注释/测试规范变更 | 对应 `guides/*` + AGENTS.md §7–§9 双向同步 |
| 缺陷修复 | `issues.md` 勾选条目、CHANGELOG（行为可见时） |

### 6.2 防过期机制

1. **同 PR 同步原则**：文档更新与代码变更同一 PR 提交；无法同 PR 时必须显式记录跟进任务并关联。
2. **状态字段**：Deprecated/Superseded 明确标记并迁入 `docs/archive/`（保留路径重定向说明），不直接删除。
3. **代码事实优先**：文档与代码冲突时，以经过验证的仓库事实为准并立即修正文档。
4. **漂移检测**：`tools/verify_docs.py` 校验相对链接有效性、设计文档接口符号在头文件中的存在性、索引覆盖完整性、状态字段合法性。

## 7. 文档类型写作要点

### 7.1 模块设计文档

从 `docs/templates/module-design.md` 复制填充，存放于 `docs/designs/<module>/`。必须覆盖：背景与目标、职责与边界、关键数据结构、并发模型、接口定义、算法与流程、边界条件与错误处理、风险与权衡、测试要点、变更记录。接口定义表与头文件 Doxygen 完全同名同义。

### 7.2 架构总览

`architecture_overview.md` 是全系统唯一权威总览，必须包含：分层架构图（mermaid）、模块地图（与 AGENTS.md §2.1 对齐并标注实现状态）、硬性约束清单、跨层数据流、性能基线、架构级并发模型、术语指向 README.md 术语表。其他文档引用其章节而非复制内容。

### 7.3 API 参考

- 头文件 Doxygen 注释是语义事实源；`docs/api/public-api.md` 是人工可读汇总，二者**同源同步**：修改头文件注释必须同步本文件，反之亦然。
- 格式：按函数分节，每函数一张表：`简介(@brief) / 参数(@param) / 返回值(@return) / 语义(@note) / 前置条件(@pre)`，与 `docs/guides/cpp_comment_guidelines.md` 强制标签一一对应；附使用示例与"语义约束"总节。
- 构建选项/环境变量以根 `README.md` 为单一事实源，本文件只引用不复制。

### 7.4 演进提案

- `docs/improvement-plan/README.md` 为入口：编号专题目录表 + 推荐阅读路径 + 各专题状态。
- 专题文档章节规范：现状分析 / 目标架构 / 方案与备选 / 实施步骤 / 风险与依赖 / 验收标准；状态 Draft → In Progress → Implemented → Superseded。

### 7.5 变更记录（三层）

| 层 | 位置 | 记录内容 |
|---|---|---|
| ADR | `docs/decisions/NNNN-*.md` | 重大架构决策及其理由、被否定方案（长期参考） |
| 设计文档变更记录表 | 各 `NN-*.md` §10 | 模块级变更轨迹（日期/变更/原因/关联 PR/ADR） |
| CHANGELOG.md | 仓库根 | 行为可见变更（API/配置/性能/缺陷修复），按语义化版本 |

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-21 | 初始版本：文档系统分类、命名、交叉引用、质量、维护规则成文 |
