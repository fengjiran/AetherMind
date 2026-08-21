# ADR-0001: 文档系统落地决策

- **状态**: Accepted
- **日期**: 2026-08-21
- **作者**: AetherMind Team
- **关联代码**: `docs/README.md`、`docs/guides/documentation-guide.md`、`docs/templates/`、`tools/verify_docs.py`、`CHANGELOG.md`

## 背景

- `docs/designs/` 下 30+ 文档混杂设计、计划、checklist、评审与临时文件，命名无统一规范，无索引与术语表，无法回答"某类文档在哪、怎么写、状态如何"。
- 需要一套文档分类、模板、命名、交叉引用、质量与维护规范，与 ammalloc 代码库已验证的文档系统（编号设计文档、ADR、verify 脚本）对齐。
- AetherMind 已存在 `docs/agent/` 记忆子系统（含模块级 ADR），必须与其共存而非冲突。

## 决策

- 以 ammalloc 文档系统为蓝本建立 AetherMind 文档系统：`docs/README.md`（索引 + 术语表）为唯一入口；`docs/guides/documentation-guide.md` 为规范成文；`docs/templates/`（module-design.md / adr.md）为新建模板；`tools/verify_docs.py` 做链接/符号/索引漂移检测。
- 模块设计按 `docs/designs/<module>/NN-*.md` 组织，模块子目录对齐 AGENTS.md §2.1 模块表；`architecture_overview.md` 为全系统唯一架构总览。
- 人类 ADR 新建 `docs/decisions/NNNN-*.md`，与 `docs/agent/memory/modules/<module>/adrs/` 的 agent 记忆 ADR 并存，双向链接、编号互不合并。
- 存量文档渐进式迁移：临时文件删除，过期文档标记 Deprecated 迁入 `docs/archive/`，有效文档归位重命名；迁移分批执行，每批同步索引。
- 设计文档只描述已验证实现；草稿/提案放 `docs/improvement-plan/`。

## 权衡

| 备选方案 | 结论 | 原因 |
|---|---|---|
| 存量文档全量一次性迁移 | 否 | 30+ 文档逐篇审阅改写工作量大且阻塞规范生效；渐进式批次迁移同样可达终态，风险更低 |
| 沿用 agent memory ADR 作为唯一 ADR 载体 | 否 | agent 记忆面向恢复工作流，生命周期与人类长期参考不一致；`docs/decisions/` 面向人类读者，职责互补 |
| 存量文档不动、只规范新增 | 否 | `docs/designs/` 继续混乱，索引无法建立，可发现性目标落空 |
| 按 ammalloc 模式落地（采纳） | 是 | 同仓库已验证的成熟模式，模板/脚本可直接适配，学习成本最低 |

## 后果

- 正面：文档可发现性、可追溯性、防过期能力建立；与 ammalloc 文档风格统一。
- 负面：迁移期索引含"待迁移"标记，`docs/designs/` 短暂双轨运行；verify 脚本对目录级引用豁免，迁移完成后需收紧。
- 影响文档：`docs/README.md`、`docs/guides/documentation-guide.md`、`AGENTS.md`（§7–§9 与指南双向同步）。

## 关联

- 规范：[docs/guides/documentation-guide.md](../guides/documentation-guide.md)
- 模板：[docs/templates/module-design.md](../templates/module-design.md) / [docs/templates/adr.md](../templates/adr.md)
- 迁移规则：[docs/archive/README.md](../archive/README.md)
- 对应 agent 记忆：无（本决策属于项目级文档基础设施，不绑定单一模块）。
