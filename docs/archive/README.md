# AetherMind 文档归档区

> 本目录存放标记为 Deprecated / Superseded / 历史快照的文档。归档文档**不删除**，保留历史可追溯；其内容以当前生效文档为准。

## 归档索引

| 文档 | 原位置 | 归档日期 | 被取代/废弃原因 | 当前权威文档 |
|---|---|---|---|---|
| [aethermind_arch_design.md](aethermind_arch_design.md) | `docs/designs/aethermind_arch_design.md` | 2026-08-21 | 早期架构设计；有效内容（并发模型/内存架构/确定性策略/冻结决策）已并入架构总览 | [architecture_overview.md](../designs/architecture/architecture_overview.md) |
| [model_loader/](model_loader/)（4 篇） | `docs/designs/model_loader/` | 2026-08-21 | 早期加载/prepack 设计，已被当前实现取代（`ModelLoader::Load` 返回 `LoadedModel`，prepack 拆分；文件头部均有历史警示） | [architecture_overview.md](../designs/architecture/architecture_overview.md) 第三章 / PRD |
| [kernel_dev/](kernel_dev/)（8 篇） | `docs/designs/kernel_dev/` | 2026-08-21 | 历史算子设计方案（已实现或被取代）、审查清单与开发记录；契约类文档保留于 `docs/designs/kernel_dev/` 待归位 | [kernel_dev 契约](../designs/kernel_dev/) |
| [designs-legacy/](designs-legacy/)（14 篇） | `docs/designs/` 顶层 | 2026-08-21 | 历史计划/评审快照（phase1_*、backend_phase1_*、*_review_and_minimal_fix_plan、路线图等），均已完成或过时 | [architecture_overview.md](../designs/architecture/architecture_overview.md) |

## 迁移规则

1. 归档前在文档头部添加 Deprecated 标记与重定向说明。
2. 在 `docs/README.md` 索引中更新对应条目状态。
3. 归档动作不得触碰 `docs/agent/`（独立子系统）。
