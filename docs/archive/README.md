# AetherMind 文档归档区

> 归档区只读、不维护，不作为事实来源。当前权威入口见 [docs/README.md](../README.md)。

## 设计文档归档

`designs/` 保存原 `docs/designs/` 的 23 篇设计文档。它们已按模块合并为 `docs/modules/` 的 12 篇，原文仅用于追溯：

- 架构与模块设计：[docs/modules/](../modules/)
- 架构总览：[docs/architecture.md](../architecture.md)
- 产品范围：[docs/product.md](../product.md)

## 过程资料归档

| 目录 | 内容 | 当前权威 |
|---|---|---|
| [reviews/](reviews/)（5 篇） | 历史设计/代码评审快照 | `docs/modules/` 与 `docs/issues.md` |
| [tests/](tests/)（20 篇） | 历史验证报告与基准记录 | `CHANGELOG.md` 与 `docs/operators/*/benchmarks/` |
| [logs/](logs/)（1 篇） | 开发过程日志 | git history |
| [decisions/](decisions/)（2 篇） | 旧 ADR；关键结论已并入模块文档 | `docs/modules/` 与 `docs/README.md` |
| [templates/](templates/)（2 篇） | 旧模块/ADR 模板 | `docs/README.md` 的写作规则 |
| [guides/](guides/)（2 篇） | 已废弃的文档规范与审查指南 | `docs/README.md` |

## 其他历史资料

| 文档 | 原位置 | 说明 | 当前权威文档 |
|---|---|---|---|
| [aethermind_arch_design.md](aethermind_arch_design.md) | `docs/designs/aethermind_arch_design.md` | 早期架构设计；有效内容已并入架构总览 | [architecture.md](../architecture.md) |
| [model_loader/](model_loader/)（4 篇） | `docs/designs/model_loader/` | 早期加载/prepack 设计，已被当前实现取代 | [modules/model.md](../modules/model.md) |
| [kernel_dev/](kernel_dev/)（8 篇） | `docs/designs/kernel_dev/` | 历史算子设计方案；现行契约见 `docs/operators/` | [operators/](../operators/) |
| [designs-legacy/](designs-legacy/)（14 篇） | `docs/designs/` 顶层 | 历史计划/评审快照 | [architecture.md](../architecture.md) |

## 规则

1. 归档文档不修改、不索引、不参与链接检查。
2. 过时文档移入本目录；活跃目录不保留历史副本。
3. `docs/agent/` 不属于本归档区。
