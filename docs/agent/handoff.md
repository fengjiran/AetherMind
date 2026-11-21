# 上轮文档整理记录

> 当前阅读入口见 [文档导航](../README.md)，在研工作见 [路线图](../roadmap/README.md)。

## 已完成

- 删除陈旧 agent 记忆、旧 handoff、agent ADR、模板与治理脚本；`docs/agent/` 仅保留 `memory/project.md` 与本文件。
- 建立 `docs/README.md`、`docs/architecture.md`、`docs/product.md`，把分散设计归并为模块文档；`reviews/tests/logs/decisions/templates` 与废弃指南移入 `docs/archive/`。
- 以 [项目手册](../handbook/README.md) 为入口重建实现文档，退役按模块编号的旧页面；扩展 `tools/check_docs_links.py` 覆盖根入口与标题锚点。
- 收敛 `组件速查`：删除 267 个碎片页与重复的嵌套目录，合并为单页 [组件速查](../handbook/组件速查.md)；同步修正 `docs/modules/`、`docs/designs/` 的残留引用。

## 当前目录结构

```
README.md / AGENTS.md / CHANGELOG.md
docs/
  README.md  architecture.md  product.md  roadmap.md  issues.md
  handbook/   # 实现文档主入口：概览、核心模块、API、算子、测试、部署、组件速查
  guides/  api/  operators/
  roadmap/    # 优先事项与提案
  archive/    # 冷存储，不索引、不参与链接检查
  agent/      # project.md + handoff.md
```
