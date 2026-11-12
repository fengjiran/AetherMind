# 当前工作 handoff

## 已完成

- Phase 0：删除陈旧 agent 记忆、旧 handoff、agent ADR、模板与治理脚本；仅保留 `memory/project.md`。
- Phase 1：建立 `docs/README.md`、`docs/architecture.md`、`docs/product.md` 与 `docs/modules/`。
- Phase 2：23 篇 `docs/designs/` 合并为 `docs/modules/` 12 篇；源文档归档。
- Phase 3：保留 9 篇常用指南；建立 `docs/roadmap.md`；`docs/product.md` 收敛到 479 行。
- Phase 4：`reviews/tests/logs/decisions/templates` 与废弃指南移入 `docs/archive/`；删除 PRD 符号链接；新增 `tools/check_docs_links.py`，活跃文档链接检查为 0 断链。

## 当前结构

```
README.md / AGENTS.md / CHANGELOG.md
docs/
  README.md  architecture.md  product.md  roadmap.md  issues.md
  modules/   guides/  api/  operators/
  improvement-plan/   # 按用户要求暂时保留
  archive/            # 冷存储，不索引、不参与链接检查
  agent/              # project.md + handoff.md
```

## 注意

- `docs/improvement-plan/` 仍包含已实现/被取代/在研提案；当前优先级以 `docs/roadmap.md` 为准。
- `docs/archive/` 只读、不维护；需要追溯历史设计或验证报告时再查阅。
- 任何新增或移动文档后运行 `python3 tools/check_docs_links.py`。
