# 当前工作 handoff

## 已完成

- Phase 0：删除 `docs/agent/` 下 ammalloc 模块记忆、旧 handoff、agent ADR、模板、prompts 与测试；删除 `tools/agent_memory/contract_check.py`、`tools/archive_old_handoffs.sh`、`tools/verify_docs.py`。仅保留 `memory/project.md`。
- Phase 1：重写 `docs/README.md`；将架构总览迁移为 `docs/architecture.md` 并修正 C ABI、Generate、W16/A32 与 KV16 状态；将 PRD 迁移为 `docs/product.md`；建立 `docs/modules/` 骨架。

## 待办

- Phase 2：把 `docs/designs/` 的 24 篇按模块合并为 `docs/modules/` 的 12 篇。
- Phase 3：收敛 `docs/guides/`，建立单文件在研方向入口，瘦身 `product.md`。
- Phase 4：全量修复引用，加入相对链接检查，决定 `docs/archive/` 去留。

## 注意

- 旧路径 `docs/designs/architecture/architecture_overview.md` 与 `docs/products/aethermind_prd.md` 目前是指向新文件的符号链接，Phase 4 删除。
- 旧 `docs/designs/`、`docs/improvement-plan/`、`docs/reviews/`、`docs/tests/` 尚未迁移，仍可被引用，但不再是权威入口。
