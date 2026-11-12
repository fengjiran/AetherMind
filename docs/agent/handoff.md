# 当前工作 handoff

## 已完成

- Phase 0：删除 `docs/agent/` 下 ammalloc 模块记忆、旧 handoff、agent ADR、模板、prompts 与测试；删除 `tools/agent_memory/contract_check.py`、`tools/archive_old_handoffs.sh`、`tools/verify_docs.py`。仅保留 `memory/project.md`。
- Phase 1：重写 `docs/README.md`；将架构总览迁移为 `docs/architecture.md` 并修正 C ABI、Generate、W16/A32 与 KV16 状态；将 PRD 迁移为 `docs/product.md`；建立 `docs/modules/` 骨架。
- Phase 2：把 `docs/designs/` 的 23 篇内容文档合并为 `docs/modules/` 的 12 篇（foundation、amstring、shape-inference、graph、operators、compiler、execution、kv-cache、runtime、model、inference、backend）；源文档归档到 `docs/archive/designs/`；根 README、AGENTS、API/算子/指南中的活跃引用已切到模块文档与 `product.md`。

## 待办

- Phase 3：收敛 `docs/guides/`，建立 `docs/roadmap.md` 单文件在研方向入口，瘦身 `product.md`，处理 `docs/improvement-plan/` 的已实现/被取代提案。
- Phase 4：清理 `archive/` 与剩余历史目录引用，加入相对链接检查，删除过渡符号链接。

## 注意

- `docs/products/aethermind_prd.md` 仍是符号链接；`docs/designs/` 已移除，原设计文档只存在于 `docs/archive/designs/`。
- `docs/improvement-plan/`、`docs/reviews/`、`docs/tests/`、`docs/logs/`、`docs/decisions/`、`docs/templates/` 尚未迁移，仍可被旧文档引用，但不再是权威入口。
- 模块文档的事实来源是头文件与源码；发现模块文档与代码不一致时，以代码为准并更新模块文档。
