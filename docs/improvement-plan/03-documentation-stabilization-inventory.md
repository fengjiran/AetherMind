# AetherMind 文档清查表（Batch -1 D2 交付物）

- **状态**: Draft
- **版本**: 1.4
- **日期**: 2026-09-17
- **主方案**: [03-documentation-stabilization.md](03-documentation-stabilization.md)
- **规范来源**: [文档系统规范](../guides/documentation-guide.md)

本表是 [03 号方案](03-documentation-stabilization.md) §5 D2 的一次性清查交付物，登记范围内所有 .md 文档的类型、状态、事实权威、当前性、问题、处置与优先级。**本清查阶段不批量重命名或移动文件**，"处置" 列仅记录建议值，实际执行放到 D3 及后续 workstream。

**当前状态（2026-09-17）**：本表为 D2 规划工件，已建立 143 行清查记录。治理范围 58 篇中：Current 16、Unverified 26、Historical Snapshot 10、Draft 4、Deprecated 1、Accepted 1。核心 9 篇当前性均为"已核验"（结构核验，见 [03 号方案 §4.4](03-documentation-stabilization.md)）；其余 26 篇 Unverified 均有处置+优先级值，满足修订后的 E2。D0 frontmatter 落地、D3 P1/P2 处置、D4 非核心术语迁移、D5b metadata schema 未启动。Batch -1 整体状态见 [03 号方案 §1.3](03-documentation-stabilization.md)。

## 1. 字段与取值

| 字段 | 取值 |
|---|---|
| 路径 | 相对仓库根的路径 |
| 类型 | design / guide / plan / review / test / archive / index / api / prd / handoff / memory / template / decision / log |
| 状态 | Current / Draft / Historical Snapshot / Superseded / Deprecated / Unverified / Out of Scope（未核验前统一取值，见 [03 号方案 §3.2](03-documentation-stabilization.md)） |
| 事实权威 | 是（注明类别）/ 否 |
| 当前性 | 已核验 / 部分核验 / 未核验 |
| 问题 | 漂移 / 重复 / 分类错误 / 过长 / 术语过时 / 命名违规 / 失效命令 / 无 frontmatter / `-` |
| 处置 | 保留 / 修订 / 拆分 / 移动 / 归档 / 待 D3 排序确认 |
| 优先级 | P0 / P1 / P2（映射规则见 [03 号方案 §6](03-documentation-stabilization.md)） |

**首次填写规则**：状态未核验时统一填 `Unverified`（不填 `Current`）；事实权威只对 [03 号方案 §3.1](03-documentation-stabilization.md) 表中列出的 10 类事实来源填"是"；当前性首次填 `未核验`；问题依据已核验事实与结构化证据填写；处置注明"待 D3 排序确认"。

## 2. 清查范围

- **包含（治理范围）**：仓库根 `README.md`、`AGENTS.md`、`CHANGELOG.md`；`docs/` 下全部 .md（含 `archive/`、`improvement-plan/`、`templates/`）。
- **包含（仅登记，Out of Scope）**：`docs/agent/` 下全部 .md——按 [documentation-guide.md §1](../guides/documentation-guide.md) 准入条件，`docs/agent/` 是独立子系统。清查表为全景统计列出，但排除在 E1–E9、状态迁移、命名和 metadata 门禁之外。
- **排除**：`3rdparty/`、`build*/`、`.models/`、`.qoder/`、`.trae/`、`.omo/`、`.sisyphus/`、`.codex/`、`.gemini/`、`.idea/`、`.vscode/`、`.cache/`、`.agents/`、`tools/agent_memory/`。
- **总数**：143 篇（根 3 + `docs/` 140）；其中治理范围 89 篇，Out of Scope 54 篇（`docs/agent/`）。

**docs/agent/ 覆盖规则**：清查表中所有 `docs/agent/` 路径的行统一适用以下覆盖，无需逐行修改：
- 状态 → `Out of Scope`
- 优先级 → `-`（不参与 E1–E9 门禁）
- 处置 → 保留（agent 独立子系统）
- 当前性 → 不适用

## 3. 清查表

按路径字典序排列，每篇文档一行。

| 路径 | 类型 | 状态 | 事实权威 | 当前性 | 问题 | 处置 | 优先级 |
|---|---|---|---|---|---|---|---|
| AGENTS.md | guide | Current | 是（模块 ownership） | 已核验 | - | 保留（D1 核验） | P0 |
| CHANGELOG.md | log | Current | 否 | 部分核验 | 内容极少（22 行） | 保留 | P2 |
| README.md | index | Current | 是（构建选项） | 已核验 | - | 保留（D1 核验） | P0 |
| docs/README.md | index | Current | 是（文档索引 + 术语表） | 已核验 | 术语已迁移（2026-09-17） | 保留 | P0 |
| docs/agent/decisions/template.md | template | Unverified | 否 | 未核验 | 独立子系统，本规范不适用 | 保留 | P2 |
| docs/agent/handoff/README.md | index | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260313T163000Z--ses_31a1b709effemOwSr0RspyMwiV--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留（agent 独立子系统） | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260314T100000Z--ses_3137f3e4bffeqJcbXrLRnYVMb3--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260314T222700Z--ses_318bd5c17ffeP3CBYxVYXTJuAj--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260318T090000Z--ses_3001f7ddcffe764XhsH3T9LQqc--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260318T172004Z--ses_2fe5b0754ffeG8dQuKNwBedMqj--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260319T064417Z--ses_2fe5b0754ffeG8dQuKNwBedMqj--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_allocator/20260319T131159Z--ses_2fe5b0754ffeG8dQuKNwBedMqj--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_cache/20260316T120000Z--ses_page_cache_v2--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_cache/20260316T171857Z--ses_page_cache_complete--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_cache/20260316T183731Z--ses_page_cache_review--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__page_cache/20260317T092600Z--ses_305b2bc9fffeJhe1zQSYl4MU0a--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__size_class/20260311T173411Z--ses_32289f06dffeW3U7ptRE7LLU0z--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__size_class/20260313T100000Z--ses_current--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库；文件名含 `current` 但已过期 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__size_class/20260324T103236Z--ses_2e617b00affe5Tx2X40IXVEfUJ--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__thread_cache/20260311T103000Z--ses_example_001--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__thread_cache/20260323T125544Z--ses_ammalloc_tc_20260323--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/ammalloc__thread_cache/20260323T165526Z--ses_ammalloc_tc_code_review--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/handoff/workstreams/project__agent-memory-v1.1/20260312T060000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | - | 保留 | P2 |
| docs/agent/handoff/workstreams/project__agent-memory-v1.1/20260312T080000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | - | 保留 | P2 |
| docs/agent/handoff/workstreams/project__docs-reorg/20260311T180000Z--ses_docs_reorg--sisyphus.md | handoff | Historical Snapshot | 否 | 未核验 | 与本文档主题相关，可作历史参考 | 保留 | P2 |
| docs/agent/memory/QUICKSTART.md | memory | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/memory/README.md | memory | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/memory/mainmodule_memory_template.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-001.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-002.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-003.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-004.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-005.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-006.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/adrs/ADR-007.md | decision | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/module.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/central_cache.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/page_allocator.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/page_cache.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/page_heap_scavenger.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/size_class.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/spin_lock.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/modules/ammalloc/submodules/thread_cache.md | memory | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |
| docs/agent/memory/project.md | memory | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/memory/submodule_memory_template.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/memory_system.md | memory | Unverified | 否 | 未核验 | 独立子系统架构参考 | 保留 | P2 |
| docs/agent/prompts/README.md | index | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/prompts/generate_module_memory.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/prompts/handoff_style_guide.md | guide | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/prompts/handoff_template.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/prompts/memory_update_and_adr.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/prompts/new_session_template.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/prompts/quick_resume.md | template | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/tests/README.md | index | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/tests/memory_system_automation_plan.md | plan | Unverified | 否 | 未核验 | 独立子系统 | 保留 | P2 |
| docs/agent/tests/memory_system_automation_plan_initial.md | plan | Historical Snapshot | 否 | 未核验 | 独立子系统；文件名含 `initial` 提示历史版本 | 保留 | P2 |
| docs/agent/tests/memory_system_test_suite.md | test | Unverified | 否 | 未核验 | 独立子系统；953 行接近 1000 行阈值 | 保留 | P2 |
| docs/api/public-api.md | api | Current | 是（Public API 汇总） | 已核验 | - | 保留（D1 核验） | P0 |
| docs/archive/README.md | index | Deprecated | 否 | 未核验 | 归档区索引 | 保留 | P2 |
| docs/archive/aethermind_arch_design.md | archive | Deprecated | 否 | 未核验 | 有效内容已并入 architecture_overview | 保留 | P2 |
| docs/archive/designs-legacy/amstring_design_and_execution_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/any_review_and_minimal_fix_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/array_review_and_minimal_fix_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/backend_phase1_development_steps.md | archive | Deprecated | 否 | 未核验 | 术语过时（`phase1`），历史语境保留 | 保留 | P2 |
| docs/archive/designs-legacy/backend_phase1_implementation_checklist.md | archive | Deprecated | 否 | 未核验 | 术语过时（`phase1`） | 保留 | P2 |
| docs/archive/designs-legacy/backend_phase1_implementation_plan.md | archive | Deprecated | 否 | 未核验 | 术语过时（`phase1`） | 保留 | P2 |
| docs/archive/designs-legacy/dispatch_implementation_task_list.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/object_review_and_minimal_fix_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/phase1_development_plan_draft.md | archive | Deprecated | 否 | 未核验 | 术语过时（`phase1`） | 保留 | P2 |
| docs/archive/designs-legacy/phase1_implementation_breakdown.md | archive | Deprecated | 否 | 未核验 | 术语过时（`phase1`） | 保留 | P2 |
| docs/archive/designs-legacy/phase1_m1_execution_checklist.md | archive | Deprecated | 否 | 未核验 | 术语过时（`phase1`） | 保留 | P2 |
| docs/archive/designs-legacy/string_review_and_minimal_fix_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/tensor_review_and_minimal_fix_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/designs-legacy/从0到1流程打通路线图.md | archive | Deprecated | 否 | 未核验 | 中文文件名（归档区保留） | 保留 | P2 |
| docs/archive/kernel_dev/LinearOp算子设计与实现方案_v1.0.md | archive | Deprecated | 否 | 未核验 | 中文文件名 | 保留 | P2 |
| docs/archive/kernel_dev/Operator语义层接口实施步骤_v1.0.md | archive | Deprecated | 否 | 未核验 | 过长（2020 行），归档区不强制拆分 | 保留 | P2 |
| docs/archive/kernel_dev/RoPEOp算子设计与实现方案_v1.0.md | archive | Deprecated | 否 | 未核验 | 过长（1086 行） | 保留 | P2 |
| docs/archive/kernel_dev/Step_A4_虚函数Operator体系实施方案_v1.0.md | archive | Deprecated | 否 | 未核验 | 中文文件名 | 保留 | P2 |
| docs/archive/kernel_dev/rmsnorm_contract_vs_impl_review_notes.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/kernel_dev/rmsnorm_development_notes.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/kernel_dev/rmsnorm_kernel_performance_review.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/kernel_dev/高性能算子开发策略设计文档_v1.0.md | archive | Deprecated | 否 | 未核验 | 过长（1748 行），中文文件名 | 保留 | P2 |
| docs/archive/model_loader/model_loader_design.md | archive | Deprecated | 否 | 未核验 | 已被 designs/model/01-model-loader.md 取代 | 保留 | P2 |
| docs/archive/model_loader/model_loader_implementation_plan.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/archive/model_loader/model_validator_design.md | archive | Deprecated | 否 | 未核验 | 过长（1330 行） | 保留 | P2 |
| docs/archive/model_loader/model_weight_packing_strategy.md | archive | Deprecated | 否 | 未核验 | - | 保留 | P2 |
| docs/decisions/0001-documentation-system.md | decision | Accepted | 否 | 部分核验 | 唯一 ADR，与 documentation-guide.md 同步 | 保留 | P1 |
| docs/designs/amstring/BasicStringCore_design.md | design | Unverified | 否 | 未核验 | 命名违规（无 NN- 前缀） | 待 D3 排序确认（重命名） | P2 |
| docs/designs/amstring/CharLayoutPolicy_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认（重命名） | P2 |
| docs/designs/amstring/GenericLayoutPolicy_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认（重命名） | P2 |
| docs/archive/designs-legacy/amstring_development_plan.md | archive | Deprecated | 否 | 未核验 | 分类错误已修复（原位于 designs/amstring/，2026-09-14 归档） | 保留（归档） | P2 |
| docs/archive/designs-legacy/amstring_milestones.md | archive | Deprecated | 否 | 未核验 | 分类错误已修复（原位于 designs/amstring/，2026-09-14 归档） | 保留（归档） | P2 |
| docs/designs/amstring/amstring_policy_based_architecture_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认（重命名） | P2 |
| docs/designs/amstring/amstring_storage_architecture_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认（重命名） | P2 |
| docs/archive/designs-legacy/amstring_task_checklist.md | archive | Deprecated | 否 | 未核验 | 分类错误已修复（原位于 designs/amstring/，2026-09-14 归档） | 保留（归档） | P2 |
| docs/designs/architecture/architecture_overview.md | design | Current | 是（当前架构） | 已核验 | 术语已迁移（2026-09-17） | 保留（D1 核验） | P0 |
| docs/designs/backend_design.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 命名违规（应在 backend/NN-*.md） | 曾暂迁 operators，2026-09-19 回流 docs/designs/backend_design.md | Resolved |
| docs/designs/cpu_capability_design.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 命名违规 | 曾暂迁 operators，2026-09-19 回流 docs/designs/cpu_capability_design.md | Resolved |
| docs/designs/dispatch_design.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 过长（1061 行）+ 命名违规 | 曾暂迁 operators，2026-09-19 回流 docs/designs/dispatch_design.md | Resolved |
| docs/designs/executor_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认 | P2 |
| docs/designs/graph_compilation_flow.md | design | Unverified | 否 | 未核验 | 命名违规 + 与 docs/reviews/graph_compilation_flow.md 同名冲突 | 待 D3 排序确认（重命名或移动） | P1 |
| docs/designs/graph_invariants_and_validator_architecture.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认 | P2 |
| docs/designs/graph_lowering_design.md | design | Unverified | 否 | 未核验 | 过长（1005 行）+ 命名违规 | 待 D3 排序确认（拆分 + 移动） | P2 |
| docs/designs/kernel_dev/LinearOp算子契约.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 中文文件名 + 命名违规 | 已归入 docs/operators/linear/LinearOp算子契约.md | Resolved |
| docs/designs/kernel_dev/RMSNorm算子契约.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 中文文件名 + 命名违规 | 已归入 docs/operators/rmsnorm/RMSNorm算子契约.md | Resolved |
| docs/designs/kernel_dev/RoPE算子契约.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 中文文件名 + 命名违规 | 已归入 docs/operators/rope/RoPE算子契约.md | Resolved |
| docs/designs/kernel_dev/算子系统设计.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 中文文件名 + 命名违规 | 已归入 docs/operators/算子系统设计.md | Resolved |
| docs/designs/kv_cache_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认 | P2 |
| docs/designs/model/01-model-loader.md | design | Unverified | 否 | 未核验 | -（canonical 命名） | 保留（D1 核验后决定是否升级为权威） | P1 |
| docs/designs/model_graph_design.md | design | Unverified | 否 | 未核验 | 过长（1706 行）+ 命名违规 | 待 D3 排序确认（拆分 + 移动） | P2 |
| docs/designs/op_evaluator.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 命名违规 | 曾暂迁 operators，2026-09-19 回流 docs/designs/op_evaluator.md | Resolved |
| docs/designs/operator_contract_design.md | design | **Moved**（2026-09-19） | 否 | 已处理 | 命名违规 | 已归入 docs/operators/operator_contract_design.md | Resolved |
| docs/designs/status设计方案.md | design | Unverified | 否 | 未核验 | 中文文件名 + 命名违规 | 待 D3 排序确认（重命名） | P2 |
| docs/designs/tensor_view_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认 | P2 |
| docs/designs/unified_allocator_design.md | design | Unverified | 否 | 未核验 | 命名违规 | 待 D3 排序确认 | P2 |
| docs/designs/已证明约束的执行阶段保障方案.md | design | Unverified | 否 | 未核验 | 中文文件名 + 命名违规 | 待 D3 排序确认（重命名） | P2 |
| docs/guides/code_review_guide.md | guide | Deprecated | 否 | 已核验 | 失效命令（第 49 行）+ ammalloc 专项残留（5 处）+ 硬阈值无依据（第 26 行）+ 与 AGENTS.md §6 不一致 | 已标 Deprecated（2026-09-14），待重写 | P0 |
| docs/guides/cpp_coding_style_guidelines.md | guide | Current | 否 | 部分核验 | - | 保留 | P1 |
| docs/guides/cpp_comment_guidelines.md | guide | Current | 否 | 部分核验 | - | 保留 | P1 |
| docs/guides/documentation-guide.md | guide | Current | 是（文档规范元） | 已核验 | - | 保留（D1 核验） | P0 |
| docs/guides/operator_optimization_guide.md | guide | **Removed**（2026-09-19） | 否 | 已处理 | 过长（2110 行） | 已压缩合并入算子开发工作流附录 C 后删除 | Resolved |
| docs/guides/test_writing_guidelines.md | guide | Current | 否 | 部分核验 | - | 保留 | P1 |
| docs/improvement-plan/01-inference-session-generate-readiness.md | plan | Draft | 否 | 部分核验 | 漂移已修复（§2.2 Linear/RoPE/Argmax 更新为 FP32 reference / 可用，2026-09-14） | 保留（其余章节待核验） | P0 |
| docs/improvement-plan/02-engineering-quality-system.md | plan | Draft | 否 | 未核验 | Workstream A 与 Batch 0 含文档治理项，需移交本方案 | 修订（本 Plan 已处理） | P0 |
| docs/improvement-plan/03-documentation-stabilization-inventory.md | plan | Draft | 否 | 未核验 | 本文档 | 保留 | P0 |
| docs/improvement-plan/03-documentation-stabilization.md | plan | Draft | 否 | 未核验 | 本方案主文档（Draft 提案，不是治理权威） | 保留（Batch -1 完成后迁移到 Implemented） | P0 |
| docs/improvement-plan/README.md | index | Current | 是（演进提案索引） | 已核验 | - | 保留 | P0 |
| docs/issues.md | index | Current | 是（已知问题） | 已核验 | - | 保留 | P0 |
| docs/logs/development_log.md | log | Current | 否 | 部分核验 | 追加式过程记录，696 行 | 保留 | P2 |
| docs/products/aethermind_prd.md | prd | Current | 是（产品能力） | 已核验 | 术语已迁移（2026-09-17） | 保留（D1 核验） | P0 |
| docs/reviews/graph_compilation_flow.md | review | Historical Snapshot | 否 | 未核验 | 命名违规（无日期）+ 与 designs/ 同名冲突 + docs/README.md 已标"已过时" | 重命名加日期（D3 P1） | P1 |
| docs/reviews/model_graph_data_structure_review.md | review | Historical Snapshot | 否 | 未核验 | 命名违规（无日期）+ docs/README.md 标 Current 但无快照日期 | 重命名加日期（D3 P1） | P1 |
| docs/reviews/operator_kernel_architecture_review_2026-07-18.md | review | Historical Snapshot | 否 | 未核验 | -（canonical 命名） | 保留 | P2 |
| docs/reviews/operator_semantic_layer_review.md | review | Historical Snapshot | 否 | 未核验 | 命名违规（无日期） | 重命名加日期 | P1 |
| docs/reviews/prd/prd_review.md | review | Historical Snapshot | 否 | 未核验 | 命名违规（无日期） | 重命名加日期 | P1 |
| docs/templates/adr.md | template | Current | 否 | 部分核验 | - | 保留 | P2 |
| docs/templates/module-design.md | template | Current | 否 | 部分核验 | - | 保留 | P2 |
| docs/tests/ammalloc_benchmark_rigorous_20260303.md | test | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留（历史验证快照） | P2 |
| docs/tests/amstring_charlayout_m7_validation_20260429.md | test | Historical Snapshot | 否 | 未核验 | - | 保留 | P2 |
| docs/tests/amstring_m6_validation_20260428.md | test | Historical Snapshot | 否 | 未核验 | - | 保留 | P2 |
| docs/tests/amstring_sso_boundary_validation_20260502.md | test | Historical Snapshot | 否 | 未核验 | - | 保留 | P2 |
| docs/tests/size_class_benchmark_20260310.md | test | Historical Snapshot | 否 | 未核验 | ammalloc 已移出仓库 | 保留 | P2 |

## 4. 汇总统计

以下计数由本清查表 §3 直接汇总，与表内行数一致（合计 143）。

### 4.1 按类型

| 类型 | 数量 |
|---|---|
| design | 26 |
| archive | 27 |
| handoff | 20 |
| memory | 12 |
| template | 10 |
| plan | 9 |
| index | 8 |
| guide | 8 |
| decision | 8 |
| test | 6 |
| review | 5 |
| log | 2 |
| api | 1 |
| prd | 1 |
| **合计** | **143** |

### 4.2 按状态

| 状态 | 数量 |
|---|---|
| Out of Scope | 54（`docs/agent/` 覆盖规则，见 §2） |
| Deprecated | 32（archive/ 31 + code_review_guide 1） |
| Unverified | 26（治理范围内 designs/ 25 + guides/ 1，均有处置+优先级） |
| Current | 16（核心 9 篇 + guides 4 + templates 2 + CHANGELOG 1） |
| Historical Snapshot | 10（reviews/ 5 + tests/ 5） |
| Draft | 4（improvement-plan/ 01–03 + inventory） |
| Accepted | 1（decisions/0001） |
| **合计** | **143**（治理范围 58 + archive 31 + OOS 54） |

### 4.3 按优先级

| 优先级 | 数量 | 说明 |
|---|---|---|
| P0 | 14 | 权威事实源 + 已知漂移 + 失效门禁 |
| P1 | 13 | 分类错误 + 无日期 review + 命名冲突 |
| P2 | 62 | 归档 + 命名违规 + 超长（内容待核验） |
| OOS | 54 | `docs/agent/` 独立子系统，不参与门禁 |
| **合计** | **143** | |

### 4.4 按处置建议

| 处置 | 数量 |
|---|---|
| 保留（含 D1 核验后保留、历史快照保留等变体） | 105 |
| 待 D3 排序确认（重命名 / 移动 / 拆分） | 24 |
| 修订（Batch -1 内处理或本 Plan 已处理） | 6 |
| 重命名加日期（reviews/） | 4 |
| 移动到 improvement-plan/ 或归档（designs/ 中的 plan 类） | 3 |
| 待 D1 后核验决定是否拆分 | 1 |
| **合计** | **143** | |

## 5. 已知债务快照

以下条目在 [03 号方案 §2](03-documentation-stabilization.md) 中已核验，Batch -1 内必须处理或显著标记。锚点指向本清查表对应行。

| 债务 | 证据 | 清查表锚点 | 处置窗口 |
|---|---|---|---|
| `docs/issues.md` 未解决条目为空 | 第 9 行 `- （无）` | docs/issues.md 行 | Batch -1 P0 |
| 01 号方案 §2.2 Linear/RoPE/Argmax 状态漂移 | 实际存在 `src/backend/cpu/kernels/{linear,rope,argmax}/*_f32_reference.cpp` | docs/improvement-plan/01-* 行 | Batch -1 P0 |
| `code_review_guide.md` 失效命令 + ammalloc 残留 | 第 49 行 `make --build build`；第 32/52/199/216/312/482-698 行 ammalloc 专项 | docs/guides/code_review_guide.md 行 | Batch -1 P0 |
| `docs/designs/` 混入 development plan / milestones / task checklist | 已修复：3 篇归档到 `docs/archive/designs-legacy/`（2026-09-14） | 对应 3 行 | 已完成 |
| `graph_compilation_flow.md` 同名冲突 | designs/ 与 reviews/ 并存 | 对应 2 行 | Batch -1 P1 |
| 4 篇 review 无日期 | `graph_compilation_flow.md`、`model_graph_data_structure_review.md`、`operator_semantic_layer_review.md`、`prd/prd_review.md` | docs/reviews/ 对应 4 行 | Batch -1 P1 |
| 3 篇非归档文档超长（>1000 行）（原 4 篇，operator_optimization_guide.md 已于 2026-09-19 移除） | ~~`operator_optimization_guide.md` 2110~~、`model_graph_design.md` 1706、`dispatch_design.md` 1061、`graph_lowering_design.md` 1005 | 对应 3 行 | 后续 workstream（P2） |
| 24 篇 designs/ 命名违规 | 全部 29 篇中：canonical 1（`model/01-model-loader.md`）、存量例外 1（`architecture_overview.md`）、plan 分类错误 3、真正命名违规 24 | docs/designs/ 对应 24 行 | 后续 workstream（P2） |
| 术语过时（`Phase 1`） | `docs/README.md` 第 37/158 行；`docs/products/aethermind_prd.md`；architecture_overview 索引行 | 对应各行 | D4 术语迁移（Batch -1 内启动，后续完成） |
| `verify_docs.py` 覆盖不足 | 第 7 行 anchors 不校验；第 39 行 improvement-plan 排除；第 128-134 行 loose symbol 匹配；第 155-158 行目录级索引覆盖 | 不在清查表（工具而非文档） | D5a Batch -1 内完成 |

## 6. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-09-13 | 初始清查：144 篇文档，8 列字段，P0/P1/P2 优先级映射 |
| 1.1 | 2026-09-14 | 头部增加"当前状态"说明：本表为 D2 规划工件，D1/D3/D4/D5 未启动；指向 03 号方案 §1.3 快照 |
| 1.2 | 2026-09-14 | 移除 authority 类型（改为 guide/index/api/design/prd/plan）；状态枚举增加 Out of Scope；§2 增加 docs/agent/ 覆盖规则（排除 E1–E9 门禁）；§4 统计重算 |
| 1.3 | 2026-09-17 | 核心 9 篇置 Current、4 篇 review 转 Historical Snapshot、重算状态计数；与 03 号方案 v2.0/2.1 快照同步 |
| 1.4 | 2026-09-17 | 删除 superseded 草案 `docs/designs/kernel_dev/CPU_FP32_GEMM优化方案.md`：移除对应行并重算 §4/§5 统计（144→143 行、治理 59→58、Unverified 27→26、命名违规 25→24） |
