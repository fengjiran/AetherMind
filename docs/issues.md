# AetherMind 问题跟踪

> 已知缺陷与优化待办清单，短生命周期条目。长生命周期/重大方向见 [decisions/](decisions/) 与 [improvement-plan/](improvement-plan/)。
>
> 条目格式：`- [ ] <背景与方案简述>（关联：文件/PR/issue 链接）`。修复完成后勾选并在 CHANGELOG 登记（行为可见时）。

## 未解决

- [ ] `HfModelValidator` 接受 `gelu/relu`，但 `ModelGraphBuilder::BuildLlamaDense` 固定生成 `SiluMul`，存在 accepted config 被静默编译为错误 MLP 语义的风险；当前产品应收窄到 `silu`，或先补齐对应 operator/graph/kernel（关联：[系统能力演进路线图 §5.2](improvement-plan/06-system-capability-evolution-roadmap.md#52-p0统一接受的模型和实际语义)）
- [ ] `ElementwiseMul` operator inference 与 CPU kernel 支持 broadcast，但 `GraphOpBuilder::AddElementwiseMul` 拒绝不同完整 `TensorSpec`，三层语义不一致（关联：[系统能力演进路线图 §6.2](improvement-plan/06-system-capability-evolution-roadmap.md#62-p0修复-graphopbuilder-与-operator-semantic-漂移)）
- [ ] `WorkspaceRequirement::lifetime/reusable` 尚未影响 offset 规划，`PlanWorkspaceRequirements` 仍顺序累加全部 requirement（关联：[系统能力演进路线图 §13.3](improvement-plan/06-system-capability-evolution-roadmap.md#133-p1workspace-lifetime-真正生效)）
- [ ] activation arena 为所有 activation 顺序分配，尚无 producer/last-consumer liveness reuse，峰值接近全部 activation bytes 之和（关联：[系统能力演进路线图 §9.3](improvement-plan/06-system-capability-evolution-roadmap.md#93-p1activation-liveness-planning)）
- [ ] PRD 的单线程/OpenMP、compile-time dispatch/planning-time resolve、多 Session 验收/单 active KV slot 等表述需要按当前事实与产品目标重新裁决（关联：[系统能力演进路线图 §17.1](improvement-plan/06-system-capability-evolution-roadmap.md#171-当前需要裁决的漂移)）
- [ ] designs/graph_compilation_flow.md 与 reviews/graph_compilation_flow.md 同名冲突，易混淆权威来源（关联：[docs/designs/graph_compilation_flow.md](designs/graph_compilation_flow.md)、[docs/reviews/graph_compilation_flow.md](reviews/graph_compilation_flow.md)）
- [ ] 3 篇非归档文档超长（>1000 行）：model_graph_design.md 1706、dispatch_design.md 1061、graph_lowering_design.md 1005（关联：[docs/designs/](designs/)）；~~operator_optimization_guide.md~~ 已于 2026-09-19 压缩合并入 [算子开发工作流附录 C](guides/operator-development-workflow.md) 并删除
- [ ] 大量文档无明确状态字段：designs/、guides/、reviews/ 中多数文档头部无状态标记（关联：[docs/designs/](designs/)、[docs/guides/](guides/)、[docs/reviews/](reviews/)）
- [ ] 4 篇 review 无日期命名：graph_compilation_flow.md、model_graph_data_structure_review.md、operator_semantic_layer_review.md、prd/prd_review.md（关联：[docs/reviews/](reviews/)）
- [ ] 24 篇 designs/ 命名违规（非 canonical `NN-<kebab-name>.md`）待迁移（关联：[docs/designs/](designs/)）
- [ ] GEMM G0 尚缺三项 production 门禁证据：裸机 perf/governor/microcode、稳定 5% 级性能门禁（WSL2 噪声 floor 在 binding/小形状超阈值）、raw artifact 的 CI/object-storage retention URL（关联：[GEMM 工作包状态](operators/gemm/README.md)、[配对 A/B 验证报告](operators/gemm/benchmarks/gemm_g0_paired_ab_validation_2026-09-18.md)）

## 已解决

- [x] 核心 9 篇文档中 `Phase 1 / Phase 2` 术语 83 处待分类迁移（修复：2026-09-17 全部迁移为具体 capability 表述；AGENTS.md/README.md/docs/README.md/prd/architecture_overview/public-api 均为 0 处，仅本文件保留元描述）
- [x] verify_docs.py 覆盖不足（修复：2026-09-14 实现 D5a 全部 7 项，389 行，report-only + --strict-* 框架）
- [x] designs/amstring/ 中 development_plan、milestones、task_checklist 分类违规（修复：2026-09-14 归档到 docs/archive/designs-legacy/）
- [x] 01 号方案 §2.2 kernel 覆盖表漂移：Linear、RoPE、Argmax 标为"无/阻塞"，实际存在 reference kernel（修复：2026-09-14 更新为 FP32 reference / 可用）
- [x] code_review_guide.md 失效命令与 ammalloc 残留（修复：2026-09-14 标 Deprecated，待重写；docs/README.md 索引同步）
- [x] docs/README.md 未登记 03 号方案：演进提案表只列到 01（修复：2026-09-14 新增 02/03 条目）
- [x] 03 号方案自相矛盾：Draft 提案自称治理权威、核心集含 Draft 提案、agent/ 纳入门禁、authority 作为类型、--no-* 语义矛盾、命名违规计数不准（修复：2026-09-14 版本 1.2 修正 6 条）
- [x] docs/issues.md 未解决条目为空，与已知缺口不一致（修复：2026-09-14 登记 6 条未解决 + 7 条已解决）
