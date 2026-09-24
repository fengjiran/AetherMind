# AetherMind 演进提案索引

> 本目录存放未来架构演进提案（"将来要做什么"）。写作与命名规范见 [文档系统规范](../guides/documentation-guide.md#74-演进提案)。
>
> 专题文档章节规范：现状分析 / 目标架构 / 方案与备选 / 实施步骤 / 风险与依赖 / 验收标准。
> 状态流转：Draft → In Progress → Implemented → Superseded。

## 专题目录

| 编号 | 专题 | 状态 | 最后更新 |
|---|---|---|---|
| 01 | [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md) | Implemented | 2026-09-24 |
| 02 | [工程质量体系建设方案](02-engineering-quality-system.md) | Draft | 2026-09-14 |
| 03 | [文档系统稳定化方案](03-documentation-stabilization.md) | In Progress | 2026-09-17 |
| 05 | [KVCache Manager 演进方案](05-kv-cache-manager-evolution.md) | Superseded | 2026-09-24 |
| 06 | [AetherMind 系统能力演进路线图](06-system-capability-evolution-roadmap.md) | Draft | 2026-09-16 |
| 07 | [ExecutableModel 生产准备入口方案](07-executable-model-preparation.md) | Implemented | 2026-09-23 |
| 08 | [KV Cache 与 CPU Attention 能力演进方案](08-kv-cache-and-attention-capability-evolution.md) | Draft | 2026-09-24 |

## 推荐阅读路径

- [01-inference-session-generate-readiness.md](01-inference-session-generate-readiness.md)：从 execution state/resource contract、ExecutableModel、reference kernel 到 public Generate 的实施门禁。
- [02-engineering-quality-system.md](02-engineering-quality-system.md)：以 capability、风险台账、变更 profile 和 vertical slice 组织全仓库质量建设。
- [03-documentation-stabilization.md](03-documentation-stabilization.md)：Batch -1 前置里程碑，D0–D5 与退出条件 E1–E9；配套清查表 [03-documentation-stabilization-inventory.md](03-documentation-stabilization-inventory.md)。
- [05-kv-cache-manager-evolution.md](05-kv-cache-manager-evolution.md)：历史 KV 草案，当前事实与实施优先级由 08 号提案取代。
- [06-system-capability-evolution-roadmap.md](06-system-capability-evolution-roadmap.md)：全仓库 capability gap、模块级演进裁决、01–05 提案关系与 Batch A–F 实施排序。
- [07-executable-model-preparation.md](07-executable-model-preparation.md)：01 号计划 M2 的细化，`LoweredModelArtifact → ExecutableModel` 唯一生产准备入口、权重绑定映射与 inference 模块归属。
- [08-kv-cache-and-attention-capability-evolution.md](08-kv-cache-and-attention-capability-evolution.md)：当前 KV owner/epoch correctness、CPU Attention 优化证据门禁和 Paged KV 长期准入。
- 原 04 号 GEMM 专项提案已于 2026-09-19 迁出本目录：方案、路线与工作包状态见 [CPU GEMM 优化方案](../operators/gemm/cpu-gemm-optimization.md)，机器级证据见 [GEMM 算子索引](../operators/gemm/README.md)；算子专项提案的位置见 [算子开发与优化工作流](../guides/operator-development-workflow.md)。
