# AetherMind 演进提案索引

> 本目录存放未来架构演进提案（"将来要做什么"）。写作与命名规范见 [文档系统规范](../guides/documentation-guide.md#74-演进提案)。
>
> 专题文档章节规范：现状分析 / 目标架构 / 方案与备选 / 实施步骤 / 风险与依赖 / 验收标准。
> 状态流转：Draft → In Progress → Implemented → Superseded。

## 专题目录

| 编号 | 专题 | 状态 | 最后更新 |
|---|---|---|---|
| 01 | [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md) | Draft | 2026-09-03 |
| 02 | [工程质量体系建设方案](02-engineering-quality-system.md) | Draft | 2026-09-14 |
| 03 | [文档系统稳定化方案](03-documentation-stabilization.md) | In Progress | 2026-09-17 |
| 04 | [CPU GEMM 优化方案](04-cpu-gemm-optimization.md) | In Progress | 2026-09-19 |
| 05 | [KVCache Manager 演进方案](05-kv-cache-manager-evolution.md) | Draft | 2026-09-16 |
| 06 | [AetherMind 系统能力演进路线图](06-system-capability-evolution-roadmap.md) | Draft | 2026-09-16 |

## 推荐阅读路径

- [01-inference-session-generate-readiness.md](01-inference-session-generate-readiness.md)：从 execution state/resource contract、ExecutableModel、reference kernel 到 public Generate 的实施门禁。
- [02-engineering-quality-system.md](02-engineering-quality-system.md)：以 capability、风险台账、变更 profile 和 vertical slice 组织全仓库质量建设。
- [03-documentation-stabilization.md](03-documentation-stabilization.md)：Batch -1 前置里程碑，D0–D5 与退出条件 E1–E9；配套清查表 [03-documentation-stabilization-inventory.md](03-documentation-stabilization-inventory.md)。
- [04-cpu-gemm-optimization.md](04-cpu-gemm-optimization.md)：共享 CPU GEMM engine、scalar/SIMD/packed-weight 路径、shape specialization、分层 benchmark 与证据门禁。只承载优化原理与工作包状态，机器级数值见 [GEMM 实验记录与验证报告索引](../tests/operators/gemm/README.md)。
- [05-kv-cache-manager-evolution.md](05-kv-cache-manager-evolution.md)：静态 contiguous baseline 的 correctness 修复、lease/transaction/kernel binding 以及 Paged KV 长期演进边界。
- [06-system-capability-evolution-roadmap.md](06-system-capability-evolution-roadmap.md)：全仓库 capability gap、模块级演进裁决、01–05 提案关系与 Batch A–F 实施排序。
