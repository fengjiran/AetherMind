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
| 06 | [AetherMind 系统能力演进路线图](06-system-capability-evolution-roadmap.md) | Draft | 2026-10-21 |
| 07 | [ExecutableModel 生产准备入口方案](07-executable-model-preparation.md) | Implemented | 2026-09-23 |
| 08 | [KV Cache 与 CPU Attention 能力演进方案](08-kv-cache-and-attention-capability-evolution.md) | Draft | 2026-09-24 |
| 09 | [Qwen 与 MoE 模型架构支持方案](09-qwen-and-moe-architecture-support.md) | Draft | 2026-10-20 |
| 10 | [上层 Runtime 组装与推理入口实施方案](10-upper-level-runtime-assembly.md) | Draft | 2026-10-19 |
| 11 | [FP16 / BF16 执行支持方案](11-fp16-bf16-execution-support.md) | Draft | 2026-10-20 |
| 12 | [CUDA 后端与执行支持方案](12-cuda-backend-execution-support.md) | Draft | 2026-10-21 |

## 推荐阅读路径

- [01-inference-session-generate-readiness.md](01-inference-session-generate-readiness.md)：从 execution state/resource contract、ExecutableModel、reference kernel 到 public Generate 的实施门禁。
- [02-engineering-quality-system.md](02-engineering-quality-system.md)：以 capability、风险台账、变更 profile 和 vertical slice 组织全仓库质量建设。
- [03-documentation-stabilization.md](03-documentation-stabilization.md)：Batch -1 前置里程碑，D0–D5 与退出条件 E1–E9；配套清查表 [03-documentation-stabilization-inventory.md](03-documentation-stabilization-inventory.md)。
- [05-kv-cache-manager-evolution.md](05-kv-cache-manager-evolution.md)：历史 KV 草案，当前事实与实施优先级由 08 号提案取代。
- [06-system-capability-evolution-roadmap.md](06-system-capability-evolution-roadmap.md)：全仓库 capability gap、模块级演进裁决与 Batch A–F 实施排序；关联 09 模型家族、11 精度与 12 CUDA 的独立主线，§11.4 记录单权重 identity 条件式零复制的待优化项与验收。
- [07-executable-model-preparation.md](07-executable-model-preparation.md)：01 号计划 M2 的细化，`LoweredModelArtifact → ExecutableModel` 唯一生产准备入口、权重绑定映射与 inference 模块归属。
- [08-kv-cache-and-attention-capability-evolution.md](08-kv-cache-and-attention-capability-evolution.md)：当前 KV owner/epoch correctness、CPU Attention 优化证据门禁和 Paged KV 长期准入。
- [09-qwen-and-moe-architecture-support.md](09-qwen-and-moe-architecture-support.md)：公共合同→Qwen3 Dense→Qwen3 MoE 主线；含 R1–R8 改造、binding 冷路径 workspace specialization、统一 decoder I/O、多 packed bindings 与验收，另列其他家族优先级和独立分支。
- [10-upper-level-runtime-assembly.md](10-upper-level-runtime-assembly.md)：从模型配置派生 Runtime、一次编译准备 kBoth plan 的独立 C++ 入口；包含生命周期、CPU preset、分阶段 C ABI 与集成验收门禁。
- [11-fp16-bf16-execution-support.md](11-fp16-bf16-execution-support.md)：显式 FP32 兼容执行→W16/A32→独立 KV16→A16 的分阶段提案；包含 logical weight ownership、构图精度、数值/ISA 合同、selector/layout 迁移与 production 验收。
- [12-cuda-backend-execution-support.md](12-cuda-backend-execution-support.md)：单 GPU 同步生成基线；明确 device placement、显存权重所有权、stream 完成与 KV 事务，并按 FP32→FP16/BF16→性能优化分阶段验收。
- 原 04 号 GEMM 专项提案已于 2026-09-19 迁出本目录：方案、路线与工作包状态见 [CPU GEMM 优化方案](../operators/gemm/cpu-gemm-optimization.md)，机器级证据见 [GEMM 算子索引](../operators/gemm/README.md)；算子专项提案的位置见 [算子开发与优化工作流](../guides/operator-development-workflow.md)。
