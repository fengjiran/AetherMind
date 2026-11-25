# AetherMind 路线图

本页维护项目级优先事项与提案索引。各专题维护自身设计、实施步骤和验收标准。已交付能力见 [产品能力矩阵](../product.md#14-当前能力与验证状态)，已知缺陷见 [问题清单](../issues.md#未解决)。

## 当前优先事项

| 方向 | 当前目标 | 实施入口 |
|---|---|---|
| CPU 算子覆盖与性能 | 按生产路径 profiling 推进剩余热点和算子覆盖 | [CPU 算子路线图](../operators/cpu-operator-evolution.md) |
| KV / Attention 演进 | 维护静态 KV 的布局预算和提交合同，按真实负载选择下一阶段 Attention 优化 | [08 KV Cache 与 CPU Attention](08-kv-cache-and-attention-capability-evolution.md) |
| FP16 / BF16 执行 | 补齐 A16 的 AVX-512 真机数值与性能证据 | [11 FP16 / BF16 执行方案](11-fp16-bf16-execution-support.md#64-性能资源与发布状态) |
| 生产验证与质量 | 补齐剩余模型数值、平台组合和交付验证 | [13 生产验证实施方案](13-deepinfer-quality-and-validation.md) |

GEMM 专项见 [GEMM 优化方案（原 04）](../operators/gemm/cpu-gemm-optimization.md) 和 [基准索引](../operators/gemm/README.md)。

## 候选提案

这些专题处于 Draft。

- [02 工程质量体系](02-engineering-quality-system.md)
- [06 系统能力演进路线图](06-system-capability-evolution-roadmap.md)
- [09 Qwen 与 MoE](09-qwen-and-moe-architecture-support.md)
- [12 CUDA 后端](12-cuda-backend-execution-support.md)
- [14 CPU GEMM 算子内多线程并行方案（CPU-08）](14-cpu-gemm-intra-op-parallelism.md)

## 历史提案

| 专题 | 状态 |
|---|---|
| [01 InferenceSession / Generate 前置闭环](01-inference-session-generate-readiness.md) | Implemented |
| [03 文档系统稳定化](03-documentation-stabilization.md) · [配套清查表](03-documentation-stabilization-inventory.md) | Superseded |
| [05 KVCache Manager 演进草案](05-kv-cache-manager-evolution.md) | Superseded，由 08 承接 |
| [07 ExecutableModel 生产准备](07-executable-model-preparation.md) | Implemented |
| [10 上层 Runtime 组装与推理入口](10-upper-level-runtime-assembly.md) | Implemented |

历史验收和设计快照见 [归档](../archive/README.md)。
