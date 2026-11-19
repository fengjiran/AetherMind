# AetherMind 演进提案索引

跨专题优先级见 [路线图](../roadmap.md)。每篇专题维护自身方案、实施步骤和验收标准，产品能力以 [产品需求](../product.md#14-当前能力与验证状态) 为准。

## 专题目录

| 专题 | 状态 |
|---|---|
| [02 工程质量体系](02-engineering-quality-system.md) | Draft |
| [06 系统能力演进路线图](06-system-capability-evolution-roadmap.md) | Draft |
| [08 KV Cache 与 CPU Attention](08-kv-cache-and-attention-capability-evolution.md) | In Progress |
| [09 Qwen 与 MoE](09-qwen-and-moe-architecture-support.md) | Draft |
| [11 FP16 / BF16 执行](11-fp16-bf16-execution-support.md) | In Progress |
| [12 CUDA 后端](12-cuda-backend-execution-support.md) | Draft |
| [13 生产验证与质量](13-deepinfer-quality-and-validation.md) | In Progress |

算子专项见 [CPU 算子路线图](../operators/cpu-operator-evolution.md) 和 [GEMM 优化方案（原 04）](../operators/gemm/cpu-gemm-optimization.md)。机器级记录见 [GEMM 基准索引](../operators/gemm/README.md)。

## 历史专题

| 专题 | 归档状态 |
|---|---|
| [01 InferenceSession / Generate 前置闭环](01-inference-session-generate-readiness.md) | Implemented |
| [03 文档系统稳定化](03-documentation-stabilization.md) · [配套清查表](03-documentation-stabilization-inventory.md) | Superseded |
| [05 KVCache Manager 演进草案](05-kv-cache-manager-evolution.md) | Superseded，由 08 承接 |
| [07 ExecutableModel 生产准备](07-executable-model-preparation.md) | Implemented |
| [10 上层 Runtime 组装与推理入口](10-upper-level-runtime-assembly.md) | Implemented |

历史验收和设计快照见 [归档](../archive/README.md)。
