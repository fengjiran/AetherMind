# 在研方向

> 本文件只记录当前活跃方向。实现完成后，结论并入 `architecture.md` / `modules/` 与 `CHANGELOG.md`，并从本表移除。
> 详细提案与历史记录保留在 `improvement-plan/`，但本表才是当前优先级入口。

| 方向 | 状态 | 当前目标 | 提案 |
|---|---|---|---|
| 工程质量体系 | Draft | capability 台账、验证 profile 与风险治理 | [02](improvement-plan/02-engineering-quality-system.md) |
| 系统能力路线图 | Draft | 全仓库 capability gap 与实施排序 | [06](improvement-plan/06-system-capability-evolution-roadmap.md) |
| KV / Attention 演进 | In Progress | KV owner/epoch、静态布局预算与 CPU Attention 优化准入 | [08](improvement-plan/08-kv-cache-and-attention-capability-evolution.md) |
| Qwen 与 MoE | Draft | 公共合同、Qwen3 Dense 到 MoE 的改造顺序 | [09](improvement-plan/09-qwen-and-moe-architecture-support.md) |
| FP16 / BF16 执行 | In Progress | W16/A32、KV16、A16 的精度合同与验收门禁 | [11](improvement-plan/11-fp16-bf16-execution-support.md) |
| CUDA 后端 | Draft | 单 GPU 同步基线、target-aware 编译与验证矩阵 | [12](improvement-plan/12-cuda-backend-execution-support.md) |
| 生产验证与质量 | In Progress | 模型 golden、teacher-forcing 等价、CTest/sanitizer/ISA 与 benchmark | [13](improvement-plan/13-deepinfer-quality-and-validation.md) |

已实现或被取代的方案（01、03、05、07、10）仍保留在 `improvement-plan/` 供追溯，不再作为在研方向。

已知缺陷与短周期待办见 [issues.md](issues.md)；产品范围与验收标准见 [product.md](product.md)；系统现状见 [architecture.md](architecture.md)。
