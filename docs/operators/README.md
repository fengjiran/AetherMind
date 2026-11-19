# 算子文档总入口

本目录是 **AetherMind 全部算子文档的唯一存放位置**（2026-09-19 由 `docs/designs/kernel_dev/`、`docs/improvement-plan/04-*` 与 `docs/tests/operators/` 归一）。规范、门禁与方法见唯一规范文档 [算子开发与优化工作流](../guides/operator-development-workflow.md)；实例文档一律按算子沉淀在本目录。

后续实施顺序见 [CPU 算子覆盖与性能演进路线图](cpu-operator-evolution.md)：覆盖台账与生产基线、矩阵计算、Attention、小算子/精度覆盖、线程与资源、量化及目标平台扩展。该入口组织跨算子工作包；具体合同、优化方案与机器证据继续归各算子目录。

跨算子覆盖台账与公共基准（参考机器/构建身份、代表模型 revision、真实生产选核身份、覆盖缺口）见 [CPU 算子覆盖与生产基线台账](cpu-coverage-baseline.md)（CPU-01）。

## 目录结构

```text
docs/operators/
├── README.md                      # 本入口
├── cpu-operator-evolution.md       # CPU 算子演进顺序、依赖与验收
├── cpu-coverage-baseline.md        # CPU-01 跨算子覆盖台账与公共基准（机器/模型/选核身份）
├── operator_contract_design.md    # Operator Contract 层设计（跨算子）
├── 算子系统设计.md                 # 算子体系设计（原 kernel_dev/算子系统设计.md）
└── <operator>/                    # 每算子一个自包含目录
    ├── README.md                  # 算子首页：门禁状态 + 证据索引
    ├── <op>-optimization.md       # 专项提案（如有；非平凡优化方案成文）
    ├── *算子契约.md                # 算子契约（如有）
    └── benchmarks/                # 证据记录：一机器一文件，追加式，不覆盖
```

现有目录包括 `gemm/`、`linear/`、`embedding/`、`rmsnorm/`、`rope/`、`attention/`、`kv_cache_update/`，以及 backend 数值 primitive `fp16_conversion/`、`bf16_conversion/`。

## 规则

- 新增算子的契约、专项提案、证据记录全部建在本目录对应算子子目录下；**不得**再写入 `docs/archive/`、`docs/improvement-plan/` 或新建独立验证目录。
- 专项提案（原 `improvement-plan/04-*`）已迁入本目录对应算子子目录（2026-09-19）：提案只承载优化原理、合同与工作包状态，证据记入 `benchmarks/` 记录文件，机器级数值不回填提案。
- 本目录维护算子契约与验证证据。相关实现说明见 [后端系统](../handbook/核心模块设计/后端系统%20%28Backend%29/后端系统%20%28Backend%29.md) 与 [计算图系统](../handbook/核心模块设计/计算图系统%20%28Graph%29/计算图系统%20%28Graph%29.md)。

## 算子索引

| 算子 | 目录 | 当前证据 |
|---|---|---|
| GEMM | [gemm/](gemm/) | 证据按机器记录（一机器一文件，54H5MMI / QHIHOGQ）；2026-09-21 清空重采 |
| RMSNorm | [rmsnorm/](rmsnorm/) | 契约（contract basis） |
| AddRmsNorm | [add_rmsnorm/](add_rmsnorm/) | FP32 plain/identity AVX2、O2 融合与未融合对照 |
| SiluMul | [silu_mul/](silu_mul/) | FP32/BF16 AVX2、prepared 和模型对照 |
| Cast | [cast/](cast/) | F32 ↔ F16 动态 Cast 和 FP16 A16 模型路径 |
| RoPE | [rope/](rope/) | 契约（contract basis） |
| Linear | [linear/](linear/) | F32 与 W16/A32 reference、plain/identity-packed、单机基线 |
| Embedding | [embedding/](embedding/) | F32 output、W16 scalar/F16C/AVX2 行级 widening 与单机基线 |
| Attention | [attention/](attention/) | F32 与 A32/KV16 reference、四组合模型验收与单机 baseline |
| KVCacheUpdate | [kv_cache_update/](kv_cache_update/) | F32 与 A32→KV16 update/transaction、四组合验收与单机 baseline |
| FP16 批量转换（backend 数值 primitive，无 GraphOp） | [fp16_conversion/](fp16_conversion/) | scalar/F16C/AVX2 合同、权重准备接入及单机证据 |
| BF16 批量转换（backend 数值 primitive，无 GraphOp） | [bf16_conversion/](bf16_conversion/) | 双向 scalar/AVX2 合同、独立 prepared 函数表与权重准备接入 |
