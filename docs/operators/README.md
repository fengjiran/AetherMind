# 算子文档总入口

本目录是 **AetherMind 全部算子文档的唯一存放位置**（2026-09-19 由 `docs/designs/kernel_dev/`、`docs/improvement-plan/04-*` 与 `docs/tests/operators/` 归一）。规范、门禁、骨架与方法见唯一规范文档 [算子开发与优化工作流](../guides/operator-development-workflow.md)；实例文档一律按算子沉淀在本目录。

## 目录结构

```text
docs/operators/
├── README.md                      # 本入口
├── operator_contract_design.md    # Operator Contract 层设计（跨算子）
├── 算子系统设计.md                 # 算子体系设计（原 kernel_dev/算子系统设计.md）
└── <operator>/                    # 每算子一个自包含目录
    ├── README.md                  # 算子首页：门禁状态 + 证据索引
    ├── <op>-optimization.md       # 专项提案（如有）
    ├── *算子契约.md                # 算子契约（如有）
    └── benchmarks/                # 证据文件：一工作包一文件，结论按 commit@date 冻结追加
```

现有算子目录：`gemm/`、`linear/`、`rmsnorm/`、`rope/`。

## 规则

- 新增算子的契约、提案、证据全部建在本目录对应算子子目录下；**不得**再写入 `docs/designs/`、`docs/improvement-plan/` 或 `docs/tests/`。
- 专项提案迁移自 `improvement-plan/04-*`：仅保留优化原理、工作包状态与证据链接；机器级数值与噪声 floor 在各自算子目录 `benchmarks/` 内。
- 本目录只放**算子层**文档：跨算子共享的算子体系/契约设计平铺在本目录（上述两份）。`backend_design.md`、`dispatch_design.md`、`cpu_capability_design.md`、`op_evaluator.md` 属 backend/graph 模块设计，仍留在 [docs/designs/](../designs/)，不搬入本目录。

## 算子索引

| 算子 | 目录 | 当前证据 |
|---|---|---|
| GEMM | [gemm/](gemm/) | G0 本地 reference baseline、配对 A/B 噪声 floor 与 Roofline 定位 Accepted（历史快照，两机器并列）；production gate Needs More Data |
| RMSNorm | [rmsnorm/](rmsnorm/) | 契约（contract basis） |
| RoPE | [rope/](rope/) | 契约（contract basis） |
| Linear | [linear/](linear/) | 契约（contract basis） |