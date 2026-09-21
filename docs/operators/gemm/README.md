# GEMM 算子文档

- **优化提案**: [CPU GEMM 优化方案](cpu-gemm-optimization.md)（方案、合同与实施顺序）
- **规范**: [算子开发与优化工作流](../../guides/operator-development-workflow.md)

## 证据记录

证据存放在 [benchmarks/](benchmarks/)：**一机器一文件**，文件名只含机器名与算子名。机器级数值按机器各自成立、互不替代，不跨机器引用；未列出的机器为 Not Collected。

| 机器 | 记录 | 状态 |
|---|---|---|
| 54H5MMI | [54h5mmi-gemm.md](benchmarks/54h5mmi-gemm.md) | 待重采（2026-09-21 清空） |
| QHIHOGQ | 首次采集时新建 | 待重采（2026-09-21 清空） |

原始 artifact 目录 `benchmark-results/operators/gemm/<run-id>/` 被 `.gitignore` 忽略，只存在于各采集机本地。与机器无关、可由源码复核的部分（benchmark/测试/采集脚本、correctness 契约、候选入口与 fallback 边界）随代码评审，不进入本目录。
