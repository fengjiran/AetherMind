# GEMM 算子文档

- **优化提案**: [CPU GEMM 优化方案](cpu-gemm-optimization.md)（方案、合同与实施顺序）
- **专项提案**: [CPU GEMM Packed Weight 提案](cpu-gemm-packed-weight.md)（exact recipe 与 packed-B 具体化）
- **规范**: [算子开发与优化工作流](../../guides/operator-development-workflow.md)

## 证据记录

证据存放在 [benchmarks/](benchmarks/)：**一机器一文件**，文件名只含机器名与算子名。机器级数值按机器各自成立、互不替代，不跨机器引用；未列出的机器为 Not Collected。

| 机器 | 记录 | 状态 |
|---|---|---|
| 54H5MMI | [54h5mmi-gemm.md](benchmarks/54h5mmi-gemm.md) | Needs More Data（2026-09-23 WSL2 diagnostic；不用于 promotion） |
| QHIHOGQ | 首次采集时新建 | 待重采（2026-09-21 清空） |

原始 artifact 目录 `benchmark-results/operators/gemm/<run-id>/` 被 `.gitignore` 忽略，只存在于各采集机本地。与机器无关、可由源码复核的部分（benchmark/测试/采集脚本、correctness 契约、候选入口与 fallback 边界）随代码评审，不进入本目录。

## 实现与证据状态

| 能力 | 状态 | 说明 |
|---|---|---|
| exact recipe 生产链 | Implemented | descriptor → backend query → inference request → pack artifact → exact-key store / plan |
| graph-wide packed preparation | Implemented | Linear、Embedding、RmsNorm、QKV、GateUp、AddRmsNorm identity consumers 已覆盖；tiny-Llama packed preparation test 通过 |
| AVX2 bpanel | Candidate / correctness verified | KC512 layout、Linear/QKV/GateUp scan/blocked 与尾块测试已落地；global identity descriptor 保持默认 |
| 性能与 KC 选择 | **Needs More Data** | 当前 WSL2 小样本不能量化可信噪声 floor；无 KC256 同机对照，不提升 candidate priority |

决策合同见 [ADR-0002](../../decisions/0002-cpu-gemm-packed-weight.md)。本机诊断采样见 [54H5MMI GEMM 记录](benchmarks/54h5mmi-gemm.md)；其他机器数据不得由该记录代替。
