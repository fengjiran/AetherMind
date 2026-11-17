# GEMM 算子文档

- **优化提案**: [CPU GEMM 优化方案](cpu-gemm-optimization.md)（方案、合同与实施顺序）
- **专项提案**: [CPU GEMM Packed Weight 提案](cpu-gemm-packed-weight.md)（exact recipe 与 packed-B 具体化）
- **规范**: [算子开发与优化工作流](../../guides/operator-development-workflow.md)

## 证据记录

证据存放在 [benchmarks/](benchmarks/)：**一机器一文件**，文件名只含机器名与算子名。机器级数值按机器各自成立、互不替代，不跨机器引用；未列出的机器为 Not Collected。

| 机器 | 记录 | 状态 |
|---|---|---|
| 54H5MMI | [54h5mmi-gemm.md](benchmarks/54h5mmi-gemm.md) | Needs More Data（2026-11-17 scalar N-contiguous 修复与 WSL2 diagnostic；不用于 promotion） |
| QHIHOGQ | 首次采集时新建 | 待重采（2026-09-21 清空） |

原始 artifact 目录 `benchmark-results/operators/gemm/<run-id>/` 被 `.gitignore` 忽略，只存在于各采集机本地。与机器无关、可由源码复核的部分（benchmark/测试/采集脚本、correctness 契约、候选入口与 fallback 边界）随代码评审，不进入本目录。

## 实现与证据状态

| 能力 | 状态 | 说明 |
|---|---|---|
| 共享参数边界 | Implemented | 非模板 `GemmArgs`，A/B/output 为 byte 指针，stride 保持元素单位；明确的 `RunGemmF32*` entry 决定存储解释与数值合同，packed-B 参数及 F32 布局独立保留 |
| portable scalar N-contiguous | Implemented / local diagnostic | 独立 NC/KC 分块与连续 N 遍历，覆盖 M=1/small-M/generic-M；correctness、guard、升序 K 与零 malloc 已回归；不改变 plain Linear/MatMul 默认 resolve，性能证据见本机记录 |
| exact recipe 生产链 | Implemented | descriptor → backend query → inference request → pack artifact → exact-key store / plan |
| graph-wide packed preparation / execution | Implemented | Linear、Embedding、RmsNorm、QKV、GateUp、AddRmsNorm identity consumers 已覆盖；tiny tied-GQA Llama 的 O1 plain / O2 packed 全模型回归覆盖 scalar-oracle logits/KV、Prefill→Decode、重复 Generate 与 prepared Decode 零 malloc；HF 目录加载入口亦覆盖两种准备路径 |
| AVX2 bpanel | Candidate / correctness verified | KC512 layout、Linear/QKV/GateUp scan/blocked 与尾块测试已落地；global identity descriptor 保持默认 |
| 性能与 KC 选择 | **Needs More Data** | 当前 WSL2 小样本不能量化可信噪声 floor；无 KC256 同机对照，不提升 candidate priority |

决策合同见 [backend 模块设计](../../modules/backend.md) 的 packed weight 与布局章节。本机诊断采样见 [54H5MMI GEMM 记录](benchmarks/54h5mmi-gemm.md)；其他机器数据不得由该记录代替。当前生产默认 resolve 身份（F32 Linear/QKV/GateUp packed 仍为 identity reference，B-panel candidate 已注册 eligible 但 priority tie 未胜出）见 [CPU 覆盖台账 §3.1/§4](../cpu-coverage-baseline.md#31-矩阵计算)。
