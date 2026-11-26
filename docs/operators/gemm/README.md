# GEMM 算子文档

- **优化提案**: [GEMM 优化方案](cpu-gemm-optimization.md)（FP32、FP16、BF16 的 CPU 优化与 CUDA 接续路线）
- **专项提案**: [CPU GEMM Packed Weight 提案](cpu-gemm-packed-weight.md)（exact recipe 与 packed-B 具体化）
- **规范**: [算子开发与优化工作流](../../guides/operator-development-workflow.md)

## 证据记录

证据存放在 [benchmarks/](benchmarks/)：**一机器一文件**，文件名只含机器名与算子名。机器级数值按机器各自成立、互不替代，不跨机器引用；未列出的机器为 Not Collected。

| 机器 | 记录 | 状态 |
|---|---|---|
| 54H5MMI | [54h5mmi-gemm.md](benchmarks/54h5mmi-gemm.md) | CPU-03 identity 已验收（WSL2 限定），KC 选择与裸机证据待补 |
| QHIHOGQ | 首次采集时新建 | 待重采（2026-09-21 清空） |

原始 artifact 目录 `benchmark-results/operators/gemm/<run-id>/` 被 `.gitignore` 忽略，只存在于各采集机本地。与机器无关、可由源码复核的部分（benchmark/测试/采集脚本、correctness 契约、候选入口与 fallback 边界）随代码评审，不进入本目录。

## 实现与证据状态

| 能力 | 状态 | 说明 |
|---|---|---|
| 共享参数边界 | Implemented | 非模板 `GemmArgs`，A/B/output 为 byte 指针，stride 保持元素单位；明确的 `RunGemm*` entry 决定存储解释与数值合同，packed-B 参数及 F32 布局独立保留 |
| portable scalar K-contiguous Prefill | Implemented / 本机 prepared 验证 | `M>=3,K>=32,N>=32` 的完整列块使用 16 KiB 临时权重 panel（`M=2` 只有一次 B 遍历，保留 4 列路径）；plain/identity 及 streaming 结果见[单机记录](benchmarks/54h5mmi-gemm.md#scalar-k-连续-prefill-的行块与临时权重块对照)。 |
| portable scalar N-contiguous | Implemented / 本机 prepared 验证 | 单行使用专用入口，多行按 4/2/1 行分解。NC/KC、升序 K 与零分配合同沿用原实现。N 连续 plain Linear 的同形状对照见[单机记录](benchmarks/54h5mmi-gemm.md#scalar-n-连续路径的四行分块)。 |
| exact recipe 生产链 | Implemented | descriptor → backend query → inference request → pack artifact → exact-key store / plan |
| graph-wide packed preparation / execution | Implemented | Linear、Embedding、RmsNorm、QKV、GateUp、AddRmsNorm identity consumers 已覆盖；tiny tied-GQA Llama 的 O1 plain / O2 packed 全模型回归覆盖 scalar-oracle logits/KV、Prefill→Decode、重复 Generate 与 prepared Decode 零 malloc；HF 目录加载入口亦覆盖两种准备路径 |
| AVX2 bpanel | Candidate / correctness verified | KC512 layout、Linear/QKV/GateUp scan/blocked 与尾块测试已落地；global identity descriptor 保持默认 |
| identity/scalar/AVX2 生产性能 | Accepted（本机限定） | [CPU-03 记录](benchmarks/54h5mmi-gemm.md#cpu-03-fp32-gemm-生产闭环与布局-promotion-决策)：registered/prepared、Generate、M sweep、fused/unfused 与 packing-inclusive 均已对照；参考实现独立保留 |
| 性能与 KC 选择 | **B-panel 全局 promotion 拒绝；KC 选择仍 Needs More Data** | CPU-02/03 的本机实现对照单独记账；B-panel 与 identity 的全边界比较决定 promotion，KC256 未测，不宣称 KC512 最优 |
| AVX2 Prefill 与 W16 Decode | Implemented | FP32 direct A 与 B panel 复用，W16/A32 SIMD panel 转置与单行幅值分链，同型 A16 的 FP32 分块累加。见 [实施顺序](cpu-gemm-optimization.md#71-cpu-0203-之后的实施顺序) 和 [机器记录](benchmarks/54h5mmi-gemm.md) |
| FP32 AVX2 权重复用 | Implemented / 本机 prepared 验证 | K 连续的小 M 路径按 16 列复用权重，大投影的 blocked 尾行复用 B 面板。N 连续的多行 Linear 直接沿 N 向量化。[形状对照](benchmarks/54h5mmi-gemm.md#fp32-avx2-权重复用与-n-连续路径) |
| FP32 AVX2 6×16 寄存器分配 | Implemented / 本机 prepared 验证 | plain 和 identity 的直接 A tile 每次推进一个 K，热循环中的 12 个累加器驻留 YMM。[汇编与对照](benchmarks/54h5mmi-gemm.md#fp32-avx2-6x16-微内核的指令吞吐) |
| A16/W16 GEMM driver | Implemented | 同型 FP16/BF16 的 reference 和 ISA 路径通过共享参数供 Linear 调用，结果以同型 16-bit 或 FP32 写回。见 [内部合同](cpu-gemm-optimization.md#41-backend-internal-contracts) |
| W16/A32 长 Prefill | Implemented | direct A 与有界 M block 的 B panel 复用。Decode 的单核带宽对照见 [机器记录](benchmarks/54h5mmi-gemm.md) |
| CUDA | Proposed | 从 FP32 O1/plain 基线开始。见 [CUDA 接入顺序](cpu-gemm-optimization.md#74-cuda-接入顺序) |

决策合同见 [backend 模块设计](../../handbook/核心模块设计/后端系统%20%28Backend%29/后端系统%20%28Backend%29.md) 的 packed weight 与布局章节。F32 Linear/QKV/GateUp 的默认 packed descriptor 为 identity optimized，B-panel candidate 以较低 priority 注册。各组合见 [CPU 覆盖台账 §3.1/§4](../cpu-coverage-baseline.md#31-矩阵计算)。
