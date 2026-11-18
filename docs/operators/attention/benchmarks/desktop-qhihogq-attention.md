# Attention 基准记录 — desktop-qhihogq

| 项 | 值 |
|---|---|
| CPU | Intel Core i9-12900H，guest 20 logical CPUs / 10 cores / SMT 2，检测可用 AVX2/FMA/F16C，无 AVX-512，guest L3 报告 24 MiB |
| OS / kernel | WSL2，6.18.58.1-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2，Release `-O3 -DNDEBUG`，SIMD TUs 禁用 fast-math 与 FMA contraction |
| 备注 | 单线程；prepared 绑 guest CPU 2，模型绑 CPU 1。不能推导宿主 P/E 核类别、DRAM 带宽或跨机器阈值；无 PMU 样本。 |

## CPU-04 首段：两遍 causal Attention SIMD

- 基线：`0a18f48c` 的 reference build；候选：同 commit + CPU-04 工作树实现。同期 GEMM 格式差异没有改动运算。保留两份共享库及 SHA-256、构建输出、源码差异和独立精度 fixture。性能表用隔离修订前的 Release 二进制；最终修订防止 Debug 中 AVX 版浮点比较/exp helper 被 baseline 链接选用。Release 的短 Decode 和 Prefill 重测与 oracle 检查见 `final/smoke.json`，两份候选库的 hash 分别保存。
- Prepared：`taskset -c 2 ./attention-benchmark-final --benchmark_filter='^BenchmarkAttention/(.*/1/128/1|.*/1/4096/1|F32/1/4096/8|.*/64/64/1)$' --benchmark_min_time=1x --benchmark_repetitions=2`。实际 registry/params binding/invocation；32 Q heads、8 KV heads、head_dim 128。每次准备后检查所有 causal 输出；两组 A/A（四进程）和四组 AB/BA 交错（八进程）。Decode 的 4096/8 轮转 8 份 F32 K/V（256 MiB），每个计时 batch 完整访问八份，实际内存流量未测。其他组复用同一份 cache。

| Prepared 路径 | Decode 128 / hot | Decode 4096 / hot | Prefill 64 / hot |
|---|---:|---:|---:|
| F32 | 0.441 → 0.380 | 15.764 → 14.321 | 7.160 → 5.834 |
| A32 / KV FP16 | 14.551 → 0.478 | 485.001 → 16.900 | 259.799 → 8.029 |
| A32 / KV BF16 | 11.744 → 0.498 | 387.074 → 16.474 | 194.308 → 8.096 |
| BF16 A16 | 35.193 → 0.520 | 1119.439 → 16.411 | 569.577 → 11.018 |

表中为进程内 CPU 时间中位数再取四进程中位数，单位 ms。F32 4096 轮转 256 MiB：15.690 → 13.976 ms；F32 长 Decode/Prefill 的进程区间分离，但部分 F32 组的 A/A 离散接近差幅，不能据此设回退阈值。KV16/BF16 A16 的收益相对逐元素 reference；BF16 A16 还减少了重算 QK 与重复 widening，不能全部归因于指令宽度。

- 模型：独立合成的 1-layer Llama GQA（4 Q heads、2 KV heads、head_dim 128、hidden 512、vocab 128），相同精度/权重与 token 序列、O2/packed、生成 4 token。通过 `tools/run_model_benchmark.py` 走实际 Compile/Prepare、内部 ExecutionPlan 与公开 Generate；两个 context × 四精度 × 四对独立进程交错 A/B，每进程 1 warmup、3 repetitions。32 对报告通过 workload/轨迹对照，生成 token 与 KV 水位一致。O1/O2 两阶段 descriptor 另用 Load/plan 探针核对；O1 不在计时表内。最终库另复验四精度 × O1/O2 共 8 对短 context 的模型轨迹，见 `final/model-smoke/`。

| 模型路径 | 16-token prompt Generate（ms） | 256-token prompt Generate（ms） |
|---|---:|---:|
| F32 | 4.195 → 4.402 | 54.340 → 51.964 |
| A32 / KV FP16 | 8.647 → 3.981 | 854.216 → 68.552 |
| A32 / KV BF16 | 8.519 → 5.012 | 648.336 → 68.786 |
| BF16 A16 | 9.890 → 4.173 | 1036.958 → 71.550 |

F32 Generate 两个 context 的进程区间均重叠，端到端仍为 Needs More Data。该模型只用于执行与性能对照。模型级冻结校准 bundle 本地缺失，既有独立校准验收未重跑；代表性 checkpoint、超长模型 Prefill、其他硬件与 KV16 大工作集轮转未重新验收。

- 正确性：Release 聚焦集 **80/80**，含实际选核、2048-token Decode 与 causal Prefill 的 reference 对拍、stride/tail/fallback、Inf/overflow、非法 visibility 写出前拒绝、KV transaction 和 BF16/KV16 reduced model 回归。现有 glibc interposer 在 prepared Decode 调用中未检测到 malloc-family 调用。ASan/UBSan **26/26**（LSan 关闭）；独立 F16C=OFF、AVX2=OFF 构建各 **26/26**，并核对编译选项。禁用 AVX2 在本机验证 fallback，不代表物理无 AVX 的主机。
- Raw：`benchmark-results/operators/attention/20261118T155102Z_0a18f48c_desktop-qhihogq_cpu04-simd/`，本地 gitignored，无远端 retention。包括 fixture 生成脚本、两份库的 hash、prepared 全部 repetitions、模型报告与对照、失败尝试、测试日志及实际 kernel 名称。
- 结论：**Accepted（本机首段实现和所测生产路径）**，合法 feature/shape 选用 SIMD；F32 公开请求性能、代码路径之外的机器和模型规模仍需补证。继续 CPU-04 的 GQA/KV 复用；单独验证 scratch、归约与长 context。

## CPU-04 第二段：Decode GQA/KV 复用

- 基线使用 `0a18f48c` + 首段最终 SIMD 工作树的 Release 库。候选只替换兼容单 token Decode 的 compute body，保持相同 descriptor、精度、layout 和 KV contract。两阶段的库、hash、ISA 编译命令与源码快照均保存。初轮允许有历史的全部长度，复测版以 16 个可见 token 为启用下限。
- Prepared A/B：`taskset -c 2 ./attention-benchmark --benchmark_filter='^BenchmarkAttention/' --benchmark_min_time=0.012s --benchmark_repetitions=3`。同一 harness 走 registered/resolved + prepared params；独立 analytic guard 用相邻 query heads 的不同 Q 值检查 causal 输出。两组 A/A（四进程），再四组 AB/BA（八进程）。主表用 32 Q heads / 8 KV heads、head_dim 128。单位为 ms，先取进程内 CPU 时间中位数，再取四进程中位数。

| Prepared 路径 | Decode 16 / 单 KV 集合 | Decode 4096 / 单 KV 集合 | Decode 4096 / 四组轮转 |
|---|---:|---:|---:|
| F32 | 0.0456 → 0.0365 | 13.675 → 10.712 | 13.629 → 10.980 |
| A32 / KV FP16 | 0.0626 → 0.0448 | 16.436 → 11.610 | 16.361 → 11.883 |
| A32 / KV BF16 | 0.0615 → 0.0423 | 16.120 → 11.131 | 16.569 → 11.181 |
| BF16 A16 | 0.1110 → 0.0964 | 16.497 → 11.157 | 16.661 → 11.193 |

  4096 各组进程区间分离；轮转 working set 为 128 MiB（F32）或 64 MiB（KV16），不等同于 DRAM 流量。4096 / 2:1 GQA 的观察降幅为 28%～34%；1024 / 3:1 / head_dim 65 的 F32 区间重叠，其余组合降幅 16%～17%。MHA、无历史单 token 和 Prefill 32 作为回退对照，差异与各组波动重叠；不能用全局固定百分比判断回退。

- 模型 A/B：沿用上一段的 1-layer 合成 Llama GQA（4 Q heads / 2 KV heads、head_dim 128），64/512-token prompt，O2/packed，生成 64 token。用 `tools/run_model_benchmark.py`（guest CPU 1）计时真实 ExecutionPlan 和公开 Generate。每组四对交错进程，各 1 warmup + 3 repetitions。32 对报告通过逐步 token、生成轨迹和 KV 水位对照。下表为 512-token prompt，内部 Decode 累加 63 步 Execute 时间，Generate 为整个公开请求。

| 模型路径 | 内部 Decode 总时间（ms） | 公开 Generate 时间（ms） |
|---|---:|---:|
| F32 | 28.625 → 24.319 | 109.571 → 106.876 |
| A32 / KV FP16 | 30.715 → 24.308 | 138.604 → 130.350 |
| A32 / KV BF16 | 31.519 → 24.811 | 133.221 → 127.559 |
| BF16 A16 | 40.998 → 35.671 | 200.059 → 190.562 |

  512-token 内部 Decode 四组进程区间均分离；有些公开 Generate 组区间重叠，百分比不推广到不同模型/机器。短 prompt 各分组及 Prefill 对照见 `model-summary.csv`。未捕获 PMU 指标，未重跑冻结的模型数值校准 bundle。

- Score tile 备选：同一 Release 链接参数，另编译最多 256 个 token 的固定 2 KiB 配对 score tile，超出仍用两遍重算。与本段最终两遍版另做 A/A + 四对 prepared AB/BA；128-token 的局部延迟再降约 12%～17%，4096 没有独立收益。128-token prompt + 128-token generation 的 16 对模型报告均通过轨迹检查，但四种精度的内部 Decode 进程区间均重叠；只有 KV FP16 的公开 Generate 区间分离。未启用原型。Online softmax 会改变归约顺序，本轮未计时。
- 正确性：Release 聚焦集 **83/83**（数值对拍、16-token 边界、odd group/MQA、head tail、可见但不得读的后缀、NaN head 隔离、KV 事务、分配计数）；ASan/UBSan、禁用 F16C、禁用 AVX2 的聚焦集各 **29/29**，LSan 关闭。ISA build fixture 通过。
- raw：`benchmark-results/operators/attention/20261118T161423Z_0a18f48c_desktop-qhihogq_cpu04-decode/`。包含两轮 prepared A/B、score 原型及模型比较、全部 repetitions、库与测试日志。本地 gitignored，无远端 retention。
- 结论：**Accepted（本机已验证的 Decode GQA/KV 两遍路径）**。保留兼容阈值及 fallback；score tile 未达到生产提升门禁。下一段独立处理 Prefill query/key blocking。

## CPU-04 第三段：Prefill 双 query 行

- 基线为 `0a18f48c` + CPU-04 第二段工作树的 Release 库，与上段保存的最终库一致。候选在同一 AVX2 descriptor 内并行处理同一 head 的相邻 query 行，第一行只走公共 K/V 前缀，第二行单独处理多出的对角 token。首轮从 2 行启用；最终库保守设为 32 行。库、源码和 ISA 编译命令分别留存。
- Prepared A/B：guest CPU 2；同一个 `aethermind_benchmark --benchmark_filter=^BenchmarkAttention/... --benchmark_min_time=1x --benchmark_repetitions=2` 交替链接两份库。逐行、逐 head 的 analytic guard 在计时前验证 causal 输出。两组 A/A（四进程），再四组 AB/BA（八进程）。下表为最终库，32 Q / 8 KV heads，head_dim 128，单位 ms，先取进程内 CPU 时间中位数，再取四进程中位数。

| Prepared Prefill | 32 token | 128 token | 512 token |
|---|---:|---:|---:|
| F32 | 1.660 → 1.264 | 25.363 → 19.782 | 399.952 → 318.880 |
| A32 / KV FP16 | 2.235 → 1.465 | 33.294 → 24.191 | 557.428 → 373.588 |
| A32 / KV BF16 | 2.224 → 1.514 | 34.276 → 22.925 | 535.629 → 363.738 |
| BF16 A16 | 3.826 → 3.010 | 40.632 → 30.353 | 561.958 → 389.677 |

  31-token Prefill 与 128-context 单 token Decode 对照的进程区间重叠。129-token MHA / head_dim 65（奇数行与 head tail）四种精度进程区间分离。另在 guest CPU 4 用首轮候选做两对 AB/BA 长形状诊断：完整 2048/4096 Prefill 用 4 Q / 2 KV heads，F32 观察降幅约 23%/28%，BF16 A16 约 31%/34%；64 行、4096 context 的 wide 形状单独覆盖非零 `query_begin`。长诊断不是 4096-token 完整模型性能结论。未捕获 PMU，不推断实际 DRAM 流量。

- 生产模型：沿用 1-layer 合成 Llama GQA（4 Q / 2 KV heads，head_dim 128）。通过 `tools/run_model_benchmark.py` 在 guest CPU 1 运行真实 ExecutionPlan，使用 `--max-new-tokens 1` 单独测公开首 token 请求。最终库 O2/packed 的 64/512-token prompt 共 32 对交错进程报告，每进程 1 warmup + 3 repetitions。2047-token 另做 F32、BF16 A16 各两对，每进程 0 warmup + 2 repetitions；O1/plain 四精度做短请求轨迹复验。所有比较通过相同 token、轨迹与 KV 水位检查。

| O2 模型路径 | 512-token 内部首 token 请求（ms） | 512-token 公开 Generate(1)（ms） |
|---|---:|---:|
| F32 | 85.165 → 74.159 | 87.661 → 75.269 |
| A32 / KV FP16 | 110.293 → 84.903 | 109.370 → 86.616 |
| A32 / KV BF16 | 107.096 → 82.081 | 107.832 → 82.434 |
| BF16 A16 | 169.370 → 147.367 | 167.719 → 145.721 |

  512-token F32 的进程区间重叠，其余三组分离。2047-token 首 token 请求 F32 是 950.152 → 806.951 ms，BF16 A16 是 1509.433 → 1191.343 ms；样本限于两对独立进程。初轮最终库矩阵的 64-token F32 曾出现较大变慢中位数，进程区间重叠。待其他任务结束后，对 F32/BF16 A16 分别补采四进程 A/A 和四对 AB/BA（2 warmups + 11 repetitions），首 token 请求中位数变化均约 −0.9%，早期变慢未复现，短模型组不作性能结论。未重跑冻结的模型数值校准 bundle；合成模型的 token 一致性不等于代表性 checkpoint 的质量验收。

- 正确性：Release 聚焦集 **84/84**，含 31/32 行门槛、不同 Q/output 行 stride、长 KV 的有限值逐位对拍、MHA/odd tail、遮蔽的 NaN V、KV 事务和 malloc-family 计数。ASan/UBSan、禁用 F16C、禁用 AVX2 的聚焦集各 **30/30**；LSan 关闭，ISA build fixture 通过。既有 Decode 用同一配对代码的无对角尾部实例，无新增 planner workspace。
- Raw：`benchmark-results/operators/attention/20261118T163806Z_0a18f48c_desktop-qhihogq_cpu04-prefill/`，包含两轮 prepared、长形状诊断、模型报告与短请求独立复测、编译及测试日志；本地 gitignored，无远端 retention。
- 结论：**Accepted（本机已验证的 Prefill 双行路径）**。32 行及以上的兼容 Prefill 启用；更大分块、score strip 与 online softmax 由长 context profile 决定，保持独立验收。
