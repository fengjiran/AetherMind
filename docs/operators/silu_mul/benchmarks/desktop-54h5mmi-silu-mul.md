# SiluMul 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H，WSL2 guest 16 logical CPUs，L3 24 MiB，effective AVX2/FMA/F16C 可用 |
| OS | WSL2 6.6.87.2-microsoft-standard-WSL2 |
| Build | GCC 14.2，Release `-O3 -DNDEBUG`，无全局 fast-math，SiluMul AVX2 独立编译 |
| Affinity | 单线程，prepared 和模型 timing 绑 guest CPU 2。独立诊断的 F16 W16/A32 和 BF16 A16 绑 guest CPU 3 |

## CPU-05 收口：FP32/BF16 SiluMul

基线为 `4a048033` 加首段 AddRmsNorm 的工作树，候选在同一基线上接入 SiluMul。两臂使用相同 model benchmark executable。原始目录含两份库的 SHA-256、基线源码快照、源码差异、编译和运行命令、固定 seed 的模型与全部测试记录。

### Profiling

在 hidden 1024、intermediate 2816、16 Q / 4 KV heads、head_dim 64、2 层的合成 Llama GQA 上测试 O2/packed。对于 FP32、F16/BF16 W16-A32 和 BF16 A16，各测 128 和 512 token prompt 与 16 个生成 token。分开计时各 kernel invocation，诊断库不进入模型 A/B timing。

| 路径 | 基线 SiluMul Prefill 占比 | SIMD 后 SiluMul Prefill 占比 |
|---|---:|---:|
| FP32 | 3.87%–5.38% | 0.49%–0.61% |
| F16 W16/A32 + F16 KV | 3.17%–4.79% | 0.40%–0.51% |
| BF16 W16/A32 + BF16 KV | 4.01%–4.95% | 0.50%–0.57% |
| BF16 A16 | 12.50%–14.75% | 0.60%–0.70% |

优化后，RoPE 是所测 BF16 A16 的下一项小算子热点（Prefill 约 3.63%–4.45%），F16 KVUpdate 为 1.37%–1.84%，独立 BF16 Add 为 1.31%–1.68%。FP32 KVUpdate 已对连续 head 使用 memcpy。保留这些现有路径，后续分别验证 RoPE 的 FP64 旋转与位置边界、KV16 写入及 padding、Add 的 residual 舍入和别名，按目标负载决定投入。上表为包含诊断开销的相对占比，未采集 PMU。

### Prepared

命令使用 `aethermind_benchmark --benchmark_filter=^Benchmark(Bf16)?SiluMul/<mode>/ --benchmark_min_time=0.02s --benchmark_repetitions=3`。实际 registry、params builder 和调用。每个 mode 四个独立进程交错，另有四个同实现进程。以下为进程 CPU 时间中位数再取四进程中位数，单位 μs。计时前用 FP64 公式检查全部输出。

| 输入形状 | FP32 scalar → SIMD | BF16 scalar → SIMD |
|---|---:|---:|
| Decode 1 × 2819（带尾部） | 8.451 → 1.550 | 79.333 → 6.177 |
| Decode 1 × 11008 | 32.630 → 7.308 | 300.124 → 22.751 |
| Prefill 32 × 2816 | 255.594 → 52.196 | 2457.770 → 137.049 |
| Prefill 128 × 11008 | 4908.422 → 932.585 | 53802.422 → 2758.482 |

常规组的 scalar 和 SIMD 进程区间分离。1 × 16 和全部 gate 超界（1 × 2819）是回退对照，各自的区间重叠。极端 BF16 对照的中位数有变慢观察，保留全量样本。FP32 还用两个实际 prepared kernel 运行独立 SiLU + Mul；该对照有中间输出，原始结果列于 `prepared/summary.json`。Llama Dense 在 O1 和 O2 都直接构建 SiluMul，模型 A/B 的对照是同一语义节点上的 fused scalar 与 fused SIMD。

### 生产模型 A/B

使用 `tools/run_model_benchmark.py --preset o2-packed` 运行真实 ExecutionPlan 和公开 Generate。每组四对独立进程，按 AB/BA 交错，每进程 2 次 warmup 和 5 次测量，128 token 另有四个同实现进程。表中为四个进程内中位数再取中位数，单位 ms。

| 路径 / prompt 长度 | Prefill Execute | 15 步 Decode Execute 合计 | 公开 Generate |
|---|---:|---:|---:|
| FP32 / 128 | 106.60 → 79.37 | 64.29 → 56.44 | 157.17 → 130.78 |
| FP32 / 512 | 432.99 → 341.62 | 67.19 → 59.93 | 514.55 → 406.33 |
| F16 W16/A32 + F16 KV / 128 | 134.58 → 150.42 | 62.09 → 64.69 | 222.71 → 198.68 |
| BF16 W16/A32 + BF16 KV / 128 | 74.25 → 77.34 | 43.85 → 45.60 | 119.60 → 118.42 |
| BF16 W16/A32 + BF16 KV / 512 | 416.74 → 388.84 | 53.89 → 54.79 | 474.82 → 454.76 |
| BF16 A16 / 128 | 119.89 → 101.76 | 62.90 → 62.29 | 184.61 → 165.61 |
| BF16 A16 / 512 | 608.02 → 519.01 | 75.92 → 72.32 | 713.51 → 592.74 |

基线与候选共 28 对模型比较通过。同一组的生成 token、逐步轨迹及 KV 水位一致，raw/packed/KV/workspace/activation 字节数相同。BF16 A16 的 128-token 内部 Prefill 与 512-token Generate 各自的 A/B 进程区间分离。512-token 另有四个候选同实现进程，Generate 中位数为 596.11 ms，接近 A/B 候选的 592.74 ms。其 577.29–664.23 ms 区间反映本机进程间波动。各 A32 组的进程区间重叠，端到端收益为 Needs More Data。结论限定于该机器与合成 GQA。

O1/plain 的四种模型策略另外做短请求轨迹复验。两臂生成和 KV 水位一致，候选的 Prefill 与 Decode 均实际选中对应的 SIMD descriptor。O2 的 FP32 和 BF16 A16 allocation 诊断各执行一轮 Prefill 和三步 Decode，prepared Execute 内的 Runtime allocator 请求均为 0。

### 正确性

Release 聚焦集 **201/201**，包含固定 HF logits/KV 轨迹和 2048-token KV16/BF16 A16 校准。禁用 AVX2 的专项 **23/23**，ASan/UBSan **23/23**（LSan 关闭），主构建和 AVX2-off 的 ISA 检查通过。FP32 密集快域对拍使用独立 FP64 公式和 6 epsilon 相对误差预算。BF16 测试覆盖所有 gate 位模式及 up 的全部 128 个尾数，除 NaN payload 外结果逐位一致。向量主体和尾部、对 gate/up 的精确原地写入、非有限与 signed zero、定向舍入、非连续和广播也有回归。

**Accepted（当前本机生产路径）**。CPU-05 的 AddRmsNorm 和 SiluMul 均已接入生产，保留各自的数值与绑定合同。其余小算子按上面的分精度 profile 进入后续实验。原始目录为 `benchmark-results/operators/silu_mul/20261119T025157Z_4a048033_desktop-54h5mmi_cpu05-closeout/`，本地留存。
