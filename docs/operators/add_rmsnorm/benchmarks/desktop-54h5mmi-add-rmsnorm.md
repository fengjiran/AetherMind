# AddRmsNorm 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H，guest 16 logical CPUs，L3 24 MiB，effective AVX2/FMA/F16C 可用 |
| OS | WSL2 6.6.87.2-microsoft-standard-WSL2 |
| Build | GCC 14.2，Release `-O3 -DNDEBUG`，AddRmsNorm AVX2 独立翻译单元，禁用 fast-math、FMA contraction 和 LTO |
| Affinity | 单线程，独立进程绑 guest CPU 2 |

## CPU-05 首段：FP32 AddRmsNorm SIMD 与 O2 回归

基线为干净的 `4a048033`，候选为该 commit 加 CPU-05 工作树。原始目录包括载入的库、SHA-256、源码和运行命令。Scalar fused 和 SIMD fused 的模型 benchmark executable 逐字节相同。未融合对照使用候选库，与其余 O2 设置相同，仅在测量程序中关闭 `enable_fused_add_rms_norm`。运行诊断确认该对照执行独立 Add 和 RMSNorm。

Prepared 比较使用 `aethermind_benchmark --benchmark_filter=^BenchmarkAddRmsNorm/<mode>/ --benchmark_min_time=0.018s --benchmark_repetitions=3`。每组四个独立进程交错运行，另有四个同实现进程作噪声对照。通过实际 registry、params builder 和 invocation 测量，packed 产物由 CPU packer 生成。下面为进程内 CPU 时间中位数再取四个进程中位数，单位 μs。每个 case 重用同一份绑定和数据。

| Identity-packed FP32 | Add + RMSNorm | Fused scalar | Fused AVX2 |
|---|---:|---:|---:|
| Decode，1 × 16 | 0.0233 | 0.0232 | 0.0126 |
| Decode，1 × 513 | 0.434 | 0.406 | 0.328 |
| Decode，1 × 4096 | 3.460 | 3.173 | 2.548 |
| Prefill，32 × 4096 | 123.485 | 101.190 | 79.888 |
| Prefill，256 × 4096 | 1215.090 | 924.332 | 704.450 |

表中 fused AVX2 与 fused scalar 的四进程区间分离，后三组候选耗时下降约 20%–24%。5 列走 scalar，区间重叠。Plain 的 1 × 4096 下，独立 Add 加 plain AVX2 RMSNorm 为 1.144 μs，融合 AVX2 为 2.496 μs。独立 AVX2 RMSNorm 使用 FP32 归约，融合路径保留 FP64 合同。该对照表明本轮 O2/packed 的融合收益有明确的布局和 kernel 边界。

模型对照通过 `tools/run_model_benchmark.py --preset o2-packed` 分别跑真实 ExecutionPlan 和公开 Generate。两个 FP32 Llama fixture 均用 128 token prompt。冻结的 tiny HF MHA（hidden 16，2 层）生成 16 token，固定种子的合成 GQA（hidden 512，8 Q / 2 KV heads，head_dim 64，2 层）生成 24 token。两轮均为每个模型每臂四个交错进程和四个同实现进程，首轮每进程 1 次 warmup 和 5 次测量，复测为 3 次 warmup 和 11 次测量。共 64 份报告、32 组逐对比较，生成 token、逐步 Decode 轨迹与 KV 水位一致。raw/packed/KV/workspace/activation 预算量三臂一致。

| O2 复测中位数（ms，未融合 → 融合 scalar → 融合 AVX2） | Prefill Execute | 内部 Decode Execute 累计 | 公开 Generate |
|---|---:|---:|---:|
| tiny MHA | 7.375 → 6.261 → 7.135 | 1.389 → 1.178 → 1.295 | 10.947 → 9.343 → 10.247 |
| 合成 GQA | 21.409 → 24.191 → 21.910 | 18.069 → 20.509 → 18.867 | 42.275 → 45.214 → 44.545 |

两轮模型的三组区间均有重叠，端到端时延为 **Needs More Data**。复测的 GQA 中位数方向与首轮相反。另用诊断库独立测量每次 kernel invocation，记录相对占比，诊断库不用于上面的时延对照。两种模型的 Prefill 和 Decode 均实际执行 `cpu::add_rmsnorm_f32_w32_packed_identity_avx2`。该步在所测四个 phase/模型组合中占 kernel invocation 耗时约 0.08%–0.86%。合成 GQA 的 Prefill 中，矩阵投影合计约 67%，Attention 约 22%，SiluMul 约 7.6%，RoPE 约 2.0%，KVUpdate 约 0.1%。诊断包含逐 kernel 定时开销，暂无 PMU 样本和代表性大模型数据。

正确性门禁包括 Release 聚焦 173/173（含 HF logits/KV 轨迹、W16 与 BF16 A16 回归）、禁用 AVX2 编译的 AddRmsNorm 专项 11/11、ASan/UBSan 专项 11/11（LSan 关闭）及 ISA 构建检查。AddRmsNorm 对拍覆盖两个输出、FP32 加法舍入、FP64 平方和的大有限值、尾部、行 padding、合法 stride、拒绝非法 alias 和 O2 identity-packed 绑定。另用 `--mode allocations` 验证合成 GQA 的一次 Prefill 和七步 Decode，prepared Execute 区间的 Runtime allocator 请求均为 0。

**Accepted（当前生产路径与本机 prepared 收益）**。本机 M2 的模型 profiling 已完成。所测 GQA 的下一批小算子候选以 SiluMul Prefill 为先，扩大模型形状后再确定投入。raw 为 `benchmark-results/operators/add_rmsnorm/20261119T021116Z_4a048033_desktop-54h5mmi_cpu05-add-rmsnorm/`，本地保留。
