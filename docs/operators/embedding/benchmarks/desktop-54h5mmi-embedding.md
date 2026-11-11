# Embedding 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA/F16C |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；build，Release -O3 -DNDEBUG，C++20，AETHERMIND_ENABLE_AVX2=ON，AETHERMIND_ENABLE_F16C=ON；sanitizer=none |
| 备注 | WSL2；taskset -c 0 单核固定；数值为有限样本观测，不推导真实模型加速 |

## P2 O1/plain W16 reference kernel baseline（vocab 32000 × hidden 4096）

- commit：`ea95702e`，dirty；含 P2 步骤 2–7 工作树补丁。
- 命令：`taskset -c 0 ./build/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_Embedding.*' --benchmark_out=<raw>/embedding-w16-benchmark.json --benchmark_out_format=json`。
- 正确性守卫：每个 case 计时前按独立 widening oracle 逐位校验所选行（F32 为精确拷贝，W16 为 bit-exact widening）；全部通过，无 SkipWithError。
- 结果（prepared hot，median real_time）：

| tokens | F32 reference | W16 F16 reference | W16 BF16 reference |
|---|---:|---:|---:|
| 1（Decode） | 122 ns | 6.30 µs | 971 ns |
| 16（Prefill） | 6.00 µs | 114.11 µs | 15.55 µs |
| 64（Prefill） | 25.09 µs | 415.07 µs | 68.37 µs |

  W16 权重工作集：F16/BF16 各 250 MiB，F32 500 MiB。F16 行 widening 当前复用 trusted scalar batch（约 1.5 ns/元素）；BF16 约 0.24 ns/元素。
- raw：`benchmark-results/operators/embedding/20261111T044713Z_ea95702e_desktop-54h5mmi_w16-o1-baseline/`（metadata.json、commands.json、JSON、log）。
- 结论：**Accepted，限定 O1 correctness baseline**。Embedding 的 W16 成本完全来自行级 widening；现有 F16C/AVX2 batch 转换已具备，行级接入是首个优化候选（预期接近 memcpy 量级）。不宣称吞吐收益；优化效果须由本算子执行链独立测量。

## F16C / AVX2 行级 widening 接入

- commit：`ea95702e` + dirty。baseline/candidate 使用同一构建，scalar 由空 effective features 选择，selected 由真实 CPU/OS features 选择，prepared 后仅执行 kernel。
- 命令：`taskset -c 0 ./cmake-build-release/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_EmbeddingPrepared... --benchmark_min_time=0.01s --benchmark_repetitions=3 --benchmark_enable_random_interleaving=true`，完整 filter 和 JSON 位于 raw。n=3，为本机限定观察。
- 正确性：独立 bit/数值 oracle、完整 65,536 patterns、非对齐/tail、feature policy fallback、O1/O2 model regression 与零分配通过。

| dtype / tokens，hidden=4096 | Scalar median CPU time | Selected median CPU time |
|---|---|---|
| f16 / 1 | 4053.0 ns | 296.4 ns |
| f16 / 16 | 75741.4 ns | 6762.8 ns |
| bf16 / 1 | 809.3 ns | 215.6 ns |
| bf16 / 16 | 11934.0 ns | 6002.4 ns |

- raw：`benchmark-results/operators/w16_execution/20261111T072153Z_ea95702e_desktop-54h5mmi_closeout/` 中的 `aethermind-p2-embedding-ab-final.json`、log、commands/metadata 与测试 XML；保存所有 repetitions。
- 结论：Accepted，限定本机 prepared 行级 kernel。两种 dtype 的 plain/identity-packed descriptor 只在所需有效 ISA 可用时优先选择，reference fallback 保留。该记录的时间只覆盖 gathered rows 的 hot workload。
