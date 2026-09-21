# GEMM 基准记录 — 54H5MMI

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H（Arrow Lake-H，16 核，SMT off，AVX2+FMA+AVX-VNNI，无 AVX-512/AMX） |
| OS / kernel | WSL2，kernel 6.6.87.2-microsoft-standard-WSL2（与宿主共享物理核） |
| Compiler / build | GCC 14.2.0，C++20，Release（-O3） |
| 备注 | WSL2 下 `taskset` 仅咨询性 |

## reference 基线（direct GEMM）

- commit：`5cbd378695bb`、`fc5371323bec`（工作树含未提交改动）
- 命令：
  - `taskset -c 2 ./build-release/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_GemmF32Reference' --benchmark_min_time=1s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true`（单线程，每组独立进程）
  - 同参数，filter 为 `'BM_GemmF32Reference.*(/M:(8|9)/|/M:1/K:11008/|/M:1/K:4096/N:(6144|22016|32000))'`
- correctness：`aethermind_unit_tests` 80/80 通过；框架不变量测试 15/15 通过
- 结果（median of 10；GFLOP/s = 2·M·N·K / t）：

| 组 | case | median | 派生指标 |
|---|---|---|---|
| GEMM · K-contiguous | M=1, K=4096, N=4096 | 10.91 ms | 3.07 GFLOP/s |
| | M=1, K=4096, N=6144 | 16.68 ms | 3.02 GFLOP/s |
| | M=1, K=4096, N=11008 | 27.99 ms | 3.22 GFLOP/s |
| | M=1, K=4096, N=22016 | 60.11 ms | 3.00 GFLOP/s |
| | M=1, K=11008, N=4096 | 30.13 ms | 2.99 GFLOP/s |
| | M=1, K=4096, N=32000 | 87.75 ms | 2.99 GFLOP/s |
| | M=4, K=4096, N=4096 | 41.22 ms | 3.26 GFLOP/s |
| | M=8, K=4096, N=4096 | 88.06 ms | 3.05 GFLOP/s |
| | M=9, K=4096, N=4096 | 99.99 ms | 3.02 GFLOP/s |
| | M=16, K=4096, N=4096 | 165.66 ms | 3.24 GFLOP/s |
| GEMM · N-contiguous（整体比 K-contiguous 慢 10–14×） | M=1, K=4096, N=4096 | 150.04 ms | 0.22 GFLOP/s |
| | M=1, K=4096, N=6144 | 179.46 ms | 0.28 GFLOP/s |
| | M=1, K=4096, N=11008 | 286.01 ms | 0.32 GFLOP/s |
| | M=1, K=4096, N=22016 | 620.50 ms | 0.29 GFLOP/s |
| | M=1, K=11008, N=4096 | 478.64 ms | 0.19 GFLOP/s |
| | M=1, K=4096, N=32000 | 796.74 ms | 0.33 GFLOP/s |
| | M=4, K=4096, N=4096 | 603.64 ms | 0.22 GFLOP/s |
| | M=8, K=4096, N=4096 | 939.26 ms | 0.29 GFLOP/s |
| | M=9, K=4096, N=4096 | 1081.59 ms | 0.28 GFLOP/s |
| | M=16, K=4096, N=4096 | 1970.84 ms | 0.27 GFLOP/s |
| 边界形状 | M=1, K=33, N=31 | — | 4.6–5.5 GFLOP/s（L1 驻留） |
| | M=2, K=32, N=33 | — | 4.7–5.4 GFLOP/s |
| Roofline 上限（本机实测） | cpufp FMA f32；STREAM 单线程 | — | 123.74 GFLOP/s；Triad 22.11 / Copy 37.05 GB/s |
| 重复性检查（同二进制、另一进程） | microkernel | — | 5% 阈值自动判定被误判：9/12 REGRESS（逐 case 见 raw 的 `repeatability-comparison.txt`） |

- raw：
  - `benchmark-results/operators/gemm/20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_baseline/`（同一 run 的 Linear 组未收录）
  - `benchmark-results/operators/gemm/20260921T023356Z_fc5371323bec_DESKTOP-54H5MMI_reference-shapefill/`
- 结论：reference 基线 Accepted（本地观察）。`perf stat`/governor/microcode 在 WSL2 不可用，百分比级门禁需裸机重采

## reference 自身交错 A/B（环境噪声 floor）

- commit：`5cbd378695bb`
- 命令：`taskset -c 2 ./build-release/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_GemmF32Reference' --benchmark_min_time=1s --benchmark_repetitions=10`（A/B 各独立进程，相邻成对、保留 raw repetitions；脚本见 raw 目录 `collect_ab.sh`）
- 结果：|Δ| = |B−A| / A，aggregate mean real_time

| 组 | median \|Δ\| | p90 \|Δ\| | max \|Δ\| |
|---|---|---|---|
| microkernel | 2.62% | 7.09% | 13.20% |

- raw：`benchmark-results/operators/gemm/20260918T023421Z_5cbd378695bb_DESKTOP-54H5MMI_ab/`（含 Linear 组，未收录）
- 结论：同一二进制自比即有 case 超过 5%（max 13.20%，median 2.62%），5% 自动门禁在本机不可直接使用，判读需按 case 人工
