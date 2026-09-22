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

## blocked GEMM（4x16 micro-kernel，G2a 诊断级）

- commit：`42e2df28`（工作树 dirty）
- 命令：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter='M:16/K|M:64/K|M:128/K'`（WSL2，默认多进程无 taskset；本机 noise floor 结论见上）
- correctness：`aethermind_unit_tests --gtest_filter='*GemmAvx2*'` 33/33 通过（含 BlockedShapes 参数化）
- 实现形态：MR=4 × NR=16（acc 沿 N，每 k 步 A 标量广播 × B 行 16 列横向向量，B 转置 k-major 打包 [kc][NR]）；KC=512 面板、MB=48 行驻留；M/N 尾边条复用行对/多目标内核；运行时对 M≥9 K-contiguous 分派（G2b 注册 prepared path 后重验）
- 结果（单次 run，median n=1；GFLOP/s = 2·M·N·K / t）：

| case | baseline（scalar fallback，同入口） | candidate（blocked） | 提速 |
|---|---|---|---|
| M=16, K=4096, N=4096 | 58.78 ms（9.13 GF） | 10.19 ms（**52.7 GF**） | **5.8x** |
| M=64, K=4096, N=4096 | 238.58 ms（9.00 GF） | 29.79 ms（**72.1 GF**） | **8.0x** |
| M=128, K=4096, N=4096 | 487.53 ms（8.81 GF） | 53.70 ms（**80.0 GF**） | **9.1x** |
| M=128, K=4096, N=11008 | 1274.99 ms（9.05 GF） | 150.39 ms（**76.8 GF**） | **8.5x** |

- raw：`benchmark-results/operators/gemm/`（本次未存档 raw repetitions，单次观察）
- 结论：Needs More Data（方向 Accepted）——单次采样、未跑 repetitions/交错 A/B；80 GF 约达本机 FMA 上限 123.7 GF 的 65%。下一步：MB/NR/KC 调参矩阵 + 交错 A/B 复跑 + G2b 接入 Linear prepared path 后以 `BM_LinearPreparedHot` 为主门禁重验
