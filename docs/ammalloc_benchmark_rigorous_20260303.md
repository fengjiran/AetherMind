# ammalloc 严谨性能对比报告（2026-03-03）

## 1. 测试目标

对 `ammalloc` 与系统 `std::malloc/std::free` 做一轮更严谨的对比，重点关注：

- 固定 CPU 亲和性后的稳定性与吞吐
- 多次重复下的波动区间
- 单线程与多线程场景下的性能差异

---

## 2. 测试环境

- 日期：`2026-03-03`
- CPU：`12th Gen Intel(R) Core(TM) i9-12900H`
- 逻辑核：`20`（10 物理核 / 20 线程）
- 亲和性绑定：`0,2,4,6,8,10,12,14,16,18`（每个物理核取 1 个线程）
- benchmark 可执行文件：`build/tests/benchmark/aethermind_benchmark`

---

## 3. 测试方法

### 3.1 执行参数

- `--benchmark_repetitions=5`
- `--benchmark_report_aggregates_only=true`
- `--benchmark_min_time=0.01s`
- `--benchmark_out_format=json`

### 3.2 数据来源

- 单线程/随机/DeepChurn：
  - `/tmp/ammalloc_st_20260303.json`
- 多线程：
  - `/tmp/ammalloc_mt_20260303.json`
- 合并汇总（mean/stddev/CV/95%CI）：
  - `/tmp/ammalloc_rigorous_combined_summary.tsv`
- 配对对照（am vs std）：
  - `/tmp/ammalloc_rigorous_pairwise_20260303.tsv`

> 注：`CV` 为变异系数（标准差 / 均值），`95%CI` 为均值 95% 置信区间半宽（近似 `1.96 * stddev / sqrt(5)`）。

---

## 4. 核心结果（am vs std）

`speedup(std/am) > 1` 表示 `ammalloc` 更快。

| Benchmark | am mean (ns) | std mean (ns) | speedup(std/am) | am CV | std CV |
|---|---:|---:|---:|---:|---:|
| `BM_Malloc_Churn<8, 1, ...>` | 3.519 | 6.714 | 1.91x | 1.96% | 1.04% |
| `BM_Malloc_Churn<64, 1, ...>` | 3.483 | 8.329 | 2.39x | 1.43% | 1.09% |
| `BM_Malloc_Churn<8, 256, ...>` | 3.643 | 6.688 | 1.84x | 1.09% | 2.62% |
| `BM_Malloc_Churn<64, 256, ...>` | 3.625 | 7.622 | 2.10x | 0.44% | 4.13% |
| `BM_Malloc_Churn<8, 1024, ...>` | 3.689 | 7.046 | 1.91x | 0.48% | 2.70% |
| `BM_Malloc_Churn<4096, 1024, ...>` | 7.474 | 17.257 | 2.31x | 0.22% | 3.17% |
| `BM_am_malloc_free_pair_random_size` | 8.703 | 82.323 | 9.46x | 2.46% | 1.28% |
| `BM_Malloc_Deep_Churn<8, 2000, ...>` | 8510.896 | 19948.091 | 2.34x | 0.65% | 2.38% |
| `BM_am_malloc_multithread<8>/threads:1` | 4177.409 | 20978.295 | 5.02x | 1.29% | 0.49% |
| `BM_am_malloc_multithread<8>/threads:2` | 4112.224 | 21191.096 | 5.15x | 0.57% | 0.48% |
| `BM_am_malloc_multithread<8>/threads:4` | 4302.045 | 21159.073 | 4.92x | 2.73% | 0.63% |
| `BM_am_malloc_multithread<8>/threads:8` | 5092.295 | 23452.451 | 4.61x | 7.15% | 2.28% |
| `BM_am_malloc_multithread<8>/threads:16` | 8159.673 | 23997.927 | 2.94x | 10.96% | 4.20% |
| `BM_am_malloc_multithread<64>/threads:1` | 4393.304 | 21179.544 | 4.82x | 0.60% | 0.39% |
| `BM_am_malloc_multithread<64>/threads:2` | 4387.153 | 21636.487 | 4.93x | 1.00% | 0.95% |
| `BM_am_malloc_multithread<64>/threads:4` | 4493.355 | 21658.601 | 4.82x | 2.24% | 2.16% |
| `BM_am_malloc_multithread<64>/threads:8` | 6042.418 | 24408.043 | 4.04x | 4.79% | 4.21% |
| `BM_am_malloc_multithread<64>/threads:16` | 10029.618 | 25485.575 | 2.54x | 7.74% | 3.93% |

---

## 5. 波动区间（示例）

### 5.1 高并发下波动明显增加（`real_time`）

- `BM_am_malloc_multithread<8>/threads:16`
  - mean: `8159.673 ns`
  - stddev: `894.219 ns`
  - CV: `10.96%`
  - 95%CI 半宽: `±783.818 ns`

- `BM_std_malloc_multithread<8>/threads:16`
  - mean: `23997.927 ns`
  - stddev: `1008.265 ns`
  - CV: `4.20%`
  - 95%CI 半宽: `±883.783 ns`

### 5.2 随机大小 + 16 线程是最抖场景

- `BM_am_malloc_multithread_random<1000>/threads:16`
  - mean: `2203287.762 ns`
  - stddev: `355473.549 ns`
  - CV: `16.13%`
  - 95%CI 半宽: `±311586.304 ns`

---

## 6. 结论

- 在本机上，`ammalloc` 在全部配对场景均快于 `std::malloc`，加速比约 `1.84x ~ 9.46x`。
- 最大优势出现在随机尺寸单线程对（`~9.46x`）。
- 多线程下仍显著领先，但线程数升高到 `16` 后，`real_time` 波动明显增加（`CV` 上升）。

---

## 7. 复现实验命令

```bash
# 单线程/随机/DeepChurn
taskset -c 0,2,4,6,8,10,12,14,16,18 \
  ./build/tests/benchmark/aethermind_benchmark \
  --benchmark_filter='BM_(Malloc_Churn.*|am_malloc_free_pair_random_size|std_malloc_free_pair_random_size|Malloc_Deep_Churn.*)' \
  --benchmark_min_time=0.01s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=/tmp/ammalloc_st_20260303.json \
  --benchmark_out_format=json

# 多线程
taskset -c 0,2,4,6,8,10,12,14,16,18 \
  ./build/tests/benchmark/aethermind_benchmark \
  --benchmark_filter='BM_(am_malloc_multithread|std_malloc_multithread).*' \
  --benchmark_min_time=0.01s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=/tmp/ammalloc_mt_20260303.json \
  --benchmark_out_format=json
```

---

## 8. 图表生成

已提供脚本：

- `tools/plot_ammalloc_benchmark.py`

使用方式：

```bash
python3 tools/plot_ammalloc_benchmark.py \
  --input /tmp/ammalloc_rigorous_pairwise_20260303.tsv \
  --out-dir docs/figures/ammalloc_bench_20260303
```

输出图表：

- 若安装了 `matplotlib`：输出 PNG
  - `speedup_std_over_am.png`
  - `latency_mean_ns.png`
  - `throughput_multithread_gibs.png`
  - `cv_variability_pct.png`
- 若未安装 `matplotlib`：自动输出同名 SVG
  - `speedup_std_over_am.svg`
  - `latency_mean_ns.svg`
  - `throughput_multithread_gibs.svg`
  - `cv_variability_pct.svg`
