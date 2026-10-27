# 模型基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；Release -O3 -DNDEBUG；sanitizer=none |
| 执行 | 单请求；继承 affinity 0–15；无核类别/SMT 裸机结论 |
| 可见配置 | 客体总内存 66,886,721,536 bytes、NUMA node 0；governor/turbo unavailable |
| 模型 | 固定 revision 的 tiny random Llama FP32 untied MHA；来源与摘要见 numerical fixture |
| 测量边界 | 预读文件 cache；每 sample 先 internal 后 public Generate；非冷启动 |

## 模型性能采集链与同实现噪声观察

- commit：ff2f94797033938e8e5d7c2e39810904142f67fd，dirty；包含第三至第五批工作树补丁。
- 命令：`python3 tools/run_model_benchmark.py --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark --model .models/inference-numerical/mha-untied/model --prompt 1,7,3 --max-new-tokens 8 --preset o1-plain --warmup 1 --repetitions 7 --mode timing`；O2 及完整 sweep 命令见 raw collection.json。
- 顺序：A1/B1/A2/B2/A3/B3/A4/B4，每次独立进程；随后扫描 P=31/127/511/2046。
- raw：`benchmark-results/models/20261027T153722Z_ff2f94797033_desktop-54h5mmi_baseline-suite-a7fa449c/`，各原始 report 由 collection.json 引用。
- 数值/合同：正式两种 Release 构建及 sanitizer CLI 各 14 项通过；参考模型/轨迹范围见 [第五批验收](../../model_benchmark_validation_2026-10-27.md)。
- 结论：**Needs More Data**。以下是本机有限样本观测，没有 Accepted 性能候选或 production priority 变更。

下表为每个独立进程的 median（ms），保留四对全部结果，n=7：

| 进程 | Prepare | 内部 Prefill Execute | 连续 Decode Execute 总和 | 公开 Generate |
|---|---:|---:|---:|---:|
| aa-o1-1 | 0.046 | 0.634 | 1.454 | 2.178 |
| ab-o2-1 | 0.453 | 0.598 | 1.319 | 1.972 |
| aa-o1-2 | 0.050 | 0.663 | 1.536 | 2.094 |
| ab-o2-2 | 0.496 | 0.595 | 1.359 | 2.108 |
| aa-o1-3 | 0.046 | 0.688 | 1.440 | 2.124 |
| ab-o2-3 | 0.566 | 0.599 | 1.495 | 2.085 |
| aa-o1-4 | 0.045 | 0.657 | 1.508 | 2.125 |
| ab-o2-4 | 0.547 | 0.636 | 1.466 | 2.080 |

同实现 Generate 的 process-median 最大两两相对离散观测为 O1 4.046%、O2 6.906%，连续 Decode Execute 为 O1 6.670%、O2 13.363%。这不是可靠的全局 noise threshold，也不构成小于该数值即等价的判定。

不同 prompt 的公开 Generate median（ms）：P=31/127 每进程 n=7；P=511/2046 每进程 n=3。G=8，唯 P=2046 时 G=3，最终 KV=2048。

| P / G | O1/plain | O2/packed |
|---|---:|---:|
| 31 / 8 | 9.412 | 9.257 |
| 127 / 8 | 31.302 | 31.662 |
| 511 / 8 | 131.962 | 129.991 |
| 2046 / 3 | 555.270 | 507.611 |

内部阶段和公开 Generate 来自不同请求、不同缓存时点，不能将列间差值当作调用开销，不能将各阶段相加重建 Generate。各过程样本量不足以给出稳定 tail 或置信结论；不以这组 tiny/WSL 结果推导真实模型加速。

首轮采样因缺 THP 输入 metadata 已保留并标记不用于最终基线；详见验收记录。计时、profile、Runtime allocator 诊断分进程，全部 repetitions 和两两对照保留。下一步需要代表性 workload、可控平台与 PMU/重复性证据，再评审优化候选。
