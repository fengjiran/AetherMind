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

## P2 W16 policy 生产路径 O1 baseline（admission tiny fixture）

- commit：`ea95702e`，dirty；含 P2 步骤 2–8 工作树补丁（model benchmark `--weights` 扩展与 W16 reference kernel）。
- 模型：`.models/inference-admission/` 的 `f16-auto`、`bf16-auto`、`f32-explicit`（tiny random Llama，2 层，hidden 16，intermediate 64，vocab 32000，4 heads，head_dim 4，untied MHA）；非 numerical fixture，不承载数值 oracle 结论。
- 命令：`python3 tools/run_model_benchmark.py --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark --model .models/inference-admission/<fixture> --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain --weights <policy> --warmup 1 --repetitions 20 --mode timing --variant <variant>`；四次独立采集，policy/fixture 对应 keep-fp16/f16-auto、keep-bf16/bf16-auto、convert-fp32/f16-auto、require-fp32/f32-explicit。
- 合同：verifier 72 项、driver 15 项、真实二进制 CLI 14 项通过；四个 report 均 accepted，internal 与 public Generate 轨迹一致。
- 结果（n=20 median；时间 µs；raw bytes 为去重逻辑权重视图）：

| policy | raw bytes | conversion | Load | Prepare | 内部 Prefill Execute | 单步 Decode Execute | 公开 Generate | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| keep-fp16 | 2,064,704 | 30.4 | 171.7 | 73.6 | 7268.1 | 2374.8 | 11945.5 | 22815, 27693, 6327 |
| keep-bf16 | 2,064,704 | 31.1 | 183.6 | 70.6 | 6162.7 | 2009.3 | 10552.2 | 18568, 25068, 22665 |
| convert-fp32（P1） | 4,129,088 | 382.6 | 102.4 | 51.5 | 650.5 | 221.0 | 1278.7 | 22815, 27693, 6327 |
| require-fp32 | 4,129,088 | — | 93.8 | 49.1 | 713.1 | 215.9 | 1189.3 | 22815, 27693, 6327 |

- 账本：W16 持久逻辑权重恰为 F32 的一半；Norm-only 转换约 30 µs，P1 全量转换约 383 µs；O1/plain 无 packed 副本。F16 keep 与 P1/F32 轨迹 token 一致；BF16 keep 因权重精度不同 token 不同，数值差异由算子/模型容差门禁覆盖（见提案 §2.6 冻结门禁），不以 token 逐位一致验收。
- raw：`benchmark-results/models/20261111T045408Z_ea95702e06cf_desktop-54h5mmi_w16-f16-o1-2995e91d/`、`20261111T045409Z_ea95702e06cf_desktop-54h5mmi_w16-bf16-o1-d67c8910/`、`20261111T045409Z_ea95702e06cf_desktop-54h5mmi_p1-convert-f16-o1-0778bd09/`、`20261111T045409Z_ea95702e06cf_desktop-54h5mmi_f32-require-o1-92fc7630/`。
- 结论：**Accepted，限定 O1 correctness/资源 baseline**。W16 reference 执行比 F32 慢约 10 倍，与 kernel 级 widening 成本一致（见 linear/embedding 算子记录）；内存目标（矩阵 2 字节、仅 Norm 转换）达成。不宣称吞吐收益；widening SIMD 化为首个优化候选，须独立证据门禁。


## P2 当前工作树 O1/O2 生产报告与账本复核

- commit：`ea95702e` + dirty。以 F16/BF16 admission fixture、O1/plain 与 O2/identity-packed 各采集一个 report，CPU affinity 0，warmup=1，每 report repetitions=3，均由 driver/verifier 接受。
- raw：`benchmark-results/operators/w16_execution/20261111T072153Z_ea95702e_desktop-54h5mmi_closeout/aethermind-p2-model-runs.json` 保存完整命令、返回值和四个不可覆盖的 report 路径。
- 字节口径补充：该 untied fixture 的 F32 矩阵为 4,128,768 bytes，W16 矩阵为 2,064,384 bytes，Norm 为 320 bytes F32。因此两种 raw 总量分别为 4,129,088 与 2,064,704 bytes。O2 的 packed storage 在独立字段报告。
- 结论：Accepted，限定执行链、policy 与资源报告可用性；本轮三个样本保留完整时间字段，性能推广需按更大 workload 与独立实验验收。
