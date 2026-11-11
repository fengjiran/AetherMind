# Linear 基准记录 — desktop-54h5mmi

| 项 | 值 |
|---|---|
| CPU | Intel Core Ultra 9 285H；客体可见 16 logical CPU；AVX2/FMA/F16C |
| OS / kernel | Ubuntu 24.04.1 LTS / 6.6.87.2-microsoft-standard-WSL2 |
| Compiler / build | GCC 14.2.0；build，Release -O3 -DNDEBUG，C++20，AETHERMIND_ENABLE_AVX2=ON，AETHERMIND_ENABLE_F16C=ON；sanitizer=none |
| 备注 | WSL2；taskset -c 0 单核固定；数值为有限样本观测，不推导真实模型加速 |

## P2 O1/plain W16 reference kernel baseline（F16/BF16 权重，F32 激活/累加）

- commit：`ea95702e`，dirty；含 P2 步骤 2–7 工作树补丁。
- 命令：`taskset -c 0 ./build/tests/benchmark/aethermind_benchmark --benchmark_filter='BM_LinearW16.*|BM_LinearPreparedHot.*' --benchmark_out=<raw>/linear-w16-benchmark.json --benchmark_out_format=json`。
- 正确性守卫：每个 case 计时前对独立 FP64 oracle 按冻结合同 `gamma(2K)*sum|A·W| + 1e-6` 校验（K 升序、分离 F32 乘加）；全部通过，无 SkipWithError。
- 结果（prepared hot，median real_time，n 见 JSON iterations；同构建同 affinity）：

| M/K/N | F32 reference | W16 F16 reference | W16 BF16 reference |
|---|---:|---:|---:|
| 1/4096/4096（Decode GEMV） | 11.70 ms | 65.49 ms | 51.11 ms |
| 1/4096/32000（LM head） | 135.73 ms | 551.34 ms | 399.30 ms |
| 16/4096/4096（Prefill） | 288.89 ms | 1035.73 ms | 781.52 ms |
| 64/4096/4096（Prefill） | 643.11 ms | 4358.97 ms | 3286.86 ms |

  streaming（256 MiB 权重工作集轮换）与 hot 基本一致：1/4096/4096 F16 64.07 ms、BF16 50.26 ms，说明当前瓶颈是逐元素 scalar widening 而非权重缓存缺失。binding specialization（冷路径 params 构建）约 0.37–0.45 µs。
- raw：`benchmark-results/operators/linear/20261111T044713Z_ea95702e_desktop-54h5mmi_w16-o1-baseline/`（metadata.json、commands.json、JSON、log）。
- 结论：**Accepted，限定 O1 correctness baseline**。W16 reference 比 F32 reference 慢约 4–7 倍，成本集中在内层逐元素 `LoadW16`（memcpy + dtype 分支 + 软件转换）；BF16 快于 F16（widening 为移位）。这是正确性基线，不是性能候选；不宣称吞吐收益。优化顺序决策见提案 P2 步骤 8 记录：先 widening SIMD 化（F16C/AVX2），再 Decode GEMV 与 Prefill 分块。

## W16 B-panel 布局评估（P2 步骤 9 决策）

- commit：`ea95702e`，dirty；依据上述 O1 baseline 与 O2/packed 验收。
- 观察：streaming 与 hot 时延基本一致，瓶颈是 scalar widening 而非权重布局/缓存缺失；identity packing 已保持 2 字节元素并支持 QKV/GateUp composite 拼接，O2 fused consumer 全部 resolve。
- 决策：**暂不新增 W16 B-panel**。现有 `kCpuBPanelF32Kc512Nr16` 按冻结合同仅限 F32，W16 不得复用其布局身份；新物理格式须独立 `PackingLayout` 并同步 byte formula、packer、validator、artifact key 与 consumer。在 widening SIMD 化落地且 profiling 证明布局收益之前，新增 W16 B-panel 缺乏证据支撑，列为后续 O4 优化批次候选。
- 结论：**Accepted（评估决策）**；O2/packed 首批以 identity 布局交付，W16 B-panel 不进入本批支持范围。