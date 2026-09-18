# GEMM G0 Roofline 定位分析

- **状态**: Current
- **日期**: 2026-09-18
- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **工作包**: G0 合同与证据基线（Roofline 定位）
- **采集机**: `DESKTOP-54H5MMI` — Intel Core Ultra 9 285H（16 核、SMT off、单 NUMA、无 AVX-512/AMX），WSL2 kernel 6.6.87.2
- **数据来源**: baseline run `benchmark-results/operators/gemm/20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline/`（gitignored、不随仓库分发，**仅存在于上述采集机本地磁盘**）
- **Commit**: `5cbd378695bb90d97fa4273a7359bf2bfea20e8f`
- **关联证据**: [G0 baseline 验证报告](gemm_g0_baseline_validation_2026-09-18.md)（§6.3 Roofline 输入）、[配对 A/B 验证报告](gemm_g0_paired_ab_validation_2026-09-18.md)（噪声 floor）

> **机器归属**：Roofline 的计算峰值与带宽上限是采集机 `DESKTOP-54H5MMI` 的硬件属性（cpufp 123.74 GFLOP/s、STREAM Triad 22.11 GB/s）。其他机器的峰值不同，因此本文的百分比定位**不可跨机套用**；换机须用该机自己的峰值与 benchmark JSON 重算。

## 1. 目标与结论

把 G0 baseline 的全部 canonical case 放到单线程 roofline 上定位，回答"当前 reference 距硬件上限多远、瓶颈在哪一侧"。本分析只做定位与归因，不构成对 candidate 可达性能的承诺。

结论（2026-09-18 baseline，单线程，Intel Core Ultra 9 285H / WSL2）：

- **没有任何 canonical 形状接近 roofline**：最接近的是 M=1 decode 路径（K-contiguous / prepared Linear），约为记忆侧上限的 **25–27%**；M≥16 prefill 形状约为计算峰值的 **2.2–2.4%**，与反汇编"标量内层循环"一致；
- **两面瓶颈分明**：M=1（AI=0.5）位于记忆侧（ridge=5.60），M≥16（AI≥7.9）位于计算侧；
- **访问顺序是当前 reference 的首要瓶颈**：同一形状 N-contiguous 比 K-contiguous 慢 11–14×，两者都远低于 roofline——差距来自访存模式，不是算力；
- 小形状（L1 驻留）实测 4.2–4.9 GFLOP/s，仅为计算峰值的 ~4%，再次指向代码本身（标量、无 FMA 向量化）。

## 2. Roofline 模型与假设

| 参数 | 值 | 来源 |
|---|---|---|
| 计算峰值 P | 123.74 GFLOP/s | cpufp FMA(f32,f32,f32) 256b 单线程（run 内 `roofline.txt`） |
| 内存带宽（主口径） | 22.11 GB/s | STREAM Triad 单线程（2 读 1 写，与 GEMM 流式读写模式一致） |
| 内存带宽（乐观上界） | 37.05 GB/s | STREAM Copy 单线程 |
| Ridge（Triad 口径） | 5.60 FLOP/byte | P / 22.11 |
| Ridge（Copy 口径） | 3.34 FLOP/byte | P / 37.05 |

```text
GFLOP/s
123.74 ┤━━━━━━━━━━━━━━━━━━━━━●  P = 123.74（计算屋顶）
       │                  ╱
       │                ╱
       │              ╱     计算侧（AI > 5.60）
       │            ╱
       │          ╱  记忆侧（AI < 5.60）
       │        ╱
   0.0 └───────┬──┴──────────────► AI (FLOP/byte)
             0.5        5.60 (ridge, Triad)
```

口径与假设：

- 算术强度采用 compulsory traffic 模型：`AI = 2MNK / (4·(MK + KN + MN))`（FP32；A、B 各读一次，C 写一次，不重复计入）；
- 全部为**单线程**口径（采集时 `taskset -c 2`），不含多线程与 NUMA 效应；
- 定位值 `%cap = 实测 / min(P, AI×22.11)`；
- 每个 case 的"cap"只代表单线程硬件上限，不代表 candidate 应达到的验收值。

## 3. 数据来源与复现

- Run ID：`20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline`；
- 环境：Intel Core Ultra 9 285H（Arrow Lake-H，16 核、SMT off、单 NUMA）、WSL2（kernel 6.6.87.2-microsoft-standard-WSL2）、GCC 14.2.0、Release（-O3）、repetitions=10；
- 输入文件：`roofline.txt`（cpufp + STREAM）、`microkernel.json`（direct GEMM reference，N/K contiguous 两族）、`linear_hot.json` / `linear_streaming.json`（prepared Linear 两族）；取各 case 的 `mean` 聚合值；
- 采集命令与 checksum：见 run 内 `collect.sh` / `context.json`（各文件 SHA256）；benchmark 协议细节见 [G0 baseline 验证报告](gemm_g0_baseline_validation_2026-09-18.md) §5。

## 4. 逐形状定位

### 4.1 Direct GEMM reference（N-contiguous vs K-contiguous）

| 形状 M×K×N | AI (F/B) | cap (GFLOP/s) | N-contig (GFLOP/s) | N %cap | K-contig (GFLOP/s) | K %cap |
|---|---:|---:|---:|---:|---:|---:|
| 1×4096×4096 | 0.50 | 11.05 | 0.207 | 1.9% | 2.785 | 25.2% |
| 1×4096×11008 | 0.50 | 11.05 | 0.289 | 2.6% | 2.952 | 26.7% |
| 4×4096×4096 | 2.00 | 44.13 | 0.204 | 0.5% | 2.954 | 6.7% |
| 16×4096×4096 | 7.94 | 123.74（计算侧） | 0.250 | 0.2% | 2.938 | 2.4% |
| 1×33×31（L1 驻留） | 0.47 | 10.40 | 4.223 | 40.6% | 4.941 | 47.5% |
| 2×32×33（L1 驻留） | 0.89 | 19.69 | 4.217 | 21.4% | 4.780 | 24.3% |

### 4.2 Prepared Linear（hot vs streaming）

| 形状 M×K×N | AI (F/B) | cap (GFLOP/s) | hot (GFLOP/s) | streaming (GFLOP/s) | streaming logical GB/s | streaming %cap |
|---|---:|---:|---:|---:|---:|---:|
| 1×4096×4096 | 0.50 | 11.05 | 2.847 | 2.991 | 5.98 | 27.1% |
| 1×4096×6144 | 0.50 | 11.05 | 2.977 | 2.971 | 5.94 | 26.9% |
| 1×4096×11008 | 0.50 | 11.05 | 2.918 | 2.957 | 5.92 | 26.8% |
| 1×4096×22016 | 0.50 | 11.05 | 2.980 | 3.038 | 6.08 | 27.5% |
| 1×11008×4096 | 0.50 | 11.05 | 2.947 | 2.949 | 5.90 | 26.7% |
| 1×4096×32000 | 0.50 | 11.05 | 2.997 | 2.826 | 5.65 | 25.6% |
| 16×4096×4096 | 7.94 | 123.74 | 3.023 | 2.771 | 0.35 | 2.2% |
| 16×4096×22016 | 7.96 | 123.74 | 2.961 | 2.983 | 0.37 | 2.4% |
| 16×11008×4096 | 7.96 | 123.74 | 2.901 | 2.950 | 0.37 | 2.4% |
| 64×4096×4096 | 31.03 | 123.74 | 2.971 | 2.902 | 0.09 | 2.3% |

说明：`logical GB/s` 为 compulsory 流量速率（= Google Benchmark `bytes_per_second`，二者同值）；M=1 时它即有效权重流读带宽，M≥16 时很小（每 FLOP 流量低）。小形状（1×33×31、2×32×33）仅 hot 族采集，见 §5.4。

## 5. 解读

### 5.1 Decode（M=1）：记忆侧，实测约 26% 有效带宽

- AI=0.5，cap=11.05 GFLOP/s（Triad 口径）；最优路径（K-contiguous / prepared Linear）实测 2.79–3.04 GFLOP/s，即 cap 的 25–27%；换算为有效（compulsory）带宽 5.6–6.1 GB/s，约为单线程 Triad 的 26%；
- hot 与 streaming 几乎相同（差异小于噪声 floor），且工作集 64–500 MiB 均远超 25 MiB LLC——reference 在 M=1 由 DRAM 读取主导，权重驻留与否不改变量级；
- 上限含义：即便打满 Triad 带宽，单线程 M=1 也只有 ~11 GFLOP/s（当前 ~3.0 的 3.7×）；这是 code-level 优化的天花板参照，多线程目标需另行口径；
- 对照：N-contiguous 在同一形状只有 0.2–0.3 GFLOP/s（cap 的 2–3%），有效带宽被压到 0.4–0.6 GB/s。

### 5.2 Prefill（M≥16）：计算侧，实测约 2.4% 计算峰值

- AI 7.9–31，越过 ridge 5.60，cap 由 P=123.74 决定；实测 2.77–3.02 GFLOP/s = 2.2–2.4% 峰值；
- 该侧内存不是瓶颈：logical GB/s 仅 0.09–0.38（Triad 的 <2%）；
- 与 [G0 baseline 验证报告](gemm_g0_baseline_validation_2026-09-18.md) §7 的反汇编证据一致（内层循环标量、未向量化）——这一侧的上限由代码质量决定，向量化/loop 优化的理论空间约一个数量级。

### 5.3 访问顺序主导

- 同形状 N-contiguous vs K-contiguous：M≈1 时 13.4×、M=4 时 14.5×、M=16 时 11.8×，且两者都远在 roofline 之下——第一瓶颈是行/列访问方向（跨步访问破坏预取与 cache 局部性）；
- 这正是 04 提案把 "K-contiguous（Linear 实际布局）" 列为首个优化对象、并把 M=1 GEMV 排在 Prefill GEMM 之前的实测依据；
- 小形状（L1 驻留）两族差距缩小到 1.17×：跨步代价被 cache 吸收，佐证差距源于 DRAM 访问模式。

### 5.4 小形状（L1 驻留）：代码效率的直接对照

- 1×33×31 / 2×32×33 数据量 4–8 KiB，L1 驻留；实测 4.2–4.9 GFLOP/s；
- 按 DRAM 口径 cap（10.4 / 19.7）的 21–48% 会误导——它们不经过 DRAM。更有意义的对照是计算峰值：仅为 3.4–4.0%——即便完全 cache-resident，标量代码也远未跑满流水线；
- 用途：小形状可作为后续 scalar/SIMD microkernel 的 instruction-level 灵敏度对照，不适合做带宽判定。

## 6. 限制与后续

限制：

- WSL2 与宿主共享频率，cpufp/STREAM 为近似值；production 百分比级门禁需裸机隔离 CPU 复采后才可用；
- 单线程口径；多线程、NUMA 与 packed-weight 路径未在本分析建模；
- AI 为 compulsory 模型（理想 cache）；真实流量只会更高，实测点不会优于模型定位；
- 无硬件计数器（`perf stat` 在 WSL2 不可用）：DRAM/L3 归因基于工作集尺寸与 hot/streaming 对照，而非计数器证据；
- 百分比精度受测量噪声约束：配对协议噪声 floor（中位）为 microkernel 2.62% / hot 1.95% / streaming 3.08%，单进程重复可达 ±10–25%；本分析百分比仅为指示性定位，不作为门禁值。

后续用法建议：

- G1S（portable scalar optimized）以 **M=1 K-contiguous** 为首个灵敏度目标，判定看"有效带宽是否显著超过 5.6–6.1 GB/s"；
- SIMD microkernel 阶段以"距 123.74 GFLOP/s 峰值的百分比 + 反汇编证据"双证据判定计算侧进展；
- candidate 对照采用配对 A/B 协议，以各组噪声 floor 作为最小可信 delta 参照。
