# GEMM G0 Roofline 定位分析 — DESKTOP-QHIHOGQ

- **状态**: Current（限 `DESKTOP-QHIHOGQ`）
- **日期**: 2026-09-19（采集窗口 2026-09-18T15:19Z–16:02Z）
- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **工作包**: G0 合同与证据基线（Roofline 定位）
- **采集机**: `DESKTOP-QHIHOGQ` — Intel Core i9-12900H（Alder Lake-H，6P+8E），WSL2 kernel 6.18.33.2，GCC 14.2.0，Release `-O3 -DNDEBUG`
- **数据来源**: baseline run `benchmark-results/operators/gemm/20260918T151928Z_162ab3e7583f_DESKTOP-QHIHOGQ_g0-baseline/`（gitignored，仅存在于本机）
- **Commit**: `162ab3e7583ffad11c9a7f5bbaa345dade665a9c`
- **关联证据**: [G0 baseline 验证报告（本机）](gemm_g0_baseline_validation_desktop-qhihogq_2026-09-19.md)、[G0 实验日志（本机）](g0-baseline-log-desktop-qhihogq.md)
- **前序机器**: 本文**不替代** [`gemm_g0_roofline_analysis_2026-09-18.md`](gemm_g0_roofline_analysis_2026-09-18.md)（采集机 `DESKTOP-54H5MMI` / Core Ultra 9 285H）。两台的 ceiling 不同，百分比不可互相引用。

> **机器归属**：Roofline 的计算峰值与带宽上限是本机的硬件属性。换机必须用该机自己的 ceiling 与 benchmark JSON 重算，按提案 §6.5「事件名与解释必须随 CPU 型号记录，不能跨微架构直接比较原始计数」。

## 1. 目标与结论

把本机 G0 baseline 的全部 canonical case 放到单线程 roofline 上定位，回答「当前 reference 距本机硬件上限多远、瓶颈在哪一侧」。本分析只做定位与归因，不构成对 candidate 可达性能的承诺。

结论（本机，单线程，`taskset -c 16`）：

- **没有任何 canonical 形状接近 roofline**：最接近的是 `M=1` decode 路径，约达记忆侧上限的 **27.7–29.4%**；`M≥16` prefill 形状仅达计算峰值的 **2.6–2.8%**，与反汇编「reference 内层为纯标量、零 packed FP 算术」一致；
- **两面瓶颈分明**：`M=1`（AI≈0.50）位于记忆侧，`M≥16`（AI≥7.94）位于计算侧，ridge point 为 **5.14 FLOP/byte**；
- **访问顺序是首要瓶颈**：同形状 N-contiguous 比 K-contiguous 慢 **16–18×**，但当工作集可驻留 L1 时差距收敛到 **1.18×**——证明慢化来自 stride 访存而非代码路径；
- 结构与采集机的定性结论一致（`M=1` 记忆侧约 26%、`M≥16` 计算侧约 2.4%），但**绝对百分比不同且不可迁移**。

## 2. 本机 ceiling 输入

单线程，`taskset -c 16`，测量程序见 `roofline.txt` 与采集脚本：

| 量 | 本机值 | 测量方式 | 备注 |
|---|---:|---|---|
| 计算峰值 P | **130.21 GFLOP/s** | 8 条独立 FMA 链的 `vfmadd` 循环，best-of-7 | 对应 32 FLOP/cycle × ≈4.07 GHz；**实测值**，非 cpufp 理论值 |
| STREAM Triad | **25.32 GB/s** | `c[i] = a[i] + s*b[i]`，3×384 MiB double 缓冲，best-of-7 | 主口径，用于记忆侧 ceiling |
| STREAM Copy | **41.12 GB/s** | `memcpy`，2×384 MiB，best-of-7 | glibc memcpy 在大尺寸下切换为 non-temporal store，故为**上界**，不是经典 STREAM Copy |
| ridge point | **5.14 FLOP/byte** | P / Triad | |

工作集 3×384 MiB = 1152 MiB，远大于本机 24 MiB L3，确保 Triad 为 DRAM-bound。

**测量边界**：`perf` 不可用，因此没有硬件计数器可验证实际 DRAM traffic 或频率；P 与带宽均为 best-of-7 的经验上限，`taskset` 在本机只是咨询性绑定（见实验日志 G0-QHIHOGQ-004），物理核与 P/E core 类别不可控。这些 ceiling 应视为**近似值**。

## 3. 算术强度模型

采用提案 §6.5 要求的 compulsory-traffic 模型，FP32：

```text
FLOPs = 2 * M * N * K
bytes = sizeof(float) * (M*K + N*K + M*N)
AI    = FLOPs / bytes
```

`logical GB/s` 由 benchmark counter 直接给出，它是**流量下界，不是实测 DRAM traffic**。

| 形状 | AI (FLOP/byte) | 相对 ridge=5.14 |
|---|---:|---|
| M=1, K=4096, N=4096 | 0.50 | 记忆侧 |
| M=1, K=11008, N=4096 | 0.50 | 记忆侧 |
| M=2, K=32, N=33 | 0.89 | 记忆侧 |
| M=4, K=4096, N=4096 | 2.00 | 记忆侧 |
| M=16, K=4096, N=4096 | 7.94 | 计算侧 |
| M=64, K=4096, N=4096 | 31.03 | 计算侧 |

## 4. 定位表

「%ceiling」按适用侧取 `min(P, AI × Triad)`。完整机器可读版本见 `roofline-positioning.txt`。

### 4.1 prepared Linear（production path，`kernel=cpu::linear_f32_reference`）

| 形状 | GFLOP/s | 侧 | 有效 GB/s | %ceiling |
|---|---:|---|---:|---:|
| hot M=1 K=4096 N=4096 | 3.505 | 记忆 | 7.013 | 27.7% |
| hot M=1 K=4096 N=6144 | 3.524 | 记忆 | 7.050 | 27.8% |
| hot M=1 K=4096 N=11008 | 3.585 | 记忆 | 7.172 | 28.3% |
| hot M=1 K=11008 N=4096 | 3.572 | 记忆 | 7.147 | 28.2% |
| hot M=1 K=4096 N=22016 | 3.704 | 记忆 | 7.409 | 29.3% |
| hot M=1 K=4096 N=32000 | 3.719 | 记忆 | 7.440 | 29.4% |
| hot M=16 K=4096 N=4096 | 3.690 | 计算 | — | 2.8% |
| hot M=16 K=4096 N=22016 | 3.486 | 计算 | — | 2.7% |
| hot M=16 K=11008 N=4096 | 3.413 | 计算 | — | 2.6% |
| hot M=64 K=4096 N=4096 | 3.432 | 计算 | — | 2.6% |
| hot M=1 K=33 N=31（L1 驻留） | 4.582 | 记忆 | 9.738 | 38.5% |
| hot M=2 K=32 N=33（L1 驻留） | 4.691 | 记忆 | 5.268 | 20.8% |
| streaming M=1 K=4096 N=4096 | 3.598 | 记忆 | 7.200 | 28.4% |
| streaming M=1 K=4096 N=32000 | 3.696 | 记忆 | 7.393 | 29.2% |
| streaming M=16 K=4096 N=4096 | 3.556 | 计算 | — | 2.7% |

### 4.2 direct GEMM reference（microkernel 层，访问顺序对照）

| 形状 | K-contig | %ceiling | N-contig | %ceiling | N/K 慢化 |
|---|---:|---:|---:|---:|---:|
| M=1 K=4096 N=4096 | 3.659 | 28.9% | 0.209 | 1.7% | 17.5× |
| M=1 K=4096 N=11008 | 3.766 | 29.8% | 0.227 | 1.8% | 16.6× |
| M=4 K=4096 N=4096 | 3.634 | 7.2% | 0.199 | 0.4% | 18.3× |
| M=16 K=4096 N=4096 | 3.468 | 2.7% | 0.210 | 0.2% | 16.5× |
| M=1 K=33 N=31（L1） | 4.435 | 37.2% | 3.766 | 31.6% | 1.18× |
| M=2 K=32 N=33（L1） | 4.456 | 19.8% | 3.779 | 16.8% | 1.18× |

单位 GFLOP/s。

## 5. 归因

1. **记忆侧（`M=1` Decode）**：有效带宽 7.0–7.4 GB/s，为 Triad 25.32 GB/s 的 28–29%。reference 的 `row → col → k` 循环对权重做的是 strided 重复流读，无法达到纯 Triad 的带宽利用率。**G1S/G1V 在 `M=1` 上的首个判据可直接写成「有效带宽是否显著超过 7.4 GB/s」**，这比 GFLOP/s 更贴近真实瓶颈。
2. **计算侧（`M≥16` Prefill）**：仅达峰值 2.6–2.8%，且反汇编证实内层为纯标量 `mulsd`/`addsd`、零 packed FP 算术。这一侧的空间几乎完全来自 SIMD 宽度与 register blocking，即 G1V/G3 的目标。
3. **`hot ≈ streaming`**：`M=1` 下 hot 3.51–3.72 与 streaming 3.60–3.70 GFLOP/s 几乎相同，说明权重工作集（64 MiB 起）已远超 24 MiB L3，reference 本身就是 DRAM-bound，cache 复用策略在 reference 上没有可挖掘空间。
4. **访问顺序**：K-contiguous（Linear 的实际布局）比 N-contiguous 快 16–18×，但 L1 驻留时只差 1.18×。慢化源于大形状下 N-contiguous 的 stride 访存（cache line 利用率与 TLB），不是代码路径差异。由于 production Linear 走 K-contiguous，**「修复 N-contiguous」不是优化目标**，它只是 microkernel 层的对照实验。
5. **binding 与 packing 不在 roofline 上**：binding 0.414–0.433 µs/call 且与 shape 无关，packing 5.45–6.13 GB/s，两者都是 cold-path 成本，按 §6.1 单独报告、不混入 compute geomean。

## 6. 与采集机结论的关系

| 维度 | `DESKTOP-54H5MMI`（285H） | `DESKTOP-QHIHOGQ`（12900H） | 可否跨机引用 |
|---|---|---|---|
| 计算峰值 | 123.74 GFLOP/s（cpufp 理论） | 130.21 GFLOP/s（实测 FMA 循环） | **否**，口径也不同 |
| Triad | 22.11 GB/s | 25.32 GB/s | 否 |
| ridge | 5.60 | 5.14 | 否 |
| `M=1` 记忆侧占比 | ~26% | 27.7–29.4% | 否（仅定性可比） |
| `M≥16` 计算侧占比 | 2.2–2.4% | 2.6–2.8% | 否（仅定性可比） |
| N/K-contiguous 慢化 | 10–14× | 16–18× | 否 |
| reference 内层未向量化 | 是 | 是 | **是**（源码事实） |
| `hot ≈ streaming` at `M=1` | 是 | 是 | 定性可参考 |

可跨机引用的只有**源码层面的事实**（reference 未向量化、访问顺序主导、`M=1` 为记忆侧）；所有百分比与绝对数值都必须按机器各自成立。

## 7. 局限

- `perf` 不可用：无法验证实际 DRAM traffic、cache-miss、branch-miss 或频率行为，记忆侧定位只能依赖 compulsory-traffic 模型；
- governor 与真实 microcode 不可得，频率在采集期间不可观测；
- `taskset` 为咨询性绑定，P/E core 类别不可控（见实验日志 G0-QHIHOGQ-004），ceiling 可能混合了两类核的行为；
- Copy 使用 glibc `memcpy`，大尺寸下走 non-temporal store，41.12 GB/s 是上界而非经典 STREAM Copy，因此本文的 ridge 只用 Triad 计算；
- 未建模 cache 级（L2/L3）分层带宽，因此 L1 驻留小形状的百分比只具指示性；
- 百分比受本机跨进程偏移约束（microkernel 1.72% / hot 3.89% / streaming 5.17% median），**不替代门禁值**。

## 8. 后续动作

- G1S 本机 candidate 比较沿用交错 A/B，并以各组噪声 floor 而非固定 5% 判读；`M=1` 优先看有效带宽是否超过 7.4 GB/s；
- G1V 需在本机重采 ceiling（AVX2+FMA 8-wide 的实测峰值），因为 G1S 的自动向量化目前只到 SSE2 4-wide；
- 裸机环境就绪后重采 perf 计数器、governor、真实 microcode 与 P/E 分离数据，并重算 roofline；
- 多线程口径与 packed-weight 路径的定位待 G2/G6 之后补充。
