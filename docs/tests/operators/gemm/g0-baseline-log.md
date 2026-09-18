# GEMM G0 基线实验日志

- **算子/工作包**: GEMM / G0 合同与证据基线
- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **采集机**: `DESKTOP-54H5MMI` — Intel Core Ultra 9 285H（Arrow Lake-H，16 核、SMT off、单 NUMA、AVX2+FMA+AVX-VNNI、无 AVX-512/AMX），WSL2 kernel 6.6.87.2
- **日志范围**: 2026-09-18 首次机器级 baseline、独立复跑与 Roofline 定位；2026-09-19 提案 §7 汇总范围值归档
- **原始数据位置**: `benchmark-results/operators/gemm/20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline/` — 该目录 gitignored、不随仓库分发，**仅存在于上述采集机本地磁盘**，在其他机器上不存在且不可恢复
- **正式报告**: [G0 baseline validation](gemm_g0_baseline_validation_2026-09-18.md)；[Roofline 定位分析](gemm_g0_roofline_analysis_2026-09-18.md)

> **适用范围**：本日志中所有性能数值、噪声 floor 与 Roofline 百分比只对采集机 `DESKTOP-54H5MMI` 成立。按提案 §6.7（baseline/candidate 必须同机）与 §6.5（不得跨微架构比较原始计数），这些数据不可作为其他机器的基线或门禁参照；其他目标机必须各自重采并归档自己的 raw artifact。下文出现的“本机”一律指该采集机。

## 日志索引

| 日期 | 实验 ID | 假设 | 结论 | 状态 |
|---|---|---|---|---|
| 2026-09-18 | G0-BASELINE-001 | 当前 benchmark 能分离 direct GEMM、prepared Linear、binding、streaming 与 packing 成本 | 结构和调用链有效；WSL2 噪声不支持百分比级 production gate | Needs More Data |
| 2026-09-18 | G0-BASELINE-002 | 相邻配对交错 A/B 与 raw repetitions 能给出可归因的本机噪声 floor | canonical compute 中位噪声 ~2–3%（顺序两轮为 ±10–25%）；binding/小形状仍超 5% | Closed locally / Production gate Needs More Data |
| 2026-09-18 | G0-BASELINE-003 | 现有 baseline 数据足以对全部 canonical 形状做单线程 Roofline 定位 | canonical 形状 ≤27% cap；M=1 记忆侧（~26% Triad）/ M≥16 计算侧（~2.4% 峰值）；访问顺序主导 | Closed locally（指示性定位） |
| 2026-09-19 | G0-BASELINE-004 | 提案 §7 粘贴的本机汇总范围值在本目录已有权威副本 | 部分范围值无副本，已补录并标注与单 case median 的口径差异；提案侧删除副本 | Closed locally |

## 2026-09-18 — G0-BASELINE-001：首次机器级 reference baseline

### 1. 假设

- 现有 G0 benchmark 可以把 microkernel、prepared compute、streaming、binding 和 cold packing 分开测量；
- K-contiguous Linear weight layout 相比 N-contiguous RHS 更适合当前 `row → col → k` reference loop；
- WSL2 可以验证采集流程和数量级，但未必能提供稳定的百分比级回归门禁。

### 2. 代码与环境

```text
candidate/baseline commit: 5cbd378695bb90d97fa4273a7359bf2bfea20e8f
code working tree at audited commit: clean
artifact finalization state: docs/improvement-plan/04-cpu-gemm-optimization.md modified
CPU: Intel Core Ultra 9 285H, 16 cores, SMT off, one NUMA node
OS: Ubuntu 24.04 under WSL2, kernel 6.6.87.2-microsoft-standard-WSL2
compiler: GCC 14.2.0
build: Release, C++20
affinity: taskset -c 2
repetitions/min_time: 10 / 1s
run ID: 20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline
context.json SHA256: e12077e9b1c87c3b5a16f268b250aa8a801522156a5f14175284bf439daea621
```

采集 README 写“工作树干净”，而最终 `context.json` 记录一项 proposal 文档修改。审计确认 benchmark 代码对应上述 commit，dirty 项不涉及 `src/`、`include/` 或 `tests/`；正式报告保留这一差异，不把两者视为完全一致的元数据。

### 3. 覆盖内容

- `BM_GemmF32Reference{N,K}Contiguous`；
- `BM_LinearPreparedHot`；
- `BM_LinearPreparedStreaming`；
- `BM_LinearBindingSpecialization`；
- `BM_WeightPackingCpuIdentity`；
- 80 个 GEMM/Linear/MatMul/fused Linear/prepack/resolve 合同测试；
- 15 个 prepared bindings/execution framework invariant 测试；
- Release reference/Linear 反汇编；
- 单线程 cpufp 与 STREAM Roofline 输入。

### 4. 已验证事实

- 80/80 合同测试和 15/15 framework invariant 测试通过；
- K-contiguous RHS 的 canonical `M=1` reference 约为 2.8–3.0 GFLOP/s，N-contiguous 约为 0.2–0.29 GFLOP/s；
- prepared Linear 与 direct K-contiguous reference 的数量级一致；
- binding specialization 为亚微秒量级；
- `cpu_identity` packing 的 size amplification 为 1.0，计时包含 allocate/copy/free；
- Release reference 内层使用 scalar floating-point instructions，没有 packed SIMD loop；
- `perf stat` 在该 WSL2 kernel 上不可用；
- 独立进程复跑 delta 可达约 ±10–25%。

### 5. 推断

- 当前 reference 的主要差异来自 RHS 访问顺序，而非 dispatch/binding overhead；
- G1S 应优先覆盖 K-contiguous Linear layout，并用 strict non-SIMD 与 portable scalar 分离 loop/layout 和 auto-vectorization 收益；
- WSL2 数据适合建立数量级和采集流程，不适合执行 5% 级 production regression gate。

### 6. 尚未满足

- JSON 使用 aggregates-only，仅保留 mean/median/stddev/cv，没有保存每次 repetition 原始行；
- streaming 组没有独立复跑；
- baseline/repeat 不是交错 A/B 次序；
- perf hardware counters、governor 和可信 microcode 不可得；
- raw artifact 仅保存在采集机 `DESKTOP-54H5MMI` 的 gitignored 目录，不随仓库分发，尚无 CI/object-storage retention URL；换机器即丢失；

### 7. 决定

- **Accepted**：作为该机器、该 commit 的 G0 reference 数量级与采集流程基线；
- **Needs More Data**：作为 G1S/G1V production 百分比级性能门禁；
- 可以开始 G1S backend-private candidate 和 correctness 工作；
- 在调整 production descriptor priority 前，必须在裸机/隔离 CPU 上按完整 repetitions、交错 A/B 和 streaming repeat 协议重采。

## 2026-09-18 — G0-BASELINE-002：raw repetitions 与交错 A/B 噪声 floor

### 1. 假设

- 相邻配对（A/B 背靠背、顺序平衡）可以将环境漂移与实现差异分离，给出该机器的可归因噪声 floor；
- 保存全部 repetitions 后，比较链路不再依赖 aggregates-only JSON；
- streaming 独立复跑与 hot 复跑对称。

### 2. 代码与环境

```text
candidate/baseline commit: 5cbd378695bb90d97fa4273a7359bf2bfea20e8f
code working tree at audited commit: clean
artifact finalization state: docs/ 审计文档 modified（见 context.json working_tree）
run ID: 20260918T023421Z_5cbd378695bb_DESKTOP-54H5MMI_g0-ab
related run: 20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline
context.json SHA256: 902717d37212ecf0946a6d5ecbeffa01f0d6d5112dcf242ad454679f3e2d967a
```

其余环境同 G0-BASELINE-001（CPU/OS/compiler/affinity 不变）。

### 3. 覆盖内容

- 5 组 × 相邻配对 A/B = 10 个独立进程 run；组 1/3/5 采 A→B，组 2/4 采 B→A；
- 未使用 `--benchmark_report_aggregates_only`，每个 JSON 保留每 repetition 原始行 + 聚合行；
- 对比与统计：`noise-floor-comparison.txt`（compare 脚本）、`noise-floor-summary.txt`（中位/p90/max）；
- `checksums.sha256` 通过 `sha256sum -c` 核验。

### 4. 已验证事实

- 噪声 floor（|B−A|/A 中位 / p90 / max）：microkernel 2.62% / 7.09% / 13.20%；linear_hot 1.95% / 6.94% / 14.51%；linear_streaming 3.08% / 17.13% / 17.13%；linear_binding 12.08% / 15.32% / 24.22%；packing 4.70% / 14.84% / 14.84%；
- canonical compute（K-contiguous 微内核 / hot / streaming 主形状）A/B delta 大多在 ±6%；
- `>5%` 案例数：microkernel 4/12、hot 4/12、streaming 4/10、binding 11/12、packing 1/3；
- 10 个 run 全部 rc=0。

### 5. 推断

- 相邻配对把 canonical compute 的可判读性从顺序两轮的 ±10–25% 提升到中位 ~2–3%，系统性漂移被大部分抵消；
- G1S candidate 比较应沿用配对协议，并以各组噪声 floor（而非固定 5%）作为最小可信 delta 参照；
- 本机仍不足以支撑 5% 级 production 门禁（binding/tail 与个别大 N case 超阈值），需裸机环境。

### 6. 尚未满足

- bare-metal perf/governor/microcode；
- durable CI/object-storage artifact URL；
- 稳定百分比级 production 门禁。

### 7. 决定

- **Closed locally**：保存全部 repetition 原始数据、streaming 独立复跑、交错 A/B 协议（噪声 floor 已量化）；
- **Needs More Data**：production 百分比级门禁、bare-metal perf、artifact retention；
- G1S candidate 采集沿用 `collect_ab.sh` 配对协议。

## 2026-09-18 — G0-BASELINE-003：Roofline 定位分析

### 1. 假设

现有 baseline 数据（cpufp/STREAM 峰值 + 各组 benchmark JSON）足以把全部 canonical case 放到单线程 Roofline 上定位，回答"距硬件上限多远、瓶颈在哪一侧"。

### 2. 代码与环境

- 数据来源：G0-BASELINE-001 baseline run（`20260918T012602Z_..._g0-baseline`），commit `5cbd378695bb90d97fa4273a7359bf2bfea20e8f`；
- 分析产物：[gemm_g0_roofline_analysis_2026-09-18.md](gemm_g0_roofline_analysis_2026-09-18.md)；
- 无新增采集，仅消费既有 JSON 与 `roofline.txt`。

### 3. 覆盖内容

- direct GEMM reference（N/K-contiguous × 6 形状）；prepared Linear（hot/streaming × 10–12 形状）；
- Roofline 参数：P=123.74 GFLOP/s（cpufp FMA 256b）、Triad 22.11 GB/s（主口径）、ridge 5.60 FLOP/byte；
- 算术强度采用 compulsory 流量模型。

### 4. 已验证事实

- canonical 形状均远低于 Roofline 上限：M=1（AI=0.5，记忆侧）为 cap 的 25–27%（有效带宽 5.6–6.1 GB/s ≈ Triad 的 26%）；M≥16（AI≥7.9，计算侧）为峰值的 2.2–2.4%；
- 同形状 N-contiguous 比 K-contiguous 慢 11–14×，且两者都远在 Roofline 之下；
- 小形状（L1 驻留）实测 4.2–4.9 GFLOP/s，仅为计算峰值 ~4%。

### 5. 推断

- M=1 侧的首个优化判据可直接用"有效带宽是否显著超过 5.6–6.1 GB/s"；计算侧用"距峰值百分比 + 反汇编证据"双证据判定；
- 访问顺序（K-contiguous 实际布局优先）应排在 SIMD 之前验证；
- 百分比为指示性定位，受噪声 floor（microkernel 2.62% / hot 1.95% / streaming 3.08%）约束，不替代门禁值。

### 6. 尚未满足

- 裸机复采后重算 Roofline（峰值近似值）；多线程口径与 packed-weight 路径定位；cache 级（L2/L3）分层带宽建模。

### 7. 决定

- **Closed locally**：Roofline 定位分析归档、纳入 [GEMM 实验记录与验证报告索引](README.md)；
- G0 门禁清单不变（production 百分比级门禁仍 Needs More Data）。

## 2026-09-19 — G0-BASELINE-004：提案 §7 汇总范围值归档

### 1. 假设

专项提案 `04-cpu-gemm-optimization.md` §7 曾直接粘贴本机 baseline 的跨形状汇总范围值。按工作流「专项提案只保留工作包状态、当前结论和正式验证报告链接」与「同一事实只在一个位置详述」，这些数值必须先在本日志拥有权威副本，才能从提案删除。

### 2. 代码与环境

- 数据来源：G0-BASELINE-001 baseline run（commit `5cbd378695bb90d97fa4273a7359bf2bfea20e8f`），**本次不新增采集**；
- 触发原因：2026-09-19 文档拓扑拆分时逐值 grep，发现下列范围值在 `docs/tests/operators/gemm/` 内无完整副本（既有报告只记了单 case median）；
- 本文所有数值仍只对 `DESKTOP-54H5MMI` 成立，口径与不可跨机复用约束与本日志开头一致。

### 3. 覆盖内容

补录跨 canonical shape 的汇总范围，并显式标注它与既有报告单 case median 的聚合口径关系。

### 4. 已验证事实（补录自提案 §7）

| 指标 | 本机汇总范围 | 与既有报告副本的关系 |
|---|---|---|
| reference 访问顺序归因（direct GEMM） | K-contiguous 2.6–3.0 GFLOPS；N-contiguous 0.2–0.29 GFLOPS（约 10–14×） | 跨形状范围；单 case median 见验证报告 §6.1（2.818 / 0.205）；Roofline 报告按 6 形状统计的倍差为 11–14×，两者为不同形状子集 |
| prepared Linear compute | 2.85–3.2 GFLOPS（全 canonical 形状稳定带） | 跨形状范围；单 case median 2.916（hot）/ 2.974（streaming） |
| Decode `M=1` 权重流读 | 5.3–6.3 GiB/s logical | §6.5 logical 下界口径，非 DRAM traffic；与 Roofline 报告「有效带宽 5.6–6.1 GB/s」分母不同，不可直接互换 |
| binding specialization | 0.28–0.40 µs/call | 跨形状范围；单 case median 359 ns |
| `cpu_identity` cold packing | 3.4–4.2 GB/s effective；size amplification 1.0 | 与验证报告单 case 3.128 effective GB/s 存在口径差异（疑为是否计入目标缓冲写入与分配开销），差值未超本机噪声 floor；权威为本机 gitignored JSON |

### 5. 推断

提案里的"范围"与报告里的"单 case median"是同一原始数据的两种聚合口径；删除提案副本不损失信息，但聚合口径必须显式标注，否则换机重采时无法与本机对齐。

### 6. 尚未满足

上述范围的逐 case 原始行仅存在于本机 gitignored JSON；durable retention URL 仍缺（与提案 G0 未满足项一致）。`3.128` 与 `3.4–4.2` 的 packing 口径差异未在本文裁决。

### 7. 决定

- **Closed locally**：本条为上述范围值的权威归档位置，提案 §7 对应段落改为指向本日志与三份报告；
- 不修改既有报告结论；packing 口径差异记为待复核项，在裸机复采时用同一脚本口径消除。


