# GEMM G0 基线实验日志 — DESKTOP-QHIHOGQ

- **算子/工作包**: GEMM / G0 合同与证据基线
- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **采集机**: `DESKTOP-QHIHOGQ` — Intel Core i9-12900H（Alder Lake-H，6P+8E 混合），WSL2 kernel 6.18.33.2，GCC 14.2.0
- **日志范围**: 2026-09-18/19 本机首次 G0 采集：correctness、五组 benchmark、交错 A/B 噪声分解、反汇编与 Roofline 输入
- **原始数据位置**: `benchmark-results/operators/gemm/20260918T151928Z_162ab3e7583f_DESKTOP-QHIHOGQ_g0-baseline/`（gitignored，仅存在于本机）
- **正式报告**: [G0 baseline validation（本机）](gemm_g0_baseline_validation_desktop-qhihogq_2026-09-19.md)；[Roofline 定位分析（本机）](gemm_g0_roofline_analysis_desktop-qhihogq_2026-09-19.md)
- **前序机器**: 本日志**不替代**也不修改 [`g0-baseline-log.md`](g0-baseline-log.md)（采集机 `DESKTOP-54H5MMI` / Core Ultra 9 285H）。两台机器的数值不可互相比较或替换，见提案 §6.5/§6.7。

> **为什么需要这份日志**：既有 G0 证据全部产自 `DESKTOP-54H5MMI`，其 raw artifact 为 gitignored 且只留在该机器本地，在本机既不存在也无法复核。按 §6.7「baseline/candidate 使用同一机器、同一配置」，本机的 G0 必须独立采集，不能沿用另一台机器的结论作为门禁参照。

## 日志索引

| 日期 | 实验 ID | 假设 | 结论 | 状态 |
|---|---|---|---|---|
| 2026-09-18 | G0-QHIHOGQ-001 | 仓库内的 G0 benchmark/测试基础设施可在另一台机器上原样复现采集流程 | 可复现；106/106 契约 + 30/30 框架不变量测试通过；五组 benchmark 全部 rc=0 | Closed |
| 2026-09-18 | G0-QHIHOGQ-002 | 访问顺序主导 reference 性能的结论跨机器成立 | 成立且更强：K=4096 形状 N-contiguous 比 K-contiguous 慢 16–18×；但 L1 可驻留的小形状只差 1.18× | Closed |
| 2026-09-18 | G0-QHIHOGQ-003 | 本机噪声 floor 与采集机同类，5% 门禁同样不可用 | 部分成立但**成因不同**：进程内 CV 仅 0.03–0.04%，跨进程系统偏移达 0.28–5.17%（max 15.30%）；binding/packing 反而可用 5% 门禁 | Closed |
| 2026-09-18 | G0-QHIHOGQ-004 | WSL2 下可以确定 P/E core 归属以固定采集 CPU | **不成立**：guest 拓扑被伪造为对称 10c/20t，`taskset` 只是咨询性绑定 | Closed（负面结论） |
| 2026-09-18 | G0-QHIHOGQ-005 | G1S scalar candidate 是纯标量数据流 | **不成立**：`-O3` 下被自动向量化为 SSE2 4-wide（16 `mulps` + 12 `addps`），且 shuffle/unpack 达 55 条 | Closed（符合 §4.3 已接受的风险，但开销可优化） |

## 2026-09-18 — G0-QHIHOGQ-001：本机首次 G0 采集

### 1. 假设

- G0 的 benchmark、测试与采集脚本属仓库资产，换机器后应能原样复现；
- correctness 契约与机器无关，应在本机同样全部通过；
- 五组 benchmark（microkernel / hot / streaming / binding / packing）应能各自独立归因。

### 2. 代码与环境

```text
commit: 162ab3e7583ffad11c9a7f5bbaa345dade665a9c
code paths modified under src/include/tests: 0   (working tree dirty only for docs/)
CPU: 12th Gen Intel Core i9-12900H (Alder Lake-H, 6P+8E), 1 socket, 1 NUMA node
     guest-visible: 10 cores x 2 threads = 20 vCPUs (fabricated, see G0-QHIHOGQ-004)
     L1d 480 KiB / L1i 320 KiB / L2 12.5 MiB / L3 24 MiB
effective ISA: avx2 avx_vnni bmi1 bmi2 f16c fma   (no avx512, no amx)
OS: Ubuntu 24.04 LTS under WSL2, kernel 6.18.33.2-microsoft-standard-WSL2
compiler: GCC 14.2.0
build: Release, -O3 -DNDEBUG, C++20
AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE: OFF  (baseline = reference descriptor)
affinity: taskset -c 16
repetitions/min_time: 10 / 1s
report_aggregates_only: false  (full repetition rows retained)
run ID: 20260918T151928Z_162ab3e7583f_DESKTOP-QHIHOGQ_g0-baseline
load average: 0.64/5.62/3.98 at start, 1.03/1.08/1.37 at end
```

`build-release/` 在采集前是陈旧产物：benchmark 二进制日期为 5 月 10 日，早于全部 scalar GEMM 工作，且 CMakeCache 里连 `AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE` 选项都不存在（该选项是后来加的）。已重新配置并重建两个 target 后才采集。

`context.json` 记录 `cpu_microcode: 0xffffffff`。这是 Hyper-V 的虚拟值，**不是真实 microcode revision**，不可用于跨机比较或安全审计。

采集期间机器为共享开发工作站，存在 CLion、Rider.Backend 与其他 agent 进程；起始 1 分钟 load average 0.64、瞬时 idle 99.1%。这一背景是 G0-QHIHOGQ-003 中跨进程偏移的可能来源之一，但无法在本环境内隔离验证。

### 3. 覆盖内容

- correctness：`CPUKernelGemmReference` / `CPUKernelGemmScalar` / `FastPathBoundaries/CPUKernelGemmScalarFastPathTest` / `CPUKernelLinear{,Entry}` / `CPUKernelMatMul{,Entry}` / `CPUKernelQkvLinear` / `CPUKernelGateUpLinear` / `CpuWeightPrepacker` / `CpuPrepareKernel` / `CpuIdentityPackedWeight` / `PackedWeightStoreOwnership` / `KernelSelector`；
- framework invariant：`Executor` / `PreparedExecutionBindings` / `ExecutionContext` / `ExecutionBindings` / `NoHotpathPrepare` / `ExecutionPlanImmutability` / `WorkspaceRequirementPlanning`；
- benchmark 五组 × 交错 A/B = 10 个独立进程 run，共 49 case × 10 repetitions；
- 反汇编：`RunGemmF32Reference` 与 `RunGemmF32ScalarOptimized`；
- Roofline 输入：本机 AVX2+FMA 峰值、STREAM-like Copy/Triad；
- 比较链路：`tools/compare_benchmark_json.py` 与本机 `summarize_g0.py` / `roofline_positioning.py`。

### 4. 已验证事实

- **106/106** GEMM/Linear/MatMul/Qkv/GateUp/Prepack/Resolve 契约测试通过；**30/30** 框架不变量测试通过（含 steady-state 零分配与 params 只构建一次）；
- 全部 10 个 benchmark run `rc=0`，repetition 原始行完整（microkernel/hot/binding 各 120 iteration rows + 48 aggregate rows；streaming 100+40；packing 30+12）；
- benchmark `label` 字段确认为 `kernel=cpu::linear_f32_reference`，即 baseline 定义正确、opt-in scalar candidate 未参与；
- 本机 baseline 数值（median of 10 repetitions，pass A）：
  - prepared Linear **hot**：canonical 大形状 3.41–3.72 GFLOP/s；L1 驻留小形状 4.58–4.69 GFLOP/s；
  - prepared Linear **streaming**：`M=1` 3.60–3.70 GFLOP/s，`M=16/64` 3.39–3.56 GFLOP/s；
  - **binding specialization**：0.414–0.433 µs/call，且几乎与 shape 无关；
  - **`cpu_identity` cold packing**：5.45–6.13 GB/s effective；
- `hot` 与 `streaming` 在 `M=1` 上几乎相同（3.51–3.72 vs 3.60–3.70 GFLOP/s），复现「权重已超出 LLC、reference 本身即 DRAM-bound」的结论；
- `perf`、`cpufreq/scaling_governor`、内存频率在本 WSL2 内核均不可用。

### 5. 推断

- G0 的**基础设施**确实跨机可用，correctness 契约与机器无关；
- binding 与 packing 两组的跨进程稳定性远好于计算组，说明其成本主要由确定性的 cold-path 工作量决定，受频率/调度漂移影响小；
- `hot ≈ streaming` 意味着在 reference 实现下，为 Decode 设计更复杂的 cache 策略没有收益空间——瓶颈已经是权重流读本身。

### 6. 尚未满足

- production 百分比级性能门禁（microkernel/hot/streaming 三组的跨进程偏移超过 5%）；
- bare-metal `perf` 硬件计数器、governor、可信 microcode；
- durable CI/object-storage artifact retention URL；
- P/E core 归属不可控（见 G0-QHIHOGQ-004），因此「固定到某一类核」这一采集前提在本机无法建立。

### 7. 决定

- **Accepted**：作为 `DESKTOP-QHIHOGQ` + commit `162ab3e7` 的 G0 reference 基线、采集流程复现证据与噪声分解证据；
- **Needs More Data**：作为 G1S/G1V 的 production 百分比级门禁；
- 不修改任何 production descriptor priority；
- 本机 G0 与采集机 G0 **并列存在、互不替代**；两台的百分比不得互相引用。

## 2026-09-18 — G0-QHIHOGQ-002：访问顺序归因在本机复现且更强

### 1. 假设

采集机记录的「访问顺序主导 reference 性能（N-contiguous 比 K-contiguous 慢 10–14×）」是实现与内存层级交互的结果，应跨机器成立。

### 2. 覆盖内容

`BM_GemmF32Reference{N,K}Contiguous` × 6 形状，median of 10 repetitions。

### 3. 已验证事实

| 形状 | K-contiguous | N-contiguous | N/K 慢化 |
|---|---:|---:|---:|
| M:1/K:4096/N:4096 | 3.659 | 0.209 | **17.5×** |
| M:1/K:4096/N:11008 | 3.766 | 0.227 | **16.6×** |
| M:4/K:4096/N:4096 | 3.634 | 0.199 | **18.3×** |
| M:16/K:4096/N:4096 | 3.468 | 0.210 | **16.5×** |
| M:1/K:33/N:31 | 4.435 | 3.766 | 1.18× |
| M:2/K:32/N:33 | 4.456 | 3.779 | 1.18× |

单位 GFLOP/s。前四行工作集远超 L1/L2；后两行工作集可驻留 L1。

### 4. 推断

- 差距在 K=4096 形状上是 **16–18×**，比采集机记录的 10–14× 更大；
- **关键新证据**：当工作集可驻留 L1 时（K=33/N=31、M=2/K=32/N=33），N-contiguous 与 K-contiguous 只差 **1.18×**，且两者都达到记忆侧上限的 31–37%。这说明慢化**不是**代码路径或分支结构造成的，而是 N-contiguous 大形状下的 stride 访存（cache line 利用率与 TLB 行为）造成的；
- 因此 G1S/G1V 的首要收益来源应是 K-contiguous（Linear 实际布局）下的数据流与 SIMD 宽度，而不是「修复 N-contiguous」——后者在 production Linear 路径上并不出现。

### 5. 决定

- **Closed**：访问顺序归因在本机独立复现，且补充了「L1 驻留时差距消失」这一更强的判据；
- 提案 §2.1 第 1 条瓶颈描述在本机同样成立。

## 2026-09-18 — G0-QHIHOGQ-003：噪声分解——进程内极稳、跨进程系统偏移

### 1. 假设

本机与采集机同为 WSL2，噪声结构应类似（采集机：顺序两轮 ±10–25%，配对交错后 canonical 中位 ~2–3%）。

### 2. 覆盖内容

五组 × 交错 A/B（组 1/3/5 采 A→B，组 2/4 采 B→A），每 pass 10 repetitions，保留全部原始行；`summarize_g0.py` 分解进程内 CV 与跨进程偏移；`tools/compare_benchmark_json.py` 做端到端比较。

### 3. 已验证事实

| 组 | 进程内 CV（median/p90/max） | 跨进程 \|delta\| median | max | >5% 案例 |
|---|---|---:|---:|---|
| microkernel | 0.04% / 0.06% / 0.11% | 1.72% | 15.30% | 3/12 |
| linear_hot | 0.04% / 0.05% / 0.05% | 3.89% | 7.16% | 4/12 |
| linear_streaming | 0.03% / 0.04% / 0.07% | 5.17% | 5.98% | 5/10 |
| linear_binding | 0.03% / 0.04% / 0.04% | 0.90% | 3.42% | 0/12 |
| packing | 0.03% / 0.03% / 0.04% | 0.28% | 0.34% | 0/3 |

带符号方向：`linear_hot` 12/12 同向（+2.07%~+7.16%）；`linear_streaming` 9/10 同向（−5.98%~+0.79%），其中 5 个 `M=1` case 高度一致地落在 −5.75%~−5.98%；`microkernel` 9/12 反向。方向在组之间与 35 分钟采集窗口内**不稳定**，无法用单纯 warm-up 或单调热漂移解释。

用 `tools/compare_benchmark_json.py` 把**同一份实现**的 pass A 当 baseline、pass B 当 candidate 比较（即零代码改动），脚本按默认 5% 阈值判出：

- `microkernel`：9 个 "IMPROVE"，最大 −16.40%（`NContiguous/M:4/K:4096/N:4096`）；
- `linear_hot`：12 个 "IMPROVE"，最大 −7.71%；
- `linear_streaming`：**4 个 "REGRESS"**（+5.52%~+7.39%）；
- `linear_binding`：0 regression，max \|delta\| 1.47%；
- `packing`：0 regression，max \|delta\| 2.02%。

### 4. 推断

- 本机的不确定性来源是**每进程系统性偏移**，不是随机抖动。进程内 CV 只有 0.03–0.04%，所以**增加 `--benchmark_repetitions` 不会降低 A/B floor**——这与「多跑几次取平均就能压噪声」的直觉相反；
- 有效缓解手段是：重复多个独立进程并比较「每进程 median 的分布」，或把 baseline/candidate 调度进同一进程内交替执行；
- 与采集机相比，本机的 `binding`（0.90% vs 12.08%）与 `packing`（0.28% vs 4.70%）**显著更稳**，因此这两组在本机可以适用 §6.8 的 5% 门禁；而 `microkernel`/`linear_hot`/`linear_streaming` 三组不可；
- 「零改动被判出 4 个 REGRESS 与 −16.40% 的 IMPROVE」是 §6.8 的 5% 阈值在本机不可作为自动门禁的直接证据。

### 5. 尚未满足

- 跨进程偏移的物理成因（频率漂移 / hypervisor 调度 / 共享工作站背景负载 / P-E core 迁移）在本环境内不可分离；
- 稳定的百分比级 production 门禁；bare-metal perf/governor/microcode。

### 6. 决定

- **Closed**：本机噪声 floor 已量化并完成进程内/跨进程分解，结论与原始数据归档于 `noise-floor-summary.txt` 与 `noise-floor-comparison.txt`；
- **Needs More Data**：production 百分比级门禁；
- G1S/G1V 的本机 candidate 比较必须沿用交错 A/B，并以**各组自己的噪声 floor**（而非固定 5%）作为最小可信 delta；对 binding/packing 可直接用 5%。

## 2026-09-18 — G0-QHIHOGQ-004：WSL2 下无法确定 P/E core 归属（负面结论）

### 1. 假设

i9-12900H 是 6P+8E 混合架构，P-core 与 E-core 的 SIMD 峰值相差数倍，因此必须把采集固定到已知类别的核上，否则数据不可解释。

### 2. 覆盖内容

sysfs 拓扑与 cache 层级、`cpu_capacity`、`/proc/cpuinfo`；对 20 个 vCPU 逐个实测 AVX2+FMA 吞吐；sysfs 声称的 SMT 兄弟对并发争用测试。

### 3. 已验证事实

- guest 拓扑为**对称** 10 核 × 2 线程；每个「核」的 L2 都是 1280K、`cpu_capacity` 全为 1024、`cpu MHz` 全部静态报为 2918.399；`/sys/devices/system/cpu/cpu0/` 下**没有** `core_type`，也没有任何 hybrid/Thread Director 信息；
- retail 12900H 为 14C/20T（6P+8E），guest 报 10C/20T，`.wslconfig` 设 `processors=20`——线程数吻合，核数不吻合；
- 逐 vCPU 实测（8 条独立 FMA 链）：20 个 vCPU 全部落在 **112–135 GFLOP/s**，换算有效频率 3.6–4.2 GHz。**不存在 E-core 那一档**（Gracemont 的 256-bit FMA 吞吐约为 Golden Cove 的一半，若暴露应出现明显低档）；
- sysfs 声称的 SMT 兄弟对（cpu0/cpu1）并发跑纯 FMA 时各得 63.80 / 67.14 GFLOP/s、合计 131 GFLOP/s，**无争用**；不同「核」（cpu0/cpu10）并发同样无争用。若 cpu0/cpu1 真是同一物理核的两个 SMT 线程，两者会争抢同一组 FMA 端口，合计吞吐不可能翻倍；
- `cpufreq` 目录不存在，governor 不可读、不可设。

### 4. 推断

- WSL2 呈现的核拓扑是**伪造的**：20 个 vCPU 由 Hyper-V/Windows 调度器动态放到物理硬件线程上，guest 看到的「10 核 × 2 线程」配对不对应物理 SMT 配对；
- 因此 `taskset -c N` 在本机**只是咨询性绑定**：它固定的是 vCPU 编号，不是物理核，更不是 P/E core 类别。同一 vCPU 在不同时刻可能落在不同物理核上；
- 这解释了 G0-QHIHOGQ-003 中「跨进程偏移方向不稳定」的一部分：偏移可能包含 hypervisor 层的物理核迁移，而 guest 无法观测也无法阻止。

### 5. 决定

- **Closed（负面结论）**：在本机无法建立「固定到某一类物理核」这一采集前提；
- 采集 CPU 选定 **cpu16**（重复测量 spread 5.2%，为候选中最稳），并在 `context.json` 的 `affinity_caveat` 中显式记录绑定的咨询性质；
- 任何需要 P/E 分离或 SMT 控制的性能结论，必须在裸机（可 `isolcpus`、可关 SMT、可读 `core_type`）上采集；本机数据只能支撑结构性归因与数量级判断。

## 2026-09-18 — G0-QHIHOGQ-005：reference 为纯标量，scalar candidate 已被 SSE2 自动向量化

### 1. 假设

- `RunGemmF32Reference` 的内层循环在本机编译器下同样未被向量化（采集机已确认）；
- G1S 的 `RunGemmF32ScalarOptimized` 按 §4.3「不使用 intrinsic 但允许 auto-vectorization」，其实际向量化程度需核对反汇编。

### 2. 覆盖内容

`objdump -d` 提取两个符号的完整函数体，按助记符分类 packed（`*ps`/`*pd`）与 scalar（`*ss`/`*sd`）；核对 `build-release/compile_commands.json` 中两个 TU 的实际编译标志。

### 3. 已验证事实

| 符号 | 行数 | `ymm` | `zmm` | packed FP 算术 | scalar FP 算术 | shuffle/unpack |
|---|---:|---:|---:|---|---|---:|
| `RunGemmF32Reference` | 124 | 0 | 0 | **0** | `mulsd`×2, `addsd`×2, `movss`×1 | 0 |
| `RunGemmF32ScalarOptimized` | 595 | 0 | 0 | **`mulps`×16, `addps`×12** | `addss`×14, `mulss`×6, `movss`×22 | **55**（`shufps`×25, `unpcklps`×16, `movlhps`×8, `unpckhps`×6） |

编译标志：`gemm_f32_reference.cpp` 与 `gemm_f32_scalar.cpp` 均只有 `-O3`，**没有** `-mavx2`/`-mfma`。`src/CMakeLists.txt:55-59` 的 `set_source_files_properties(... COMPILE_FLAGS "-mavx2 -mfma")` 只施加于 `common/dot_product_f32_avx2.cpp` 与 `rmsnorm/rmsnorm_f32_avx2.cpp`。

### 4. 推断

- **reference 是纯标量**（零 packed FP 算术），本机复现采集机的反汇编结论，也解释了 `M≥16` 计算侧只有峰值 2.6–2.8% 的原因；
- **G1S candidate 不是纯标量数据流**：`-O3` 在基线 x86-64 ISA 上把它 SLP 向量化成了 **SSE2 4-wide**（`mulps`/`addps` 用 xmm，故 `ymm=0`）。这正是 §4.3 与 §8 已明确接受的风险（「scalar source 被编译器自动向量化 → 收益归因模糊（已接受：不做拆分）」），2026-09-18 的简化决策已撤销 strict non-SIMD 变体，因此本项**不构成阻塞**；
- 但有一个**可操作的新发现**：shuffle/unpack 共 55 条，接近 packed 算术（28 条）的 2 倍。这是 auto-vectorizer 处理「多 accumulator + 水平归约」模式时的典型低效特征——当前 `NR=4` + K unroll 2 的写法让编译器不得不做大量 lane 重排；
- 由于两个 GEMM TU 都没有 `-mavx2 -mfma`，G1S 的自动向量化上限是 **SSE2 4-wide**，完全没有用到本机的 AVX2+FMA（8-wide + FMA）。**G1V 相对 G1S 仍有真实的、未被占用的空间**，两者不是重复工作。

### 5. 决定

- **Closed**：反汇编证据归档于 `disassembly-reference.txt` 与 `disassembly-scalar.txt`；
- reference 未向量化的事实可作为 G1S/G1V 收益归因的起点证据；
- 建议 G1S 调优时把「降低 shuffle/unpack 占比」列为一个显式观察项（调整 accumulator 数量与 K unroll，使 lane 布局无需重排），但**不引入 strict non-SIMD 变体**，遵守 2026-09-18 简化决策；
- G1V 的 AVX2+FMA TU 必须按 §4.4 单独施加 `-mavx2 -mfma`，不得对整个 target 使用 `-march=native`。
