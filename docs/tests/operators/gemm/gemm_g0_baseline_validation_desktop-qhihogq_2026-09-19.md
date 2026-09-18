# GEMM G0 Baseline 验证报告 — DESKTOP-QHIHOGQ

- **状态**: Current（限 `DESKTOP-QHIHOGQ`）
- **日期**: 2026-09-19（采集窗口 2026-09-18T15:19Z–16:02Z）
- **专项提案**: [CPU GEMM 优化方案](../../../improvement-plan/04-cpu-gemm-optimization.md)
- **工作包**: G0 合同与证据基线
- **Change Profile**: Reference / Optimized benchmark infrastructure / Packing/Layout measurement（无 production 代码改动）
- **Candidate commit**: `162ab3e7583ffad11c9a7f5bbaa345dade665a9c`
- **Baseline commit**: `162ab3e7583ffad11c9a7f5bbaa345dade665a9c`（同实现独立复跑）
- **Working tree**: 采集时 `src/`、`include/`、`tests/` 下 **0** 个修改路径（见 `context.json` 的 `code_paths_modified_under_src_include_tests: 0`），dirty 项仅为本次机器归属修正的 `docs/` 文件。**采集结束后**观察到 `src/backend/cpu/kernels/gemm/gemm_f32_scalar.cpp` 出现一处单空行插入（mtime `2026-09-19T00:00:28+08:00`，晚于最后一个 benchmark run 的 `23:57:37`），非本次采集所为；该改动为纯空白、不改变 codegen，且两个二进制已于 `23:18` 由干净的 `162ab3e7` 源码构建完成，因此**不影响本报告任何测量与反汇编结论**
- **采集机**: `DESKTOP-QHIHOGQ` — Intel Core i9-12900H（Alder Lake-H，6P+8E），WSL2 kernel 6.18.33.2，GCC 14.2.0
- **Raw artifact**: `benchmark-results/operators/gemm/20260918T151928Z_162ab3e7583f_DESKTOP-QHIHOGQ_g0-baseline/`（gitignored、不随仓库分发，**仅存在于本机本地磁盘**；34 个文件、1.7 MiB）
- **Artifact checksum**: `context.json` SHA256 `9d94f0626209d8c489dc2029453b3530882173721ef79f645f65abfae9d97e3e`；全部产物见 `checksums.sha256`（`sha256sum -c` 通过）
- **关联证据**: [G0 实验日志（本机）](g0-baseline-log-desktop-qhihogq.md)、[Roofline 定位分析（本机）](gemm_g0_roofline_analysis_desktop-qhihogq_2026-09-19.md)
- **关联 ADR**: 无

> **机器归属**：本报告的全部数值只对采集机 `DESKTOP-QHIHOGQ` 成立。它与 [`gemm_g0_baseline_validation_2026-09-18.md`](gemm_g0_baseline_validation_2026-09-18.md)（采集机 `DESKTOP-54H5MMI` / Core Ultra 9 285H）**并列存在、互不替代**；按提案 §6.7（baseline/candidate 必须同机）与 §6.5（不得跨微架构比较原始计数），两台的百分比与绝对值不得互相引用。

## 1. 验证目标与结论

本报告的直接动因是：既有 G0 证据全部产自 `DESKTOP-54H5MMI`，其 raw artifact 为 gitignored 且只留在该机器本地，在本机不存在也无法复核。因此需要验证 G0 的**仓库内基础设施**能否在另一台机器上原样复现，并为本机建立独立的 reference baseline 与噪声 floor。

最终判定：

- **Accepted**：作为 `DESKTOP-QHIHOGQ` + commit `162ab3e7` 上的首次 G0 reference baseline、采集流程复现证据、噪声分解证据与反汇编/Roofline 定位证据；
- **Needs More Data**：作为任何 optimized candidate 的百分比级 production acceptance gate（microkernel / hot / streaming 三组的跨进程偏移超过 5%）；
- **Not Applicable on this machine**：bare-metal `perf` 计数器、governor、真实 microcode、P/E core 分离——本 WSL2 环境原理上不可得。

本报告不证明 scalar optimized、SIMD、packed Linear、端到端 token latency 或跨机器性能。

## 2. 被验证实现

```text
direct primitive:
    RunGemmF32Reference              (correctness oracle, double accumulation)
    RunGemmF32ScalarOptimized        (G1S candidate, 仅反汇编核对，未参与 baseline 计时)
production path:
    cpu::linear_f32_reference        (由 benchmark label 字段确认)
    AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE = OFF
packing:
    CpuWeightPrepacker cpu_identity recipe
```

采集前发现 `build-release/` 为陈旧产物（benchmark 二进制日期 5 月 10 日，早于全部 scalar GEMM 工作；CMakeCache 中不存在 `AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE` 选项）。已重新配置（Release、`BUILD_TESTS=ON`、`BUILD_BENCHMARKS=ON`、scalar candidate `OFF`）并重建 `aethermind_unit_tests` 与 `aethermind_benchmark` 后才采集，两个二进制时间戳一致。

## 3. 环境

| 项 | 值 |
|---|---|
| hostname | `DESKTOP-QHIHOGQ` |
| CPU | 12th Gen Intel Core i9-12900H（Alder Lake-H，6P+8E），1 socket，1 NUMA node |
| guest 可见拓扑 | 10 cores × 2 threads = 20 vCPUs（**伪造的对称拓扑**，见 §7） |
| cache | L1d 480 KiB / L1i 320 KiB / L2 12.5 MiB / L3 24 MiB |
| 有效 ISA | `avx2 avx_vnni bmi1 bmi2 f16c fma`（无 avx512、无 amx） |
| OS / kernel | Ubuntu 24.04 LTS / 6.18.33.2-microsoft-standard-WSL2 |
| compiler | GCC 14.2.0 |
| build | Release，`-O3 -DNDEBUG`，C++20 |
| affinity | `taskset -c 16`（**咨询性**，见 §7） |
| repetitions / min_time | 10 / 1s |
| aggregates-only | **false**（保留全部 repetition 原始行） |
| load average | 起始 0.64 / 5.62 / 3.98，结束 1.03 / 1.08 / 1.37 |
| governor | **UNAVAILABLE**（无 `/sys/.../cpufreq`） |
| `perf` | **UNAVAILABLE** |
| 内存频率 | **UNAVAILABLE** |
| microcode | `0xffffffff` — Hyper-V 虚拟值，**不是真实 revision** |
| `.wslconfig` | `processors=20`，`memory=64GB` |

采集期间机器为共享开发工作站，存在 CLion、Rider.Backend 与其他 agent 进程；起始瞬时 idle 99.1%。该背景负载是跨进程偏移的可能来源之一，但在本环境内无法隔离验证。

## 4. Correctness 与安全

| 组 | 结果 |
|---|---|
| GEMM/Linear/MatMul/Qkv/GateUp/Prepack/Resolve 契约测试 | **106 / 106 通过** |
| Executor / PreparedExecutionBindings / ExecutionContext / ExecutionBindings / NoHotpathPrepare / ExecutionPlanImmutability / WorkspaceRequirementPlanning | **30 / 30 通过** |

契约测试覆盖 `CPUKernelGemmReference`、`CPUKernelGemmScalar`、`FastPathBoundaries/CPUKernelGemmScalarFastPathTest`、`CPUKernelLinear{,Entry}`、`CPUKernelMatMul{,Entry}`、`CPUKernelQkvLinear`、`CPUKernelGateUpLinear`、`CpuWeightPrepacker`、`CpuPrepareKernel`、`CpuIdentityPackedWeight`、`PackedWeightStoreOwnership`、`KernelSelector`。

框架不变量测试确认 steady-state 零分配与 params 只构建一次，即 params build/validation **未混入** compute loop。

本机 106 个契约测试多于采集机记录的 80 个，差额来自 G1S 提交新增的 `CPUKernelGemmScalar` 与 `FastPathBoundaries` 两组，**不是**覆盖范围回退。correctness 契约与机器无关，此项在两台上等价成立。

## 5. Benchmark 协议

按提案 §6.7 执行：Release 构建、单线程 `taskset` 固定、10 repetitions、`min_time=1s`、独立进程、组内交错 A/B（组 1/3/5 采 A→B，组 2/4 采 B→A）。

与采集机协议的两处**改进**：

1. **不使用** `--benchmark_report_aggregates_only`，保留全部 repetition 原始行（采集机 baseline run 列为「尚未满足」的第 1 项，本机直接满足）；
2. 噪声分析同时输出「进程内 CV」与「跨进程偏移」两个量，而非只报单一 delta，使噪声来源可归因。

五组共 49 case × 10 repetitions × 2 pass = 10 个独立进程 run，全部 `rc=0`，总耗时约 35 分钟。hot / streaming / binding / packing / microkernel 各自保存独立 JSON，未合并为单一 geomean（符合 §6.1 与 §7 G0 的要求）。

## 6. 结果摘要

全部为 median of 10 repetitions，pass A。机器可读版本见 `g0-summary.txt`。

### 6.1 Canonical reference cases

| 组 | 形状 | 值 |
|---|---|---:|
| prepared Linear hot | `M=1` K=4096 N∈{4096,6144,11008,22016,32000} | **3.505–3.719 GFLOP/s** |
| prepared Linear hot | `M=1` K=11008 N=4096 | 3.572 GFLOP/s |
| prepared Linear hot | `M=16/64` canonical | **3.413–3.690 GFLOP/s** |
| prepared Linear hot | L1 驻留小形状（K=33/N=31、M=2/K=32/N=33） | 4.582 / 4.691 GFLOP/s |
| prepared Linear streaming | `M=1` canonical | **3.598–3.696 GFLOP/s** |
| prepared Linear streaming | `M=16/64` canonical | 3.388–3.556 GFLOP/s |
| direct GEMM **K-contiguous** | K=4096 canonical | **3.468–3.766 GFLOP/s** |
| direct GEMM **N-contiguous** | K=4096 canonical | **0.199–0.227 GFLOP/s** |

`hot` 与 `streaming` 在 `M=1` 上几乎相同（3.505–3.719 vs 3.598–3.696），说明权重工作集（64 MiB 起）已远超 24 MiB L3，reference 本身即 DRAM-bound。

### 6.2 Binding 与 packing

| 组 | 值 |
|---|---|
| binding specialization | **0.414–0.433 µs/call**，与 shape 基本无关 |
| `cpu_identity` cold packing | **5.45–6.13 GB/s** effective，size amplification 1.0 |

### 6.3 Roofline 输入（本机实测）

| 量 | 值 | 说明 |
|---|---:|---|
| 计算峰值 P | **130.21 GFLOP/s** | 8 条独立 FMA 链实测 best-of-7，≈4.07 GHz × 32 FLOP/cycle |
| STREAM Triad | **25.32 GB/s** | 3×384 MiB double，主口径 |
| STREAM Copy | 41.12 GB/s | glibc `memcpy` 走 NT store，为**上界** |
| ridge point | **5.14 FLOP/byte** | P / Triad |

定位结果：`M=1` 记忆侧达 ceiling 的 **27.7–29.4%**（有效带宽 7.0–7.4 GB/s）；`M≥16` 计算侧达峰值的 **2.6–2.8%**。详见 [Roofline 定位分析（本机）](gemm_g0_roofline_analysis_desktop-qhihogq_2026-09-19.md)。

## 7. 可重复性、硬件计数器与反汇编

### 7.1 噪声分解（本机的核心新结论）

| 组 | 进程内 CV（median/p90/max） | 跨进程 \|delta\| median | max | >5% |
|---|---|---:|---:|---|
| microkernel | 0.04% / 0.06% / 0.11% | 1.72% | 15.30% | 3/12 |
| linear_hot | 0.04% / 0.05% / 0.05% | 3.89% | 7.16% | 4/12 |
| linear_streaming | 0.03% / 0.04% / 0.07% | 5.17% | 5.98% | 5/10 |
| linear_binding | 0.03% / 0.04% / 0.04% | **0.90%** | 3.42% | **0/12** |
| packing | 0.03% / 0.03% / 0.04% | **0.28%** | 0.34% | **0/3** |

**不确定性来源是每进程系统性偏移，不是随机抖动。** 进程内 CV 仅 0.03–0.04%，因此提高 `--benchmark_repetitions` **不会**降低 A/B floor。带符号方向在组之间与 35 分钟窗口内均不稳定（`linear_hot` 12/12 同向 +2.07%~+7.16%；`linear_streaming` 9/10 同向 −5.98%~+0.79%；`microkernel` 9/12 反向），无法用单纯 warm-up 或单调热漂移解释。

**阈值误判证据**：用 `tools/compare_benchmark_json.py` 把同一份实现的 pass A 当 baseline、pass B 当 candidate（零代码改动），脚本按默认 5% 阈值判出 `linear_streaming` **4 个 REGRESS**（+5.52%~+7.39%）、`microkernel` 最大 **−16.40% 的 "IMPROVE"**、`linear_hot` 12 个 "IMPROVE"。即无改动也会被自动门禁误判。详见 `noise-floor-comparison.txt`。

反之，`linear_binding`（max 1.47%）与 `packing`（max 2.02%）在本机**可以**适用 5% 门禁。

### 7.2 硬件计数器

`perf` 在本 WSL2 内核不可用（未安装且计数器未暴露），`cpufreq`/governor 目录不存在，内存频率不暴露，`microcode` 为 `0xffffffff` 虚拟值。因此**没有**任何 cycles / instructions / branch-misses / cache-misses / L1 / LLC 事件数据。这是环境原理性限制，不是采集疏漏；必须在裸机补齐。

### 7.3 拓扑与 affinity 有效性（负面结论）

- guest 拓扑为对称 10 cores × 2 threads，`cpu_capacity` 全为 1024，每「核」L2 均为 1280K，`cpu MHz` 静态报为 2918.399，`/sys/devices/system/cpu/cpu0/` 下无 `core_type` 或任何 hybrid 信息；
- retail 12900H 为 14C/20T（6P+8E），guest 报 10C/20T；`.wslconfig` 设 `processors=20`，线程数吻合而核数不吻合；
- 逐 vCPU 实测 AVX2+FMA 吞吐：20 个 vCPU 全部落在 **112–135 GFLOP/s**（有效频率 3.6–4.2 GHz），**不存在 E-core 应有的低档**；
- sysfs 声称的 SMT 兄弟对（cpu0/cpu1）并发跑纯 FMA 时各得 63.80 / 67.14 GFLOP/s、合计 131 GFLOP/s，**无争用**；不同「核」（cpu0/cpu10）并发同样无争用。若二者真是同一物理核的 SMT 线程，会争抢同一组 FMA 端口，合计吞吐不可能翻倍。

结论：WSL2 呈现的核拓扑是伪造的，「10 核 × 2 线程」的配对不对应物理 SMT 配对。**`taskset -c N` 固定的是 vCPU 编号，不是物理核，更不是 P/E core 类别**；同一 vCPU 在不同时刻可能落在不同物理核上，guest 无法观测也无法阻止。这解释了 §7.1 中偏移方向不稳定的部分成因。采集 CPU 选定 cpu16（重复测量 spread 5.2%，候选中最稳），并在 `context.json` 的 `affinity_caveat` 中显式记录该绑定的咨询性质。

### 7.4 反汇编

| 符号 | packed FP 算术 | scalar FP 算术 | shuffle/unpack | `ymm` |
|---|---|---|---:|---:|
| `RunGemmF32Reference` | **0** | `mulsd`×2, `addsd`×2, `movss`×1 | 0 | 0 |
| `RunGemmF32ScalarOptimized` | **`mulps`×16, `addps`×12** | `addss`×14, `mulss`×6, `movss`×22 | **55** | 0 |

- **reference 内层为纯标量、零 packed FP 算术**，本机复现采集机结论，解释了 `M≥16` 计算侧仅 2.6–2.8% 峰值；
- **G1S candidate 已被自动向量化为 SSE2 4-wide**（`mulps`/`addps` 用 xmm，故 `ymm=0`）。这符合 §4.3「不使用 intrinsic 但允许 auto-vectorization」与 §8 已接受的风险，**不构成阻塞**；
- 两个 GEMM TU 的编译标志均只有 `-O3`，**没有** `-mavx2`/`-mfma`（`src/CMakeLists.txt:55-59` 的 `set_source_files_properties` 只施加于 `dot_product_f32_avx2.cpp` 与 `rmsnorm_f32_avx2.cpp`）。因此 G1S 的自动向量化上限是 SSE2 4-wide，**完全没有用到本机 AVX2+FMA**——G1V 相对 G1S 仍有真实且未被占用的空间，两者不是重复工作；
- shuffle/unpack 共 55 条、约为 packed 算术（28 条）的 2 倍，是 auto-vectorizer 处理「多 accumulator + 水平归约」的典型低效特征，可作为 G1S 调优的显式观察项。

## 8. Integration 与端到端

本报告不涉及端到端。真实 `prefill latency` / `decode ms/token` / `tokens/s` 按 §6.1 必须等 Generate vertical slice 可运行后再加入，单算子 benchmark 不能替代。

`execution integration` 层 benchmark（`benchmark_execution_gemm_path.cpp`）尚未存在，因此「adapter、binding、workspace 是否抵消 kernel 收益」在本机同样未被证明。

## 9. 事实、推断与限制

### 已验证事实

- 106/106 契约测试与 30/30 框架不变量测试通过；
- 10 个 benchmark run 全部 `rc=0`，repetition 原始行完整，`label` 确认为 `cpu::linear_f32_reference`；
- 本机 baseline 数值如 §6；
- 访问顺序慢化 16–18×（K=4096），L1 驻留时仅 1.18×；
- 进程内 CV 0.03–0.04%，跨进程偏移 0.28–5.17%（median）、max 15.30%；
- 同一实现自我比较被判出 4 个 REGRESS 与 −16.40% 的 IMPROVE；
- reference 反汇编为纯标量；scalar candidate 被自动向量化为 SSE2 4-wide 且 shuffle 占比高；
- `perf` / governor / 内存频率不可用，microcode 为虚拟值；
- guest 拓扑伪造，20 vCPU 吞吐同质，SMT 兄弟对无争用。

### 推断

- G0 的仓库内基础设施跨机可用，correctness 契约与机器无关；
- `M=1` 侧的首要优化判据应写成「有效带宽是否显著超过 7.4 GB/s」，比 GFLOP/s 更贴近真实瓶颈；`M≥16` 侧用「距峰值百分比 + 反汇编」双证据判定；
- 访问顺序差距是 stride 访存（cache line 利用率与 TLB）效应，不是代码路径效应；由于 production Linear 走 K-contiguous，「修复 N-contiguous」不是优化目标；
- binding/packing 两组成本低且稳定，当前不是瓶颈，也不应成为首轮优化对象；
- 跨进程偏移可能混合了频率漂移、hypervisor 层物理核迁移与共享工作站背景负载，但在本环境内不可分离。

### 限制

- 无硬件计数器，记忆侧定位只能依赖 compulsory-traffic 模型；
- `taskset` 为咨询性绑定，无法建立「固定到某一类物理核」的采集前提；
- Copy ceiling 因 NT store 为上界，ridge 只用 Triad 计算；
- 未建模 L2/L3 分层带宽，L1 驻留小形状的百分比仅具指示性；
- raw artifact 仅存本机 gitignored 目录，无远端 retention URL；
- 所有百分比不替代门禁值，且受各组自身噪声 floor 约束。

## 10. 门禁判定

- [x] correctness 与 safety contract baseline（106 + 30 全通过，与机器无关）；
- [x] production prepared-path benchmark（`cpu::linear_f32_reference` 经 `PrepareKernel` + `KernelParamsBuilder` + `ResolvedKernel::fn`）；
- [x] microkernel / preparation / cache-mode 分层，且各组独立归档、未合并 geomean；
- [x] allocation / workspace / ownership 的框架不变量证据；
- [x] 环境、命令与 raw artifact 可追溯（`context.json` + `checksums.sha256`）；
- [x] **保存全部 repetition 原始数据**（采集机 baseline run 未满足项，本机满足）；
- [x] **streaming 独立复跑**（同上）；
- [x] **交错 A/B 协议与噪声 floor 量化**（同上，且进一步分解为进程内/跨进程两个量）；
- [ ] 稳定百分比级 production 门禁 —— microkernel/hot/streaming 跨进程偏移超 5%，保持 blocked；
- [ ] bare-metal `perf` / governor / 真实 microcode 证据 —— 本环境原理性不可得，保持 blocked；
- [ ] P/E core 分离与 SMT 控制 —— WSL2 下不可建立，保持 blocked；
- [ ] durable CI/object-storage artifact URL —— 保持 blocked。

因此：**本机 G0 的实现复现与 reference baseline 可以关闭**；G1S/G1V 的本机 candidate 比较可以开始，但必须沿用交错 A/B 并以各组噪声 floor 判读；任何 production descriptor priority 调整仍被上述未完成项阻塞。

## 11. 后续动作

- proposal：把 G0 状态记为按机器计，`DESKTOP-QHIHOGQ` 与 `DESKTOP-54H5MMI` 各自 Baseline Complete、production gate 均 Needs More Data；
- G1S 本机 candidate 比较：沿用 `collect_g0.sh` 的交错 A/B 协议；`M=1` 以「有效带宽 > 7.4 GB/s」为首要判据；把 shuffle/unpack 占比列为调优观察项；**不引入 strict non-SIMD 变体**，遵守 2026-09-18 简化决策；
- G1V：需在本机重采 AVX2+FMA 8-wide 的实测 ceiling，因为 G1S 目前只到 SSE2 4-wide；ISA TU 按 §4.4 单独施加 `-mavx2 -mfma`，不得对整个 target 使用 `-march=native`；
- 把 raw artifact 上传 CI/object storage 后补充 retention URL 与 checksum，消除「换机即丢失」；
- 裸机环境就绪后重采 `perf` 计数器、governor、真实 microcode 与 P/E 分离数据，并重算 roofline 与百分比门禁。
