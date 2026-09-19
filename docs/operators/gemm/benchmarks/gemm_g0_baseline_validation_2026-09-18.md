# GEMM G0 Baseline 验证报告

- **状态**: Current
- **日期**: 2026-09-18
- **专项提案**: [CPU GEMM 优化方案](../cpu-gemm-optimization.md)
- **工作包**: G0 合同与证据基线
- **Change Profile**: Reference / Optimized benchmark infrastructure / Packing/Layout measurement
- **Candidate commit**: `5cbd378695bb90d97fa4273a7359bf2bfea20e8f`
- **Baseline commit**: `5cbd378695bb90d97fa4273a7359bf2bfea20e8f`（同实现独立复跑）
- **Working tree**: benchmark 代码对应 commit；artifact finalization 时仅 proposal 文档 dirty
- **采集机**: `DESKTOP-54H5MMI` — Intel Core Ultra 9 285H（Arrow Lake-H，16 核、SMT off、单 NUMA、无 AVX-512/AMX），WSL2 kernel 6.6.87.2
- **Raw artifact**: `benchmark-results/operators/gemm/20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline/`（gitignored、不随仓库分发，**仅存在于上述采集机本地磁盘**，其他机器上不存在）
- **Artifact checksum**: `context.json` SHA256 `e12077e9b1c87c3b5a16f268b250aa8a801522156a5f14175284bf439daea621`；各文件 checksum 见 context manifest
- **关联 ADR**: 无

> **机器归属**：本报告的全部数值只对采集机 `DESKTOP-54H5MMI` 成立，按提案 §6.7/§6.5 不可跨机复用；其他目标机的 G0 状态须各自采集后单独判定。

## 1. 验证目标与结论

本报告验证 G0 benchmark/contract 基础设施能够分离 direct GEMM、prepared Linear、hot/streaming、binding 和 cold packing，并记录当前 reference 实现的数量级基线。

最终判定：

- **Accepted**：作为 Intel Core Ultra 9 285H + WSL2 + GCC 14.2.0 上的首次 G0 reference baseline 和采集流程证据；
- **Needs More Data**：作为任何 optimized candidate 的百分比级 production acceptance gate。

本报告不证明 scalar optimized、SIMD、packed Linear、端到端 token latency 或跨机器性能。

## 2. 被验证实现

```text
direct primitive:
    RunGemmF32Reference

production operator:
    CpuBackend::PrepareKernel(OpType::kLinear)
    → KernelParamsBuilder
    → ResolvedKernel::fn
    → RunLinearF32Reference

cold packing:
    CpuWeightPrepacker::Pack
    → cpu_identity allocate/copy/free
```

当前只有 FP32 reference descriptor；packing benchmark 不包含 packed Linear compute。reference 使用 double accumulation 并覆盖写回 `C=A×B`。

## 3. 环境

| 项目 | 值 |
|---|---|
| CPU/model/stepping | Intel Core Ultra 9 285H / stepping 2 |
| Microcode | WSL2 报告 `0xffffffff`，不可作为可信版本 |
| OS/kernel | Ubuntu 24.04 / `6.6.87.2-microsoft-standard-WSL2` |
| Compiler/version | GCC 14.2.0 |
| Build type/flags | Release，默认 `-O3 -DNDEBUG`，C++20 |
| Effective relevant features | AVX2、FMA、AVX-VNNI；无 AVX-512/AMX |
| Threads/affinity | 单线程，`taskset -c 2` |
| NUMA | 单节点 |
| Governor/turbo/SMT | governor 不可读；宿主频率共享；SMT off |
| Memory | WSL2 约 62.3 GiB 可见内存 |
| Repetitions/min_time | 10 / 1s |

## 4. Correctness 与安全

| 维度 | 覆盖 | 结果 | 证据 |
|---|---|---|---|
| GEMM/Linear/MatMul/fused Linear | canonical、stride、zero-size、tail | PASS 80/80 | `correctness-tests.txt` |
| alias/injectivity/unknown overlap | Linear/MatMul/QKV/Gate-Up | PASS | `correctness-tests.txt` |
| overflow/invalid metadata | adapter 与 prepacker | PASS | `correctness-tests.txt` |
| packed ownership/recipe identity | store/prepacker | PASS | `correctness-tests.txt` |
| prepared params/zero allocation invariant | execution framework | PASS 15/15 | `correctness-tests.txt` supplementary section |
| numerical guard | benchmark 计时前对照 double reference | PASS | 各 benchmark 无 SkipWithError |
| sanitizers | 本工作包未运行，且本报告不声称 sanitizer 证据 | NOT RUN | 范围外 |

## 5. Benchmark 协议

- 每个 benchmark group 使用独立进程；
- 单线程固定 CPU 2；
- `--benchmark_min_time=1s`；
- `--benchmark_repetitions=10`；
- microkernel、hot、streaming、binding、packing 分开输出；
- correctness guard 在 timed loop 外；
- params build 只进入 binding benchmark；packing 只进入 packing benchmark；
- baseline/repeat 为先后两轮独立进程，不是交错 A/B；
- JSON 使用 `--benchmark_report_aggregates_only=true`，只保留 mean/median/stddev/cv。

精确命令见 raw artifact 的 `collect.sh`、`collect_repeat.sh` 和 `collect_disasm.sh`。

## 6. 结果摘要

### 6.1 Canonical reference cases（median）

| Case | Real time | GFLOP/s | Logical GB/s | 结论 |
|---|---:|---:|---:|---|
| Direct GEMM N-contiguous, M1 K4096 N4096 | 150.0 ms | 0.205 | 0.410 | inner-K 跨 N stride，reference 极慢 |
| Direct GEMM K-contiguous, M1 K4096 N4096 | 10.91 ms | 2.818 | 5.639 | Linear 实际 weight layout |
| Prepared Linear hot, M1 K4096 N4096 | 10.55 ms | 2.916 | 5.835 | 与 direct K-contiguous 数量级一致 |
| Prepared Linear streaming, M1 K4096 N4096 | 10.35 ms | 2.974 | 5.951 | 本轮与 hot 接近，不构成普遍 cache 结论 |

### 6.2 Binding 与 packing（median）

| Case | Real time | 其他指标 | 结论 |
|---|---:|---:|---|
| Linear binding, M1 K4096 N4096 | 359 ns | 64 MiB weight view | 亚微秒冷路径 metadata/params 构建 |
| cpu_identity packing, N4096 K4096 | 39.34 ms | 3.128 effective GB/s；amplification 1.0 | 包含 allocate/copy/free，不是 compute |

### 6.3 Roofline 输入

| 指标 | 结果 | 限制 |
|---|---:|---|
| cpufp AVX2+FMA FP32 单线程峰值 | 123.74 GFLOP/s | WSL2/宿主共享频率，近似值 |
| STREAM Copy 单线程 | 37.1 GB/s | best rate |
| STREAM Triad 单线程 | 22.1 GB/s | best rate |

reference 与 SIMD 峰值的比值只用于说明优化空间，不能当作未来 kernel 可达到的承诺。

## 7. 可重复性、硬件计数器与反汇编

- baseline/repeat 的结构性排序一致，但不同 case 的跨进程 delta 约为 ±10–25%；
- 当前 `compare_benchmark_json.py` 使用 aggregate mean，适合验证命名和比较链路，不足以建立 5% production gate；
- streaming 没有独立 repeat；
- `perf stat` 因 WSL2 kernel 对应 linux-tools 不可用，已保存失败输出；
- disassembly 显示 reference 内层为 scalar `cvtss2sd/mulsd/addsd`，虽然使用 XMM 标量寄存器，但没有 packed SIMD loop。

## 8. Integration 与端到端

- prepared Linear production entry 已覆盖；
- ExecutionPlan/PreparedExecutionBindings 的 params stability 和 zero-allocation invariant 有聚焦测试；
- 完整 Generate/Prefill/Decode token latency：`Not Available`；
- packed Linear compute：`Not Available`。

## 9. 事实、推断与限制

**已验证事实**：

- G0 benchmark 分层和 JSON 采集路径可运行；
- K-contiguous reference 显著快于 N-contiguous reference；
- prepared Linear framework overhead 相对 canonical compute 很小；
- 当前 reference 未形成 packed SIMD loop。

**基于数据的推断**：

- G1S 应优先优化 K-contiguous Linear/Decode 路径；
- scalar loop/layout specialization 具有明确机制空间。

**限制**：

- WSL2 运行噪声高；
- 原始 repetitions 未保存；
- streaming 未复跑；
- perf/governor/可信 microcode 不可用；
- raw artifact 当前仅保留在采集机 `DESKTOP-54H5MMI` 的 gitignored 目录，没有远端 retention URL，其他机器无法复核；
- README 的“工作树干净”与最终 context 的 proposal-only dirty 状态不一致。

## 10. 门禁判定

- [x] correctness 与 safety contract baseline；
- [x] production prepared-path benchmark；
- [x] microkernel/preparation/cache-mode 分层；
- [x] allocation/workspace/ownership 的聚焦框架证据；
- [x] 环境、命令和本地 raw artifact 可追溯；
- [ ] 保存全部 repetition 原始数据；
- [ ] streaming 独立复跑；
- [ ] 交错 A/B 与稳定百分比级门禁；
- [ ] bare-metal perf/governor/microcode 证据；
- [ ] durable CI/object-storage artifact URL。

因此：G0 的实现（benchmark/测试/采集脚本，属仓库资产）可以关闭；采集机 `DESKTOP-54H5MMI` 上的 reference baseline 可以关闭；G1S 可以开始。**其他机器的 G0 仍为 Not Collected**，且任何 production priority 调整仍被上述未完成项阻塞。

## 11. 后续动作

- proposal：将 G0 标记为“Baseline Complete on `DESKTOP-54H5MMI` Only / Production Gate Needs More Data / Not Collected on Other Machines”；
- G1S：开始 backend-private scalar optimized candidate，不改变 production descriptor priority；
- 正式 candidate 比较前，在裸机或隔离 CPU 环境补采完整 raw repetitions、streaming repeat 和 perf；
- **每台目标机（含当前开发机）各自完成一次 G0 采集并归档 raw artifact**，因为 §6.7 要求 baseline/candidate 同机；
- 将 raw artifact 上传 CI/object storage 后补充 retention URL 与 checksum。

