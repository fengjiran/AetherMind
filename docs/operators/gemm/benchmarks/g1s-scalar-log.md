# GEMM G1S Scalar 实验日志

> 本文是追加式过程记录。它不构成 production acceptance；正式验证只在满足完整配对 A/B、bare-metal 与 production-path 门禁后另建报告。

- **算子/工作包**: GEMM / G1S portable scalar optimized
- **专项提案**: [CPU GEMM 优化方案](../cpu-gemm-optimization.md)
- **采集机**: `DESKTOP-54H5MMI` — Intel Core Ultra 9 285H（16 核、SMT off、单 NUMA），WSL2 kernel 6.6.87.2，GCC 14.2.0
- **日志范围**: 2026-09-18 backend-private candidate、本地 diagnostic smoke 与同日简化重构
- **原始数据位置**: `benchmark-results/operators/gemm/20260918T141000Z_g1s-local-diagnostic/`（gitignored、不随仓库分发，**仅存在于上述采集机本地磁盘**，其他机器上不存在）
- **正式报告**: 尚无；production gate 仍为 `Needs More Data`

> **机器归属**：本日志中的 smoke 时延、汇编与噪声观察均来自采集机 `DESKTOP-54H5MMI`，不可跨机复用。代码层面的事实（candidate 入口、fallback 边界、opt-in descriptor 注册、测试通过与否）与机器无关，在任何机器上都可由仓库源码复核；性能数字则必须在目标机重采。

## 日志索引

| 日期 | 实验 ID | 假设 | 结论 | 状态 |
|---|---|---|---|---|
| 2026-09-18 | G1S-SCALAR-001 | `M=1` 下的 `NR=4`、K unroll 2 与连续 RHS 能改善 reference 的 loop/layout 开销 | candidate、fallback 与 FP32 error budget 已实现并通过聚焦测试 | In Progress |
| 2026-09-18 | G1S-SCALAR-002 | strict non-SIMD 可分离 loop/layout 与 compiler auto-vectorization 的贡献 | strict 汇编没有 packed SIMD；本机单次 smoke 不能作为性能门禁 | Superseded（strict 机制已移除，见 G1S-SCALAR-003） |
| 2026-09-18 | G1S-BINDING-001 | Linear 可在 binding time 冻结 exact scalar/reference driver，而非在 Execute 二次选择 | opt-in descriptor、prepared bindings 和 resolved-entry smoke 已通过；production acceptance 仍需正式证据 | Superseded（冻结机制移除，descriptor 保留简化直调版；见 G1S-SCALAR-003） |

## 2026-09-18 — G1S-SCALAR-001：backend-private portable candidate

### 1. 假设

- 目标机制：对 `M=1` 的连续 activation/output 与 K-contiguous（Linear weight）或 N-contiguous RHS，以四个 output accumulator、pointer bumping 和 K unroll 2 降低 reference 的地址计算、分支与单 accumulator 依赖；
- 目标 shape/layout：`M=1`，`lhs_k_stride=1`，`output_n_stride=1`，且 `rhs_k_stride=1` 或 `rhs_n_stride=1`；
- 成功判定：与 double reference 的 FP32 `atol/rtol=1e-4` 数值合同一致，且其他 reference-legal layout 逐一 fallback；
- 反例：多 row、strided activation、strided output 或非连续 RHS 不得因 candidate 而缩窄合同。

### 2. 代码与环境

```text
baseline commit: 5cbd378695bb90d97fa4273a7359bf2bfea20e8f
candidate commit: uncommitted local G1S candidate on the baseline checkout
working tree: dirty only for G1S implementation, tests, benchmark/CMake, and this log
CPU: Intel Core Ultra 9 285H, WSL2-visible 16 CPUs, one NUMA node
OS/kernel: Ubuntu 24.04 / 6.6.87.2-microsoft-standard-WSL2
compiler: GCC 14.2.0
build: Release, C++20; strict source flags -fno-tree-vectorize -fno-tree-slp-vectorize
effective ISA used by candidate source: no explicit intrinsic; portable source permits auto-vectorization
affinity: taskset -c 2 for benchmark smoke
run ID: 20260918T141000Z_g1s-local-diagnostic
artifact checksums: checksums.txt in the raw artifact directory
```

### 3. 改动

- 新增 `RunGemmF32ScalarOptimized`，不修改 `RunGemmF32Reference`；
- fast path 仅覆盖 `M=1` 的 K-contiguous 或 N-contiguous RHS，采用 `NR=4`、K unroll 2；N tail 逐 scalar output 处理；
- `K=0`、`M!=1`、非 unit lhs-K/output-N stride 或其他 RHS layout 直接调用 double reference；
- source 通过共享 implementation include 同时构建 portable 和 strict diagnostic 函数，避免复制两份算法；strict 翻译单元的 vectorization-disable flags 经 `CheckCXXCompilerFlag` 检测，仅作用于该文件；
- `benchmark_cpu_gemm_microkernel.cpp` 现在比较 reference、portable scalar 与 strict scalar；`benchmark_cpu_linear.cpp` 新增 benchmark-only prepared scalar candidate，仍复用 registered Linear 的 params builder，未替换 `ResolvedKernel::fn` 或 descriptor。

### 4. 验证命令

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMM_STRICT_SCALAR_DIAGNOSTIC=ON
cmake --build build-release --target aethermind_unit_tests aethermind_benchmark -j
TMPDIR=/tmp ./build-release/tests/unit/aethermind_unit_tests \
  --gtest_filter='CPUKernelGemmScalar*:*CPUKernelGemmScalar*:*CPUKernelGemmReference.*'
taskset -c 2 ./build-release/tests/benchmark/aethermind_benchmark \
  --benchmark_filter='BM_GemmF32(Reference|ScalarOptimized|StrictScalar)(N|K)Contiguous/M:1/K:33/N:31$' \
  --benchmark_min_time=1ms --benchmark_repetitions=1 \
  --benchmark_out=<run-id>/direct-tail.json --benchmark_out_format=json
objdump -d -C --disassemble='aethermind::cpu::detail::RunGemmF32ScalarOptimized(...)' \
  build-release/src/libAetherMind.so
objdump -d -C --disassemble='aethermind::cpu::detail::RunGemmF32ScalarOptimizedStrictDiagnostic(...)' \
  build-release/src/libAetherMind.so
```

### 5. Correctness

| 测试 | 结果 | 备注 |
|---|---|---|
| `CPUKernelGemmScalar*`、parameterized fast-path cases、reference GEMM | PASS 28/28 | K=`0/1/31/32/33/127/128/129/4096`、N tail、N/K-contiguous、padded、unaligned、overwrite、fallback |
| Release benchmark build | PASS | portable/strict entries 和 prepared candidate 均成功链接 |
| direct/Linear benchmark correctness guards | PASS | canonical FP32 candidate 按 `1e-4` relative-or-absolute budget 验证 |

### 6. 性能结果摘要

| Case | Baseline | Portable | Strict | Repetitions | Raw artifact |
|---|---:|---:|---:|---|---|
| Direct N-contiguous, M1 K33 N31 | 641 ns | 127 ns | 192 ns | 1 smoke | `direct-tail.json` |
| Direct K-contiguous, M1 K33 N31 | 612 ns | 254 ns | 252 ns | 1 smoke | `direct-tail.json` |
| Direct K-contiguous, M1 K4096 N4096 | 12.18--15.28 ms | 4.83--5.96 ms | 4.20--4.80 ms | 2 independent smokes | `canonical-smoke.json`, `final-canonical-smoke.json` |
| Prepared Linear scalar candidate, M1 K4096 N4096 | — | 4.93--5.78 ms | — | 2 independent smokes | `canonical-smoke.json`, `final-canonical-smoke.json` |

这些是 load average 约 7--19 的 WSL2 单次 smoke 数据，未做完整 repetitions、配对 A/B、streaming repeat 或 bare-metal counter；它们只证明 benchmark path 和量级可运行，不能用于 production priority 或性能承诺。

### 7. 分析

**已验证事实**：

- portable release 反汇编含 `mulps/addps` 等 packed SSE 指令，编译器确实对 source 的部分路径自动向量化；
- strict release 反汇编只含 scalar `mulss/addss`（以及 XMM 标量寄存器搬运），没有 packed SIMD loop；
- strict and portable 与同一 double reference 使用相同数据、shape 和 direct benchmark 计数口径；
- candidate 不是 registered production kernel，现有 descriptor 名称、priority、params builder 与 `ResolvedKernel::fn` 未改变。

**推断**：

- `NR=4`、K unroll 2 的 loop/layout 改写至少在 local diagnostics 中值得继续评估；
- portable 与 strict 的差异需要在成对、低噪声环境复测后，才能量化 compiler auto-vectorization 的贡献。

**尚未验证的假设**：

- `NR=4`、K unroll 2 优于 `NR=2` 或 K unroll 1；本次只实现最小候选，未进行参数 sweep；
- hot/streaming prepared candidate 在全部 canonical Decode shapes 的稳定收益；
- 完整 Production Linear integration 的 allocation、fallback 与 end-to-end 行为。

### 8. 决定

- **In Progress / Needs More Data**：G1S-A backend-private candidate、correctness 与 diagnostic bench 已完成；
- 不创建 G1S Accepted validation report，不调整 descriptor priority，不开始 G1V production integration；
- 下一步：完成 `NR={2,4}`、K unroll=`{1,2}` 的受控 sweep，保存完整 repetitions、交错 A/B 与 streaming 数据，并在裸机采集 perf/assembly；证据达标后才评审 binding-time driver freeze 和 production descriptor 替换。

## 2026-09-18 — G1S-BINDING-001：opt-in production-like Linear integration

### 1. 假设

- `KernelParamsBuilder` 已拥有 concrete row count、stride 和 layout，可在 cold path 冻结 exact `GemmF32Driver`；
- candidate entry 只间接调用已保存 driver，就不会在 Execute 再检查 shape/layout 或重新 resolve；
- 由于 G0 production gate 仍未关闭，candidate 必须是显式 CMake opt-in，默认 descriptor 不变。

### 2. 改动

- 新增 `AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE`，默认 `OFF`；OFF 保留唯一的 `cpu::linear_f32_reference` descriptor、name、params builder 和 reference entry；
- ON 时只注册同一 selector 下唯一的 `cpu::linear_f32_scalar_candidate` descriptor，不建立第二套 registry 或 implementation selector；
- candidate params builder 对 `M=1`、nonzero K、unit input-K/output-N 的 plain Linear views 冻结 `RunGemmF32ScalarOptimizedM1KContiguous` 或 `RunGemmF32ScalarOptimizedM1NContiguous`；zero K、multi-row、strided output/input 或其他 RHS layout 冻结 `RunGemmF32Reference`；
- candidate entry 调用 `RunLinearF32FrozenGemmDriver`，后者只构造 prevalidated `GemmF32Args` 并调用已冻结的函数指针；
- prepared candidate benchmark 现在真实经过 `CpuBackend::PrepareKernel → candidate params builder → ResolvedKernel::fn`，不再使用 benchmark-local reinterpret/invoker。

### 3. 验证

```text
default OFF Release: build-release
candidate ON Release: build-gemm-scalar-candidate
candidate ON options: -DAETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE=ON
                      -DGEMM_STRICT_SCALAR_DIAGNOSTIC=ON
run ID: 20260918T150000Z_g1s-b-local-diagnostic
raw artifact: benchmark-results/operators/gemm/20260918T150000Z_g1s-b-local-diagnostic/
```

| 证据 | 结果 |
|---|---|
| OFF resolve/name + retained M=2 ExecutionPlan integration | PASS；`cpu::linear_f32_reference`，相关聚焦测试 26/26 |
| ON descriptor、K/N scalar freeze、reference fallback、zero K | PASS；candidate 聚焦测试包含 4 个 `CPUKernelLinearScalarCandidate` 用例 |
| ON unaligned fast path、padded multi-row fallback、M=1 PreparedExecutionBindings → Executor | PASS；candidate 聚焦测试总计 32/32 |
| ON prepared resolved-entry smoke | PASS；label 为 `kernel=cpu::linear_f32_scalar_candidate` |
| strict source compile/assembly | PASS；仅 strict source 有 vectorization-disable flags，汇编为 scalar `mulss/addss` |

`M=1,K=4096,N=4096` 的 candidate ON 本机 smoke 约为 5.1--6.0 ms；该 run 的 load average 约 36--65、每 case 仅 1 repetition，不能与 G0 baseline 形成可信百分比比较，也不构成 descriptor priority 或 production acceptance 证据。

### 4. 结论

- **Implemented for opt-in diagnostics**：G1S-B 的 binding-time integration、fallback 和 resolved-entry benchmark 已完成；
- **Needs More Data**：默认仍为 OFF，且尚未完成 paired A/B、完整 repetitions、streaming、bare-metal perf/governor/microcode、durable artifact retention 和 end-to-end token evidence；
- 不创建 Accepted validation report，不把该 option 解释为默认产品能力。

## 2026-09-18 — G1S-SCALAR-003：简化重构（移除 strict 与 exact/export，单入口直调）

### 1. 决策

- 不建设 strict non-SIMD 归因机制：允许编译器自动向量化，不做 loop/layout 与 vectorization 的贡献拆分（撤销 G1S-SCALAR-002 的目标）；
- 不保留 export-exact/冻结驱动机制：Linear 集成改为 candidate entry 直调单入口，hot path 的“分派”退化为入口内数次整数比较（撤销 G1S-BINDING-001 的冻结设计）；
- 保留：scalar 入口与 reference fallback、opt-in candidate descriptor（默认 OFF）、microkernel/Linear 基准与聚焦测试。

### 2. 改动

- 删除 `gemm_f32_scalar_impl.inc`、`gemm_f32_scalar_strict.cpp`；`gemm_f32_scalar.cpp` 单文件承载 helpers + `RunGemmF32ScalarOptimized` 单入口；
- 删除宏/机关：`GEMM_STRICT_SCALAR_DIAGNOSTIC`（含 `_ENABLED`/`_FLAGS`）、`GEMM_SCALAR_FUNCTION`、`GEMM_SCALAR_EXPORT_EXACT_DRIVERS`；
- 删除符号：`RunGemmF32ScalarOptimizedM1{K,N}Contiguous`、`RunGemmF32ScalarOptimizedStrictDiagnostic`、`GemmF32Driver`、`LinearF32KernelArgs::gemm_driver`、`SelectLinearF32ScalarCandidateDriver`、`RunLinearF32FrozenGemmDriver`；
- `LinearF32ScalarCandidateEntry` 组装 `GemmF32Args` 后直调 `RunGemmF32ScalarOptimized`；candidate/reference 两个 descriptor 共用同一 params builder；
- strict 基准与 driver 指针断言测试移除；`CPUKernelLinearScalarCandidate` 测试改为行为验证并重命名（FastAndFallbackViews / ZeroInnerDimension / EndToEnd）。

### 3. 验证（本机，build-release）

| 证据 | 结果 |
|---|---|
| ON 构建 + 聚焦测试（Linear/candidate/scalar/reference） | PASS 34/34 |
| OFF（默认）构建 + 聚焦测试 | PASS 31/31；candidate 套件按门控编译为空 |
| 基准清单：strict 用例 | 0（两模式） |
| 基准清单：scalar 微内核 / candidate Linear | 8 / 14（ON）；8 / 0（OFF） |
| ON 冒烟：`BM_LinearPreparedScalarCandidateHot` tail case | 234 ns；label `kernel=cpu::linear_f32_scalar_candidate`；correctness guard 通过 |
| ON 冒烟：`BM_GemmF32ScalarOptimizedKContiguous` tail case | 252 ns；correctness guard 通过 |

### 4. 决定

- **Accepted**：单入口 + 直调集成为当前 G1S 实现形态；下游 G1V/G2 不再假设存在 strict 或 exact-冻结机制；
- **Needs More Data（不变）**：G1S 的收益证据（受控 sweep、配对 A/B、完整 repetitions、streaming、bare-metal）仍待补齐；
- 归因机制（strict）与冻结机制（exact drivers）曾在同日完整实现并验证；如未来证据需要归因，可按本日志历史条目回溯引入。
