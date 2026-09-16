# CPU GEMM 优化方案

- **状态**: Draft
- **版本**: 1.0
- **日期**: 2026-09-16
- **产品边界**: [AetherMind 当前产品 PRD](../products/aethermind_prd.md)
- **架构基线**: [架构总览](../designs/architecture/architecture_overview.md)
- **优化方法基线**: [算子优化指南](../guides/operator_optimization_guide.md)
- **关联模块**: backend / execution / model / benchmark

## 1. 结论与范围

AetherMind 不应新增 semantic `Gemm` operator。当前 GEMM 是 CPU backend 的内部计算 primitive，语义入口是 `MatMul`、`Linear`、`QkvLinear` 和 `GateUpLinear`。推荐建设一个共享的 CPU GEMM engine，由各算子 adapter 完成语义、layout、alias 和 packed-weight 校验，再把 compute-ready problem 交给 GEMM driver。

实施优先级不是“先做通用大矩阵 SGEMM”，而是：

1. 建立可信 benchmark 与 Roofline 基线；
2. 优先优化 LLM Decode 的 `M=1` GEMV / skinny GEMM；
3. 再优化 Prefill 的 blocked GEMM；
4. 将真实 packed-weight recipe 接入 kernel resolve 和 materialization；
5. 在量化合同稳定后增加 INT8/INT4 路径；
6. 多线程只通过 runtime 统一线程池演进，不在 kernel 内引入 OpenMP 或私有线程。

第一条完整 production 优化路径建议为 **x86-64 FP32 AVX2+FMA、单线程、immutable packed weight、无运行时分配**。在 exact recipe 链路完成前，可以先用 plain/`cpu_identity` weight 实现直接访问的 AVX2 Decode baseline，但它只是风险收敛里程碑，不是最终 packing 方案。AVX-512、AMX、AArch64、INT8/INT4 和多线程均是后续独立里程碑，不能写成当前已实现能力。

### 1.1 包含

- `Linear` / fused Linear 的 GEMV、skinny GEMM 和 blocked GEMM；
- 通用 `MatMul` 对共享 GEMM engine 的复用边界；
- SIMD microkernel、cache blocking、packing、prefetch 和 dispatch；
- binding-time shape specialization 与 workspace 演进；
- correctness、microbenchmark、production-path benchmark、packing benchmark 和最终端到端验证设计。

### 1.2 不包含

- 修改 Graph IR 或新增 `OpType::kGemm`；
- 在本提案中直接实现 kernel、量化格式或线程池；
- 把第三方 BLAS 引入为生产依赖；
- GPU GEMM、MoE、continuous batching 或分布式执行；
- 在没有测量证据时承诺具体 tile size、prefetch distance 或性能倍数。

### 1.3 与旧草案的关系

本文替代 [2026-09-04 CPU FP32 GEMM 草案](../designs/kernel_dev/CPU_FP32_GEMM优化方案.md)。旧草案主要面向通用 `MatMul` 的无 packing AVX2 路径；本文保留其 binding-time driver 和 descriptor-internal fallback 思路，并补齐 LLM Linear/fused Linear 优先级、exact packed recipe、workspace 演进、streaming-weight benchmark 及分阶段证据门禁。旧草案保留为历史机制记录，不代表当前已实现设计。

## 2. 已验证的当前状态

| 能力 | 当前实现 | 对优化的影响 |
|---|---|---|
| GEMM primitive | `RunGemmF32Reference`，三重循环，double accumulator，支持二维 stride | correctness baseline，不是性能基线 |
| MatMul | FP32 reference；支持 batch broadcast、`transpose_rhs` 和任意已验证 stride | 通用性高，首轮优化不应受其最宽 layout 合同约束 |
| Linear | FP32 plain-weight reference；将 leading dimensions flatten 为 `row_count` | LLM unfused path 可作为 production adapter 样板 |
| QkvLinear / GateUpLinear | FP32 packed-only reference；当前 packed payload 是 `cpu_identity` | 已打通 opaque artifact 与 execution binding，但没有真实 tile packing |
| kernel resolve | `CpuBackend::PrepareKernel` 按 selector、CPU feature 和 priority 选 descriptor | resolve 时尚无 concrete shape/layout，不能按 `M/N/K` 选择算法 |
| binding specialization | `KernelParamsBuilder` 可看到固定的 pointer/shape/stride/dtype | 可在 cold path 选择 GEMV/skinny/blocked driver，并把函数指针写入 prepared params |
| workspace | `ResolvedKernel.workspace_requirement` 在 binding 前规划 | 不能表达依赖 concrete `M/N/K` 的 transient A/B packing scratch |
| packing recipe | `CpuWeightPrepacker::RecipeFor(selector)` 返回全局 `cpu_identity` | 不足以表达 AVX2/AVX-512/AMX 或 tile/version 不同的 layout |
| CPU dispatch | 已有 AVX2/FMA/AVX-512/VNNI/AMX、NEON/DotProd/I8MM/SVE 等 capability model | feature gate 基础可复用；当前仅 RMSNorm 有 AVX2+FMA optimized descriptor |
| threading | 当前产品边界为单请求、单线程，现有 kernel 也保持单线程 | 首轮优化保持单线程；并行化属于 runtime 级后续工作 |
| benchmark | Google Benchmark；RMSNorm 已示范 prepared kernel 与 binding-cost 分离 | GEMM 需新增 shape、cache-state、packing 和生产路径维度 |

### 2.1 当前主要瓶颈

1. reference loop 的 `row → col → k` 每个输出点重新流过 K，缺少寄存器 blocking，无法复用 A/B 数据；
2. 没有 SIMD microkernel、cache blocking 或 software prefetch；
3. `cpu_identity` 只是 aligned copy，不降低 microkernel 的地址计算和访存代价；
4. Decode 与 Prefill 形状差异巨大，却只能 resolve 到同一个固定 kernel entry；
5. packed recipe 由 selector 而非实际 descriptor 决定，无法安全支持同一 selector 下的多 ISA layout；
6. shape-dependent workspace 尚无合同，不能直接加入 transient panel packing；
7. 现有 benchmark 没有覆盖 LLM 真实形状，也没有区分 hot-cache、streaming-weight、packing cost 和 binding cost。

## 3. 约束与 invariant

### 3.1 语义与模块边界

- `operators/` 只定义 `MatMul`/`Linear`/fused Linear 语义，不出现 ISA、tile、packing 或 workspace 细节。
- `backend/cpu/kernels/gemm/` 拥有 GEMM problem、planner、driver、packing layout 和 microkernel。
- `Linear`、`QkvLinear`、`GateUpLinear` 和 `MatMul` adapter 继续拥有各自的 shape、stride、alias 和 output-layout 校验。
- compiler 不选择 AVX2/AVX-512/AMX，也不携带 packed pointer 或 concrete kernel。
- execution 只消费 `ResolvedKernel`、exact `PackingRecipe`、prepared params 和 workspace，不理解 tile layout。

### 3.2 热路径

- `Execute` 中不进行 registry lookup、shape validation、heap allocation、weight repack 或算法搜索。
- shape/layout 分类、tail 策略和 driver 选择在 `PrepareExecutionBindings` 的 cold path 完成。
- immutable model weights 只在 model preparation 阶段 pack；steady-state Decode 不 pack B。
- packed artifact 的 recipe、logical dtype、logical shape、alignment 必须与 resolved kernel 精确一致。

### 3.3 数值与安全

- reference kernel 保留 double accumulation，作为 correctness oracle，不用其结果定义 optimized FP32 的逐 bit 一致性。
- optimized FP32 默认使用 FP32 FMA accumulation；误差按绝对误差、相对误差和规模相关容差验收。
- output 必须 injective；Linear/fused Linear output 与 input/weight 必须 proven disjoint。
- stride-hole 或 overlap 无法证明安全时返回 `Unimplemented`，不能把 unknown 当作 disjoint。
- `M/N/K == 0`、非 tile 整除 tail、非连续 row stride 和整数溢出继续有确定行为。
- 高优先级 optimized descriptor 一旦被 resolve，就必须覆盖该 descriptor 声明的完整 layout 合同；不适合 SIMD fast path 的合法输入应在 descriptor 内选择 compatible scalar driver，不能期待 registry 在 binding 失败后自动回退到另一个 descriptor。

## 4. 目标架构

```text
Linear / QkvLinear / GateUpLinear / MatMul adapter
        │  validate semantic layout, alias and packed metadata
        ▼
Binding-time GemmPlan builder
        │  classify shape: GEMV / skinny / blocked
        │  freeze pointers, strides, tile sizes, tails and driver fn
        ▼
Prepared operator params
        │  no allocation or dispatch search in Execute
        ▼
GEMM driver
        ├── direct-A + packed-B GEMV/skinny path
        ├── blocked GEMM path
        └── scalar/reference fallback
                ▼
        ISA microkernel + epilogue
```

### 4.1 Backend-internal contracts

建议引入 backend-private 数据结构，名称仅为设计草案：

```cpp
struct GemmProblemF32 {
    const float* a;
    const void* packed_b;
    float* c;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t a_row_stride;
    int64_t a_col_stride;
    int64_t c_row_stride;
    int64_t c_col_stride;
};

using GemmDriverF32 = Status (*)(const GemmPreparedF32&) noexcept;

struct GemmPreparedF32 {
    GemmProblemF32 problem;
    GemmDriverF32 driver;
    int32_t mc;
    int32_t nc;
    int32_t kc;
    int32_t mr;
    int32_t nr;
    // tail policy and optional epilogue descriptor
};
```

这里的核心不是具体字段，而是把两层选择分开：

- plan-build time：registry 根据 dtype、weight format、phase 和 CPU capabilities 选择 ISA descriptor；
- binding time：descriptor 的 params builder 根据 concrete shape/layout 选择该 ISA 内部的 GEMV、skinny 或 blocked driver。

这样可以在不扩大 hot-path dispatch 的前提下绕开当前 `PrepareKernel` 无 shape 的限制。若以后 workspace 或 recipe 也依赖 concrete shape，再升级公开 prepare/specialization 合同，而不是把 runtime shape 塞入 `KernelSelector`。

### 4.2 三类算法路径

| 路径 | 典型形状 | 首选并行/复用方向 | 设计重点 |
|---|---|---|---|
| GEMV | `M=1` | N 方向多个输出通道 | packed B 连续流读、A 驻留 L1、多个 accumulator、减少水平归约 |
| Skinny GEMM | `2 <= M <= threshold` | 同时复用 A 行和 B panel | M-specialized microkernel，避免大 GEMM packing 固定成本 |
| Blocked GEMM | 大 M | `MC/NC/KC` cache blocking | B 预打包、可选 A panel packing、寄存器 tile、tail kernel |

`threshold` 和 tile 参数必须由 benchmark 在目标 CPU 上确定，不写成跨平台常量。首轮可以提供保守静态表；后续只允许在 model preparation/binding cold path 做轻量选择，不在每次 Execute 自动调优。

### 4.3 Microkernel

AVX2+FMA 分成两个物理路径：

- direct/`cpu_identity` GEMV 沿 K 向量化，同时展开多个 N 输出行并做水平归约；
- interleaved packed-B GEMM 沿 N 向量化，广播 A 标量并加载连续 B vector。

真实 packed-B microkernel 建议：

- 以 N 方向向量化，C 的 `MR × NR` tile 驻留寄存器；
- K 循环广播 A 标量并加载 packed B vector；
- full-tile 与 M/N/K tail 分离，tail 不越界读取；
- microkernel 不做 shape/alias 检查；
- 不使用与合法 alias 合同冲突的 `restrict`；
- prefetch distance 和 unroll factor 通过反汇编、硬件计数器和 benchmark 决定；
- ISA translation unit 单独使用 `-mavx2 -mfma`，descriptor 显式声明 `{kAvx2, kFma}`。

每个 packed layout 必须自带能处理所有合法 tail/小 shape 的 compatible fallback driver；不能把同一字节流交给只理解 `cpu_identity` 的 reference kernel。

AVX-512、NEON/SVE 使用同一 driver contract、不同 microkernel 与 recipe。AMX 不作为 AVX2 的简单高优先级替换：tile setup 与 M 较小时的固定成本可能使 Decode 退化，必须按 shape 单独选择。

### 4.4 Packed weight

真实 packing layout 至少编码：

- format version；
- dtype / quantization scheme；
- ISA family 或兼容域；
- `NR/KR` 与 panel 顺序；
- logical N/K 和 tail padding 规则；
- alignment；
- 量化路径需要的 scale/zero-point/group metadata 布局。

例如 recipe 名可以采用 `cpu_f32_bpanel_v1_avx2_nr8_kr1`，但具体 tile 只有测量后才能冻结。

当前 `CpuWeightPrepacker::RecipeFor(selector)` 必须演进。推荐：

1. `KernelDescriptor` 提供其精确 packing recipe；
2. `CpuBackend::PrepareKernel` 把 descriptor recipe 复制到 `ResolvedKernel`；
3. packing request 从已 prepare 的 kernel 收集 exact recipe；
4. backend 提供按 recipe pack 的服务，model 层不直接实例化具体 `CpuWeightPrepacker`；
5. `PackedWeightStore` 继续以 binding + selector + exact recipe 区分 artifact。

在上述链路闭环前，只能新增与 `cpu_identity` 兼容的计算优化，不能让 optimized descriptor 悄悄解释另一种物理布局。

### 4.5 Workspace

第一阶段使用 direct A + prepacked B，保持 zero-workspace。Prefill 需要 A panel packing 时，再引入 binding-dependent workspace：

```text
Resolved ISA kernel
    → binding specialization sees M/N/K/layout
    → computes params + WorkspaceRequirement
    → PreparedExecutionBindings aggregates workspace layout
    → Execute binds preplanned slice
```

不能在 kernel 内临时 `malloc`，也不能用一个按最大模型 shape 永久膨胀的全局 scratch 规避合同设计。

### 4.6 Threading 与 NUMA

当前产品边界内保持单线程。未来引入 runtime thread pool 后：

- Decode `M=1` 优先按 N tiles 切分；
- Prefill 优先按 M/N tiles 切分；
- 避免 K-split，除非矩阵极端 skinny 且 reduction 成本有测量收益；
- 一个 output tile 只由一个 worker 写，避免 false sharing；
- worker 不得再次进入并行区，防止 oversubscription；
- packed weight placement 与 first-touch/NUMA policy 由 runtime/model preparation 管理。

## 5. 方案与备选

### 5.1 推荐：自有小型 GEMM engine

优势：

- 与 opaque packed weight、fused QKV/Gate-Up、prepared bindings 和量化 metadata 深度一致；
- 可为 Decode 的 GEMV/skinny shape 专门优化；
- hot path 可保持零分配、无通用 BLAS dispatch。

代价：

- microkernel、packing、tail、ISA 和 benchmark 的维护成本高；
- 必须建立严格 correctness 与性能回归门禁。

### 5.2 第三方 BLAS / oneDNN

可作为 benchmark 对照和 Prefill 原型，不建议立即成为 production 依赖。通用库能快速提供成熟 SGEMM，但不自动解决 AetherMind 的 fused operator、exact recipe、weight-only quantization、prepared binding 和 Decode 小 M 问题。若实测显示其覆盖主要形状且集成成本可接受，可通过 backend-private adapter 重新评估。

### 5.3 编译器自动向量化

只适合作为 baseline。它难以稳定生成跨 K blocking、B panel packing 和多个输出 accumulator 的目标代码，也不能替代 shape-specific dispatch。所有“已向量化”结论必须核对 Release 构建反汇编和硬件计数器。

## 6. Benchmark 设计

### 6.1 四层 benchmark

| 层级 | 建议文件 | 测量对象 | 用途 |
|---|---|---|---|
| Microkernel | `tests/benchmark/cpu_kernels/benchmark_cpu_gemm_microkernel.cpp` | 固定 MR/NR/K tile | 选择 unroll、tail、prefetch，不作为产品收益结论 |
| Prepared operator | `tests/benchmark/cpu_kernels/benchmark_cpu_linear.cpp` | `ResolvedKernel.fn` + prepared params | 主性能门禁，覆盖真实 dispatch/packed contract |
| Packing | `tests/benchmark/cpu_kernels/benchmark_cpu_weight_packing.cpp` | logical weight → artifact | 启动成本、吞吐、内存放大和 amortization |
| Execution integration | 后续 `benchmark_execution_gemm_path.cpp` | `ExecutionPlan` + prepared bindings + workspace | 证明 adapter、binding、workspace 没有抵消 kernel 收益 |

端到端 `prefill latency`、`decode ms/token`、`tokens/s` 必须等真实 Generate vertical slice 可运行后再加入；单算子 benchmark 不能替代它。

### 6.2 Shape 矩阵

统一用 Linear 约定 `A[M,K] × W[N,K]^T → C[M,N]`。

#### LLM canonical shapes

| 场景 | M | K | N | 说明 |
|---|---:|---:|---:|---|
| Decode projection | 1 | 4096 | 4096 | attention o/downstream projection |
| Decode QKV fused | 1 | 4096 | 6144 | 4096 Q + 1024 K + 1024 V 示例 |
| Decode Gate-Up fused | 1 | 4096 | 22016 | 2 × 11008 示例 |
| Decode Down | 1 | 11008 | 4096 | MLP down projection |
| Decode lm_head | 1 | 4096 | 32000 | vocabulary projection 示例 |
| Prefill projection | 16/64/128/512/2048 | 4096 | 4096 | 从短 prompt 到长 prompt |
| Prefill Gate-Up | 16/128/512 | 4096 | 22016 | 大 N streaming |
| Prefill Down | 16/128/512 | 11008 | 4096 | 大 K |

这些是代表性 benchmark shape，不是模型语义常量。后续应从实际 `HfModelConfig` 生成 4096/11008、4096/14336、GQA 和 vocabulary 变体。

#### Boundary shapes

- `M={1,2,3,4,7,8,15,16,17}`；
- `N/K` 覆盖 `NR/KR - 1`、整除和 `+1`；
- `K={0,1,31,32,33,127,128,129,4095,4096,4097}`；
- padded row stride、合法非连续 stride、unaligned input/output；
- zero M/N/K；
- batch broadcast MatMul 作为独立兼容性组，不混入 Linear 主性能 geomean。

### 6.3 Cache-state 模式

每个关键 shape 至少区分：

1. **hot artifact**：反复使用同一 packed weight，测 microkernel 上限；
2. **streaming artifact**：轮转总量明显大于 LLC 的多份 packed weight，模拟逐层 Decode；
3. **cold start**：单独测 prepare + pack，绝不混入 steady-state kernel loop。

不能通过 timed loop 内 `memset` 大缓冲来“清 cache”，那会把清理成本计入结果。streaming 模式应预先分配并轮转 buffer；工作集大小记录为 benchmark label/counter。

### 6.4 Baseline 与对照

- correctness oracle：`RunGemmF32Reference` 或算子 reference kernel；
- performance baseline：当前 scalar production descriptor；
- candidate：AVX2/AVX-512/量化 descriptor；
- optional external ceiling：相同线程数的成熟 BLAS/oneDNN，仅作研究对照；
- scalar 与特定 ISA 通过 `CpuFeaturePolicy` 或显式测试入口固定，不能依赖运行机器“碰巧”选择某个 kernel。

### 6.5 计数器

Google Benchmark 输出：

- latency (`ns`/call)；
- `GFLOP/s`，FP32 约定 `2*M*N*K`；
- logical `GB/s` 下界：`sizeof(T) * (M*K + N*K + M*N)`，明确它不是实际 DRAM traffic；
- `items/s` 或 output elements/s；
- packing `GB/s`、packed bytes、size amplification；
- resolved kernel name、recipe、M/N/K、cache mode。

外部 `perf stat` 至少采集 cycles、instructions、branches、branch-misses、cache-misses；可用时增加 L1/LLC、FP arithmetic 和 memory-controller 事件。事件名与解释必须随 CPU 型号记录，不能跨微架构直接比较原始计数。

### 6.6 Correctness guard

correctness 不放进 timed loop。每个 benchmark case 在计时前运行一次 reference 对照：

- 输入使用固定 seed；
- 覆盖随机值、正负抵消、大/小量级和特殊 tail；
- 拒绝 NaN/Inf（除非该 case 专门验证传播语义）；
- 记录 max absolute error、max relative error；
- 量化路径另记录 per-row/channel 误差分布和模型级质量门禁。

unit tests 负责完整 correctness；benchmark 的 guard 只防止“错误实现跑得更快”。

### 6.7 运行协议

推荐 Release 构建，并保存原始 JSON：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON
cmake --build build-release --target aethermind_benchmark -j

taskset -c <isolated-cpu> \
  ./build-release/tests/benchmark/aethermind_benchmark \
  --benchmark_filter='BM_(Gemm|Linear|WeightPacking)' \
  --benchmark_min_time=1s \
  --benchmark_repetitions=10 \
  --benchmark_report_aggregates_only=true \
  --benchmark_out=/tmp/aethermind_gemm_candidate.json \
  --benchmark_out_format=json

python3 tools/compare_benchmark_json.py \
  --baseline /tmp/aethermind_gemm_baseline.json \
  --candidate /tmp/aethermind_gemm_candidate.json
```

测量报告必须记录 CPU 型号、microcode、kernel、compiler/version、CMake flags、CPU governor、SMT、NUMA binding、内存频率和有效 CPU feature。baseline/candidate 使用同一机器、同一配置、独立进程，并采用交错 A/B 次序。小于系统噪声的单次差异不是收益证据。

### 6.8 性能判定

初始门禁建议：

- 所有 correctness case 通过；
- canonical Decode 与 Prefill 分别计算 geomean，不用大 GEMM 掩盖 `M=1` 退化；
- 关键路径的 candidate 置信区间显示稳定收益；
- 任一 canonical shape 回退超过 5% 必须解释或增加 shape dispatch fallback；
- steady-state prepared benchmark 内零 heap allocation；
- pack 成本单独报告，并给出 `pack_time / (baseline_time - optimized_time)` 的 break-even invocation count；
- 性能结论同时给出 hot 与 streaming artifact 结果。

不预先规定“必须达到某个 GFLOP/s”。在首轮基线采集后，结合目标 CPU 的可实现峰值和 memory bandwidth，为各 shape 类别设置硬门禁。

## 7. 分阶段实施

### G0：证据基线

- 新增 prepared Linear/GEMM、packing 和 cache-mode benchmark；
- 补齐 canonical/boundary correctness tests；
- 记录 scalar 的反汇编、Roofline 输入和 JSON baseline；
- 验证 benchmark 不把 params build、pack 或 allocation 混入 steady-state loop。

退出条件：baseline 可重复，shape/cache/packing 维度可区分，结果可以由脚本比较。

### G1：FP32 AVX2+FMA Decode

- 增加 AVX2+FMA translation unit 和 feature-gated descriptor；
- 先覆盖 plain Linear，沿 K 向量化并展开多个输出行；若本阶段做 fused consumer 原型，其字节解释必须继续严格兼容现有 `cpu_identity`；
- binding time 选择 `M=1` GEMV、小 M skinny 或 compatible scalar driver；
- 本阶段不改变 packed bytes 的解释方式；plain、`cpu_identity` 和未来 interleaved recipe 不能混用；
- 覆盖 N/K tail、padded stride 和 alias rejection；
- 检查 Release 反汇编确认目标指令与无意 spill。

退出条件：Decode canonical geomean 稳定提升，无 correctness/zero-allocation 回归。

### G2：真实 packed B + Prefill blocked GEMM

- descriptor-owned exact recipe；
- backend-owned pack service；
- B panel packing 与 blocked driver；
- direct-A zero-workspace 版本先落地；
- benchmark hot/streaming artifact 和 packing break-even。

退出条件：packed artifact 全链路数值测试通过，Prefill canonical geomean 稳定提升，启动成本和内存放大可接受。

### G3：binding-dependent workspace

- 扩展 specialization/workspace 合同；
- 增加 A panel packing 或其他 shape-dependent scratch；
- execution 聚合 specialization 后的 workspace；
- 保持 Execute 无分配。

退出条件：workspace size/alignment/lifetime 被端到端验证，收益覆盖额外 copy 成本。

### G4：fused Linear consumers

- QkvLinear/GateUpLinear 复用同一 packed GEMM engine；
- 以一个 combined N traversal 取代三次/两次独立 reference GEMM；
- epilogue 只融合已有 operator 语义，不私自加入 bias/activation；
- 比较 fused 与 unfused 的 bytes、latency 和 packing overhead。

### G5：INT8/INT4 与新 ISA

- 先冻结 quantization/dequantization、scale、group、rounding 和 accumulator 合同；
- AVX2/VNNI/AMX、NEON/DotProd/I8MM 分别注册 capability requirements；
- Decode 与 Prefill 分开选择，AMX 不覆盖小 M fallback；
- 增加模型级数值/质量门禁，而非只比较单算子误差。

### G6：runtime 线程池与 NUMA

- 仅在产品边界与 runtime thread-pool ownership 明确后实施；
- 单线程仍是 correctness/performance baseline；
- 单独测 scaling efficiency、barrier cost、oversubscription 和 NUMA placement。

## 8. 风险与依赖

| 风险 | 影响 | 缓解 |
|---|---|---|
| shape 在 kernel resolve 后才可见 | 无法选 GEMV/blocked path | binding-time internal driver；workspace/recipe 依赖 shape 时升级 specialization 合同 |
| recipe 只由 selector 决定 | 多 ISA layout 被误绑定 | descriptor-owned exact recipe + prepare-first packing request |
| 通用 MatMul 合同拖累 Linear | optimized fast path 被任意 stride/broadcast 复杂化 | adapter 分层；合法但不适合 fast path 的 layout 在 descriptor 内走 compatible scalar driver，原本无法证明安全的 layout 才返回 `Unimplemented` |
| benchmark 重复同一权重 | 高估 Decode cache locality | 同时报 hot 与 streaming artifact |
| 大 GEMM 平均值掩盖 M=1 回退 | token latency 退化 | Decode/Prefill 分组 geomean 与 per-shape gate |
| tile 参数过拟合单机 | 跨 CPU 退化 | ISA/微架构静态 profile + fallback；保留原始数据 |
| kernel 内部并行 | oversubscription、lifetime 不清 | runtime 统一线程池，kernel 只消费并行上下文 |
| 量化过早耦合 | layout/scale 合同返工 | FP32 engine 先稳定，量化作为独立 recipe/microkernel family |

## 9. 验收标准

### 9.1 架构

- 没有新增 semantic `Gemm` operator 或 graph→ISA 依赖；
- shared GEMM engine 为 backend-private，算子 adapter 继续负责自己的语义安全；
- optimized descriptor 通过 CPU capabilities 筛选，scalar/reference fallback 始终可用；
- exact recipe 从 resolved kernel 到 packed artifact 一致；
- hot path 无 registry lookup、validation、packing 或 heap allocation。

### 9.2 Correctness

- reference 与 optimized 覆盖 canonical、boundary、tail、zero-size 和合法 stride；
- alias/injectivity policy 未弱化；
- feature-disabled 环境确定选择 fallback；
- packed layout mismatch、alignment mismatch 和 metadata mismatch 明确失败。

### 9.3 Performance

- benchmark 遵守 §6 的分层、shape、cache-state 和运行协议；
- Decode/Prefill 分开报告，并保存原始 JSON；
- 关键收益有重复测量、置信区间、反汇编和硬件计数器支持；
- packing break-even、内存放大和端到端影响均有证据；
- 未达到门禁的 optimized path 不提高 registry priority。

## 10. 相关文档

- [AetherMind 当前产品 PRD](../products/aethermind_prd.md)：产品范围、单线程边界、INT8/INT4 目标。
- [架构总览](../designs/architecture/architecture_overview.md)：模块边界与执行数据流。
- [算子优化指南](../guides/operator_optimization_guide.md)：Roofline、SIMD、packing、blocking、fusion 与 benchmark 通用方法。
- [Dispatch 设计](../designs/dispatch_design.md)：kernel registry、selector 与 capability dispatch。
- [InferenceSession / Generate 前置闭环计划](01-inference-session-generate-readiness.md)：真实 Prefill/Decode vertical slice 与端到端门禁。
