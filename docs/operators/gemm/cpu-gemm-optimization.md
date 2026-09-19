# CPU GEMM 优化方案

- **状态**: In Progress
- **版本**: 2.1
- **日期**: 2026-09-19
- **文档定位**: 本算子的优化原理、合同、工作包与状态。**机器级实测数值、噪声 floor 与 Roofline 百分比不在本文详述**，权威位置见[实验记录与验证报告索引](README.md)
- **产品边界**: [AetherMind 当前产品 PRD](../../products/aethermind_prd.md)
- **架构基线**: [架构总览](../../designs/architecture/architecture_overview.md)
- **优化方法基线**: [算子开发与优化工作流附录 C](../../guides/operator-development-workflow.md#附录-c优化方法)
- **工作流规范**: [算子开发与优化工作流](../../guides/operator-development-workflow.md)
- **Change Profile**: Optimized / Packing/Layout / Threading（G6 起）
- **关联代码**: `src/backend/cpu/kernels/gemm/`、`src/backend/cpu/cpu_backend.cpp`、`src/backend/cpu/kernels/cpu_weight_prepacker.cpp`
- **关联测试**: `tests/unit/backend/cpu/kernels/`、`tests/benchmark/cpu_kernels/`
- **关联 ADR**: 无（G2 exact recipe 合同落地时新建）
- **关联模块**: backend / execution / model / benchmark

## 1. 结论与范围

AetherMind 不应新增 semantic `Gemm` operator。当前 GEMM 是 CPU backend 的内部计算 primitive，语义入口是 `MatMul`、`Linear`、`QkvLinear` 和 `GateUpLinear`。推荐建设一个共享的 CPU GEMM engine，由各算子 adapter 完成语义、layout、alias 和 packed-weight 校验，再把 compute-ready problem 交给 GEMM driver。

实施优先级不是“先做通用大矩阵 SGEMM”。顺序权威是 §7 的工作包依赖图：先建立可信 benchmark 与 Roofline 基线（G0），再由 portable scalar optimized（G1S）分离 loop/layout/register reuse 的收益，之后才量化 SIMD 的额外贡献（G1V）、真实 packing（G2）与 Prefill blocking（G3）。两条不随路线漂移的原则：多线程只通过 runtime 统一线程池引入，kernel 内不得使用 OpenMP 或私有线程；任何百分比级收益结论必须先有**同一台机器**的 baseline。

第一条完整 production 优化路径建议为 **portable FP32 scalar optimized → x86-64 FP32 AVX2+FMA → immutable packed weight**，保持单线程和运行时零分配。scalar optimized 用来分离数据访问、loop structure、unroll 和 register blocking 的收益；它不能取代 double-accumulation reference oracle。在 exact recipe 链路完成前，可以先用 plain/`cpu_identity` weight 实现直接访问的 scalar/AVX2 Decode baseline，但它们只是风险收敛里程碑，不是最终 packing 方案。AVX-512、AMX、AArch64、INT8/INT4 和多线程均是后续独立里程碑，不能写成当前已实现能力。

### 1.1 包含

- `Linear` / fused Linear 的 GEMV、skinny GEMM 和 blocked GEMM；
- 通用 `MatMul` 对共享 GEMM engine 的复用边界；
- 独立于 reference 与 SIMD 的 portable scalar optimized driver；
- SIMD microkernel、cache blocking、packing、prefetch 和 dispatch；
- binding-time shape specialization 与 workspace 演进；
- correctness、microbenchmark、production-path benchmark、packing benchmark 和最终端到端验证设计。

### 1.2 不包含

- 修改 Graph IR 或新增 `OpType::kGemm`；
- 在本提案中直接实现 kernel、量化格式或线程池；
- 把第三方 BLAS 引入为生产依赖；
- GPU GEMM、MoE、continuous batching 或分布式执行；
- 在没有测量证据时承诺具体 tile size、prefetch distance 或性能倍数。

## 2. 已验证的当前状态

| 能力 | 当前实现 | 对优化的影响 |
|---|---|---|
| GEMM primitive | `RunGemmF32Reference`，三重循环，double accumulator，支持二维 stride | correctness baseline，不是性能基线 |
| scalar optimized | `RunGemmF32ScalarOptimized` 已落地，但只覆盖 `M=1` 且 lhs K / output N 单位 stride 的两种 RHS 连续布局，其余布局委托 reference；仅在 `AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE=ON` 时经 candidate descriptor 进入 production 路径 | small-M 之外的 loop/register reuse 收益尚未取得 production 证据；不能取代 double-accumulation reference oracle |
| MatMul | FP32 reference；支持 batch broadcast、`transpose_rhs` 和任意已验证 stride | 通用性高，首轮优化不应受其最宽 layout 合同约束 |
| Linear | FP32 plain-weight reference，当前是 Linear 私有的 double 累加循环，**尚未复用共享 GEMM primitive**；leading dimensions 在 binding 期 flatten 为 `row_count` | LLM unfused path 可作为 production adapter 样板，但需先接到共享 engine 上 |
| QkvLinear / GateUpLinear | FP32 packed-only reference；当前 packed payload 是 `cpu_identity` | 已打通 opaque artifact 与 execution binding，但没有真实 tile packing |
| kernel resolve | `CpuBackend::PrepareKernel` 按 selector、CPU feature 和 priority 选 descriptor | resolve 时尚无 concrete shape/layout，不能按 `M/N/K` 选择算法 |
| binding specialization | `KernelParamsBuilder` 可看到固定的 pointer/shape/stride/dtype | 可在 cold path 选择 GEMV/skinny/blocked driver，并把函数指针写入 prepared params |
| workspace | `ResolvedKernel.workspace_requirement` 在 binding 前规划 | 不能表达依赖 concrete `M/N/K` 的 transient A/B packing scratch |
| packing recipe | `CpuWeightPrepacker::RecipeFor(selector)` 返回全局 `cpu_identity` | 不足以表达 AVX2/AVX-512/AMX 或 tile/version 不同的 layout |
| CPU dispatch | 已有 AVX2/FMA/AVX-512/VNNI/AMX、NEON/DotProd/I8MM/SVE 等 capability model | feature gate 基础可复用；当前仅 RMSNorm 有 AVX2+FMA optimized descriptor |
| threading | 当前产品边界为单请求、单线程，现有 kernel 也保持单线程 | 首轮优化保持单线程；并行化属于 runtime 级后续工作 |
| benchmark | Google Benchmark；G0 已提供 Linear prepared-path、binding-cost 与 `cpu_identity` packing 基线 | 当前只有 reference descriptor；机器级基线**按机器各自成立、互不替代**，逐机采集状态与数值见 [GEMM 实验记录与验证报告索引](README.md)（§6.7 同机要求；未采集的机器须各自重采） |

### 2.1 当前主要瓶颈

1. reference loop 的 `row → col → k` 每个输出点重新流过 K，缺少 scalar optimized 的 layout specialization、multi-accumulator 和 register blocking，无法复用 A/B 数据；
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
- scalar/SIMD optimized FP32 默认使用 FP32 accumulation；SIMD 路径可使用 FMA。两者均按绝对误差、相对误差和规模相关容差验收。
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
        ├── portable scalar optimized driver
        └── correctness reference fallback
                │
                ▼
        scalar register block / ISA microkernel
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

### 4.3 Scalar optimized 路径

scalar optimized 必须是独立实现，不能修改或覆盖 `RunGemmF32Reference`。建议使用 backend-private 的 `RunGemmF32ScalarOptimized` 或等价 driver。

优先验证以下机制：

- 针对 N-contiguous 与 K-contiguous RHS 分别选择循环和访问顺序；
- 使用 pointer bumping/strength reduction 减少地址计算；
- 使用 2–8 个独立 accumulator 打断单一 dependency chain；
- 对 K loop 做小规模 unroll，并检查 code size 与 instruction-cache 影响；
- 为 `M=1`、small-M 和 generic M 建立不同 driver；
- 使用 `1×4`、`2×4`、`4×4` 等 scalar register block 候选，具体大小由 benchmark 决定；
- 不适配 fast path 的合法 stride/layout 走 reference-compatible fallback，不缩窄 operator 合同。

scalar optimized 不使用 intrinsic，但**允许编译器 auto-vectorization**：不建设 strict non-SIMD 归因变体，也不做 loop/layout 与 vectorization 的贡献拆分（strict 目标曾实现并验证，2026-09-18 按简化决策撤销，过程与理由见 [G1S scalar 实验日志](benchmarks/g1s-scalar-log.md)）。scalar 候选、未来 SIMD 候选与 reference 使用同一 shape/layout 测试矩阵与正确性合同。

### 4.4 ISA microkernel

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

### 4.5 Packed weight

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

### 4.6 Workspace

首个实现包使用 direct A + prepacked B，保持 zero-workspace。Prefill 需要 A panel packing 时，再引入 binding-dependent workspace：

```text
Resolved ISA kernel
    → binding specialization sees M/N/K/layout
    → computes params + WorkspaceRequirement
    → PreparedExecutionBindings aggregates workspace layout
    → Execute binds preplanned slice
```

不能在 kernel 内临时 `malloc`，也不能用一个按最大模型 shape 永久膨胀的全局 scratch 规避合同设计。

### 4.7 Threading 与 NUMA

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
- performance baseline：当前 reference production descriptor；
- scalar candidate：scalar optimized（允许 auto-vectorization；不建设 strict 归因变体）；
- ISA candidate：AVX2/AVX-512/量化 descriptor；
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

#### 6.8.1 噪声 floor 是机器属性，必须按机器按组量化

上述 5% 回退阈值**不是**一个可以直接套用的常数：已采集机器的实测表明，同一 benchmark 组的噪声 floor 在不同机器之间可以相差一个数量级，且最差组可以完全不同。量化方法已上移为机器无关的通用规范，见 [算子开发工作流附录 C.2.4 噪声 floor、repetitions 与最小可信 delta](../../guides/operator-development-workflow.md#附录-c优化方法)。本提案只保留门禁要求：

- 每台目标机、每个 benchmark 组必须先量化自身的噪声 floor，只有 floor 明显低于 5% 的组才可作为自动门禁；判读 candidate 时使用**该机器该组的噪声 floor**作为最小可信 delta，而不是固定 5%；
- 必须区分**进程内方差**与**跨进程系统偏移**。当不确定性几乎全部来自后者时，提高 `--benchmark_repetitions` **无效**，有效手段是重复多个独立进程并比较每进程 median 的分布，或把 baseline/candidate 调度进同一进程内交替执行；
- 采集脚本必须能证明阈值本身有效：把**同一份实现**的两轮 A/B 互比，若被判出 REGRESS/IMPROVE，则该阈值在该组上不可用作自动门禁；
- 虚拟化环境可能把非对称拓扑伪造成对称 CPU，使 `taskset` 退化为咨询性绑定；需要核类别或 SMT 控制的门禁结论只在裸机采集；
- 逐机逐组的 floor 数值、被误判的具体 case 与可用组清单属于机器级证据，见 [GEMM 实验记录与验证报告索引](README.md)，本文不重复。

## 7. 实施路线与证据门禁

G0、G1S、G1V 与 G2–G6 是带依赖关系的工作包，不是要求机械串行执行的产品阶段。G0 是所有性能工作的前置门禁；G1S 先建立独立 scalar optimized 证据，G1V 再量化 SIMD 的额外贡献；G2 可以在 scalar 接口边界冻结后并行推进 exact recipe；G3、G5 和 G6 只有在 benchmark 或产品需求提供明确证据时才进入实施。任何工作包未达到退出条件时，不得仅凭 microbenchmark 提高 production descriptor priority。

```text
G0 合同与证据基线
 ├──> G1S portable scalar optimized
 │       └──> G1V Decode direct-weight AVX2
 └──> G2 exact recipe 与 packed B
          ├──> G3 Prefill blocked GEMM / 可选 workspace 演进
          └──> G4 fused Linear consumers
G2/G4 + quantization contract ──> G5 量化与新 ISA
runtime thread-pool contract + 单线程证据 ──> G6 并行与 NUMA
```

### G0：合同与证据基线

**状态**：Baseline Complete on `DESKTOP-54H5MMI` and `DESKTOP-QHIHOGQ` / Production Gate Needs More Data / **Not Collected on Other Machines**

状态按机器计。G0 的 benchmark、测试与采集脚本属于仓库资产，跨机可用；但机器级 baseline 数据不可跨机复用，每台目标机必须各自完成一次采集并归档自己的 raw artifact。

目标是先建立可重复、可解释的 correctness 与性能事实，不修改 production 优先级。

- 保留 `RunGemmF32Reference` 的 double accumulation，作为 correctness oracle；
- 明确外部语义为 `C = A × B`；K blocking 所需的 accumulate 仅为 microkernel 内部状态，不扩张成公共 `alpha/beta` API；
- 新增 prepared Linear/GEMM、packing 和 cache-mode benchmark；
- 覆盖 Decode/Prefill canonical shapes、tile boundary、zero-size、合法 stride、unaligned pointer、alias/injectivity 和 overflow；
- 分离 hot artifact、streaming artifact 和 cold preparation；
- 记录 scalar Release 反汇编、Roofline 输入、硬件计数器和原始 JSON；
- 验证 params build、validation、packing 和 allocation 不混入 steady-state compute loop。

2026-09-18 已完成首次机器级基线采集（记录见下方“G0 基线采集记录”）。当前代码提供以下基础设施：

- `tests/benchmark/cpu_kernels/benchmark_cpu_gemm_microkernel.cpp` 的 backend-private direct GEMM reference 基线，分别覆盖 N-contiguous 与 K-contiguous RHS；该 benchmark 用于后续 microkernel、unroll 和 blocking 调优，不替代 production Linear benchmark；
- `tests/benchmark/cpu_kernels/benchmark_cpu_linear.cpp` 的 hot/streaming prepared Linear 与 binding-specialization 测量。每个 compute case 都经 `CpuBackend::PrepareKernel`、`KernelParamsBuilder` 和 `ResolvedKernel::fn`，且在计时外将结果与 `RunGemmF32Reference` 对照；
- `tests/benchmark/cpu_kernels/benchmark_cpu_weight_packing.cpp` 的独立 `cpu_identity` cold-packing 测量。当前没有 packed Linear descriptor，因此该数据仅表示 pack/allocate/copy 成本，不能解释为 packed Linear compute 性能；
- `tests/unit/backend/cpu/kernels/test_cpu_gemm_reference.cpp` 对 `C = A × B` 覆盖写回合同的显式测试，以及 Linear 对未声明 SIMD alignment 的合法 view 的回归测试。

采集时分别保存 hot、streaming、binding 与 packing JSON；不要将它们合并为一个几何平均值。可使用 [第 6.7 节](#67-运行协议) 的命令和 `tools/compare_benchmark_json.py` 对同一模式、同一 shape 的 baseline/candidate JSON 比较。

**逐机采集状态**（机器级数值、噪声 floor 与 Roofline 百分比不在本文详述，权威见 [GEMM 实验记录与验证报告索引](README.md)）：

| 采集机 | 环境 | run id | 证据 | 本机定性结论 |
|---|---|---|---|---|
| `DESKTOP-54H5MMI` | Core Ultra 9 285H，WSL2，GCC 14.2.0 Release，单线程 `taskset`，10 repetitions | `20260918T012602Z_5cbd378695bb_DESKTOP-54H5MMI_g0-baseline` | [实验日志](benchmarks/g0-baseline-log.md)、[验证报告](benchmarks/gemm_g0_baseline_validation_2026-09-18.md)、[Roofline 定位](benchmarks/gemm_g0_roofline_analysis_2026-09-18.md)、[配对 A/B](benchmarks/gemm_g0_paired_ab_validation_2026-09-18.md) | correctness 契约与框架不变量全通过，确认 params build 未混入 compute loop；访问顺序主导 reference；`M=1` 落在记忆侧、`M≥16` 落在计算侧且均远低于 ceiling；reference 反汇编为纯标量；顺序两轮 A/B 的 delta 超过 5% 阈值，百分比级门禁不成立 |
| `DESKTOP-QHIHOGQ` | Core i9-12900H（6P+8E），WSL2，GCC 14.2.0 `-O3`，`taskset -c 16`，10 repetitions 保留全部原始行 | `20260918T151928Z_162ab3e7583f_DESKTOP-QHIHOGQ_g0-baseline` | [实验日志](benchmarks/g0-baseline-log-desktop-qhihogq.md)、[验证报告](benchmarks/gemm_g0_baseline_validation_desktop-qhihogq_2026-09-19.md)、[Roofline 定位](benchmarks/gemm_g0_roofline_analysis_desktop-qhihogq_2026-09-19.md) | 复现访问顺序归因且更强；进程内 CV 远小于跨进程系统偏移，同实现自比被误判，因此仅 binding 与 packing 两组可适用 5% 门禁；WSL2 伪造对称拓扑使 `taskset` 仅具咨询性；scalar candidate 被自动向量化为 SSE2 4-wide，未触及本机 AVX2+FMA，G1V 仍有空间 |
| 其他目标机 | — | — | — | **Not Collected**；不得沿用上述任一台的数值作为基线或门禁参照 |

原始 artifact 位于各采集机本地 gitignored 的 `benchmark-results/operators/gemm/<run-id>/`，**不随仓库分发**，在其他机器上不可恢复、不可复核；这是 G0 尚未满足 durable artifact retention 的直接后果（见 §6.7 与 G0 退出条件）。

**工作流状态**：G0 的实现（benchmark / 测试 / 采集脚本）已完成并属于仓库资产，correctness baseline 完整，shape/cache state/packing/binding 成本可独立归因，baseline/candidate 可由脚本比较。仍未满足：production 百分比级性能门禁（两台 WSL2 环境都不支持）、bare-metal `perf`、durable artifact URL，以及**其余目标机各自的 baseline 采集** —— 必须在调整 descriptor priority 前补齐。**噪声 floor 是机器属性，不得跨机沿用**：两台的可用门禁组并不相同。

退出条件：correctness baseline 完整；benchmark 可重复；shape、cache state、packing 与 binding 成本可独立归因；baseline/candidate 可以由脚本比较。

### G1S：portable scalar optimized

**状态**：In Progress（backend-private candidate 与 opt-in production-like binding integration 已完成；production acceptance blocked by formal evidence）

目标是在不依赖显式 SIMD 的前提下，独立验证 loop/layout、address generation、unroll、multi-accumulator 和 scalar register blocking 的收益，同时建立可移植的 optimized fallback。

- 新增独立 `RunGemmF32ScalarOptimized` 或等价 backend-private driver，不修改 `RunGemmF32Reference`；
- 分别覆盖 N-contiguous 与 K-contiguous RHS，优先优化 `M=1`/small-M；
- 测试 pointer bumping、K unroll、multi-accumulator 与小型 scalar register block，所有参数由 benchmark 决定；
- binding time 根据 concrete `M/N/K/stride` 选择 scalar fast driver 或 reference-compatible fallback；
- scalar optimized source 不使用 intrinsic，但允许编译器 auto-vectorization（不建设 strict 归因变体，见 §4.3）；
- direct microkernel 与 production Linear benchmark 使用相同 shape/layout/correctness matrix；
- optimized FP32 accumulation 的误差相对 double reference 单独验收。

**当前进展**：`RunGemmF32ScalarOptimized` 已作为 backend-private 单入口 candidate 落地，并在 `AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE=ON` 时注册为 opt-in descriptor `cpu::linear_f32_scalar_candidate`；默认 OFF 时 `cpu::linear_f32_reference` 的 descriptor/name/entry 保持不变。首版覆盖的 shape/layout 范围、fast path 参数与 fallback 边界属于已实现设计，以源码与 [G1S scalar 实验日志](benchmarks/g1s-scalar-log.md) 为准，该日志同时记录 strict 归因变体的实现与撤销过程。opt-in integration 与首次 smoke 数据均不构成 production acceptance，且 smoke 数值同样不可跨机复用。

退出条件：scalar optimized 在目标 Decode canonical geomean 上有稳定收益；合法但不适合 fast path 的布局确定 fallback；reference oracle 完全不变；hot path 无新增分配（SIMD 归因要求已按 2026-09-18 简化决策撤销）。

### G1V：Decode direct-weight AVX2

目标是在 G1S 已验证的数据流和 shape 分类上引入 AVX2+FMA，并明确分离 SIMD 相对 scalar optimized 的额外收益。

- 增加独立 AVX2+FMA translation unit 和 feature-gated descriptor，不对整个 target 使用 `-march=native`；
- 先覆盖 plain Linear，沿 K 向量化并展开多个 N 输出行；能复用 G1S 的 driver/planner 合同时不建立平行体系；
- binding time 根据 concrete `M/N/K/stride` 选择 `M=1` GEMV、small-M skinny 或 compatible scalar driver；
- optimized descriptor 必须覆盖其声明的完整合法 layout 合同，不适合 SIMD 的输入在 descriptor 内 fallback；
- 若验证 fused consumer 原型，packed bytes 仍必须严格解释为现有 `cpu_identity`；
- 覆盖 N/K tail、padded stride、unaligned access 和 alias rejection；
- 同时报告 reference → scalar optimized 与 scalar optimized → AVX2 两组差值；
- 检查 Release 反汇编，确认 FMA、unroll、register pressure 和 spill 情况。

退出条件：AVX2 相对 scalar optimized 的 Decode canonical geomean 有稳定增益；任一 canonical shape 的显著回退均有 shape fallback；feature-disabled 环境确定选择 scalar optimized 或 reference；hot path 无新增分配。

### G2：exact recipe 与 packed B

目标是建立真实 immutable weight packing，而不是继续把 aligned identity copy 当作优化布局。

- `KernelDescriptor`/prepared kernel 提供其精确 `PackingRecipe`；
- packing request 从已 prepare 的 kernel 收集 exact recipe；
- backend 提供按 recipe pack 的服务，model 层不建立平行 `WeightLayout` enum，也不在 `ModelLoader` 中 prepack；
- packing 发生在 semantic graph optimization/fusion 之后，由具体 `WeightBinding` 驱动；
- 实现 N-interleaved packed-B layout、tail padding、alignment 和 compatible fallback driver；
- `PackedWeightStore` 继续按 binding + selector + exact recipe 区分 artifact；
- 单独测 packing latency、GB/s、size amplification、hot/streaming compute 和 break-even invocation count。

退出条件：exact recipe 从 kernel resolve、materialization、store 到 execution binding 全链路一致；layout/alignment/metadata mismatch 明确失败；packed numerical tests 通过；packing amortization 与内存开销可接受。

### G3：Prefill blocked GEMM 与按需 workspace 演进

目标是在真实 Prefill shapes 上建立 register/cache reuse，同时避免无证据扩张 workspace 合同。

- 基于 packed-B microkernel 联合选择 `MR/NR` 与 `MC/NC/KC`，不把 register blocking、microkernel、cache blocking 和 packing 当作互不相关的固定步骤；
- 首版使用 direct A + prepacked B，保持 zero-workspace；
- full tile、M/N/K tail 和小 M fallback 使用同一 recipe 语义；
- 只有 profiling 证明 A packing 或其他 scratch 能覆盖 copy 成本时，才扩展 binding-dependent workspace specialization；
- 若引入 workspace，由 `PreparedExecutionBindings` 聚合 size/alignment/lifetime，Execute 只绑定已规划 slice；
- 比较 compute-only、packing-inclusive 和 execution-integration 结果。

退出条件：Prefill canonical geomean 稳定提升；block/tile 参数有反汇编、cache counter 和 benchmark 支持；workspace size/alignment/lifetime 端到端验证；Execute 保持零分配。

### G4：fused Linear consumers

目标是让 fused semantic operators 复用同一 GEMM engine，同时保持 graph/operator 与 backend 边界。

- `QkvLinear` 和 `GateUpLinear` 使用同一 packed GEMM engine；
- 以 combined-N traversal 取代三次/两次独立 reference GEMM；
- adapter 继续负责 output split、shape、stride、alias 和 packed metadata 校验；
- epilogue 只实现现有 operator 已定义的语义，不增加任意 Bias/Activation/Residual 组合器；
- 分别比较 fused/unfused latency、logical bytes、streaming-weight 行为和 packing overhead。

退出条件：fused operator 全链路数值测试通过；semantic port、weight binding 和 output split 保持一致；收益不是由遗漏写回或缩窄合法 layout 获得。

### G5：量化与新 ISA

目标是在量化合同稳定后引入独立的 recipe/microkernel family，而不是把 dtype 与 ISA 写成固定线性序列。

- 先冻结 activation/weight dtype、scale、zero-point、group size、rounding、accumulator 和 output conversion 合同；
- INT8/INT4 packing recipe 显式编码 quantization metadata；
- dequant/unpack 在 microkernel 内融合，避免完整反量化 tensor；
- AVX2、AVX-VNNI、AVX-512/VNNI、AMX 与 NEON/DotProd/I8MM/SVE 分别声明 capability requirements；
- GEMV/small-M/blocked 路径独立选择，AMX 不覆盖小 M fallback；
- 除单算子误差外，增加 logits/token 或模型级质量门禁。

退出条件：quantization contract 可独立验证；reference/optimized error budget 明确；目标 ISA 与 fallback 均有数值覆盖；模型级质量、内存节省和 latency/throughput 收益同时达标。

### G6：runtime threading 与 NUMA

目标是在单线程 engine 稳定后扩展多核能力，线程和内存 placement 由 runtime 统一拥有。

- 仅在 runtime-owned persistent thread pool 的 ownership、lifetime 和线程数配置明确后实施；
- Decode `M=1` 优先按 N tiles 切分，Prefill 按 M/N output tiles 切分；
- 默认避免 K-split，除非极端 shape 的 reduction 收益有测量证据；
- 一个 output tile 只由一个 worker 写，避免 false sharing；
- 禁止 kernel 创建临时线程或进入嵌套并行区；
- NUMA/first-touch/affinity 作为多 socket 部署策略，不作为所有服务器 CPU 的无条件要求；
- 单独测 scaling efficiency、barrier cost、load balance、bandwidth saturation、oversubscription 和 local/remote NUMA traffic。

退出条件：单线程结果不回退；多线程 scaling 与同步成本可解释；无数据竞争和 false-sharing 热点；线程池与 packed-weight placement 的 ownership 清晰；不同 core/socket 配置均有原始数据。

## 8. 风险与依赖

| 风险 | 影响 | 缓解 |
|---|---|---|
| shape 在 kernel resolve 后才可见 | 无法选 GEMV/blocked path | binding-time internal driver；workspace/recipe 依赖 shape 时升级 specialization 合同 |
| “scalar source” 被编译器自动向量化 | 收益归因模糊（已接受：不做拆分） | 不再区分 strict 变体；如需归因可回溯引入（该机制曾实现并验证后撤销，过程见 [G1S 实验日志](benchmarks/g1s-scalar-log.md)） |
| recipe 只由 selector 决定 | 多 ISA layout 被误绑定 | descriptor-owned exact recipe + prepare-first packing request |
| 通用 MatMul 合同拖累 Linear | optimized fast path 被任意 stride/broadcast 复杂化 | adapter 分层；合法但不适合 fast path 的 layout 在 descriptor 内走 compatible scalar driver，原本无法证明安全的 layout 才返回 `Unimplemented` |
| benchmark 重复同一权重 | 高估 Decode cache locality | 同时报 hot 与 streaming artifact |
| 大 GEMM 平均值掩盖 M=1 回退 | token latency 退化 | Decode/Prefill 分组 geomean 与 per-shape gate |
| tile 参数过拟合单机 | 跨 CPU 退化 | ISA/微架构静态 profile + fallback；保留原始数据 |
| reference 与 scalar candidate 编译期互斥（`linear_entry.cpp:170-204` 的 `#if/#else`，一个二进制里只有一个 descriptor） | 无法在同一进程内做 reference↔candidate 交错 A/B；跨二进制比较会混入链接与代码布局差异 | 两个 Release 二进制分别采集，并在 context 中记录构建选项；若需要同进程交错 A/B，须先把二者改为并存 descriptor（不同 priority）—— 这是 G1S 采集协议的前置决策 |
| 跨进程系统偏移被当成 candidate 收益/回退 | 错误接受或否决 optimized kernel | 按 §6.8.1 与 [工作流附录 C.2.4](../../guides/operator-development-workflow.md#附录-c优化方法) 逐机逐组量化噪声 floor；分离进程内 CV 与跨进程偏移；用同实现 A/B 自比校验阈值有效性 |
| 虚拟化环境伪造拓扑，`taskset` 只是咨询性绑定 | 采集可能跨物理核/P-E core 迁移，数据不可解释 | 采集前实测逐 vCPU 吞吐与 SMT 兄弟争用以识别伪造拓扑；在 context 中记录 affinity 的咨询性质；需 P/E 或 SMT 控制的结论只在裸机采集 |
| 基线 raw artifact 仅存单机 gitignored 目录 | 换机即丢失、无法复核，跨机误用他人基线 | 上传 CI/object storage 并记录 retention URL；报告中显式标注采集机与「不可跨机复用」 |
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
- 未达到门禁的 optimized path 不提高 registry priority；
- baseline 与 candidate 在**同一台机器**采集，报告显式标注采集机身份；任何性能数值不得跨机引用或替换（§6.7、§6.5）；
- 每台目标机各自的噪声 floor 已按 §6.8.1 与 [工作流附录 C.2.4](../../guides/operator-development-workflow.md#附录-c优化方法) 逐组量化，并区分进程内方差与跨进程偏移；用于自动门禁的组其 floor 必须明显低于阈值，且经同实现 A/B 自比验证不会误判；
- raw artifact 有可复核的存放位置（本机 gitignored 目录或 durable retention URL），报告中记录 run id 与 checksum。

## 10. 相关文档

- [算子开发与优化工作流](../../guides/operator-development-workflow.md)：Change Profile 证据等级、O0–O6 门禁、Benchmark 规范与工作文件证据。
- [AetherMind 当前产品 PRD](../../products/aethermind_prd.md)：产品范围、单线程边界、INT8/INT4 目标。
- [架构总览](../../designs/architecture/architecture_overview.md)：模块边界与执行数据流。
- [算子开发与优化工作流附录 C](../../guides/operator-development-workflow.md#附录-c优化方法)：Roofline、SIMD、packing、blocking、fusion 与 benchmark 通用方法；§C.2.4 为噪声 floor 与最小可信 delta 的方法权威。
- [GEMM 实验记录与验证报告索引](README.md)：本算子全部机器级数值、噪声 floor、反汇编与 Roofline 定位的权威位置。
- [Dispatch 设计](../../designs/dispatch_design.md)：kernel registry、selector 与 capability dispatch。
- [InferenceSession / Generate 前置闭环计划](../../improvement-plan/01-inference-session-generate-readiness.md)：真实 Prefill/Decode vertical slice 与端到端门禁。

## 11. 变更记录

| 日期 | 版本 | 变更 | 原因 | 证据/PR |
|---|---|---|---|---|
| 2026-09-19 | 2.1 | 核对代码后修正 §2 三处过期事实：scalar optimized 已落地（仅覆盖 `M=1`、编译期 opt-in）；默认 Linear descriptor 走 Linear 私有的 double 累加循环、**未复用共享 GEMM primitive**；candidate 与 reference 编译期互斥，因此 reference↔candidate 对比必须构建两个二进制（新增 §8 风险行） | 提案的"当前状态"与仓库事实不符会直接误导 G1S/G1V 的采集协议 | `src/backend/cpu/kernels/linear/linear_f32_reference.cpp:20-32`、`linear_entry.cpp:170-204` |
| 2026-09-19 | 2.0 | 按文档拓扑拆分：§6.8.1 的逐机噪声 floor 表与判读改为原则+链接，方法权威上移指南 §2.4.2；§7 两段 G0 机器级采集记录与 G1S 实施记录改为逐机状态表+报告链接；补模板要求的 Change Profile / 关联代码 / 关联测试 / 关联 ADR / 本文档定位字段 | 工作流要求专项提案只保留工作包状态、当前结论与验证报告链接，不粘贴原始 benchmark 数据；同时消除与 `docs/operators/gemm/` 的重复事实 | `70b1f8d8` |
| 2026-09-18/19 | 1.9 | 采集机 `DESKTOP-QHIHOGQ` G0 基线与逐机门禁归属 | 第二台机器独立采集，验证噪声 floor 的机器属性 | [G0 验证报告（QHIHOGQ）](benchmarks/gemm_g0_baseline_validation_desktop-qhihogq_2026-09-19.md) |
