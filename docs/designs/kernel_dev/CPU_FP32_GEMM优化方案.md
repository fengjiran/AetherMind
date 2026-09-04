# CPU FP32 GEMM 优化方案

版本：v0.1
状态：设计方案，尚未实现 optimized kernel
适用范围：AetherMind Phase 1 CPU FP32 GEMM / MatMul
更新时间：2026-09-04

本文定义 CPU FP32 GEMM 从 reference primitive 演进到 SIMD、cache blocking、packing 和多线程实现的工程方案。本文中的性能判断均为机制分析；micro-tile、cache block、dispatch 阈值和实际加速比必须由 benchmark 决定，不能作为已测量结论引用。

---

## 1. 目标与非目标

### 1.1 目标

- 保留现有 double-accumulation reference GEMM 作为 correctness oracle。
- 建立单线程 AVX2+FMA optimized GEMM，并对不适配的 layout 安全回退 reference。
- 分别优化 N-contiguous 和 K-contiguous 两类主要访问模式。
- 在 benchmark 证明必要后加入 cache blocking 和 panel packing。
- 保持 Execute hot path 零堆分配、无 shape/layout 重验证、无 registry 查询。
- 为后续 AVX-512、AArch64 NEON 和 runtime-owned persistent thread pool 保留稳定边界。

### 1.2 非目标

第一阶段不包含：

- 完整 BLAS `alpha` / `beta` API；
- bias、activation 或其他 operator fusion；
- FP16、BF16、FP8、INT8 或 INT4；
- RHS weight prepack；MatMul 的 RHS 是 activation，不应伪装成 model weight；
- kernel 内临时创建 `std::thread`；
- split-K reduction；
- 全局 `-ffast-math`；
- 为 GEMM 新增 Graph IR、`OpType::kGemm` 或 `OperatorSchema`。

---

## 2. 当前基线与性能模型

当前 `RunGemmF32Reference` 使用 `M -> N -> K` 三重循环和 double accumulation：

```text
for m
  for n
    double sum = 0
    for k
      sum += double(A[m,k]) * double(B[k,n])
    C[m,n] = float(sum)
```

该实现结构简单、累加顺序固定，适合作为数值 oracle，但没有 SIMD、register blocking、cache blocking、packing 或线程并行。

GEMM 的理论工作量为：

$$
\operatorname{FLOPs} \approx 2MNK
$$

不同 shape 的性能性质不同：

- Decode / GEMV-like（典型 `M = 1`）：每个 RHS 元素通常只参与约一次乘加，算术强度接近 `0.5 FLOP/byte`，容易受 memory bandwidth 限制。
- Prefill / 大 GEMM：通过 register/cache blocking 可以跨多个输出元素复用 A/B，更可能接近 compute bound。
- packing 只有在复用收益超过 packing 成本时才有价值；小矩阵或单次 `M = 1` 调用不应默认 packing。

以上是静态机制分析，不构成性能数据。

---

## 3. 架构边界

### 3.1 MatMul 与 GEMM

```text
Registered MatMul kernel
    -> binding-time dtype/shape/broadcast/stride/alias validation
    -> normalize transpose_rhs into effective RHS strides
    -> normalize batch broadcast strides
    -> freeze selected GemmF32Kernel in prepared params
    -> for each logical batch invoke GEMM

Backend-internal GEMM primitive
    -> compute one C[M,N] = A[M,K] * B[K,N]
```

MatMul 负责 operator 语义和 physical binding；GEMM 只消费已经验证的二维 compute-ready 参数。GEMM 不注册到 `KernelRegistry`，不读取 `TensorView`、`OpParams` 或 attrs。

### 3.2 Reference 与 optimized 必须并存

- `RunGemmF32Reference` 保持单线程、double accumulation 和固定 K 顺序。
- optimized kernel 使用 FP32 vector accumulation、FMA、多 accumulator 和不同 reduction order。
- optimized kernel 不覆盖或改写 reference 实现。
- 所有 optimized 路径必须能够与同一个 reference oracle 对比。

### 3.3 Hot/cold path

```text
PrepareKernel（plan cold path）
    -> 根据 selector 和 CpuCapabilities 选择 MatMul descriptor

PrepareExecutionBindings（binding cold path）
    -> 完整验证 concrete TensorViews
    -> 根据 shape/layout 选择 GemmF32Kernel
    -> 构造 trivially destructible prepared params

Execute（hot path）
    -> batch offset 解码
    -> 间接调用已冻结的 GemmF32Kernel
```

hot path 不重新判断 dtype、rank、broadcast 或 ISA eligibility。

---

## 4. 内部接口设计

现有 `GemmF32Args` 继续作为所有 FP32 实现共享的数据合同：

```cpp
struct GemmF32Args {
    const float* lhs{};
    const float* rhs{};
    float* output{};

    int64_t m{};
    int64_t n{};
    int64_t k{};

    int64_t lhs_m_stride{};
    int64_t lhs_k_stride{};
    int64_t rhs_k_stride{};
    int64_t rhs_n_stride{};
    int64_t output_m_stride{};
    int64_t output_n_stride{};
};
```

增加内部函数类型和 ISA 实现：

```cpp
using GemmF32Kernel = Status (*)(const GemmF32Args&) noexcept;

Status RunGemmF32Reference(const GemmF32Args&) noexcept;
Status RunGemmF32Avx2NVectorized(const GemmF32Args&) noexcept;
Status RunGemmF32Avx2KVectorized(const GemmF32Args&) noexcept;
```

`MatMulF32KernelArgs` 增加：

```cpp
GemmF32Kernel gemm_kernel = &RunGemmF32Reference;
```

该函数指针由 params builder 在 binding-time 冻结。类型必须继续满足：

```cpp
static_assert(std::is_trivially_destructible_v<MatMulF32KernelArgs>);
static_assert(sizeof(MatMulF32KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(MatMulF32KernelArgs) <= alignof(std::max_align_t));
```

若 batch driver 不再只代表 reference，建议将 `RunMatMulF32Reference` 重命名为内部的 `RunPreparedMatMulF32`；这是 backend-internal rename，不改变 public API。

---

## 5. 第一阶段：无 packing AVX2+FMA

第一阶段只使用原始 A/B/C storage，不需要 workspace，是最小可验证优化闭环。

### 5.1 N-vectorized outer-product 路径

适用条件：

```text
rhs_n_stride == 1
output_n_stride == 1
```

典型布局是 A `[M,K]`、B `[K,N]`、C `[M,N]` 的 row-major MatMul。沿 N 维加载 B vector，将 A scalar broadcast 后执行 FMA：

```text
for each MR x NR output tile
  keep C tile in vector registers
  for k
    load contiguous B[k, n:n+NR]
    for each row in MR
      broadcast A[m+row, k]
      C[row, :] = FMA(A_scalar, B_vector, C[row, :])
  store C tile once
```

首个候选 micro-tile：

```text
MR = 6
NR = 16
```

AVX2 有 16 个 YMM registers；`6 x 16` 需要 12 个 accumulator registers，仍给 B vectors、A broadcast 和临时值留下空间。该尺寸只是起始候选，必须与 `MR={4,6,8}`、`NR={8,16,24}` 做 benchmark sweep。

### 5.2 Tail

- 完整 `MR x NR` 走 AVX2+FMA。
- N tail 先处理可用的 8-wide vector，剩余 `< 8` 元素走 scalar。
- M tail 可以使用 `MR=1..5` 模板实例，首版也可回退 scalar row。
- 不允许越界 load 后依赖结果 mask 丢弃；AVX2 访问不能跨越未映射页。
- 不要求调用方提供 32-byte alignment；使用 unaligned load/store，后续只有 benchmark 证明收益时才增加 aligned fast path。

### 5.3 K unroll

可从 `K_UNROLL={1,2,4}` benchmark。unroll 用于隐藏 load/FMA latency，但不能让 register pressure 导致 accumulator spill。是否 unroll 及具体因子属于 benchmark 结果，不在接口中冻结。

---

## 6. 第二条路径：K-vectorized dot product

适用条件：

```text
lhs_k_stride == 1
rhs_k_stride == 1
```

典型布局是：

```text
A physical: [M,K] row-major
B physical: [N,K] row-major
transpose_rhs = true
```

每个输出元素成为两个连续 K vectors 的 dot product：

```text
for m
  for n
    C[m,n] = DotProductAvx2(A[m,:], B[n,:])
```

该路径特别适合不值得 pack RHS 的 `M=1` decode workload。实现可以复用项目现有 AVX2 dot-product 的多 accumulator 和 horizontal reduction 思路，但 GEMM 仍应拥有独立的内部接口，不将 MatMul 语义泄漏到 common dot-product API。

### 6.1 Binding-time 路径选择

初始规则：

```text
if rhs_n_stride == 1 && output_n_stride == 1:
    candidate = N-vectorized
else if lhs_k_stride == 1 && rhs_k_stride == 1:
    candidate = K-vectorized
else:
    candidate = reference
```

当两种路径同时适用时，最终规则必须根据 shape benchmark 决定。可以使用 `M`、`N`、`K` 和估算 FLOPs，但不要只用总 FLOPs：`M=1` 与方阵即使 FLOPs 相近，cache reuse 和并行策略也不同。

---

## 7. ISA 编译、注册与 fallback

### 7.1 Translation unit

建议文件：

```text
src/backend/cpu/kernels/gemm/
├── gemm_internal.h
├── gemm_f32_reference.cpp
├── gemm_f32_avx2.cpp
├── gemm_f32_avx512.cpp      # 后续
└── gemm_f32_neon.cpp        # 后续
```

只给 ISA-specific translation unit 添加编译选项：

```cmake
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/kernels/gemm/gemm_f32_avx2.cpp
    PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
)
```

不得给整个 `AetherMind` target 添加 `-mavx2` 或 `-mfma`。

### 7.2 MatMul descriptors

建议 priority：

| descriptor | CPU requirements | priority |
|---|---|---:|
| FP32 reference | none | 10 |
| FP32 AVX2+FMA | AVX2 + FMA | 20 |
| FP32 NEON | NEON | 20 |
| FP32 AVX-512 | AVX512F | 30，暂定 |

priority 只决定 eligible descriptor 的初步顺序，不代表所有 shape 上更宽 ISA 一定更快。AVX-512 descriptor 后续仍应对小 shape 选择 AVX2 或 reference，以规避 frequency downclock 和启动开销。

### 7.3 必须保证 layout fallback

当前 backend 先选中最高优先级 descriptor，params builder 失败后不会尝试次优 descriptor。因此 AVX2 params builder 不能因为 stride/layout 不适配而返回 `Unimplemented`。

正确行为：

```cpp
Status BuildMatMulF32Avx2Args(...) {
    AM_RETURN_IF_ERROR(ValidateAndBuildCommonArgs(...));

    if (SupportsNVectorized(args)) {
        args.gemm_kernel = &RunGemmF32Avx2NVectorized;
    } else if (SupportsKVectorized(args)) {
        args.gemm_kernel = &RunGemmF32Avx2KVectorized;
    } else {
        args.gemm_kernel = &RunGemmF32Reference;
    }

    // placement-new prepared params
}
```

AVX2 descriptor 对 reference 已接受的合法 layout 必须仍能构造 prepared params；layout 不适配时内部回退 reference。

---

## 8. 第二阶段：cache blocking

register blocking 稳定后再增加 cache blocking：

| 参数 | 作用 |
|---|---|
| `MR x NR` | register tile |
| `KC` | K panel，主要受 L1/L2 和 packing buffer 限制 |
| `MC` | A/C 的 M panel |
| `NC` | B/C 的 N panel |

推荐循环顺序：

```text
for jc in N step NC
  for pc in K step KC
    prepare or pack B[pc:pc+KC, jc:jc+NC]

    for ic in M step MC
      prepare or pack A[ic:ic+MC, pc:pc+KC]

      for jr in NC step NR
        for ir in MC step MR
          microkernel(MR, NR, KC)
```

初始 benchmark 搜索范围：

```text
MR = {4, 6, 8}
NR = {8, 16, 24}
KC = {64, 128, 256}
MC = {48, 72, 96, 144}
NC = {64, 128, 256}
```

最终值不能仅由 cache size 公式决定，还受 associativity、TLB、prefetcher、load/store ports、AVX frequency 和真实 shape 分布影响。

---

## 9. Packing 设计

### 9.1 优先 pack B

B packing 优先于 A packing：

- B panel 可跨多个 M rows 复用；
- arbitrary/transposed stride 被转换为连续 `KC x NR`；
- N tail 可 zero-pad；
- microkernel 不再执行复杂 RHS address generation。

packed B layout：

```text
for each N micro-panel
  for k in KC
    store NR contiguous B values
```

即每个 K 对应连续 NR 个元素，直接服务 N-vectorized microkernel。

### 9.2 A packing

当 `lhs_k_stride == 1` 且 A row-major 时，首版可以不 pack A。只有 profiling 证明 A stride、TLB/cache miss 或 address-generation 已成为瓶颈时再加入 A packing。

### 9.3 Packing 成本

必须同时维护两类 benchmark：

- end-to-end：包含 packing，是对外性能结论；
- diagnostic：分离 packing 和 compute，只用于定位成本。

不能用 compute-only 结果声称整体 MatMul 已获得同等加速。

---

## 10. Workspace

packing 禁止在 Execute hot path 使用 `std::vector`、`new` 或其他 heap allocation。

当前 `KernelContext` 已提供 `WorkspaceBinding`，但 `PrepareKernel` 不掌握 concrete M/N/K，且 `KernelDescriptor` 当前没有直接携带 shape-dependent workspace builder。因此 packing 需要先选择明确的 workspace 策略。

### 10.1 推荐：固定 panel workspace

首版使用由编译期 block 参数决定的固定上限：

```text
B panel bytes = KC * NR * sizeof(float)
```

例如候选 `KC=128, NR=16` 对应 8 KiB。大矩阵按 panel 循环复用同一 slice，workspace 大小不依赖完整 M/N/K。

落地前需要给 descriptor/backend prepared result 增加静态 `WorkspaceRequirement` 来源，使 plan-time 可以冻结该固定需求。不得在 params builder 中临时改变已经规划的 workspace requirement。

### 10.2 暂不推荐：shape-dependent workspace specialization

如果未来必须按 M/N/K 动态决定 packing buffer，需要扩展当前 plan/binding workspace 合同。该改动涉及 execution/runtime 边界，不应与第一版 AVX2 GEMM 合并。

### 10.3 K panel 累加

GEMM 对外仍是 `C = A * B`，不暴露 BLAS `beta`。内部 microkernel 使用：

```text
pc == 0: accumulator 从 0 开始
pc > 0:  accumulator 从已有 C tile 加载
```

可以使用内部 `accumulate` flag，但不扩张 operator 或 GEMM 对外语义。

---

## 11. 多线程

当前仓库没有供生产 kernel 使用的持久线程池或 OpenMP 集成。单线程 SIMD/packing 稳定前不实现多线程。

禁止每次 GEMM 临时创建线程。后续多线程由 runtime-owned persistent thread pool 提供，并避免与更高层 executor 发生 nested parallelism。

### 11.1 切分策略

| workload | 推荐切分 |
|---|---|
| Decode，`M ~= 1` | 沿 N/NC blocks |
| Prefill，M 较大 | `(M block, N block)` 二维 output tiles |
| batch_count 较大 | 优先 batch，再切 M/N |

每个 task 独占不相交 output tile，避免锁、原子操作和 false sharing。

第一版不使用 split-K：它需要额外 reduction workspace 和同步，也会引入新的数值顺序与确定性问题。

### 11.2 并行阈值

初始工作量估计：

```text
work = 2 * batch_count * M * N * K
```

但 dispatch 还必须考虑 shape；不能仅根据 FLOPs 判断。最终 thread count 和阈值必须由不同 core count、decode/prefill shape 的 benchmark 决定。

---

## 12. AVX-512 与 AArch64

### 12.1 AVX-512

候选 `NR=32`、`MR=6` 或 `8`，利用 32 个 ZMM registers 和 masked tail。必须测量 AVX-512 frequency downclock；小矩阵可能继续选择 AVX2。

### 12.2 AArch64 NEON

候选 `NR=8` 或 `16`、`MR=4..8`。共享 `GemmF32Args`、blocking/packing layout 和 correctness tests，但 NEON 与 AVX intrinsics 保持独立 translation unit，不使用大量条件宏拼成一个 microkernel。

---

## 13. Benchmark 设计

建议新增：

```text
tests/benchmark/cpu_kernels/benchmark_cpu_gemm_f32.cpp
```

### 13.1 对比实现

- reference；
- AVX2 N-vectorized；
- AVX2 K-vectorized；
- packed AVX2；
- 完整 MatMul prepared-entry 路径；
- 可选 vendor BLAS 只作为外部基线，不进入核心依赖。

### 13.2 Shape 矩阵

Decode / GEMV-like：

```text
M = 1
K,N = {128, 512, 4096, 11008, 32000}
```

Prefill：

```text
M = {16, 128, 512, 2048}
K,N = {128, 512, 4096, 11008}
```

Attention-like：

```text
K = {64, 128, 256}
M,N = representative sequence lengths
batch/head dimensions > 1
```

还必须覆盖：

- `transpose_rhs = false/true`；
- contiguous 与 strided layout；
- M/N/K tails；
- broadcast RHS；
- 非 2 的幂和质数尺寸；
- 小矩阵和高度 rectangular 矩阵。

### 13.3 指标

主要指标：

```text
GFLOP/s = 2 * M * N * K * batch_count / elapsed_seconds
```

同时记录：

- end-to-end latency；
- packing latency；
- compute-only latency；
- packed bytes；
- single-thread / multi-thread；
- CPU feature set；
- 最终选择的 kernel name/path；
- correctness 的 max absolute / relative error。

benchmark 必须在计时循环外完成 kernel resolution 和 prepared params 构造；如果测量完整 binding 成本，应作为独立 benchmark 命名。

---

## 14. Correctness 与安全门禁

每个 optimized implementation 必须覆盖：

- 随机 M/N/K；
- 所有 SIMD tail；
- `transpose_rhs`；
- padded/strided layouts；
- batch broadcast；
- `M/N/K/batch == 0`；
- output injectivity；
- input/output partial alias 拒绝；
- address/offset overflow；
- NaN/Inf 和大/小 FP32 值；
- 重复执行确定性；
- guard-zone 或等价的越界检测；
- ASAN/UBSAN；
- 多线程落地后的 TSAN。

optimized kernel 允许 FP32/FMA accumulation 和不同 reduction order，因此与 double reference 使用明确的 `atol/rtol`，不要求 bitwise equal。误差阈值必须结合 K、输入分布和端到端 logits/token regression 校准。

---

## 15. 分批实施与验收门

### Batch 1：AVX2 N-vectorized

范围：

- `gemm_f32_avx2.cpp`；
- `MR x NR` register microkernel；
- M/N/K tail；
- ISA-specific CMake flags；
- direct GEMM correctness tests；
- microbenchmark。

验收：

- reference 全部测试保持通过；
- randomized/tail correctness 通过；
- 无越界、无 heap allocation；
- 至少获得可复现 benchmark 数据，不预设必须达到的加速比。

### Batch 2：AVX2 K-vectorized

范围：

- K-contiguous dot-product path；
- `transpose_rhs=true` 和 `M=1` workload；
- 与 N-vectorized 的 shape dispatch benchmark。

验收：

- 两种 layout 路径均有独立 correctness tests；
- dispatch 阈值由保存的 benchmark 数据支持。

### Batch 3：MatMul binding-time dispatch

范围：

- `GemmF32Kernel`；
- `MatMulF32KernelArgs::gemm_kernel`；
- AVX2 MatMul descriptor；
- 不适配 layout 回退 reference；
- production prepared-entry benchmark。

验收：

- hot path 不查询 registry、不重新验证 shape/layout；
- AVX2 descriptor 对所有 reference-legal layout 均成功或明确返回真正的合同错误；
- CPU feature policy 禁用 AVX2/FMA 时解析 reference。

### Batch 4：cache blocking 与 B packing

前置条件：Batch 1-3 benchmark 已证明内核计算而非其他层是主要瓶颈。

范围：

- MC/NC/KC blocking；
- fixed panel workspace；
- B packing；
- packing-inclusive benchmark。

验收：

- Execute 零 heap allocation；
- workspace 生命周期与大小验证完整；
- 对小矩阵 packing 不产生无条件回归；
- 结论使用包含 packing 的端到端结果。

### Batch 5：persistent multi-threading

前置条件：runtime thread-pool contract 已单独设计并批准。

范围：

- batch/M/N output-tile parallelism；
- small-workload single-thread threshold；
- false-sharing 和 NUMA 检查；
- TSAN 和多 core benchmark。

---

## 16. 推荐实施顺序

```text
AVX2 N-vectorized
    -> AVX2 K-vectorized
    -> binding-time function selection and safe fallback
    -> benchmark-driven MR/NR/dispatch tuning
    -> fixed-workspace B packing and cache blocking
    -> persistent thread pool
    -> AVX-512 / NEON
```

最优先的是 Batch 1-3。它们不要求 workspace 和线程架构扩张，可以建立完整的 SIMD correctness、dispatch 和 benchmark 闭环。packing、多线程和更高级 ISA 只有在前一阶段数据证明收益后再进入实现。
