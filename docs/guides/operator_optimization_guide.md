---
title: 算子优化指南
type: topic
category: 算子开发
updated: 2026-05-31
tags:
  - operator-optimization
  - cpu
  - gpu
  - performance
  - roofline
  - simd
  - profiling
confidence: high
---

# 概述

## 适用范围与平台定位

本指南以 **CPU（x86 / ARM）平台** 为主要推演脉络，代码示例和底层优化章节（如 SIMD、NUMA、Cache Blocking）均围绕 CPU 微架构展开。

但以下核心哲学与方法论 **完全等效适用于 GPU（CUDA / Triton）平台**：

- 算子契约定义（输入输出语义、数值精度、shape 约束）；
- 内存层级分析与 Roofline 建模（FLOPs / Bytes / Arithmetic Intensity）；
- Profiling 驱动的瓶颈验证流程；
- 算法级优化（算子融合、减少 pass 数、消除冗余计算）；
- 多线程 / 多 stream 切分策略；
- 回归测试与停止准则。

GPU 开发者可将本指南视为"同一套方法论在 CPU 上的具体展开"，并将 SIMD / Cache / NUMA 等概念映射到对应的 GPU 概念（Warp / Shared Memory / SM / HBM）。

---

## 总体原则

算子优化不是简单地“加多线程”或“套 SIMD 指令”，而是一个由 **算子契约定义、正确性验证、理论建模、真实性能剖析、分层优化、回归验证** 构成的闭环工程。

一个成熟的算子优化流程应满足以下原则：

1. **先定义契约，再写代码**
   不明确输入输出语义、数值精度、内存布局、shape 范围和目标硬件，就无法判断优化是否正确。

2. **先保证正确，再追求极限性能**
   快但不正确的算子没有工程价值。正确性体系必须覆盖边界 shape、随机数据、特殊数值、误差容忍度和端到端影响。

3. **先建立性能模型，再选择优化方向**
   需要通过 FLOPs、Bytes、算术强度、Roofline、Cache 层级和硬件带宽建立性能假设。

4. **用 Profiling 验证假设，而不是凭经验猜测瓶颈**
   算子看似 memory-bound，实际可能受 shuffle、reduction、特殊函数、TLB、分支、寄存器溢出或线程同步限制。

5. **优先做高层收益，再做底层微调**
   算子融合、减少 pass 数、减少中间 tensor、权重预打包通常比局部指令优化收益更大。

6. **优化必须面向目标 workload**
   大 batch、small batch、prefill、decode、单请求、多请求对应完全不同的瓶颈。脱离 workload 的优化没有稳定价值。

7. **优化结果必须可复现、可回归、可维护**
   每一个优化版本都应能回答：为什么快、快了多少、在哪些 shape 上快、是否有回退路径、是否破坏数值稳定性。

---

## 推荐的工业级优化流程

建议将算子优化流程组织为九个阶段：

```text
1. 算子契约与目标场景定义
2. Reference、Baseline 与验证体系
3. 理论模型：FLOPs / Bytes / Roofline / 上限估算
4. Profiling 验证真实瓶颈
5. 算法级优化与算子融合
6. 内存层级优化
7. SIMD / 指令级 / Micro-kernel 优化
8. 多线程 / NUMA / 系统集成优化
9. 回归测试、性能对比与停止准则
```

这不是机械线性流程。实际工程中，第 3、4、5、6、7、8、9 步通常会反复迭代。

---

# 第 1 步：定义算子契约与目标场景

## 1.1 为什么要先定义契约

算子优化的第一步不是写 baseline，而是明确算子 contract。因为任何优化本质上都在改变执行方式，如果 contract 不清楚，就无法判断：

- 哪些数值行为可以变化；
- 哪些 layout 可以假设；
- 是否允许 in-place；
- 是否允许 fast-math；
- 是否支持非连续输入；
- 是否支持任意 stride；
- 是否需要 bitwise equal；
- 是否需要跨 ISA 路径结果一致。

例如 RMSNorm 可以有多种实现差异：

- sum of squares 使用 float 累加还是 double 累加；
- 是否允许 `_mm_rsqrt_ps` 近似；
- 是否支持 `hidden_size` 非 16 对齐；
- 是否允许输入输出指针相同；
- NaN/Inf 是否需要严格传播；
- epsilon 放在 `mean(x^2) + eps` 内还是其他位置。

这些都必须在优化前定义清楚。

---

## 1.2 算子契约清单

设计每个算子时，建议明确以下内容。

### 1.2.1 输入输出语义

| 项目 | 需要明确的问题 |
|---|---|
| 输入 tensor | 数量、shape、dtype、layout、stride |
| 输出 tensor | shape、dtype、layout、是否预分配 |
| shape 约束 | 是否固定维度，是否支持动态 shape |
| 内存连续性 | 是否要求 contiguous |
| 对齐要求 | 是否要求 32B / 64B alignment |
| aliasing | 输入输出是否允许重叠 |
| in-place | 是否支持原地计算 |
| 空 tensor | 是否允许 size = 0 |
| batch 维度 | batch 维是否参与并行切分 |
| 特殊维度 | hidden size、head dim、seq len、tile size 等 |

### 1.2.2 数值语义

| 项目 | 需要明确的问题 |
|---|---|
| 精度要求 | bitwise equal / tolerance equal / top-k equal |
| 累加精度 | FP32 累加、FP64 reference、BF16/FP16 输入 |
| NaN/Inf | 是否严格传播 |
| denormal | 是否 flush-to-zero |
| rounding | 是否依赖默认舍入模式 |
| fast-math | 是否允许近似 sqrt/div/exp |
| epsilon | 位置、dtype、默认值 |
| 溢出处理 | exp、sum、square 是否需要稳定策略 |
| 端到端影响 | logits/token 是否必须保持一致 |

### 1.2.3 性能目标

| 项目 | 说明 |
|---|---|
| latency 优先 | 单请求、decode、小 batch |
| throughput 优先 | 多请求、prefill、大 batch |
| 单核目标 | 单线程 kernel 极限 |
| 多核目标 | socket 内扩展能力 |
| NUMA 目标 | 跨 socket 是否允许 |
| 内存分配 | 是否要求 steady-state zero allocation |
| 启动开销 | 小算子是否要求低 call overhead |
| ISA 路径 | scalar / AVX2 / AVX-512 / AMX / NEON / SVE |
| fallback | 不支持高级 ISA 时是否有退化路径 |

### 1.2.4 目标硬件与执行平台

| 项目 | 说明 |
|---|---|
| 目标平台 | CPU / GPU / NPU / 异构混合 |
| 目标微架构 | x86 (Skylake/Zen/Alder Lake) / ARM (Neon/SVE) / CUDA SM |
| 内存层级假设 | 共享内存 / HBM / LPDDR / 片上 SRAM |
| 异构策略 | 是否允许 offload 到 GPU/NPU；CPU fallback 路径 |
| 跨平台一致性 | 不同平台输出是否要求 bitwise / tolerance 一致 |
| 运行时环境 | OS (Linux/WSL2/Windows)、编译器、驱动版本约束 |
| 虚拟化影响 | 容器 / WSL2 / 云实例对性能或可用 ISA 的限制 |

---

## 1.3 面向大模型推理的 workload 定义

对大模型推理引擎，算子优化必须区分 **prefill** 和 **decode**。

### 1.3.1 Prefill 阶段

特点：

- 输入 token 数较多；
- GEMM 通常较大；
- 算子吞吐更重要；
- Attention 的矩阵规模较大；
- 数据复用潜力更高；
- 多线程和 blocking 收益明显。

典型优化方向：

- GEMM packing；
- AMX/AVX-512 micro-kernel；
- 多线程矩阵分块；
- Attention blocking；
- KV cache 批量写入；
- 融合 epilogue。

### 1.3.2 Decode 阶段

特点：

- 每步通常只生成一个 token；
- batch 可能很小；
- GEMM 退化为 GEMV 或 skinny GEMM；
- kernel launch/call overhead 更敏感；
- KV cache 读取成为重要瓶颈；
- 过度多线程可能不划算。

典型优化方向：

- small-batch GEMV 专用路径；
- 权重预打包；
- KV cache layout 优化；
- 减少临时内存；
- 低开销 dispatch；
- 避免每个小算子启动线程；
- 适度融合 RMSNorm、RoPE、Elementwise。

---

# 第 2 步：建立 Reference、Baseline 与验证体系

## 2.1 Reference 与 Baseline 的区别

建议区分两个概念：

1. **Reference Implementation**
   - 用于正确性对齐；
   - 逻辑尽量直接；
   - 可以使用更高精度；
   - 不追求性能；
   - 可以用 double 累加；
   - 可以用标准库函数；
   - 可以写得更清晰。

2. **Baseline Implementation**
   - 用于性能对比；
   - 通常是朴素 C++ 标量版；
   - 不使用复杂优化；
   - 作为后续优化收益的基准。

例如 RMSNorm：

```cpp
// Reference: correctness first.
void rmsnorm_ref(const float* x,
                 const float* gamma,
                 float* y,
                 int n,
                 float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[i]);
        sum_sq += v * v;
    }

    const double mean = sum_sq / static_cast<double>(n);
    const double scale = 1.0 / std::sqrt(mean + static_cast<double>(eps));

    for (int i = 0; i < n; ++i) {
        y[i] = static_cast<float>(
            static_cast<double>(x[i]) *
            scale *
            static_cast<double>(gamma[i]));
    }
}
```

Baseline 则可以使用 float 累加，并保持结构简单。

---

## 2.2 正确性测试体系

### 2.2.1 固定测试用例

每个算子至少应覆盖：

- 最小 shape；
- 常见 shape；
- 大 shape；
- 非 2 的幂 shape；
- 质数维度；
- SIMD 宽度整除；
- SIMD 宽度不整除；
- cache 边界附近 shape；
- batch = 1；
- batch > 1；
- hidden size = 4096 / 8192 / 11008 等 LLM 常见尺寸。

### 2.2.2 随机测试

建议覆盖：

- uniform random；
- normal random；
- small magnitude；
- large magnitude；
- mixed sign；
- all positive；
- all negative；
- sparse-like 数据；
- repeated value；
- extreme value。

### 2.2.3 特殊数值测试

根据算子 contract 决定是否覆盖：

- `0.0`
- `-0.0`
- `NaN`
- `+Inf`
- `-Inf`
- denormal/subnormal
- extremely large value
- extremely small value

对于 fast-math 路径，必须明确这些值是否需要严格处理。

---

## 2.3 误差评价指标

不同算子适合不同误差指标。

| 算子类型 | 推荐指标 |
|---|---|
| Elementwise | max abs diff、max relative diff |
| Reduction | max abs diff、relative diff、double reference 对比 |
| RMSNorm/LayerNorm | max diff、mean diff、RMS error |
| Softmax | max diff、sum-to-one、argmax 一致性 |
| GEMM | max diff、relative diff、误差分布、必要时 ULP |
| Attention | output diff、attention score 稳定性 |
| LLM logits | max diff、top-k 一致性、top-1 一致性 |
| Greedy decode | token 序列一致性 |

### 2.3.1 Max Absolute Difference

```text
max_abs_diff = max(abs(y_opt[i] - y_ref[i]))
```

适合检查绝对误差，但当结果幅值很大或很小时不够充分。

### 2.3.2 Relative Difference

```text
rel_diff = abs(y_opt[i] - y_ref[i]) / max(abs(y_ref[i]), eps)
```

适合幅值变化较大的结果，但 reference 接近 0 时需要保护。

### 2.3.3 Cosine Similarity

适合向量输出整体方向比较，但不能替代逐元素误差检查。

### 2.3.4 Top-k 一致性

LLM 推理中，单个算子误差最终影响 logits。对于 greedy decoding，top-1 是否一致非常重要。对于采样路径，top-k 分布稳定性更重要。

---

## 2.4 Benchmark 基础规范

性能测试必须可复现。建议统一规范：

1. 固定 CPU governor；
2. 固定线程数；
3. 固定绑核策略；
4. 固定 NUMA 节点；
5. warmup 后再计时；
6. 重复运行多轮；
7. 记录 p50 / p90 / p99；
8. 区分 cold cache 和 warm cache；
9. 记录编译器版本；
10. 记录编译参数；
11. 记录 ISA 路径；
12. 记录输入 shape；
13. 禁止 benchmark 内部频繁 malloc/free；
14. 防止编译器把计算优化掉；
15. 使用稳定计时器。

### 2.4.1 推荐输出格式

```text
op=RMSNorm
dtype=fp32
shape=[1,4096]
isa=avx2
threads=1
latency_p50_us=...
latency_p90_us=...
latency_p99_us=...
bandwidth_GBps=...
GFLOPS=...
speedup_vs_baseline=...
max_abs_diff=...
max_rel_diff=...
```

---

# 第 3 步：理论模型与性能假设

## 3.1 计算 FLOPs

FLOPs 是算子执行的浮点运算数量。它用于估算理论计算压力。

> **FLOP 统计口径**：业界通常将乘加（Multiply-Accumulate）算作 **2 个 FLOPs**（1 次乘 + 1 次加），即便硬件上是由一条 FMA 指令完成的。统一口径对 Roofline 建模和跨团队对比至关重要。

### 示例：RMSNorm

对长度为 `N` 的向量：

1. 计算平方和：
   - `x[i] * x[i]`：N 次乘法；
   - 累加：N - 1 次加法。

2. 计算均值和倒数平方根：
   - 除法：1 次；
   - 加 epsilon：1 次；
   - sqrt 或 rsqrt：1 次。

3. 输出：
   - `x[i] * scale * gamma[i]`：约 2N 次乘法。

粗略估算：

```text
FLOPs ≈ 4N + 常数
```

实际是否将 sqrt/div 计为 1 FLOP，取决于统计口径。工程中更重要的是保持口径一致。

---

## 3.2 估算 Bytes

Bytes 是性能建模中更容易出错的部分。不能只按 tensor 逻辑大小估算，还要考虑 cache 行为。

以 FP32 RMSNorm 为例：

- 读 `x`：4N bytes；
- 读 `gamma`：4N bytes；
- 写 `y`：4N bytes；
- 如果两遍扫描时 `x` 未命中 cache，第二遍再次读 `x`：额外 4N bytes；
- 如果写分配产生 write allocate，可能产生额外读流量；
- 如果使用 non-temporal store，流量又不同。

因此实际 Bytes 可能是：

```text
理想缓存命中：约 12N bytes
两遍 DRAM 读取：约 16N bytes 或更高
```

结论：

> 算术强度中的 Bytes 应尽量按实际内存层级流量估算，而不是只按张量逻辑大小估算。必要时通过 PMU 计数器验证 DRAM/L2/L1 traffic。

---

## 3.3 算术强度

算术强度定义为：

```text
Arithmetic Intensity = FLOPs / Bytes
```

含义：

- AI 低：单位数据搬运产生的计算少，大概率 memory-bound；
- AI 高：单位数据搬运产生的计算多，具备 compute-bound 的可能。

但 AI 只是第一层判断。实际瓶颈还可能来自：

- cache miss；
- TLB miss；
- load/store port；
- shuffle；
- horizontal reduction；
- branch；
- front-end decode；
- register spilling；
- synchronization；
- NUMA remote access。

---

## 3.4 Roofline 模型

Roofline 用于估算算子性能上限：

```text
Attainable Performance = min(Peak Compute, Arithmetic Intensity × Memory Bandwidth)
```

其中：

- Peak Compute：硬件峰值计算能力；
- Memory Bandwidth：实测内存带宽；
- Arithmetic Intensity：算术强度。

### 3.4.1 机器平衡点

```text
Machine Balance Point = Peak Compute / Memory Bandwidth
```

如果：

```text
AI < Machine Balance Point
```

算子大概率受数据搬运限制。

如果：

```text
AI > Machine Balance Point
```

算子才具备计算受限的可能。

### 3.4.2 注意事项

不要直接使用厂商标称峰值作为唯一依据。建议测量：

1. 实际 STREAM 内存带宽；
2. 单核内存带宽；
3. 多核内存带宽；
4. 单 socket 带宽；
5. 跨 socket 带宽；
6. AVX2/AVX-512 实际 FMA 峰值；
7. AVX-512 降频影响；
8. AMX 实际 tile 计算吞吐。

---

## 3.5 层级 Roofline

CPU 算子经常不是单纯 DRAM-bound，而是受不同层级限制：

- Register bandwidth；
- L1 bandwidth；
- L2 bandwidth；
- L3 bandwidth；
- DRAM bandwidth；
- Load/store port throughput；
- Front-end throughput；
- Special function throughput。

因此建议建立多层性能假设：

```text
该算子是否受 DRAM 限制？
该算子是否受 L2/L1 带宽限制？
该算子是否受 load/store port 限制？
该算子是否受 reduction dependency chain 限制？
该算子是否受 shuffle 指令限制？
该算子是否受线程同步限制？
```

---

# 第 4 步：Profiling 验证真实瓶颈

## 4.1 Profiling 应贯穿全过程

Profiling 不应该只放在最后。推荐流程是：

```text
建立 baseline -> profiling -> 提出瓶颈假设 -> 做单一优化 -> 再 profiling -> 验证收益 -> 继续迭代
```

每一次优化都应该能解释：

1. 优化前瓶颈是什么；
2. 优化手段针对什么瓶颈；
3. 优化后指标是否改善；
4. 是否引入了新瓶颈；
5. 对不同 shape 是否稳定有效。

---

## 4.2 CPU Profiling 工具

常用工具：

| 工具 | 用途 |
|---|---|
| Linux perf | 低开销采样、PMU 计数器 |
| Intel VTune Profiler | pipeline、memory、threading 分析 |
| AMD uProf | AMD 平台性能分析 |
| likwid | 拓扑、带宽、硬件计数器 |
| pmu-tools | Intel PMU 辅助分析 |
| numactl | NUMA 绑定和验证 |
| taskset | 绑核 |

---

## 4.3 CPU 关键指标

| 指标 | 含义 | 可能问题 |
|---|---|---|
| cycles | 总周期 | 总体耗时 |
| instructions | 指令数 | 算法复杂度、展开过度 |
| IPC | 每周期指令数 | pipeline 利用率 |
| L1 miss | L1 未命中 | 数据局部性差 |
| L2 miss | L2 未命中 | blocking 不合理 |
| LLC miss | 末级缓存未命中 | DRAM 压力大 |
| DRAM bandwidth | 内存带宽 | memory-bound |
| branch miss | 分支预测失败 | 主循环分支多 |
| TLB miss | 地址转换开销 | 跨页访问、数据过大 |
| vectorization ratio | 向量化比例 | SIMD 未生效 |
| stalled cycles | 停顿周期 | 等待内存或依赖 |
| port pressure | 执行端口压力 | load/store/FMA/shuffle 瓶颈 |
| register spilling | 寄存器溢出 | 变量过多、展开过度 |
| remote NUMA access | 远端内存访问 | 绑核/first-touch 问题 |

---

## 4.4 GPU Profiling 工具

如涉及 GPU，可使用：

| 工具 | 用途 |
|---|---|
| Nsight Systems | 系统级 timeline、kernel launch、数据传输 |
| Nsight Compute | 单 kernel 指标、occupancy、memory、warp stall |
| nvprof | 老版本 CUDA profiling |
| rocprof | AMD GPU profiling |

GPU 常看指标：

- occupancy；
- global memory throughput；
- shared memory bank conflict；
- warp stall reason；
- tensor core utilization；
- register usage；
- achieved occupancy；
- memory coalescing；
- L2 hit rate；
- instruction throughput。

---

## 4.5 Profiling 结论模板

每次 profiling 后建议输出类似结论：

```text
算子：RMSNorm fp32
shape：[1,4096]
线程：1
ISA：AVX2

观察：
1. DRAM bandwidth 仅达到 STREAM 的 35%。
2. IPC 偏低。
3. 主循环存在 horizontal reduction dependency。
4. 无明显 branch miss。
5. 无 register spilling。

判断：
当前瓶颈不是纯 DRAM 带宽，而是 SIMD reduction 和 load/store pipeline 利用不足。

下一步：
1. 使用多累加器降低依赖链。
2. 使用 AVX2 向量化主循环。
3. 单独优化 tail 路径。
```

---

# 第 5 步：算法级优化与算子融合

## 5.1 优先级原则

底层指令优化前，应先判断是否能从算法和算子图层面降低复杂度。

优先级通常是：

```text
减少计算/访存次数 > 减少中间 tensor > 提高数据复用 > SIMD/指令级优化 > 多线程扩展
```

---

## 5.2 算子融合

算子融合的核心目标是减少中间结果的写回和再读取。

### 5.2.1 GEMM Epilogue Fusion

典型融合：

```text
GEMM + Bias
GEMM + Bias + ReLU
GEMM + Bias + GELU
GEMM + Scale
GEMM + Residual Add
```

更准确的表达是：

> 将 Bias、Activation 等后处理逻辑融合到 GEMM epilogue 中，在 tile 或 micro-kernel 写回阶段完成，避免额外生成、读取和写回完整中间矩阵。

### 5.2.2 LLM 常见融合方向

| 算子组合 | 融合目的 |
|---|---|
| RMSNorm + residual | 减少额外读写 |
| Linear + Bias | epilogue 融合 |
| Linear + Activation | MLP 后处理融合 |
| RoPE + Q/K projection 后处理 | 减少中间 Q/K 写回 |
| Attention score + mask + softmax | 减少 score matrix 中间存储 |
| Softmax + value matmul | streaming attention 思路 |
| Dequant + MatMul | 权重量化路径减少中间反量化 tensor |

---

## 5.3 减少 Pass 数

Memory-bound 算子通常对 pass 数非常敏感。

例如 RMSNorm 朴素实现通常两遍：

1. 第一遍计算平方和；
2. 第二遍归一化并乘 gamma。

如果 hidden size 很小，第一遍读取的 `x` 可能仍在 cache 中；如果 hidden size 很大，则第二遍可能重新访问 DRAM。

优化方向：

- 分块计算；
- cache-friendly 两遍扫描；
- 与前后算子融合；
- 尽量避免额外中间 buffer。

---

## 5.4 消除冗余计算

示例：

### Softmax

稳定 softmax 通常为：

```text
max_val = max(x)
sum = Σ exp(x_i - max_val)
y_i = exp(x_i - max_val) / sum
```

可以优化：

- `max_val` 只计算一次；
- `1 / sum` 提前算好；
- 输出阶段用乘法代替除法；
- 对小维度使用专用路径；
- 对大维度使用分块 reduction。

### RMSNorm

```text
scale = 1 / sqrt(mean(x^2) + eps)
y_i = x_i * scale * gamma_i
```

可以优化：

- `scale` 只计算一次；
- 输出阶段只做乘法；
- `gamma` 连续读取；
- 尽量避免重复除法。

---

## 5.5 Fast Math 与近似计算

Fast Math 可以提升性能，但必须受 contract 控制。

### 5.5.1 可选手段

- reciprocal approximation；
- reciprocal sqrt approximation；
- Newton-Raphson refinement；
- polynomial approximation；
- table lookup；
- vector exp approximation；
- fused multiply-add 重写。

### 5.5.2 风险

- 改变 NaN/Inf 传播；
- 改变 signed zero 行为；
- 改变 rounding；
- 改变 denormal 处理；
- 累积误差；
- 影响 logits top-k；
- 影响生成 token 一致性。

### 5.5.3 建议策略

1. fast-math 不应默认全局开启；
2. 每个算子显式声明是否允许近似；
3. 对 LLM 输出敏感路径做端到端验证；
4. 使用 `_mm_rsqrt_ps` 后通常需要 Newton-Raphson refinement；
5. 允许按精度档位提供不同 kernel。

---

# 第 6 步：内存层级优化

## 6.1 为什么内存优化最关键

现代 CPU/GPU 的计算单元通常远快于内存系统。大量算子性能不是受计算数量限制，而是受数据搬运限制。

常见 memory-bound 算子：

- Add；
- Mul；
- RMSNorm；
- LayerNorm；
- Softmax；
- RoPE；
- Copy；
- Cast；
- Quant/Dequant；
- 小 batch GEMV；
- KV cache 读写。

这些算子的关键不是“多算得快”，而是“少搬、连续搬、复用后再丢”。

---

## 6.2 连续访问

优先确保主循环访问是连续的：

```cpp
for (int i = 0; i < n; ++i) {
    y[i] = x[i] * scale;
}
```

连续访问优点：

- cache line 利用率高；
- 硬件预取器容易识别；
- SIMD load/store 简单；
- TLB 压力较低。

跨步访问问题：

```cpp
for (int i = 0; i < n; ++i) {
    y[i] = x[i * stride] * scale;
}
```

可能导致：

- cache line 利用率低；
- 硬件预取失效；
- TLB miss 增加；
- SIMD gather 成本高。

---

## 6.3 Memory Layout

不同算子对 layout 要求不同。

### 6.3.1 GEMM Packing

GEMM 中的 packing 是将矩阵 A/B 重排成适合 micro-kernel 和 cache blocking 的 panel layout。

目标：

- 连续访问；
- 提高 cache 复用；
- 减少 TLB miss；
- 匹配 SIMD/AMX tile；
- 简化 micro-kernel 内部寻址。

### 6.3.2 Conv Layout

CNN 中常见：

- NCHW；
- NHWC；
- NCHWc；
- blocked layout。

不同 layout 会影响 vectorization 和 cache locality。

### 6.3.3 LLM KV Cache Layout

大模型推理中，KV cache layout 直接影响 decode 性能。

常见布局考虑：

```text
[layer][seq][head][dim]
[layer][head][seq][dim]
[block][head][token][dim]
paged/block KV cache
```

选择 layout 时要考虑：

- decode 阶段按 head 读取是否连续；
- attention 访问历史 token 是否连续；
- head_dim 是否 SIMD 对齐；
- 写入新 token 是否低开销；
- 是否支持 paged attention；
- 是否减少 TLB 和 cache miss。

---

## 6.4 Cache Blocking / Tiling

Blocking 的核心思想：

> 将大问题切成能放进 cache 的小块，在数据被逐出 cache 前尽量复用。

### 6.4.1 GEMM Blocking 层级

典型层级：

```text
Register Blocking
L1 Blocking
L2 Blocking
L3 Blocking
NUMA / Socket Blocking
```

每一级解决不同问题：

| 层级 | 目标 |
|---|---|
| Register | 累加 tile 常驻寄存器 |
| L1 | 小 panel 复用 |
| L2 | A/B block 复用 |
| L3 | 多核共享数据复用 |
| NUMA | 避免远端内存访问 |

### 6.4.2 Blocking 参数选择

参数不是越大越好，要考虑：

- cache 容量；
- associativity；
- cache line 大小；
- TLB；
- prefetch；
- register 数量；
- SIMD 宽度；
-线程切分；
- 是否造成 conflict miss。

### 6.4.3 缓存冲突未命中（Cache Thrashing / Conflict Miss）

一个经典的性能陷阱是 **2 的幂次步长诅咒**：

当矩阵的行宽（如 `hidden_size`）恰好是 $2^N$（例如 4096、8192），按列跨步访问时，所有列索引的低位地址位相同，导致数据全部映射到同一个 Cache Set。这会引发极为严重的**缓存冲突未命中（Conflict Miss）**——即使总数据量远小于 Cache 容量，实际命中率却极低。

典型场景：

- GEMM 中按列访问矩阵 B（列优先存储时）；
- 跨步读取 `hidden_size=4096` 的张量；
- 多路并行时多个线程访问不同行但相同列偏移。

解决方法：

- **内存填充（Padding）**：将行宽从 $2^N$ 扩展到 $2^N + \delta$（如 4096 → 4104），打破地址对齐；
- **交错（Swizzling）**：对内存布局做位变换，使相邻逻辑行映射到不同物理 Cache Set；
- **调整 blocking 参数**：使 tile 边界不与 $2^N$ 对齐。

---

## 6.5 Alignment

对 SIMD 和 cache 都很重要。

建议：

- 大 buffer 至少 64B 对齐；
- AVX2 路径考虑 32B 对齐；
- AVX-512 路径考虑 64B 对齐；
- AMX tile 数据按 kernel 要求对齐；
- 权重预打包结果按 cache line 对齐；
- 避免跨 cache line 的非必要访问。

但也要注意：

- 现代 CPU 对 unaligned load 支持较好；
- 只要不跨 cache line/page，成本未必高；
- 不要为了对齐引入复杂分支导致收益下降。

---

## 6.6 Software Prefetch

软件预取不是通用优化项。

### 6.6.1 适合场景

- 间接访问；
- 跨步访问；
- 链表或 block list；
- paged KV cache；
- 访问地址可预测但硬件预取器识别困难；
- memory latency stall 明显。

### 6.6.2 不适合场景

- 简单连续线性访问；
- 数据很快会被硬件预取；
- 预取距离难以确定；
- cache 容量紧张；
- 预取导致 cache pollution。

建议：

> 对连续线性访问，应优先依赖硬件预取器。只有 profiling 证明存在明显 memory latency stall 时，再引入手工 prefetch。

---

## 6.7 避免 False Sharing

多线程写入相邻小数据时，可能造成 false sharing。

示例：

```cpp
struct ThreadState {
    float sum;
};
ThreadState states[num_threads];
```

多个线程写不同 `sum`，但这些 `sum` 可能在同一 cache line，导致 cache line ping-pong。

优化：

```cpp
struct alignas(64) ThreadState {
    float sum;
    char padding[60];
};
```

或者使用更合理的 reduction buffer 布局。

---

## 6.8 Non-temporal Store

对于只写一次且短期不会读取的大输出，可以考虑 non-temporal store，避免污染 cache。

适合：

- 大规模 copy；
- 大输出 tensor；
- 结果短期不会被读取。

不适合：

- 输出很快被下一个算子读取；
- 小 tensor；
- cache 复用较强；
- 会破坏后续融合机会。

在推理引擎中，如果下一个算子马上读取该输出，普通 store 往往更合适。

---

## 6.9 量化路径的内存与计算权衡

LLM 推理中，W8A8、W4A16、INT8 等量化方案已成为标配。量化算子优化的核心矛盾是：**减少内存带宽的同时，引入了额外的反量化计算开销。**

### 6.9.1 反量化开销（Dequantization Overhead）

量化权重在计算前需要反量化回 FP32/BF16。这个过程包含：

- 提取 Weight、Scale、Zero-point；
- 位移（Bit-shift）和乘法；
- Shuffle / Permute 操作（尤其是 INT4 打包时，一个字节存两个值）。

当量化粒度较细（如 per-channel 或 per-group）时，反量化可能成为新的瓶颈——算子从 **Memory-bound 变成 ALU-bound**。

优化方向：

- 在 SIMD 主循环中**边反量化边计算**，避免写回中间 FP32 buffer；
- 利用 SIMD 的 unpack / shuffle 指令加速 INT4 解包；
- 对 W4A16 等混合精度路径，保持激活值 FP16/BF16，仅反量化权重。

### 6.9.2 Block-wise 量化与 Cache Blocking 的交互

LLM 常用 Group 量化（如 `group_size=64/128`），这意味着：

- 每个 group 有独立的 Scale 和 Zero-point；
- Cache Blocking 的步长（tile size）需要与 group_size 对齐，否则跨 group 边界读取会引入额外的 Scale 查找开销；
- 如果 blocking 参数不是 group_size 的整数倍，可能导致同一 group 的 Scale 被重复加载。

建议：

> Blocking 参数（MC/NC/KC）应尽量是 `group_size` 的整数倍，使每个 tile 内的量化参数连续且可复用。

---

# 第 7 步：SIMD / 指令级 / Micro-kernel 优化

## 7.1 SIMD 优化的本质

SIMD 的目标是让一条指令处理多个元素，例如：

- AVX2：256-bit，一次处理 8 个 FP32；
- AVX-512：512-bit，一次处理 16 个 FP32；
- NEON：128-bit；
- SVE：向量长度可变；
- AMX：tile matrix compute；
- ARM SME：matrix extension。

但 SIMD 优化不只是替换成 intrinsic，还包括：

- 主循环设计；
- tail 处理；
- alignment 策略；
- 多累加器；
- horizontal reduction；
- shuffle 成本；
- 寄存器压力；
- ISA dispatch；
- scalar fallback。

---

## 7.2 主循环与 Tail 路径

向量化时通常需要：

```text
vectorized main loop
+
tail scalar loop
```

例如 `N` 不是 8 或 16 的倍数时，需要处理剩余元素。

AVX-512 可使用 mask tail：

```cpp
__mmask16 mask = (1u << tail) - 1;
```

AVX2 通常选择：

- scalar tail；
- masked load/store 模拟；
- over-read with padding；
- 专用 remainder kernel。

选择取决于 contract 是否允许 padding 和安全 over-read。

---

## 7.3 多累加器降低依赖链

Reduction 类算子中，单个累加器会形成依赖链：

```cpp
sum += x[i] * x[i];
```

优化为多个累加器：

```cpp
sum0 += ...
sum1 += ...
sum2 += ...
sum3 += ...
```

收益：

- 增加 ILP；
- 降低累加依赖；
- 提高 FMA pipeline 利用率；
- 帮助乱序执行。

但累加器太多会增加寄存器压力，可能导致 spilling。

---

## 7.4 FMA

FMA 将：

```text
a * b + c
```

融合为一条指令，通常有更高吞吐和更好精度。

常见场景：

- dot product；
- GEMM；
- polynomial approximation；
- RMSNorm sum of squares；
- vector transform。

注意：

- FMA 可能导致与非 FMA 路径结果不 bitwise equal；
- 如果 contract 要求严格一致，需要明确是否允许。

---

## 7.5 Horizontal Reduction 成本

很多算子需要向量内归约，例如：

- sum；
- max；
- sum of squares；
- dot product。

SIMD 横向归约通常需要 shuffle/add，成本不可忽略。

优化方向：

- 多 vector 累加器；
- 最后统一 horizontal reduce；
- 避免循环内频繁横向归约；
- 对固定维度写专用 reduce；
- AVX-512 利用更宽寄存器和 mask；
- 对小 N 使用 scalar 可能更快。

---

## 7.6 Shuffle 成本

某些操作看似计算简单，但需要大量 shuffle，例如：

- transpose；
- interleave/deinterleave；
- layout transform；
- RoPE；
- complex multiply；
- gather/scatter。

如果 shuffle 成本过高，可能导致算子不是 compute-bound，而是 shuffle-bound。

优化方向：

- 重新设计 layout；
- 减少跨 lane 操作；
- 批量处理；
- 用更适合的 SoA/AoS 布局；
- 针对 head_dim 固定值写专用路径。

---

## 7.7 Register Pressure 与 Spilling

循环展开和多累加器可以提高 ILP，但也会增加寄存器使用。

如果寄存器不够，编译器会 spill 到栈上，性能可能大幅下降。

检查方式：

- 查看汇编；
- 使用 compiler report；
- 使用 perf/VTune；
- 检查 stack load/store；
- 使用 Godbolt 或 objdump。

优化方向：

- 减少展开因子；
- 减少临时变量；
- 拆分循环；
- 缩小 live range；
- 调整 blocking 参数。

---

## 7.8 ISA Dispatch

工程中通常需要多条路径：

```text
Scalar fallback
SSE/NEON fallback
AVX2 kernel
AVX-512 kernel
AMX kernel
```

推荐机制：

1. 启动时检测 CPU feature；
2. 根据 dtype、shape、layout、ISA 选择 kernel；
3. 小 shape 和大 shape 可走不同 kernel；
4. 预留 fallback；
5. 保证不同路径共享测试体系。

示例选择逻辑：

```cpp
KernelFunc select_rmsnorm_kernel(CpuFeatures f, DataType dtype, int hidden_size) {
    if (dtype == DataType::kFloat32) {
        if (f.avx512f && hidden_size >= 1024) {
            return rmsnorm_fp32_avx512;
        }
        if (f.avx2) {
            return rmsnorm_fp32_avx2;
        }
        return rmsnorm_fp32_scalar;
    }
    return nullptr;
}
```

---

# 第 8 步：多线程、NUMA 与系统集成优化

## 8.1 多线程优化的前提

多线程不是越早越好。应先让单线程 kernel 达到合理效率，再扩展到多核。

原因：

- 多线程会掩盖单线程低效；
- 小算子多线程启动成本可能大于计算收益；
- 线程同步和 reduction 可能成为新瓶颈；
- 过度并行可能破坏 cache locality。

推荐顺序：

```text
单线程 correctness
单线程 baseline
单线程 profiling
单线程 SIMD/memory 优化
多线程 scaling
NUMA 优化
```

---

## 8.2 线程池

高性能推理引擎中，不应每次算子调用创建线程。

推荐：

- Executor 持有线程池；
- 算子提交任务到线程池；
- worker 长驻；
- 支持绑核；
- 支持按 NUMA 分组；
- 支持低开销 barrier；
- 避免 steady-state malloc/free。

---

## 8.3 任务切分策略

不同算子适合不同切分维度。

| 算子 | 常见切分维度 |
|---|---|
| Elementwise | contiguous range |
| RMSNorm | batch/row |
| LayerNorm | batch/row |
| Softmax | row/head |
| GEMM | M/N tile |
| GEMV | output rows or columns |
| Attention | batch/head/query block |
| KV cache copy | token/head/block |

切分原则：

1. 每个线程工作量足够大；
2. 每个线程访问尽量连续；
3. 避免多个线程写同一 cache line；
4. 避免过多同步；
5. reduction 合并成本可控。

---

## 8.4 Load Balancing

负载不均会造成长尾。

常见问题：

- shape 不能被线程数整除；
- 不同行计算量不同；
- 稀疏结构；
- 变长序列；
- attention mask 导致有效计算不同。

解决：

- 静态均匀切分；
- block cyclic；
- work stealing；
- 动态调度；
- 对小任务合并；
- 对大任务分块。

但动态调度有额外开销，应根据 workload 选择。

---

## 8.5 Reduction 并行化

Reduction 多线程优化要注意：

1. 每线程私有 partial sum；
2. partial sum cache line 对齐；
3. 最后归并；
4. 避免 atomic；
5. 对小 N 不并行；
6. 对大 N 分块；
7. 注意数值结果与单线程不同。

---

## 8.6 NUMA 优化

多路 CPU 服务器中，NUMA 影响很大。

### 8.6.1 常见问题

- 线程运行在 socket 0；
- 数据分配在 socket 1；
- 跨 socket 访问导致延迟和带宽下降；
- 多线程争抢远端内存；
- first-touch 策略不正确。

### 8.6.2 优化策略

- thread affinity；
- NUMA first-touch；
- 按 socket 分配内存；
- 按 socket 切分任务；
- 避免跨 socket 写共享数据；
- 对大模型权重考虑 socket 本地副本或分片；
- 使用 `numactl` 验证。

### 8.6.3 验证方法

```bash
numactl --hardware
numactl --cpunodebind=0 --membind=0 ./benchmark
numastat -p <pid>
```

---

## 8.7 系统集成注意事项

算子不能只在 microbenchmark 中快，还要在推理引擎中稳定。

需要检查：

1. 是否引入临时内存分配；
2. 是否破坏 tensor 生命周期；
3. 是否兼容 executor 调度；
4. 是否兼容 kernel registry；
5. 是否支持 runtime dispatch；
6. 是否影响异常处理；
7. 是否破坏可测试性；
8. 是否支持 tracing/profiling；
9. 是否对小 shape 有过高 overhead；
10. 是否对 decode 阶段不划算。

对 CPU-first LLM 推理引擎，尤其要坚持：

```text
steady-state zero allocation
single-request low latency
predictable execution
clear fallback path
```

---

# 第 9 步：回归、调优与停止准则

## 9.1 每次优化后的检查

每一次优化都必须做：

1. 正确性回归；
2. 性能 benchmark；
3. 多 shape 验证；
4. 多 ISA 验证；
5. profiling 对比；
6. 代码复杂度评估；
7. 端到端影响检查。

---

## 9.2 性能对比维度

不要只看单个耗时数字。

建议记录：

- latency p50；
- latency p90；
- latency p99；
- throughput；
- GB/s；
- GFLOPS；
- speedup；
- CPU utilization；
- memory bandwidth；
- scaling efficiency；
- tail latency；
- 端到端 token/s；
- decode latency per token；
- **MFU（Model FLOPs Utilization）**：有效算力利用率 = 实际 GFLOPS / 硬件峰值 GFLOPS，衡量整体硬件算力发挥了多少；
- **MBU（Memory Bandwidth Utilization）**：内存带宽利用率 = 实际 GB/s / STREAM 峰值 GB/s，针对 Memory-bound 算子的"终极考卷"。

---

## 9.3 停止准则

停止优化不应机械使用统一百分比。

可以认为算子达到工程可接受状态的情况：

1. 达到目标延迟；
2. 达到目标吞吐；
3. 已接近实测硬件上限；
4. profiling 显示瓶颈已落到物理带宽或指令吞吐极限；
5. 继续优化收益低于阈值；
6. 复杂度超过维护收益；
7. 对主 workload 无明显收益；
8. 端到端收益不明显。

建议写成：

> 如果算子在目标 workload 下已经接近实测硬件上限，且 profiling 显示主要瓶颈已经落在内存带宽、计算吞吐或硬件指令极限上，则可以认为该算子达到工程可接受状态。具体阈值应按算子类型、shape 和硬件平台确定，不宜机械套用统一比例。

---

## 9.4 自动化调优（Autotuning）

工业界不仅依赖人工手算 MC/NC/KC 或展开因子，还可以引入**离线自动调优（Offline Autotuning）**机制。

### 9.4.1 为什么需要 Autotuning

手动选择 blocking 参数、展开因子、线程切分策略存在以下问题：

- 不同微架构（Skylake vs Zen 4 vs Alder Lake）的最优参数差异显著；
- 同一 CPU 上，不同 shape 的最优参数也不同；
- 人工调优覆盖的 shape 组合有限，容易遗漏边界情况。

### 9.4.2 典型做法

类似 TVM、Halide 或 cuBLAS 的 heuristic 搜索：

1. **定义搜索空间**：blocking 参数范围、展开因子、线程数、ISA 路径等；
2. **定义目标函数**：latency、throughput、或加权综合指标；
3. **搜索策略**：
   - 穷举搜索（小搜索空间）；
   - 随机采样 + 贝叶斯优化（大搜索空间）；
   - 遗传算法 / 强化学习。
4. **缓存最优配置**：将搜索结果缓存为 lookup table，服务启动时或预热阶段加载。

### 9.4.3 工程建议

- Autotuning 应在**目标硬件上离线运行**，结果持久化；
- 对 LLM 推理引擎，可在**服务启动预热阶段**运行轻量级 autotuning，覆盖常用 shape；
- 保留 fallback 配置，防止 autotuning 结果异常；
- 定期回归测试，确保 autotuning 结果不因编译器/系统更新而退化。

---

# 典型算子优化策略

## RMSNorm 优化方法

### 算子公式

```text
rms = sqrt(mean(x_i^2) + eps)
y_i = x_i / rms * gamma_i
```

等价：

```text
scale = 1 / sqrt(mean(x_i^2) + eps)
y_i = x_i * scale * gamma_i
```

### 性能特征

RMSNorm 通常是 memory-bound + reduction-bound。

主要开销：

- 第一遍读 `x` 计算 sum of squares；
- 第二遍读 `x` 和 `gamma`，写 `y`；
- reduction 依赖链；
- sqrt/rsqrt；
- tail 处理；
- 对小 hidden size，函数调用和线程调度开销明显。

### 优化方向

1. SIMD 计算平方和；
2. 多累加器降低 reduction 依赖；
3. 横向归约只在循环末尾做；
4. 使用 `rsqrt` 或 `sqrt + div`，由 contract 控制；
5. 第二遍 SIMD 输出；
6. 对 hidden size 固定值做专用路径；
7. batch/row 维度并行；
8. 小 batch decode 阶段避免过度多线程；
9. 与 residual 或后续 elementwise 融合；
10. 保证 tail 正确。

### 注意事项

- FP32/BF16/FP16 输入的累加精度要明确；
- fast rsqrt 需要误差验证；
- hidden size 不整除 SIMD 宽度时 tail 不能越界；
- gamma 访问应连续；
- 输入输出 aliasing 要明确。

---

## Softmax 优化方法

### 算子特点

Softmax 通常包含：

1. max reduction；
2. exp；
3. sum reduction；
4. normalize。

它既可能 memory-bound，也可能受 exp 特殊函数吞吐限制。

### 稳定实现

```text
m = max(x)
s = Σ exp(x_i - m)
y_i = exp(x_i - m) / s
```

### 优化方向

1. max reduction SIMD 化；
2. exp 向量化或近似；
3. sum reduction 多累加器；
4. 用 `inv_sum` 替代逐元素除法；
5. 对小维度使用专用路径；
6. 对大维度分块；
7. 融合 mask；
8. 融合 scale；
9. 融合后续 dropout 或 value matmul；
10. 对 attention softmax 考虑 streaming/blocking。

### 注意事项

- 不能直接 `exp(x)`，容易溢出；
- mask 后的 `-inf` 处理要正确；
- 全 mask 行要定义行为；
- fast exp 要验证误差；
- softmax 输出 sum-to-one 只是必要条件，不是充分条件。

---

## GEMM / MatMul 优化方法

### 算子特点

GEMM 理论上通常 compute-bound，但前提是实现足够好。

朴素 GEMM 可能因为内存访问差而 memory-bound。

### 优化层级

1. loop order；
2. packing；
3. cache blocking；
4. register blocking；
5. SIMD/AMX micro-kernel；
6. 多线程 tile 切分；
7. epilogue fusion；
8. small shape 专用 kernel；
9. quantized weight path；
10. prepack weight。

### Micro-kernel 思路

一个典型 GEMM micro-kernel 会让 C 的一个小 tile 常驻寄存器：

```text
C[mr x nr] += A[mr x kc] * B[kc x nr]
```

优化目标：

- A/B 连续读取；
- C tile 寄存器累加；
- FMA 吞吐最大化；
- 减少 load/store；
- 减少地址计算；
- 控制寄存器压力。

### LLM 推理中的特殊点

在 decode 阶段，很多 Linear 实际上是：

```text
[1 x K] * [K x N]
```

这更接近 GEMV 或 skinny GEMM。

优化重点变成：

- 权重读取带宽；
- cache 复用；
- batch 合并；
- quantized weight；
- NUMA 本地性；
- 避免过高线程开销。

---

## RoPE 优化方法

### 算子特点

RoPE 对 Q/K 的相邻维度进行旋转：

```text
x_even' = x_even * cos - x_odd * sin
x_odd'  = x_even * sin + x_odd * cos
```

### 优化方向

1. sin/cos 预计算；
2. 连续读取 Q/K；
3. SIMD 处理相邻 pair；
4. 减少 shuffle；
5. 针对 head_dim 固定值优化；
6. 与 Q/K projection 后处理融合；
7. 对 decode 使用低开销路径。

### 注意事项

- layout 会显著影响 SIMD 难度；
- interleaved 和 split-half RoPE 处理方式不同；
- sin/cos 表的访问也可能造成 cache 压力。

---

## Attention / KV Cache 优化方法

### Decode 阶段瓶颈

decode attention 主要访问历史 KV cache：

```text
q: 当前 token
K/V: 历史所有 token
```

瓶颈常来自：

- KV cache 读取带宽；
- cache miss；
- TLB miss；
- layout 不连续；
- head 维度访问不友好；
- softmax 归约；
- 小矩阵计算效率低。

### 优化方向

1. KV cache layout 设计；
2. block/paged cache；
3. head_dim 对齐；
4. K/V 分开存储；
5. prefetch 历史 block；
6. streaming softmax；
7. 减少 score 中间存储；
8. 多 head 并行；
9. 多 query batch 合并；
10. 对短上下文和长上下文分路径。

---

# 工程检查清单

## 算子开发前检查清单

```text
[ ] 输入输出 shape 已定义
[ ] dtype 已定义
[ ] layout/stride 已定义
[ ] 是否要求 contiguous 已定义
[ ] 是否允许 in-place 已定义
[ ] aliasing 行为已定义
[ ] NaN/Inf 行为已定义
[ ] fast-math 策略已定义
[ ] 误差容忍度已定义
[ ] 目标 workload 已定义
[ ] 目标硬件已定义
[ ] 单线程/多线程目标已定义
[ ] benchmark shape 集合已定义
```

---

## Correctness 检查清单

```text
[ ] reference 实现已完成
[ ] baseline 实现已完成
[ ] 固定 shape 测试已覆盖
[ ] 随机测试已覆盖
[ ] 边界 shape 已覆盖
[ ] SIMD tail 已覆盖
[ ] NaN/Inf 已按 contract 测试
[ ] FP32/BF16/FP16 路径已分别测试
[ ] max abs diff 已记录
[ ] max relative diff 已记录
[ ] 端到端 logits/token 影响已评估
```

---

## Performance 检查清单

```text
[ ] benchmark 不包含 malloc/free 干扰
[ ] benchmark 有 warmup
[ ] benchmark 有多轮重复
[ ] 记录 p50/p90/p99
[ ] 固定线程数
[ ] 固定绑核策略
[ ] 固定编译参数
[ ] 记录 ISA 路径
[ ] 记录 GB/s 或 GFLOPS
[ ] 与 baseline 对比
[ ] 与理论上限对比
[ ] profiling 证明瓶颈变化
```

---

## Memory 优化检查清单

```text
[ ] 主访问路径连续
[ ] 避免不必要 stride/gather
[ ] 数据对齐合理
[ ] cache blocking 参数合理
[ ] packing 成本可摊销
[ ] 避免多余中间 tensor
[ ] 避免 false sharing
[ ] 必要时考虑 non-temporal store
[ ] prefetch 有 profiling 依据
[ ] NUMA first-touch 正确
```

---

## SIMD 优化检查清单

```text
[ ] 主循环 SIMD 化
[ ] tail 路径正确
[ ] 对齐/非对齐 load 策略明确
[ ] 多累加器降低依赖
[ ] horizontal reduction 不在主循环频繁执行
[ ] shuffle 数量可控
[ ] 无 register spilling
[ ] ISA dispatch 正确
[ ] scalar fallback 正确
[ ] 不同 ISA 路径共享测试
```

---

## 多线程检查清单

```text
[ ] 没有每次算子调用创建线程
[ ] 使用线程池
[ ] chunk size 合理
[ ] 小 shape 不过度并行
[ ] 每线程写入无 false sharing
[ ] reduction 使用线程私有 partial
[ ] barrier 开销可控
[ ] thread affinity 明确
[ ] NUMA 访问可控
[ ] scaling efficiency 已测量
```

---

# 推荐文档模板

## 单个算子优化设计文档模板

```markdown
# 算子名称优化设计

## 1. 算子语义
- 公式：
- 输入：
- 输出：
- dtype：
- layout：
- stride：
- in-place：
- aliasing：
- fast-math：

## 2. 目标场景
- prefill / decode：
- shape 集合：
- batch：
- hidden size：
- 线程数：
- ISA：
- latency/throughput 目标：

## 3. Reference 与 Baseline
- reference 实现：
- baseline 实现：
- 误差指标：
- 测试用例：

## 4. 理论性能模型
- FLOPs：
- Bytes：
- Arithmetic Intensity：
- Roofline 判断：
- 预期瓶颈：

## 5. Profiling 结果
- baseline 性能：
- PMU 指标：
- 主要瓶颈：
- 优化假设：

## 6. 优化方案
- 算法级：
- 内存级：
- SIMD 级：
- 多线程级：
- 特殊 shape：

## 7. 实现路径
- scalar：
- AVX2：
- AVX-512：
- AMX：
- fallback：
- dispatch：

## 8. 测试与验证
- correctness：
- benchmark：
- 端到端：
- 回归：

## 9. 风险与限制
- 精度风险：
- shape 限制：
- 平台限制：
- 维护成本：

## 10. 结论
- 性能提升：
- 是否达到目标：
- 后续优化点：
```

---

# 面向 C++20 推理引擎的落地建议

## Kernel Registry 与 Dispatch

建议每个算子注册多个 kernel descriptor：

```cpp
struct KernelSelector {
    DeviceType device_type;
    DataType activation_dtype;
    DataType weight_dtype;
    WeightFormat weight_format;
    IsaLevel isa;
    ExecPhase phase;
};

struct KernelDescriptor {
    OpType op_type;
    KernelSelector selector;
    KernelFunc kernel_func;
    const char* name;
    int priority;
};
```

选择逻辑：

1. 根据 op type 找候选；
2. 根据 dtype/layout/ISA/phase 过滤；
3. 根据 priority 选择；
4. shape 特化可在 kernel 内二次分发；
5. fallback 必须存在。

---

## Steady-state Zero Allocation

算子实现中应避免：

```cpp
std::vector<float> tmp(n);
```

除非 tmp buffer 由上层 workspace 预分配。

推荐：

- Executor 提供 workspace；
- 算子声明 workspace size；
- 初始化阶段分配；
- 推理稳态不分配；
- 小临时数据使用 stack 或寄存器；
- 大临时数据复用 buffer。

---

## Benchmark 与 CI 集成

建议建立三类测试：

1. **Correctness CI**
   - 每次提交运行；
   - 覆盖 scalar/AVX2/AVX-512；
   - 覆盖关键 shape。

2. **Performance Smoke Test**
   - 检查性能没有明显回退；
   - 阈值可以宽松；
   - 适合每日运行。

3. **Full Benchmark**
   - 手动或定期运行；
   - 输出详细性能报告；
   - 用于优化决策。

---

# 总结

一个工业级算子优化流程的核心不是“把代码写得复杂”，而是让每一步优化都有明确依据：

```text
定义契约 -> 建立 reference -> 建立 baseline -> 建模 -> profiling -> 提出假设 -> 单点优化 -> 回归验证 -> 继续迭代
```

优化的最终目标是：

> 让算子在目标 workload 下尽可能接近硬件真实上限，并且瓶颈清晰、行为正确、性能稳定、代码可维护。

对于大模型推理引擎而言，尤其需要避免三个误区：

1. **只看 microbenchmark，不看端到端 token latency。**
2. **只做 SIMD，不处理 memory layout、fusion 和 dispatch。**
3. **只追求单个 shape 极限，不考虑真实 workload、fallback 和长期维护。**

高质量算子优化不是一次性的技巧堆叠，而是一套可复现、可验证、可演进的工程闭环。
