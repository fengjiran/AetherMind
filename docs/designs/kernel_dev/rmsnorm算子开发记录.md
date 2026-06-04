**RMSNorm（Root Mean Square Normalization）自 LLaMA 采用后，已成为现代开源大模型的标配归一化层，在推理中频繁调用。**
## 1. 数学定义

RMSNorm 的数学公式如下：

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma $$

其中：

- $x$ 是输入的向量（通常按 Token 或按行划分，长度为隐藏层维度 $d$）
- $d$ 是隐藏层维度（`hidden_size`）
- $\epsilon$ 是防止除零的极小值（`eps`）
- $\gamma$ 是模型训练得到的可学习权重（`weight`），长度为 $d$

每个 token（即每行）独立计算：先求该行所有元素的平方和，除以维度后取平方根的倒数，再用该标量缩放每个元素，最后逐元素乘上可学习权重 $\gamma$。

---
## 2. 算子契约

### 2.1 输入输出契约

#### 2.1.1 输入

| 输入       | 语义                 | 当前约束                                      |
| -------- | ------------------ | ----------------------------------------------- |
| `input`  | 待归一化激活张量 `X`       | `float32`，rank-2，shape `[seq_len, hidden_size]` |
| `weight` | RMSNorm 权重 `gamma` | `float32`，rank-1，shape `[hidden_size]`          |

#### 2.1.2 输出

| 输出 | 语义 | 当前约束 |
|---|---|---|
| `output` | 归一化结果 `Y` | 预分配 `float32`，rank-2，shape 与 `input` 完全一致 |

#### 2.1.3 Shape 约束

- `seq_len > 0`。
- `hidden_size > 0`。
- `input.shape == [seq_len, hidden_size]`。
- `weight.shape == [hidden_size]`。
- `output.shape == input.shape`。
- `seq_len` 和 `hidden_size` 在运行时动态确定；kernel 不能依赖编译期固定 shape。
- correctness 与 benchmark 至少覆盖：`seq_len = 1` decode、`seq_len > 1` prefill、`hidden_size = 4096 / 8192 / 11008`、SIMD 宽度整除与不整除、非 2 的幂和质数 hidden size。

#### 2.1.4 Layout / stride / alignment

- 当前只支持 contiguous row-major layout。
- `input` / `output` stride 必须等价于 `[hidden_size, 1]`。
- `weight` stride 必须等价于 `[1]`。
- 不支持 arbitrary stride、gather/scatter、blocked layout 或 packed activation layout。
- 不要求调用方提供 32B / 64B 对齐地址；CPU kernel 必须能处理 unaligned load/store。
- 后续如果引入 aligned fast path，必须保留 unaligned fallback。

#### 2.1.5 Aliasing / in-place

- 允许 `output` 与 `input` 完全相同的精确 in-place 计算，但必须由测试覆盖。
- 不允许 `output` 与 `input` 部分重叠；部分 overlap 行为未定义，后续应在 Tensor alias 检查能力完善后显式拒绝。
- 不允许 `output` 与 `weight` 重叠。
- `weight` 在 kernel 执行期间只读，不得被修改。

#### 2.1.6 空 tensor

当前不支持空 tensor。`seq_len <= 0` 或 `hidden_size <= 0` 必须返回 `InvalidArgument`。

---

### 2.2 参数契约

| 参数 | 类型 | 默认值 | 约束 | 语义 |
|---|---:|---:|---|---|
| `epsilon` | `float` | `1.0e-5F` | `epsilon > 0` | 加在 `mean(X^2)` 之后、`sqrt` 之前 |

Operator 层参数为 `RmsNormOp::Params::epsilon_`。CPU kernel attrs 为 `CpuRmsNormAttrs::Epsilon`，二者必须保持 ABI 语义一致。

---

### 2.3 数值契约

#### 2.3.1 Reference 语义

Reference 实现用于 correctness oracle，应使用更高精度累加：

$$
\operatorname{sum\_sq}_s = \sum_{h=0}^{H-1}\operatorname{double}(X_{s,h}) \cdot \operatorname{double}(X_{s,h})
$$

$$
\operatorname{mean\_square}_s = \frac{\operatorname{sum\_sq}_s}{\operatorname{double}(H)}
$$

$$
\operatorname{scale}_s = \frac{1}{\sqrt{\operatorname{mean\_square}_s + \operatorname{double}(\varepsilon)}}
$$

$$
Y_{s,h} = \operatorname{float}\left(\operatorname{double}(X_{s,h}) \cdot \operatorname{scale}_s \cdot \operatorname{double}(\gamma_h)\right)
$$

#### 2.3.2 Optimized kernel 允许的差异

- Optimized kernel 不要求 bitwise equal。
- 允许使用 FP32 累加、FMA、多累加器和不同 reduction order。
- 默认不允许 approximate rsqrt。若后续引入 `_mm_rsqrt_ps` 或近似路径，必须新增独立精度档位、测试阈值和 ISA dispatch 条件。
- 默认不启用全局 fast-math；不得依赖破坏 NaN/Inf 传播或舍入语义的编译选项作为 correctness 前提。

#### 2.3.3 误差指标

correctness 以 double reference 为基准：

- 常规输入：`max_abs_diff <= 1e-5F`。
- 对幅值较大输入，同时记录 `max_rel_diff`，建议目标 `<= 1e-5F`，但最终阈值应由真实模型 logits/token 回归确认。
- benchmark 输出应记录 `max_abs_diff`、`max_rel_diff`，不能只报告 latency。

#### 2.3.4 特殊数值

- NaN / Inf 不做额外清洗，遵循 IEEE-754 默认传播行为。
- denormal/subnormal 不在 kernel 内显式设置 flush-to-zero；实际行为继承进程和平台的浮点环境。
- rounding 依赖默认舍入模式；kernel 不应修改全局 rounding mode。
- 特殊数值测试必须按该契约补齐，再决定是否收紧或放宽行为。

---

### 2.4 性能与执行契约

#### 2.4.1 目标 workload

| 场景 | 典型 shape | 优先级 | 目标 |
|---|---|---|---|
| Decode | `[1, 4096]`、`[1, 8192]` | 最高 | 低 latency，避免线程调度开销 |
| Prefill | `[16, 4096]`、`[128, 4096]`、`[128, 8192]` | 高 | row 维吞吐，稳定多核扩展 |
| 边界测试 | 非 2 的幂、质数 hidden、SIMD tail | 高 | correctness 优先，验证 tail 不越界 |

#### 2.4.2 内存与 workspace

- Steady-state zero allocation。
- `ComputeWorkspaceRequirement()` 返回空 workspace。
- kernel 不得分配临时 heap buffer。
- 主访问路径必须连续读取 `input`、连续读取 `weight`、连续写入 `output`。
- 输出通常会被下游算子继续读取，默认不使用 non-temporal store，除非 profiling 证明收益且不会破坏后续 cache locality。

#### 2.4.3 多线程

- RMSNorm 的自然并行维度是 row / `seq_len`。
- 当前不做单行 hidden 维并行 reduction。
- Decode 小 shape 不应启动多线程；当前策略应保持小 `seq_len` 单线程执行。
- 多线程阈值必须由 benchmark / profiling 驱动，不能凭经验固定后长期不回归。

#### 2.4.4 ISA 与 fallback

- CPU backend 必须提供可运行 fallback 路径；高级 ISA 路径不能成为唯一 correctness 路径。
- AVX2/FMA、AVX-512、ARM NEON/SVE 等路径必须共享同一 reference 与测试体系。
- 不同 ISA 路径只要求 tolerance equal，不要求 bitwise equal。
- ISA dispatch 必须基于 runtime capability 和 `KernelSelector`，不能在不支持目标 ISA 的机器上执行对应指令。

---

### 2.5 硬件平台与运行时环境契约

#### 2.5.1 目标平台与微架构

| 项目     | 当前约束                                        |
| ------ | ------------------------------------------------- |
| 目标平台   | **CPU-first**，仅 x86-64                            |
| 目标微架构  | Intel Alder Lake（i9-12900H）及以上，支持 AVX2 + FMA      |
| 内存层级   | DDR4/DDR5 或 LPDDR5，通过 STREAM 实测带宽作为 Roofline 建模依据 |
| 最小 ISA | SSE2 / x86-64 baseline（scalar fallback）           |
| 可选 ISA | AVX2 + FMA、AVX-512（若可用）、AMX（未来）                   |

#### 2.5.2 运行时环境

| 项目 | 当前约束 |
|---|---|
| 操作系统 | Linux native 或 WSL2 |
| 编译器 | GCC ≥ 11 或 Clang ≥ 14，支持 `-march=native` |
| 线程模型 | OpenMP 或自研线程池；禁止每次算子调用动态创建线程 |
| 虚拟化影响 | WSL2 下 CPU 指令原生执行，计算开销 < 1%；内存带宽可能受 Hyper-V 内存管理轻微影响 |
| NUMA | 当前假设单 socket，不处理跨 NUMA 节点访问 |

#### 2.5.3 异构与扩展策略

| 项目 | 当前约束 |
|---|---|
| GPU / NPU offload | 当前不支持；所有计算必须在 CPU 上完成 |
| 跨平台一致性 | 仅需保证同一 ISA 路径内 tolerance equal；不同 ISA 路径间不要求 bitwise equal |
| 后续扩展 | Phase 2 可考虑 GPU（CUDA / ROCm）或 NPU 路径；届时需重新定义跨平台误差契约 |

---

### 2.6 Operator / Kernel 边界契约

#### 2.6.1 Operator 层职责

`RmsNormOp` 负责：

- 校验 `epsilon > 0`。
- 校验输入数量、dtype、rank、shape。
- 推导输出 shape：返回与 `input` 相同的 `ShapeInfo`。
- 声明空 workspace。
- 根据 `OperatorContext.backend` 和 `KernelSelector` 解析 kernel。
- 将 `epsilon` 作为 kernel attrs 传入。

#### 2.6.2 Kernel 层职责

执行期参数绑定：

- `ExecutionPlanBuilder::Build` 产生 `RmsNormOp` 并完成 `Prepare`（kernel 解析 + epsilon attrs 缓存），但不绑定 tensor view。
- 调用方通过 `RuntimeBindingContext::SetStepTensorBinding` 注册 per-step 的输入/输出 tensor views。
- `LayerRunner::RunStep` 检查 `OpType::kRmsNorm`，从 `RuntimeBindingContext` 取出 tensor binding，构造 `CpuRmsNormParams` 并写入 `KernelContext.packed_params`。
- `CpuRmsNormParams` 的生命周期由 `RunStep` 栈帧保证，覆盖同步 `Operator::Run` 的完整调用。

`CpuRmsNormKernelEntry` 负责：

- 校验 `ctx.packed_params` 非空。
- 校验 `epsilon`（从 `ctx.attrs` 解码）有限且大于 0。
- 校验 `TensorView` / `MutableTensorView` 有效、dtype 正确、rank 正确、contiguous。
- 校验 `seq_len`、`hidden_size`、shape、data pointer 和 stride 满足 CPU kernel 的低层参数前置条件。
- 构造 `CpuRmsNormKernelArgs` 并调用 `CpuRmsNormKernel`。

`CpuRmsNormKernel` 是已验证参数上的 typed compute primitive：

- 调用方必须保证 `CpuRmsNormKernelArgs` 中的指针非空、维度为正、stride 为正、`epsilon` 有限且大于 0，并且 backing storage 覆盖所有访问元素。
- 执行数值计算并写入预分配 output。
- 不拥有输入、权重、输出内存；不延长任何指针生命周期。

---

### 2.7 验证清单

#### 2.7.1 Correctness

- [ ] 固定小 shape：`[1, 4]`、`[3, 4]`。
- [ ] Decode shape：`[1, 4096]`、`[1, 8192]`。
- [ ] Prefill shape：`[16, 4096]`、`[128, 4096]`、`[128, 8192]`。
- [ ] SIMD tail：hidden size 不是 8 / 16 / 32 的倍数。
- [ ] 非 2 的幂和质数 hidden size。
- [ ] 随机输入：uniform、normal、mixed sign、small magnitude、large magnitude。
- [ ] `output == input` 精确 in-place case。
- [ ] NaN / Inf / denormal 行为按契约记录。
- [ ] 与 double reference 比较 `max_abs_diff` 和 `max_rel_diff`。

#### 2.7.2 Benchmark

- [ ] benchmark 不在计时循环内 malloc/free。
- [ ] 记录 `seq_len`、`hidden_size`、dtype、ISA、线程数。
- [ ] 同时报告 optimized kernel 与 reference/baseline。
- [ ] 记录 latency、items/s、GB/s、GFLOPS、speedup。
- [ ] 对 decode 小 shape 单独观察线程调度开销。
- [ ] 阈值调整必须附带 before/after benchmark 数据。

---

### 9. 当前开放问题

1. 是否将 `output == input` in-place 从"允许"提升为 Operator 层显式能力，需要 Tensor alias 检查支持。
2. ~~是否需要 scalar reference kernel 与 AVX2 optimized kernel 拆分注册，避免非 AVX2 平台编译或运行风险。~~ **（已解决：scalar 与 AVX2 路径已拆分为两个独立注册入口，通过 `IsaLevel::kScalar` / `kAVX2` 区分注册，详见 RMSNorm算子契约.md 第 8 节。）**
3. BF16 / FP16 输入的累加精度、输出 dtype 和误差阈值尚未定义。
4. Approx rsqrt 是否作为独立 fast path 引入，需要端到端 logits/token 回归后决定。
5. 多线程阈值需要按目标硬件和 workload 通过 benchmark 固化，而不是写死为永久策略。
