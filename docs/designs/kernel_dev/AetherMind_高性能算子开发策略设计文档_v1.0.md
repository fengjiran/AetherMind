# AetherMind 高性能算子开发策略设计文档

版本：v1.0  
状态：设计建议稿  
适用范围：AetherMind Phase 1 CPU-first 推理引擎  
更新时间：2026-05-19  

---

## 1. 文档目标

本文档用于明确 AetherMind 在推理引擎尚未完全实现阶段，是否以及如何开展高性能算子开发。

本文档重点回答以下问题：

1. 当前阶段是否应该开始高性能算子开发；
2. 哪些算子应该立即实现；
3. 哪些算子只适合做原型验证；
4. 哪些高性能优化应暂缓；
5. 算子开发与 Tensor、Backend、Dispatch、Executor、Model Loader、KVCache、Workspace Plan 之间应如何解耦；
6. 如何建立 Reference Kernel、Correctness Test 和 Benchmark 体系；
7. 如何从当前阶段平滑过渡到后续系统性高性能算子开发阶段。

---

## 2. 总体结论

当前 AetherMind 尚未完全实现时，应该开始算子开发，但不应大规模铺开平台相关的极致高性能算子开发。

建议采用以下策略：

> 先建设算子基础设施，优先实现 Reference Kernel，并选择少量核心热点算子做性能原型验证；待 Tensor、Backend、Dispatch、Executor、Workspace、Model Loader、PackedWeight、KVCache 等核心合同基本稳定后，再系统性进入高性能算子库开发阶段。

换句话说，当前阶段应做的是：

```text
Operator 语义层
    ↓
Kernel 注册与选择机制
    ↓
Reference Kernel
    ↓
Correctness Test
    ↓
Benchmark Harness
    ↓
少量热点算子性能原型
```

而不是立即全面开发：

```text
完整 AVX/AVX512/AMX 算子库
完整量化算子库
复杂 fused kernels
复杂 attention 优化
多 backend 高性能 kernel
```

---

## 3. 当前阶段的核心判断

### 3.1 为什么不能过早全面开发高性能算子

高性能算子强依赖底层系统合同。如果以下模块尚未稳定，过早开发高性能 kernel 会导致大量返工：

- Tensor 的 shape、stride、contiguous、layout 语义；
- Buffer / DataPtr / MemoryHandle 的所有权与生命周期语义；
- Operator 语义层接口；
- Backend 与 Runtime 的职责边界；
- KernelDescriptor / KernelSelector / KernelRegistry；
- Workspace 静态规划机制；
- Model Loader 和权重预打包格式；
- KVCache 的布局、访问方式和生命周期；
- Executor 对 prefill / decode 的调度路径；
- ThreadPool、NUMA、并行策略；
- dtype、量化格式、packed layout 扩展方向。

高性能 kernel 通常会绑定如下要素：

```text
Tensor layout
weight packing format
threading policy
workspace policy
NUMA policy
cache blocking
KVCache layout
operator fusion boundary
backend dispatch key
```

如果这些抽象没有稳定，kernel 很可能变成“局部性能很好，但难以接入系统”的孤岛代码。

### 3.2 为什么不能完全不做算子

算子不是独立模块，它会反向影响引擎架构。

例如：

- Linear / GEMM 会影响权重预打包格式；
- Attention 会影响 KVCache layout；
- RMSNorm / Silu / Softmax 会验证 Tensor contiguous 语义；
- Decode 阶段 GEMV 会影响 ThreadPool 和 batch 策略；
- Workspace 使用方式会影响 Executor 的静态 plan；
- 算子融合边界会影响 Operator 语义层设计；
- RoPE 会影响 position ids、cos/sin cache、Q/K layout 和 attention 接口。

因此，当前阶段不能等待系统全部实现后再做算子，而应通过少量关键算子提前验证架构合同。

---

## 4. Phase 1 约束前提

本文档默认 AetherMind Phase 1 遵循以下约束：

```text
CPU-first
single-process
single-model
single-request
synchronous execution
token-ids in/out
Llama-family dense model
static KV cache
greedy decoding
steady-state zero allocation
```

在该约束下，算子开发应避免过早引入以下复杂能力：

- continuous batching；
- multi-request scheduling；
- speculative decoding；
- GPU backend；
- distributed inference；
- dynamic KV cache；
- 复杂动态图 shape；
- 多模型并发；
- 复杂量化格式全覆盖；
- 完整 graph compiler / fusion pass。

---

## 5. 算子开发总体原则

### 5.1 Correctness First

第一阶段算子的首要目标是正确性，而不是极限性能。

每个 Phase 1 必需算子都应优先提供 Reference Kernel，作为后续高性能 kernel 的 correctness oracle。

Reference Kernel 应具备：

- 语义清晰；
- 行为稳定；
- 便于调试；
- 便于和 PyTorch / NumPy 对齐；
- 便于构造单元测试；
- 可以支持端到端 Llama 最小推理路径。

### 5.2 Infrastructure First

当前阶段的重点不是单点 kernel 性能，而是建立统一算子执行链路：

```text
Operator API
    ↓
Operator 参数校验
    ↓
KernelDescriptor
    ↓
KernelRegistry
    ↓
KernelSelector
    ↓
KernelContext / Workspace
    ↓
Kernel Invoke
    ↓
Correctness Test
    ↓
Benchmark
```

所有新算子都应通过统一路径接入，而不是绕过框架直接调用裸函数。

### 5.3 Benchmark First

高性能算子开发必须以 benchmark 为驱动。

在进行 SIMD、blocking、packing、threading、fusion 等优化前，应先具备稳定的 benchmark 框架，至少能够度量：

- latency；
- p50 / p90 / p99；
- throughput；
- memory bandwidth；
- FLOPS；
- allocation count；
- workspace usage；
- thread scaling；
- prefill / decode 分阶段耗时；
- kernel dispatch overhead。

没有 benchmark 的优化不应进入主线。

### 5.4 Stable Contract Before Deep Optimization

深度优化应等待核心合同稳定后再进行。

以下内容属于深度优化前置合同：

- Tensor layout；
- PackedWeight 格式；
- KernelDescriptor 字段；
- WorkspacePlan 语义；
- KVCache layout；
- Threading policy；
- dtype / quantization 语义；
- Backend dispatch key；
- Error handling / Status 规范。

### 5.5 Kernel 与 Tensor 语义解耦

高性能 kernel 不应直接依赖复杂 Tensor 对象。

推荐分层：

```text
Operator 层：
- 接收 Tensor
- 做 shape / dtype / layout 校验
- 解析参数
- 选择 kernel
- 组织 workspace
- 调用 kernel

Kernel 层：
- 接收裸指针、维度、stride、workspace、thread context
- 不关心 Tensor 生命周期
- 不做复杂动态分配
- 不依赖上层语义对象
```

推荐 kernel 低层接口形态：

```cpp
Status LinearKernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    WorkspaceView workspace,
    KernelThreadContext* thread_ctx) noexcept;
```

### 5.6 Steady-State Zero Allocation

Phase 1 已明确要求 steady-state zero allocation。

因此：

- kernel 内部不应进行堆分配；
- 临时内存应来自 Executor / WorkspacePlan 预规划；
- packed weight 应在模型加载阶段或初始化阶段完成；
- cos/sin cache、mask buffer 等应提前准备；
- decode 主循环中不得出现不可控动态分配。

---

## 6. 推荐目录结构

建议算子相关目录采用如下结构：

```text
aethermind/
  operators/
    op_type.h
    operator.h
    operator_context.h
    operator_schema.h
    rmsnorm_op.h
    linear_op.h
    rope_op.h
    attention_op.h
    softmax_op.h
    mlp_op.h

  kernels/
    kernel.h
    kernel_context.h
    kernel_descriptor.h
    kernel_registry.h
    kernel_selector.h

    reference/
      rmsnorm_ref.cc
      linear_ref.cc
      rope_ref.cc
      softmax_ref.cc
      attention_ref.cc
      silu_ref.cc
      elementwise_ref.cc
      argmax_ref.cc

    cpu/
      rmsnorm_cpu.cc
      linear_cpu.cc
      rope_cpu.cc
      softmax_cpu.cc
      attention_cpu.cc

    cpu/x86/
      rmsnorm_avx2.cc
      linear_packed_avx2.cc
      linear_packed_avx512.cc

  backend/
    cpu/
      cpu_backend.h
      cpu_backend.cc

  dispatch/
    dispatch_key.h
    dispatch_table.h

  runtime/
    thread_pool.h
    workspace.h
    workspace_plan.h

bench/
  operators/
    bench_rmsnorm.cc
    bench_linear.cc
    bench_rope.cc
    bench_softmax.cc
    bench_attention.cc

  end_to_end/
    bench_llama_decode.cc
    bench_llama_prefill.cc

tests/
  operators/
    test_rmsnorm.cc
    test_linear.cc
    test_rope.cc
    test_softmax.cc
    test_attention.cc
    test_mlp.cc

  kernels/
    test_kernel_registry.cc
    test_kernel_selector.cc
```

目录职责建议如下：

| 目录 | 职责 |
|---|---|
| `operators/` | 语义层算子定义、参数校验、形状推导、调用 kernel |
| `kernels/` | kernel 抽象、注册、选择、执行 |
| `kernels/reference/` | 正确性优先的 reference kernel |
| `kernels/cpu/` | CPU 通用 kernel |
| `kernels/cpu/x86/` | x86 ISA 相关优化 kernel |
| `backend/` | backend 能力声明、资源上下文、运行时适配 |
| `dispatch/` | dispatch key、dispatch table、kernel 选择机制 |
| `runtime/` | thread pool、workspace、执行上下文 |
| `bench/` | 性能基准测试 |
| `tests/` | 单元测试、正确性测试、registry 测试 |

---

## 7. 算子执行路径设计

推荐执行路径如下：

```text
Executor
  ↓
Operator::Run()
  ↓
Operator 参数校验与 shape 推导
  ↓
构造 KernelDescriptor
  ↓
KernelSelector::Resolve()
  ↓
KernelRegistry 查找候选 kernel
  ↓
选择最合适 kernel
  ↓
获取 WorkspaceView
  ↓
Kernel::Invoke()
  ↓
写入 output tensor
```

示意：

```text
Input Tensor
    ↓
Operator Semantic Layer
    ↓
Normalized Kernel Arguments
    ↓
Kernel Descriptor
    ↓
Kernel Selection
    ↓
Reference / Optimized Kernel
    ↓
Output Tensor
```

### 7.1 Operator 层职责

Operator 层负责语义，不负责深度性能优化。

主要职责：

- 校验输入输出数量；
- 校验 dtype；
- 校验 shape；
- 校验 stride / contiguous 约束；
- 校验 device / backend；
- 推导输出 shape；
- 构造 KernelDescriptor；
- 获取 workspace；
- 调用 kernel；
- 返回 Status。

### 7.2 Kernel 层职责

Kernel 层负责实际计算。

主要职责：

- 按给定参数执行计算；
- 不做复杂 Tensor 语义判断；
- 不进行堆分配；
- 不持有 Tensor 生命周期；
- 不直接依赖 Executor；
- 通过 KernelContext 获取线程池、ISA、workspace 等运行上下文；
- 返回 Status 或错误码。

### 7.3 KernelSelector 职责

KernelSelector 负责根据 KernelDescriptor 选择最合适 kernel。

选择依据包括：

- op type；
- dtype；
- backend；
- device；
- layout；
- ISA capability；
- shape class；
- packed weight availability；
- workspace requirement；
- phase：prefill / decode；
- implementation priority；
- fallback policy。

推荐 fallback 顺序：

```text
specialized optimized kernel
    ↓
generic optimized kernel
    ↓
cpu reference kernel
    ↓
NotFound
```

---

## 8. KernelDescriptor 设计建议

KernelDescriptor 至少应覆盖如下信息：

```cpp
struct KernelDescriptor {
    OpType op_type;
    BackendType backend;
    DeviceType device;
    DataType input_dtype;
    DataType weight_dtype;
    DataType output_dtype;
    LayoutType input_layout;
    LayoutType weight_layout;
    LayoutType output_layout;
    KernelPhase phase;        // prefill / decode / generic
    IsaCapability isa;
    bool has_packed_weight;
    bool requires_workspace;
};
```

建议支持 `ToString()`，用于 Resolve 失败时输出明确诊断信息。

示例错误信息：

```text
Kernel not found:
  op_type        = Linear
  backend        = CPU
  device         = CPU
  input_dtype    = Float32
  weight_dtype   = Float32
  output_dtype   = Float32
  input_layout   = RowMajorContiguous
  weight_layout  = PackedX86
  phase          = Decode
  isa            = AVX2
  packed_weight  = true
```

这样可以避免 `Resolve() failed` 这类弱诊断信息。

---

## 9. 算子开发优先级

### 9.1 第一优先级：必须立即实现

第一优先级算子的目标是跑通 Llama Phase 1 最小推理路径，并验证算子基础设施。

| 算子 | 当前目标 | 原因 |
|---|---|---|
| Embedding | Reference Kernel | input ids 到 hidden states 的入口 |
| RMSNorm | Reference + SIMD 原型 | Llama 必需，适合验证 Tensor / kernel 接口 |
| Linear / MatMul / GEMV | Reference + PackedWeight 原型 | 最大热点，影响权重预打包 |
| RoPE | Reference Kernel | 影响 Q/K layout、position ids、cos/sin cache |
| Attention | Reference Kernel | 验证 KVCache、causal mask、head layout |
| Softmax | Reference Kernel | Attention 必需 |
| Silu | Reference Kernel | MLP 必需 |
| Elementwise Mul | Reference Kernel | SwiGLU / gated MLP 必需 |
| Add / Residual | Reference Kernel | Transformer block 必需 |
| Argmax | Reference Kernel | greedy decoding 必需 |

### 9.2 第二优先级：小规模性能原型

第二优先级算子用于验证热点优化方向，但不要求一次性达到最终性能。

| 算子 | 当前目标 | 验证内容 |
|---|---|---|
| Linear / GEMV packed kernel | 性能原型 | PackedWeight、cache blocking、threading |
| RMSNorm SIMD kernel | 性能原型 | contiguous path、SIMD reduce |
| RoPE optimized kernel | 性能原型 | cos/sin cache、Q/K layout |
| KVCache Attention prototype | 性能原型 | KV 读取布局、decode attention |
| Softmax optimized kernel | 性能原型 | workspace、数值稳定性、SIMD |

### 9.3 第三优先级：暂缓

以下内容暂缓进入主线实现：

| 内容 | 暂缓原因 |
|---|---|
| AMX int8 GEMM | 强依赖量化格式和 packing 合同 |
| q4/q8 dequant fused matmul | 强依赖模型格式和量化规范 |
| 完整 FlashAttention-like CPU kernel | 实现复杂，依赖 KV/layout/fusion 边界 |
| 大规模 fused MLP | 需要稳定 operator fusion 机制 |
| RMSNorm + QKV fused kernel | 依赖 fusion 边界和权重布局 |
| RoPE + Attention fused kernel | 依赖 attention 接口稳定 |
| Continuous batching attention | 超出 Phase 1 单请求范围 |
| 多 backend 高性能 kernel | Phase 1 CPU-first，暂不铺开 |

---

## 10. 第一批算子详细策略

### 10.1 RMSNorm

RMSNorm 是当前最适合早期实现和优化的算子之一。

原因：

- Llama-family 必需；
- 输入输出语义简单；
- 易于测试；
- 易于 benchmark；
- 可验证 Tensor contiguous 语义；
- 可验证 KernelRegistry / KernelSelector；
- 可作为第一个 SIMD 原型。

Reference 公式：

```text
variance = mean(x_i^2)
rms = sqrt(variance + eps)
y_i = x_i / rms * weight_i
```

当前支持范围：

```text
dtype: float32
layout: contiguous
shape: [hidden_size] 或 [tokens, hidden_size]
backend: CPU
allocation: steady-state zero allocation
```

实现路径：

1. `rmsnorm_ref.cc`：标量 reference；
2. `rmsnorm_cpu.cc`：通用 CPU 版本；
3. `rmsnorm_avx2.cc`：SIMD 原型；
4. benchmark 对比 reference / cpu / avx2；
5. 与 PyTorch 结果做误差对齐。

测试重点：

- hidden size = 128 / 512 / 1024 / 4096 / 8192；
- eps 不同取值；
- 全 0 输入；
- 大值输入；
- 非法 shape；
- 非 contiguous 输入暂时拒绝；
- 输出误差阈值。

---

### 10.2 Linear / MatMul / GEMV

Linear 是 LLM 推理最大热点，应尽早实现 reference，并做少量 packed-weight 性能原型。

需要区分：

```text
Prefill: 更接近 GEMM
Decode: 更接近 GEMV 或 small-batch GEMM
```

当前建议优先覆盖：

```text
float32 input
float32 weight
float32 output
row-major contiguous input
row-major contiguous output
packed or unpacked weight
bias optional
```

Reference kernel：

```text
Y = X * W^T + bias
```

第一阶段实现：

1. 标量 reference；
2. 简单 threaded CPU 版本；
3. packed-weight 抽象；
4. decode GEMV 原型；
5. benchmark prefill / decode 分开测。

关键设计点：

- weight packing 不应发生在 decode 主循环；
- PackedWeight 应由 Model Loader 或 Model Prepare 阶段生成；
- KernelDescriptor 中应明确 `has_packed_weight`；
- kernel 不应直接读取模型结构对象；
- packed layout 必须版本化，避免后续不可兼容。

建议接口：

```cpp
struct PackedWeight {
    DataType dtype;
    LayoutType layout;
    int64_t rows;
    int64_t cols;
    int64_t block_m;
    int64_t block_n;
    const void* data;
};
```

当前不建议立即实现：

- int8 AMX；
- q4 dequant fused matmul；
- 多量化格式兼容；
- 复杂 auto-tuning；
- 多 ISA 版本全铺开。

---

### 10.3 RoPE

RoPE 应早期实现 reference，因为它直接影响 attention 接口。

当前需要明确：

- position id 来源；
- cos/sin cache 生命周期；
- Q/K tensor layout；
- head_dim；
- rotary_dim；
- prefill 与 decode 的 position 处理差异；
- GQA/MQA 下 K 的布局。

当前实现范围：

```text
dtype: float32
layout: contiguous
mode: Llama-style RoPE
position: absolute position id
cache: precomputed cos/sin table
```

不建议在第一阶段做：

- 多 RoPE scaling 策略全覆盖；
- YaRN / NTK scaling 等扩展；
- 多模型 RoPE 变体复杂兼容；
- RoPE + Attention fused kernel。

---

### 10.4 Attention / KVCache Attention

Attention 必须早做 reference，但不建议立即做极致优化。

原因：

- Attention 会验证 KVCache layout；
- 会验证 causal mask；
- 会验证 head layout；
- 会验证 prefill / decode 差异；
- 会验证 softmax workspace；
- 会验证 GQA/MQA 支持边界。

建议拆分为两个阶段：

#### 阶段一：Reference Attention

支持：

```text
single request
float32
causal attention
static KV cache
Llama-family dense
num_heads
num_kv_heads
head_dim
```

#### 阶段二：Decode KV Attention 原型

重点验证：

```text
Q: 当前 token
K/V: KVCache 中历史 token
score: QK^T
softmax
output: score * V
```

关键问题：

- KVCache 是按 layer/head/token/head_dim 组织，还是按 layer/token/head/head_dim 组织；
- decode 下连续读取是否 cache-friendly；
- GQA 下多个 Q head 如何映射到 KV head；
- softmax workspace 如何预分配；
- causal mask 在 decode 下是否可省略；
- prefill 与 decode 是否使用不同 kernel。

当前不建议实现：

- FlashAttention-like full optimization；
- block sparse attention；
- paged attention；
- continuous batching attention；
- RoPE + Attention fused；
- quantized KVCache attention。

---

### 10.5 Softmax

Softmax 是 Attention 的关键子算子。

Reference 实现必须采用数值稳定形式：

```text
max_val = max(x)
exp_sum = sum(exp(x_i - max_val))
y_i = exp(x_i - max_val) / exp_sum
```

当前实现范围：

```text
float32
1D row softmax
attention score softmax
workspace external
contiguous
```

测试重点：

- 大正数；
- 大负数；
- 全相等；
- 单元素；
- 长 context；
- NaN / Inf 策略；
- 和 PyTorch 对齐误差。

---

### 10.6 MLP 相关算子

Llama MLP 通常包含：

```text
gate_proj
up_proj
silu
elementwise_mul
down_proj
```

Phase 1 可先采用非融合路径：

```text
gate = Linear(x, gate_weight)
up = Linear(x, up_weight)
act = Silu(gate)
hidden = act * up
out = Linear(hidden, down_weight)
```

当前建议：

- Linear 使用统一 Linear kernel；
- Silu 实现 reference；
- Elementwise Mul 实现 reference；
- 后续再考虑 fused SwiGLU。

暂缓：

```text
Silu + Mul fused
Gate/Up projection fused
MLP full fused kernel
Quantized MLP fused kernel
```

---

### 10.7 Argmax / Greedy Sampling

Phase 1 采用 greedy decoding，因此需要 Argmax。

当前实现范围：

```text
input: logits [vocab_size]
output: token_id
dtype: float32
tie-break: first max index
```

注意事项：

- tie-break 规则必须固定；
- 不引入 temperature、top-k、top-p；
- 不引入随机数；
- 不引入采样状态。

---

## 11. Reference Kernel 规范

Reference Kernel 是正确性基准，不追求极限性能。

### 11.1 基本要求

每个 Reference Kernel 应满足：

- 代码简单直接；
- 行为确定；
- 易于审查；
- 不依赖平台 ISA；
- 不使用复杂并行；
- 不做隐式动态分配；
- 不吞掉错误；
- 返回明确 Status。

### 11.2 命名规范

建议命名：

```text
RmsNormReferenceKernel
LinearReferenceKernel
RoPEReferenceKernel
SoftmaxReferenceKernel
AttentionReferenceKernel
SiluReferenceKernel
ElementwiseMulReferenceKernel
ArgmaxReferenceKernel
```

### 11.3 Reference Kernel 与 Optimized Kernel 的关系

优化 kernel 必须能够通过相同输入与 reference 结果对齐。

测试结构：

```text
Generate input
    ↓
Run reference kernel
    ↓
Run optimized kernel
    ↓
Compare output
    ↓
Check error threshold
```

误差阈值建议：

| dtype | atol | rtol |
|---|---:|---:|
| float32 scalar | 1e-5 | 1e-5 |
| float32 SIMD | 1e-4 | 1e-4 |
| attention output | 1e-4 | 1e-4 |

具体阈值应根据算子和数值路径调整。

---

## 12. Correctness Test 策略

### 12.1 测试类型

算子测试至少包括：

1. 正常输入测试；
2. 边界 shape 测试；
3. 非法参数测试；
4. dtype 不匹配测试；
5. layout 不支持测试；
6. reference vs optimized 对齐测试；
7. 与 PyTorch / NumPy golden data 对齐测试；
8. allocation-free 检查；
9. determinism 检查。

### 12.2 Golden Data

建议为关键算子引入 golden data：

```text
tests/data/operators/
  rmsnorm_case_001.json
  linear_case_001.json
  rope_case_001.json
  softmax_case_001.json
  attention_case_001.json
```

可以由 Python / PyTorch 生成：

```text
input tensor
weight tensor
parameters
expected output
tolerance
```

### 12.3 Shape 覆盖建议

RMSNorm：

```text
hidden_size: 128, 512, 1024, 4096, 8192
tokens: 1, 4, 16, 128
```

Linear：

```text
M: 1, 4, 16, 128
K: 512, 1024, 4096
N: 512, 1024, 4096, 11008, vocab_size
```

Attention：

```text
seq_len: 1, 8, 128, 1024
num_heads: 8, 16, 32
num_kv_heads: 8, 16, 32
head_dim: 64, 128
```

Softmax：

```text
length: 1, 8, 128, 1024, 4096
```

### 12.4 错误路径测试

必须覆盖：

- null pointer；
- shape 不匹配；
- dtype 不支持；
- workspace 不足；
- non-contiguous input；
- unsupported layout；
- unsupported backend；
- kernel not found；
- output buffer size 不足。

---

## 13. Benchmark 策略

### 13.1 Benchmark 目标

Benchmark 应服务于两个目标：

1. 量化单算子优化收益；
2. 定位端到端推理瓶颈。

### 13.2 Benchmark 分类

建议拆分为：

```text
bench/operators/      单算子 benchmark
bench/end_to_end/     端到端 prefill/decode benchmark
bench/memory/         KVCache / workspace / bandwidth benchmark
bench/dispatch/       dispatch overhead benchmark
```

### 13.3 必测指标

每个 benchmark 至少输出：

```text
op_name
implementation
dtype
shape
phase
latency_avg
latency_p50
latency_p90
latency_p99
throughput
bytes_read
bytes_written
estimated_bandwidth
estimated_flops
allocation_count
thread_count
workspace_bytes
```

### 13.4 Prefill 与 Decode 分开测

LLM 推理中 prefill 和 decode 的热点不同，必须分开测。

| 阶段 | 典型特征 | 主要瓶颈 |
|---|---|---|
| Prefill | 多 token 输入 | GEMM / attention compute |
| Decode | 单 token 逐步生成 | GEMV / memory bandwidth / KVCache read |
| Long context decode | 长上下文 | KVCache bandwidth / softmax |
| Small batch decode | 小批量 | dispatch overhead / thread overhead |

### 13.5 Benchmark 不应做的事

Benchmark 不应：

- 在计时区间内分配内存；
- 在计时区间内初始化大 buffer；
- 在计时区间内构造模型；
- 将数据加载和计算混在一起；
- 只看平均值，不看尾延迟；
- 只测单 shape；
- 只测单线程；
- 忽略 warmup。

---

## 14. Workspace 策略

Phase 1 要求 steady-state zero allocation，因此 workspace 必须静态规划。

### 14.1 Workspace 来源

workspace 应由 Executor 或 Runtime 统一管理：

```text
Model initialization
    ↓
Operator workspace requirement collection
    ↓
WorkspacePlan generation
    ↓
Workspace buffer allocation
    ↓
Decode steady-state reuse
```

### 14.2 Operator Workspace Requirement

每个 operator 应能声明 workspace 需求：

```cpp
struct WorkspaceRequirement {
    size_t bytes;
    size_t alignment;
    bool reusable;
    WorkspaceLifetime lifetime;
};
```

建议 lifetime 类型：

```text
PerOperator
PerLayer
PerToken
PerSequence
Persistent
```

### 14.3 Kernel 内部约束

Kernel 内部：

- 不得调用 `new` / `delete`；
- 不得构造隐式动态分配容器；
- 不得在热路径申请临时 vector；
- 必须通过 `WorkspaceView` 使用预分配内存；
- workspace 不足时返回明确错误。

---

## 15. PackedWeight 策略

权重预打包是 Linear/GEMV 高性能优化的核心。

### 15.1 当前阶段目标

当前只需要建立最小 PackedWeight 抽象，验证：

- Model Loader 能否生成 packed weight；
- Linear kernel 能否识别 packed weight；
- KernelDescriptor 能否区分 packed/unpacked；
- Executor 是否能在推理过程中复用 packed weight；
- benchmark 能否比较 packed/unpacked 性能差异。

### 15.2 不应过早固化复杂格式

当前不建议立即固化：

- 多量化格式 packing；
- AMX tile-specific packing；
- 多 ISA 独立 packing；
- auto-tuning packing；
- compressed sparse packing。

### 15.3 PackedWeight 元信息

建议 PackedWeight 至少包含：

```cpp
struct PackedWeight {
    DataType original_dtype;
    DataType packed_dtype;
    LayoutType packed_layout;
    BackendType backend;
    IsaCapability target_isa;

    int64_t rows;
    int64_t cols;
    int64_t block_rows;
    int64_t block_cols;

    const void* data;
    size_t bytes;
    size_t alignment;

    uint32_t format_version;
};
```

### 15.4 生命周期

PackedWeight 应在以下阶段生成：

```text
Model loading
    ↓
Model validation
    ↓
Weight transformation
    ↓
Weight packing
    ↓
LoadedModel / PreparedModel 持有
    ↓
Executor 推理期间只读访问
```

Decode 主循环不得进行 packing。

---

## 16. 算子融合策略

### 16.1 当前阶段结论

当前阶段可以规划 fusion hook，但不应过早实现复杂 fusion pass。

### 16.2 可以保留的抽象

可以预留：

```text
OpPattern
FusionGroup
KernelCapability
FusedKernelDescriptor
```

但不要让语义层依赖具体 fused kernel。

### 16.3 暂缓的融合

以下融合暂缓：

```text
RMSNorm + QKV Projection
RoPE + Attention
Silu + Mul + DownProjection
Dequant + MatMul
MatMul + Bias + Activation
Full MLP fusion
Full Transformer block fusion
```

### 16.4 后续进入融合的条件

满足以下条件后再系统性推进 fusion：

- 单算子 reference 已完整；
- 单算子 optimized kernel 已有稳定 benchmark；
- Tensor layout 稳定；
- WorkspacePlan 稳定；
- PackedWeight 稳定；
- Executor prefill/decode 路径稳定；
- 端到端 profiler 已证明 fusion 是主要收益来源。

---

## 17. 当前不建议支持过多 dtype / layout

Phase 1 建议优先收敛到：

```text
dtype: float32
layout: row-major contiguous
backend: CPU
device: CPU
request: single request
decode: greedy
```

后续再扩展：

```text
float16
bfloat16
int8
q8
q4
blocked layout
packed layout
multi-request
batch decode
continuous batching
```

过早支持过多 dtype 和 layout 会显著增加测试矩阵和接口复杂度，不利于 Phase 1 收敛。

---

## 18. 开发阶段规划

### 18.1 阶段一：算子基础设施

目标：

> 建立统一算子执行框架，让任意算子都能以一致方式注册、选择、执行、测试、benchmark。

任务：

1. 定义 `OpType`；
2. 定义 `Operator` 基类或函数式接口；
3. 定义 `KernelDescriptor`；
4. 定义 `Kernel` 抽象；
5. 实现 `KernelRegistry`；
6. 实现 `KernelSelector`；
7. 实现 `KernelContext`；
8. 实现 `WorkspaceView`；
9. 实现 reference kernel 注册机制；
10. 建立 operator test 模板；
11. 建立 benchmark 模板。

阶段产出：

```text
operators/ 基础接口
kernels/ 基础接口
KernelRegistry 可用
KernelSelector 可用
RMSNorm reference 跑通
Linear reference 跑通
benchmark 框架可用
```

---

### 18.2 阶段二：跑通 Llama 最小 forward

目标：

> 使用 reference kernel 跑通 Llama-family dense 模型的最小 forward/decode 路径。

路径：

```text
input_ids
  → embedding
  → layer 0..N
      → rmsnorm
      → qkv projection
      → rope
      → attention
      → residual add
      → rmsnorm
      → mlp
      → residual add
  → final norm
  → lm_head
  → argmax
  → next token
```

阶段产出：

```text
最小 Llama forward 可运行
单 token decode 可运行
greedy next token 可输出
核心算子 correctness test 通过
```

---

### 18.3 阶段三：热点识别

目标：

> 基于真实端到端路径识别热点，而不是凭经验盲目优化。

任务：

1. prefill benchmark；
2. decode benchmark；
3. 单算子 benchmark；
4. dispatch overhead benchmark；
5. workspace 使用分析；
6. KVCache 访问分析；
7. thread scaling 分析。

预期热点通常包括：

```text
Linear / GEMV
MLP projection
Attention KV read
Softmax
RMSNorm
RoPE
Dispatch overhead
Thread synchronization overhead
```

但实际优化顺序必须以 profiler 和 benchmark 为准。

---

### 18.4 阶段四：关键热点算子优化

目标：

> 对已确认的热点进行系统性优化。

优先方向：

1. Linear / GEMV packed kernel；
2. RMSNorm SIMD；
3. Softmax SIMD；
4. Decode KV Attention；
5. RoPE optimized path；
6. MLP 局部融合；
7. prefill GEMM 优化；
8. NUMA-aware memory placement；
9. thread scheduling 优化。

阶段产出：

```text
核心热点算子具备 optimized kernel
optimized kernel 与 reference 对齐
benchmark 显示稳定收益
端到端 decode latency 明显下降
```

---

### 18.5 阶段五：扩展 dtype / quantization / fusion

目标：

> 在系统主路径稳定后，扩展量化、融合和多 ISA 优化。

可选方向：

```text
int8 GEMM
q8/q4 dequant matmul
AMX kernel
AVX512 kernel
fused MLP
fused RMSNorm + Linear
fused attention
batch decode
continuous batching
```

该阶段不属于当前立即目标。

---

## 19. 当前阶段建议实施清单

### 19.1 立即实施

```text
1. 建立 operators/ 语义层目录
2. 建立 kernels/ 抽象与注册机制
3. 建立 KernelDescriptor / KernelSelector
4. 建立 reference kernel 目录
5. 实现 RMSNorm reference
6. 实现 Linear reference
7. 实现 RoPE reference
8. 实现 Softmax reference
9. 实现 Attention reference
10. 实现 Silu / Mul / Add / Argmax reference
11. 建立 correctness test 框架
12. 建立 benchmark 框架
```

### 19.2 并行小规模验证

```text
1. RMSNorm SIMD 原型
2. Linear packed-weight 原型
3. Decode GEMV 原型
4. KVCache Attention 原型
5. WorkspacePlan 与算子 workspace 对接
6. PackedWeight 与 ModelLoader 对接
```

### 19.3 暂缓

```text
1. 完整量化算子库
2. 复杂 fusion pass
3. 多 backend 高性能实现
4. AMX/AVX512 全量铺开
5. continuous batching attention
6. FlashAttention-like CPU kernel
7. 多模型格式下的量化 kernel 兼容
```

---

## 20. 判断某个算子现在该不该做的标准

### 20.1 现在应该做

满足以下任一条件的算子应进入当前阶段：

```text
1. Phase 1 跑通 Llama 必须依赖；
2. 能验证关键架构合同；
3. 后续一定是热点；
4. 实现成本低、返工风险小；
5. 可作为 reference/correctness oracle；
6. 可推动 benchmark 和 dispatch 体系成熟。
```

典型算子：

```text
RMSNorm
Linear
RoPE
Softmax
Attention reference
Silu
Elementwise Mul
Add
Argmax
Embedding
```

### 20.2 现在不应深做

满足以下条件的算子不应当前深做：

```text
1. 强依赖尚未冻结的 layout；
2. 强依赖量化格式；
3. 强依赖权重预打包格式；
4. 强依赖多线程调度策略；
5. 优化复杂但对当前闭环帮助不大；
6. 一旦接口变化会大面积返工。
```

典型内容：

```text
AMX int8 GEMM
q4 fused dequant matmul
复杂 fused attention
多 backend fused MLP
continuous batching attention
复杂 graph fusion pass
```

---

## 21. 风险与规避策略

### 21.1 风险：过早优化导致返工

规避：

- 先 reference 后 optimized；
- 优化 kernel 不直接依赖复杂 Tensor；
- packed format 版本化；
- fusion hook 预留但不强绑定；
- 每个 optimized kernel 必须通过 reference 对齐测试。

### 21.2 风险：算子开发脱离 Executor

规避：

- 第一批算子必须服务于最小 Llama forward；
- benchmark 必须包括端到端 decode；
- operator workspace 必须和 Executor WorkspacePlan 对接；
- attention 必须和 KVCacheManager 对接验证。

### 21.3 风险：benchmark 不可信

规避：

- warmup 与 measurement 分离；
- 计时区间内禁止分配；
- prefill / decode 分开测；
- 输出 p50 / p90 / p99；
- shape 覆盖 Llama 典型配置；
- 单线程与多线程都测；
- benchmark 数据记录到固定格式。

### 21.4 风险：Reference Kernel 变成低质量临时代码

规避：

- reference kernel 也必须纳入测试；
- reference kernel 代码必须保持清晰；
- reference 结果作为 optimized kernel oracle；
- reference 不应包含平台相关优化；
- reference 不应被随意删除。

### 21.5 风险：KernelSelector 诊断信息不足

规避：

- `KernelDescriptor`、`KernelCapability`、`KernelCandidate` 提供 `ToString()`；
- Resolve 失败时输出完整 descriptor；
- NotFound 信息应包含 op、dtype、layout、backend、phase、ISA、packed 状态。

---

## 22. 推荐里程碑

### M1：算子框架最小闭环

完成标准：

```text
OpType 定义完成
KernelDescriptor 定义完成
KernelRegistry 可注册/查找
KernelSelector 可 resolve
RMSNorm reference 可运行
RMSNorm test 通过
RMSNorm benchmark 可运行
```

### M2：Phase 1 核心 reference 算子集

完成标准：

```text
Embedding reference
RMSNorm reference
Linear reference
RoPE reference
Attention reference
Softmax reference
Silu reference
Mul reference
Add reference
Argmax reference
```

### M3：Llama 最小 forward 跑通

完成标准：

```text
单层 Transformer block 可运行
多层 Transformer block 可运行
lm_head 可输出 logits
argmax 可输出 next token
decode step 可执行
```

### M4：Benchmark 与 profiler 闭环

完成标准：

```text
单算子 benchmark 可运行
prefill benchmark 可运行
decode benchmark 可运行
dispatch overhead 可测
allocation count 可检查
```

### M5：第一批性能原型

完成标准：

```text
RMSNorm SIMD 原型
Linear packed-weight 原型
Decode GEMV 原型
KVCache Attention 原型
optimized kernel 与 reference 对齐
benchmark 显示收益
```

---

## 23. 最终建议

当前 AetherMind 的算子开发策略应定为：

> 以 Reference Kernel 和算子执行框架为主线，选取 RMSNorm、Linear/GEMV、RoPE、Attention/KVCache Attention、Softmax、MLP 基础算子作为第一批架构验证型算子；当前不进行大规模平台相关极致优化。待 Tensor、Backend、Executor、Workspace、ModelLoader、PackedWeight、KVCache 等合同稳定后，再进入系统性高性能算子开发阶段。

该策略的核心价值是：

1. 不让算子开发滞后于引擎主线；
2. 避免过早优化造成接口返工；
3. 通过关键算子反向验证架构设计；
4. 为后续 optimized kernel 提供 correctness oracle；
5. 用 benchmark 驱动真实性能优化；
6. 保持 Phase 1 范围可控，确保系统尽快跑通闭环。

---

## 24. 建议下一步

建议下一步优先落地以下三个文件/模块：

```text
aethermind/kernels/kernel_descriptor.h
aethermind/kernels/kernel_registry.h
aethermind/operators/rmsnorm_op.h
```

并以 RMSNorm 作为第一个完整样板算子，打通：

```text
Operator
    ↓
KernelDescriptor
    ↓
KernelRegistry
    ↓
Reference Kernel
    ↓
Correctness Test
    ↓
Benchmark
```

RMSNorm 样板跑通后，再复制该模式实现 Linear、RoPE、Softmax 和 Attention。
