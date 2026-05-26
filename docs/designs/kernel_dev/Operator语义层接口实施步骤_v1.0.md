# Operator 语义层接口实施步骤设计文档

版本：v1.0  
状态：实施建议稿  
适用范围：AetherMind Phase 1 CPU-first / Llama-family dense / 单请求同步推理  
更新时间：2026-05-26  

---

## 1. 文档目标

本文档用于指导 AetherMind 推理引擎中 `Operator` 语义层接口的落地实现。

当前 AetherMind 尚未完全实现，但已经进入算子体系建设阶段。此时最优先的工作不是直接大规模开发高性能 kernel，而是先建立一个轻量、稳定、可扩展的 Operator 语义层，用于承接 Executor 与 Kernel / Backend / Dispatch 之间的边界。

本文档重点回答：

1. Operator 语义层为什么要先做；
2. Operator 语义层应该负责什么、不应该负责什么；
3. 第一版 Operator 接口应该如何设计；
4. 与 KernelDescriptor、KernelRegistry、KernelSelector 如何衔接；
5. 如何以 RMSNormOp 作为第一个样板算子打通完整闭环；
6. 每一步应新增哪些文件、类、函数和测试；
7. 如何控制 Phase 1 范围，避免过度设计。

---

## 2. 总体结论

当前阶段应该优先落地 Operator 语义层接口。

推荐实施路径：

```text
Step 1: 建立 operators/ 目录与 OpType
Step 2: 定义 OperatorContext
Step 3: 定义 WorkspaceRequirement
Step 4: 定义轻量 Operator 接口风格
Step 5: 建立 KernelDescriptor 对接点
Step 6: 以 RMSNormOp 打通完整链路
Step 7: 接入 Reference Kernel
Step 8: 增加 Correctness Test
Step 9: 增加 Benchmark
Step 10: 复制样板到 Linear / RoPE / Softmax / Attention
```

当前不要做复杂的动态图 Operator Framework，也不要照搬 PyTorch 的完整 dispatcher/operator schema 系统。

Phase 1 的目标是：

> Operator 层负责语义校验、shape 推导、workspace 声明和 kernel 调度；Kernel 层负责裸指针计算；Executor 层负责模型执行流程；Backend 层负责设备和运行环境能力。

---

## 3. Operator 语义层定位

### 3.1 Operator 层在系统中的位置

推荐整体调用链：

```text
Executor
  ↓
Operator::Run()
  ↓
Validate / InferShape / BuildKernelDescriptor
  ↓
KernelSelector::Resolve()
  ↓
KernelRegistry 查找候选 kernel
  ↓
Kernel::Invoke()
  ↓
写入 Output Tensor
```

Operator 层是 Executor 和 Kernel 之间的语义边界：

```text
Executor 层：
- 负责模型执行顺序
- 负责 prefill/decode 流程
- 负责 layer 循环
- 负责 KVCache 生命周期
- 负责 WorkspacePlan 分配

Operator 层：
- 负责算子语义
- 负责输入输出约束
- 负责 shape/dtype/layout 校验
- 负责构造 KernelDescriptor
- 负责调度 kernel

Kernel 层：
- 负责数值计算
- 负责裸指针访问
- 负责 SIMD/threading/blocking
- 不理解复杂 Tensor 语义

Backend 层：
- 负责设备能力
- 负责 ISA 能力
- 负责运行时资源
- 负责 backend-specific capability
```

### 3.2 Operator 层应该负责

Operator 层应负责：

```text
1. 算子类型标识
2. 输入 Tensor 数量校验
3. 输出 Tensor 数量校验
4. dtype 校验
5. shape 校验
6. layout / contiguous 校验
7. device / backend 校验
8. 输出 shape 推导
9. workspace 需求声明
10. KernelDescriptor 构造
11. KernelSelector 调用
12. kernel 参数规范化
13. 调用 kernel
14. 错误信息补充
15. 返回 Status
```

### 3.3 Operator 层不应该负责

Operator 层不应负责：

```text
1. 不做具体数值计算
2. 不写 SIMD 内核
3. 不做线程切分细节
4. 不做权重 packing
5. 不直接管理 KVCache 底层内存
6. 不在 decode 热路径动态分配内存
7. 不持有模型权重生命周期
8. 不替代 Executor 的执行调度
9. 不做复杂 graph fusion
10. 不做动态图 schema 解析
```

---

## 4. Phase 1 设计边界

AetherMind Phase 1 的 Operator 层只服务于以下目标：

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

因此第一版 Operator 层应避免以下复杂能力：

```text
动态图 graph
自动微分
复杂 schema parser
复杂 overload
复杂 broadcasting
复杂 type promotion
复杂 layout propagation
多 backend 动态选择
复杂 operator fusion pass
continuous batching
multi-request scheduling
GPU backend
```

第一版应保持薄、稳定、易测试。

---

## 5. 推荐目录结构

建议新增或调整如下目录：

```text
include/aethermind/operators/
  op_type.h
  operator_context.h
  workspace_requirement.h
  operator_traits.h
  rmsnorm_op.h
  linear_op.h
  rope_op.h
  softmax_op.h
  attention_op.h
  elementwise_op.h
  argmax_op.h

src/operators/
  op_type.cc
  rmsnorm_op.cc
  linear_op.cc
  rope_op.cc
  softmax_op.cc
  attention_op.cc
  elementwise_op.cc
  argmax_op.cc

include/aethermind/kernels/
  kernel_descriptor.h
  kernel_context.h
  kernel.h
  kernel_registry.h
  kernel_selector.h

src/kernels/reference/
  rmsnorm_ref.cc
  linear_ref.cc
  rope_ref.cc
  softmax_ref.cc
  attention_ref.cc

tests/operators/
  rmsnorm_op_test.cc
  linear_op_test.cc
  rope_op_test.cc
  softmax_op_test.cc
  attention_op_test.cc

bench/operators/
  rmsnorm_bench.cc
  linear_bench.cc
  rope_bench.cc
  softmax_bench.cc
  attention_bench.cc
```

第一阶段不需要一次性创建所有文件。建议先落地：

```text
include/aethermind/operators/op_type.h
include/aethermind/operators/operator_context.h
include/aethermind/operators/workspace_requirement.h
include/aethermind/operators/rmsnorm_op.h

src/operators/op_type.cc
src/operators/rmsnorm_op.cc

tests/operators/rmsnorm_op_test.cc
bench/operators/rmsnorm_bench.cc
```

---

## 6. 实施步骤总览

### 阶段 A：Operator 基础设施

目标：建立 Operator 语义层最小公共基础设施。

任务：

```text
A1. 定义 OpType
A2. 定义 ToString(OpType)
A3. 定义 WorkspaceRequirement
A4. 定义 OperatorContext
A5. 明确 Operator 参数传递风格
A6. 约定 Status 和错误信息规范
```

### 阶段 B：Kernel 对接基础

目标：让 Operator 能够构造 KernelDescriptor 并通过 KernelSelector 找到 kernel。

任务：

```text
B1. 定义 KernelDescriptor 与 ToString()
B2. 定义 KernelContext
B3. 定义 Kernel 抽象或函数式 Kernel 接口
B4. 定义 KernelRegistry
B5. 定义 KernelSelector
B6. 打通 Reference Kernel 注册路径
```

### 阶段 C：RMSNorm 样板算子

目标：以 RMSNormOp 打通 Operator → KernelDescriptor → KernelSelector → Reference Kernel → Test → Benchmark 完整闭环。

任务：

```text
C1. 定义 RmsNormOpParams
C2. 定义 RmsNormOp
C3. 实现 Validate()
C4. 实现 GetWorkspaceRequirement()
C5. 实现 Run()
C6. 实现 RmsNormReferenceKernel
C7. 注册 RmsNorm reference kernel
C8. 编写单元测试
C9. 编写 benchmark
```

### 阶段 D：复制样板到核心算子

目标：基于 RMSNorm 样板扩展 Phase 1 必需算子。

顺序：

```text
D1. LinearOp
D2. RoPEOp
D3. SoftmaxOp
D4. ElementwiseMulOp / AddOp / SiluOp
D5. AttentionOp
D6. ArgmaxOp
D7. EmbeddingOp
```

---

## 7. Step A1：定义 OpType

### 7.1 文件

```text
include/aethermind/operators/op_type.h
src/operators/op_type.cc
```

### 7.2 设计目标

`OpType` 是 Operator 层和 Kernel 层之间的基础标识，应保持稳定、轻量、可诊断。

### 7.3 建议定义

```cpp
#pragma once

#include <string_view>

namespace aethermind {

enum class OpType {
    kUnknown = 0,

    kEmbedding,
    kRmsNorm,
    kLinear,
    kRoPE,
    kAttention,
    kSoftmax,
    kSilu,
    kElementwiseMul,
    kAdd,
    kArgmax,
};

[[nodiscard]] std::string_view ToString(OpType op_type) noexcept;

}  // namespace aethermind
```

### 7.4 实现建议

```cpp
#include "aethermind/operators/op_type.h"

namespace aethermind {

std::string_view ToString(OpType op_type) noexcept {
    switch (op_type) {
        case OpType::kEmbedding:
            return "Embedding";
        case OpType::kRmsNorm:
            return "RmsNorm";
        case OpType::kLinear:
            return "Linear";
        case OpType::kRoPE:
            return "RoPE";
        case OpType::kAttention:
            return "Attention";
        case OpType::kSoftmax:
            return "Softmax";
        case OpType::kSilu:
            return "Silu";
        case OpType::kElementwiseMul:
            return "ElementwiseMul";
        case OpType::kAdd:
            return "Add";
        case OpType::kArgmax:
            return "Argmax";
        case OpType::kUnknown:
        default:
            return "Unknown";
    }
}

}  // namespace aethermind
```

### 7.5 验收标准

```text
1. OpType 可被 KernelDescriptor 引用；
2. ToString(OpType) 可用于错误日志；
3. 未知类型返回 Unknown；
4. 不引入动态分配；
5. 不依赖具体 Operator 类。
```

---

## 8. Step A2：定义 WorkspaceRequirement

### 8.1 文件

```text
include/aethermind/operators/workspace_requirement.h
```

### 8.2 设计目标

Operator 需要声明执行所需临时空间，供 Executor 或 WorkspacePlan 做静态规划。

Phase 1 目标是 steady-state zero allocation，所以 workspace 需求必须显式化。

### 8.3 建议定义

```cpp
#pragma once

#include <cstddef>

namespace aethermind {

enum class WorkspaceLifetime {
    kNone = 0,
    kPerOperator,
    kPerLayer,
    kPerToken,
    kPerSequence,
    kPersistent,
};

struct WorkspaceRequirement {
    std::size_t bytes = 0;
    std::size_t alignment = alignof(std::max_align_t);
    WorkspaceLifetime lifetime = WorkspaceLifetime::kNone;
    bool reusable = true;

    [[nodiscard]] bool empty() const noexcept {
        return bytes == 0;
    }
};

}  // namespace aethermind
```

### 8.4 当前策略

第一版可以允许大部分算子返回 empty workspace：

```cpp
WorkspaceRequirement NoWorkspace() noexcept {
    return {};
}
```

但 Attention / Softmax 后续可能需要外部 workspace。

### 8.5 验收标准

```text
1. Operator 可以声明 workspace；
2. workspace requirement 不触发动态分配；
3. 可表达 0 workspace；
4. 可表达 alignment；
5. 可被 WorkspacePlan 后续消费。
```

---

## 9. Step A3：定义 OperatorContext

### 9.1 文件

```text
include/aethermind/operators/operator_context.h
```

### 9.2 设计目标

`OperatorContext` 用于把运行时依赖传递给 Operator。第一版保持轻量，不做复杂所有权管理。

### 9.3 建议定义

```cpp
#pragma once

namespace aethermind {

class Backend;
class KernelRegistry;
class KernelSelector;
class ThreadPool;
class WorkspaceView;

enum class KernelPhase {
    kGeneric = 0,
    kPrefill,
    kDecode,
};

struct OperatorContext {
    Backend* backend = nullptr;
    KernelRegistry* kernel_registry = nullptr;
    KernelSelector* kernel_selector = nullptr;
    ThreadPool* thread_pool = nullptr;
    WorkspaceView* workspace = nullptr;
    KernelPhase phase = KernelPhase::kGeneric;

    bool enable_profiling = false;
    bool enable_debug_check = false;
};

}  // namespace aethermind
```

### 9.4 注意事项

当前可以使用裸指针，但语义必须明确：

```text
OperatorContext 不拥有 backend；
OperatorContext 不拥有 kernel_registry；
OperatorContext 不拥有 kernel_selector；
OperatorContext 不拥有 thread_pool；
OperatorContext 不拥有 workspace；
OperatorContext 仅在一次 Operator::Run 调用期间有效。
```

### 9.5 验收标准

```text
1. OperatorContext 可传入 Run；
2. 不产生所有权歧义；
3. 可携带 phase 信息；
4. 可访问 KernelSelector；
5. 可访问 WorkspaceView；
6. 可扩展 profiler/debug flag。
```

---

## 10. Step A4：确定 Operator 接口风格

### 10.1 不建议第一版做重型虚函数体系

当前不建议立即设计如下复杂接口：

```cpp
class Operator {
public:
    virtual Status Validate(...) const = 0;
    virtual Status InferShape(...) const = 0;
    virtual Status Run(...) const = 0;
};
```

原因：

```text
1. Phase 1 算子数量有限；
2. 当前重点是跑通闭环，不是构建通用动态图框架；
3. 每个 Llama 算子的输入输出形态差异很大；
4. 过早抽象容易导致接口被迫泛化；
5. 轻量静态类更容易审查和优化。
```

### 10.2 推荐第一版采用轻量静态 Operator 类

例如：

```cpp
class RmsNormOp {
public:
    explicit RmsNormOp(RmsNormOpParams params) noexcept;

    [[nodiscard]] OpType type() const noexcept;

    [[nodiscard]] Status Validate(
        const Tensor& input,
        const Tensor& weight,
        const Tensor& output) const;

    [[nodiscard]] WorkspaceRequirement GetWorkspaceRequirement(
        const Tensor& input,
        const Tensor& weight,
        const Tensor& output) const noexcept;

    [[nodiscard]] Status Run(
        OperatorContext& ctx,
        const Tensor& input,
        const Tensor& weight,
        Tensor& output) const;

private:
    RmsNormOpParams params_;
};
```

### 10.3 后续可演进方向

当算子数量增加、Executor 需要统一持有 Operator 时，再考虑抽象：

```cpp
class IOperator {
public:
    virtual ~IOperator() = default;
    virtual OpType type() const noexcept = 0;
};
```

但第一版不必强行做。

### 10.4 验收标准

```text
1. 单个 Operator 类可独立实现；
2. 不依赖复杂基类；
3. Run 接口清晰；
4. 参数强类型；
5. 易于单元测试；
6. 后续可逐步抽象。
```

---

## 11. Step B1：定义 KernelDescriptor 对接点

### 11.1 文件

```text
include/aethermind/kernels/kernel_descriptor.h
```

### 11.2 设计目标

`KernelDescriptor` 是 Operator 层向 KernelSelector 描述需求的核心对象。

Operator 不应直接选择 AVX2 / AVX512 / reference kernel，而是通过 descriptor 表达需求，由 KernelSelector 决定。

### 11.3 最小定义建议

```cpp
#pragma once

#include <string>

#include "aethermind/operators/op_type.h"

namespace aethermind {

enum class BackendType;
enum class DeviceType;
enum class DataType;
enum class LayoutType;
enum class IsaCapability;
enum class KernelPhase;

struct KernelDescriptor {
    OpType op_type = OpType::kUnknown;

    BackendType backend;
    DeviceType device;

    DataType input_dtype;
    DataType weight_dtype;
    DataType output_dtype;

    LayoutType input_layout;
    LayoutType weight_layout;
    LayoutType output_layout;

    KernelPhase phase = KernelPhase::kGeneric;
    IsaCapability isa;

    bool has_packed_weight = false;
    bool requires_workspace = false;

    [[nodiscard]] std::string ToString() const;
};

}  // namespace aethermind
```

### 11.4 当前可简化字段

如果现有 `DataType`、`DeviceType`、`BackendType`、`LayoutType` 已经定义，则直接复用。

如果 `LayoutType` 尚未稳定，第一版可以先用更小的枚举：

```cpp
enum class LayoutType {
    kUnknown = 0,
    kContiguous,
    kPacked,
};
```

### 11.5 诊断要求

`ToString()` 必须在 Resolve 失败时输出足够信息：

```text
Kernel not found:
  op_type       = RmsNorm
  backend       = CPU
  device        = CPU
  input_dtype   = Float32
  weight_dtype  = Float32
  output_dtype  = Float32
  input_layout  = Contiguous
  output_layout = Contiguous
  phase         = Decode
  isa           = AVX2
  workspace     = false
```

### 11.6 验收标准

```text
1. Operator 能构造 KernelDescriptor；
2. KernelSelector 能根据 descriptor 查找 kernel；
3. Resolve 失败时能打印完整 ToString；
4. descriptor 不持有 Tensor；
5. descriptor 不持有裸指针；
6. descriptor 是轻量值对象。
```

---

## 12. Step B2：定义 Kernel 调用接口

### 12.1 设计原则

Kernel 层不应接收复杂 Operator 对象，也不应接收复杂 Tensor 生命周期。

Kernel 应接收：

```text
裸指针
维度
stride
参数
workspace
thread context
```

### 12.2 RMSNorm Kernel 低层接口示例

```cpp
using RmsNormKernelFn = Status (*)(
    const float* input,
    const float* weight,
    float* output,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    WorkspaceView* workspace,
    KernelThreadContext* thread_ctx) noexcept;
```

### 12.3 为什么这样设计

好处：

```text
1. Kernel 不依赖 Tensor 复杂语义；
2. Tensor 结构变化不会大面积影响 kernel；
3. optimized kernel 更容易做 SIMD；
4. reference kernel 和 optimized kernel 可共享同一低层签名；
5. 更容易 benchmark；
6. 更容易接入 packed weight。
```

### 12.4 验收标准

```text
1. Kernel 不包含 Tensor 参数；
2. Kernel 不分配堆内存；
3. Kernel 通过返回 Status 表达错误；
4. Kernel 接口可被 benchmark 直接调用；
5. Operator 负责从 Tensor 提取裸指针和维度。
```

---

## 13. Step C1：实现 RMSNormOp 样板

### 13.1 文件

```text
include/aethermind/operators/rmsnorm_op.h
src/operators/rmsnorm_op.cc
```

### 13.2 参数定义

```cpp
#pragma once

#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/workspace_requirement.h"

namespace aethermind {

class Tensor;
class Status;

struct RmsNormOpParams {
    float eps = 1.0e-6f;
};

class RmsNormOp {
public:
    explicit RmsNormOp(RmsNormOpParams params) noexcept;

    [[nodiscard]] OpType type() const noexcept {
        return OpType::kRmsNorm;
    }

    [[nodiscard]] Status Validate(
        const Tensor& input,
        const Tensor& weight,
        const Tensor& output) const;

    [[nodiscard]] WorkspaceRequirement GetWorkspaceRequirement(
        const Tensor& input,
        const Tensor& weight,
        const Tensor& output) const noexcept;

    [[nodiscard]] Status Run(
        OperatorContext& ctx,
        const Tensor& input,
        const Tensor& weight,
        Tensor& output) const;

private:
    RmsNormOpParams params_;
};

}  // namespace aethermind
```

### 13.3 Validate 语义

RMSNorm 的 Validate 应检查：

```text
1. input 已初始化；
2. weight 已初始化；
3. output 已初始化；
4. input dtype == float32；
5. weight dtype == float32；
6. output dtype == float32；
7. input 和 output shape 相同；
8. weight 是一维；
9. weight.numel == input.shape[-1]；
10. input contiguous；
11. weight contiguous；
12. output contiguous；
13. eps > 0；
14. output 可写。
```

### 13.4 rows / hidden_size 推导

RMSNorm 通常沿最后一维归一化：

```text
hidden_size = input.shape[-1]
rows = input.numel / hidden_size
```

例如：

```text
input shape = [tokens, hidden_size]
weight shape = [hidden_size]
output shape = [tokens, hidden_size]
```

### 13.5 WorkspaceRequirement

第一版 RMSNorm 不需要 workspace：

```cpp
WorkspaceRequirement RmsNormOp::GetWorkspaceRequirement(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output) const noexcept {
    return {};
}
```

### 13.6 Run 流程

`Run()` 应按如下步骤执行：

```text
1. 调用 Validate；
2. 从 Tensor 提取 dtype/layout/shape；
3. 计算 rows 和 hidden_size；
4. 构造 KernelDescriptor；
5. 调用 KernelSelector::Resolve；
6. 将 Tensor 转为裸指针；
7. 构造 KernelThreadContext；
8. 调用 kernel；
9. 返回 Status。
```

伪代码：

```cpp
Status RmsNormOp::Run(
    OperatorContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    Tensor& output) const {
    AM_RETURN_IF_ERROR(Validate(input, weight, output));

    const int64_t hidden_size = input.shape().back();
    const int64_t rows = input.numel() / hidden_size;

    KernelDescriptor desc;
    desc.op_type = OpType::kRmsNorm;
    desc.backend = BackendType::kCPU;
    desc.device = DeviceType::kCPU;
    desc.input_dtype = input.dtype();
    desc.weight_dtype = weight.dtype();
    desc.output_dtype = output.dtype();
    desc.input_layout = LayoutType::kContiguous;
    desc.weight_layout = LayoutType::kContiguous;
    desc.output_layout = LayoutType::kContiguous;
    desc.phase = ctx.phase;
    desc.requires_workspace = false;

    auto kernel = ctx.kernel_selector->Resolve(desc);
    if (!kernel.ok()) {
        return kernel.status();
    }

    const auto* input_ptr = input.data<float>();
    const auto* weight_ptr = weight.data<float>();
    auto* output_ptr = output.mutable_data<float>();

    KernelThreadContext thread_ctx;
    thread_ctx.thread_pool = ctx.thread_pool;

    return kernel.value()->InvokeRmsNorm(
        input_ptr,
        weight_ptr,
        output_ptr,
        rows,
        hidden_size,
        params_.eps,
        ctx.workspace,
        &thread_ctx);
}
```

具体代码应根据现有 `Status`、`Tensor`、`Expected` 或 `Result` 风格调整。

### 13.7 验收标准

```text
1. RmsNormOp 可编译；
2. Validate 能拒绝非法输入；
3. Run 不做动态分配；
4. Run 通过 KernelSelector 调用 kernel；
5. Run 不直接调用具体 reference 函数；
6. Run 不包含 SIMD/计算逻辑；
7. 错误信息包含 op name 和关键 shape/dtype 信息。
```

---

## 14. Step C2：实现 RmsNormReferenceKernel

### 14.1 文件

```text
src/kernels/reference/rmsnorm_ref.cc
```

### 14.2 计算逻辑

RMSNorm 公式：

```text
variance = mean(x_i^2)
scale = 1 / sqrt(variance + eps)
y_i = x_i * scale * weight_i
```

### 14.3 Reference 实现要求

```text
1. 标量实现；
2. 不使用 SIMD；
3. 不使用线程池；
4. 不分配堆内存；
5. 数值逻辑清晰；
6. 作为 optimized kernel 的 correctness oracle。
```

伪代码：

```cpp
Status RmsNormReferenceKernel(
    const float* input,
    const float* weight,
    float* output,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    WorkspaceView* workspace,
    KernelThreadContext* thread_ctx) noexcept {
    if (input == nullptr || weight == nullptr || output == nullptr) {
        return Status::InvalidArgument("RmsNormReferenceKernel received null pointer");
    }
    if (rows <= 0 || hidden_size <= 0) {
        return Status::InvalidArgument("RmsNormReferenceKernel received invalid shape");
    }
    if (eps <= 0.0f) {
        return Status::InvalidArgument("RmsNormReferenceKernel received invalid eps");
    }

    for (int64_t r = 0; r < rows; ++r) {
        const float* row = input + r * hidden_size;
        float* out = output + r * hidden_size;

        double sum_sq = 0.0;
        for (int64_t i = 0; i < hidden_size; ++i) {
            const double v = static_cast<double>(row[i]);
            sum_sq += v * v;
        }

        const double mean_sq = sum_sq / static_cast<double>(hidden_size);
        const double scale = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

        for (int64_t i = 0; i < hidden_size; ++i) {
            out[i] = static_cast<float>(
                static_cast<double>(row[i]) * scale * static_cast<double>(weight[i]));
        }
    }

    return Status::OK();
}
```

### 14.4 验收标准

```text
1. reference kernel 可独立测试；
2. 不依赖 Operator；
3. 不依赖 Tensor；
4. 不分配内存；
5. 可被 KernelRegistry 注册；
6. 可被 RmsNormOp 调用。
```

---

## 15. Step C3：接入 KernelRegistry / KernelSelector

### 15.1 注册目标

需要将 `RmsNormReferenceKernel` 注册为：

```text
op_type: RmsNorm
backend: CPU
dtype: float32
layout: contiguous
implementation: reference
```

### 15.2 Registry 设计要求

Registry 至少应支持：

```text
1. 注册 kernel；
2. 按 KernelDescriptor 查询 kernel；
3. 返回 NotFound 时携带 descriptor.ToString()；
4. 支持多个候选 kernel；
5. 支持 reference fallback。
```

### 15.3 Selector fallback 策略

推荐顺序：

```text
specialized optimized kernel
    ↓
generic optimized kernel
    ↓
reference kernel
    ↓
NotFound
```

当前第一版只有 reference kernel，也必须走同样流程。

### 15.4 验收标准

```text
1. RmsNorm reference kernel 可注册；
2. RmsNormOp::Run 可 resolve 到 reference kernel；
3. 找不到 kernel 时错误信息清楚；
4. 不允许 Operator 直接绕过 selector 调用 reference；
5. 后续增加 optimized kernel 不需要修改 Executor。
```

---

## 16. Step C4：编写 RmsNormOp 单元测试

### 16.1 文件

```text
tests/operators/rmsnorm_op_test.cc
```

### 16.2 测试类型

至少覆盖：

```text
1. 正常输入；
2. 多 rows 输入；
3. hidden_size 边界；
4. eps 不同取值；
5. dtype 不支持；
6. shape 不匹配；
7. weight 不是一维；
8. output shape 不匹配；
9. non-contiguous 输入暂时拒绝；
10. null / uninitialized tensor；
11. reference 对齐；
12. Run 走 KernelSelector。
```

### 16.3 推荐测试 shape

```text
[1, 128]
[1, 4096]
[4, 4096]
[16, 4096]
[1, 8192]
```

### 16.4 误差阈值

```text
float32 reference: atol=1e-5, rtol=1e-5
optimized SIMD:   atol=1e-4, rtol=1e-4
```

### 16.5 验收标准

```text
1. Validate 测试覆盖成功和失败路径；
2. Run 输出正确；
3. 错误路径返回明确 Status；
4. 单元测试不依赖端到端模型；
5. 测试中可以替换 KernelSelector 为 mock/stub。
```

---

## 17. Step C5：编写 RmsNorm Benchmark

### 17.1 文件

```text
bench/operators/rmsnorm_bench.cc
```

### 17.2 Benchmark 目标

第一版 benchmark 主要用于验证框架，而不是追求极限性能。

需要测：

```text
1. Operator 调用开销；
2. Kernel 调用开销；
3. 不同 hidden_size 下的 latency；
4. rows 不同取值的吞吐；
5. reference kernel baseline；
6. 后续 optimized kernel 对比。
```

### 17.3 推荐 benchmark case

```text
rows=1, hidden_size=4096
rows=1, hidden_size=8192
rows=4, hidden_size=4096
rows=16, hidden_size=4096
rows=128, hidden_size=4096
```

### 17.4 输出指标

```text
op_name
implementation
rows
hidden_size
dtype
latency_avg
latency_p50
latency_p90
latency_p99
bytes_read
bytes_written
estimated_bandwidth
allocation_count
```

### 17.5 计时要求

```text
1. warmup 和 measurement 分离；
2. 计时区间内不分配内存；
3. 计时区间内不构造 Tensor；
4. 计时区间内不注册 kernel；
5. 输入输出 buffer 预先分配；
6. benchmark 可重复运行。
```

---

## 18. Step D1：扩展 LinearOp

RMSNorm 样板跑通后，第二个 Operator 应实现 `LinearOp`。

### 18.1 为什么 LinearOp 第二个做

```text
1. Llama 中最大热点；
2. 会验证 weight layout；
3. 会验证 prefill/decode phase；
4. 会验证 packed weight 抽象；
5. 会验证大矩阵 shape；
6. 会推动 benchmark 体系成熟。
```

### 18.2 第一版范围

```text
dtype: float32
input layout: contiguous
weight layout: contiguous
output layout: contiguous
bias: optional
packing: 第一版可不启用，只预留字段
```

### 18.3 LinearOp Validate

应检查：

```text
1. input rank >= 1；
2. weight rank == 2；
3. input.shape[-1] == weight.shape[1]；
4. output.shape == input.shape[:-1] + [weight.shape[0]]；
5. dtype 均为 float32；
6. layout 均为 contiguous；
7. bias 如果存在，bias.shape == [weight.shape[0]]。
```

### 18.4 KernelDescriptor 差异

LinearOp 的 descriptor 应加入：

```text
has_packed_weight
phase = prefill / decode
weight_layout
```

### 18.5 暂缓内容

```text
1. int8 AMX；
2. q4/q8 dequant fused matmul；
3. full auto-tuning；
4. complex packed layout；
5. fused bias/activation。
```

---

## 19. Step D2：扩展 RoPEOp

### 19.1 第一版范围

```text
dtype: float32
layout: contiguous
Llama-style RoPE
precomputed cos/sin cache
position id 显式传入
```

### 19.2 Validate

应检查：

```text
1. q/k dtype == float32；
2. q/k shape 合法；
3. head_dim 可被 2 整除；
4. cos/sin cache shape 覆盖 position；
5. output shape 与 input shape 一致；
6. layout contiguous。
```

### 19.3 注意事项

RoPEOp 不应负责生成 cos/sin cache。cache 应由 Model 或 Runtime 初始化阶段准备。

---

## 20. Step D3：扩展 SoftmaxOp

### 20.1 第一版范围

```text
dtype: float32
axis: last dimension
layout: contiguous
numerically stable softmax
```

### 20.2 Validate

应检查：

```text
1. input/output shape 一致；
2. dtype == float32；
3. last_dim > 0；
4. layout contiguous。
```

### 20.3 Workspace

Softmax 可以第一版不使用 workspace，直接两 pass 实现：

```text
pass 1: max
pass 2: exp sum and output
```

后续 attention 可使用更专门的 softmax kernel。

---

## 21. Step D4：扩展 Elementwise / Silu / Add

### 21.1 算子集合

```text
SiluOp
ElementwiseMulOp
AddOp
```

### 21.2 第一版范围

```text
dtype: float32
same shape
contiguous
no broadcasting
```

### 21.3 暂不支持

```text
broadcasting
type promotion
strided tensor
in-place alias 复杂语义
```

---

## 22. Step D5：扩展 AttentionOp

AttentionOp 应在 RMSNorm、Linear、RoPE、Softmax 等基础算子稳定后再实现。

### 22.1 第一版范围

```text
single request
float32
causal attention
static KV cache
Llama-family dense
GQA/MQA 可预留但不复杂化
```

### 22.2 Operator 层职责

AttentionOp 负责：

```text
1. q/k/v shape 校验；
2. num_heads / num_kv_heads / head_dim 校验；
3. context length 校验；
4. causal mask 语义校验；
5. KVCache view 合法性校验；
6. 构造 attention kernel descriptor；
7. 调用 attention reference kernel。
```

### 22.3 Operator 层不负责

AttentionOp 不负责：

```text
1. KVCache 内存分配；
2. KVCache 页表管理；
3. RoPE 计算；
4. 复杂 fused attention；
5. paged attention；
6. continuous batching。
```

---

## 23. Step D6：扩展 ArgmaxOp

### 23.1 第一版范围

```text
input: logits [vocab_size]
output: token id
dtype: float32
tie-break: first max index
```

### 23.2 注意事项

ArgmaxOp 必须固定 tie-break 规则，保证 greedy decoding 可复现。

---

## 24. Operator 层错误处理规范

### 24.1 错误信息必须包含的信息

Operator 返回错误时，应尽可能包含：

```text
op type
input shape
output shape
dtype
layout
expected condition
actual condition
```

示例：

```text
RmsNormOp Validate failed: weight.numel must equal input.shape[-1],
input_shape=[4,4096], weight_shape=[2048]
```

### 24.2 不建议的错误信息

避免：

```text
Invalid shape
Failed
Bad input
Resolve failed
```

这些信息不足以定位问题。

### 24.3 Status 类型建议

如果已有 `Status`，建议至少包含：

```text
OK
InvalidArgument
Unimplemented
NotFound
Internal
ResourceExhausted
```

---

## 25. Operator 层与 Tensor 的关系

### 25.1 Operator 可依赖的 Tensor 能力

Operator 层至少需要 Tensor 提供：

```text
dtype()
shape()
numel()
is_contiguous()
data<T>()
mutable_data<T>()
device()
layout()
is_initialized()
```

### 25.2 Operator 不应依赖的 Tensor 细节

Operator 不应依赖：

```text
BufferImpl 内部结构
MemoryHandle 内部结构
DataPtr deleter
allocator 实现
storage offset 复杂语义
```

### 25.3 Empty Tensor 语义

如果当前 AetherMind 已定义 empty Tensor 语义，Operator Validate 必须显式处理。

建议 Phase 1 中：

```text
核心推理算子不接受未初始化 Tensor；
核心推理算子不接受 numel == 0 的运行输入；
shape 推导可以允许元信息对象，但 Run 不接受空数据。
```

---

## 26. Operator 层与 Backend 的关系

### 26.1 Operator 不直接实现 backend-specific 分支

不要在 Operator 中写：

```cpp
if (backend == CPU && isa == AVX2) {
    RunAvx2(...);
} else {
    RunReference(...);
}
```

而应统一：

```text
Operator 构造 KernelDescriptor
KernelSelector 选择具体 kernel
```

### 26.2 Operator 可以读取 backend capability

Operator 可以通过 `ctx.backend` 查询：

```text
BackendType
DeviceType
IsaCapability
threading capability
alignment requirement
```

但不要把 backend-specific 逻辑散落在各个 Operator 中。

---

## 27. Operator 层与 Executor 的关系

### 27.1 Executor 调用 Operator

Executor 应按模型结构调用 Operator：

```text
RmsNormOp
LinearOp
RoPEOp
AttentionOp
AddOp
RmsNormOp
LinearOp
SiluOp
ElementwiseMulOp
LinearOp
AddOp
FinalRmsNormOp
LinearOp
ArgmaxOp
```

### 27.2 Executor 不应绕过 Operator

Executor 不应直接调用：

```text
RmsNormReferenceKernel
LinearCpuKernel
AttentionRefKernel
```

否则后续 KernelSelector、benchmark、debug、fallback 都会被破坏。

### 27.3 Operator 不应替代 Executor

Operator 不应负责：

```text
layer loop
KVCache position 更新
token loop
prefill/decode 调度
model state 管理
```

---

## 28. Operator 层与 WorkspacePlan 的关系

### 28.1 当前阶段

第一版可以只让 Operator 返回 workspace requirement。

例如：

```cpp
auto req = op.GetWorkspaceRequirement(input, weight, output);
```

### 28.2 后续阶段

Executor 在模型准备阶段收集所有 Operator 的 workspace requirement：

```text
遍历 model layers
    ↓
构造 operator
    ↓
收集 workspace requirement
    ↓
生成 WorkspacePlan
    ↓
预分配 workspace buffer
    ↓
decode steady-state 复用
```

### 28.3 验收要求

```text
1. Operator 不在 Run 内动态分配 workspace；
2. workspace 不足时返回 ResourceExhausted；
3. workspace alignment 明确；
4. workspace lifetime 明确。
```

---

## 29. Operator 层与 Benchmark 的关系

### 29.1 两类 benchmark 都需要

单算子 benchmark 应支持两种入口：

```text
1. Operator-level benchmark
2. Kernel-level benchmark
```

区别：

```text
Operator-level:
- 包含 Validate
- 包含 KernelSelector
- 包含 Tensor 访问
- 反映真实框架开销

Kernel-level:
- 直接调用低层 kernel
- 排除 Operator 开销
- 用于分析 kernel 本身性能
```

### 29.2 RMSNorm benchmark 应同时保留两种路径

这样可以分离：

```text
Operator overhead
Kernel compute cost
Dispatch overhead
```

---

## 30. 代码落地顺序建议

推荐实际提交顺序：

### Commit 1：OpType

```text
include/aethermind/operators/op_type.h
src/operators/op_type.cc
tests/operators/op_type_test.cc
```

目标：

```text
OpType + ToString 可用。
```

### Commit 2：OperatorContext / WorkspaceRequirement

```text
include/aethermind/operators/operator_context.h
include/aethermind/operators/workspace_requirement.h
```

目标：

```text
Operator 公共上下文和 workspace 描述可用。
```

### Commit 3：KernelDescriptor 对接

```text
include/aethermind/kernels/kernel_descriptor.h
src/kernels/kernel_descriptor.cc
```

目标：

```text
KernelDescriptor 可由 Operator 构造，并支持 ToString。
```

### Commit 4：RMSNormOp 头文件与 Validate

```text
include/aethermind/operators/rmsnorm_op.h
src/operators/rmsnorm_op.cc
tests/operators/rmsnorm_op_test.cc
```

目标：

```text
RMSNormOp Validate 完整覆盖。
```

### Commit 5：RMSNorm Reference Kernel

```text
src/kernels/reference/rmsnorm_ref.cc
tests/kernels/rmsnorm_ref_test.cc
```

目标：

```text
RMSNorm reference kernel 可独立测试。
```

### Commit 6：KernelRegistry / Selector 接入 RMSNorm

```text
src/kernels/kernel_registry.cc
src/kernels/kernel_selector.cc
```

目标：

```text
RMSNormOp::Run 通过 selector 调用 reference kernel。
```

### Commit 7：RMSNorm Benchmark

```text
bench/operators/rmsnorm_bench.cc
```

目标：

```text
形成第一个可复用 benchmark 样板。
```

### Commit 8：复制到 LinearOp

```text
include/aethermind/operators/linear_op.h
src/operators/linear_op.cc
tests/operators/linear_op_test.cc
bench/operators/linear_bench.cc
```

目标：

```text
Linear reference 路径跑通。
```

---

## 31. 第一阶段验收清单

Operator 语义层第一阶段完成标准：

```text
[ ] include/aethermind/operators/ 目录建立
[ ] OpType 定义完成
[ ] ToString(OpType) 完成
[ ] WorkspaceRequirement 定义完成
[ ] OperatorContext 定义完成
[ ] KernelDescriptor 可由 Operator 构造
[ ] KernelDescriptor::ToString 可用
[ ] RmsNormOp 定义完成
[ ] RmsNormOp::Validate 完成
[ ] RmsNormOp::Run 完成
[ ] RmsNorm reference kernel 完成
[ ] RmsNorm reference kernel 注册完成
[ ] RmsNormOp 单元测试完成
[ ] RmsNorm benchmark 完成
[ ] Run 路径无动态分配
[ ] Executor 后续可调用 RmsNormOp
```

---

## 32. 第二阶段验收清单

核心 Operator 扩展完成标准：

```text
[ ] LinearOp reference 路径跑通
[ ] RoPEOp reference 路径跑通
[ ] SoftmaxOp reference 路径跑通
[ ] SiluOp reference 路径跑通
[ ] ElementwiseMulOp reference 路径跑通
[ ] AddOp reference 路径跑通
[ ] ArgmaxOp reference 路径跑通
[ ] AttentionOp reference 路径跑通
[ ] 所有 Operator 有 Validate 测试
[ ] 所有 Operator 有正常路径测试
[ ] 关键 Operator 有 benchmark
```

---

## 33. 常见设计误区

### 33.1 误区一：Operator 直接调用具体 kernel

错误：

```text
RmsNormOp 直接调用 RmsNormAvx2Kernel
```

正确：

```text
RmsNormOp 构造 KernelDescriptor，由 KernelSelector 选择 kernel。
```

### 33.2 误区二：Operator 中写计算逻辑

错误：

```text
RmsNormOp::Run 里直接写 RMSNorm for loop。
```

正确：

```text
RmsNormOp::Run 只做校验、descriptor、resolve、invoke。
```

### 33.3 误区三：第一版做复杂基类

错误：

```text
一开始构建完整动态图 Operator Framework。
```

正确：

```text
先用轻量静态 Operator 类跑通 Phase 1。
```

### 33.4 误区四：忽略错误诊断

错误：

```text
return Status::InvalidArgument("invalid input");
```

正确：

```text
return Status::InvalidArgument(
    "RmsNormOp expected weight.numel == input.shape[-1], "
    "but got weight.numel=2048, hidden_size=4096");
```

### 33.5 误区五：Operator 层过度支持通用性

错误：

```text
第一版支持 broadcasting、type promotion、strided layout、in-place alias。
```

正确：

```text
第一版只支持 Phase 1 必需的 float32 + contiguous。
```

---

## 34. 推荐最终路线

当前最合理的路线是：

```text
1. 先实现 OpType / OperatorContext / WorkspaceRequirement；
2. 再实现 KernelDescriptor 对接；
3. 以 RMSNormOp 为样板打通完整链路；
4. 将 RMSNorm 的模式复制到 Linear / RoPE / Softmax；
5. 在基础算子稳定后实现 AttentionOp；
6. 等端到端 Llama decode 跑通后，再考虑高性能 kernel 和 fusion。
```

一句话总结：

> AetherMind 当前需要先做 Operator 语义层，但必须做成薄语义层；第一目标不是抽象完美，而是通过 RMSNormOp 打通 Operator → KernelDescriptor → KernelSelector → Reference Kernel → Test → Benchmark 的可复用闭环。

---

## 35. 下一步建议

建议下一步实际编码从以下文件开始：

```text
include/aethermind/operators/op_type.h
src/operators/op_type.cc

include/aethermind/operators/workspace_requirement.h
include/aethermind/operators/operator_context.h

include/aethermind/operators/rmsnorm_op.h
src/operators/rmsnorm_op.cc
tests/operators/rmsnorm_op_test.cc
```

第一轮不要急于实现 Linear 和 Attention。先确保 RMSNorm 的完整闭环质量足够高。

RMSNorm 样板完成后，再进入：

```text
LinearOp
RoPEOp
SoftmaxOp
SiluOp / MulOp / AddOp
AttentionOp
ArgmaxOp
EmbeddingOp
```

这样可以最大限度降低接口返工风险，同时保证算子体系尽快服务 AetherMind Phase 1 推理闭环。
