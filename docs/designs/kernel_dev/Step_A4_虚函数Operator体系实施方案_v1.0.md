# Step A4 实施方案：基于虚函数的 Operator 语义层

> **关联文档**：[Operator语义层接口实施步骤_v1.0.md](./Operator语义层接口实施步骤_v1.0.md) Step A4
> **版本**：v1.0
> **状态**：Approved for implementation
> **调整说明**：本文档在原始 Step A4 "轻量静态类"建议基础上，调整为完整的虚函数 Operator 体系。

---

## 一、背景与设计目标

### 1.1 原始文档立场

[Operator语义层接口实施步骤_v1.0.md](./Operator语义层接口实施步骤_v1.0.md) 明确建议 Phase 1 不做重型虚函数体系，理由包括：

- Phase 1 算子数量有限
- 重点是跑通闭环，不是构建通用动态图框架
- 每个 Llama 算子的输入输出形态差异很大
- 过早抽象容易导致接口被迫泛化
- 轻量静态类更容易审查和优化

### 1.2 调整后的目标

在原始文档基础上调整为完整的虚函数体系。**关键约束**：不能把原始文档中的"轻量静态类"简单加个虚函数前缀，而要设计一套**真正有架构价值的 Operator 抽象**。

### 1.3 核心架构原则

| 原则 | 说明 |
|------|------|
| **最小抽象** | 只在"Executor 需要统一持有不同类型的 Operator"的地方使用虚函数 |
| **非虚优先** | 能静态分发的地方不做虚调用 |
| **两阶段分离** | Plan-time（Prepare/Validate/InferShapes）与 Runtime（Run）明确分开 |
| **不泛化接口** | Phase 1 不强行设计通用的 "any operator can accept any inputs" 接口；每个派生类保持强类型构造函数 |

---

## 二、现有系统集成分析

### 2.1 当前运行时调用链

```
ModelLoader::Load()
  → BuildExecutionPlan(runtime, model_instance, nodes)
    → ExecutionPlanBuilder::ResolveKernelForNode(backend, node)
      → Backend::ResolveKernelInfo(op_type, selector)
        → KernelRegistry::Resolve(request)
          → returns ResolvedKernel { .fn = KernelFunc, .attrs = ..., }
    → plan.AddStep(ExecutionStep { .op_type, .fn, .packed_params, ... })
  → returns ExecutionPlan

Executor::Execute(plan, bindings)
  → LayerRunner::Run(plan, bindings)
    → for each step: step.fn(invocation, op_ctx, workspace)
```

关键接口定义已就绪：

- **`KernelFunc`** (`include/aethermind/backend/kernel_types.h`)：
  ```cpp
  using KernelFunc = Status (*)(const KernelInvocation&,
                                const OpKernelContext&,
                                const WorkspaceBinding&) noexcept;
  ```

- **`OpKernelContext`** (`include/aethermind/backend/op_kernel_context.h`)：
  包含 `Device`, `Stream*`, `WorkspaceArena*`, `packed_params`, `attrs`, `debug_name`

- **`OperatorContext`** (`include/aethermind/operators/operator_context.h`)：
  包含 `Backend*`, `KernelRegistry*`, `WorkspaceArena*`, `KernelSelector`

- **`ExecutionStep`** (`include/aethermind/execution/execution_plan.h`)：
  ```cpp
  struct ExecutionStep {
      OpType op_type;
      KernelInvocation invocation;
      KernelFunc fn;              // ← 当前是裸函数指针
      const void* packed_params;
      WorkspaceRequirement workspace_requirement;
      std::span<const std::byte> attrs;
      const char* debug_name;
  };
  ```

### 2.2 集成切入点

虚函数 Operator 体系在以下两点切入现有系统：

1. **`ExecutionStep` 增加 `std::shared_ptr<const Operator>` 成员**：替代裸 `KernelFunc` 指针，执行时通过 Operator 做二次分发
2. **`LayerRunner::RunStep` 调用 `step.operator_->Run(...)` 而非 `step.fn(...)`**：但内部 Operator 仍使用已缓存的 `KernelFunc`

### 2.3 架构对比

```
                    当前架构                              目标架构
                    ────────                              ────────

ExecutionStep:                              ExecutionStep:
  op_type: OpType                             op_type: OpType
  fn: KernelFunc ─────────────┐              operator: shared_ptr<const Operator>
  packed_params               │                packed_params
  workspace_requirement       │                workspace_requirement
  attrs                       │                debug_name
  debug_name                  │                │
                              │                │
LayerRunner::RunStep:         │              LayerRunner::RunStep:
  step.fn(invocation, ...) ◄──┘ 直接调用       step.operator_->Run(invocation, ...)
                                                  │
                                                Operator::Run() 内部:
                                                  - 构造 OpKernelContext
                                                  - KERNEL_FN(inputs, ...)  ← 仍通过计划期缓存的 KernelFunc 调用
```

**关键变化**：`ExecutionStep` 从持有裸函数指针变为持有 `shared_ptr<const Operator>`，运行时通过虚函数 `Run()` 分发。但 Operator 内部的 kernel 调用仍然是非虚的（计划期已解析缓存），所以热路径没有额外虚函数调用开销。

---

## 三、虚函数接口设计

### 3.1 Operator 抽象基类

> **文件**：`include/aethermind/operators/operator.h`

```cpp
#ifndef AETHERMIND_OPERATORS_OPERATOR_H
#define AETHERMIND_OPERATORS_OPERATOR_H

#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"
#include "macros.h"

#include <cstddef>
#include <span>
#include <string>

namespace aethermind {

struct KernelInvocation;
struct OpKernelContext;
struct WorkspaceBinding;

/// Abstract base class for all semantic-layer operators.
///
/// An Operator encapsulates:
/// - Shape inference
/// - Workspace requirement estimation
/// - Kernel resolution and preparation
/// - Runtime execution dispatch
///
/// Lifecycle: Construct → Validate → Prepare → Run (repeated) → Destroy
///
/// Thread safety: Prepare and Run are NOT thread-safe by default.
/// Each invocation should use its own Operator instance or external
/// synchronization.
class Operator {
public:
    virtual ~Operator() = default;

    // ── Identity ────────────────────────────────────────────

    /// Returns the OpType enum for this operator.
    AM_NODISCARD virtual OpType type() const noexcept = 0;

    /// Returns a human-readable name for diagnostics and logging.
    /// Default implementation delegates to ToString(type()).
    AM_NODISCARD virtual const char* name() const noexcept {
        return ToString(type());
    }

    // ── Plan-time (单次调用): Validation ─────────────────────

    /// Validates the operator's internal configuration.
    ///
    /// Called after construction and parameter assignment, before Prepare().
    /// Implementations should verify that all required parameters are set
    /// and consistent (e.g., epsilon > 0 for RmsNorm, hidden_size > 0).
    ///
    /// Returns Ok if valid; otherwise returns a Status describing the error.
    AM_NODISCARD virtual Status Validate() const = 0;

    /// Validates that input tensors are compatible with this operator.
    ///
    /// Called during plan building. Checks shape compatibility, dtype
    /// constraints, contiguity requirements, etc.
    ///
    /// Returns Ok if compatible; otherwise returns a descriptive error Status.
    AM_NODISCARD virtual Status ValidateInputs(
            std::span<const TensorView> inputs) const = 0;

    // ── Plan-time (单次调用): Shape Inference ────────────────

    /// Infers output shapes from input shapes without executing.
    ///
    /// Used during graph construction and workspace planning.
    ///
    /// \return A vector of ShapeInfo descriptors (one per output),
    ///         or a Status error if inference fails.
    AM_NODISCARD virtual StatusOr<std::vector<ShapeInfo>> InferOutputShapes(
            std::span<const ShapeInfo> inputs) const = 0;

    // ── Plan-time (单次调用): Workspace ──────────────────────

    /// Computes the workspace requirement for this operator.
    ///
    /// Called during execution plan building. The result is stored in the
    /// ExecutionStep and used for unified workspace planning.
    ///
    /// Default returns zero-byte requirement (no scratch space needed).
    ///
    /// \param inputs  Input tensor shapes (for size-dependent estimation).
    AM_NODISCARD virtual WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const ShapeInfo> inputs) const noexcept {
        (void) inputs;
        return {};
    }

    // ── Plan-time (单次调用): Preparation ────────────────────

    /// Prepares this operator for execution using the provided context.
    ///
    /// Called once during plan building, BEFORE any Run() calls.
    /// Implementations should:
    /// - Resolve the kernel from OperatorContext::backend
    /// - Cache the resolved KernelFunc and attrs
    /// - Validate the resolved kernel matches expectations
    ///
    /// After successful Prepare(), the operator is ready for repeated Run() calls.
    ///
    /// \param ctx  Runtime context providing backend, workspace, and
    ///             kernel selector for resolution.
    /// \return Ok on success; Status error if kernel resolution fails.
    virtual Status Prepare(OperatorContext& ctx) = 0;

    // ── Runtime (可重复调用): Execution ──────────────────────

    /// Executes this operator on the given inputs and outputs.
    ///
    /// Called once per execution step. Must only be called after
    /// successful Prepare().
    ///
    /// Implementations should:
    /// - Construct an OpKernelContext using cached kernel metadata
    /// - Invoke the cached KernelFunc with inputs, outputs, and workspace
    ///
    /// \param invocation  Kernel invocation descriptor (op_type + selector).
    /// \param op_ctx      Kernel execution context (device, workspace, etc.).
    /// \param workspace   Pre-bound workspace slice from unified plan.
    /// \return Ok on success; Status error if execution fails.
    virtual Status Run(
            const KernelInvocation& invocation,
            const OpKernelContext& op_ctx,
            const WorkspaceBinding& workspace) const noexcept = 0;

    // ── Diagnostics ─────────────────────────────────────────

    /// Returns the resolved kernel info for debugging and logging.
    /// Only valid after successful Prepare().
    AM_NODISCARD virtual ResolvedKernel GetResolvedKernel() const noexcept = 0;
};

/// Shared pointer alias for operator lifetime management.
using OperatorPtr = std::shared_ptr<const Operator>;

} // namespace aethermind

#endif // AETHERMIND_OPERATORS_OPERATOR_H
```

### 3.2 纯虚 vs 非纯虚划分原则

| 方法 | 派发方式 | 理由 |
|------|---------|------|
| `type()` | **纯虚** | 每个算子必须声明自己的类型 |
| `name()` | **非纯虚**（默认实现） | 有合理的默认行为 `ToString(type())` |
| `Validate()` | **纯虚** | 每个算子有不同的参数验证逻辑 |
| `ValidateInputs()` | **纯虚** | 每个算子有不同的输入约束 |
| `InferOutputShapes()` | **纯虚** | 每个算子有不同的输出形状逻辑 |
| `ComputeWorkspaceRequirement()` | **非纯虚**（默认 0） | 多数简单算子不需要 scratch space |
| `Prepare()` | **纯虚** | 每个算子有不同的 kernel 解析逻辑 |
| `Run()` | **纯虚** | 每个算子有不同的执行上下文构造逻辑 |
| `GetResolvedKernel()` | **纯虚** | 诊断时必需的 |

**划分原则总结**：
- 有**合理零操作/零返回默认行为**的方法 → 非纯虚（`name()`, `ComputeWorkspaceRequirement()`）
- **每种算子的逻辑必然不同**的方法 → 纯虚（其余 7 个）

---

## 四、类层次结构

### 4.1 总体层次

```
                    Operator (abstract)
                   /    |     \      \
    UnaryElementwise  BinaryOp  RmsNormOp  LinearOp  EmbeddingOp  RoPEOp  ...
       (future)       (future)
```

Phase 1 聚焦 Llama 家族所需的 `OpType`，不做中间层次（如 "UnaryOp"/"BinaryOp"），避免过早泛化。

### 4.2 Phase 1 算子清单（映射自 `OpType`）

| OpType | 类名 | 文件 |
|--------|------|------|
| `kEmbedding` | `EmbeddingOp` | `operators/embedding_op.h/.cpp` |
| `kRmsNorm` | `RmsNormOp` | `operators/rms_norm_op.h/.cpp` |
| `kLinear` | `LinearOp` | `operators/linear_op.h/.cpp` |
| `kMatMul` | `MatMulOp` | `operators/mat_mul_op.h/.cpp` |
| `kRoPE` | `RoPEOp` | `operators/rope_op.h/.cpp` |
| `kAttention` | `AttentionOp` | `operators/attention_op.h/.cpp` |
| `kSiLU` | `SiLUOp` | `operators/silu_op.h/.cpp` |
| `kSoftmax` | `SoftmaxOp` | `operators/softmax_op.h/.cpp` |
| `kArgMax` | `ArgMaxOp` | `operators/argmax_op.h/.cpp` |
| `kCopy` | `CopyOp` | `operators/copy_op.h/.cpp` |
| `kAdd` | `AddOp` | `operators/add_op.h/.cpp` |
| `kMultiply` | `MultiplyOp` | `operators/multiply_op.h/.cpp` |

### 4.3 典型派生类示例：`RmsNormOp`

> **文件**：`include/aethermind/operators/rms_norm_op.h`

```cpp
#ifndef AETHERMIND_OPERATORS_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_RMS_NORM_OP_H

#include "aethermind/operators/operator.h"

namespace aethermind {

/// Parameters for RMS Normalization operator.
struct RmsNormOpParams {
    int64_t hidden_size = 0;   ///< Size of the last dimension.
    float epsilon = 1e-6f;     ///< Small constant for numerical stability.
};

/// RMS Normalization operator.
///
/// Computes: output = input * weight * rsqrt(mean(input^2) + epsilon)
///
/// Inputs (2): [input: (..., hidden_size), weight: (hidden_size,)]
/// Outputs (1): [output: (..., hidden_size)]
class RmsNormOp final : public Operator {
public:
    explicit RmsNormOp(RmsNormOpParams params) noexcept;

    // ── Identity ────────────────────────────────────────────
    AM_NODISCARD OpType type() const noexcept override;
    AM_NODISCARD const char* name() const noexcept override;

    // ── Validation ──────────────────────────────────────────
    AM_NODISCARD Status Validate() const override;
    AM_NODISCARD Status ValidateInputs(
            std::span<const TensorView> inputs) const override;

    // ── Shape Inference ─────────────────────────────────────
    AM_NODISCARD StatusOr<std::vector<ShapeInfo>> InferOutputShapes(
            std::span<const ShapeInfo> inputs) const override;

    // ── Workspace ───────────────────────────────────────────
    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const ShapeInfo> inputs) const noexcept override;

    // ── Preparation ─────────────────────────────────────────
    Status Prepare(OperatorContext& ctx) override;

    // ── Execution ───────────────────────────────────────────
    Status Run(const KernelInvocation& invocation,
               const OpKernelContext& op_ctx,
               const WorkspaceBinding& workspace) const noexcept override;

    // ── Diagnostics ─────────────────────────────────────────
    AM_NODISCARD ResolvedKernel GetResolvedKernel() const noexcept override;

private:
    RmsNormOpParams params_;
    ResolvedKernel resolved_kernel_{};
};

} // namespace aethermind

#endif
```

> **文件**：`src/operators/rms_norm_op.cpp`

```cpp
#include "aethermind/operators/rms_norm_op.h"
#include "aethermind/operators/operator_context.h"

namespace aethermind {

RmsNormOp::RmsNormOp(RmsNormOpParams params) noexcept
    : params_(params) {}

OpType RmsNormOp::type() const noexcept {
    return OpType::kRmsNorm;
}

const char* RmsNormOp::name() const noexcept {
    return "RmsNorm";
}

Status RmsNormOp::Validate() const {
    if (params_.hidden_size <= 0) {
        return Status::InvalidArgument(
                "RmsNorm: hidden_size must be positive, got " +
                std::to_string(params_.hidden_size));
    }
    if (params_.epsilon <= 0.0f) {
        return Status::InvalidArgument(
                "RmsNorm: epsilon must be positive, got " +
                std::to_string(params_.epsilon));
    }
    return Status::Ok();
}

Status RmsNormOp::ValidateInputs(
        std::span<const TensorView> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 inputs (hidden_states, weight), got " +
                std::to_string(inputs.size()));
    }
    const auto& hidden = inputs[0];
    const auto& weight = inputs[1];

    if (hidden.rank() < 1) {
        return Status::InvalidArgument("RmsNorm input must have at least 1 dimension");
    }
    if (hidden.dim(-1) != params_.hidden_size) {
        return Status::InvalidArgument(
                "RmsNorm input last dim mismatch: expected " +
                std::to_string(params_.hidden_size) + ", got " +
                std::to_string(hidden.dim(-1)));
    }
    if (weight.rank() != 1 || weight.dim(0) != params_.hidden_size) {
        return Status::InvalidArgument(
                "RmsNorm weight must be 1-D with size " +
                std::to_string(params_.hidden_size));
    }
    return Status::Ok();
}

StatusOr<std::vector<ShapeInfo>> RmsNormOp::InferOutputShapes(
        std::span<const ShapeInfo> inputs) const {
    if (inputs.size() < 1) {
        return Status::InvalidArgument(
                "RmsNorm requires at least 1 input for shape inference");
    }
    return std::vector<ShapeInfo>{inputs[0]};
}

WorkspaceRequirement RmsNormOp::ComputeWorkspaceRequirement(
        std::span<const ShapeInfo> inputs) const noexcept {
    (void) inputs;
    return WorkspaceRequirement{
            .bytes = static_cast<size_t>(params_.hidden_size) * sizeof(float),
            .alignment = 64,
            .lifetime = WorkspaceLifetime::kPerOperator,
            .reusable = true,
    };
}

Status RmsNormOp::Prepare(OperatorContext& ctx) {
    if (!Validate().ok()) {
        return Validate();
    }
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("OperatorContext.backend is null");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kRmsNorm, ctx.selector);
    if (!resolved.ok()) {
        return resolved.status();
    }
    resolved_kernel_ = resolved.value();
    resolved_kernel_.debug_name = "RmsNorm";
    return Status::Ok();
}

Status RmsNormOp::Run(
        const KernelInvocation& invocation,
        const OpKernelContext& op_ctx,
        const WorkspaceBinding& workspace) const noexcept {
    return resolved_kernel_.fn(invocation, op_ctx, workspace);
}

ResolvedKernel RmsNormOp::GetResolvedKernel() const noexcept {
    return resolved_kernel_;
}

} // namespace aethermind
```

### 4.4 派生类设计规范

| 规范 | 说明 |
|------|------|
| **标记 `final`** | Phase 1 所有派生类标记 `final`，不设计中间层次 |
| **构造函数强类型** | 每个算子有自己的 Params 结构体，不接受通用 `map<string, any>` |
| **参数值语义** | 构造时拷贝 Params，不持有外部引用 |
| **`Prepare()` 只解析 Kernel** | 不在此阶段做输入校验（那是 `ValidateInputs()` 的职责） |
| **`Run()` 只转发** | `Run()` 委托给 `resolved_kernel_.fn()`，不做额外逻辑 |
| **`GetResolvedKernel()` 直接返回缓存** | 不在每次调用时重新解析 |

---

## 五、与现有系统的兼容性处理

### 5.1 `ExecutionStep` 变更

> **文件**：`include/aethermind/execution/execution_plan.h`

**变更前**（当前）：

```cpp
struct ExecutionStep {
    OpType op_type = OpType::kUnknown;
    KernelInvocation invocation{};
    KernelFunc fn = nullptr;                  // ← 裸函数指针
    const void* packed_params = nullptr;
    WorkspaceRequirement workspace_requirement{};
    std::span<const std::byte> attrs{};
    const char* debug_name = nullptr;
};
```

**变更后**：

```cpp
struct ExecutionStep {
    OpType op_type = OpType::kUnknown;
    KernelInvocation invocation{};
    std::shared_ptr<const Operator> op;      // ← 共享 Operator 所有权
    const void* packed_params = nullptr;
    WorkspaceRequirement workspace_requirement{};
    const char* debug_name = nullptr;
};
```

**变更说明**：
- 移除 `KernelFunc fn` — 通过 `op->GetResolvedKernel().fn` 访问
- 移除 `std::span<const std::byte> attrs` — 通过 `op->GetResolvedKernel().attrs` 访问
- 新增 `std::shared_ptr<const Operator> op` — 统一管理生命周期
- **`owned_attrs_` 也可移除**（`attrs` 生命周期由 `Operator` 内部管理）

### 5.2 `ExecutionPlanBuilder` 变更

> **文件**：`src/execution/execution_plan_builder.cpp`

**变更后**：

```cpp
StatusOr<std::shared_ptr<const Operator>> ExecutionPlanBuilder::CreateAndPrepareOperator(
        RuntimeContext& runtime,
        const ExecutionPlanNodeSpec& node) noexcept {
    // 1. 通过 OperatorRegistry 创建算子
    auto op = OperatorRegistry::Create(node.op_type, node.op_params);
    if (!op.ok()) return op.status();

    // 2. 验证内部配置
    if (auto status = (*op)->Validate(); !status.ok()) return status;

    // 3. 构造 OperatorContext 并 Prepare
    OperatorContext op_ctx = MakeOperatorContext(runtime, node);
    if (auto status = (*op)->Prepare(op_ctx); !status.ok()) return status;

    return std::shared_ptr<const Operator>(std::move(*op));
}
```

### 5.3 `LayerRunner` 变更

> **文件**：`src/execution/layer_runner.cpp`

**变更后**：

```cpp
Status LayerRunner::RunStep(const ExecutionStep& step,
                            RuntimeBindingContext& bindings) noexcept {
    if (step.op == nullptr) {
        return Status::InvalidArgument("Execution step operator cannot be null");
    }
    const auto workspace_binding = bindings.BindWorkspace(step.workspace_requirement);
    if (!workspace_binding.ok()) return workspace_binding.status();

    const auto resolved = step.op->GetResolvedKernel();
    OpKernelContext op_ctx = BuildKernelContext(step, bindings, resolved);
    return step.op->Run(step.invocation, op_ctx, workspace_binding.value());
}
```

### 5.4 `OperatorRegistry` 工厂

> **文件**：`include/aethermind/operators/operator_registry.h`

```cpp
#ifndef AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H
#define AETHERMIND_OPERATORS_OPERATOR_REGISTRY_H

#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator.h"

#include <any>
#include <functional>
#include <memory>
#include <unordered_map>

namespace aethermind {

/// Factory for creating Operator instances from OpType.
///
/// Each concrete operator registers its factory function at static init time.
/// The factory accepts an opaque std::any parameter (typically the operator's
/// Params struct) and returns a unique_ptr<Operator>.
class OperatorRegistry {
public:
    using FactoryFunc = std::function<
            StatusOr<std::unique_ptr<Operator>>(const std::any& params)>;

    /// Register a factory for an OpType. Called at static init.
    /// Returns false if a factory already exists for this OpType.
    static bool Register(OpType op_type, FactoryFunc factory);

    /// Create an Operator instance for the given OpType with the given params.
    /// The params must contain the correct Params struct for this OpType.
    static StatusOr<std::unique_ptr<Operator>> Create(
            OpType op_type,
            const std::any& params);

private:
    static std::unordered_map<OpType, FactoryFunc>& Registry();
};

/// Helper macro for registering an operator.
/// Usage: AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)
#define AM_REGISTER_OPERATOR(op_type, OpClass)                                  \
    namespace {                                                                 \
        static const bool _am_reg_##OpClass = OperatorRegistry::Register(       \
                op_type,                                                        \
                [](const std::any& params) -> StatusOr<std::unique_ptr<Operator>> { \
                    try {                                                       \
                        auto p = std::any_cast<typename OpClass::Params>(params); \
                        return std::make_unique<OpClass>(p);                    \
                    } catch (const std::bad_any_cast&) {                        \
                        return Status::InvalidArgument(                         \
                                "Wrong params type for " #OpClass);             \
                    }                                                           \
                });                                                             \
    }

} // namespace aethermind

#endif
```

### 5.5 `CpuWeightPrepacker` 与新 Operator 的集成

当前 prepacker 通过 `TensorView` 消费权重，不依赖 Operator 接口。**不需要变更**。Operator 体系通过 `ExecutionPlanBuilder` 在 plan 构建阶段解析 kernel，prepacker 独立运行在模型加载阶段。

---

## 六、具体实施步骤

### Step 1：基础设施（1 个 PR）

**文件变更**：

| 操作 | 文件 |
|------|------|
| 新建 | `include/aethermind/operators/operator.h` — `Operator` 抽象基类 |
| 新建 | `include/aethermind/operators/operator_registry.h` — 工厂注册 |
| 新建 | `src/operators/operator_registry.cpp` — 工厂实现 |
| 修改 | `src/operators/CMakeLists.txt` — 添加编译目标 |
| 新建 | `tests/unit/operators/test_operator_registry.cpp` — 注册/创建测试 |

**验证方法**：

```bash
cmake --build build --target AetherMind -j
./build/tests/unit/aethermind_unit_tests --gtest_filter=OperatorRegistry.*
```

### Step 2：首个算子 + 执行路径打通（1 个 PR）

选择 **`RmsNormOp`**（最简单的路径，2 输入 1 输出，逻辑清晰）。

**文件变更**：

| 操作 | 文件 |
|------|------|
| 新建 | `include/aethermind/operators/rms_norm_op.h` |
| 新建 | `src/operators/rms_norm_op.cpp` |
| 修改 | `include/aethermind/execution/execution_plan.h` — `ExecutionStep` 增加 `op` |
| 修改 | `src/execution/execution_plan_builder.cpp` — 使用 `Operator` |
| 修改 | `src/execution/layer_runner.cpp` — 通过 `step.op->Run()` 分发 |

**验证方法**：

```bash
# 构建
cmake --build build --target AetherMind -j

# 运行现有 RmsNorm 相关测试确认无回归
./build/tests/unit/aethermind_unit_tests --gtest_filter=*RmsNorm*

# 运行完整单元测试套件
./build/tests/unit/aethermind_unit_tests
```

### Step 3：RmsNormOp 专项测试（1 个 PR）

```cpp
// tests/unit/operators/test_rms_norm_op.cpp

TEST(RmsNormOp, ValidatesPositiveHiddenSize) { ... }
TEST(RmsNormOp, ValidatesPositiveEpsilon) { ... }
TEST(RmsNormOp, RejectsIncorrectInputCount) { ... }
TEST(RmsNormOp, RejectsDimensionMismatch) { ... }
TEST(RmsNormOp, InfersSameOutputShapeAsInput) { ... }
TEST(RmsNormOp, ComputesNonZeroWorkspaceForLargeHiddenSize) { ... }
TEST(RmsNormOp, PrepareResolvesKernelFromBackend) { ... }
TEST(RmsNormOp, RunProducesNormalizedOutput) { ... }
```

### Step 4-8：剩余算子（每个算子 1 个 PR）

按 `OpType` 的顺序逐个实现：

- Step 4: `EmbeddingOp`
- Step 5: `LinearOp`
- Step 6: `MatMulOp`
- Step 7: `RoPEOp`
- Step 8: `AttentionOp`, `SiLUOp`, `SoftmaxOp`, `ArgMaxOp`, `CopyOp`, `AddOp`, `MultiplyOp`

### Step 9：移除兼容层

移除 `ExecutionStep` 中的 `fn` 和 `attrs` 字段，以及 `ExecutionPlan` 中的 `owned_attrs_`。

### Step 10：文档更新

更新 [Operator语义层接口实施步骤_v1.0.md](./Operator语义层接口实施步骤_v1.0.md) 的 Step A4 章节，记录调整决策。

---

## 七、技术验证方法

### 7.1 编译期验证

```bash
# 配置
cmake -S . -B build -DBUILD_TESTS=ON

# 全量构建（验证所有算子通过编译）
cmake --build build --target AetherMind -j
cmake --build build --target aethermind_unit_tests -j
```

### 7.2 虚函数接口契约验证

```cpp
// tests/unit/operators/test_operator_contract.cpp

// 验证所有已注册算子的接口完整性
TEST(OperatorContract, AllOperatorsOverrideAllPureVirtuals) {
    // 编译期验证：如果派生类遗漏纯虚函数，编译失败
}

// 验证 name() 不与 type() 矛盾
TEST(OperatorContract, NameMatchesType) {
    for (auto [op_type, _] : GetRegisteredOps()) {
        auto op = OperatorRegistry::Create(op_type, DefaultParamsFor(op_type));
        EXPECT_STREQ(op->name(), ToString(op->type()));
    }
}

// 验证 Prepare() 必须在 Validate() 之后
TEST(OperatorContract, InvalidOperatorsRejectPrepare) {
    // 创建一个无效算子（如 hidden_size = 0），Prepare() 应失败
}
```

### 7.3 性能回归检测

```bash
# 微基准测试（比较 Operator::Run 与直接 KernelFunc 调用的开销）
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_OpDispatch

# 端到端推理时延对比（修改前后）
./build/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_Inference
```

### 7.4 TSAN 线程安全验证

```bash
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON
cmake --build build-tsan --target aethermind_unit_tests -j
./build-tsan/tests/unit/aethermind_unit_tests --gtest_filter=Operator.*
./build-tsan/tests/unit/aethermind_unit_tests --gtest_filter=ExecutionPlan.*
```

### 7.5 文件结构验证（最终状态）

```
include/aethermind/operators/
    operator.h            ← 抽象基类
    operator_registry.h    ← 工厂
    operator_context.h     ← 已有，不变
    op_type.h              ← 已有，不变
    rms_norm_op.h          ← RmsNormOp
    embedding_op.h         ← EmbeddingOp
    linear_op.h            ← LinearOp
    mat_mul_op.h           ← MatMulOp
    rope_op.h              ← RoPEOp
    ...                     ← 其余算子

src/operators/
    CMakeLists.txt
    operator_registry.cpp
    rms_norm_op.cpp
    embedding_op.cpp
    linear_op.cpp
    mat_mul_op.cpp
    rope_op.cpp
    ...

tests/unit/operators/
    CMakeLists.txt
    test_operator_contract.cpp
    test_operator_registry.cpp
    test_rms_norm_op.cpp
    test_embedding_op.cpp
    ...
```

---

## 八、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 虚函数调用开销 | 每次 `Run()` 多一次间接跳转 | `Run()` 内部不做虚调用；benchmark 验证 |
| `shared_ptr` 原子引用计数开销 | 每个 `ExecutionStep` 持有 `shared_ptr` | Plan 构建一次，执行时计数不变（无原子操作） |
| `std::any` 类型擦除开销 | `OperatorRegistry::Create` 使用 `any_cast` | 只在模型加载阶段调用一次，不在热路径 |
| 接口过于泛化 | 过早抽象导致每个算子都要适配 | 派生类保持强类型构造，基类只定义最小契约 |
| `ExecutionStep` 变更破坏现有代码 | `fn`/`attrs` 字段移除 | Step 9 最后执行，中间阶段兼容共存 |
| `owned_attrs_` 移除导致悬挂 | `attrs` 生命周期由 `Operator` 管理 | `GetResolvedKernel()` 返回拷贝的 `ResolvedKernel`（不含悬空引用） |

---

## 九、与原文档的差异决策记录

| 原建议 | 调整决定 | 理由 |
|--------|---------|------|
| 不使用重型虚函数体系 | 使用基础虚函数体系（7 纯虚 + 2 非纯虚） | Executor 需要统一持有不同类型的 Operator 对象；当前已具备 Backend/KernelRegistry 等基础设施，虚函数抽象成本可控 |
| 轻量静态类，Run 接口直接暴露 | `Run()` 通过 `OperatorContext` 和缓存的 `KernelFunc` 转发 | 不暴露强类型 Run()，避免使用者绕过 Operator 层直接调 kernel |
| Phase 2 再抽象 | Phase 1 完成抽象 | 在算子数量有限时先建立正确接口，避免后续大规模重构 |
