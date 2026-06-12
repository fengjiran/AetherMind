# Operator 语义层全面审核报告

> **审核日期**：2026-05-30
> **审核范围**：`include/aethermind/operators/operator.h` 及其子类实现、OperatorRegistry、OperatorContext、FunctionOperator，以及与 ExecutionPlanBuilder、LayerRunner、KernelRegistry 的交互代码

---

## 目录

1. [架构概览](#1-架构概览)
2. [设计亮点](#2-设计亮点)
3. [问题缺陷](#3-问题缺陷)
4. [接口设计评估](#4-接口设计评估)
5. [错误处理机制评估](#5-错误处理机制评估)
6. [与其他模块的交互评估](#6-与其他模块的交互评估)
7. [改进建议](#7-改进建议)
8. [总结](#8-总结)

---

## 1. 架构概览

Operator 语义层是 AetherMind 算子系统的 **中间层**，向上承接 `ExecutionPlanBuilder` 的编译请求，向下通过 `ResolvedKernel` 调度 kernel 执行。

### 1.1 组件清单

| 组件 | 文件 | 职责 |
|------|------|------|
| `Operator`（基类） | `include/aethermind/operators/operator.h` | 定义算子生命周期接口 |
| `RmsNormOp` | `include/aethermind/operators/rmsnorm_op.h` / `src/operators/rmsnorm_op.cpp` | RMS 归一化算子 |
| `EmbeddingOp` | `include/aethermind/operators/embedding_op.h` / `src/operators/embedding_op.cpp` | 嵌入查找算子 |
| `FunctionOperator` | `include/aethermind/operators/function_operator.h` | Raw KernelFunc 的轻量适配器 |
| `OperatorRegistry` | `include/aethermind/operators/operator_registry.h` / `src/operators/operator_registry.cpp` | 算子工厂注册表 |
| `OperatorContext` | `include/aethermind/operators/operator_context.h` | Prepare 阶段上下文 |

### 1.2 类层次结构

```
Operator (abstract base)
  ├─ RmsNormOp       (final, AM_REGISTER_OPERATOR)
  ├─ EmbeddingOp     (final, AM_REGISTER_OPERATOR)
  └─ FunctionOperator (final, 未注册，手动构造)
```

### 1.3 生命周期模型

```
Construct → ValidateParams → Prepare → Run (repeated) → Destroy
     │            │              │           │
     │            │              │           └─ 同步调用 resolved_kernel_.fn(ctx)
     │            │              └─ kernel resolve + attrs 缓存
     │            └─ 参数语义校验
     └─ OperatorRegistry::Create 或手动构造
```

### 1.4 OpType 枚举状态

| OpType | Operator 实现 | Kernel 实现 | 状态 |
|--------|-------------|------------|------|
| `kEmbedding` | EmbeddingOp | CpuEmbeddingKernel | ✅ 完整 |
| `kRmsNorm` | RmsNormOp | CpuRmsNormKernelEntry | ✅ 完整 |
| `kLinear` | ❌ | ❌ | 待实现 |
| `kMatMul` | ❌ | ❌ | 待实现 |
| `kRoPE` | ❌ | ❌ | 待实现 |
| `kAttention` | ❌ | ❌ | 待实现 |
| `kSilu` | ❌ | ❌ | 待实现 |
| `kSiluMul` | ❌ | ❌ | 待实现 |
| `kElementwiseMul` | ❌ | ❌ | 待实现 |
| `kAdd` | ❌ | ❌ | 待实现 |
| `kSoftmax` | ❌ | ❌ | 待实现 |
| `kArgmax` | ❌ | ❌ | 待实现 |

---

## 2. 设计亮点

### 2.1 严格的生命周期契约

`operator.h` 明确定义了 `Construct → ValidateParams → Prepare → Run(repeated) → Destroy` 生命周期，每个阶段有明确的前置条件：

- `ValidateParams()` 在 `Prepare()` 之前调用
- `Prepare()` 一次性完成 kernel resolve 和缓存
- `Run()` 仅做数据绑定和直接调用，可重复执行

`ExecutionPlanBuilder::CreateAndPrepareOperator` 严格遵循此顺序。

### 2.2 InferenceResult 同时返回输出形状和运行时约束

`InferenceResult` 将 `outputs` 和 `runtime_checks` 绑定在一起返回，确保约束是形状推导的副产品而非独立操作。`RmsNormOp::InferOutputShapes` 正确利用了这一点——当符号阶段无法证明 `input.dim[1] == weight.dim[0]` 时，自动生成 `DimEqualConstraint`。

```cpp
struct InferenceResult {
    std::vector<TensorSpec> outputs{};
    std::vector<ShapeConstraint> runtime_checks{};
};
```

### 2.3 FunctionOperator 的优雅降级

`FunctionOperator` 将 raw `KernelFunc` 包装为 `Operator` 接口，所有虚方法提供空实现：

- `ValidateParams()` → `Ok()`
- `CheckInputSpecs()` → `Ok()`
- `InferOutputShapes()` → `InferenceResult{}`
- `Prepare()` → `Ok()`
- `Run()` → `resolved_kernel_.fn(ctx)`

这保证了 `ExecutionPlanBuilder` 的双路径分派逻辑统一，无需为 fallback 路径维护特殊代码。

### 2.4 AM_REGISTER_OPERATOR 宏的自动注册

宏同时注册 `factory_` 和 `make_default_params_`，算子开发者只需在 `.cpp` 文件中一行宏即可完成注册：

```cpp
AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)
```

且 `ExecutionPlanBuilder` 可通过 `CreateDefaultParams` 自动获取默认参数，无需硬编码 if-else。

### 2.5 OperatorRegistry 的先拷贝后解锁模式

`OperatorRegistry::Create` 在锁内拷贝 `factory`，解锁后再调用，避免在持锁期间执行用户代码导致死锁：

```cpp
FactoryFunc factory;
{
    std::lock_guard<std::mutex> lock(Mutex());
    const auto& registry = Registry();
    const auto it = registry.find(op_type);
    if (it == registry.end()) {
        return Status::NotFound(...);
    }
    factory = it->second.factory_;
}
return factory(params);  // 锁外调用
```

### 2.6 热路径零间接调用

`Operator::Run` 通过 `resolved_kernel_.fn(ctx)` 直接调用 kernel 函数指针，无注册表查找、无虚函数分派（`Run` 本身是虚调用，但只有一次，且编译器通常能 devirtualize `final` 类）。

---

## 3. 问题缺陷

### 🔴 P0：Operator::Run 的 const 语义与 OperatorPtr 的 const 组合依赖隐式调用顺序

`Run()` 被声明为 `const`：

```cpp
AM_NODISCARD virtual Status Run(KernelContext& ctx,
                                const RuntimeBindingContext& bindings,
                                size_t step_index) const noexcept = 0;
```

而 `OperatorPtr = shared_ptr<const Operator>` 意味着所有 Operator 实例都是 const 的。`Prepare()` 是非 const 方法（修改 `resolved_kernel_`），必须在 `shared_ptr<const Operator>` 构造之前调用。

**当前调用顺序**（正确）：

```
CreateAndPrepareOperator:
  1. OperatorRegistry::Create() → unique_ptr<Operator>  (非 const)
  2. op->Prepare()              (非 const，修改 resolved_kernel_)
  3. OperatorPtr(std::move(op)) → shared_ptr<const Operator>  (转为 const)
```

**风险**：如果未来有人重构 `ExecutionPlanBuilder`，将步骤 3 提前到步骤 2 之前，将导致编译错误（const 不能调用非 const 方法）。这是正确的编译期保护，但依赖隐式顺序，没有在类型层面强制。

**改进方案**：在 `Operator` 基类文档中明确声明 "Prepare() 必须在 OperatorPtr 构造前完成"，或引入 Builder 类型强制顺序。

---

### 🟡 P1：RmsNormOp::InferOutputShapes 与 CheckInputSpecs 存在大量重复校验

`CheckInputSpecs` 校验：
- 输入数量 == 2
- dtype == float32
- rank == 2 / rank == 1
- hidden_size 正性
- Unify 维度

`InferOutputShapes` 再次校验：
- 输入数量 == 2
- rank == 2 / rank == 1
- hidden_size / weight_length 正性

`EmbeddingOp` 同样存在此问题。

**根因**：`ExecutionPlanBuilder` 中 `CheckInputSpecs` 和 `InferOutputShapes` 是顺序调用的，但 `InferOutputShapes` 不能假设 `CheckInputSpecs` 已经被调用——因为接口层面没有强制调用顺序。

**改进方案**：在基类文档中明确声明 "InferOutputShapes 的调用前提是 CheckInputSpecs 已返回 Ok"，移除 `InferOutputShapes` 中的重复校验。

---

### 🟡 P1：FunctionOperator::Run 忽略 bindings 和 step_index

```cpp
Status Run(KernelContext& ctx,
           const RuntimeBindingContext& bindings,
           size_t step_index) const noexcept override {
    UNUSED(bindings);
    UNUSED(step_index);
    // ...
    return resolved_kernel_.fn(ctx);
}
```

`bindings` 和 `step_index` 被完全忽略。这意味着 `FunctionOperator` 无法从 `RuntimeBindingContext` 获取输入/输出张量绑定，也无法设置 `ctx.kernel_params`。

**影响**：FunctionOperator 路径的 kernel 无法获取运行时张量绑定，只能处理不需要张量绑定的简单 kernel。对于需要输入/输出张量的 kernel（如 RmsNorm），必须走 Operator 路径。

**当前状态**：这是有意为之的设计——FunctionOperator 仅用于不需要张量绑定的 fallback 场景。但接口层面没有阻止用户错误地期望 FunctionOperator 能传递张量绑定。

**改进方案**：在 `FunctionOperator` 文档中明确声明 "FunctionOperator 不支持运行时张量绑定，仅用于不需要输入/输出绑定的 fallback kernel"。

---

### 🟡 P1：RmsNormOp::Prepare 修改 resolved_kernel_ 但 Prepare 不是 const

```cpp
Status RmsNormOp::Prepare(OperatorContext& ctx) {
    // ...
    resolved_kernel_ = resolved.value();  // 修改成员
    resolved_kernel_.attrs.assign(eps_bytes.begin(), eps_bytes.end());  // 修改成员
    return Status::Ok();
}
```

`Prepare()` 是非 const 方法，修改 `resolved_kernel_` 成员。与 `OperatorPtr = shared_ptr<const Operator>` 的组合意味着 `Prepare()` 必须在 `shared_ptr<const Operator>` 构造之前调用。当前 `ExecutionPlanBuilder` 的调用顺序正确，但依赖隐式约定。

---

### 🟠 P2：ComputeWorkspaceRequirement 的默认实现返回零

```cpp
AM_NODISCARD virtual WorkspaceRequirement ComputeWorkspaceRequirement(
        std::span<const TensorSpec> inputs) const noexcept {
    UNUSED(inputs);
    return {};
}
```

默认返回零字节 workspace。`RmsNormOp` 显式 override 了此方法（也返回零），但 `EmbeddingOp` 未 override——依赖默认实现。如果未来 Embedding kernel 需要 scratch space（如分块处理），忘记 override 将导致 workspace 不足。

**改进方案**：将默认实现改为纯虚函数（`= 0`），强制每个子类显式声明 workspace 需求。

---

### 🟠 P2：OperatorContext 包含未使用的字段

```cpp
struct OperatorContext {
    Backend* backend = nullptr;
    const KernelRegistry* kernel_registry = nullptr;  // 未使用
    WorkspaceArena* workspace = nullptr;               // 未使用
    KernelSelector selector{};
    bool enable_profiling = false;                     // 未使用
    bool enable_debug_check = false;                   // 未使用
};
```

当前 `Prepare()` 实现只使用 `backend` 和 `selector`。其余字段是为未来功能预留的，但增加了接口复杂度。

**改进方案**：移除未使用字段，待需要时再加。遵循 YAGNI 原则。

---

### 🟠 P2：OperatorName 遗留类与 OpType 枚举并存

`include/operator_name.h` 中的 `OperatorName` 类（含 `name_` + `overload_name_` 字符串）与 `OpType` 枚举并存于代码库中。`OperatorName` 不被 Operator 体系使用，属于遗留代码。

**改进方案**：确认 `OperatorName` 无使用方后删除。

---

### 🟠 P2：RmsNormOp::Run 中 std::to_string 在 noexcept 函数中

```cpp
AM_NODISCARD Status Run(KernelContext& ctx,
                        const RuntimeBindingContext& bindings,
                        size_t step_index) const noexcept override {
    // ...
    return Status::InvalidArgument(
            "RMSNorm requires 2 input tensor bindings, got " +
            std::to_string(b->inputs.size()));  // std::to_string 可能抛 bad_alloc
}
```

`Run()` 标记为 `noexcept`，但 `std::to_string` 在内存不足时可能抛 `std::bad_alloc`。虽然实践中几乎不会发生，但严格来说违反了 `noexcept` 契约——如果抛出异常，将调用 `std::terminate()`。

**改进方案**：改用 `std::to_chars` 或预格式化字符串，或在错误路径使用不抛异常的整数转字符串方法。

---

### 🟠 P2：RmsNormOp::Run 中构造的 params 是栈上局部变量

```cpp
cpu::CpuRmsNormParams params{
    .input_tensor = b->inputs[0],
    .weight_tensor = b->inputs[1],
    .output_tensor = b->outputs[0],
};
ctx.kernel_params = &params;
return resolved_kernel_.fn(ctx);
```

`params` 是栈上局部变量，`ctx.kernel_params` 指向它。由于 `resolved_kernel_.fn(ctx)` 是同步调用，`params` 在调用期间有效。但如果未来 kernel 变为异步执行，`params` 将悬垂。

**当前状态**：安全，因为所有 CPU kernel 都是同步的。

---

### 🟢 P3：TensorSpec 缺少 operator==

```cpp
struct TensorSpec {
    DataType dtype{};
    SymbolicShape shape{};
};
```

没有定义 `operator==`，无法直接比较两个 `TensorSpec` 是否相等。在测试和调试中需要手动比较 `dtype` 和 `shape`。

---

### 🟢 P3：InferenceResult 缺少 operator==

同上，`InferenceResult` 也缺少比较运算符，测试中无法直接断言输出形状。

---

## 4. 接口设计评估

### 4.1 虚接口完整性

| 方法 | 纯虚/有默认实现 | 评估 |
|------|----------------|------|
| `Type()` | 纯虚 | ✅ 每个算子必须声明 |
| `Name()` | 有默认（委托 `ToString(Type())`） | ✅ 合理默认 |
| `ValidateParams()` | 纯虚 | ✅ 必须实现 |
| `CheckInputSpecs()` | 纯虚 | ✅ 必须实现 |
| `InferOutputShapes()` | 纯虚 | ✅ 必须实现 |
| `ComputeWorkspaceRequirement()` | 有默认（返回零） | ⚠️ 建议改为纯虚 |
| `Prepare()` | 纯虚 | ✅ 必须实现 |
| `Run()` | 纯虚 | ✅ 必须实现 |
| `GetResolvedKernel()` | 纯虚 | ✅ 必须实现 |

### 4.2 noexcept 一致性

| 方法 | noexcept | 评估 |
|------|----------|------|
| `Type()` | ✅ | 正确 |
| `Name()` | ✅ | 正确 |
| `ValidateParams()` | ❌ | 可能分配字符串（错误消息） |
| `CheckInputSpecs()` | ❌ | 同上 |
| `InferOutputShapes()` | ❌ | 同上 |
| `ComputeWorkspaceRequirement()` | ✅ | 正确 |
| `Prepare()` | ❌ | 可能分配字符串 + kernel resolve |
| `Run()` | ✅ | 正确（热路径不应抛异常） |
| `GetResolvedKernel()` | ✅ | 正确 |

`Run()` 标记 `noexcept` 是正确的设计——热路径不应抛异常。但 `RmsNormOp::Run` 中调用了 `std::to_string()`，在极端情况下可能抛 `bad_alloc`，严格来说违反了 `noexcept` 契约。

### 4.3 AM_NODISCARD 一致性

所有返回 `Status` 或 `StatusOr` 的方法都标记了 `AM_NODISCARD`，防止忽略返回值。一致性良好。

### 4.4 参数传递方式

| 方法 | 参数方式 | 评估 |
|------|---------|------|
| `CheckInputSpecs` | `span<const TensorSpec>` | ✅ 零拷贝视图 |
| `InferOutputShapes` | `span<const TensorSpec>` | ✅ 零拷贝视图 |
| `ComputeWorkspaceRequirement` | `span<const TensorSpec>` | ✅ 零拷贝视图 |
| `Prepare` | `OperatorContext&` | ✅ 非常量引用，允许修改 |
| `Run` | `KernelContext&`, `const RuntimeBindingContext&`, `size_t` | ✅ 合理 |

---

## 5. 错误处理机制评估

### 5.1 错误处理方式

| 方法 | 错误处理方式 | 评估 |
|------|-------------|------|
| `ValidateParams()` | 返回 `Status` | ✅ 一致 |
| `CheckInputSpecs()` | 返回 `Status` | ✅ 一致 |
| `InferOutputShapes()` | 返回 `StatusOr<InferenceResult>` | ✅ 一致 |
| `Prepare()` | 返回 `Status` | ✅ 一致 |
| `Run()` | 返回 `Status` + `noexcept` | ✅ 一致 |

### 5.2 错误消息质量

| 算子 | 方法 | 错误消息示例 | 评估 |
|------|------|-------------|------|
| RmsNormOp | ValidateParams | `"RmsNorm epsilon must be positive, got 0.000000"` | ✅ 含具体值 |
| RmsNormOp | CheckInputSpecs | `"RmsNorm expects exactly 2 inputs, got 3"` | ✅ 含具体值 |
| RmsNormOp | Run | `"RMSNorm requires 2 input tensor bindings, got 1"` | ✅ 含具体值 |
| EmbeddingOp | CheckInputSpecs | `"Embedding token ids must be int32, int64, or uint32"` | ✅ 明确 |
| FunctionOperator | Run | `"FunctionOperator kernel function cannot be null"` | ✅ 明确 |

所有错误消息都包含上下文信息，便于诊断。

### 5.3 错误传播路径

```
Operator::ValidateParams() → Status
  ↓ ExecutionPlanBuilder 检查
Operator::CheckInputSpecs() → Status
  ↓ ExecutionPlanBuilder 检查
Operator::InferOutputShapes() → StatusOr<InferenceResult>
  ↓ ExecutionPlanBuilder 检查
Operator::Prepare() → Status
  ↓ ExecutionPlanBuilder 检查
Operator::Run() → Status
  ↓ LayerRunner 检查
  ↓ Executor 返回给调用者
```

全链路使用 `Status`/`StatusOr` + `AM_RETURN_IF_ERROR`，错误立即传播，不吞没。

---

## 6. 与其他模块的交互评估

### 6.1 与 ExecutionPlanBuilder 的交互

```
ExecutionPlanBuilder::CreateAndPrepareOperator
  → OperatorRegistry::Create(op_type, params) → unique_ptr<Operator>
  → op->ValidateParams()
  → op->CheckInputSpecs(input_specs)
  → op->InferOutputShapes(input_specs) → InferenceResult
  → op->Prepare(op_ctx)  // kernel resolve
  → OperatorPtr(std::move(op))  // 转为 shared_ptr<const Operator>
  → ExecutionStep{op, inference.outputs, inference.runtime_checks}
```

交互清晰，职责边界明确。`ExecutionPlanBuilder` 负责调用顺序，`Operator` 负责各阶段的逻辑。

### 6.2 与 LayerRunner 的交互

```
LayerRunner::RunStep
  → BuildKernelContext(step, bindings) → KernelContext
  → ValidateShapeConstraints(step.runtime_checks, ...)
  → step.op->Run(ctx, bindings, step_index)
      → RmsNormOp::Run:
          ctx.kernel_params = &params  // 设置参数
          resolved_kernel_.fn(ctx)     // 直接调用 kernel
```

`Operator::Run` 通过修改 `KernelContext::kernel_params` 传递算子特定参数，通过 `resolved_kernel_.fn(ctx)` 直接调用 kernel。热路径零间接调用。

### 6.3 与 KernelRegistry 的交互

```
Operator::Prepare
  → ctx.backend->ResolveKernelInfo(op_type, selector)
    → KernelRegistry::Resolve(op_type, selector)  // 无锁读取
      → ResolvedKernel{op_type, fn, attrs, debug_name}
```

`Prepare()` 通过 `Backend` 间接访问 `KernelRegistry`，不直接依赖 `KernelRegistry`。解耦良好。

### 6.4 与 Shape Constraint 系统的交互

```
RmsNormOp::InferOutputShapes
  → 当 input.shape[1] != weight.shape[0] 时:
    → 生成 DimEqualConstraint{input[0].dim[1], input[1].dim[0]}
    → 写入 InferenceResult::runtime_checks

LayerRunner::RunStep
  → ValidateShapeConstraints(step.runtime_checks, inputs, outputs)
  → 违反 → 返回错误，不执行 kernel
```

约束声明与求值完全分离，`Operator` 只负责声明，`LayerRunner` 负责求值。

### 6.5 交互依赖图

```
                    ┌──────────────────┐
                    │ ExecutionPlanBuilder │
                    └────────┬─────────┘
                             │ 调用
                    ┌────────▼─────────┐
                    │ OperatorRegistry  │
                    │  ::Create()       │
                    └────────┬─────────┘
                             │ 返回
                    ┌────────▼─────────┐
                    │ Operator          │
                    │  ::ValidateParams │
                    │  ::CheckInputSpecs│
                    │  ::InferOutputShapes│
                    │  ::Prepare()      │
                    └────────┬─────────┘
                             │ Prepare 内部
                    ┌────────▼─────────┐
                    │ Backend           │
                    │  ::ResolveKernelInfo│
                    └────────┬─────────┘
                             │ 委托
                    ┌────────▼─────────┐
                    │ KernelRegistry    │
                    │  ::Resolve()      │
                    └──────────────────┘

                    ┌──────────────────┐
                    │ LayerRunner       │
                    │  ::RunStep()      │
                    └────────┬─────────┘
                             │ 调用
                    ┌────────▼─────────┐
                    │ Operator          │
                    │  ::Run()          │
                    └────────┬─────────┘
                             │ 直接调用
                    ┌────────▼─────────┐
                    │ resolved_kernel_  │
                    │  .fn(ctx)         │
                    └──────────────────┘
```

---

## 7. 改进建议

### 7.1 优先级排序

| 优先级 | 问题 | 建议 | 影响 | 实现难度 |
|--------|------|------|------|---------|
| 🔴 P0 | `OperatorPtr` 的 const 语义依赖隐式调用顺序 | 在基类文档中明确声明 "Prepare() 必须在 OperatorPtr 构造前完成" | 防止重构时破坏不变量 | 低 |
| 🟡 P1 | `InferOutputShapes` 与 `CheckInputSpecs` 重复校验 | 在基类文档中声明前置条件，移除 `InferOutputShapes` 中的重复校验 | 减少代码重复 | 低 |
| 🟡 P1 | `FunctionOperator::Run` 忽略 bindings | 在文档中明确声明 FunctionOperator 不支持运行时张量绑定 | 防止误用 | 低 |
| 🟡 P1 | `Prepare()` 非 const 与 `OperatorPtr` const 的隐式约束 | 同 P0，通过文档明确 | 同 P0 | 低 |
| 🟠 P2 | `ComputeWorkspaceRequirement` 有默认实现 | 改为纯虚函数，强制每个子类显式声明 | 防止遗漏 | 低 |
| 🟠 P2 | `OperatorContext` 含未使用字段 | 移除未使用字段，遵循 YAGNI | 减少接口复杂度 | 低 |
| 🟠 P2 | `OperatorName` 遗留类 | 确认无使用方后删除 | 减少死代码 | 低 |
| 🟠 P2 | `std::to_string` 在 noexcept 函数中 | 改用不抛异常的整数转字符串方法 | noexcept 安全 | 低 |
| 🟠 P2 | 栈上 params 指针 | 当前安全，异步 kernel 时需重新设计 | 未来风险 | 中 |
| 🟢 P3 | `TensorSpec` 缺少 `operator==` | 添加比较运算符 | 便于测试 | 低 |
| 🟢 P3 | `InferenceResult` 缺少 `operator==` | 同上 | 同上 | 低 |

### 7.2 推荐实施顺序

1. **短期**（可立即实施）：
   - 在 `Operator` 基类文档中明确生命周期约束（5 分钟）
   - 在 `FunctionOperator` 文档中声明不支持张量绑定（5 分钟）
   - 移除 `InferOutputShapes` 中的重复校验（30 分钟）

2. **中期**（随新算子接入时实施）：
   - 将 `ComputeWorkspaceRequirement` 改为纯虚函数
   - 移除 `OperatorContext` 中未使用字段
   - 删除 `OperatorName` 遗留类

3. **长期**（Phase 2 规划）：
   - 解决 `std::to_string` 在 noexcept 中的问题
   - 为异步 kernel 场景重新设计 `kernel_params` 传递机制

---

## 8. 总结

### 8.1 整体评价

Operator 语义层的设计整体成熟，**生命周期契约**、**InferenceResult 双输出**、**FunctionOperator 降级**、**AM_REGISTER_OPERATOR 自动注册** 都是优秀的架构决策。核心热路径（`Operator::Run → resolved_kernel_.fn(ctx)`）实现了零间接调用，满足推理引擎的性能要求。

### 8.2 关键指标

| 指标 | 评估 |
|------|------|
| 接口设计 | ★★★★☆ — 虚接口完整，但 ComputeWorkspaceRequirement 应为纯虚 |
| 生命周期管理 | ★★★★★ — 严格契约，编译期保护 |
| 错误处理 | ★★★★★ — 全链路 Status，错误消息含上下文 |
| 代码复用 | ★★★☆☆ — InferOutputShapes 与 CheckInputSpecs 重复校验 |
| 可扩展性 | ★★★★★ — 新增算子只需实现子类 + 一行宏 |
| 文档完整性 | ★★★★☆ — 接口注释清晰，但隐式约束未文档化 |
| noexcept 安全 | ★★★★☆ — Run() 标记 noexcept，但 std::to_string 有隐患 |

### 8.3 核心结论

主要问题集中在 **接口冗余**（`InferOutputShapes` 与 `CheckInputSpecs` 重复校验）和 **隐式约束**（`OperatorPtr` 的 const 语义依赖调用顺序）。这些问题不会在当前使用模式下导致错误，但增加了重构风险和维护成本。

建议优先处理 P1 的重复校验问题（通过文档明确前置条件），P2 的 `ComputeWorkspaceRequirement` 改为纯虚函数可在新增算子时一并处理。其余问题属于技术债，可在后续迭代中逐步清理。
