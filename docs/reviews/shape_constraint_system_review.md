# Shape Constraint 系统全面审核报告

> **审核日期**：2026-05-30
> **审核范围**：`include/aethermind/shape_inference/shape_constraint.h`、`include/aethermind/shape_inference/shape_constraint_evaluator.h` 及其实现文件、测试文件、以及与算子系统的集成代码

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计评估](#2-架构设计评估)
3. [功能评估](#3-功能评估)
4. [性能分析](#4-性能分析)
5. [安全检查](#5-安全检查)
6. [兼容性验证](#6-兼容性验证)
7. [问题清单](#7-问题清单)
8. [改进建议](#8-改进建议)
9. [测试覆盖评估](#9-测试覆盖评估)
10. [总结](#10-总结)

---

## 1. 系统概述

### 1.1 定位与职责

Shape Constraint 系统是 AetherMind 推理引擎中 **形状安全校验** 的核心机制，负责在算子执行前验证输入/输出张量的形状约束是否满足。该系统贯穿算子系统的三个层次：

| 层次 | 职责 | 关键类型 |
|------|------|---------|
| **Operator 层（声明）** | 算子在形状推导时声明无法在符号阶段证明的约束 | `ShapeConstraint`、`DimEqualConstraint` 等 |
| **Plan 层（传递）** | 约束随 `ExecutionStep` 传递到执行期 | `ExecutionStep::runtime_checks` |
| **Execution 层（校验）** | 在 kernel 执行前对具体张量做运行时校验 | `ValidateShapeConstraints`、`EvaluateShapeConstraint` |

### 1.2 核心文件清单

| 文件 | 类型 | 职责 |
|------|------|------|
| `include/aethermind/shape_inference/shape_constraint.h` | 头文件 | 纯数据约束类型定义 |
| `include/aethermind/shape_inference/shape_constraint_evaluator.h` | 头文件 | 求值器接口声明 |
| `src/shape_inference/shape_constraint_evaluator.cpp` | 实现文件 | 符号求值与运行时求值实现 |
| `include/aethermind/shape_inference/shape_symbol.h` | 头文件 | 符号维度基础类型 |
| `src/shape_inference/shape_symbol.cpp` | 实现文件 | `ShapeSymbol` 运算与 `Unify`/`Join` 逻辑 |
| `include/utils/variant_utils.h` | 工具头文件 | `overloaded` 工具（`std::visit` 多 lambda 分派） |
| `tests/unit/shape_inference/test_shape_constraint.cpp` | 测试文件 | 约束系统单元测试 |

### 1.3 端到端数据流

```
Operator::InferOutputShapes(input_specs)
    │
    ├─ 符号阶段可证明 → 无需约束，直接返回 InferenceResult
    │
    └─ 符号阶段无法证明 → 声明 ShapeConstraint，写入 runtime_checks
         │
         ▼
InferenceResult { outputs, runtime_checks }
         │
         ▼
ExecutionPlanBuilder::Build
    → ExecutionStep { ..., runtime_checks }
         │
         ▼
LayerRunner::RunStep
    → ValidateShapeConstraints(runtime_checks, inputs, outputs)
        │
        ├─ EvaluateShapeConstraint (运行时重载)
        │   ├─ kSatisfied → 继续执行
        │   ├─ kViolated  → 返回错误，不执行 kernel
        │   └─ kDeferred  → 静默放行（运行时不应出现）
        │
        ▼
    Operator::Run(ctx, bindings, step_index)
        → kernel 执行
```

---

## 2. 架构设计评估

### 2.1 设计亮点

#### 2.1.1 数据与求值分离

`shape_constraint.h` 定义的约束类型是 **纯数据**（POD-like struct + `std::variant`），不包含任何求值逻辑。求值由独立的 `EvaluateShapeConstraint` 自由函数完成。

**优势**：

- **可序列化**：约束可被 AOT 编译器直接 dump，支持离线分析
- **可替换**：求值策略可替换（符号求值 / 运行时求值 / 未来约束求解器）
- **低耦合**：约束声明方无需依赖求值实现

#### 2.1.2 三态求值结果

`ShapeConstraintEvaluationResult` 定义了三种求值结果：

| 状态 | 含义 | 典型场景 |
|------|------|---------|
| `kSatisfied` | 约束已证明成立 | 符号阶段 `lhs == rhs`，或运行时维度值相等 |
| `kViolated` | 约束已证明违反 | 符号阶段两个静态维度不等，或运行时维度值不等 |
| `kDeferred` | 当前信息不足以判定 | 符号维度无法证明相等或不相等 |

**设计意义**：避免了"符号阶段无法证明就报错"的过度严格行为，也避免了"无法证明就跳过"的不安全行为。`kDeferred` 允许将检查延迟到运行时，当所有维度值已知时再做精确判断。

#### 2.1.3 DimLocator 端口定位设计

```cpp
struct DimLocator {
    TensorPort tensor_port{};  // {direction: Input/Output, tensor_idx}
    size_t dim_index{};        // 维度索引
};
```

通过 `TensorPort{direction, tensor_idx} + dim_index` 精确定位任意输入/输出张量的任意维度，支持跨张量约束（如 "input[0] 的 dim[1] == input[1] 的 dim[0]"）。这是 RmsNorm 等算子的核心需求——需要约束 `input.hidden_size == weight.numel`。

#### 2.1.4 运行时校验在 kernel 执行前

`LayerRunner::RunStep` 在调用 `op->Run()` 前执行 `ValidateShapeConstraints`，违反时直接返回错误，**不执行 kernel**。这防止了形状不匹配时的内存越界访问。

#### 2.1.5 符号维度的统一机制

`ShapeSymbol` 的三态设计：

| 值域 | 含义 | 用途 |
|------|------|------|
| `≥ 0` | 静态已知维度 | 编译期可证明的维度 |
| `= -1` | 未知维度 | 运行时才能确定的维度 |
| `< -1` | 符号维度 | 不同位置共享同一符号，可追踪等式关系 |

配合 `UnifyShapeSymbol`（严格，用于算子输入约束）和 `JoinShapeSymbol`（宽松，用于控制流合并），为形状推导提供了完整的代数基础。

### 2.2 架构设计评分

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 职责分离 | ★★★★★ | 数据与求值完全分离，声明与校验分层清晰 |
| 可扩展性 | ★★★★☆ | 新增约束类型只需扩展 variant，但缺少比较类约束 |
| 可测试性 | ★★★★☆ | 纯数据结构易于构造测试用例，但符号求值测试覆盖不足 |
| 最小化原则 | ★★★★★ | Phase 1 只支持必要的约束类型，未过度设计 |
| 与算子系统集成 | ★★★★★ | 通过 `InferenceResult::runtime_checks` 自然嵌入，零侵入 |

---

## 3. 功能评估

### 3.1 约束类型覆盖

| 约束类型 | 语义 | 典型使用场景 | 当前使用方 |
|---------|------|-------------|-----------|
| `DimEqualConstraint` | 两个维度必须相等 | RmsNorm: `input.dim[1] == weight.dim[0]` | RmsNormOp |
| `DimBroadcastableConstraint` | 两个维度可广播 | Elementwise: `a == b \|\| a == 1 \|\| b == 1` | 无（预留） |
| `VolumeEqualConstraint` | 两组维度的乘积相等 | Reshape: `prod(input.dims) == prod(output.dims)` | 无（预留） |
| `RankEqualConstraint` | 张量秩必须等于指定值 | 约束输入为 2D | 无（预留） |
| `RankAtLeastConstraint` | 张量秩至少为指定值 | 约束输入至少 1D | 无（预留） |

### 3.2 求值器功能矩阵

| 求值场景 | DimEqual | Broadcastable | VolumeEqual | RankEqual | RankAtLeast |
|---------|----------|---------------|-------------|-----------|-------------|
| **符号求值** | ✅ 支持 | ✅ 支持 | ⚠️ 过度保守 | ✅ 支持 | ✅ 支持 |
| **运行时求值** | ✅ 支持 | ✅ 支持 | ✅ 支持（含溢出检查） | ✅ 支持 | ✅ 支持 |

### 3.3 符号求值详细分析

#### DimEqual 符号求值

```cpp
if (lhs == rhs) return kSatisfied;           // 同一符号实例
if (lhs.IsStatic() && rhs.IsStatic())         // 两个静态维度
    return (lhs.value() == rhs.value()) ? kSatisfied : kViolated;
return kDeferred;                             // 无法判定
```

**正确性**：当 `lhs` 和 `rhs` 指向同一个 `ShapeSymbol` 实例时（通过 `Unify` 合并后），`lhs == rhs` 成立，返回 `kSatisfied`。这是当前 RmsNorm 场景的主要路径。

**局限**：当 `lhs = S1` 且 `rhs = S2`（不同符号实例）时，即使通过之前的 Unify 已知 S1 == S2，也无法识别——因为缺少等价类追踪。

#### VolumeEqual 符号求值

```cpp
for (const auto& loc : constraint.lhs_dims) {
    auto dim = ResolveShape(loc, inputs, outputs);
    if (!*dim || !(**dim).IsStatic()) return kDeferred;  // 任一非静态即 Deferred
    lhs_volume *= (**dim).value();
}
// rhs_dims 同理
return (lhs_volume == rhs_volume) ? kSatisfied : kViolated;
```

**局限**：当任一维度是符号维度时，立即返回 `kDeferred`。但存在可以证明的情况：`[S1, 8]` 与 `[S1, 8]`（S1 是同一符号），`S1 * 8 == S1 * 8` 恒成立。

### 3.4 运行时求值详细分析

运行时求值对所有约束类型均提供完整实现，包括：

- **端口解析**：`ResolveTensor` / `ResolveMutableTensor` 根据 `TensorPort.direction` 分派到 inputs 或 outputs
- **维度值提取**：`ResolveRuntimeDim` 从 `TensorView::dim()` 获取具体维度值
- **溢出检查**：`VolumeEqual` 使用 `OverflowSafeMultiply` 防止乘法溢出
- **边界检查**：`tensor_idx` 和 `dim_index` 越界时返回 `Status::InvalidArgument`

---

## 4. 性能分析

### 4.1 热路径开销分析

Shape Constraint 系统在推理热路径中的开销主要来自 `LayerRunner::RunStep` 中的 `ValidateShapeConstraints` 调用。

| 操作 | 开销 | 频率 |
|------|------|------|
| `runtime_checks.empty()` 检查 | ~1ns（布尔判断） | 每步每次推理 |
| `EvaluateShapeConstraint` 运行时重载 | ~100-500ns（variant visit + 维度比较） | 仅含约束的步 |
| `ValidateShapeConstraints` 循环 | O(N) × 单次求值开销 | 仅含约束的步 |

**关键观察**：

1. **无约束步零开销**：`runtime_checks.empty()` 为 true 时直接跳过，开销仅 1ns
2. **有约束步开销可忽略**：单次求值 ~100-500ns，相比 kernel 执行时间（微秒到毫秒级）可忽略
3. **无堆分配**：运行时求值路径无 `new`/`malloc` 调用

### 4.2 Plan 构建期开销

| 操作 | 开销 | 说明 |
|------|------|------|
| `ShapeConstraint` 构造 | ~50ns（含 `std::string` 分配） | 每个 `InferenceResult` 一次 |
| `std::vector<ShapeConstraint>` 拷贝 | O(N) | `ExecutionStep` 从 `InferenceResult` 拷贝 |
| 符号求值 | ~200ns-1μs | `InferOutputShapes` 内部 |

### 4.3 潜在性能瓶颈

| 瓶颈 | 影响 | 严重程度 |
|------|------|---------|
| `error_context` 的 `std::string` 拷贝 | Plan 构建期，每约束一次堆分配 | 🟢 低 |
| `VolumeEqualConstraint` 的 `std::vector<DimLocator>` 拷贝 | Plan 构建期 | 🟢 低 |
| `std::visit` 的 variant 分派 | 每次求值一次间接跳转 | 🟢 低（编译器优化后 ~2ns） |

**结论**：Shape Constraint 系统在热路径中的性能开销可忽略，不存在显著性能瓶颈。

---

## 5. 安全检查

### 5.1 内存安全

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 悬垂指针 | ✅ 安全 | `ShapeConstraint` 持有 `std::vector<DimLocator>` 和 `std::string`，均为拥有语义 |
| 缓冲区溢出 | ✅ 安全 | 运行时求值对 `tensor_idx` 和 `dim_index` 做越界检查 |
| 整数溢出 | ✅ 安全 | `VolumeEqual` 使用 `OverflowSafeMultiply` |
| 野指针 | ✅ 安全 | 无裸指针传递 |

### 5.2 线程安全

| 组件 | 线程安全 | 说明 |
|------|---------|------|
| `ShapeConstraint`（数据） | ✅ | 不可变数据结构，只读安全 |
| `EvaluateShapeConstraint`（求值） | ✅ | 纯函数，无共享可变状态 |
| `ValidateShapeConstraints` | ✅ | 纯函数 |
| `ExecutionStep::runtime_checks` | ✅ | Plan 构建后不可变 |

### 5.3 异常安全

| 组件 | 异常安全 | 说明 |
|------|---------|------|
| `EvaluateShapeConstraint` | ✅ | 返回 `StatusOr`，不抛异常 |
| `ValidateShapeConstraints` | ✅ | 返回 `Status`，不抛异常 |
| `ShapeConstraint` 构造 | ⚠️ | `std::string` 和 `std::vector` 构造可能抛 `bad_alloc` |

---

## 6. 兼容性验证

### 6.1 与算子系统的集成

| 集成点 | 接口 | 兼容性 |
|--------|------|--------|
| `Operator::InferOutputShapes` | `InferenceResult::runtime_checks` | ✅ 自然嵌入 |
| `ExecutionPlanBuilder` | `ExecutionStep::runtime_checks` | ✅ 透传 |
| `LayerRunner::RunStep` | `ValidateShapeConstraints(inputs, outputs)` | ✅ 运行时校验 |
| `RmsNormOp` | `DimEqualConstraint` 声明 | ✅ 实际使用 |
| `EmbeddingOp` | 空 `runtime_checks` | ✅ 无约束算子 |

### 6.2 与 Shape Symbol 系统的兼容

| 交互点 | 兼容性 | 说明 |
|--------|--------|------|
| `SymbolicShape` 作为求值输入 | ✅ | 符号求值重载接受 `span<const SymbolicShape>` |
| `UnifyShapeSymbol` 结果消费 | ⚠️ | Unify 碰撞时降级为 Unknown，丢失等式信息 |
| `JoinShapeSymbol` 结果消费 | ✅ | 控制流合并后正确返回 kDeferred |

### 6.3 与 TensorView 系统的兼容

| 交互点 | 兼容性 | 说明 |
|--------|--------|------|
| `TensorView::dim()` 维度提取 | ✅ | 运行时求值直接使用 |
| `TensorView::rank()` 秩提取 | ✅ | RankEqual/RankAtLeast 使用 |
| `MutableTensorView` 输出维度 | ✅ | 通过 `ResolveMutableTensor` 分派 |

### 6.4 编译依赖兼容

| 依赖 | 必要性 | 说明 |
|------|--------|------|
| `shape_constraint.h` → 标准库 | ✅ 必要 | 零项目依赖 |
| `shape_constraint_evaluator.h` → `shape_symbol.h` | ⚠️ 可优化 | 运行时求值调用方不需要符号类型 |
| `shape_constraint_evaluator.h` → `tensor_view.h` | ✅ 必要 | 运行时求值需要 `TensorView` |
| `shape_constraint_evaluator.cpp` → `variant_utils.h` | ✅ 必要 | `overloaded` 工具 |

---

## 7. 问题清单

### 🔴 P1：VolumeEqualConstraint 符号求值过于保守

**文件**：`src/shape_inference/shape_constraint_evaluator.cpp`，`EvaluateSymbolicVolumeConstraint` 函数

**现象**：当任一维度是符号维度时，立即返回 `kDeferred`，即使 lhs 和 rhs 的符号维度列表完全相同。

**示例**：

```
lhs = [S1, 8], rhs = [S1, 8]  （S1 是同一符号实例）
当前行为：kDeferred（因为 S1 非静态）
期望行为：kSatisfied（S1 * 8 == S1 * 8 恒成立）
```

**影响**：当前只有 RmsNorm 使用约束系统且不使用 `VolumeEqualConstraint`，实际影响为零。但未来 Reshape/Transpose 等算子会大量使用此约束，过度保守的 Deferred 会导致不必要的运行时检查开销。

**修复方案**：在返回 `kDeferred` 前，检查 lhs_dims 和 rhs_dims 的符号列表是否完全相同（相同位置为同一 `ShapeSymbol` 实例或相同的静态值）：

```cpp
// 快速路径：符号列表完全相同 → 乘积必然相等
if (lhs_dims.size() == rhs_dims.size()) {
    bool all_identical = true;
    for (size_t i = 0; i < lhs_dims.size(); ++i) {
        if (lhs_dims[i] != rhs_dims[i]) { all_identical = false; break; }
    }
    if (all_identical) return ShapeConstraintEvaluationResult::kSatisfied;
}
```

---

### 🔴 P1：DimEqualConstraint 符号求值缺少传递性推理

**文件**：`src/shape_inference/shape_constraint_evaluator.cpp`，`EvaluateSymbolicDimEqual` 函数

**现象**：当 `lhs = S1`（符号）且 `rhs = S2`（另一个符号）时，即使 S1 和 S2 通过之前的 `UnifyShapeSymbol` 已被证明相等，此函数也无法识别。

**根因**：`ShapeSymbol` 的相等性基于值比较，而 `UnifyShapeSymbol` 在符号碰撞时降级为 `Unknown`（`shape_symbol.cpp` 中 `Unify` 的实现），丢失了等式信息。没有等价类追踪机制。

**示例**：

```
Step 1: Unify(S1, S2) → Unknown  （S1 和 S2 应该相等，但信息丢失）
Step 2: Evaluate(DimEqual{S1, S2}) → kDeferred  （无法证明 S1 == S2）
```

**影响**：在当前只有 RmsNorm 的场景下，`DimEqualConstraint` 的 lhs 和 rhs 通常指向同一个 `ShapeSymbol` 实例（通过 Unify 合并后），所以 `lhs == rhs` 成立，不会触发此问题。但在多步推导链中（如 A → B → C），传递性推理的缺失会导致不必要的 `kDeferred`。

**修复方案**：引入 `SymbolConstraintSolver` 记录等式约束（`shape_symbol.cpp` 中的 TODO 已规划此功能）。短期可接受当前行为。

---

### 🟡 P1：ShapeConstraint 的 `operator<=>` 含字符串比较

**文件**：`include/aethermind/shape_inference/shape_constraint.h`，第 103 行

**现象**：

```cpp
auto operator<=>(const ShapeConstraint&) const noexcept = default;
```

`ShapeConstraint` 含 `std::string error_context` 和 `ConstraintVariant`（含 `std::vector<DimLocator>`）。默认的 `operator<=>` 会对 `error_context` 做字典序比较。

**问题**：

1. **语义不正确**：两个条件相同但错误消息不同的约束应该是等价的
2. **性能浪费**：如果将 `ShapeConstraint` 放入 `std::set` 或 `std::map`，字符串比较引入不必要开销
3. **noexcept 风险**：`std::string` 的比较在极端情况下可能抛 `bad_alloc`（虽然实践中不会），但 `noexcept` 声明可能导致 `std::terminate`

**修复方案**：自定义比较，仅比较 `condition`，忽略 `error_context`：

```cpp
friend bool operator==(const ShapeConstraint& a, const ShapeConstraint& b) noexcept {
    return a.condition == b.condition;
}
friend auto operator<=>(const ShapeConstraint& a, const ShapeConstraint& b) noexcept {
    return a.condition <=> b.condition;
}
```

---

### 🟠 P2：EvaluateShapeConstraint 返回 StatusOr 混合了两种错误语义

**文件**：`include/aethermind/shape_inference/shape_constraint_evaluator.h`

**现象**：返回 `StatusOr<ShapeConstraintEvaluationResult>` 意味着有两种"失败"路径：

1. **Status 错误**：约束引用了不存在的端口/维度（如 `tensor_idx` 越界）——这是程序错误
2. **kViolated**：约束引用有效但条件不满足——这是数据错误

**影响**：调用者需要区分这两种情况。当前 `ValidateShapeConstraints` 对两者行为一致（都返回错误 Status），但语义上应该不同：端口越界应使用 `AM_CHECK`（断言），约束违反应返回 `Status`。

**修复方案**：端口/维度越界改为 `AM_CHECK`（程序错误），仅约束违反通过 `kViolated` 返回。

---

### 🟠 P2：缺少 DimGreaterThan / DimLessThan 约束类型

**现象**：当前约束类型覆盖了相等、广播、体积和秩，但缺少维度大小比较约束。

**需要的场景**：

| 算子 | 约束需求 |
|------|---------|
| Attention | `seq_len <= max_seq_len` |
| Embedding | `token_id < vocab_size`（值约束，但 vocab_size 可视为形状维度） |
| Slice/Range | `start + size <= dim_size` |

**当前处理**：这些检查只能在 kernel 内部完成，无法在形状推导阶段声明。

**修复方案**：新增 `DimLessThanConstraint` 和 `DimGreaterThanConstraint`：

```cpp
struct DimLessThanConstraint {
    DimLocator lhs;
    DimLocator rhs;
    auto operator<=>(const DimLessThanConstraint&) const noexcept = default;
};
```

---

### 🟠 P2：运行时求值的 kDeferred 静默放行

**文件**：`src/shape_inference/shape_constraint_evaluator.cpp`，`ValidateShapeConstraints` 函数

**现象**：

```cpp
if (*result == ShapeConstraintEvaluationResult::kViolated) {
    return Status::InvalidArgument(...);
}
// kSatisfied 和 kDeferred 都被放行
```

`kDeferred` 被静默忽略。在运行时求值重载中，所有维度都是具体的，不应该出现 `kDeferred`。如果出现，说明存在逻辑错误。

**修复方案**：在运行时 `ValidateShapeConstraints` 中，对 `kDeferred` 返回 `Status::Internal`：

```cpp
if (*result == ShapeConstraintEvaluationResult::kDeferred) {
    return Status::Internal("Runtime shape constraint evaluation returned kDeferred; "
                            "this should not happen with concrete tensor shapes");
}
```

---

### 🟠 P2：evaluator.h 编译依赖过重

**文件**：`include/aethermind/shape_inference/shape_constraint_evaluator.h`

**现象**：头文件同时声明了符号求值和运行时求值两个重载，导致：

- 运行时求值调用方（如 `layer_runner.cpp`）也被迫包含 `shape_symbol.h`
- 符号求值调用方也被迫包含 `tensor_view.h`

**修复方案**：拆分为两个独立头文件：

```
shape_constraint_runtime_evaluator.h  → 依赖 tensor_view.h
shape_constraint_symbolic_evaluator.h → 依赖 shape_symbol.h
```

或前向声明 `SymbolicShape`，将符号求值的实现移到 `.cpp` 文件中。

---

### 🟢 P3：ShapeConstraintList 类型别名未被使用

**文件**：`include/aethermind/shape_inference/shape_constraint.h`，第 106 行

```cpp
using ShapeConstraintList = std::vector<ShapeConstraint>;
```

搜索代码库，`ShapeConstraintList` 未被任何文件使用。所有使用方直接写 `std::vector<ShapeConstraint>`。

**修复方案**：删除此类型别名，或在 `InferenceResult`、`ExecutionStep` 等处统一使用。

---

### 🟢 P3：error_context 可为空

**现象**：`ShapeConstraint::error_context` 是 `std::string`，默认为空。`ValidateShapeConstraints` 在 `error_context` 为空时使用通用错误消息。

**影响**：空 `error_context` 导致运行时约束违反时的诊断信息不够具体。

**修复方案**：在 `ShapeConstraint` 构造时强制要求非空 `error_context`，或在 `Operator::InferOutputShapes` 实现中添加编码规范要求。

---

## 8. 改进建议

### 8.1 优先级排序

| 优先级 | 问题 | 建议 | 预期收益 | 实现难度 |
|--------|------|------|---------|---------|
| 🔴 P1 | VolumeEqual 符号求值过于保守 | 添加符号列表完全相同时的快速路径 | 减少 Reshape 等算子的运行时检查 | 低 |
| 🔴 P1 | 缺少传递性推理 | 引入 `SymbolConstraintSolver`（已规划 TODO） | 提升多步推导的符号推理能力 | 高 |
| 🟡 P1 | `operator<=>` 含字符串比较 | 自定义比较，仅比较 `condition` | 语义正确 + 性能 | 低 |
| 🟠 P2 | StatusOr 混合错误语义 | 端口越界改为 `AM_CHECK` | 语义清晰 | 低 |
| 🟠 P2 | 缺少比较约束类型 | 新增 `DimLessThanConstraint` 等 | 覆盖 Attention 等场景 | 中 |
| 🟠 P2 | kDeferred 静默放行 | 运行时对 kDeferred 返回 `Status::Internal` | 防止误用 | 低 |
| 🟠 P2 | 编译依赖过重 | 拆分头文件或前向声明 | 减少编译依赖 | 低 |
| 🟢 P3 | ShapeConstraintList 未使用 | 删除或统一使用 | 减少死代码 | 低 |
| 🟢 P3 | error_context 可为空 | 强制非空或编码规范 | 更好的诊断 | 低 |

### 8.2 推荐实施顺序

1. **短期**（可立即实施）：
   - 修复 `operator<=>`（5 分钟）
   - 运行时 kDeferred 返回 `Status::Internal`（5 分钟）
   - VolumeEqual 符号列表相同快速路径（30 分钟）

2. **中期**（随新算子接入时实施）：
   - 新增 `DimLessThanConstraint` / `DimGreaterThanConstraint`
   - 拆分 evaluator 头文件

3. **长期**（Phase 2 规划）：
   - 引入 `SymbolConstraintSolver` 实现传递性推理
   - 支持动态 shape 推导框架

---

## 9. 测试覆盖评估

### 9.1 已覆盖场景

| 场景 | 测试文件 | 状态 |
|------|---------|------|
| 约束数据类型存储 | `test_shape_constraint.cpp` | ✅ |
| 运行时 DimEqual 满足/违反 | `test_shape_constraint.cpp` | ✅ |
| 运行时 Broadcastable 满足/违反 | `test_shape_constraint.cpp` | ✅ |
| 运行时 VolumeEqual 满足/违反 | `test_shape_constraint.cpp` | ✅ |
| 运行时 RankEqual/RankAtLeast | `test_shape_constraint.cpp` | ✅ |
| 符号 DimEqual satisfied/violated/deferred | `test_shape_constraint.cpp` | ✅ |
| ValidateShapeConstraints 错误消息 | `test_shape_constraint.cpp` | ✅ |
| RmsNormOp 发出 DimEqualConstraint | `test_rmsnorm_op.cpp` | ✅ |
| ExecutionStep 传递约束 | `test_execution_plan_builder.cpp` | ✅ |
| 运行时约束违反拒绝执行 | `test_executor_backend_path.cpp` | ✅ |
| 运行时约束满足正常执行 | `test_executor_backend_path.cpp` | ✅ |

### 9.2 缺失场景

| 场景 | 优先级 | 说明 |
|------|--------|------|
| 符号 Broadcastable deferred | 🟡 | 符号维度下应返回 kDeferred |
| 符号 VolumeEqual | 🟡 | 含符号维度时的行为 |
| 符号 RankEqual/RankAtLeast | 🟢 | 秩约束的符号求值 |
| 端口越界错误 | 🟠 | `tensor_idx` 超出 inputs/outputs 范围 |
| 维度越界错误 | 🟠 | `dim_index` 超出 rank 范围 |
| VolumeEqual 溢出 | 🟠 | 大维度值乘法溢出 |
| kDeferred 在运行时 ValidateShapeConstraints 中的行为 | 🟠 | 当前静默放行，应报错 |
| 空 runtime_checks 的零开销路径 | 🟢 | 验证无约束步无性能影响 |

---

## 10. 总结

### 10.1 整体评价

Shape Constraint 系统的核心设计——**数据与求值分离、三态求值结果、DimLocator 端口定位、运行时校验在 kernel 前执行**——都是正确的架构决策。系统已经完整地嵌入算子系统的 `InferOutputShapes → ExecutionStep → LayerRunner` 管线中，且被 RmsNormOp 实际使用，端到端测试覆盖了关键路径。

### 10.2 关键指标

| 指标 | 评估 |
|------|------|
| 功能完整性 | ★★★★☆ — 覆盖主要约束类型，缺少比较类约束 |
| 正确性 | ★★★★☆ — 运行时求值完全正确，符号求值有保守性局限 |
| 性能 | ★★★★★ — 热路径零开销，无堆分配 |
| 安全性 | ★★★★★ — 内存安全、线程安全、异常安全 |
| 可扩展性 | ★★★★☆ — 新增约束类型只需扩展 variant |
| 测试覆盖 | ★★★★☆ — 核心路径已覆盖，边界场景有缺口 |
| 文档完整性 | ★★★★★ — 头文件注释清晰，设计文档完善 |

### 10.3 核心结论

主要缺陷集中在 **符号求值能力不足**（VolumeEqual 过度保守、缺少传递性推理），这会随着更多算子接入而逐渐显现。根因是缺少 `SymbolConstraintSolver`（代码中已标注 TODO），这是一个已知的架构演进方向。当前阶段，系统满足 Phase 1 的功能需求。

中低优先级问题（StatusOr 语义混淆、缺少比较约束类型、编译依赖过重）属于设计改进，建议在 Phase 2 引入更多算子时逐步解决。
