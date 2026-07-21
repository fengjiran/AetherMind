# Shape Inference 模块系统接入方案

## 现状：已完成的接入

shape_inference 模块的核心组件已经接入系统的以下环节：

### 数据流全景

```
ModelGraph::AddNode
  → AnalyzeOperator(op_type, params, input_specs)
    → detail::Analyze*() 使用 InferBroadcastShape / SymbolicShape 推导
    → 返回 InferenceResult { outputs: vector<TensorSpec>, runtime_checks: vector<ShapeConstraint> }
  → GraphNode.runtime_checks 存储
  → GraphValue.spec 存储推导出的 TensorSpec

OptimizeModelGraph (pass pipeline)
  → GraphRewriteSession 重放 AnalyzeOperator 验证替换节点

LowerModelGraph
  → GraphNode.runtime_checks → ExecutionPlanNodeSpec.runtime_checks (line 151)

ExecutionPlanBuilder::Build
  → ExecutionPlanNodeSpec.runtime_checks → ExecutionStep.runtime_checks

LayerRunner::RunStep (runtime)
  → ValidateShapeConstraints(step.runtime_checks, inputs, outputs)
  → 在 kernel 执行前校验所有 deferred 约束
```

### 已接入的组件

| 组件 | 头文件 | 接入点 |
|------|--------|--------|
| ShapeSymbol / SymbolicShape | `shape_symbol.h` | TensorSpec、所有 Analyze* 函数 |
| TensorSpec | `tensor_spec.h` | GraphValue.spec、ExecutionStep |
| InferBroadcastShape | `broadcast.h` | AnalyzeAdd、AnalyzeSiluMul、AnalyzeElementwiseMul |
| ShapeConstraint 变体 | `shape_constraint.h` | GraphNode.runtime_checks、ExecutionStep |
| EvaluateShapeConstraint | `shape_constraint_evaluator.h` | LayerRunner 运行时验证 |
| ValidateShapeConstraints | `shape_constraint_evaluator.h` | LayerRunner::RunStep |

### 未接入的组件

| 组件 | 头文件 | 设计意图 |
|------|--------|----------|
| **SymbolConstraintSolver** | `symbol_constraint_solver.h` | 编译期 union-find 等式求解，消除冗余 runtime_checks |

---

## 剩余缺口：SymbolConstraintSolver 接入

### 设计意图（来自头文件注释）

> Plan-time symbolic equality solver using union-find.
> Designed for the graph compilation phase (plan time), not for runtime.
> Proves equality facts that allow the builder to eliminate redundant runtime shape checks.

### 接入方案

#### 接入位置：OptimizeModelGraph 之后、LowerModelGraph 之前（或作为 lowering 的第一步）

理由：
- 此时图已经过优化，节点稳定
- 所有 GraphNode.runtime_checks 已就绪
- 可以在 lowering 前将已证明的约束消除，减少 runtime 开销

#### 具体步骤

**Step 1: 新增编译期约束求解 pass**

在 `src/model/graph/compilation/` 下新增 `shape_constraint_resolution.cpp`（或作为 lowering 内部步骤）：

```cpp
// 伪代码
StatusOr<ResolvedShapeFacts> ResolveShapeConstraints(const ModelGraph& graph) {
    SymbolConstraintSolver solver;

    // 1. 收集所有节点的 DimEqualConstraint，喂入 solver
    for (const GraphNode& node : graph.GetNodes()) {
        for (const ShapeConstraint& check : node.runtime_checks) {
            if (auto* eq = std::get_if<DimEqualConstraint>(&check.condition)) {
                // 解析 DimLocator → 找到对应 GraphValue 的 SymbolicShape[dim_index]
                // solver.AddEqual(lhs_symbol, rhs_symbol)
            }
        }
    }

    // 2. 利用 solver 传播静态绑定
    //    如果某个 symbolic dim 被绑定到 static value，更新 GraphValue.spec

    // 3. 重新评估每个 runtime_check：
    //    如果 solver 已证明 kSatisfied → 从 runtime_checks 中移除
    //    如果 solver 发现 kViolated → 编译期报错（fail fast）
    //    如果仍为 kDeferred → 保留到 runtime

    return resolved_facts;
}
```

**Step 2: 在 CompileModelGraph 管线中调用**

```
ModelGraph
  → OptimizeModelGraph
  → [NEW] ResolveShapeConstraints  ← SymbolConstraintSolver 接入点
  → LowerModelGraph
  → LoweredGraph
```

或者作为 `LowerModelGraph` 内部的第一个步骤（不改变公开 API）。

**Step 3: 收益**

- 编译期发现静态维度冲突（如 hidden_size=4096 的 activation 接 hidden_size=2048 的 weight）
- 消除已证明的 runtime_checks，减少 LayerRunner 热路径开销
- 为后续 workspace 精确规划提供静态 shape 信息

#### 替代方案：作为 Optimization Pass

将 SymbolConstraintSolver 包装为一个 graph pass（如 `ShapeConstraintResolutionPass`），注册到 O1/O2 pipeline 中。

优点：符合现有 pass 架构，可配置开关
缺点：需要修改 GraphValue.spec（当前 pass 通过 GraphRewriteSession 操作，但 spec 更新需要 AnalyzeOperator 重放）

#### 推荐方案

**作为 LowerModelGraph 的内部前置步骤**（非独立 pass），理由：
1. 不改变 ModelGraph 的不可变语义（lowering 本身是只读消费 graph）
2. 直接在生成 ExecutionPlanNodeSpec 时过滤已证明的 runtime_checks
3. 最小侵入性，不需要修改 pass pipeline 注册机制
4. 编译期冲突检测可以在 lowering 阶段 fail fast

---

## 测试策略

- 单元测试：`tests/unit/shape_inference/test_symbol_constraint_solver.cpp`（已存在）
- 集成测试：在 `tests/unit/model/graph/compilation/test_graph_lowering.cpp` 中增加用例，验证：
  - 两个相同 static dim 的 DimEqualConstraint 被消除
  - 静态冲突在 lowering 阶段报错
  - 无法证明的约束仍保留到 runtime

---

## 总结

shape_inference 模块的**核心数据流已完整接入**（图构建 → 优化 → lowering → 运行时验证）。唯一的剩余工作是将 `SymbolConstraintSolver` 接入编译管线，作为 lowering 的前置约束求解步骤，实现编译期约束消除和冲突检测。这是一个增量改进，不影响现有架构。
