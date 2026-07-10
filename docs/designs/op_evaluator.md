在深度学习编译器（如 MLIR、XLA、ONNX Runtime、TVM）中，常量折叠的底层支撑通常被称为 **Constant Evaluator（常量评估器）** 或 **Folder**。

设计这个组件时，最容易犯的错误是**直接复用推理引擎的底层运行态 Kernel（如 CPU/CUDA 算子）**。这会导致编译期依赖极其沉重的运行时环境（比如需要初始化 CUDA Context、线程池，或者跨平台交叉编译时直接崩溃）。

主流框架（如 XLA 的 HLO Evaluator 和 MLIR 的 Fold 接口）的标准做法是：**为编译期单独实现一套纯 C++ 的、完全同步的、数学等价的“参考算子（Reference Kernels）”**。

---

### 一、架构设计原则

1. **编译期与运行期解耦**：Evaluator 不复用运行时 Kernel，不依赖 CUDA Context、线程池、Backend、Allocator 或执行计划。它只依赖 IR 元数据、常量字节数据和纯 C++ 参考实现。
2. **Plan 与 Evaluate 分离**：`Plan()` 负责验证输入/输出数量、dtype、shape、layout、broadcast、参数语义和成本预算；`Evaluate()` 只执行 `Plan()` 已经接受的情况。
3. **Pass 与算子语义解耦**：`ConstantFoldingPass` 不知道某个算子支持哪些 dtype、broadcast 或多输出规则。Pass 只收集图信息、构造 view、分配输出 buffer、注册常量和重定向 value。
4. **无状态与无所有权**：Evaluator 不持有输入/输出 view，不保存 graph/session 指针，不管理常量生命周期。所有内存由 Pass 分配，并通过 `GraphRewriteSession::AddConstant()` 移交给图重写系统。
5. **优雅降级**：不支持的 op、dtype、shape、layout、参数组合返回 `Status::Unimplemented`，Pass 跳过该节点并保留原图。图结构不变量破坏、buffer 尺寸不一致等应返回错误并中止 pass。
6. **确定性优先**：只有 schema 表示 deterministic、无副作用且可编译期求值的节点才允许进入 Evaluator。浮点 reference kernel 需要定义清楚舍入、近似函数和容差预期。

---

### 二、核心接口：Plan + Evaluate

Evaluator 使用“先规划、后执行”的契约。`Plan()` 是完整的可折叠性判定入口，`Evaluate()` 是纯数据计算入口。

```cpp
// const_evaluator.h
#pragma once

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/graph/graph_types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace aethermind {

struct ConstEvalPolicy {
    // 单个节点折叠后允许物化的最大总字节数。
    size_t max_output_bytes = 64U * 1024U;

    // 单个节点允许执行的最大标量元素操作量。
    // 复杂 evaluator 可用该预算估算计算成本。
    size_t max_compute_elements = 64U * 1024U;
};

struct PlannedConstOutput {
    // 输出 TensorSpec，shape 必须为 fully static（所有维度为非负整数），
    // 否则 Pass 无法分配输出 buffer 或验证不变量。
    TensorSpec spec{};
    QuantizationSpec quantization{};
    std::vector<int64_t> strides{};
    size_t nbytes = 0;
    // Evaluator 产出 debug_name 时需遵守命名约定，见「新增 evaluator 的要求」。
    std::string debug_name{};
};

struct ConstEvalPlan {
    std::vector<PlannedConstOutput> outputs{};
};

class ConstEvaluator {
public:
    virtual ~ConstEvaluator() = default;

    // 验证该 evaluator 是否接受当前节点的输入、输出、参数和成本预算。
    // 返回 Ok(plan) 表示可以折叠；返回 Unimplemented 表示不支持并应跳过；
    // 返回其他错误表示图或 pass 不变量异常，应向上传播。
    // Plan 必须保证返回的 output spec.shape 为 fully static（所有维度为非负整数），
    // 否则 Pass 无法分配输出 buffer（nbytes 和 rowsizes 无法确定）。
    AM_NODISCARD virtual StatusOr<ConstEvalPlan> Plan(
            std::span<const NodeOutputDesc> inputs,
            std::span<const NodeOutputDesc> outputs,
            const OpParams& params,
            const ConstEvalPolicy& policy) const = 0;

    // 执行编译期求值。调用方必须保证 inputs/outputs 与 Plan() 接受的
    // 元数据一致，且 outputs 的 backing storage 已分配完成。
    AM_NODISCARD virtual Status Evaluate(
            std::span<const TensorView> inputs,
            std::span<MutableTensorView> outputs,
            const OpParams& params) const = 0;
};

// 根据 OpType 查找 evaluator。查找表为静态只读结构，避免可变全局注册状态。
AM_NODISCARD const ConstEvaluator* FindConstEvaluator(OpType op_type) noexcept;

} // namespace aethermind
```

设计要点：

- `Plan()` 接收 `NodeOutputDesc`，不是裸 `TensorSpec`，因为它需要同时判断 payload、quantization、debug name 和 dtype/shape。
- `Evaluate()` 接收 `TensorView` / `MutableTensorView`，不接收 graph/session。它只能读输入字节、写输出字节。
- 多输出天然由 `std::span<MutableTensorView>` 表达。单输出、多输出、shape-only、数值型算子都使用同一个契约。
- `ConstEvalPolicy` 嵌入 `PassContext`，用户通过 `PassContext` 统一配置常量折叠行为。`Plan()` 接收的 `policy` 即从 `ctx.const_eval_policy` 取得，不独立传递。

---

### 三、Evaluator 的职责边界

每个具体 evaluator 必须在 `Plan()` 中完整验证自己需要的前提：

1. 输入数量与输出数量。
2. dtype 是否支持，以及 dtype promotion / accumulation / cast 规则。
3. shape 是否足够确定；需要静态 shape 时必须显式检查 `SymbolicShape::IsStatic()`。
4. broadcast、stride、layout、rank、axis、transpose 等参数语义。
5. 输出是否可以物化为 compile-time constant。
6. 输出字节数、计算量和临时空间需求是否满足 `ConstEvalPolicy`。
7. 数学域限制，例如除零、非法 axis、非法 reshape、溢出等。

输入 payload 为 `ConstantValue` 且 `ConstantBinding::inline_data` 的字节数与 `TensorSpec`（dtype × 静态 shape 元素数）一致是**图级 precondition**，由 Pass 在调用 `Plan()` 前保证（见 §七 图级条件 item 4-5）。`WeightValue` 虽然在推理期不可变，但不属于本 evaluator 的可折叠输入集合；权重折叠必须通过独立的权重数据解析契约提供可验证的实际字节数据，不能由 `WeightValue` 绑定本身隐式启用。evaluator 不重复检查这一条件，只验证 dtype/shape/layout 等算子语义条件。

`Evaluate()` 不应重复推导复杂规则。它可以做轻量防御性检查，但主要假设 `Plan()` 已经完成验证。这样可以避免 Pass 复制算子语义，也避免 `Evaluate()` 在写输出 buffer 时才发现尺寸或 layout 不匹配。

错误策略：

| 场景 | 返回值 | Pass 行为 |
|---|---|---|
| evaluator 不支持该 dtype / shape / layout / params | `Status::Unimplemented` | 跳过该节点 |
| 数学域不适合安全折叠（例如按语义需要保留运行时行为） | `Status::Unimplemented` | 跳过该节点 |
| 输入数量、输出数量、buffer 大小与 Plan 不一致 | `Status::InvalidArgument` 或 `Status::Internal` | 中止 pass |
| 成本预算超限 | `Status::Unimplemented` | 跳过该节点 |
| 求值成功 | `Status::Ok()` | 注册常量并重定向输出 |

---

### 四、ConstantFoldingPass 的通用流程

`ConstantFoldingPass` 只负责图级装配，不实现算子规则。

```cpp
class ConstantFoldingPass : public GraphPass {
public:
    std::string_view Name() const noexcept override {
        return "ConstantFoldingPass";
    }

    Status Run(GraphRewriteSession& session, const PassContext& ctx) override {
        const ConstEvalPolicy& policy = ctx.const_eval_policy;

        auto topo = session.GetTopologicalOrder();
        AM_RETURN_IF_ERROR(topo.status());

        for (GraphNodeId node_id : *topo) {
            auto node_or = session.GetNodeView(node_id);
            if (!node_or.ok()) {
                continue;
            }
            const GraphNodeView& node = *node_or;

            const auto schema = GetOperatorSchema(node.op_type);
            if (!schema.ok() || !IsCompileTimeEvaluable(*schema)) {
                continue;
            }

            const ConstEvaluator* evaluator = FindConstEvaluator(node.op_type);
            if (evaluator == nullptr) {
                continue;
            }

            auto input_descs = CollectInputDescs(session, node.inputs);
            AM_RETURN_IF_ERROR(input_descs.status());
            if (!AllInputsAreInlineConstantValues(*input_descs)) {
                continue;
            }

            auto output_descs = CollectOutputDescs(session, node.outputs);
            AM_RETURN_IF_ERROR(output_descs.status());

            auto plan = evaluator->Plan(*input_descs, *output_descs, node.op_params, policy);
            if (plan.status() == StatusCode::kUnimplemented) {
                continue;
            }
            AM_RETURN_IF_ERROR(plan.status());

            auto input_views = BuildInputViews(*input_descs);
            AM_RETURN_IF_ERROR(input_views.status());

            auto output_storage = AllocateOutputViews(*plan);
            AM_RETURN_IF_ERROR(output_storage.status());

            Status eval_status = evaluator->Evaluate(
                    input_views->views,
                    output_storage->views,
                    node.op_params);
            if (eval_status == StatusCode::kUnimplemented) {
                continue;
            }
            AM_RETURN_IF_ERROR(eval_status);

            // Invariant: Plan()'s output specs must match the graph's declared
            // output specs. The folded constant replaces the original value via
            // ReplaceValue, so downstream consumers must see the same dtype/shape.
            AM_DCHECK(plan->outputs.size() == node.outputs.size());

            for (size_t i = 0; i < node.outputs.size(); ++i) {
                const PlannedConstOutput& out = plan->outputs[i];
                ConstantBinding binding{
                        .name = out.debug_name,
                        .inline_data = std::move(output_storage->buffers[i]),
                };

                GraphValueId folded = session.AddConstant(
                        out.spec,
                        std::move(binding),
                        out.quantization,
                        out.debug_name);

                AM_RETURN_IF_ERROR(session.ReplaceValue(node.outputs[i], folded));
            }
        }

        return Status::Ok();
    }
};
```

关键点：

- Pass 不判断某个 op 是否支持 broadcast、某种 dtype 或多输出。这些全部交给 evaluator 的 `Plan()`。
- Pass 不主动 `RemoveNode()`。折叠后，原输出通过 `ReplaceValue()` 重定向到新常量；无消费者的原节点由管线中的 DCE 统一清理。
- Pass 只对 `Unimplemented` 降级跳过。其他错误表示图不变量或实现不变量异常，应向上传播。

---

### 五、数据收集与 View 构造

#### 1. 收集输入和输出描述

Pass 通过 `GraphRewriteSession` 查询当前 session 视图，而不是直接访问 `ModelGraph`：

```cpp
StatusOr<std::vector<NodeOutputDesc>> CollectInputDescs(
        const GraphRewriteSession& session,
        std::span<const GraphValueId> inputs) {
    std::vector<NodeOutputDesc> descs;
    descs.reserve(inputs.size());
    for (GraphValueId input : inputs) {
        auto desc = session.GetValueOutputDesc(input);
        AM_RETURN_IF_ERROR(desc.status());
        descs.push_back(std::move(*desc));
    }
    return descs;
}
```

`GetNodeView()` 返回的 inputs 已经经过 `GetResolvedValue()` 解析，因此如果上游节点在同一个 session 中被折叠成 session constant，下游节点会自然看到该 session constant 的 `ConstantValue` 描述。

#### 2. 构造 TensorView

`TensorSpec` 使用 `SymbolicShape`，而 `TensorView` / `MutableTensorView` 需要 `IntArrayView` shape 和 strides。由于 `IntArrayView` 是借用的 `std::span<const int64_t>`，Pass 必须持有 shape/stride 数组的所有权，并保证其生命周期覆盖 view 使用。

```cpp
struct BorrowedViewStorage {
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
};

StatusOr<std::vector<int64_t>> ExtractStaticShape(const TensorSpec& spec) {
    if (!spec.shape.IsStatic()) {
        return Status::Unimplemented("constant evaluator requires materializable shape");
    }

    const size_t rank = spec.shape.rank().value_or(0);
    std::vector<int64_t> shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        shape[i] = spec.shape[i].GetStaticValue();
    }
    return shape;
}

StatusOr<std::vector<int64_t>> MakeContiguousStrides(std::span<const int64_t> shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (size_t i = shape.size() - 1; i > 0; --i) {
        int64_t product = 0;
        if (CheckOverflowMul(strides[i], shape[i], &product)) {
            return Status::ResourceExhausted("contiguous stride computation overflow");
        }
        strides[i - 1] = product;
    }
    return strides;
}
```

`MakeContiguousStrides` 使用检查乘法 (checked multiplication) 拒绝其连续步幅计算会溢出 `int64_t` 的 shape（例如：前导维度为 0 但尾部维度极大的张量）。溢出返回 `Status::ResourceExhausted`；evaluator 和折叠 pass 中的调用方必须处理此情况。evaluator 将 `ResourceExhausted` 转换为 `Unimplemented`，使 pass 跳过该节点而不触发未定义行为。

#### 3. 输出 buffer 生命周期

`AllocateOutputViews(plan)` 为每个 `PlannedConstOutput` 分配 `std::shared_ptr<std::vector<std::byte>>`，用其 data 指针构造 `MutableTensorView`，并在 Evaluate 成功后转成 `ConstantBinding::inline_data`：

```cpp
struct OutputStorage {
    std::vector<MutableTensorView> views;
    std::vector<std::shared_ptr<std::vector<std::byte>>> buffers;
    BorrowedViewStorage metadata;
};
```

`ConstantBinding::inline_data` 的类型是 `std::shared_ptr<const std::vector<std::byte>>`。`std::shared_ptr<std::vector<std::byte>>` 可以移动给它，之后 `Commit()` 只增加引用计数，不深拷贝常量数据。

---

### 六、Evaluator 查找表

查找表保持静态只读，不使用可变 singleton registry。这样没有注册顺序问题，也避免测试间共享可变全局状态。

```cpp
namespace {

class AddConstEvaluator final : public ConstEvaluator {
public:
    StatusOr<ConstEvalPlan> Plan(
            std::span<const NodeOutputDesc> inputs,
            std::span<const NodeOutputDesc> outputs,
            const OpParams& params,
            const ConstEvalPolicy& policy) const override;

    Status Evaluate(
            std::span<const TensorView> inputs,
            std::span<MutableTensorView> outputs,
            const OpParams& params) const override;
};

const AddConstEvaluator kAddEvaluator;

struct Entry {
    OpType op_type;
    const ConstEvaluator* evaluator;
};

constexpr Entry kEvaluators[] = {
        {OpType::kAdd, &kAddEvaluator},
        // 新增 evaluator 时在此追加条目。
};

} // namespace

const ConstEvaluator* FindConstEvaluator(OpType op_type) noexcept {
    for (const Entry& entry : kEvaluators) {
        if (entry.op_type == op_type) {
            return entry.evaluator;
        }
    }
    return nullptr;
}
```

如果需要外部扩展 evaluator，应优先通过显式依赖注入或构造时传入只读表，而不是开放运行期全局注册接口。

---

### 七、常量折叠安全条件

一个节点可以折叠必须同时满足图级安全条件和 evaluator 语义条件。

图级条件由 Pass 统一检查：

1. `GetOperatorSchema(op_type)` 成功。
2. `IsCompileTimeEvaluable(schema)` 为 true。
3. 当前 session 中该节点 live，且可以取得 `GraphNodeView`。
4. 所有输入 value 在当前 session 视图中可解析为 `ConstantValue`；`WeightValue` 不满足折叠条件，即使 `GraphRewriteSession::IsConstant()` 将权重视为推理期不变量。
5. 所有输入常量的 `ConstantBinding::inline_data` 存在，且 `inline_data->size()` 与输入 `TensorSpec`（dtype × 静态 shape 元素数）一致。零元素 tensor 的预期字节数为 0，因此 `inline_data->size() == 0` 也合法，只要 dtype × numel == 0。
6. 找得到对应 `ConstEvaluator`。

`ConstantFoldingPass` 必须使用独立的 `AllInputsAreInlineConstantValues()` 检查，而不能直接使用 `GraphRewriteSession::AreAllInputsConstant()`。后者回答的是“输入是否推理期不变”，会把 `WeightValue` 也视为常量；前者回答的是“输入是否有 evaluator 可直接读取的内联常量字节”。

算子语义条件由 `Plan()` 检查（假设图级条件已满足，即输入已是 `ConstantValue` 且 `inline_data` 字节数与 `TensorSpec` 一致）：

1. 输入/输出数量和端口语义。
2. dtype、shape、rank、broadcast、stride、layout。
3. `OpParams` 的合法性。
4. 输出是否可物化为 `ConstantValue`。
5. 输出总字节数、计算成本和临时空间预算。
6. 数学域安全性与确定性。

这种拆分保证 Pass 不复制算子逻辑，同时 evaluator 不关心图 rewiring。

---

### 八、错误处理策略

`Status::Unimplemented` 是“该节点不折叠”的正常控制流，不应污染优化管线；其他错误表示不变量问题，应向上传播。

```cpp
auto plan = evaluator->Plan(inputs, outputs, params, policy);
if (plan.status() == StatusCode::kUnimplemented) {
    continue;
}
AM_RETURN_IF_ERROR(plan.status());

Status eval = evaluator->Evaluate(input_views, output_views, params);
if (eval == StatusCode::kUnimplemented) {
    continue;
}
AM_RETURN_IF_ERROR(eval);
```

建议约定：

- `Unimplemented`：不支持、成本超限、数学域不适合折叠。
- `InvalidArgument`：调用方传入与 `Plan()` 不一致的 views、错误数量或非法参数。
- `Internal`：Evaluator 自身不变量被破坏，例如输出 buffer 小于 plan 所需字节数。

---

### 九、新增 evaluator 的要求

新增一个算子的编译期求值实现时，必须满足以下要求：

1. 不调用运行时 Kernel，不访问 Backend、Device、线程池或执行计划。
2. `Plan()` 完整验证输入、输出、dtype、shape、layout 和参数语义。
3. `Evaluate()` 不保存 view，不保存 data 指针，不持有 graph/session 对象。
4. 不在 evaluator 内部长期持有内存；所有输出内存由 Pass 分配。
5. 对不支持组合返回 `Unimplemented`，不要崩溃或静默写错结果。
6. 对浮点算子写明精度策略和测试容差。
7. 对整数算子写明溢出语义：按 C++ 安全检查、饱和、wrap，或直接不折叠。
8. 对多输出算子，`Plan().outputs.size()` 必须与 graph node outputs 数量一致。
9. `PlannedConstOutput::debug_name` 命名约定为 `"folded_" + outputs[i].debug_name`，确保折叠后的常量在 dump 中可追踪到来源。`Plan()` 从 `outputs[i].debug_name` 读取原始输出名并拼接前缀，不依赖 graph/node 上下文。
10. 对 shape-only 算子，可以不做数值计算，但仍必须通过 plan 产生可物化常量输出。
11. `Plan().outputs[i].spec` 必须与 graph 声明的第 i 个输出 `TensorSpec` 一致（dtype 匹配，shape 必须为 fully static 且元素数匹配）。evaluator 只能补充 strides/layout 元数据，不能改变语义 spec。这保证 `ReplaceValue` 重定向后下游消费者看到的 spec 不变，且 `AllocateOutputViews` 能根据 plan 计算出确定的输出字节数。

---

### 十、方案特点总结

1. **Evaluator 通用化**：Pass 完全 op-agnostic；算子差异全部封装在 evaluator 的 `Plan()` 与 `Evaluate()` 中。
2. **职责边界清晰**：Pass 管 graph、session、buffer、constant binding；Evaluator 管数学语义和数据写入。
3. **安全降级**：不支持的情况返回 `Unimplemented`，原图保留，优化管线继续。
4. **零拷贝常量流转**：输出 buffer 通过 `ConstantBinding::inline_data` 交给 session，`Commit()` 时只增加引用计数。
5. **多输出自然支持**：接口从一开始就是 `span<MutableTensorView>`，不需要为多输出算子另开通道。
6. **跨编译友好**：纯 C++ reference kernel 避免运行期依赖，适合离线编译、交叉编译和无设备环境。
