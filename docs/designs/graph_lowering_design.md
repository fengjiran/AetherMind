# AetherMind Graph Lowering 模块设计方案

> 版本：v1.1（v1.0 为前瞻设计，v1.1 起补充实现现状对照）  
> 目标语言：C++20  
> 适用范围：AetherMind 大模型推理引擎 Graph IR → LoweredGraph 转换  
> 设计目标：建立清晰、可扩展、与 Kernel 选择及 Execution Plan Generation 解耦的 Lowering 层

---

## 0. 实现现状总览

> 本文 v1.0 为前瞻性设计方案；v1.1 起补充「实现现状对照」。当前代码事实以 `graph_compilation_flow.md` 与 compiler/execution 模块代码为准，本节是判断本文各章节「已落地 / 未落地」的权威索引。

### 0.1 当前三层链路（代码事实）

```text
ModelGraph（语义 DAG，graph 模块）
    ↓ OptimizeModelGraph（compiler 模块，O0/O1/O2 pipeline）
Optimized ModelGraph
    ↓ LowerModelGraph（compiler 模块）
LoweredGraph（immutable compiler artifact）
    │   └─ LoweredStep{LoweredStepSpec, LoweredStepBinding}[]
    │      LoweredValueDesc[] / model_inputs / model_outputs / state_aliases
    ↓ ExecutionPlanBuilder::Build（execution 模块）
ExecutionPlan（ExecutionStep[] + StateAliasPlan）
```

### 0.2 设计 vs 实现差异对照表

| 本文章节 | 设计方案 | 实现现状 | 状态 |
|---|---|---|---|
| §3/§8/§9/§37-44 | Semantic Op → Execution Primitive（PrimitiveOpType/PrimitiveParams/LoweringRegistry） | 语义 OpType 直沉为步骤（`LoweredStepSpec.op_type` = OpType），无 Primitive 层 | 🔮 未落地-未来演进方向 |
| §31/§41/§59 | Prefill/Decode 特化（AttentionPrefill/AttentionDecode 分裂图） | `ExecPhase` 作为 `KernelSelector.phase` 属性，单图 | 🔀 已偏离-替代方案 |
| §24-28 | KV Cache 独立建模（LoweredResource/ResourceUse/ControlDependency） | `StateValue` payload + `state_aliases`（schema 声明式 must-alias 坐标记录） | 🔀 已偏离-替代方案 |
| §22/§43/§44/§45/§17 | View Primitive + LayoutConstraint/StrideExpr 两阶段布局 | kReshape/kPermute/kReorder 为普通语义算子，无布局约束表达 | 🔮 未落地-未来演进方向 |
| §33/§34 | GraphLowerer 类 + LoweringRegistry 注册表 | 自由函数 `LowerModelGraph` 单一入口，无 per-op 规则注册 | 🔀 简化-当前无需 |
| §12 | LoweredGraph immutable + builder/finalize | `LoweredGraph` 私有 storage + 嵌套 `LoweredGraph::Builder::Build() &&` | ✅ 已落地 |
| §13 | StrongId<Tag> 强类型 ID | `GraphValueId`/`GraphNodeId`（distinct struct） | ✅ 已落地（非模板形式） |
| §49/§50 | LoweredGraphVerifier 类 | `ValidateLoweredGraph` 自由函数 + `Builder::Validate` | ✅ 已落地（自由函数形式） |
| §7.1-7.4/§52/§53/§61 | 不变量与禁止清单 | 全部成立（selector 仅 base 层属性，kernel/workspace/layout 决策在 execution） | ✅ 已落地 |
| §19-21 | Weight 作为 LoweredValue + 复合 binding | `WeightValue` payload + `QkvWeightBinding`/`GateUpWeightBinding` | ✅ 已落地 |
| §55 | 独立 lowering/ + planner/ + kernel/ 目录 | compiler/（lowering+optimize）+ execution/（planning）+ backend/（kernel） | 🔀 目录布局不同，分层对应 |
| §54 | Dump() 第一版实现 | 未实现 | 🔮 待办 |

### 0.3 当前实现路径声明

当前实现采用**「语义 OpType 直沉 + KernelSelector 属性」**的简化路径：Lowering 不做 Semantic Op → Primitive 的抽象映射，步骤类型直接复用算子语义类型（OpType），执行差异（device/ISA/weight format/phase/dtype）通过 base 层 `KernelSelector` 表达，kernel 选择完全推迟到 ExecutionPlanBuilder。Primitive 化（跨算子重排、schedule 化、backend 差异化展开需求出现时）为未来可选演进，触发条件见 §64.5。

---

## 1. 设计背景

在大模型推理引擎中，Graph IR 通常用于表达模型计算的高层语义，例如（当前 19 个 OpType 全量清单，见 `include/aethermind/operators/op_type.h`）：

- `kEmbedding`
- `kRmsNorm`
- `kLinear`
- `kQkvLinear`
- `kMatMul`
- `kRoPE`
- `kAttention`
- `kSilu`
- `kSiluMul`
- `kElementwiseMul`
- `kKVCacheUpdate`
- `kAdd`
- `kAddRmsNorm`
- `kSoftmax`
- `kArgmax`
- `kReshape`
- `kPermute`（文档旧称 Transpose）
- `kReorder`（文档旧称 Transpose）
- `kGateUpLinear`

这些算子描述的是“模型需要完成什么计算”，但并不适合作为最终执行表示。

另一方面，Execution Plan 需要解决的是更具体的执行问题，包括：

- 选择具体 Kernel 实现；
- 确定 ISA / Backend；
- 选择线程数和 Kernel tuning 参数；
- 查询 Workspace Requirement；
- 确定最终物理 Layout；
- 生成权重预打包格式；
- 进行 In-place / Alias 决策；
- 执行调度；
- 生命周期分析；
- Activation / Workspace 内存规划；
- 绑定 PackedWeightHandle；
- 生成 KernelInvocation。

如果直接从高层 Graph IR 跳到 Execution Plan，会导致模型语义、执行原语、Kernel 实现和资源规划高度耦合。因此 AetherMind 需要一个独立的 Lowering 层。

Lowering 的核心定位是：

> **将经过优化的模型语义 Graph IR 转换为与模型框架无关、但仍未绑定具体 Kernel 实现和执行资源的执行 Primitive Graph。**

最终形成三层主要表示：

```text
Graph IR
    ↓
LoweredGraph
    ↓
ExecutionPlan
```

分别回答：

```text
Graph IR：
    算什么？

LoweredGraph：
    由哪些执行 Primitive 完成？

ExecutionPlan：
    具体使用哪个 Kernel，以及如何配置执行资源？
```

---

## 2. 总体架构

AetherMind 推荐采用如下编译与执行流程：

```text
Model Loader / Graph Builder
           │
           ▼
┌──────────────────────────────┐
│           Graph IR           │
│                              │
│ Semantic Operators           │
│ Logical TensorSpec           │
│ WeightBinding                │
└──────────────┬───────────────┘
               │
               │ Graph Optimization
               ▼
┌──────────────────────────────┐
│      Optimized Graph IR      │
│                              │
│ Constant Folding             │
│ QKV Fusion                   │
│ GateUp Fusion                │
│ SiluMul Fusion               │
│ AddRmsNorm Fusion            │
│ DCE                          │
└──────────────┬───────────────┘
               │
               │ Graph Lowering
               ▼
┌──────────────────────────────┐
│         LoweredGraph         │
│                              │
│ PrimitiveOp                  │
│ LoweredTensorSpec            │
│ View Relationship            │
│ Weight Usage                 │
│ Resource Semantics           │
│ Control Dependency           │
└──────────────┬───────────────┘
               │
               │ Execution Planning
               ▼
┌──────────────────────────────┐
│        ExecutionPlan         │
│                              │
│ Concrete Kernel              │
│ Physical Layout              │
│ Packed Weight                │
│ Workspace                    │
│ Buffer Allocation            │
│ Alias Decision               │
│ Execution Schedule           │
└──────────────┬───────────────┘
               │
               ▼
            Runtime
```

---

## 3. Lowering 的正式定义（🔮 定义成立；Primitive 映射体系未落地，当前实现为 OpType 直沉，见 §0）

AetherMind 中 Graph Lowering 定义为：

> **Graph Lowering 将经过 Graph Optimization 的语义 Graph IR 转换为由 Execution Primitive 构成的 LoweredGraph。该阶段负责消除模型级算子抽象、规范化计算结构、明确 View 关系和持久化资源语义，并为后续 Kernel Implementation Selection 提供稳定的执行原语表示；但不负责选择具体 Kernel、Workspace 计算、最终物理 Layout、权重打包格式、Alias 决策、内存规划和执行调度。**

其核心转换关系为：

```text
Semantic Operator
        ↓
Execution Primitive
```

而不是：

```text
Semantic Operator
        ↓
Concrete Kernel
```

具体 Kernel 选择推迟到 Execution Plan Generation。

---

## 4. Lowering 与其他阶段的边界

### 4.1 Graph Optimization

负责改变 Graph 的语义表达方式：

```text
Linear(q) + Linear(k) + Linear(v)
        ↓
QkvLinear
```

包括：

- Constant Folding；
- QKV Fusion；
- GateUp Fusion；
- SiluMul Fusion；
- AddRmsNorm Fusion；
- DCE；
- 其他语义级 Rewrite。

Graph Optimization 不关心具体 Kernel。

---

### 4.2 Lowering（🔮 映射示意；当前实现为语义算子直沉为步骤）

负责将语义 Operator 映射为执行 Primitive：

```text
Linear
    ↓
Gemm

QkvLinear
    ↓
QkvGemm

GateUpLinear
    ↓
GateUpGemm

Attention
    ↓
AttentionPrefill / AttentionDecode

kAddRmsNorm
    ↓
AddRmsNorm
```

Lowering 允许：

```text
1 Semantic Op
→
0 / 1 / N Primitive Op
```

但不选择 AVX2、AVX-512、AMX、CUDA 等具体 Kernel。

---

### 4.3 Execution Plan Generation

负责：

- Kernel implementation selection；
- Kernel configuration；
- Workspace Requirement；
- Physical Layout；
- Weight Packing；
- Layout legalization；
- In-place decision；
- Execution scheduling；
- Liveness；
- Buffer reuse；
- Activation planning；
- Workspace planning；
- KernelInvocation materialization。

---

### 4.4 Runtime

负责：

- 本次请求输入绑定；
- KV Cache 实例绑定；
- 当前 token position；
- 当前 sequence length；
- dynamic runtime parameter；
- 按 ExecutionPlan 执行。

---

## 5. 职责划分表

| 功能 | Graph Pass | Lowering | Execution Plan | Runtime |
|---|---:|---:|---:|---:|
| Constant Folding | ✓ | | | |
| QKV Pattern Fusion | ✓ | | | |
| GateUp Fusion | ✓ | | | |
| AddRmsNorm Fusion | ✓ | | | |
| DCE | ✓ | | | |
| Semantic Op → Primitive | | ✓ | | |
| Prefill / Decode Primitive Specialization | | ✓ | | |
| View Normalization | | ✓ | | |
| Persistent Resource Semantics | | ✓ | | |
| Primitive-level Shape / DType Constraint | | ✓ | | |
| Concrete Kernel Selection | | | ✓ | |
| ISA / Backend Implementation Selection | | | ✓ | |
| Kernel Tuning Parameters | | | ✓ | |
| Workspace Requirement | | | ✓ | |
| Final Physical Layout | | | ✓ | |
| Packed Weight Format | | | ✓ | |
| Layout Reorder / Pack | | | ✓ | |
| In-place / Alias Decision | | | ✓ | |
| Execution Scheduling | | | ✓ | |
| Liveness Analysis | | | ✓ | |
| Buffer Allocation / Reuse | | | ✓ | |
| Runtime Input Binding | | | | ✓ |
| KV Cache Instance Binding | | | | ✓ |
| Current Position / Seq Len | | | | ✓ |

---

## 6. LoweredGraph 的定位

LoweredGraph 是一个：

> **Kernel-unbound、resource-unbound、primitive-level execution graph。**

它不再保留模型框架级高层 Operator，但仍然不包含具体 Kernel 实现和运行时资源。

LoweredGraph 应满足：

```text
Graph IR
    model semantic

LoweredGraph
    execution primitive semantic

ExecutionPlan
    concrete implementation
```

---

## 7. LoweredGraph 核心设计原则（✅ 已落地，见 §0.2）

### 7.1 LoweredGraph 不保存 KernelId

错误设计：

```cpp
struct LoweredNode {
    KernelId kernel_id;
};
```

推荐：

```cpp
struct LoweredNode {
    PrimitiveOp op;
};
```

因为 Kernel 选择属于 Execution Planner。

---

### 7.2 LoweredGraph 不保存 WorkspaceRequirement

Workspace 是 Kernel implementation 的执行资源需求。

LoweredGraph 尚未选择 Kernel，因此不应该出现：

```cpp
WorkspaceRequirement workspace;
```

Planner 在选定 Kernel 后：

```text
KernelDescriptor
    ↓
QueryWorkspace()
    ↓
WorkspaceRequirement
```

---

### 7.3 LoweredGraph 不保存最终 Physical Layout

不同 Kernel 可能要求：

```text
AVX512:
    K32N16

AMX:
    AMX_TILE

Generic:
    RowMajor
```

LoweredGraph 只保存：

```text
LayoutConstraint
```

而非：

```text
PhysicalLayout
```

---

### 7.4 LoweredGraph 不保存 PackedWeight Format

例如 QKV 权重只需要表达：

```text
Composite Q/K/V Weight
used by QkvGemm
```

Planner 在选择 Kernel 后决定：

```text
QKV_K32_N16
AMX_QKV_TILE
INT4_GROUPWISE_K64
...
```

---

### 7.5 LoweredGraph 保留 View 的 mandatory alias 语义

例如：

```text
Reshape(A) → B
```

如果 lowering 判断这是纯 metadata operation，则：

```text
B MUST alias A
```

这是 LoweredGraph 的一部分。

但是 Kernel 的：

```text
output MAY alias input
```

属于 Kernel capability，由 Planner 决定。

---

## 8. Execution Primitive（🔮 未来演进方向，当前未引入）

### 8.1 PrimitiveOpType

推荐：

```cpp
enum class PrimitiveOpType : uint16_t {
    // Matrix computation
    kGemm,
    kQkvGemm,
    kGateUpGemm,

    // Normalization
    kRmsNorm,
    kAddRmsNorm,

    // Activation
    kSilu,
    kSiluMul,

    // Attention
    kAttentionPrefill,
    kAttentionDecode,

    // Positional encoding
    kRoPE,

    // Tensor
    kSoftmax,
    kEmbedding,

    // KV cache
    kKvCacheUpdate,

    // Metadata-only operation
    kView,
};
```

注意：

```text
PrimitiveOpType
```

不是 Kernel 类型，也不是 Graph IR Operator 类型。

---

## 9. Graph Operator 与 Primitive 的映射（🔮 未来演进方向）

推荐映射：

| Graph IR Operator | Lowered Primitive |
|---|---|
| Linear | Gemm |
| QkvLinear | QkvGemm |
| GateUpLinear | GateUpGemm |
| RmsNorm | RmsNorm |
| kAddRmsNorm | AddRmsNorm |
| Silu | Silu |
| SiluMul | SiluMul |
| Attention | AttentionPrefill / AttentionDecode |
| RoPE | RoPE |
| Softmax | Softmax |
| Embedding | Embedding |
| KVCacheUpdate | KvCacheUpdate |
| Reshape | View |
| Transpose | View（如果语义上可表示为 stride 变化） |

Lowering 不保证每个 Primitive 最终一定由单个 Kernel 实现。

例如：

```text
QkvGemm
```

Planner 可以选择：

```text
Fused QKV Kernel
```

或者：

```text
Gemm Q
Gemm K
Gemm V
```

---

## 10. Primitive 参数设计

推荐：

```cpp
struct PrimitiveOp {
    PrimitiveOpType type;
    PrimitiveParams params;
};
```

参数使用强类型：

```cpp
using PrimitiveParams =
    std::variant<
        NoPrimitiveParams,
        GemmParams,
        QkvGemmParams,
        GateUpGemmParams,
        RmsNormParams,
        AddRmsNormParams,
        AttentionParams,
        RopeParams,
        SoftmaxParams,
        EmbeddingParams,
        KvCacheUpdateParams,
        ViewParams>;
```

不建议：

```cpp
unordered_map<string, Attribute>
```

原因：

- 类型安全差；
- 容易拼写错误；
- 不利于编译期检查；
- 不利于序列化；
- 不利于重构。

---

## 11. 典型 Primitive 参数

### 11.1 GemmParams

```cpp
struct GemmParams {
    bool transpose_a = false;
    bool transpose_b = true;

    bool has_bias = false;
};
```

---

### 11.2 QkvGemmParams

```cpp
struct QkvGemmParams {
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    bool has_bias = false;
};
```

---

### 11.3 RmsNormParams

```cpp
struct RmsNormParams {
    float epsilon;
};
```

---

### 11.4 AttentionParams

```cpp
struct AttentionParams {
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    float scale;

    bool causal;
};
```

---

### 11.5 RopeParams

```cpp
struct RopeParams {
    uint32_t rotary_dim;

    float theta;

    RopeScalingType scaling_type;
};
```

---

## 12. LoweredGraph 数据模型（✅ 已落地，具体结构见 §64.1）

推荐：

```cpp
class LoweredGraph final {
public:
    AM_NODISCARD std::span<const LoweredNode>
    nodes() const noexcept;

    AM_NODISCARD std::span<const LoweredValue>
    values() const noexcept;

    AM_NODISCARD std::span<const LoweredResource>
    resources() const noexcept;

    AM_NODISCARD std::span<const LoweredValueId>
    inputs() const noexcept;

    AM_NODISCARD std::span<const LoweredValueId>
    outputs() const noexcept;

    AM_NODISCARD const LoweredGraphMetadata&
    metadata() const noexcept;

private:
    friend class LoweredGraphBuilder;

    std::vector<LoweredNode> nodes_;
    std::vector<LoweredValue> values_;
    std::vector<LoweredResource> resources_;

    std::vector<LoweredValueId> inputs_;
    std::vector<LoweredValueId> outputs_;

    LoweredGraphMetadata metadata_;
};
```

LoweredGraph 在 `Finalize()` 后应保持 immutable。

---

## 13. 强类型 ID（✅ 已落地，实现为 distinct struct 而非模板 StrongId）

推荐：

```cpp
template <typename Tag, typename T = uint32_t>
struct StrongId {
    T value{};
};

using LoweredNodeId =
    StrongId<struct LoweredNodeTag>;

using LoweredValueId =
    StrongId<struct LoweredValueTag>;

using LoweredResourceId =
    StrongId<struct LoweredResourceTag>;
```

避免：

```cpp
uint32_t node_id;
uint32_t value_id;
```

之间误用。

---

## 14. LoweredNode

推荐：

```cpp
struct LoweredNode {
    LoweredNodeId id;

    PrimitiveOp op;

    std::vector<LoweredValueId> inputs;
    std::vector<LoweredValueId> outputs;

    std::vector<ResourceUse> resources;

    std::vector<LoweredNodeId> control_dependencies;

    DebugOrigin debug_origin;
};
```

注意 LoweredNode 中不包含：

```text
KernelId
KernelFn
Workspace
Buffer
Arena Offset
PackedWeightHandle
```

---

## 15. LoweredValue

LoweredValue 表示：

> 已经经过语义 Lowering，但仍未进行最终 Kernel-specific physicalization 的 Tensor Value。

推荐：

```cpp
struct LoweredValue {
    LoweredValueId id;

    LoweredTensorSpec spec;

    LoweredValueOrigin origin;

    std::optional<LoweredNodeId> producer;

    DebugOrigin debug_origin;
};
```

---

## 16. LoweredTensorSpec

推荐：

```cpp
struct LoweredTensorSpec {
    DataType dtype;

    ShapeExpr shape;

    LayoutConstraint layout_constraint;

    AlignmentConstraint alignment_constraint;
};
```

它不是最终：

```cpp
PhysicalTensorSpec
```

因为 Planner 尚未选择 Kernel。

---

## 17. LayoutConstraint（🔮 未来演进方向，当前无布局约束表达）

推荐不要直接使用：

```cpp
Layout::kContiguous
Layout::kPacked
```

而应表达约束。

例如：

```cpp
enum class LayoutConstraintKind : uint8_t {
    kAny,
    kDense,
    kDenseLastDim,
    kStridedView,
    kWeightCompatible,
};
```

进一步：

```cpp
struct LayoutConstraint {
    LayoutConstraintKind kind;

    std::optional<StrideExpr> strides;
};
```

Planner 最终根据 Kernel capability 将其 materialize 为：

```text
RowMajor
K32N16
AMXTile
...
```

---

## 18. LoweredValueOrigin

推荐：

```cpp
enum class LoweredValueOriginKind : uint8_t {
    kInternal,
    kGraphInput,
    kWeight,
    kConstant,
};
```

```cpp
struct LoweredValueOrigin {
    LoweredValueOriginKind kind;

    std::variant<
        std::monostate,
        GraphInputBinding,
        LoweredWeightSpec,
        ConstantBinding
    > payload;
};
```

Graph Output 不作为一种 Value Origin。

Graph output 应保存为：

```cpp
std::vector<LoweredValueId> outputs_;
```

---

## 19. Weight 在 LoweredGraph 中的表示（✅ 已落地）

权重应统一作为 LoweredValue。

例如：

```text
hidden ─────────────┐
                    ▼
                 QkvGemm
                    ▲
                    │
               qkv_weight
```

而不是：

```cpp
PrimitiveOp {
    WeightRequirement weight;
}
```

这样 Tensor 数据流更加统一。

---

## 20. LoweredWeightSpec（✅ 概念已落地：WeightValue payload + 语义 role）

推荐：

```cpp
struct LoweredWeightSpec {
    CompositeWeightBinding binding;

    WeightUsage usage;
};
```

例如：

```cpp
enum class WeightUsage : uint8_t {
    kLinearKernel,
    kQkvGemm,
    kGateUpGemm,
    kRmsNormScale,
    kEmbedding,
};
```

---

## 21. CompositeWeightBinding（✅ 已落地：QkvWeightBinding/GateUpWeightBinding）

为了支持 QKV / GateUp 等复合权重：

```cpp
struct CompositeWeightBinding {
    SmallVector<WeightBinding, 4> components;

    WeightComposition composition;
};
```

例如：

```cpp
enum class WeightComposition : uint8_t {
    kSingle,
    kConcatOutputChannel,
    kInterleavedLogical,
};
```

QKV：

```text
components:
    q_proj.weight
    k_proj.weight
    v_proj.weight

composition:
    kConcatOutputChannel
```

LoweredGraph 不决定具体 packing format。

---

## 22. View Primitive（🔮 未来演进方向，当前未引入）

View 是一个特殊 Primitive。

推荐：

```cpp
struct ViewParams {
    int64_t byte_offset = 0;
};
```

output 的：

- shape；
- strides；
- dtype；

由对应 `LoweredTensorSpec` 保存。

例如：

```text
A:
    [1,16,4096]

Reshape
    ↓

B:
    [16,4096]
```

Lowered：

```text
A
  │
  ▼
View
  │
  ▼
B
```

其语义为：

```text
B MUST alias A
```

Planner 只负责实际 buffer binding。

---

## 23. Mandatory Alias 与 Optional In-place

必须严格区分。

## View

```text
output MUST alias input
```

属于 LoweredGraph 语义。

---

## Kernel In-place

```text
output MAY alias input
```

属于 Kernel implementation capability。

例如：

```text
AddRmsNorm Kernel A:
    output 0 may alias input 0

Kernel B:
    no inplace
```

Planner 在 Kernel 选择后决定。

LoweredGraph 不保存 `InplaceCandidate`。

---

## 24. 持久化 Resource（🔀 已偏离：实现采用 StateValue + state_aliases 值语义建模，见 §64.3）

KV Cache 不建议建模为普通 SSA Tensor Value。

它具有：

```text
persistent
mutable
request-scoped
side-effectful
```

因此独立建模。

---

## 25. LoweredResource（🔀 已偏离，无独立 Resource 类型）

推荐：

```cpp
enum class LoweredResourceKind : uint8_t {
    kKvCache,
};
```

```cpp
struct LoweredResource {
    LoweredResourceId id;

    LoweredResourceKind kind;

    ResourceSpec spec;

    DebugOrigin debug_origin;
};
```

---

## 26. KV Cache ResourceSpec（🔀 已偏离：对应 KVCacheStateBinding）

例如：

```cpp
struct KvCacheResourceSpec {
    uint32_t decoder_layer_index;

    uint32_t num_kv_heads;
    uint32_t head_dim;

    DataType dtype;
};
```

这里不保存：

```text
KV cache pointer
cache length
page table
request id
physical buffer
```

---

## 27. ResourceUse（🔀 已偏离，无显式 ResourceUse 记录）

Node 显式声明 Resource 使用：

```cpp
enum class ResourceAccess : uint8_t {
    kRead,
    kWrite,
    kReadWrite,
};
```

```cpp
struct ResourceUse {
    LoweredResourceId resource;

    ResourceAccess access;
};
```

例如：

```text
KVCacheUpdate:
    KVCache Write

AttentionDecode:
    KVCache Read
```

或者 fused decode attention：

```text
AttentionDecode:
    KVCache ReadWrite
```

---

## 28. Control Dependency（🔀 已偏离：实现以拓扑序线性化表达顺序）

Tensor edge 无法完全表达 mutable resource 的顺序约束。

因此推荐：

```cpp
std::vector<LoweredNodeId> control_dependencies;
```

例如：

```text
KVCacheUpdate
      │
      │ control dependency
      ▼
AttentionDecode
```

Planner 在 scheduling 时必须遵守：

```text
Data Dependency
+
Control Dependency
```

---

## 29. DebugOrigin

建议从第一版就实现。

```cpp
struct DebugOrigin {
    std::optional<GraphNodeId> graph_node;

    std::optional<GraphValueId> graph_value;

    std::optional<uint32_t> decoder_layer_index;
};
```

用途：

- Debug；
- Dump；
- Graph → LoweredGraph 对照；
- 融合问题排查；
- decoder layer 语义追踪；
- WeightBinding 错误定位。

---

## 30. LoweredGraphMetadata

推荐：

```cpp
struct LoweredGraphMetadata {
    ExecutionMode execution_mode;

    ShapeSpecialization shape_specialization;

    LoweredGraphVersion version;
};
```

当前可以包含：

```text
Prefill / Decode
fixed model dimension
symbolic seq_len
```

不建议放：

```text
AVX512
AMX
thread count
```

因为 Kernel selection 尚未发生。

---

## 31. Prefill / Decode Specialization（🔀 已偏离：实现为 ExecPhase selector 属性，单图）

建议在 Lowering 阶段完成 Primitive specialization。

Graph：

```text
Attention
```

Lowering Context：

```cpp
enum class ExecutionMode : uint8_t {
    kPrefill,
    kDecode,
};
```

得到：

```text
Prefill:
    AttentionPrefill

Decode:
    AttentionDecode
```

因此可以生成：

```text
PrefillLoweredGraph
DecodeLoweredGraph
```

然后分别进行 Planning。

---

## 32. LoweringContext（🔀 已偏离：实现为 GraphLoweringConfig{KernelSelector}）

推荐：

```cpp
struct LoweringContext {
    ExecutionMode execution_mode;

    const ShapeEnvironment& shape_env;

    const ModelContext& model_context;

    LoweringOptions options;
};
```

注意不再依赖：

```text
KernelRegistry
KernelSelector
MemoryPlanner
WorkspacePlanner
WeightManager
```

这是新设计的重要收益。

---

## 33. GraphLowerer（🔀 已偏离：实现为自由函数 LowerModelGraph）

推荐接口：

```cpp
class GraphLowerer {
public:
    AM_NODISCARD Result<LoweredGraph> Lower(
        const Graph& graph,
        const LoweringContext& context) const;

private:
    AM_NODISCARD Result<void> LowerNode(
        const Node& node,
        LoweringContext& context,
        LoweredGraphBuilder& builder) const;
};
```

核心特点：

> `GraphLowerer` 不依赖 KernelRegistry。

因此 Lowering 与具体 Kernel 实现完全解耦。

---

## 34. LoweringRegistry（🔀 已偏离：当前 1:1 直沉无需 per-op 规则注册）

不同 Graph Operator 需要不同 lowering rule。

推荐：

```cpp
using LoweringFn =
    Result<void> (*)(
        const Node&,
        LoweringContext&,
        LoweredGraphBuilder&);

class LoweringRegistry {
public:
    void Register(
        OperatorType type,
        LoweringFn lowering_fn);

    AM_NODISCARD LoweringFn Lookup(
        OperatorType type) const;
};
```

注册：

```cpp
registry.Register(
    OperatorType::kLinear,
    &LowerLinear);

registry.Register(
    OperatorType::kQkvLinear,
    &LowerQkvLinear);

registry.Register(
    OperatorType::kAttention,
    &LowerAttention);
```

---

## 35. LoweredGraphBuilder（🔀 简化：实现为嵌套 LoweredGraph::Builder，见 §64.2）

推荐：

```cpp
class LoweredGraphBuilder {
public:
    LoweredValueId AddGraphInput(
        LoweredTensorSpec spec,
        GraphInputBinding binding);

    LoweredValueId AddWeight(
        LoweredTensorSpec spec,
        LoweredWeightSpec weight);

    LoweredValueId AddInternalValue(
        LoweredTensorSpec spec);

    LoweredNodeId AddPrimitive(
        PrimitiveOp op,
        std::span<const LoweredValueId> inputs,
        std::span<const LoweredValueId> outputs);

    LoweredValueId AddView(
        LoweredValueId source,
        LoweredTensorSpec output_spec,
        ViewParams params);

    LoweredResourceId AddResource(
        ResourceSpec spec);

    void MarkOutput(
        LoweredValueId value);

    AM_NODISCARD Result<LoweredGraph>
    Finalize();
};
```

Builder 负责维护：

- SSA；
- producer；
- Graph input / output；
- Resource；
- Debug origin；
- Node / Value ID。

---

## 36. Value Mapping（🔀 简化：实现直接保留 GraphValueId 引用，无独立映射表）

GraphLowerer 需要维护：

```text
GraphValueId
    ↓
LoweredValueId
```

建议：

```cpp
class LoweringValueMap {
public:
    void Bind(
        GraphValueId graph_value,
        LoweredValueId lowered_value);

    AM_NODISCARD LoweredValueId Lookup(
        GraphValueId graph_value) const;
};
```

处理：

```text
Graph Value
    → Lowered Value
```

---

## 37. Lowering Result 不要求 1:1（🔮 未来演进方向；当前实现为 1:1 直沉）

Lowering rule 必须允许：

```text
1 Graph Node
→
0 / 1 / N Primitive Nodes
```

例如：

## Reshape

```text
Reshape
→ View
```

没有 executable kernel。

---

## RmsNorm

```text
RmsNorm
→ RmsNorm Primitive
```

---

## 复杂语义算子

未来可以：

```text
SemanticOp
→ Primitive A
→ Primitive B
→ Primitive C
```

---

## 38. Linear Lowering（🔮 映射示意，见 §0.3 路径声明）

Graph：

```text
Linear(X, W)
```

Lowered：

```text
Gemm(X, W)
```

示意实现：

```cpp
Result<void> LowerLinear(
    const Node& node,
    LoweringContext& context,
    LoweredGraphBuilder& builder) {

    const auto input =
        context.value_map.Lookup(node.input(0));

    const auto weight =
        context.value_map.Lookup(node.input(1));

    auto output =
        builder.AddInternalValue(
            LowerTensorSpec(node.output(0)));

    PrimitiveOp op{
        .type = PrimitiveOpType::kGemm,
        .params = GemmParams{
            .transpose_a = false,
            .transpose_b = true,
            .has_bias = false,
        },
    };

    builder.AddPrimitive(
        std::move(op),
        {input, weight},
        {output});

    context.value_map.Bind(
        node.output(0),
        output);

    return {};
}
```

---

## 39. QkvLinear Lowering（🔮 映射示意）

Optimized Graph：

```text
QkvLinear(
    hidden,
    qkv_weight
)
```

其中 qkv weight 可以是 CompositeWeightBinding：

```text
Q
K
V
```

Lowered：

```text
QkvGemm(
    hidden,
    composite_qkv_weight
)
```

示意：

```cpp
PrimitiveOp op{
    .type = PrimitiveOpType::kQkvGemm,
    .params = QkvGemmParams{
        .num_q_heads = ...,
        .num_kv_heads = ...,
        .head_dim = ...,
        .has_bias = false,
    },
};
```

此时不决定：

```text
AVX512 QKV kernel
AMX QKV kernel
3 × GEMM fallback
```

---

## 40. GateUpLinear Lowering（🔮 映射示意）

Graph：

```text
GateUpLinear(
    hidden,
    gate_up_weight
)
```

Lowered：

```text
GateUpGemm
```

Planner 后续可以选择：

```text
Fused GateUp GEMM
```

或：

```text
Gate GEMM
+
Up GEMM
```

---

## 41. Attention Lowering（🔀 已偏离：无 Prefill/Decode 分裂，见 §31）

Graph：

```text
Attention
```

根据 `ExecutionMode`：

```text
Prefill
    ↓
AttentionPrefill

Decode
    ↓
AttentionDecode
```

注意：

```text
FlashAttention
```

不应该成为 Graph IR / LoweredGraph Operator 类型。

它属于具体 Kernel implementation。

---

## 42. AddRmsNorm Lowering（🔮 映射示意：语义算子 kAddRmsNorm → Primitive AddRmsNorm）

Graph：

```text
kAddRmsNorm
```

Lowered：

```text
AddRmsNorm
```

Planner 可以选择：

```text
fused AddRmsNorm kernel
```

如果没有：

```text
Add Kernel
+
RmsNorm Kernel
```

具体 fallback 属于 implementation selection。

---

## 43. Reshape Lowering（🔮 未来演进方向：无 View 体系）

Graph：

```text
Reshape(A)
```

Lowering 判断 reshape 在逻辑 stride 语义上是否可表示为 view。

如果可以：

```text
View(A)
```

Lowered Value 保存：

```text
new shape
new stride expression
```

不产生 copy。

如果不能表示为纯 view，则 Lowering 不应直接选择某个 copy kernel，而应转换为需要物化的数据变换 Primitive；如果第一版暂不支持这种情况，可以直接返回 `UnsupportedLowering`。

---

## 44. Transpose Lowering（🔮 未来演进方向：对应 kPermute/kReorder 语义算子）

Transpose 通常可以先表示为 stride-changing View：

```text
Transpose
    ↓
View
```

后续 Kernel 是否支持该 strided layout，由 Planner 在 Kernel Selection 后判断。

例如：

```text
View(strided)
    ↓
QkvGemm
```

Planner：

```text
Kernel A supports strided
    → directly use

Kernel B requires contiguous
    → insert Reorder implementation
```

因此 Lowering 不提前插入 Kernel-specific reorder。

---

## 45. Layout 的两阶段处理（🔮 未来演进方向：当前无布局约束表达）

Layout 必须拆成：

## Lowering

确定：

```text
logical / primitive-level layout constraint
```

例如：

```text
dense
strided view
dense last dimension
```

---

## Execution Planning

确定：

```text
concrete physical layout
```

例如：

```text
RowMajor
K32N16
AMXTile
CUDA_COL32
```

这避免 Lowering 与 Kernel implementation 耦合。

---

## 46. Weight Packing 的两阶段处理（🔀 部分落地：PackedWeightStore/WeightPrepackPlanner 为兼容设施，graph-driven materialization 待 P2）

LoweredGraph：

```text
Weight:
    logical source
    composition
    usage
```

Execution Planner：

```text
Primitive
+
Selected Kernel
    ↓
required packed format
    ↓
WeightManager
    ↓
PackedWeightHandle
```

例如：

```text
QkvGemm
+
AMXQkvKernel
    ↓
AMX_QKV_K32N16
```

---

## 47. Kernel Selection 接口属于 Planner（🔀 部分落地：KernelSelector 已实现，基于 OpType 而非 Primitive）

KernelSelector 应基于 Primitive 查询。

例如：

```cpp
struct KernelQuery {
    PrimitiveOpType op_type;

    PrimitiveParams params;

    std::span<const PlannedTensorConstraint> inputs;
    std::span<const PlannedTensorConstraint> outputs;

    TargetInfo target;

    PlanningPolicy policy;
};
```

Planner：

```text
Lowered Primitive
    ↓
KernelSelector
    ↓
Implementation
```

---

## 48. Implementation 可以是单 Kernel 或 Kernel Sequence（🔮 未来演进方向）

为了支持 fused primitive fallback：

```cpp
using ImplementationPlan =
    std::variant<
        SingleKernelImplementation,
        KernelSequenceImplementation>;
```

例如：

```text
QkvGemm
```

可以：

```text
Single:
    FusedQkvKernel
```

或者：

```text
Sequence:
    GemmQ
    GemmK
    GemmV
```

这样 LoweredGraph 不需要因为目标平台不同而改变。

---

## 49. LoweredGraph Verifier（✅ 已落地：ValidateLoweredGraph 自由函数，见 §64.4）

推荐：

```cpp
class LoweredGraphVerifier {
public:
    AM_NODISCARD Status Verify(
        const LoweredGraph& graph) const;
};
```

由于 LoweredGraph 不含 Kernel，因此 Verifier 不需要 KernelRegistry。

---

## 50. Verifier 检查项（✅ 核心检查项已落地，与实现的差异见 §64.4）

### 50.1 SSA

每个 Internal Value：

```text
必须且只有一个 producer
```

GraphInput / Weight / Constant：

```text
不得有 producer
```

---

### 50.2 Primitive Arity

例如：

```text
Gemm:
    X
    W

QkvGemm:
    hidden
    weight
```

必须符合 Primitive 定义。

---

### 50.3 DType 约束

例如：

```text
RmsNorm:
    floating type

Embedding index:
    integer type
```

---

### 50.4 Shape 约束

例如：

```text
QkvGemm:
    input hidden dimension
    与 weight K dimension 一致
```

---

### 50.5 View 合法性

检查：

- rank；
- element count；
- dtype size；
- byte offset；
- stride expression；
- storage range 关系。

---

### 50.6 Resource 合法性

例如：

```text
AttentionDecode
```

必须使用合法 KVCache Resource。

---

### 50.7 DAG

检查：

```text
data dependency
+
control dependency
```

不能产生非法 cycle。

---

## 51. LoweredGraph 不变量（✅ 已落地，见 ValidateLoweredGraph）

建议正式固化以下 invariant：

1. LoweredGraph 不包含高层模型语义 Operator。
2. LoweredGraph 不包含 Concrete Kernel。
3. LoweredGraph 不包含 Workspace。
4. LoweredGraph 不包含 Buffer allocation。
5. LoweredGraph 不包含 PackedWeightHandle。
6. LoweredGraph 不包含最终 In-place decision。
7. LoweredGraph 不包含 execution schedule。
8. LoweredGraph 不包含 runtime request state。
9. View 表示 mandatory alias。
10. Internal Tensor Value 保持 SSA。
11. Mutable Resource 显式建模。
12. Graph 必须是 data/control dependency 意义下的合法 DAG。

---

## 52. LoweredGraph Forbidden List（✅ 已落地）

LoweredGraph **禁止包含**：

```text
× KernelId
× KernelFn
× AVX2 / AVX512 / AMX implementation ID
× WorkspaceRequirement
× WorkspaceSlice
× Buffer*
× void*
× Arena offset
× BufferSlice
× Concrete packed layout
× PackedWeightHandle
× final inplace binding
× liveness interval
× execution schedule
× thread assignment
× NUMA assignment
× stream assignment
× request id
× current sequence length
× current token position
× KVCacheHandle
```

---

## 53. LoweredGraph 应包含（✅ 大部分落地，与实现的差异见 §64）

```text
✓ PrimitiveOp
✓ Primitive static parameters
✓ LoweredTensorSpec
✓ ShapeExpr
✓ DType
✓ LayoutConstraint
✓ Graph Input / Output
✓ Weight logical source
✓ CompositeWeightBinding
✓ WeightUsage
✓ View relationship
✓ Persistent Resource definition
✓ Resource access
✓ Control dependency
✓ Debug origin
✓ ExecutionMode specialization
```

---

## 54. Dump 设计（🔮 待办：当前未实现 LoweredGraph::Dump）

建议 LoweredGraph 从第一版提供：

```cpp
std::string Dump() const;
```

示例：

```text
lowered_graph @llama_decode {
  mode = decode

  resource %r0 :
      kv_cache<
          layer=0,
          dtype=bf16,
          kv_heads=8,
          head_dim=128
      >

  %0 = input :
      bf16[1,4096]

  %w0 = weight :
      usage=qkv_gemm,
      source=[
          q_proj.weight,
          k_proj.weight,
          v_proj.weight
      ]

  %1 =
      qkv_gemm
      (%0, %w0)
      {
          q_heads=32,
          kv_heads=8,
          head_dim=128
      }

  %2 =
      view %1
      shape=[1,48,128]

  %3 =
      attention_decode
      (%2)
      resource(%r0, read_write)
}
```

---

## 55. 推荐目录结构（🔀 实际目录为 compiler/ + execution/ + backend/，见下方对照）

```text
aethermind/
├── graph/
│   ├── graph.h
│   ├── node.h
│   ├── value.h
│   ├── tensor_spec.h
│   └── passes/
│
├── lowering/
│   ├── graph_lowerer.h
│   ├── graph_lowerer.cc
│   ├── lowering_context.h
│   ├── lowering_registry.h
│   ├── lowering_value_map.h
│   ├── lowered_graph_builder.h
│   │
│   ├── ir/
│   │   ├── lowered_graph.h
│   │   ├── lowered_node.h
│   │   ├── lowered_value.h
│   │   ├── lowered_tensor_spec.h
│   │   ├── lowered_resource.h
│   │   ├── primitive_op.h
│   │   ├── primitive_params.h
│   │   ├── layout_constraint.h
│   │   └── lowered_weight_spec.h
│   │
│   ├── rules/
│   │   ├── linear_lowering.cc
│   │   ├── qkv_linear_lowering.cc
│   │   ├── gate_up_linear_lowering.cc
│   │   ├── attention_lowering.cc
│   │   ├── rms_norm_lowering.cc
│   │   ├── add_rms_norm_lowering.cc
│   │   ├── reshape_lowering.cc
│   │   ├── transpose_lowering.cc
│   │   └── kv_cache_lowering.cc
│   │
│   └── verify/
│       └── lowered_graph_verifier.h
│
├── kernel/
│   ├── kernel_registry.h
│   ├── kernel_selector.h
│   └── kernel_descriptor.h
│
├── planner/
│   ├── execution_planner.h
│   ├── implementation_selector.h
│   ├── layout_legalizer.h
│   ├── weight_planner.h
│   ├── workspace_planner.h
│   ├── alias_planner.h
│   ├── liveness_analysis.h
│   └── memory_planner.h
│
└── runtime/
    ├── execution_plan.h
    ├── kernel_invocation.h
    └── executor.h
```

### 55.1 实际目录结构对照（代码事实）

```text
include/aethermind/ + src/
├── compiler/          （≈ 本文 lowering/：optimize + lower + artifact）
│   ├── optimize_graph.h / .cpp        OptimizeModelGraph（O0/O1/O2 pipeline）
│   ├── graph_lowering.h / .cpp        GraphLoweringConfig + LowerModelGraph + ValidateLoweredGraph
│   ├── lowered_graph.h / .cpp         LoweredGraph（immutable）+ 嵌套 LoweredGraph::Builder + ValidateLoweredGraph
│   └── model_compiler.h / .cpp        ModelCompiler / ModelCompileOptions / LoweredModelArtifact
├── execution/         （≈ 本文 planner/：计划构建与 runtime 契约）
│   ├── execution_plan_builder.h / .cpp  ExecutionPlanBuilder + ResolveStateAliasesForExecution
│   ├── execution_node_spec.h             untrusted ExecutionPlanNodeSpec
│   ├── execution_plan.h / state_alias_plan.h / executor.h / layer_runner.h ...
├── backend/           （≈ 本文 kernel/：kernel registry / selector / CPU kernels）
│   ├── kernel_selector.h（转发 base/kernel_selector.h）
│   ├── kernel_registry.h / kernel_descriptor.h / packed_weights.h ...
├── base/              kernel_attrs.h / workspace_types.h / kernel_selector.h（跨模块纯数据契约）
├── graph/             语义 IR（ModelGraph）+ 优化 passes
└── operators/         算子语义契约层（OpType / OperatorSchema / OpParams / Infer*）
```

---

## 56. 推荐实现顺序（🔀 五阶段建议已全部越过，当前状态见 §64）

第一阶段建议只实现最核心骨架：

```text
PrimitiveOpType
    ↓
PrimitiveParams
    ↓
LoweredTensorSpec
    ↓
LoweredValue
    ↓
LoweredNode
    ↓
LoweredGraph
```

---

第二阶段实现：

```text
LoweredGraphBuilder
LoweringValueMap
LoweredGraphVerifier
Dump
```

---

第三阶段实现基础 lowering：

```text
Linear
RmsNorm
Silu
SiluMul
Reshape
Transpose
```

---

第四阶段实现 LLM 特有 Primitive：

```text
QkvLinear
GateUpLinear
kAddRmsNorm
RoPE
Attention
KVCacheUpdate
```

---

第五阶段再连接：

```text
ExecutionPlanner
KernelSelector
WeightPlanner
WorkspacePlanner
MemoryPlanner
```

这样可以确保 Lowering 模块本身保持独立。

---

## 57. 与现有 Graph Pass 的衔接（✅ 已落地：默认 pipeline 注册于 compiler 的 optimize_graph）

当前已实现：

```text
ConstantFoldingPass
QkvLinearFusionPass
GateUpLinearFusionPass
SiluMulFusionPass
AddRmsNormFusionPass
DeadCodeEliminationPass
```

推荐 pipeline：

```text
Graph
 │
 ├── ConstantFolding
 ├── QkvLinearFusion
 ├── GateUpLinearFusion
 ├── SiluMulFusion
 ├── AddRmsNormFusion
 ├── DCE
 │
 ▼
OptimizedGraph
 │
 │ Lowering
 ▼
LoweredGraph
```

注意：

```text
Graph Pass
```

产生的是：

```text
QkvLinear
GateUpLinear
kAddRmsNorm
```

这些仍然是语义算子。

Lowering 再映射：

```text
QkvLinear
    → QkvGemm

GateUpLinear
    → GateUpGemm

kAddRmsNorm
    → AddRmsNorm
```

---

## 58. QKV 完整路径

```text
Graph Builder
────────────────

Linear Q
Linear K
Linear V


        │
        │ QkvLinearFusionPass
        ▼


Optimized Graph
────────────────

QkvLinear
    hidden
    Q/K/V logical weight


        │
        │ Graph Lowering
        ▼


LoweredGraph
────────────────

QkvGemm
    hidden
    composite Q/K/V weight

dtype:
    BF16

shape:
    ...


        │
        │ Execution Planning
        ▼


Implementation Selection
────────────────

candidate:
    AVX512 QKV
    AMX QKV
    Generic 3 × GEMM


        │
        ▼


Selected Implementation
────────────────

AMX QKV GEMM


        │
        ▼


Weight Planning
────────────────

AMX_QKV_K32N16


        │
        ▼


Workspace / Memory Planning
────────────────

workspace
activation buffer
packed weight


        │
        ▼


ExecutionPlan
```

---

## 59. Attention 完整路径

```text
Graph:
    Attention
```

Lowering：

```text
execution_mode == prefill
    → AttentionPrefill

execution_mode == decode
    → AttentionDecode
```

Planner：

```text
AttentionDecode
    ↓
candidate implementations

CPU decode attention kernel
generic decomposition
future specialized kernel
```

因此：

```text
FlashAttention
```

永远属于 Kernel implementation，而不是 Primitive 类型。

---

## 60. Reshape / Transpose 完整路径

## Reshape

```text
Graph:
    Reshape

Lowering:
    View

Planning:
    same storage binding
```

---

## Transpose

```text
Graph:
    Transpose

Lowering:
    Strided View
```

Planning：

```text
next kernel supports strided
    → no copy

next kernel requires dense
    → insert implementation-level reorder
```

这可以避免过早 materialization。

---

## 61. Lowering 模块的最重要设计约束（✅ 已落地）

推荐正式固化以下四条：

### 规则 1

> Lowering 不选择 Concrete Kernel。

---

### 规则 2

> Lowering 不处理任何具体执行资源。

---

### 规则 3

> Lowering 只将模型语义转换为稳定的 Execution Primitive。

---

### 规则 4

> 所有 Kernel-specific implementation decision 统一进入 Execution Planning。

这四条是整个模块边界的核心。

---

## 62. 最终推荐架构

```text
                     Graph IR
                        │
                        │
                 Graph Optimization
                        │
                        ▼
                 Optimized Graph
                        │
                        │
                      Lower
                        │
                        ▼
        ┌────────────────────────────┐
        │        LoweredGraph        │
        │                            │
        │ PrimitiveOp                │
        │ LoweredTensorSpec          │
        │ LayoutConstraint           │
        │ CompositeWeightBinding     │
        │ View                       │
        │ ResourceUse                │
        │ ControlDependency          │
        └──────────────┬─────────────┘
                       │
                       │
               Execution Planning
                       │
                       ▼
        ┌────────────────────────────┐
        │        ExecutionPlan       │
        │                            │
        │ Concrete Kernel            │
        │ Kernel Configuration       │
        │ Physical Layout            │
        │ Packed Weight              │
        │ Workspace                  │
        │ Alias Decision             │
        │ Buffer Allocation          │
        │ Execution Schedule         │
        └──────────────┬─────────────┘
                       │
                       ▼
                    Runtime
```

---

## 63. 结论

AetherMind 的 Lowering 模块应当被定义为一个纯粹的 **Semantic Graph → Execution Primitive Graph** 转换层。

其最终边界是：

```text
Graph Optimization
    =
    semantic transformation

Lowering
    =
    abstraction lowering

Execution Planning
    =
    implementation selection
    +
    resource realization

Runtime
    =
    request binding
    +
    execution
```

因此：

```text
Graph IR
```

负责：

> 模型在数学和语义层面需要完成什么计算。

```text
LoweredGraph
```

负责：

> 这些计算由哪些稳定的执行 Primitive 表达。

```text
ExecutionPlan
```

负责：

> 每个 Primitive 最终使用哪个 Kernel、什么物理布局以及哪些执行资源。

这一设计能够最大程度降低 Graph IR、Kernel 实现、内存规划和 Runtime 之间的耦合，也最适合作为 AetherMind 后续支持 AVX2、AVX-512、AMX、量化 Kernel、CUDA 后端和多种执行策略的长期架构基础。

---

## 64. 实现特有机制（v1.1 新增，代码事实）

本章描述当前实现中已落地、但 v1.0 设计未覆盖的核心机制，作为 §0 对照表的展开。

### 64.1 LoweredGraph 产物结构

```cpp
// include/aethermind/compiler/lowered_graph.h
struct LoweredStepSpec {        // 每步的执行规格（不携带 workspace_requirement）
    OpType op_type;             // 语义 OpType 直沉，非 PrimitiveOpType
    KernelSelector selector;    // base 层纯数据：device/isa/weight_format/phase/act_dtype/weight_dtype
    std::vector<TensorSpec> input_specs;   // 完整 schema 端口序，含 state 端口
    std::vector<TensorSpec> output_specs;
    std::vector<ShapeConstraint> runtime_checks;  // 图构建期推导，透传不重推断
    OpParams op_params;
};
struct LoweredStepBinding {     // 图值绑定：node + input_values/output_values（schema 端口序）
    GraphNodeId node;
    std::vector<GraphValueId> input_values;
    std::vector<GraphValueId> output_values;
};
struct LoweredStep {            // 1:1 配对，类型级不变量
    LoweredStepSpec spec;
    LoweredStepBinding binding;
};
struct LoweredValueDesc {       // 按 GraphValueId 稠密索引的 value 元数据
    TensorSpec spec;
    GraphValuePayload payload;  // ModelInput/Activation/Weight/Constant/State
    QuantizationSpec quantization;
    std::string name;
};

class LoweredGraph {            // immutable：私有 storage + const span accessor
public:
    std::span<const LoweredStep> steps() const;
    std::span<const LoweredValueDesc> values() const;
    std::span<const GraphValueId> model_inputs() const;
    std::span<const GraphValueId> model_outputs() const;
    std::span<const LoweredStateAlias> state_aliases() const;
    class Builder;              // 唯一构造路径 + 测试 seam（见 §64.2）
};
```

契约要点：

- **1:1 配对**：spec 与 binding 在同一 `LoweredStep` 中，平行向量漂移不可能；
- **端口序**：input/output 向量按 `OperatorSchema` 端口顺序（M5 起端口 index 即向量位置，无独立 index 字段）；
- **compact 视图**：运行时 tensor 绑定由 `MakeCompactInputSpecs` 按 `contributes_tensor_spec` 派生，state 端口不进入 kernel 输入；
- **不变量**：不携带 workspace/kernel/layout/packed-weight——`LoweredStepSpec` 无 workspace 字段，selector 仅 base 层属性。

### 64.2 LoweredGraph::Builder 构造模式

实现采用嵌套 Builder（原独立 `LoweredGraphDraft` 折叠而来）：

```cpp
class LoweredGraph::Builder {   // 嵌套类天然可访问私有 storage，无需 friend
public:
    std::vector<LoweredStep> steps;
    std::vector<LoweredValueDesc> values;
    std::vector<GraphValueId> model_inputs;
    std::vector<GraphValueId> model_outputs;
    std::vector<LoweredStateAlias> state_aliases;

    Status Validate() const;                 // 校验累积结构，不消费
    StatusOr<LoweredGraph> Build() &&;       // 校验 + move 冻结，消费自身
};
```

- `LowerModelGraph` 是唯一生产路径（build → validate → freeze 三段式）；
- Builder 同时是畸形 artifact 注入的测试 seam（`LoweredGraphBuilder.*` 测试套件）。

### 64.3 State Alias 机制（KV Cache 的 must-alias 建模）

实现放弃 v1.0 的独立 Resource 图建模，改为「schema 声明 + 坐标记录 + 执行期校验」三段式：

1. **声明**：算子在其 `OperatorSchema::state_alias_ports` 中声明（输入端口名, 输出端口名）对（如 kKVCacheUpdate 声明 k_cache_in→k_cache_out、v_cache_in→v_cache_out）。新增有状态算子仅需声明，lowering 与执行零改动；
2. **记录**：`LowerModelGraph` 按拓扑序逐节点消费 schema 声明，以当时已知的 step 索引与端口位置生成 `LoweredStateAlias{step_index, input_port, output_port, input, output}`——不扫描不匹配；
3. **校验与解析**：compiler 侧 `ValidateLoweredGraph` 校验（声明存在、端口 kind==kState、绑定值一致、payload 为 StateValue、state binding 同槽、无重复/冲突），失败为 Internal 错误；execution 侧 `ResolveStateAliasesForExecution`（execution-private）在 trust boundary 重查后按 (step_index, input_port, output_port) 确定性排序为 `StateAliasPlan`，`ForStep()` 二分查询。

### 64.4 ValidateLoweredGraph 检查项（当前实现）

对应 v1.0 §50，实际检查项：

- **步骤结构**：schema 注册存在、GraphNodeId 无重复、input/output binding 与 spec 的 arity 均等于 schema 端口数；
- **selector dtype**：act_dtype / weight_dtype 均非 Undefined；
- **值一致性**：所有绑定 GraphValueId 合法，且 step spec 与 `LoweredValueDesc.spec` 逐位相等；
- **模型 I/O**：model_inputs 均携带 `ModelInputValue` payload；model_outputs 值 id 合法；
- **state alias**：见 §64.3 第 3 点。

未实现（v1.0 §50.5/§50.6）：View 合法性、Resource 合法性（无对应概念）、DAG 环检测（拓扑序天然无环，且 `ValidateAndTopologicalOrder` 已在上游保证）。

### 64.5 dtype 推导契约与配置

`LowerModelGraph` 对每个 step 按以下契约推导 selector dtype：

- **act_dtype**：首个 contributes_tensor_spec 的 activation 输入端口 → 无 activation 输入时回退首个 activation 输出端口；
- **weight_dtype**：首个 contributes_tensor_spec 的 weight 输入端口 → 无 weight 输入时回退 act_dtype；
- **schema 契约**：每个算子必须暴露至少一个 activation 输入或输出端口（`EveryOperatorSchemaExposesActivationPort` 测试守卫），否则 selector 将携带 Undefined dtype；
- **配置**：`GraphLoweringConfig{KernelSelector}` 提供 device/isa/weight_format/phase 前缀（默认 CPU/Scalar/Plain/Both），dtype 由 lowering 推导覆盖——设计 v1.0 的 `LoweringContext{execution_mode, shape_env, model_context}` 未实现，后续需要模型级上下文时再扩展。

### 64.6 未来演进触发条件

- **Primitive 化**（§3/§8/§9）：出现跨算子重排、schedule 化、backend 差异化展开需求时引入 `PrimitiveOpType` + per-op lowering rule；
- **View 体系**（§22/§43/§44）：需要 stride 级布局表达（如 strided kernel 复用）时引入 `LayoutConstraint`；
- **Prefill/Decode 分裂图**（§31/§41）：单图 + phase selector 无法满足 kernel 差异时再分裂；
- **LoweredGraph::Dump()**（§54）：诊断需求出现时按 v1.0 示例实现；
- **P2 接线**：runtime tensor/state binding 与 graph-driven weight materialization 落地后，`LoweredStepBinding`/`LoweredValueDesc` 的消费面补齐。
