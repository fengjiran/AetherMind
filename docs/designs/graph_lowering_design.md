# AetherMind Graph Lowering 模块设计方案

> 版本：v1.0  
> 目标语言：C++20  
> 适用范围：AetherMind 大模型推理引擎 Graph IR → LoweredGraph 转换  
> 设计目标：建立清晰、可扩展、与 Kernel 选择及 Execution Plan Generation 解耦的 Lowering 层

---

## 1. 设计背景

在大模型推理引擎中，Graph IR 通常用于表达模型计算的高层语义，例如：

- `Linear`
- `QkvLinear`
- `GateUpLinear`
- `Attention`
- `RmsNorm`
- `FusedAddRmsNorm`
- `SiluMul`
- `RoPE`
- `Reshape`
- `Transpose`
- `KVCacheUpdate`

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

# 2. 总体架构

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

# 3. Lowering 的正式定义

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

# 4. Lowering 与其他阶段的边界

## 4.1 Graph Optimization

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

## 4.2 Lowering

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

FusedAddRmsNorm
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

## 4.3 Execution Plan Generation

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

## 4.4 Runtime

负责：

- 本次请求输入绑定；
- KV Cache 实例绑定；
- 当前 token position；
- 当前 sequence length；
- dynamic runtime parameter；
- 按 ExecutionPlan 执行。

---

# 5. 职责划分表

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

# 6. LoweredGraph 的定位

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

# 7. LoweredGraph 核心设计原则

## 7.1 LoweredGraph 不保存 KernelId

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

## 7.2 LoweredGraph 不保存 WorkspaceRequirement

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

## 7.3 LoweredGraph 不保存最终 Physical Layout

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

## 7.4 LoweredGraph 不保存 PackedWeight Format

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

## 7.5 LoweredGraph 保留 View 的 mandatory alias 语义

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

# 8. Execution Primitive

## 8.1 PrimitiveOpType

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

# 9. Graph Operator 与 Primitive 的映射

推荐映射：

| Graph IR Operator | Lowered Primitive |
|---|---|
| Linear | Gemm |
| QkvLinear | QkvGemm |
| GateUpLinear | GateUpGemm |
| RmsNorm | RmsNorm |
| FusedAddRmsNorm | AddRmsNorm |
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

# 10. Primitive 参数设计

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

# 11. 典型 Primitive 参数

## 11.1 GemmParams

```cpp
struct GemmParams {
    bool transpose_a = false;
    bool transpose_b = true;

    bool has_bias = false;
};
```

---

## 11.2 QkvGemmParams

```cpp
struct QkvGemmParams {
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    bool has_bias = false;
};
```

---

## 11.3 RmsNormParams

```cpp
struct RmsNormParams {
    float epsilon;
};
```

---

## 11.4 AttentionParams

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

## 11.5 RopeParams

```cpp
struct RopeParams {
    uint32_t rotary_dim;

    float theta;

    RopeScalingType scaling_type;
};
```

---

# 12. LoweredGraph 数据模型

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

# 13. 强类型 ID

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

# 14. LoweredNode

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

# 15. LoweredValue

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

# 16. LoweredTensorSpec

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

# 17. LayoutConstraint

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

# 18. LoweredValueOrigin

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

# 19. Weight 在 LoweredGraph 中的表示

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

# 20. LoweredWeightSpec

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

# 21. CompositeWeightBinding

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

# 22. View Primitive

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

# 23. Mandatory Alias 与 Optional In-place

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

# 24. 持久化 Resource

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

# 25. LoweredResource

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

# 26. KV Cache ResourceSpec

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

# 27. ResourceUse

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

# 28. Control Dependency

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

# 29. DebugOrigin

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

# 30. LoweredGraphMetadata

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

# 31. Prefill / Decode Specialization

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

# 32. LoweringContext

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

# 33. GraphLowerer

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

# 34. LoweringRegistry

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

# 35. LoweredGraphBuilder

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

# 36. Value Mapping

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

# 37. Lowering Result 不要求 1:1

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

# 38. Linear Lowering

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

# 39. QkvLinear Lowering

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

# 40. GateUpLinear Lowering

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

# 41. Attention Lowering

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

# 42. AddRmsNorm Lowering

Graph：

```text
FusedAddRmsNorm
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

# 43. Reshape Lowering

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

# 44. Transpose Lowering

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

# 45. Layout 的两阶段处理

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

# 46. Weight Packing 的两阶段处理

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

# 47. Kernel Selection 接口属于 Planner

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

# 48. Implementation 可以是单 Kernel 或 Kernel Sequence

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

# 49. LoweredGraph Verifier

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

# 50. Verifier 检查项

## 50.1 SSA

每个 Internal Value：

```text
必须且只有一个 producer
```

GraphInput / Weight / Constant：

```text
不得有 producer
```

---

## 50.2 Primitive Arity

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

## 50.3 DType 约束

例如：

```text
RmsNorm:
    floating type

Embedding index:
    integer type
```

---

## 50.4 Shape 约束

例如：

```text
QkvGemm:
    input hidden dimension
    与 weight K dimension 一致
```

---

## 50.5 View 合法性

检查：

- rank；
- element count；
- dtype size；
- byte offset；
- stride expression；
- storage range 关系。

---

## 50.6 Resource 合法性

例如：

```text
AttentionDecode
```

必须使用合法 KVCache Resource。

---

## 50.7 DAG

检查：

```text
data dependency
+
control dependency
```

不能产生非法 cycle。

---

# 51. LoweredGraph 不变量

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

# 52. LoweredGraph Forbidden List

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

# 53. LoweredGraph 应包含

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

# 54. Dump 设计

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

# 55. 推荐目录结构

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

---

# 56. 推荐实现顺序

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
FusedAddRmsNorm
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

# 57. 与现有 Graph Pass 的衔接

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
FusedAddRmsNorm
```

这些仍然是语义算子。

Lowering 再映射：

```text
QkvLinear
    → QkvGemm

GateUpLinear
    → GateUpGemm

FusedAddRmsNorm
    → AddRmsNorm
```

---

# 58. QKV 完整路径

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

# 59. Attention 完整路径

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

# 60. Reshape / Transpose 完整路径

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

# 61. Lowering 模块的最重要设计约束

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

# 62. 最终推荐架构

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

# 63. 结论

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
