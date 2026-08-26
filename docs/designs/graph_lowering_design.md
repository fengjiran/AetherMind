# Graph Lowering 模块设计方案

## 1. 概述

Lowering 位于 compiler 模块，将优化后的语义 `ModelGraph` 转换为不可变的 compiler artifact `LoweredGraph`，再由 execution 模块消费构建 `ExecutionPlan`。步骤类型直接复用算子语义类型（OpType），执行差异通过 base 层 `KernelSelector` 表达，kernel 选择完全推迟到 ExecutionPlanBuilder。

### 1.1 三层链路

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

### 1.2 模块结构与关键类型

| 模块 | 关键类型 / 入口 | 职责 |
|---|---|---|
| compiler | `LowerModelGraph` / `GraphLoweringConfig` | 语义图 → LoweredGraph 的转换；步骤类型为 OpType 直沉；派生 selector dtype |
| compiler | `LoweredGraph` + 嵌套 `LoweredGraph::Builder` | 不可变产物：LoweredStep[] / LoweredValueDesc[] / model I/O / state_aliases；构造态负责校验与冻结 |
| compiler | `ValidateLoweredGraph` | 产物结构不变量校验（schema arity、绑定、值表、selector dtype、state alias） |
| compiler | `OperatorSchema::state_alias_ports` | 有状态算子声明 must-alias 端口对，lowering 与 execution 零特判 |
| execution | `ExecutionPlanBuilder::Build` / `ResolveStateAliasesForExecution` | 消费 LoweredGraph，验证 artifact、解析 state aliases、resolve kernel、规划 workspace |
| execution | `ExecutionPlanNodeSpec` | untrusted 执行请求：测试/手工构造路径，经 InferOperator 重验证 |
| base | `KernelSelector`（含 device/isa/weight_format/phase/dtype） | 跨模块执行请求描述符：lowering 记录，execution/backend 匹配 kernel |

### 1.3 Lowering 的职责

- **转换**：按拓扑序将每个语义节点直沉为一个 `LoweredStep`（`LoweredStepSpec` + `LoweredStepBinding` 1:1 配对）；
- **执行属性**：将 `GraphLoweringConfig.KernelSelector` 前缀写入每步，并按 dtype 推导契约填充 act_dtype/weight_dtype；
- **状态别名**：消费 `OperatorSchema::state_alias_ports` 声明，以已知的 step/port 坐标记录 `LoweredStateAlias`；
- **校验**：`LoweredGraph::Builder::Build()` 经 `ValidateLoweredGraph` 冻结产物，失败为 Internal 错误；
- **不越界**：不选择 kernel、不计算 workspace、不决定物理布局与 packed weight 格式——这些属于 execution/backend 决策。

### 1.4 LoweredGraph 的定位

LoweredGraph 是一个：

> **Kernel-unbound、resource-unbound、primitive-level execution graph。**

它不再保留模型框架级高层 Operator，但仍然不包含具体 Kernel 实现和运行时资源。

```text
Graph IR
    model semantic

LoweredGraph
    execution primitive semantic

ExecutionPlan
    concrete implementation
```

---

## 2. 设计背景与正式定义

### 2.1 设计背景

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

> **将经过优化的模型语义 Graph IR 转换为与模型框架无关、但仍未绑定具体 Kernel 实现和执行资源的步骤（LoweredStep）图。**

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
    由哪些执行步骤（LoweredStep）完成？

ExecutionPlan：
    具体使用哪个 Kernel，以及如何配置执行资源？
```

### 2.2 正式定义

AetherMind 中 Graph Lowering 定义为：

> **Graph Lowering 将经过 Graph Optimization 的语义 Graph IR 转换为由步骤（`LoweredStep`）构成的不可变 compiler artifact `LoweredGraph`。步骤类型直接复用算子语义类型（OpType），不引入独立的执行原语层；该阶段负责携带已验证的语义元数据（specs、dtype、runtime_checks）与图值绑定，并以声明式方式记录状态别名；但不负责选择具体 Kernel、Workspace 计算、最终物理 Layout、权重打包格式和内存规划。**

其核心转换关系为：

```text
Semantic Operator（OpType）
        ↓
LoweredStep（OpType 直沉 + KernelSelector 属性）
```

而不是：

```text
Semantic Operator
        ↓
Concrete Kernel
```

具体 Kernel 选择推迟到 ExecutionPlanBuilder。

---

## 3. 总体架构与数据流

### 3.1 编译与执行流程

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
│ LoweredStep[]                │
│ LoweredValueDesc[]           │
│ KernelSelector               │
│ StateAlias                   │
│ Weight Binding               │
│ Node 溯源                    │
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

### 3.2 最终架构

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
        │ LoweredStep[]              │
        │   (LoweredStepSpec +       │
        │    LoweredStepBinding)     │
        │ LoweredValueDesc[]         │
        │ model_inputs / model_outputs│
        │ StateAlias                 │
        │ KernelSelector             │
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

## 4. 阶段边界与职责

### 4.1 Graph Optimization

负责改变 Graph 的语义表达方式：

```text
Linear(q) + Linear(k) + Linear(v)
        ↓
QkvLinear
```

包括：Constant Folding、QKV Fusion、GateUp Fusion、SiluMul Fusion、AddRmsNorm Fusion、DCE、其他语义级 Rewrite。Graph Optimization 不关心具体 Kernel。

### 4.2 Lowering

将每个语义算子直沉为一个 `LoweredStep`（1:1），步骤类型即 OpType，不进行算子拆分或重组：

```text
Linear        → Linear 步骤
QkvLinear     → QkvLinear 步骤
GateUpLinear  → GateUpLinear 步骤
Attention     → Attention 步骤
kAddRmsNorm   → kAddRmsNorm 步骤
kRmsNorm      → kRmsNorm 步骤
...           → 同 OpType 步骤
```

保证 `1 Semantic Op → 1 LoweredStep`。执行差异（device/ISA/weight format/phase/dtype）通过 `KernelSelector` 属性表达，不选择 AVX2、AVX-512、AMX、CUDA 等具体 Kernel。

### 4.3 Execution Plan Generation

负责 Kernel implementation selection、Kernel configuration、Workspace Requirement、Physical Layout、Weight Packing、Layout legalization、In-place decision、Execution scheduling、Liveness、Buffer reuse、Activation planning、Workspace planning、KernelInvocation materialization。

### 4.4 Runtime

负责本次请求输入绑定、KV Cache 实例绑定、当前 token position、当前 sequence length、dynamic runtime parameter、按 ExecutionPlan 执行。

### 4.5 职责划分表

| 功能 | Graph Pass | Lowering | Execution Plan | Runtime |
|---|---:|---:|---:|---:|
| Constant Folding | ✓ | | | |
| QKV Pattern Fusion | ✓ | | | |
| GateUp Fusion | ✓ | | | |
| AddRmsNorm Fusion | ✓ | | | |
| DCE | ✓ | | | |
| Semantic Op → LoweredStep（1:1 直沉） | | ✓ | | |
| ExecPhase 属性表达 Prefill / Decode 差异 | | ✓ | | |
| State alias 记录（must-alias） | | ✓ | | |
| 持久化状态语义（StateValue） | | ✓ | | |
| Shape / DType 约束（TensorSpec + runtime_checks 透传） | | ✓ | | |
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

### 4.6 最重要设计约束

1. Lowering 不选择 Concrete Kernel。
2. Lowering 不处理任何具体执行资源。
3. Lowering 只将模型语义转换为稳定的步骤（LoweredStep）。
4. 所有 Kernel-specific implementation decision 统一进入 Execution Planning。

这四条是整个模块边界的核心。

---

## 5. LoweredGraph 核心原则

### 5.1 不保存 KernelId

错误设计：

```cpp
struct LoweredStepSpec {
    KernelId kernel_id;
};
```

正确：

```cpp
struct LoweredStepSpec {
    OpType op_type;
};
```

因为 Kernel 选择属于 ExecutionPlanBuilder。

### 5.2 不保存 WorkspaceRequirement

Workspace 是 Kernel implementation 的执行资源需求。LoweredGraph 尚未选择 Kernel，因此不应该出现 `WorkspaceRequirement workspace;`。Planner 在选定 Kernel 后 `KernelDescriptor → QueryWorkspace() → WorkspaceRequirement`。

### 5.3 不保存最终 Physical Layout

不同 Kernel 可能要求 `K32N16` / `AMX_TILE` / `RowMajor` 等不同布局。LoweredGraph 只保存 `LayoutConstraint`（当前实际为无约束），而非 `PhysicalLayout`。

### 5.4 不保存 PackedWeight Format

例如 QKV 权重只需要表达 `Composite Q/K/V Weight used by QkvGemm`，Planner 在选择 Kernel 后决定 `QKV_K32_N16` / `AMX_QKV_TILE` / `INT4_GROUPWISE_K64` 等具体格式。

### 5.5 保留 View 的 mandatory alias 语义

例如 `Reshape(A) → B` 若为纯 metadata operation 则 `B MUST alias A`，属于 LoweredGraph 语义；而 Kernel 的 `output MAY alias input` 属于 capability，由 Planner 决定。LoweredGraph 不保存 `InplaceCandidate`。

---

## 6. 步骤类型与 1:1 对应

### 6.1 步骤类型

`LoweredStepSpec.op_type` 直接复用算子语义类型 `OpType`（19 个枚举，见 §2.1），不存在独立的执行原语类型层：

```cpp
OpType op_type = OpType::kUnknown;
```

算子拆分为多个执行步骤（如 Attention 按 phase 分裂）由 Kernel 选择与运行时 phase 语义承担，Lowering 不进行此类拆分。

### 6.2 语义算子与步骤的对应

| Graph IR Operator（OpType） | LoweredStep 类型 |
|---|---|
| kLinear | kLinear |
| kQkvLinear | kQkvLinear |
| kGateUpLinear | kGateUpLinear |
| kRmsNorm | kRmsNorm |
| kAddRmsNorm | kAddRmsNorm |
| kSilu | kSilu |
| kSiluMul | kSiluMul |
| kAttention | kAttention |
| kRoPE | kRoPE |
| kSoftmax | kSoftmax |
| kEmbedding | kEmbedding |
| kKVCacheUpdate | kKVCacheUpdate |
| kReshape / kPermute / kReorder | 同 OpType 步骤 |

一个 `LoweredStep` 最终由单个 Kernel 实现；kernel 是否存在、如何匹配由 `KernelSelector` 与 backend 注册表决定。保证 `1 Graph Node → 1 LoweredStep`，不存在 0 步或 N 步展开。

### 6.3 各类算子步骤

- **kLinear**：直沉，specs 按 schema 端口序（input, weight）填充。
- **kQkvLinear**：权重为 `QkvWeightBinding`（Q、K、V 拼接配方），此时不决定 AVX512/AMX/3×GEMM 等实现。
- **kGateUpLinear**：权重为 `GateUpWeightBinding`，单 kernel 或拆分实现由 kernel 选择决定。
- **kAttention**：Prefill/Decode 差异由 `ExecPhase` 属性表达（见 §12.2），`FlashAttention` 是 Kernel 而非步骤类型。
- **kAddRmsNorm**：直沉，融合或拆分由 implementation selection 决定。
- **kReshape / kPermute / kReorder**：直沉，Lowering 不做 stride 分析、不折叠为 View、不选择 copy kernel；布局适配由 execution 决定（见 §10）。

---

## 7. 参数与类型系统

### 7.1 算子参数设计

步骤携带类型化算子参数（typed variant），不使用属性字典：

```cpp
using OpParams = std::variant<std::monostate, RmsNormParams, AttentionParams,
                              EmbeddingParams, KVCacheUpdateParams, AddParams, ...>;
```

原因：类型安全差、易拼写错误、不利于编译期检查/序列化/重构。`LoweredStepSpec.op_params` 在 lowering 时从图节点拷贝，kernel prepare 时按类型提取。

### 7.2 典型算子参数

**RmsNormParams**
```cpp
struct RmsNormParams {
    float eps = 1.0e-5F;
};
```

**AttentionParams**
```cpp
struct AttentionParams {
    uint32_t num_attention_heads = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t head_dim = 0;
};
```

**其他**：`EmbeddingParams` / `KVCacheUpdateParams` / `AddParams` 等定义于 `operators/op_params.h`，与 `OperatorSchema` 一一对应（完整清单见该头文件）。

### 7.3 强类型 ID

```cpp
struct GraphNodeId { uint32_t index = 0; };
struct GraphValueId { uint32_t index = 0; };
```

distinct struct 类型，编译期不可隐式互转。

### 7.4 TensorSpec

value 规格直接使用 shape 推导层的 `TensorSpec`（dtype + `SymbolicShape`），不引入独立的 lowered 规格类型：

```cpp
struct TensorSpec {
    DataType dtype{};
    SymbolicShape shape{};
};
```

### 7.5 Value 来源（payload）

```cpp
using GraphValuePayload = std::variant<std::monostate, ModelInputValue,
                                       ActivationValue, WeightValue,
                                       ConstantValue, StateValue>;
```

模型输出不作为独立的 value 来源；`LoweredGraph::model_outputs()` 保存输出集合。

---

## 8. 数据模型

### 8.1 LoweredGraph 产物结构

```cpp
class LoweredGraph {
public:
    std::span<const LoweredStep> steps() const;
    std::span<const LoweredValueDesc> values() const;
    std::span<const GraphValueId> model_inputs() const;
    std::span<const GraphValueId> model_outputs() const;
    std::span<const LoweredStateAlias> state_aliases() const;
    uint64_t artifact_id() const;
    class Builder;

private:
    std::vector<LoweredStep> steps_;
    std::vector<LoweredValueDesc> values_;
    std::vector<GraphValueId> model_inputs_;
    std::vector<GraphValueId> model_outputs_;
    std::vector<LoweredStateAlias> state_aliases_;
    uint64_t artifact_id_;
};
```

`Builder::Build()` 后 immutable（storage 私有，仅 const span accessor）。完整字段见附录 A.1。

### 8.2 LoweredStep

```cpp
struct LoweredStep {
    LoweredStepSpec spec{};      // 执行规格（op_type/selector/specs/checks/params）
    LoweredStepBinding binding{};// 图值绑定（node/input_values/output_values）
};
```

spec 与 binding 1:1 配对是类型级不变量。不包含 `KernelId/KernelFn/Workspace/Buffer/Arena Offset/PackedWeightHandle`。

### 8.3 LoweredValueDesc

按 `GraphValueId` 稠密索引的 value 元数据表：

```cpp
struct LoweredValueDesc {
    TensorSpec spec{};
    GraphValuePayload payload{};   // WeightValue/ConstantValue/StateValue 等
    QuantizationSpec quantization{};
    std::string name{};
};
```

### 8.4 图值引用

直接保留 `GraphValueId` 引用（`LoweredStepBinding.input_values/output_values`、`model_inputs/model_outputs`），值元数据按 ID 索引 `values()` 表，无独立的 GraphValueId → LoweredValueId 映射表。

---

## 9. 权重表示与绑定

### 9.1 Weight 在 LoweredGraph 中的表示

权重统一作为 value 元数据表中的 `WeightValue` payload（而非节点属性）：

```text
hidden ─────────────┐
                    ▼
               kQkvLinear 步骤
                    ▲
                    │
               qkv_weight（WeightValue payload）
```

### 9.2 Weight 绑定

```cpp
struct WeightValue {
    WeightBinding binding{};   // decoder_layer_index + WeightBindingSpec
};

using WeightBindingSpec = std::variant<DirectWeightBinding,
                                       QkvWeightBinding,
                                       GateUpWeightBinding>;
```

绑定描述逻辑来源（直接权重或 QKV/GateUp 复合配方），不决定 packing format。

### 9.3 复合权重绑定

```cpp
struct QkvWeightBinding {};    // concat(Q, K, V, axis=0)
struct GateUpWeightBinding {}; // concat(Gate, Up, axis=0)
```

配方固定为 Q、K、V 顺序的 output-channel 拼接；拆分边界属于 `QkvLinearParams`，形状属于 `TensorSpec`。LoweredGraph 不决定具体 packing format。

### 9.4 Weight Packing

LoweredGraph 携带逻辑绑定，packed format 由执行规划阶段决定：

```text
LoweredStep + Selected Kernel → required packed format → PackedWeightStore 查找 → PackedWeightHandle
```

例如 `kQkvLinear + AMXQkvKernel → AMX_QKV_K32N16`。

---

## 10. 形状变换与布局约束

### 10.1 布局约束现状

当前 LoweredGraph 不表达布局约束（无 LayoutConstraint/StrideExpr 类型）；物理布局是 kernel 与 execution 规划阶段的决策。

### 10.2 形状变换算子

`kReshape / kPermute / kReorder` 作为普通语义算子直沉，Lowering 不将其折叠为 View，也不表达 stride 级布局约束；其别名/拷贝行为由 kernel 与 execution 规划决定。

### 10.3 布局适配与决策归属

完全属于 execution/backend：

```text
kernel 支持 strided → 直接使用
kernel 要求 contiguous → 执行期插入 reorder
```

Lowering 不提前插入 Kernel-specific reorder，不携带 `concrete physical layout（RowMajor / K32N16 / AMXTile）` 约束，避免与 Kernel implementation 耦合。

---

## 11. 状态与别名语义

### 11.1 Mandatory Alias 与 Optional In-place

- **State Alias（must-alias）**：`state input MUST alias state output`，属于 LoweredGraph 语义，由 `LoweredStateAlias` 记录（见附录 A.3），当前唯一载体是 KV cache。
- **Kernel In-place（may-alias）**：`output MAY alias input`，属于 Kernel capability，由 execution/backend 决定。LoweredGraph 不保存 `InplaceCandidate`。

### 11.2 持久化状态与绑定

KV Cache 建模为 value payload 中的 `StateValue`，辅以 `state_aliases` 表达 must-alias：

```cpp
struct StateValue {
    StateBinding binding{};   // KVCacheStateBinding / ...
};
struct KVCacheStateBinding {
    uint32_t decoder_layer_index;
    KVCacheSlot slot;         // kKey / kValue
};
```

状态具有 persistent / mutable / request-scoped / side-effectful 语义；绑定类型（`KVCacheStateBinding` 等）定义于 `graph_types.h`，由 `ValidateLoweredGraph` 校验 alias 一致性（同槽）。状态值携带 `decoder_layer_index + slot`，不保存 `KV cache pointer / cache length / page table / request id / physical buffer`。

### 11.3 State 使用与顺序约束

状态端口在 schema 中声明（kState，`contributes_tensor_spec = false`），读写语义由算子本身与 step 拓扑序表达，无 `ResourceUse` 记录。顺序由 `LoweredGraph::steps()` 拓扑序线性化保证——`kKVCacheUpdate` 先于消费者，无需 control dependency。

### 11.4 溯源信息

```cpp
struct LoweredStepBinding {
    GraphNodeId node{};   // 源语义节点
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
};
```

用于 Debug、Graph→LoweredGraph 对照、融合问题排查、WeightBinding 错误定位。

---

## 12. 产物元数据与配置

### 12.1 产物元数据

携带实例身份 `artifact_id`（Build 时分配），用于将 packed-weight artifact 与来源图绑定；不存放 `AVX512/AMX/thread count` 等执行属性。

```cpp
uint64_t artifact_id();
```

### 12.2 Prefill / Decode 表达

通过 `ExecPhase`（kPrefill/kDecode/kBoth）作为 `KernelSelector.phase` 属性表达，不生成分裂的图：

```cpp
ExecPhase phase = ExecPhase::kBoth;
```

同一 `LoweredGraph` 可被不同 phase 复用；phase 由 kernel 选择阶段消费。

### 12.3 Lowering 配置

```cpp
struct GraphLoweringConfig {
    KernelSelector selector{
        .device_type = DeviceType::kCPU,
        .weight_format = WeightFormat::kPlain,
        .isa = IsaLevel::kScalar,
        .phase = ExecPhase::kBoth,
    };
};
```

不依赖 `KernelRegistry/MemoryPlanner/WorkspacePlanner/WeightManager`。

---

## 13. 构建与生成规则

### 13.1 LowerModelGraph

```cpp
StatusOr<LoweredGraph> LowerModelGraph(
        const ModelGraph& graph,
        const GraphLoweringConfig& config = {});
```

`LowerModelGraph` 不依赖 `KernelRegistry`，与具体 Kernel 实现完全解耦。

### 13.2 步骤生成规则

统一的 OpType 直沉流程（拓扑序遍历 + schema 端口解释），不存在 per-op lowering rule 注册表：对每个节点按 schema 端口序收集 specs 与绑定、派生 selector dtype、按 `state_alias_ports` 记录别名，保证 `1 Graph Node → 1 LoweredStep`。不存在 0 步或 N 步展开，`kReshape/kPermute/kReorder` 同样直沉。

### 13.3 LoweredGraph::Builder

```cpp
class LoweredGraph::Builder {
public:
    std::vector<LoweredStep> steps;
    std::vector<LoweredValueDesc> values;
    std::vector<GraphValueId> model_inputs;
    std::vector<GraphValueId> model_outputs;
    std::vector<LoweredStateAlias> state_aliases;

    Status Validate() const;
    StatusOr<LoweredGraph> Build() &&;
};
```

嵌套类天然可访问私有 storage，无需 friend。负责累积、校验（`ValidateLoweredGraph`）、冻结与 `artifact_id` 分配；`LowerModelGraph` 是唯一生产路径（build → validate → freeze），Builder 同时是畸形 artifact 注入的测试 seam（`LoweredGraphBuilder.*`）。

---

## 14. Kernel 选择与实现

### 14.1 Kernel 选择接口

基于 `KernelSelector`（base 层纯数据描述符）匹配：

```cpp
struct KernelSelector {
    DeviceType device_type;
    DataType act_dtype;
    DataType weight_dtype;
    WeightFormat weight_format;
    IsaLevel isa;
    ExecPhase phase;
};
```

```text
LoweredStepSpec.selector → Backend::PrepareKernel(op_type, selector, op_params) → ResolvedKernel
```

### 14.2 Implementation 形态

每个步骤最终绑定一个 `ResolvedKernel`（kernel 函数指针 + 冻结参数）。fused primitive 的拆分实现（如单 QKV kernel 与 3 × GEMM）不属于 Lowering；如需拆分由未来 kernel 展开层承担。

---

## 15. 校验与不变量

### 15.1 校验入口

```cpp
Status ValidateLoweredGraph(const LoweredGraph& lowered);
```

由 `LoweredGraph::Builder::Validate()/Build()` 调用，不含 Kernel，不需要 `KernelRegistry`。

### 15.2 检查项

**步骤结构**：schema 注册存在、GraphNodeId 无重复、input/output binding 与 spec 的 arity 均等于 schema 端口数、selector act_dtype/weight_dtype 均非 Undefined。

**值一致性**：所有绑定 GraphValueId 合法，step spec 与 `LoweredValueDesc.spec` 逐位相等；model_inputs 均携带 `ModelInputValue` payload；model_outputs 值 id 合法。

**State Alias**：alias 的 step/port 坐标在范围内、端口 kind 均为 kState、别名对由 schema `state_alias_ports` 声明、绑定值与记录值一致、payload 均为 `StateValue` 且同槽、无重复/冲突。

未覆盖：View 合法性、Resource 合法性（无概念）、DAG 环检测（拓扑序天然无环，上游 `ValidateAndTopologicalOrder` 已保证）。

### 15.3 不变量

1. 步骤类型为语义 OpType（直沉），无独立 primitive 层。
2. 不包含 Concrete Kernel / Workspace / Buffer allocation / PackedWeightHandle / In-place decision / execution schedule / runtime request state。
3. State alias 表示 mandatory alias（唯一载体为 KV cache）。
4. Internal Tensor Value 保持 SSA。
5. 持久化状态显式建模（StateValue + state_aliases）。
6. 步骤序列为合法拓扑序。

### 15.4 Forbidden List

```
× KernelId / KernelFn / AVX2/AVX512/AMX ID
× WorkspaceRequirement / WorkspaceSlice / Buffer* / void* / Arena offset / BufferSlice
× Concrete packed layout / PackedWeightHandle / final inplace binding
× liveness interval / execution schedule / thread/NUMA/stream assignment
× request id / current sequence length / current token position / KVCacheHandle
```

### 15.5 应包含清单

```
✓ LoweredStep（Spec + Binding 1:1）
✓ OpType / KernelSelector / TensorSpec / 完整 schema 端口序 specs / runtime_checks
✓ Graph Input/Output / Weight 逻辑绑定 + 复合配方 / LoweredValueDesc / State alias / 溯源 node / artifact_id
```

---

## 16. 工程结构

### 16.1 目录结构

```text
include/aethermind/ + src/
├── compiler/          优化 + lowering + artifact
│   ├── optimize_graph.h / .cpp        OptimizeModelGraph（O0/O1/O2 pipeline）
│   ├── graph_lowering.h / .cpp        GraphLoweringConfig + LowerModelGraph + ValidateLoweredGraph
│   ├── lowered_graph.h / .cpp         LoweredGraph（immutable）+ 嵌套 Builder + ValidateLoweredGraph
│   └── model_compiler.h / .cpp        ModelCompiler / ModelCompileOptions / LoweredModelArtifact
├── execution/         计划构建与 runtime 契约
│   ├── execution_plan_builder.h / .cpp  ExecutionPlanBuilder + ResolveStateAliasesForExecution
│   ├── execution_node_spec.h             untrusted ExecutionPlanNodeSpec
│   └── execution_plan.h / state_alias_plan.h / executor.h / layer_runner.h ...
├── backend/           kernel registry / selector / CPU kernels
│   ├── kernel_selector.h（转发 base/kernel_selector.h）
│   └── kernel_registry.h / kernel_descriptor.h / packed_weights.h ...
├── base/              kernel_attrs.h / workspace_types.h / kernel_selector.h
├── graph/             语义 IR（ModelGraph）+ 优化 passes
└── operators/         算子语义契约层（OpType / OperatorSchema / OpParams / Infer*）
```

### 16.2 Dump 设计

当前未提供 `Dump()`；诊断基于 `values()` 的 name 与 `LoweredStepBinding.node`。如需文本 dump，可按下列格式扩展：

```text
lowered_graph @artifact_id {
  %0 = input : bf16[1,4096]
  %1 = kQkvLinear(%0, %w0) { q_heads=32, kv_heads=8, head_dim=128 }
  %2 = kAttention(%1) alias(k_cache_in → k_cache_out)
}
```

---

## 17. 实现状态与衔接

### 17.1 当前实现状态

已实现：`LowerModelGraph` 拓扑序直沉、selector dtype 推导（M6 契约）、schema 声明式 state alias 记录、`ValidateLoweredGraph`、Builder 冻结与 artifact_id、`ExecutionPlanBuilder::Build(LoweredGraph)` 消费路径（kernel resolve + workspace 规划）。

待推进：runtime tensor/state binding 接线（P2）与 graph-driven weight materialization。

### 17.2 与 Graph Pass 的衔接

已实现 passes：`ConstantFoldingPass / QkvLinearFusionPass / GateUpLinearFusionPass / SiluMulFusionPass / AddRmsNormFusionPass / DeadCodeEliminationPass`

默认 O2 pipeline：

```text
Graph → ConstantFolding → QkvLinearFusion → GateUpLinearFusion → SiluMulFusion → AddRmsNormFusion → DCE → OptimizedGraph → Lowering → LoweredGraph
```

Graph Pass 产生的是语义算子（`QkvLinear / GateUpLinear / kAddRmsNorm`），Lowering 直沉为同名步骤。

---

## 18. 端到端路径示例

### 18.1 QKV 完整路径

```text
Graph Builder: Linear Q/K/V
    ↓ QkvLinearFusionPass
Optimized Graph: QkvLinear(hidden, Q/K/V logical weight)
    ↓ Graph Lowering
LoweredGraph: kQkvLinear(hidden, composite weight, BF16, shape...)
    ↓ Execution Planning
candidate: AVX512 QKV / AMX QKV / 3×GEMM → Selected: AMX QKV GEMM → Weight: AMX_QKV_K32N16 → Workspace/Memory → ExecutionPlan
```

### 18.2 Attention 完整路径

```text
Graph: Attention
Lowering: kAttention 步骤（phase 属性表达 prefill/decode 差异）
Planner: kAttention → CPU decode / generic decomposition / specialized kernel
FlashAttention 属于 Kernel implementation，不是步骤类型。
```

### 18.3 形状变换完整路径

```text
Graph: kReshape → Lowering: kReshape 步骤（直沉）→ Planning: kernel/execution 决定布局适配
Graph: kPermute / kReorder → Lowering: 同 OpType 步骤 → Planning: strided 直用或插入 reorder
Lowering 不提前插入 Kernel-specific reorder。
```

---

## 19. 结论

AetherMind 的 Lowering 模块是 **Semantic Graph → LoweredStep Graph** 转换层：

```text
Graph Optimization = semantic transformation
Lowering          = abstraction lowering
Execution Planning = implementation selection + resource realization
Runtime           = request binding + execution
```

- Graph IR 负责模型在数学和语义层面需要完成什么计算；
- LoweredGraph 负责这些计算由哪些稳定的步骤表达；
- ExecutionPlan 负责每个步骤最终使用哪个 Kernel、什么物理布局以及哪些执行资源。

该设计最大程度降低 Graph IR、Kernel 实现、内存规划和 Runtime 之间的耦合，适合作为后续支持 AVX2/AVX-512/AMX、量化、CUDA 及多种执行策略的长期架构基础。

---

## 附录 A 实现机制详述

与 §1 / §8 / §15 相互印证。

### A.1 产物结构

```cpp
// include/aethermind/compiler/lowered_graph.h
struct LoweredStepSpec {        // 不携带 workspace_requirement
    OpType op_type;
    KernelSelector selector;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    std::vector<ShapeConstraint> runtime_checks;
    OpParams op_params;
};
struct LoweredStepBinding {     // schema 端口序
    GraphNodeId node;
    std::vector<GraphValueId> input_values;
    std::vector<GraphValueId> output_values;
};
struct LoweredStep {            // 1:1 配对
    LoweredStepSpec spec;
    LoweredStepBinding binding;
};
struct LoweredValueDesc {
    TensorSpec spec;
    GraphValuePayload payload;
    QuantizationSpec quantization;
    std::string name;
};
class LoweredGraph {            // immutable
public:
    std::span<const LoweredStep> steps() const;
    std::span<const LoweredValueDesc> values() const;
    std::span<const GraphValueId> model_inputs() const;
    std::span<const GraphValueId> model_outputs() const;
    std::span<const LoweredStateAlias> state_aliases() const;
    class Builder;
};
```

契约：1:1 配对、端口序即向量位置、compact 视图按 `contributes_tensor_spec` 派生、不携带 workspace/kernel/layout/packed-weight。

### A.2 Builder 构造模式

```cpp
class LoweredGraph::Builder {
public:
    std::vector<LoweredStep> steps;
    std::vector<LoweredValueDesc> values;
    std::vector<GraphValueId> model_inputs;
    std::vector<GraphValueId> model_outputs;
    std::vector<LoweredStateAlias> state_aliases;
    Status Validate() const;
    StatusOr<LoweredGraph> Build() &&;
};
```

`LowerModelGraph` 为唯一生产路径（build → validate → freeze）；Builder 同时是畸形 artifact 注入的测试 seam（`LoweredGraphBuilder.*`）。

### A.3 State Alias 机制

1. **声明**：`OperatorSchema::state_alias_ports` 声明（输入端口名, 输出端口名）对；
2. **记录**：`LowerModelGraph` 按拓扑序逐节点消费，以 step 索引与端口位置生成 `LoweredStateAlias{step_index, input_port, output_port, input, output}`；
3. **校验与解析**：compiler 侧 `ValidateLoweredGraph` 校验（声明存在、kind==kState、绑定一致、payload 为 StateValue、同槽、无重复），execution 侧 `ResolveStateAliasesForExecution` 排序为 `StateAliasPlan`，`ForStep()` 二分查询。

### A.4 校验检查项

对应 §15.2：步骤结构、selector dtype、值一致性、模型 I/O、state alias（见 A.3），未覆盖 View/Resource/DAG 环检测。

### A.5 dtype 推导契约与配置

- act_dtype：首个 contributes 的 activation 输入 → 无则回退首个 activation 输出；
- weight_dtype：首个 contributes 的 weight 输入 → 无则回退 act_dtype；
- 每个算子必须暴露至少一个 activation 端口（`EveryOperatorSchemaExposesActivationPort` 守卫）；
- `GraphLoweringConfig{KernelSelector}` 提供 device/isa/weight_format/phase 前缀，dtype 由 lowering 推导覆盖。

### A.6 后续演进

- 步骤展开层、布局约束类型、Prefill/Decode 分裂、`LoweredGraph::Dump()`、P2 runtime 接线与 graph-driven weight materialization。
