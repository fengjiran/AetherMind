# ModelGraph 设计方案

## 1. 目标定位

`ModelGraph` 是 **后端无关、设备无关、内存无关的语义 DAG IR**。

它回答的问题是：

> 模型在语义上由哪些计算组成，哪些 value 在节点之间流动，权重/常量/状态如何被引用，shape/type/量化语义是什么。

它不回答的问题是：

> 这次在哪个设备执行、使用哪个 kernel、权重如何 pack、workspace 多大、buffer 如何复用、是否走 tensor parallel / pipeline parallel。

这些执行相关决策应放在 lowering / planning / backend-specific IR 中。

## 2. 架构分层

```text
ModelGraph
  - 语义 DAG
  - stable node/value ids
  - logical tensor/value specs
  - weight / constant / state references
  - op semantic params
  - validation metadata

        ↓ semantic passes

GraphPassPipeline
  - canonicalize
  - shape inference / constraint propagation
  - constant folding
  - semantic pattern marking
  - rewrite-friendly fusion representation

        ↓ lowering

LoweredGraph / GraphPlan
  - backend / device / execution-mode selection
  - device placement
  - sharding / partitioning strategy
  - packed weight mapping
  - workspace planning
  - inserted communication / resource ops

        ↓ build

ExecutionPlan / ExecutionStep
  - topological execution schedule
  - resolved operator / kernel
  - runtime checks
  - workspace offsets
  - packed weight handles
  - executor-facing debug info
```

核心原则：**ModelGraph 是模型语义；ExecutionPlan 是某次部署和请求上下文下的可执行计划。**

## 3. 明确边界

### 3.1 属于 ModelGraph

- 语义算子：`Embedding`、`RMSNorm`、`Linear`、`RoPE`、`MatMul`、`Softmax`、`Add`、`SiluMul`、`Argmax` 等；
- 数据流 use-def：node inputs / outputs、producer，以及按需从 node inputs 派生的 consumers 索引；
- 逻辑 tensor/value 类型：dtype、rank、symbolic shape、dynamic shape 约束；
- value role：model input、activation、weight、constant、state / resource；
- 语义参数：epsilon、axis、head count、RoPE config、activation type、quantization scheme；
- 权重逻辑引用：checkpoint key、role、layer index、shape、dtype；
- debug 信息：stable id、debug name、decoder layer index、source location；
- pass metadata：用于 rewrite / validation / debug 的语义标记。

### 3.2 不属于 ModelGraph

- `WorkspaceRequirement`；
- `KernelSelector`；
- device / backend / ISA；
- packed weight pointer / packed weight layout；
- buffer allocation / memory offset / reuse plan；
- thread pool / stream / NUMA policy；
- concrete fused kernel choice；
- distributed placement / sharding / collective schedule；
- runtime batching strategy。

这些内容属于 `LoweredGraph`、`GraphPlan`、`ExecutionPlan` 或 backend-specific IR。

## 4. ID 类型

```cpp
struct GraphNodeId {
    uint32_t index = 0;
};

struct GraphValueId {
    uint32_t index = 0;
};
```

可以使用 `std::vector` 下标作为 ID，但必须满足两个约束：

1. graph 构建期间 append-only，或由 builder 统一管理 ID 重写；
2. graph 构造完成后默认 immutable，rewrite pass 默认通过 mutation patch 生成新 graph；只有受控 mutable API 才允许原地更新 use-def。

如果需要跨进程序列化、增量编译或稳定 debug replay，可以在 metadata 中增加 stable textual id，但不要替代轻量的 index ID。

## 5. GraphValuePayload

GraphValue 的数据流实体类型通过 `std::variant` 表达，variant 的类型本身即为 tag，无需独立 `GraphValueKind` enum。这样 kind 与 binding 的一致性由类型系统在编译期保证，无需运行时校验。
`GraphValuePayload` 是 value category 与 category-specific binding 的唯一来源，不应再并行保存 kind enum。

```cpp
struct ModelInputValue {};
struct ActivationValue {};
struct WeightValue {
    WeightBinding binding{};
};
struct ConstantValue {
    ConstantBinding binding{};
};
struct StateValue {
    StateBinding binding{};
};
struct ResourceValue {
    ResourceBinding binding{};
};

using GraphValuePayload = std::variant<
        std::monostate,
        ModelInputValue,
        ActivationValue,
        WeightValue,
        ConstantValue,
        StateValue,
        ResourceValue>;
```

语义说明：

- `std::monostate`：构造期非法哨兵，不是一种真实 graph value category；`Validate()` 必须拒绝它；
- `ModelInputValue`：请求输入，例如 `token_ids`、position ids、attention mask；
- `ActivationValue`：普通中间激活；
- `WeightValue`：模型权重的逻辑引用，携带 `WeightBinding`，不拥有真实权重 payload；
- `ConstantValue`：编译期常量，携带 `ConstantBinding`，例如标量、固定 mask、量化 scale/zero-point table。注意 RoPE sin/cos table 属于派生常量（可从 `RoPEParams` 完全确定），应在 lowering 阶段生成，不应作为 `ConstantValue` 输入；
- `StateValue`：跨 step 持久状态，携带 `StateBinding`，例如 KV cache、decode state、streaming state；
- `ResourceValue`：外部 runtime resource 的逻辑占位，携带 `ResourceBinding`，例如 paged attention handle、外部分配 buffer handle；实际落地应晚于 `StateValue`，等存在明确 runtime handle use case 后再实现。

KV cache 不应伪装成普通 activation，也不应在语义图中直接表达为 runtime resource handle。它的生命周期跨 token / request step，serving、paged attention、continuous batching 都会依赖显式 state 语义。ModelGraph 中的 KV cache 应优先建模为版本化 `StateValue`：每次状态更新消费一个 state input，并产生一个新的 state output。

若需要序列化 / debug 输出，用 helper 将 variant index 映射为字符串或显式 enum；不要让序列化 tag 反向侵入核心类型设计。

## 6. GraphValue

```cpp
struct GraphValue {
    GraphValuePayload payload{std::monostate{}};

    TensorSpec spec{};

    // nullopt:
    // - ModelInputValue
    // - WeightValue
    // - ConstantValue
    // - StateValue / ResourceValue 的外部入口
    //
    // has value:
    // - ActivationValue produced by a node
    // - node-produced state/resource update
    std::optional<GraphNodeId> producer{};

    QuantizationSpec quantization{};

    std::string debug_name{};
};
```

`payload` 中的 binding 类型是逻辑引用，不是物理存储所有权：

- `WeightBinding` 指向模型权重角色、layer、checkpoint key 或 weight id；
- `ConstantBinding` 描述常量来源和小型常量 payload / external constant id；
- `StateBinding` 描述 KV cache 等状态的逻辑身份；
- `ResourceBinding` 描述外部 runtime resource 的逻辑身份；
- `QuantizationSpec` 描述模型语义量化方案，例如 int4/int8、group size、scale dtype、zero point policy。

`QuantizationSpec` 放置在 `GraphValue` 外层而非 `WeightValue` 内部，是为了覆盖 activation 量化（例如 int8 activation）等场景，避免在 variant 每个分支重复定义。

注意：**quantization scheme 属于 ModelGraph；packed weight 属于 backend artifact。**

### 6.1 Binding 与 Quantization 类型草案

以下类型是语义层占位设计，用于让 `ModelGraph` 文档自洽。它们只描述逻辑身份和语义属性，不持有 backend buffer、device handle 或 packed artifact。

```cpp
struct ConstantBinding {
    std::string name{};
    std::vector<std::byte> inline_data{};
};

enum class KVCacheSlot : uint8_t {
    kKey,
    kValue,
};

struct KVCacheStateBinding {
    uint32_t decoder_layer_index = 0;
    KVCacheSlot slot = KVCacheSlot::kKey;
};

struct DecodeStateBinding {
    // Milestone 1 placeholder. Add fields only when decode-state semantics land.
};

struct StreamingStateBinding {
    // Milestone 1 placeholder. Add fields only when streaming-state semantics land.
};

using StateBinding = std::variant<
        KVCacheStateBinding,
        DecodeStateBinding,
        StreamingStateBinding>;

enum class ResourceKind : uint8_t {
    kUnknown,
    kPagedAttentionHandle,
    kExternalBuffer,
    kAllocatorArena,
};

struct ResourceBinding {
    std::string logical_id{};
    ResourceKind kind = ResourceKind::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::string debug_name{};
};

enum class QuantizationKind : uint8_t {
    kNone,
    kInt8,
    kInt4,
};

struct QuantizationSpec {
    QuantizationKind kind = QuantizationKind::kNone;
    uint32_t group_size = 0;
    DataType scale_dtype{};
    bool has_zero_point = false;
};
```

语义边界：

- `ConstantBinding` 是逻辑常量引用，可携带小型 inline 常量；大型常量可以通过 `name` / external id 解析；
- `StateBinding` 是 KV cache、decode state、streaming state 的结构化逻辑身份，不描述具体 cache layout；variant alternative 是 state 类型标签。KV cache 由 `KVCacheStateBinding{decoder_layer_index, slot}` 构成 lowering/runtime 解析 state handle 的 key，其中 `slot` 是强类型 `KVCacheSlot::kKey` / `KVCacheSlot::kValue`，不是字符串；`GraphValue::debug_name` 只用于诊断，不参与绑定解析；
- `ResourceBinding` 是 runtime resource 的结构化逻辑身份，不持有设备句柄；`logical_id` / `kind` / `decoder_layer_index` 共同构成 lowering/runtime 解析 resource handle 的 key，`debug_name` 只用于诊断；它不是早期 KV cache 建模的默认选择，应等 lowering / runtime 出现真实外部资源句柄需求后再落地；
- `QuantizationSpec` 描述模型语义量化，例如 int4/int8、group size、scale dtype、zero point policy，不描述 packed weight 格式。

`name` / `debug_name` 不应作为唯一绑定键。State / resource binding 的解析流程应依赖结构化 logical key：

```text
GraphValue(StateValue{KVCacheStateBinding{decoder_layer_index=3, slot=KVCacheSlot::kKey}})
        ↓ lowering
RuntimeStateRegistry::Resolve(binding)
        ↓
ExecutionPlan state handle / buffer alias group
```

对于 `state_k_cache_in, state_v_cache_in -> KVCacheUpdate -> state_k_cache_out, state_v_cache_out`，输入和输出 state value 可以是不同 `GraphValueId`，但同一 layer、同一 `KVCacheSlot` 的输入/输出 `StateBinding` 应解析到同一个 logical state family；同一 layer 的 K/V state 属于同一个 state collection。lowering 再通过 must-alias 约束绑定到对应物理 buffer / state handle。

## 7. GraphNode

```cpp
struct GraphNode {
    OpType op_type = OpType::kUnknown;

    std::optional<uint32_t> decoder_layer_index{};
    std::string debug_name{};

    // Schema-ordered logical operands.
    // inputs[i] binds to OperatorSchema.inputs[i].
    // Includes model input, activation, weight, constant, state, resource values
    // when the operator schema declares them as input ports.
    std::vector<GraphValueId> inputs{};

    // Schema-ordered logical results.
    // outputs[i] binds to OperatorSchema.outputs[i].
    // Outputs usually include activation values, and may include state/resource updates.
    std::vector<GraphValueId> outputs{};

    // Typed operator parameters. The active execution path carries the same
    // OpParams variant into ExecutionPlanNodeSpec and OperatorRegistry.
    OpParams op_params{};

    // Extension metadata only. Registered operators should prefer op_params.
    ModelGraphAttrs attrs{};
};
```

`GraphNode` 不应包含：

```cpp
std::vector<WeightBinding> weights{};
WorkspaceRequirement workspace_requirement{};
DeviceType device_type{};
IsaLevel isa{};
WeightFormat weight_format{};
KernelSelector selector{};
```

原因：

- weight 已由 `WeightValue` payload 表达；
- workspace、device、ISA、weight format、kernel selector 都是 lowering / planning 决策；
- `decoder_layer_index` 只是 semantic/debug metadata，不是图结构依赖。真实依赖必须由 edges 表达。

### 7.1 Operator schema 输入顺序

`GraphNode.inputs` 是 operator schema 顺序下的完整 logical operand 列表，不是 runtime activation input 列表。它必须包含 operator schema 声明的所有输入端口，包括 activation、model input、weight、constant、state、resource。

默认约定：

```text
GraphNode.inputs[i]  <->  OperatorSchema.inputs[i]
GraphNode.outputs[i] <->  OperatorSchema.outputs[i]
```

这个顺序是 `CheckInputSpecs`、`InferOutputShapes`、runtime binding 和 lowering 派生执行视图的共同语义 ABI。Lowering 和 optimization pass 不允许根据 payload 类型删除、重排或重新解释 `GraphNode.inputs`。

例如 `RMSNorm` 的输入顺序应为：

```text
inputs[0] = hidden activation
inputs[1] = RMSNorm weight
```

`Embedding` 的输入顺序应为：

```text
inputs[0] = token ids
inputs[1] = embedding weight
```

即使 weight 在 execution 层被解析为 plain weight pointer 或 packed weight handle，它在 `ModelGraph` 中仍然是 schema-ordered logical input。

如果需要支持 named、optional 或 variadic ports，可引入显式 port binding：

```cpp
struct GraphNodeInput {
    GraphValueId value{};
    uint32_t port_index = 0;
};

struct GraphNodeOutput {
    GraphValueId value{};
    uint32_t port_index = 0;
};
```

但在没有 named / optional / variadic port 需求时，`std::vector<GraphValueId>` 加 schema 顺序不变量更简单，也与现有 `CheckInputSpecs(std::span<const TensorSpec>)` 兼容。

### 7.2 OperatorSchema 类型

`OperatorSchema` 描述 operator 的语义 ABI：输入输出端口顺序、端口名称、期望 payload 类型和该端口是否参与 tensor spec validation。它用于 graph validation 和 lowering，不表达 backend kernel 选择。

`OperatorSchema` 不是后续优化项，而是 `GraphNode.inputs` / `GraphNode.outputs` 采用 schema 顺序后的前置约束。只要 `ModelGraph` 存储的是 `GraphValueId` 而不是裸 `TensorSpec`，validator 就必须依赖 schema 判断端口数量、端口顺序、payload 类型以及哪些端口参与 tensor spec 校验。

Milestone 1 必须提供当前已构图算子的最小 schema registry，至少覆盖 `BuildLlamaDense()` 会产生的固定 arity ops：`Embedding`、`RMSNorm`、`Linear`、`RoPE`、`MatMul`、`Softmax`、`Add`、`SiluMul`、`Argmax`。named / optional / variadic ports 可以后移；固定顺序端口不能后移。

```cpp
enum class OperatorPortKind : uint8_t {
    kModelInput,
    kActivation,
    kWeight,
    kConstant,
    kState,
    kResource,
};

struct OperatorInputPort {
    uint32_t index = 0;
    std::string name{};
    OperatorPortKind kind = OperatorPortKind::kActivation;
    bool required = true;
    bool variadic = false;

    // If true, this port contributes its TensorSpec to CheckInputSpecs /
    // InferOutputShapes in schema order.
    bool contributes_tensor_spec = true;
};

struct OperatorOutputPort {
    uint32_t index = 0;
    std::string name{};
    OperatorPortKind kind = OperatorPortKind::kActivation;
    bool variadic = false;
};

struct OperatorSchema {
    OpType op_type = OpType::kUnknown;
    std::vector<OperatorInputPort> inputs{};
    std::vector<OperatorOutputPort> outputs{};
};
```

例如 `RMSNorm` schema 可以表达为：

```text
inputs[0] = {name: "input",  kind: kActivation, contributes_tensor_spec: true}
inputs[1] = {name: "weight", kind: kWeight,     contributes_tensor_spec: true}
outputs[0] = {name: "output", kind: kActivation}
```

即使 `weight` 在 execution 层被解析为 packed handle，它仍然是 schema 输入端口，并参与 shape / dtype 校验。

## 8. OpParams 与 attrs

当前实现已经将 graph、execution plan 和 operator registry 的参数路径统一为 typed `OpParams`。新增 operator params 必须有明确的 typed `Params` 类型，并进入 `OpParams` variant；不得通过 `attrs` 或无类型容器绕过 schema / params 校验。

当前参数表示为：

```cpp
using OpParams = std::variant<
        std::monostate,
        EmbeddingParams,
        RmsNormParams,
        LinearParams,
        RoPEParams,
        MatMulParams,
        SoftmaxParams,
        AddParams,
        SiluMulParams,
        ArgmaxParams>;
```

收益：

- validation 能检查 `OpType` 与 params 类型匹配；
- serialization / debug dump 更可靠；
- 避免无类型参数错误只在运行期暴露；
- pass pipeline 可以安全读取和重写 op 参数。

`attrs` 不应成为无类型垃圾桶。核心语义参数应进入 typed `op_params`；`attrs` 只用于 extension metadata、raw-kernel fallback 或暂未定型的实验性标记。

## 9. ModelGraph 容器

```cpp
class ModelGraph {
public:
    struct Input {
        GraphValueId value{};
        std::string name{};
    };

    struct Output {
        GraphValueId value{};
        std::string name{};
    };

    struct NodeOutputDecl {
        TensorSpec spec{};
        GraphValuePayload payload{std::monostate{}};
        QuantizationSpec quantization{};
        std::string debug_name{};
    };

    struct AddedNode {
        GraphNodeId node{};
        std::vector<GraphValueId> outputs{};
    };

    GraphValueId AddInput(TensorSpec spec, std::string name);

    GraphValueId AddWeight(
            TensorSpec spec,
            WeightBinding binding,
            QuantizationSpec quantization,
            std::string debug_name);

    GraphValueId AddConstant(
            TensorSpec spec,
            ConstantBinding binding,
            std::string debug_name);

    GraphValueId AddState(
            TensorSpec spec,
            StateBinding binding,
            std::string debug_name);

    AddedNode AddNode(
            OpType op_type,
            std::optional<uint32_t> decoder_layer_index,
            std::vector<GraphValueId> inputs,
            std::vector<NodeOutputDecl> outputs,
            OpParams op_params,
            ModelGraphAttrs attrs,
            std::string debug_name);

    void MarkOutput(GraphValueId value, std::string name);

    Status Validate() const;
    StatusOr<std::vector<GraphNodeId>> TopologicalOrder() const;

    std::span<const GraphNode> GetNodes() const noexcept;
    std::span<const GraphValue> GetValues() const noexcept;
    std::span<const Input> GetInputs() const noexcept;
    std::span<const Output> GetOutputs() const noexcept;

    const GraphNode& GetNode(GraphNodeId id) const;
    const GraphValue& GetValue(GraphValueId id) const;

private:
    HfModelConfig config_{};
    ModelGraphMetadata metadata_{};

    std::vector<GraphNode> nodes_{};
    std::vector<GraphValue> values_{};
    std::vector<Input> inputs_{};
    std::vector<Output> outputs_{};
};
```

`AddNode()` 必须是创建 node 与 node-produced values 的唯一原子入口。调用方只提供输入 value、输出 value 声明和 op metadata；`ModelGraph` 内部按以下顺序完成构造：

1. 分配新的 `GraphNodeId`；
2. 为每个 `NodeOutputDecl` 分配新的 `GraphValueId`，并设置其 `producer = node_id`；
3. 构造 `GraphNode`，将新输出 value 写入 `GraphNode.outputs`；
4. 返回 `{node_id, output_value_ids}`。

这样避免 `AddActivation(spec, producer)` 与 `AddNode(GraphNode node)` 之间的构造循环：调用方不需要先拿到 producer node id 才能声明 node outputs，也不需要先构造完整 `GraphNode` 才能获得 output value id。

`AddInput()`、`AddWeight()`、`AddConstant()`、`AddState()` 只用于创建无 producer 的外部入口 value。所有由 node 产生的 activation / state / resource update 都必须通过 `AddNode()` 的 `NodeOutputDecl` 创建。

`ModelGraph` 不持久化 `consumers`。consumer/use-list 是从 `GraphNode.inputs` 派生出的索引或查询结果，例如 `BuildConsumerIndex()` / `GetConsumers(GraphValueId)` 可以按需扫描或构建临时缓存。这样 use-def 的持久语义来源只有两处：`GraphNode.inputs` 表达 use，`GraphValue.producer` 表达 def，避免双向边缓存与 node inputs 漂移。

如果允许 graph rewrite pass 修改节点输入输出，必须提供受控 mutation API 来同步维护 `producer` 与 node inputs，并在 mutation 后按需重新派生 consumers 索引。

## 10. Graph Rewrite API

`ModelGraph` 对外是 immutable semantic graph。Graph pass 不应直接修改 `nodes_`、`values_`、`inputs_`、`outputs_`，也不应让外部观察到部分 rewrite 后的中间非法状态。

但 immutable 不等于每个 pass 都全量拷贝 / 重建一次 graph。Semantic pass pipeline 默认采用 **immutable snapshot + rewrite transaction** 模式：`ModelGraph` 是公开的合法快照，`PassManager` 在 pass group 内打开受控 `GraphRewriteSession`，批量应用 mutations，并只在 phase checkpoint materialize 新的 immutable `ModelGraph`。

```text
ModelGraph snapshot
    ↓ open GraphRewriteSession
Pass group:
    pass 1 → mutations / session API
    pass 2 → mutations / session API
    pass 3 → mutations / session API
    ...
    ↓ commit at phase checkpoint
New ModelGraph snapshot
```

这样既保持 use-def 安全，又避免长 pipeline 中 `Graph_0 -> Graph_1 -> ... -> Graph_N` 的重复全量重建成本。

### 10.1 Immutable Snapshot Contract

`ModelGraph` 一旦构造完成即视为 immutable snapshot：

1. `GraphNodeId` / `GraphValueId` 是 snapshot-local index id；
2. `nodes_`、`values_`、`inputs_`、`outputs_` 不暴露 mutable 引用；
3. 持久 graph 只保存 `GraphValue::producer`，不保存 `consumers`；
4. consumers/use-list 由 `GraphNode.inputs` 按需派生；
5. external caller、lowering、debug dump 只消费完整合法 snapshot。

### 10.2 Rewrite Session 与事务

`GraphRewriteSession` 是 `PassManager` 内部的受控 mutable overlay。Pass 可以返回 mutation list，也可以通过 session 的受控 API 表达局部修改，但不能直接写底层 vector。

```cpp
struct ReplaceNodeMutation {
    GraphNodeId old_node{};
    std::vector<GraphNode> replacement_nodes{};
};

struct RemoveNodeMutation {
    GraphNodeId node{};
};

struct RedirectInputMutation {
    GraphNodeId node{};
    size_t input_index = 0;
    GraphValueId new_value{};
};

struct ReplaceValueMutation {
    GraphValueId old_value{};
    GraphValueId new_value{};
};

struct GraphMetadataPatch {
    std::string key{};
    std::string value{};
};

struct MarkMetadataMutation {
    GraphNodeId node{};
    GraphMetadataPatch patch{};
};

using GraphMutation = std::variant<
        ReplaceNodeMutation,
        RemoveNodeMutation,
        RedirectInputMutation,
        ReplaceValueMutation,
        MarkMetadataMutation>;

class GraphRewriteSession {
public:
    Status Apply(std::span<const GraphMutation> mutations);

    Status ReplaceNode(GraphNodeId node, std::vector<GraphNode> replacement_nodes);
    Status RemoveNode(GraphNodeId node);
    Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);

    GraphValueId GetResolvedValue(GraphValueId value) const;
    StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;

    Status ValidateEdits() const;
    StatusOr<ModelGraph> Commit();
};
```

session 内部可以维护：

- append-only 新 nodes / values；
- deleted nodes / values 的 tombstone；
- value replacement map；
- node input override；
- 临时 consumers/use-list cache；
- metadata patches。

这些结构只在 rewrite transaction 内部存在。`Commit()` 后生成新的 immutable `ModelGraph`，临时 cache 不进入持久 graph。

#### 10.2.1 链式替换解析与 Virtual View

批量 pass 会产生链式 value replacement。例如：

```text
Pass A: ReplaceValue(1, 2)
Pass B: ReplaceValue(2, 3)
```

此时 session 内部可能记录到：

```text
replacement_map = {1 -> 2, 2 -> 3}
```

任何后续 pass 查询 `Value 1` 的当前语义值时，必须得到最终结果 `Value 3`，不能只得到中间值 `Value 2`。因此 `GraphRewriteSession` 必须提供带路径压缩的 replacement resolution，或者使用并查集（Disjoint Set / Union-Find）维护 value alias 集合。

伪代码：

```cpp
GraphValueId GraphRewriteSession::GetResolvedValue(GraphValueId value) const {
    auto it = replacement_map_.find(value);
    if (it == replacement_map_.end()) {
        return value;
    }

    GraphValueId resolved = GetResolvedValue(it->second);
    // 实际实现可在非 const 查询或 mutable cache 中做路径压缩。
    return resolved;
}
```

如果允许在 session 内继续重写已被替换的 value，所有 mutation API 必须先调用 `GetResolvedValue()` 规范化输入，例如：

```text
RedirectInput(node, port, old_value) 先解析 old_value 的最终 replacement；
ReplaceValue(old_value, new_value) 先解析 old_value 与 new_value；
ValidateEdits() / Commit() 只能基于 resolved value view 判断 use-def。
```

Pass 在 session 期间不允许直接读取原始 `ModelGraph::GetNode()` 后自行解释 inputs。它们必须通过 session 的 wrapper API 读取 virtual view，例如 `GetNodeView(GraphNodeId)`。这个 view 需要：

1. 过滤 tombstone node / value；
2. 将 `GraphNode.inputs` 中的每个 `GraphValueId` 映射为 `GetResolvedValue(input)`；
3. 合并 node input override 与 append-only 新节点；
4. 对 use-list / consumers 查询返回基于 resolved inputs 的结果。

这条规则避免 pass 在同一个 transaction 内看到过期 value、deleted value 或未解析的中间 replacement。

### 10.3 Mutation Batching

Pass 可以继续以 `GraphMutation` 作为可测试的输出格式，但 `PassManager` 不应每个 pass 后都调用一次全量 graph materialization。推荐按 phase batching：

```text
canonicalize pass group     → Commit()
shape/type legalization     → Commit()
semantic fusion marking     → Commit()
final pre-lowering cleanup   → Commit()
```

`GraphRewriteSession::Apply()` 负责：

1. 校验 mutation 引用的 node / value 是否存在；
2. 检查 mutation 之间是否冲突；
3. 更新 overlay 中的 node inputs / outputs；
4. 更新或失效临时 consumers/use-list cache；
5. 记录需要在 commit 时重建的 producer / metadata / graph input-output 状态。

`GraphRewriteSession::Commit()` 负责：

1. compact live nodes / values；
2. 重建 `GraphValue::producer`；
3. 重建 graph inputs / outputs；
4. 丢弃 tombstone、replacement map、临时 consumers cache；
5. 运行完整 `Validate()`；
6. 返回新的 immutable `ModelGraph`。

### 10.4 Validation 分层

为了避免每个 pass 都做完整 DAG validation，validation 分为三层：

1. **Edit-time cheap checks**：每次 controlled edit 检查 ID 有效、端口存在、payload 类型与 schema 兼容、不能引用 deleted value；
2. **Transaction checkpoint validation**：`Commit()` 时运行完整 `Validate()`，检查 producer、schema、shape、DAG、graph input/output；
3. **Debug / CI validation**：可选地在每个 pass 后 full validate 或 dump 当前 session graph，用于调试 rewrite pass。

release 默认不要求每个 pass 后 materialize + full validate。debug 模式可以打开更严格策略。

### 10.5 Storage 与性能策略

第一版不需要实现完整 persistent graph 或 chunked copy-on-write。优先采用：

- pass group 内批量 mutation，减少 materialization 次数；
- session 内临时 consumers cache，加速 `ReplaceValue`；
- arena / `std::pmr` 管理 rewrite 临时分配；
- 对 shape、attrs、debug string、constant binding 等大 metadata 做 interning 或共享存储，避免 deep copy。

如果 profiling 证明 `Commit()` 仍是启动期瓶颈，再考虑 chunked vector COW / persistent storage。COW 是后续性能优化，不是 v2 DAG IR 的第一版要求。

### 10.6 Controlled Mutable Escape Hatch

复杂局部 rewrite 可以在 `GraphRewriteSession` 内部使用 mutable scratch graph，但 scratch state 不能逃逸出 `PassManager`。唯一对外发布的结果必须是 `Commit()` 返回的 immutable `ModelGraph`。

这种模式的边界是：

- semantic pass 可以快速连续修改 session；
- session API 负责维护或失效 use-def 派生索引；
- 外部永远只看到合法 snapshot；
- phase checkpoint 控制 materialization 和 full validation 成本。

## 11. Metadata

`ModelGraph` 需要可调试、可序列化、可演进的 metadata：

```cpp
struct ModelGraphMetadata {
    uint32_t ir_version = 1;
    std::string model_family{};
    std::string architecture{};
    std::string source_model_id{};
    std::string debug_name{};
};
```

可选扩展：

- source checkpoint file / tensor key；
- original HuggingFace config 摘要；
- graph build options；
- pass pipeline history；
- debug dump / round-trip validation 信息。

## 12. Llama 单层数据流示例

```text
v_hidden_in
    ↓
RMSNorm(v_hidden_in, w_input_norm) → v_normed

Linear(v_normed, w_q) → v_q
Linear(v_normed, w_k) → v_k
Linear(v_normed, w_v) → v_v

RoPE(v_q, v_k, position_ids) → v_q_rope, v_k_rope

MatMul(v_q_rope, v_k_rope) → v_scores
Softmax(v_scores) → v_probs
MatMul(v_probs, v_v) → v_attn
Linear(v_attn, w_o) → v_attn_out

Add(v_hidden_in, v_attn_out) → v_post_attn

RMSNorm(v_post_attn, w_post_norm) → v_mlp_normed
Linear(v_mlp_normed, w_gate) → v_gate
Linear(v_mlp_normed, w_up) → v_up
SiluMul(v_gate, v_up) → v_mlp_act
Linear(v_mlp_act, w_down) → v_mlp_out

Add(v_post_attn, v_mlp_out) → v_hidden_out
```

如果显式建模 KV cache，attention 相关数据流可以扩展为：

```text
RoPE(v_q, v_k, position_ids) → v_q_rope, v_k_rope
KVCacheUpdate(v_k_rope, v_v, state_k_cache_in, state_v_cache_in) → state_k_cache_out, state_v_cache_out
Attention(v_q_rope, state_k_cache_out, state_v_cache_out, mask_or_metadata) → v_attn
```

这使 prefill / decode / paged attention 的状态语义在 graph 中可见，但具体 cache layout、page table、device buffer 仍留给 lowering / runtime。

KV cache state 建模规则：

1. `StateValue` 表达逻辑状态身份和版本，例如 `kv_cache.layer_3.key.before_update` 与 `kv_cache.layer_3.key.after_update`；K/V 分别用强类型 `KVCacheSlot::kKey` / `KVCacheSlot::kValue` 表达，不使用字符串 slot；
2. 状态更新必须产生新的 `StateValue`，不能原地修改或复用同一个 value 表示更新前后两个版本；
3. state edge 只表达语义依赖顺序，不描述 cache layout、page table、device buffer、paged cache handle；
4. `ResourceValue` 不用于早期 KV cache 建模；paged cache handle、外部分配 buffer、backend resource 属于 lowering / runtime，等有实际 runtime resource use case 后再进入 ModelGraph；
5. validation 应检查 state update node 的 state input / state output 端口符合 operator schema，不能通过普通 activation edge 隐式表达跨 step 状态。

## 13. Validation 规则

`Validate()` 至少检查：

1. 所有 `GraphNodeId` / `GraphValueId` 有效；
2. 每个 node input / output 均引用有效 value；
3. `GraphValuePayload` 不允许为 `std::monostate`；
4. 每个 `ActivationValue` 必须有 producer；
5. `ModelInputValue` / `WeightValue` / `ConstantValue` 默认不允许有 producer；
6. `StateValue` 允许无 producer 的外部入口，也允许由节点产生更新后的新 state 版本；`ResourceValue` 是延后落地的 runtime resource 逻辑占位，不应作为早期 KV cache 默认表达；
7. 每个 node output 对应 value 的 producer 必须是该 node；
8. 按需派生的 consumers 索引必须与所有 `GraphNode.inputs` 完全一致；
9. node 输入数量、输出数量、schema port 顺序、payload 类型必须符合 operator schema；
10. 输入输出 `TensorSpec` dtype / shape / rank 兼容；
11. dynamic shape 符号和约束一致；
12. 对普通 activation 数据流，DAG 必须无环；
13. 对 state token，如需表达跨 step 状态更新，应通过显式 state input/output 建模，不能制造隐式环；state update node 的 state input / state output 必须符合 operator schema；KV cache state binding 必须使用 `KVCacheStateBinding`，K/V 槽位必须分别匹配 `KVCacheSlot::kKey` / `KVCacheSlot::kValue`，且 `decoder_layer_index` 必须与对应 decoder layer node 一致。

注：payload 与 binding 的一致性由 `GraphValuePayload` variant 类型在编译期保证，无需额外的运行时一致性校验。

## 14. Shape 与动态维度

动态 shape 是语义信息，应该进入 `ModelGraph`；具体 runtime shape binding 不应该进入 `ModelGraph`。

建议模型：

```text
ModelGraph:       token_ids [batch, seq_len]
                  hidden    [batch, seq_len, hidden_size]
                  kv_cache  [batch, kv_len, num_kv_heads, head_dim]

Runtime ShapeEnv: batch = 1
                  seq_len = prompt length 或 1
                  kv_len = previous length + current length
```

`ModelGraph` 负责表达 `batch`、`seq_len`、`kv_len` 等符号及约束；runtime / execution context 负责给这些符号绑定具体值。

## 15. Quantization 与 packed weight

必须区分两类信息：

### 15.1 ModelGraph 中的量化语义

- int4 / int8 / fp16 / bf16；
- per-tensor / per-channel / per-group；
- group size；
- scale dtype；
- zero point policy；
- weight logical shape；
- checkpoint tensor key。

### 15.2 Backend artifact 中的物理表示

- AVX2 / AMX / CUDA-specific packed format；
- transposed / interleaved layout；
- precomputed kernel-specific scales；
- NUMA / device placement；
- backend-owned packed buffers。

`ModelGraph` 只保存 15.1。15.2 必须在 lowering / backend cache 中产生和持有。

## 16. Semantic Pass Pipeline

`ModelGraph` 应允许语义 pass，但不要过早变成完整 MLIR 式系统。推荐最小 pass 能力：

- graph validation；
- shape inference / constraint propagation；
- canonicalize；
- constant folding；
- dead value detection；
- pattern marking，例如 attention pattern、MLP pattern；
- semantic fusion rewrite，例如把一组节点标记为可 fusion 的 high-level op。

Semantic pass 默认采用第 10 节定义的 immutable snapshot + rewrite transaction 模式：pass 读取合法 graph snapshot 或 `GraphRewriteSession` virtual view，输出 `GraphMutation` 列表或调用 session 受控 API；`PassManager` 在 phase checkpoint 统一 commit 新的 `ModelGraph`。

注意：semantic fusion rewrite 可以发生在 `ModelGraph`，但具体 fused kernel、workspace、launch config 仍属于 lowering / backend plan。

### 16.1 TVM / Relax 参考原则与非目标

`ModelGraph` rewrite 设计可以参考 Apache TVM / Relax 的工程思想，但不能照搬 TVM 的对象模型或编译栈。AetherMind 的目标是借鉴成熟的 pass 组织与 pattern matching 思路，同时保持 ID-based、扁平、轻量的推理引擎 IR。

建议参考的方向：

1. **Visitor / Mutator 思想**：pass 可以提供类似 `GraphMutator` 的 helper，按拓扑序遍历 graph，识别可重写节点或子图，并产生 `GraphMutation`。但它不直接修改 `ModelGraph`，也不拥有 IR 存储。
2. **Dataflow Pattern Matching**：可以实现 TVM DFPattern-inspired 的极简结构匹配器，用于匹配 `OpType`、输入输出连接、payload 类型和基础 attrs。复杂 shape / params predicate 应等 typed `OpParams` 落地后再扩展。
3. **Semantic Pass 与 Lowering Pass 分层**：target-independent 的数学等价变换、canonicalize、dead value detection、fusion marking 属于 semantic pass；kernel dispatch、packed weight、workspace、buffer aliasing、device/resource binding 属于 lowering / execution planning。
4. **显式 StateValue**：KV cache、decode state 等跨 step 状态通过显式 state input / output 表达，避免隐藏副作用。

必须避免的方向：

1. 不引入 TVM `ObjectRef` / AST pointer ownership 模型；`ModelGraph` 保持 `GraphNodeId` / `GraphValueId` + `std::vector` 存储。
2. 不引入 TVM TIR / TensorIR 级别 kernel generation；AetherMind lowering 只做 kernel dispatch / fused kernel selection，不做运行时代码生成。
3. 不引入完整 Relax/Relay type inference 系统；仅实现 LLM 推理所需的符号 shape、dtype、schema validation 与约束传播。
4. 不实现完整 TVM DFPattern 语言；第一版只做保守的结构匹配。

可选 helper 形态：

```cpp
class GraphMutator {
public:
    virtual ~GraphMutator() = default;

    virtual std::optional<GraphMutation> MutateNode(
            const GraphRewriteSession& session,
            GraphNodeId node_id) {
        return std::nullopt;
    }

    Status Run(GraphRewriteSession& session) {
        std::vector<GraphMutation> mutations;
        for (GraphNodeId node_id: session.TopologicalOrderView()) {
            if (auto mutation = MutateNode(session, node_id)) {
                mutations.push_back(std::move(*mutation));
            }
        }
        return session.Apply(mutations);
    }
};
```

关键约束：pass 在 transaction 内读取节点时必须通过 `GraphRewriteSession` 的 virtual view，例如 `GetNodeView()`，以获得已解析 replacement、过滤 tombstone 后的最新视图；不能直接读取原始 `ModelGraph::GetNode()` 后自行解释 inputs。

## 17. Lowering

DAG lowering 的职责：

1. 拓扑排序或生成合法执行 schedule；
2. 按 operator schema 顺序解释 `GraphNode.inputs` / `GraphNode.outputs`；
3. 解析 `WeightValue` 的逻辑权重引用；
4. 根据 backend / execution mode / device 选择 operator 或 kernel；
5. 生成 packed weight mapping；
6. 推导 buffer aliasing / in-place update 约束；
7. 计算 workspace；
8. 插入必要的 runtime resource / communication / synchronization；
9. 生成 `ExecutionPlanNodeSpec`、`LoweredGraph` 或 `ExecutionPlan`。

Lowering 可以派生 execution-specific views，例如 runtime activation inputs、bound weight handles、state handles、resource handles。但这些派生视图必须保留原始 schema `port_index` 或 `port_name`，不能替代 `GraphNode.inputs` 本身。

`ExecutionPlanNodeSpec::input_specs` 用于 operator schema validation / shape inference，应按 schema port 顺序构造。对于声明为 tensor-like 的端口，weight / constant / state 也必须保留对应 `TensorSpec`，不能因为 payload 类型是 `WeightValue` 就过滤掉。

### 17.1 State Buffer Aliasing

`ModelGraph` 中的 state update 必须保持 SSA-style 语义：`state_kv_in` 与 `state_kv_out` 是两个不同的 `GraphValueId`，分别表示更新前和更新后的逻辑状态版本。这保证 graph 仍然是清晰的 use-def DAG，避免在语义层引入隐式原地修改。

但在物理执行层，某些 state update 必须 lower 为 in-place buffer alias。例如 KV cache update：

```text
ModelGraph:     state_kv_in -> KVCacheUpdate -> state_kv_out
ExecutionPlan:  state_kv_in.buffer == state_kv_out.buffer
```

Lowering planner 必须识别 operator schema 中的 state input / state output 端口关系，并生成显式 alias 约束：

```cpp
struct BufferAliasConstraint {
    GraphValueId source{};  // e.g. state_kv_in
    GraphValueId target{};  // e.g. state_kv_out
    bool must_alias = true;
};
```

KV cache 这类状态更新应使用 **must-alias** 约束：planner 必须把更新前后的 state value 绑定到同一物理地址 / runtime state handle，严禁为了满足普通 tensor SSA 规则而分配新 buffer 或插入拷贝。若 backend 无法满足该 alias 约束，lowering 应失败，而不是静默退化为 copy-based update。

Buffer aliasing 属于 lowering / execution plan，不属于 `ModelGraph`：

- `ModelGraph` 表达不同的逻辑 state 版本；
- lowering 推导这些版本之间的 physical alias 关系；
- memory planner 根据 alias 约束分配同一 buffer / state handle；
- executor 根据 plan 执行 in-place update。

对于普通 activation，aliasing 是优化机会；对于 KV cache state update，aliasing 是正确性和性能约束。

示例伪代码：

```cpp
for (GraphNodeId node_id: graph.TopologicalOrder()) {
    const GraphNode& node = graph.GetNode(node_id);

    ExecutionPlanNodeSpec spec{};
    spec.op_type = node.op_type;

    const OperatorSchema& schema = registry.GetSchema(node.op_type);
    for (size_t port_index = 0; port_index < node.inputs.size(); ++port_index) {
        GraphValueId input = node.inputs[port_index];
        const GraphValue& value = graph.GetValue(input);
        const OperatorInputPort& port = schema.inputs[port_index];

        if (port.contributes_tensor_spec) {
            spec.input_specs.push_back(value.spec);
        }

        std::visit(overloaded{
            [&](const ActivationValue&) {},
            [&](const ModelInputValue&) {},
            [&](const ConstantValue&) {},
            [&](const WeightValue& w) {
                // Resolve logical weight binding to plain or packed weight handle.
                // Store port_index so execution binding preserves schema ABI.
            },
            [&](const StateValue& s) {
                // Bind runtime state according to execution context.
                // For state update input/output pairs, emit must-alias constraints
                // so logical state versions map to the same physical state buffer.
                // Store port_index so state handle remains schema-addressable.
            },
            [&](const ResourceValue& r) {
                // Bind runtime resource according to execution context.
                // Store port_index so resource handle remains schema-addressable.
            },
        }, value.payload);
    }

    spec.attrs = node.attrs.bytes;
    spec.op_params = node.op_params;
}
```

## 18. 分布式与多后端

原始 `ModelGraph` 不应直接表达具体分布式部署策略。同一个模型图应该可以被不同 strategy lowering 为不同执行计划：

- single CPU；
- multi-thread CPU；
- CPU + accelerator；
- tensor parallel；
- pipeline parallel；
- distributed serving。

send / recv / collective / shard / gather 等 communication ops 应由 planning 层根据 strategy 插入到 `LoweredGraph` 或 `ExecutionPlan`，而不是污染原始模型语义图。

如果需要在 graph 层保留分布式友好的信息，可以只保留语义 metadata，例如 tensor partitionable axis、weight sharding hint、attention state locality hint；具体 placement 仍由 planner 决定。

## 19. 落地 Milestones

### Milestone 1：最小语义 DAG 骨架

Milestone 1 只完成当前 ordered graph skeleton 到最小 use-def DAG 的结构迁移，目标是让现有 `BuildLlamaDense()` 的节点顺序、shape 断言和 weight binding 语义可以用新 IR 表达，不引入完整 compiler framework。

迁移策略采用一次性替换：直接将 `model_graph.h` 中的 v1 `GraphNode` / `ModelGraph` 结构替换为 v2 最小 DAG 设计，同时重写 `BuildLlamaDense()` 以及所有直接遍历 `ModelGraph` 的编译期调用点。不保留 v1 / v2 双轨兼容层，因为现有 graph 代码量小，双轨会增加测试、文档和语义维护成本。

如果 `ExecutionPlanBuilder` 或其他执行计划代码依赖旧的 `GraphNode.inputs` / `GraphNode.outputs` / `GraphNode.weights`，Milestone 1 只做最小适配以保持构建和现有测试通过；完整 Graph → `ExecutionPlan` lowering 仍属于 Milestone 3。

验收范围：

1. 新增 `GraphNodeId`、`GraphValueId`；
2. 新增 `GraphValuePayload` 与 `GraphValue`，首批只要求落地 `ModelInputValue`、`ActivationValue`、`WeightValue`；
3. 将 `GraphNode.inputs` / `GraphNode.outputs` 改为 `GraphValueId`；
4. 将 weight 从 `GraphNode.weights` 迁移为 `WeightValue` payload，并保留现有 `WeightBinding` 语义；
5. 从 `GraphNode` 移除 `WorkspaceRequirement`；
6. 为 `ModelGraph` 增加 `values_` / `inputs_` / `outputs_`，并提供原子 `AddNode()` 构造 node-produced values；
7. 为当前 graph builder 会产生的固定 arity ops 添加最小 `OperatorSchema` / schema registry：`Embedding`、`RMSNorm`、`Linear`、`RoPE`、`MatMul`、`Softmax`、`Add`、`SiluMul`、`Argmax`；
8. 添加最小 `Validate()`：ID 范围、producer 一致性、node output producer、weight payload binding、operator schema 端口数量 / 顺序 / payload 类型、普通 activation DAG 无环；
9. `op_params` 使用 typed `OpParams`，并禁止放入指针、backend handle、kernel config、lambda、opaque context 或 runtime resource；
10. 提供按需派生 consumers 索引的 helper / 测试，但不在 `GraphValue` 中持久化 consumers；
11. 更新现有 graph builder 单元测试，使其通过 `GraphValue` 检查 shape、producer 和 weight binding。

明确不属于 Milestone 1：

- `StateValue` / `ResourceValue` 的实际落地；
- KV cache state 建模；
- `GraphMutation` / `ModelGraphRewriter`；
- immutable rewrite pass pipeline；
- `OpParams` serialization / debug dump / round-trip tooling；
- graph dump / serialization tooling；
- backend lowering、kernel selection、workspace planning；
- named / optional / variadic port 的完整 schema 能力。

### Milestone 2：Llama GraphBuilder

1. 重写 `BuildLlamaDense()`，显式创建 activation / weight / constant value；
2. 显式表达 residual、RoPE 多输出；
3. 使用 typed `OpParams` variant 或等价的 schema-validated typed parameter object 表达 operator params；
4. 如需在 Milestone 2 表达 KV cache，只落地版本化 `StateValue` / `StateBinding`，通过 `state_in -> state_out` 建模，不引入 `ResourceValue`；
5. 添加 topology、shape、payload / binding / typed params 单元测试。

### Milestone 3：Lowering Bridge

1. 添加 Graph → `ExecutionPlanNodeSpec` 转换；
2. 保持 `ExecutionPlanBuilder` 接口尽量稳定；
3. 在 lowering 中解析 weight / constant / state；
4. workspace 继续由 operator / planner 计算。

### Milestone 4：参数与工具完善

1. 增强 `OpParams` serialization / debug dump / round-trip tooling；
2. 增加 `QuantizationSpec`；
3. 在存在明确 runtime handle use case 后增加 `ResourceValue` / `ResourceBinding`；
4. 增加 `GraphMutation` 与 `ModelGraphRewriter`；
5. 增加 semantic pass pipeline；
6. 增加 graph dump / serialization / validation tooling。

### 19.1 工程落地步骤

落地时按“先一次性替换 v1 graph，再逐步补齐高级能力”的路线推进。每一步都必须保持仓库可编译、相关测试可运行，不保留 v1 / v2 双轨兼容层。

#### M0：落地前盘点

先确认直接依赖旧 `ModelGraph` 的代码点：

- `include/aethermind/model/graph/model_graph.h`；
- `src/model/graph/model_graph.cpp`；
- `src/model/graph/model_graph_builder.cpp`；
- `tests/unit/model/graph/test_model_graph_builder.cpp`；
- 任何遍历 `GraphNode.inputs` / `GraphNode.outputs` / `GraphNode.weights` / `GraphNode.workspace_requirement` 的 execution plan 代码。

产出旧字段到新字段的迁移表：

```text
GraphNode.inputs:  vector<TensorSpec>        -> vector<GraphValueId>
GraphNode.outputs: vector<TensorSpec>        -> vector<GraphValueId>
GraphNode.weights: vector<WeightBinding> -> WeightValue payload
TensorSpec on edge                         -> GraphValue.spec
WorkspaceRequirement on GraphNode          -> lowering / execution planning
```

#### M1：最小 DAG 骨架一次性替换

1. 在 `model_graph.h` 中引入 `GraphNodeId`、`GraphValueId`、`GraphValuePayload`、`GraphValue`、`OperatorSchema`。
2. 将 `GraphNode.inputs` / `GraphNode.outputs` 替换为 `std::vector<GraphValueId>`。
3. 将 `GraphNode.weights` 迁移为 `WeightValue` payload。
4. 从 `GraphNode` 移除 `WorkspaceRequirement`。
5. 实现 `ModelGraph` 容器 API：`AddInput()`、`AddWeight()`、`AddNode()`、`MarkOutput()`、`GetNode()`、`GetValue()`、`TopologicalOrder()`、`Validate()`。
6. 添加最小 schema registry，覆盖 `Embedding`、`RMSNorm`、`Linear`、`RoPE`、`MatMul`、`Softmax`、`Add`、`SiluMul`、`Argmax`。
7. 重写 `BuildLlamaDense()`：用 `GraphValueId hidden` 串联真实数据流；weight 通过 `AddWeight()` 创建；每个 op 通过 `AddNode()` 创建 output value；保留确定性拓扑顺序。
8. 更新 graph builder 测试：从检查 `node.outputs[i].shape` 改为检查 `graph.GetValue(node.outputs[i]).spec`；从检查 `node.weights` 改为检查 weight input 对应的 `WeightValue.binding`。
9. 对依赖旧 graph 字段的 execution plan 代码做最小适配，确保构建和现有测试通过；完整 lowering 仍留到 M3。

M1 验收：graph builder 测试通过；`Validate()` 能发现非法 ID、错误 producer、schema mismatch、`std::monostate` payload、weight binding 缺失和普通 activation 环。

#### M2：GraphBuilder 语义补强

1. 保持 operator params 走 typed `OpParams` variant 或等价的 schema-validated typed parameter object。
2. 抽出 builder helper，减少 `BuildLlamaDense()` 中的重复构图逻辑，例如 `AddLinear()`、`AddRmsNorm()`、`AddBinary()`、`AddRoPE()`。
3. 显式验证 attention residual 与 MLP residual 的数据流。
4. 如需表达 KV cache，只引入版本化 `StateValue` / `StateBinding`，通过 `state_in -> state_out` 建模，不引入 `ResourceValue`。
5. 补充 typed params、payload / binding、拓扑顺序、residual 数据流单元测试。

#### M3：Lowering Bridge

1. 添加 Graph -> `ExecutionPlanNodeSpec` 转换。
2. lowering 按 operator schema 顺序读取 `GraphNode.inputs` / `GraphNode.outputs`。
3. 在 lowering 中解析 `WeightValue` 为 plain / packed weight handle。
4. workspace 继续由 operator / planner 计算，不回流到 `ModelGraph`。
5. 如果存在 `StateValue`，lowering 必须生成 buffer alias constraint，例如：

```text
state_kv_in.buffer == state_kv_out.buffer
```

M3 验收：execution plan 构建不再依赖旧 `GraphNode.weights` / `GraphNode.workspace_requirement`。

#### M4：Pass 与工具

最后再引入 compiler-like 能力：

- `GraphRewriteSession`；
- `GraphMutation`；
- `GraphMutator`；
- session 内 consumers / use-list 临时索引；
- graph dump；
- serialization；
- round-trip validation；
- pattern matcher；
- semantic pass pipeline。

这些能力不应提前阻塞 M1 / M2 的核心结构迁移。

推荐执行顺序：

```text
ID / GraphValue 类型
-> ModelGraph 容器
-> OperatorSchema
-> Validate
-> BuildLlamaDense rewrite
-> tests migration
-> minimal ExecutionPlan adaptation
-> typed OpParams
-> lowering bridge
-> rewrite / pass tooling
```

## 20. 设计红线

1. 不把 backend / device / ISA 放进 `ModelGraph`；
2. 不把 workspace / memory offset 放进 `GraphNode`；
3. 不把 packed weight 当成 `GraphValue` payload；
4. 不把 `attrs` 当成无类型参数垃圾桶；
5. 不把裸指针、backend handle、kernel config、lambda、opaque context 或 runtime resource 塞进 `OpParams`；
6. 不依赖 `decoder_layer_index` 表达数据依赖；
7. 不在 graph builder 中根据已有 kernel 覆盖率裁剪拓扑；
8. 不因起步范围较小而过早实现完整 compiler framework。

## 21. 总结

`ModelGraph` 应是稳定的 semantic use-def DAG：

- `GraphNode` 是语义算子；
- `GraphValue` 是数据流边；
- input / activation / weight / constant / state / resource 都统一为 value；
- shape、quantization、weight binding、state binding 属于语义图；
- workspace、device、ISA、packed weight、kernel selector、distributed placement 属于 lowering / planning / backend；
- `ExecutionPlan` 是某次 backend / execution mode / runtime context 下从 `ModelGraph` 派生出的可执行计划。

这样可以支撑 CPU Llama 顺序执行、fusion、多后端、KV cache、continuous batching、服务化和分布式推理。
