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

- 语义算子：`Embedding`、`RMSNorm`、`Linear`、`RoPE`、`MatMul`、`Softmax`、`Add`、`SiluMul`、`Argmax`、`Reshape`、`Permute`、`Reorder` 等；
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
    // OpParams variant into ExecutionPlanNodeSpec and Backend::PrepareKernel.
    OpParams op_params{};

    // Extension metadata only. Execution planning derives kernel metadata from
    // op_params and does not consume graph attrs as raw kernel bytes.
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
- workspace 是 execution planning 决策，`LoweredStepSpec` 不携带 workspace 需求，lowered 步骤按默认零需求参与规划；device、ISA、weight format、kernel selector 是 lowering 决策；
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
        ArgmaxParams,
        ReshapeParams>;
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
struct NodeRemoval {
    GraphNodeId node{};
};

struct InputRedirection {
    GraphNodeId node{};
    size_t input_index = 0;
    GraphValueId new_value{};
};

struct ValueReplacement {
    GraphValueId old_value{};
    GraphValueId new_value{};
};

struct RewriteOutputBinding {
    NodeOutputDesc desc{};
    std::optional<GraphValueId> replaces{};
};

struct ReplacementNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<RewriteOutputBinding> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};
    std::string debug_name{};
};

struct SubgraphReplacement {
    std::vector<GraphNodeId> old_nodes{};
    std::vector<ReplacementNode> replacement_nodes{};
};

using GraphMutation = std::variant<SubgraphReplacement,
                                   NodeRemoval,
                                   InputRedirection,
                                   ValueReplacement>;

class GraphRewriteSession {
public:
    explicit GraphRewriteSession(const ModelGraph& graph);

    AM_NODISCARD GraphValueId AllocateVirtualValue();

    AM_NODISCARD Status Apply(std::span<const GraphMutation> mutations);

    AM_NODISCARD Status RemoveNode(GraphNodeId node);
    AM_NODISCARD Status ReplaceSubgraph(std::span<const GraphNodeId> old_nodes, const std::vector<ReplacementNode>& replacement_nodes);
    AM_NODISCARD Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    AM_NODISCARD Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);

    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;
    AM_NODISCARD StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;

    AM_NODISCARD bool IsNodeLive(GraphNodeId node) const noexcept;
    AM_NODISCARD StatusOr<std::vector<GraphNodeId>> GetTopologicalOrder() const;
    AM_NODISCARD std::vector<GraphNodeId> FindNodesByOpType(OpType op_type) const;

    AM_NODISCARD bool IsValueLive(GraphValueId value) const noexcept;
    AM_NODISCARD std::vector<GraphValueId> GetLiveValues() const;
    AM_NODISCARD std::vector<GraphNodeId> FindConsumers(GraphValueId value) const;
    AM_NODISCARD bool HasLiveConsumers(GraphValueId value) const;

    AM_NODISCARD Status ValidateEdits() const;
    AM_NODISCARD StatusOr<ModelGraph> Commit() const;

private:
    struct RewriteEntry {
        std::vector<GraphNodeId> old_nodes{};
        std::vector<ReplacementNode> replacements{};
        bool active = true;
        bool exposes_node_view = false;
    };

    const ModelGraph& graph_;
    std::vector<std::optional<SessionConstant>> session_value_metadata_{};
    std::vector<RewriteEntry> rewrites_{};
    std::vector<std::optional<std::size_t>> node_to_rewrite_{};
    std::vector<std::optional<GraphValueId>> value_replacements_{};
    mutable std::vector<std::optional<GraphValueId>> resolved_value_cache_{};
};
```

session 内部维护：

- `rewrites_`：`RewriteEntry` 列表，记录旧节点集合、`ReplacementNode` 列表、active 状态，以及是否暴露 node view；
  - `active`：rewrite 是否仍然有效。`ReplaceSubgraph` 覆盖同一旧节点时会 `DeactivateRewrite` 旧 entry，使其在 `Commit` 时被跳过；
  - `exposes_node_view`：该 rewrite 是否通过 `RedirectInput` 创建的 mirror `ReplacementNode` 暴露 node view。`RedirectInput` 需要在 `GetNodeView` 中返回修改了输入的节点视图，因此 mirror replacement 标记此字段为 `true`；普通 `ReplaceSubgraph` 产生的 replacement 不暴露原节点 view，此字段为 `false`。`GetNodeView` 仅对 `active && exposes_node_view && old_nodes.size()==1 && replacements.size()==1` 的 rewrite 返回 mirror 视图，否则返回 `NotFound`。
- `node_to_rewrite_`：从旧 node 到当前 active rewrite 的索引，用于快速覆盖和失效旧 rewrite；
- `session_value_metadata_`：session-local value 元数据及唯一计数来源；有值项表示 session constant，`nullopt` 表示 virtual value；`AllocateVirtualValue()` 分配的 virtual value 仅用于同一个 `RewriteEntry` 内 replacement 子图的内部边；
- `value_replacements_`：value replacement map，支持链式替换；
- `resolved_value_cache_`：`mutable` 路径压缩缓存，每次 `ReplaceValue()` 时失效。

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

当前实现使用带路径压缩的 `mutable` 缓存：

```cpp
GraphValueId GraphRewriteSession::GetResolvedValue(GraphValueId value) const {
    if (value.index >= value_replacements_.size()) {
        return value;
    }

    if (resolved_value_cache_[value.index].has_value()) {
        return *resolved_value_cache_[value.index];
    }

    std::vector<uint32_t> path;
    GraphValueId cur = value;
    GraphValueId resolved = value;
    for (size_t depth = 0; depth < value_replacements_.size(); ++depth) {
        if (cur.index >= value_replacements_.size()) {
            resolved = cur;
            break;
        }

        if (resolved_value_cache_[cur.index].has_value()) {
            resolved = *resolved_value_cache_[cur.index];
            break;
        }

        path.push_back(cur.index);
        const auto& next = value_replacements_[cur.index];
        if (!next.has_value()) {
            resolved = cur;
            break;
        }
        cur = *next;
        resolved = cur;
    }

    for (uint32_t value_index: path) {
        resolved_value_cache_[value_index] = resolved;
    }
    return resolved;
}
```

`resolved_value_cache_` 是 `mutable` 成员，允许在 `const` 查询时缓存路径压缩结果。每次 `ReplaceValue()` 调用会清空整个缓存，保证后续查询的正确性。

如果允许在 session 内继续重写已被替换的 value，各 mutation API 的输入处理规则如下：

```text
RedirectInput(node, port, new_value) 存储原始 new_value，由 GetResolvedValue 在查询/Commit 时统一解析；
ReplaceValue(old_value, new_value) 不解析 old_value（作为 value_replacements_ 的键直接写入，覆盖语义），仅遍历 new_value 解析链做环检测；new_value 的链式解析由 GetResolvedValue 在查询时处理；
ValidateEdits() / Commit() 只能基于 resolved value view 判断 use-def。
```

`ReplaceValue` 采用消费者重定向语义（非别名合并）：直接写 `value_replacements_[old_value] = new_value`，覆盖该键的旧值。**不能**先解析 old_value 再写入其 terminal——那会把 terminal 的消费者错误重定向。例如 `ReplaceValue(A, B)` 后 `ReplaceValue(A, C)`，正确结果是 A→C 且 B 的消费者不变；若先解析 old 得到 B 再写 `value_replacements_[B] = C`，则 B 的消费者被错误改为 C。调用方若需链式重定向（A 和 B 的消费者都读 C），应调用 `ReplaceValue(B, C)`，由 `GetResolvedValue` 的链式遍历处理。

Pass 在 session 期间不允许直接读取原始 `ModelGraph::GetNode()` 后自行解释 inputs。它们必须通过 session 的 wrapper API 读取 virtual view，例如 `GetNodeView(GraphNodeId)`。这个 view 需要：

1. 过滤 tombstone node / value；
2. 将 `GraphNode.inputs` 中的每个 `GraphValueId` 映射为 `GetResolvedValue(input)`；
3. 合并 node input override 与 append-only 新节点；
4. 对 use-list / consumers 查询返回基于 resolved inputs 的结果。

这条规则避免 pass 在同一个 transaction 内看到过期 value、deleted value 或未解析的中间 replacement。

#### 10.2.2 节点与值枚举 API

Session 暴露三类枚举 API，让 pass 可以独立发现节点和值，而非依赖外部传入 id。

**节点活跃度**：`IsNodeLive(node)` 是节点活跃度的单一真理源。`GetNodeView` / `GetTopologicalOrder` / `FindNodesByOpType` / `FindConsumers` 都基于它过滤。节点 live 的条件是：未被 rewrite 触及，或仅由 `RedirectInput` 创建的 mirror replacement 暴露（`active && exposes_node_view && old_nodes.size()==1 && replacements.size()==1`）。`RemoveNode` / `ReplaceSubgraph` 产生的节点不 live。

**值活跃度**：`IsValueLive(value)` 表达"结构性存在"语义，与 `IsNodeLive` 类比。它**不**等于 DCE-reachable：一个没有 live consumer 且非 graph output 的值仍可能 `IsValueLive == true`。DCE 判定需要 `HasLiveConsumers` + graph output 根集组合判断。值 live 的三个条件（任一满足）：

1. 外部值（input / weight / constant / state），无 producer；
2. producer 是 live 节点（未被触及或仅 `RedirectInput`）；
3. active rewrite 的 replacement output 通过 `replaces` binding 接管该 value（即使原 producer 已被 remove/replace）。

`ReplaceValue` 不影响值活跃度：被替换的值仍然存在，只是其消费者被重定向到目标值。

**值枚举**：`GetLiveValues()` 返回所有 live value id（升序），排除被移除节点产生且无 replacement 接管的值，以及 virtual value（virtual value 是 rewrite 内部边，不通过 session API 可观察）。

**消费者查询**：`FindConsumers(value)` 返回 live **原图节点**中消费该值（解析后）的节点列表（拓扑序，每个节点最多出现一次）。查询值与每个节点输入都经 `GetResolvedValue` 解析后比较，因此 `ReplaceValue` 透明反映：`FindConsumers(v_old)` 与 `FindConsumers(v_new)` 在 `ReplaceValue(v_old, v_new)` 后返回相同结果。virtual value 与越界 id 返回空列表。

**注意**：`FindConsumers` **不包含** active replacement node 消费者——replacement 节点在 session 中没有 GraphNodeId（直到 `Commit()` 才分配），无法用 `std::vector<GraphNodeId>` 表示。当某个值的唯一消费者是 replacement 节点时，`FindConsumers` 返回空。

**DCE 消费者判定**：`HasLiveConsumers(value)` 是 DCE 场景的正确判定 API，它同时扫描 live 原图节点和 active replacement node 的 inputs（都经 `GetResolvedValue` 解析后比较）。对 structurally live value 而言，当 `HasLiveConsumers(v) == false` 且 `v` 非 graph output 时，`v` 可被 DCE 视为 dead。`FindConsumers` 用于 rewiring 场景（调用方能通过 `RedirectInput` 实际操作的节点），`HasLiveConsumers` 用于 liveness 判定——两者关注点分离。

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
4. 记录需要在 commit 时重建的 producer / metadata / graph input-output 状态。

`GraphRewriteSession::Commit()` 负责：

1. compact live nodes / values；
2. 重建 `GraphValue::producer`；
3. 重建 graph inputs / outputs；
4. 丢弃 tombstone、replacement map；
5. 运行完整 `Validate()`；
6. 返回新的 immutable `ModelGraph`。

### 10.4 Validation 分层

为了避免每个 pass 都做完整 DAG validation，validation 分为三层：

1. **Edit-time cheap checks**：每次 controlled edit 检查 ID 有效、端口存在、payload 类型与 schema 兼容、不能引用 deleted value；
2. **Transaction checkpoint validation**：`Commit()` 时运行完整 `Validate()`，检查 producer、schema、shape、DAG、graph input/output；
3. **Debug / CI validation**：可选地在每个 pass 后 full validate 或 dump 当前 session graph，用于调试 rewrite pass。

release 默认不要求每个 pass 后 materialize + full validate。debug 模式可以打开更严格策略。

**当前实现状态**：第 1 层由 `GraphRewriteSession` 的 controlled edit API（ID 校验、端口范围、value 存在性）实现；第 2 层由 `Commit()` 内调用 `committed.Validate()` 实现；第 3 层当前通过 `SetCheckpointEvery(1)` 达到类似效果，但代价是每 pass 都会 materialize 新图（比设计描述的"validate session without materialize"更重）。未来可增加 `ValidateSession()` 做无 materialization 的 full validate，从而真正分离"校验"与"物化"。`ValidateEdits()` 只做轻量校验（value id 有效、replacement 引用有效），不等价于 full `Validate()`。

### 10.5 Storage 与性能策略

第一版不需要实现完整 persistent graph 或 chunked copy-on-write。优先采用：

- pass group 内批量 mutation，减少 materialization 次数；
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
Attention(v_q_rope, state_k_cache_out, state_v_cache_out) → v_attn
```

这使 prefill / decode / paged attention 的状态语义在 graph 中可见，但具体 cache layout、page table、device buffer 仍留给 lowering / runtime。

KV cache state 建模规则：

1. `StateValue` 表达逻辑状态身份和版本，例如 `kv_cache.layer_3.key.before_update` 与 `kv_cache.layer_3.key.after_update`；K/V 分别用强类型 `KVCacheSlot::kKey` / `KVCacheSlot::kValue` 表达，不使用字符串 slot；
2. 状态更新必须产生新的 `StateValue`，不能原地修改或复用同一个 value 表示更新前后两个版本；
3. state edge 只表达语义依赖顺序，不描述 cache layout、page table、device buffer、paged cache handle；
4. `ResourceValue` 不用于早期 KV cache 建模；paged cache handle、外部分配 buffer、backend resource 属于 lowering / runtime，等有实际 runtime resource use case 后再进入 ModelGraph；
5. validation 应检查 state update node 的 state input / state output 端口符合 operator schema，不能通过普通 activation edge 隐式表达跨 step 状态。

### 12.1 Attention 语义与执行边界

Phase-1 Attention 算子在 graph 层面表达以下语义契约，物理 layout、page table、device buffer 归 lowering / runtime。

**逻辑签名（Phase 1）：**

| 端口 | 种类 | Rank | Shape | 约束 |
|------|------|------|-------|------|
| q (input 0) | Activation | 2 | `[seq_len, hidden]` | `hidden = num_attention_heads * head_dim` |
| k_cache (input 1) | State | 3 | `[num_key_value_heads, cache_len, head_dim]` | `contributes_tensor_spec = false` |
| v_cache (input 2) | State | 3 | `[num_key_value_heads, cache_len, head_dim]` | `contributes_tensor_spec = false` |
| output (output 0) | Activation | 2 | `[seq_len, hidden]` | 与 q spec 一致 |

**Dtype 契约：** q、k_cache、v_cache 三者 dtype 必须一致，取值 ∈ {Float32, Float16, BFloat16}。输出 dtype 跟随 q。禁止隐式类型转换。

**参数校验：**
- `num_attention_heads`、`num_key_value_heads`、`head_dim` 均为正
- `num_attention_heads % num_key_value_heads == 0`（GQA 约束）
- `num_attention_heads * head_dim` 不溢出 `int64_t`

**静态等式校验：**
- `q.shape[1]`（hidden）、`k_cache.shape[0]`（kv_heads）、`k_cache.shape[2]`（head_dim）在 Phase 1 必须**静态**；symbolic 维度拒绝
- `q.shape[1] == num_attention_heads * head_dim`
- `k_cache.shape[0] == num_key_value_heads`
- `k_cache.shape[2] == head_dim`
- `k_cache.shape == v_cache.shape`（逐维 `AreProvablyEqual`）

**Symbolic 维度处理：**
- `q.shape[0]`（seq_len）为 symbolic 时，推迟到 runtime；语义层仅发 `DimPositiveConstraint`（input port 0, dim 0）。
- `cache_len` 为 symbolic 时，仅当 k_cache 与 v_cache 共享可证相等 symbol 时接受，不发 State-port check。
- 静态零 seq_len / cache_len 拒绝。

**`cache_len` vs runtime `active_len` 边界：** graph 层面的 `cache_len` 是逻辑容量上限（对应 `KVCacheView` 的 `max_tokens`），不等于 runtime 的 `current_pos`（已提交的历史长度）。`CommitUntil(new_pos)` 推进 active frontier，prefill 写 `[0, prompt_len)`，decode 写 `[current_pos, current_pos+1)`，history 读取 `[seq_begin, seq_end)`。graph 语义层不校验 `cache_len == seq_len`。

**不属于 graph 语义层：** batch 维度、attention mask、sliding window、PagedAttention、continuous batching、物理 KV layout、kernel dispatch。

### 12.2 RoPE 语义与执行边界

Phase-1 RoPE 算子在 graph 层面表达以下语义契约（rank-2、无 batch 轴）。sin/cos table 生成、数值计算、kernel dispatch 归 lowering / runtime。

**逻辑签名（Phase 1）：**

| 端口 | 种类 | Rank | Shape | 约束 |
|------|------|------|-------|------|
| q (input 0) | Activation | 2 | `[seq_len, num_attention_heads * head_dim]` | `q[1] == num_attention_heads * head_dim`（静态时） |
| k (input 1) | Activation | 2 | `[seq_len, num_key_value_heads * head_dim]` | `k[1] == num_key_value_heads * head_dim`（静态时） |
| position_ids (input 2) | ModelInput | 1 | `[seq_len]` | dtype 必须为 Int64 |
| q_rope (output 0) | Activation | 2 | 与 q spec 完全一致 | 输出保持输入 TensorSpec |
| k_rope (output 1) | Activation | 2 | 与 k spec 完全一致 | 输出保持输入 TensorSpec |

**Dtype 契约：** q、k dtype 必须一致，取值 ∈ {Float32, Float16, BFloat16}。position_ids 必须为 Int64。输出 dtype 分别跟随 q、k。禁止隐式类型转换。

**参数校验：**
- `head_dim`、`num_attention_heads`、`num_key_value_heads`、`max_position_embeddings` 均为正
- `head_dim` 必须为偶数
- `theta` 有限且为正
- `num_attention_heads * head_dim`、`num_key_value_heads * head_dim` 独立进行溢出检查

**Scaling 契约：**
- `scaling_type == kNone`：标准 RoPE，`scaling_factor` 必须缺席
- `scaling_type == kLinear`：`scaling_factor` 必须存在、有限且 `> 0`；`1.0` 合法且不做归一化改写
- 其余 `HfRopeScalingType`（Dynamic NTK、YaRN、Llama3、LongRope、Su、Unknown）因当前 `RoPEParams` 无法表达而被拒绝（capability-oriented 诊断），并非 Phase-1 策略性 blanket 拒绝

**静态等式校验（仅静态维度）：**
- `q.shape[1] == num_attention_heads * head_dim`（静态时强制；symbolic 宽度合法，不发 product 约束）
- `k.shape[1] == num_key_value_heads * head_dim`（同上）
- 不添加 `num_attention_heads % num_key_value_heads == 0` 的 GQA 约束（该约束归 Attention）

**Symbolic 维度处理与 runtime checks：**
- q、k、position_ids 的 seq_len 三方协调：`AreProvablyEqual` 时不发约束；双方均静态且不等则拒绝；否则发 `DimEqualConstraint`
- symbolic q seq_len 先发一个 `DimPositiveConstraint`（input 0, dim 0），随后发未决的 q/k、q/position 等式检查（确定性顺序）
- 最大 check 集合：1 个 positivity + 2 个 equality；全静态合法输入不发任何 check
- 静态 q seq_len `<= 0` 拒绝

**执行边界（不属于 graph 语义层）：**
- 本语义层**不**检查 position tensor 内容。未来 executable RoPE path 必须校验非负 position ID、scaling 契约定义的 effective-position 上界、以及 symbolic q/k 宽度与 params 的一致性后才能计算
- 不声称 `position_ids < max_position_embeddings`——该 coordinate/bound 策略未在本语义任务中冻结
- Loader `allow_rope_scaling` 仍是独立策略；语义接受不等于当前 end-to-end kernel 支持
- sin/cos table 属于派生常量（可从 `RoPEParams` 完全确定），应在 lowering 阶段生成
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

### 13.1 Reshape 语义与执行边界

`Reshape` 是 ModelGraph 中的语义算子：保留 dtype 和 row-major 逻辑线性元素顺序，仅改变逻辑 shape。数学定义等价于 row-major logical flatten 后再 row-major logical unflatten 到目标 shape；输出体积必须等于输入体积，元素值与 dtype 不变。

**目标 shape 强类型**：不使用 ONNX `0/-1` 哨兵语义，避免多个动态输入维度无法表达以及 importer 约定泄漏到核心 IR。目标维度由 `ReshapeDim` variant 表达：

- `ReshapeLiteralDim{int64_t value}`：非负字面维度（如 `32`）；
- `ReshapeInputDim{uint32_t axis}`：对输入轴的引用（如 `@0` 复用输入第 0 轴的 `ShapeSymbol`），`axis` 为 canonical 非负索引；
- `ReshapeInferDim`：推断标记（如 `*`），由 `InferReshape` 解析。

`ReshapeParams{std::vector<ReshapeDim> target_shape}` 中 `target_shape.size()` 即输出 rank；空 vector 表示 rank zero（标量，体积 1）。

**边界规则**（由 `InferReshape` 强制，而非类型系统）：

- 字面值非负；
- `ReshapeInputDim.axis` 必须 < 输入 rank；
- 至多一个 `ReshapeInferDim`；
- 输入必须 ranked（unranked 拒绝）；
- infer 标记静态可解当且仅当输入维度与所有非 infer 输出维度均静态：商必须可整除且 fit `int64_t`；非 infer 乘积为零时拒绝（零体积歧义）；不可静态解析时 emit `ShapeSymbol::Unknown()` 并附一条 deferred `VolumeEqualConstraint`；
- 始终 emit 一条 `VolumeEqualConstraint` 覆盖输入和输出全维度：证伪立即拒绝，证真省略，待定持久化为 deferred runtime check。

**dtype / quantization 保留**：`AddReshape` helper 将输入的 `QuantizationSpec` 复制到输出 value descriptor；dtype 不限制为算术子集。

**当前能力边界（语义专用）**：图可以构造、推断、校验、dump、rewrite Reshape，但**不存在可执行 lowering**——没有匹配的 kernel descriptor、alias / view legality、materialization、stride propagation、workspace 或 constant evaluator。语义合法性与判定与未来 view / materialization 的物理合法性是两件事。

### 13.2 Permute 语义与执行边界

`Permute` 是 ModelGraph 中的语义算子：按显式 `permutation` 重排输入张量的轴顺序，保留 dtype、轴数量与每个轴的 `ShapeSymbol` 身份，但**不保留 row-major 逻辑线性元素顺序**（这与 `Reshape` 不同：`Reshape` 保留 row-major linear order，而 `Permute` 通常会改变它）。等价于 NumPy `numpy.transpose`、PyTorch `torch.permute`、ONNX `Transpose`（无 `perm` 默认反转轴的语义在这里不适用——AetherMind 强制要求显式 `permutation`）。

**强类型参数**：`PermuteParams{std::vector<uint32_t> permutation}`。

- `permutation[j]` 是**输出轴 `j` 对应的输入轴索引**（零基、非负、canonical）。
- `permutation.size()` 即输出 rank；空 vector 表示 rank zero（标量上的恒等）。
- 不使用 ONNX 反转默认、不使用负轴归一化、不使用 importer 特定约定；轴索引必须显式、canonical、非负。

**形状方程**（由 `InferPermute` 强制）：

```
output.shape[j] = input.shape[permutation[j]]   对 0 <= j < permutation.size()
```

例：`input.shape = [2,3,4]`，`permutation = [2,0,1]` ⇒ `output.shape = [input.shape[2], input.shape[0], input.shape[1]] = [4,2,3]`。

**坐标方程**（语义参考，不进入 IR）：

对于输出坐标 `o = (o_0, ..., o_{r-1})`，对应的输入坐标 `i` 满足 `i[permutation[j]] = o_j` 对所有 `0 <= j < r`。

**双射规则**（由 `InferPermute` 强制，而非类型系统）：

- `permutation` 必须是 `[0, input_rank)` 上的**完整双射**：每个输入轴恰好出现一次。
- 重复轴（如 `[0,0]`）非法——与 `Reshape` 不同，`Reshape` 允许同一输入轴多次出现在 `target_shape`，但 `Permute` 必须是双射。
- 缺失轴非法。
- `permutation.size()` 必须 == 输入 rank。
- `permutation[k]` 必须 < 输入 rank。
- 输入必须 ranked（unranked 拒绝）。
- 验证优先级：先 `PermuteParams` 变体检查，再扫描重复轴（即使 arity 也错时重复轴优先报告），再 arity/rank/长度/范围检查。

**恒等与边界情形**：

- rank zero（`permutation = []`）合法，是标量上的恒等。
- rank one（`permutation = [0]`）合法，是恒等。
- 任意 `permutation = [0,1,...,r-1]` 是恒等。
- 静态 `[2,3,4]` 与 `[2,0,1]` 产生 `[4,2,3]`。
- 符号维度 `[A,B,C]` 与 `[2,0,1]` 产生 `[C,A,B]`，且输出第 0 轴的 `ShapeSymbol` 与输入第 2 轴的 `ShapeSymbol` 是同一对象（身份保留），用于后续约束求解。

**dtype / quantization 保留**：`AddPermute` helper 将输入的 `QuantizationSpec` 复制到输出 value descriptor；dtype 不限制为算术子集；输出 dtype == 输入 dtype。`runtime_checks` 向量为空（语义静态完整可判定，无需 deferred 约束）。

**不 emit 约束**：`InferPermute` 不构造 `VolumeEqualConstraint` 或任何 `ShapeConstraint`——双射性保证输入输出体积自动相等，无需运行时检查；也不构造 unknown symbol、不修改 solver 状态。

**当前能力边界（语义专用）**：图可以构造、推断、校验、dump `Permute`，但**不存在可执行 lowering**——没有匹配的 kernel descriptor、alias / view legality、materialization、stride propagation、workspace、constant evaluator 或 importer 默认行为。语义合法性与未来物理 view / materialization 合法性是两件事：未来若添加对应 kernel，仍须独立验证 stride / contiguity / alias 合法性，本节语义契约不构成对物理执行的承诺。

### 13.3 Reorder 语义与执行边界

`Reorder` 是 ModelGraph 中的语义算子：逻辑上完全保持张量不变（dtype、shape、`ShapeSymbol` 身份、所有坐标值），但记录一个固定的物理目标意图——**canonical row-major contiguous 存储**。等价语义参考：oneDNN `Reorder`（`https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_reorder.html`）和 PyTorch `Tensor.contiguous()`（`https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html`）。

**强类型参数**：`ReorderParams{}`——空类型化参数对象，区分 Reorder 与逻辑 Permute 和 dtype 转换。无字段，因为目标始终是 canonical contiguous。

**坐标方程**（由 `InferReorder` 强制）：

```
Y[i_0, ..., i_(r-1)] = X[i_0, ..., i_(r-1)]   对所有合法坐标元组
```

即输出与输入在每个逻辑坐标上完全相等。`Y.dtype == X.dtype`，`Y.shape == X.shape`（精确 `ShapeSymbol` 身份，不仅是静态值相等）。

**与 Permute / Reshape / Cast 的区别**：

| 算子 | 逻辑变化 | 物理意图 |
|------|----------|----------|
| `Permute` | 改变轴顺序（shape 变化） | 无固定目标 |
| `Reshape` | 改变形状（保留 row-major linear order） | 无固定目标 |
| `Cast`/量化转换 | 改变 dtype | 无固定目标 |
| `Reorder` | **无逻辑变化** | canonical row-major contiguous |

例：`Permute` 可能将 `[2,3,4]` 变为 `[4,2,3]`；随后的 `Reorder` 保持 `[4,2,3]` 不变，仅标记存储必须 canonical 化。

**接受的输入状态**：

- ranked 静态形状（如 `[4, 256]`）
- ranked 符号形状（如 `[A, B]`，`ShapeSymbol` 身份保留）
- unranked（输出同样 unranked）
- rank zero（标量）
- 含零维度的形状（如 `[0, 3]`）
- 任意 dtype（不限于浮点子集）

**推断行为**：`InferReorder` 直接复制 `inputs[0]` 为唯一输出——不检查 rank、不检查维度值、不检查 dtype、不检查 stride/layout、不构造新 `SymbolicShape`、不 emit 约束、不填充 `runtime_checks`。验证优先级：先 `ReorderParams` 变体检查，再 arity 检查。

**dtype / quantization 保留**：`AddReorder` helper 将输入的 `QuantizationSpec` 复制到输出 value descriptor；输出 dtype == 输入 dtype。`runtime_checks` 向量为空。

**DCE 行为**：Reorder 是 pure、deterministic、`RuntimeOnly()` schema——未使用的 Reorder 节点可被 DCE 安全移除；但**不存在代数恒等消除**——即使输出 `TensorSpec` 等于输入，只要输出是活的（graph output 或有下游消费者），DCE 不会移除它。Reorder 是 materialization/layout-intent 边界，不是普通 identity。

**当前能力边界（语义专用）**：图可以构造、推断、校验、dump `Reorder`，但**不存在可执行 lowering**——没有匹配的 kernel descriptor、materialization、allocation、workspace、constant evaluator、target-format 字段、channels-last/blocked/opaque 布局支持或 importer 默认行为。未来 lowering 可能复用已经 canonical 的输入（零拷贝）或另行 materialize；本节语义契约不承诺执行时一定拷贝、一定分配新存储、一定 alias、或一定零拷贝。`TensorSpec` 有意不暴露物理变化；`OpType::kReorder` 本身携带 lowering 需求。

## 14. Shape 与动态维度

动态 shape 是语义信息，应该进入 `ModelGraph`；具体 runtime shape binding 不应该进入 `ModelGraph`。

建议模型：

```text
ModelGraph:       token_ids [seq_len]
                  hidden    [seq_len, hidden_size]
                  kv_cache  [num_kv_heads, cache_len, head_dim]

Runtime ShapeEnv: seq_len = prompt length 或 1
                  cache_len = previous length + current length
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

Semantic pass 默认采用第 10 节定义的 immutable snapshot + rewrite transaction 模式：pass 读取合法 graph snapshot 或 `GraphRewriteSession` virtual view，输出 `GraphMutation` 列表或调用 session 受控 API；`GraphPassManager` 在 phase checkpoint 统一 commit 新的 `ModelGraph`。

注意：semantic fusion rewrite 可以发生在 `ModelGraph`，但具体 fused kernel、workspace、launch config 仍属于 lowering / backend plan。

### 16.1 GraphPass 当前接口

当前实现中，所有语义 pass 继承 `GraphPass`。Pass 不接收裸 `ModelGraph&`，而是通过 `GraphRewriteSession` 读取 virtual view 并记录 mutation：

```cpp
class GraphPass {
public:
    virtual ~GraphPass() = default;

    virtual std::string_view Name() const noexcept = 0;

    /// 所有图读取和 mutation 记录都通过 session 完成。
    /// 返回非 OK Status 时，GraphPassManager 必须立即终止 pipeline。
    AM_NODISCARD virtual Status Run(GraphRewriteSession& session, const PassContext& ctx) = 0;
};
```

设计约束：

1. `Run()` 必须返回 `Status`，不能降级为 `bool`；`bool` 无法区分“未修改”和“非法图结构”。
2. Pass 在 transaction 内必须通过 `GetNodeView()` / `GetResolvedValue()` 读取逻辑图，不能直接读取原始 `ModelGraph::GetNode()` 后自行解释 inputs。
3. Pass 是否产生 mutation 不应通过返回值表达；如未来需要，可在 `GraphRewriteSession` 上增加 `HasMutations()` 或 mutation 计数查询。

### 16.2 GraphPassManager 与 checkpoint 策略

`GraphPassManager` 按注册顺序运行 pass，并通过 `SetCheckpointEvery(N)` 控制 materialization 频率：

```cpp
class GraphPassManager {
public:
    GraphPassManager() = default;

    GraphPassManager& Add(std::unique_ptr<GraphPass> pass);

    // pass_count == 0: 禁用中间 checkpoint，只在 pipeline 末尾 commit。
    // pass_count == 1: 每个 pass 后 commit。
    // pass_count == N: 每 N 个 pass 后 commit。
    GraphPassManager& SetCheckpointEvery(size_t pass_count) noexcept;

    AM_NODISCARD StatusOr<ModelGraph> Run(const ModelGraph& graph) const;
};
```

`SetCheckpointEvery(N)` 比 production/debug 二元开关更通用：

| 场景 | 推荐配置 |
|---|---|
| 生产模式，最少重建 | `SetCheckpointEvery(0)` |
| CI / 调试，每个 pass 后验证 | `SetCheckpointEvery(1)` |
| 大 pipeline 折中排查 | `SetCheckpointEvery(N)` |

`GraphPassManager::Run()` 必须遵守：

1. 任一 pass 返回非 OK `Status` 时立即停止并向上传播错误；
2. checkpoint 后的新图作为后续 pass group 的 immutable snapshot；
3. pipeline 末尾避免对刚 checkpoint 过的图重复 commit；
4. dump / instrumentation 只能增加可观测性，不能改变优化语义。

**错误语义（all-or-nothing）**：任一 pass 失败时，`Run()` 只返回错误 `Status`，**不返回部分结果**。已 checkpoint 的中间快照不暴露给调用方，session 状态不可恢复。调用方收到错误时，输入 `ModelGraph` 本身不受影响，可安全重试或换用空 pipeline。

**空 pipeline 行为**：当未注册任何 pass 时，`Run()` 返回输入图的有效副本（经 `Commit()` 生成），保证调用方始终拿到一个独立的 `ModelGraph`。

### 16.3 PassContext 扩展方向

 `PassContext` 作为 `GraphPassManager` 的增量配置层，集中管理优化级别、feature flags 和 dump 行为，和 checkpoint 机制进行配合。

建议目标形态：

```cpp
struct PassContext {
    // 0: 禁用优化；1: 基础清理；2: 语义融合；3: 激进优化。
    uint32_t opt_level = 2;

    // 继承当前 GraphPassManager 的 checkpoint 语义。
    // 0 表示仅 pipeline 末尾 commit。
    size_t checkpoint_every = 0;

    // Debug 与可观测性。第一版复用文本 DumpGraph；DOT/JSON 可后续扩展。
    bool dump_after_checkpoint = false;
    std::filesystem::path dump_dir{};

    bool enable_qkv_fusion = true;
    bool enable_swiglu_fusion = true;
    bool enable_fused_add_rms_norm = true;
};
```

`PassContext` 已引入，pass 接口为：

```cpp
AM_NODISCARD virtual Status Run(GraphRewriteSession& session,
                                const PassContext& ctx) = 0;
```

兼容策略：`SetCheckpointEvery()` 保留为便捷 API，内部写入 `ctx_.checkpoint_every`；所有错误仍通过 `Status` 传播，不能改成 `bool`。

**Fusion flags 使用约定**：`enable_qkv_fusion` / `enable_gate_up_fusion` / `enable_swiglu_fusion` / `enable_fused_add_rms_norm` 默认全 `true`（激进优化）。每个 fusion pass 在 `Run()` 入口自行检查对应 flag，若禁用则直接返回 `Status::Ok()` 跳过。约定模式：`if (!ctx.enable_qkv_fusion) { return Status::Ok(); }`。`opt_level` 用于控制 pass 是否注册到 pipeline（由 `GraphPassManager` 或上层决定），flag 用于控制已注册 pass 的运行时行为。

### 16.4 当前优化 Pass Pipeline

当前仓库已有 graph-owned pass framework 和 rewrite session；compiler-owned `OptimizeModelGraph` 负责把这些 backend-independent passes 组装为默认 pipeline。各 pass 当前状态标注如下：`[已实现]` / `[未实现]` / `[规划中]`。

#### 默认 Pipeline：优化级别与 Pass 选择

`OptimizeModelGraph` 根据 `PassContext::opt_level` 确定性地选择 pass 管道，不由 feature flags 驱动（flags 仅控制已注册 pass 的运行时行为）：

| 级别 | 注册的 Pass | 说明 |
|---|---|---|
| O0 | 无 | 输入图不经任何优化直接通过；`GraphPassManager` 生成输入图的合法副本 |
| O1 | `ConstantFoldingPass` → `DeadCodeEliminationPass` | 基础清理：折叠纯常量子图，删除不可达节点；不进行语义融合 |
| O2+ (默认) | `ConstantFoldingPass` → `QkvLinearFusionPass` → `GateUpLinearFusionPass` → `SiluMulFusionPass` → `AddRmsNormFusionPass` → `DeadCodeEliminationPass` | 完整优化：常量折叠 + QKV/Gate-Up/SwiGLU 语义融合 + Add-RMSNorm 融合 + 死代码消除 |

默认优化级别为 O2（`PassContext::opt_level` 默认值为 2）。O3 及更大的级别（如 99）使用与 O2 相同的 pipeline。

**Flag 与注册分离规则**：Pass 注册完全由 `opt_level` 决定，feature flags（`enable_constant_folding`、`enable_qkv_fusion`、`enable_gate_up_fusion`、`enable_swiglu_fusion`、`enable_fused_add_rms_norm`、`enable_dce`）在 pipeline 构建时不参与判断。Flag 通过 `PassContext` 传入每个 pass 的 `Run()` 方法，由 pass 内部自行检查是否跳过执行。这保证了 pipeline 的可观测性和可预测性；禁用某 flag 时该 pass 仍在管道中，但行为上被视为跳过。

#### 顺序约束

ConstantFolding 必须优先于 SiluMulFusion，因为：

1. CF 可能折叠 SiLU 节点（当其 gate 输入为内联常量时，`silu(gate)` 是纯常量计算，可被常量求值器整体折叠），将 SiLU 的输出替换为折叠后的常量值；此后该 SiLU 计算在运行时已不存在；
2. 融合 pass 必须在入口识别这一情况并**跳过融合**：`FindSiluMulPattern` 检查 `session.GetResolvedValue(silu_out) != silu_out`（`mul_out` 同样检查），若输出已被前置 pass 替换，融合不会通过替换链"重新触发"——重新融合会把已消除的 silu 计算以 fused kernel 形式重新引入运行时，并使折叠后的常量成为死值（其 replacement binding 被孤立）。跳过是正确的语义选择：剩余的普通 `Mul(常量, up)` 已是最优；
3. DCE 在融合之后运行：在 CF 已折叠 SiLU 的路径上，DCE 删除输出被替换的原始 SiLU producer 节点；在常规融合路径上，DCE 删除被融合替换的 SiLU 和 ElementwiseMul 节点。

最终优化图等价于无 checkpoint（`checkpoint_every=0`）的运行结果，即使设置了中间 checkpoint。

`QkvLinearFusionPass` 的注册位置同理：

1. CF 必须先于 QkvLinearFusion：`Linear` 是 `RuntimeOnly`（不可被 CF 折叠），CF 在前不会替换 Q/K/V 候选节点、不破坏融合匹配；同时保持“折叠优先、融合次之”的统一顺序，避免未来 Linear 变为可折叠时重现 SiluMul 的“重引入已折叠计算”缺陷；
2. QkvLinearFusion 先于 SiluMulFusion：两者模式完全不相交（QKV 以权重角色锚定 attention 块的三个 `kLinear`，SiluMul 匹配 MLP 块的 `kSilu`+`kElementwiseMul`），顺序无功能依赖；QKV 在前可更早收敛节点数（3→1）；
3. DCE 最后：融合后失去消费者的原 Q/K/V 权重值与不可达残留由 DCE 统一收尾（QkvLinearFusion 自身已做死值防御——候选输出必须为 graph output 或有 live consumer）。

#### Pass 实现状态

##### Phase 1：基础清理（已实现）

- **`ConstantFoldingPass`** `[已实现]`：仅折叠纯函数、compile-time constant 输入的安全子图；不得折叠 runtime input、state 或外部权重依赖。
- **`DeadCodeEliminationPass`** `[已实现]`：以 graph outputs、stateful update 和必要 side-effect 节点为 roots，通过 `GraphRewriteSession::HasLiveConsumers()` 确定节点存活状态，删除不可达节点；DCE 后必须 commit 生成一致新图。DCE 会删除被 ConstantFolding 替换输出后的原始 producer 节点（这些节点的输出已被折叠为内联常量，不再需要原节点）。

##### Phase 2：LLM 语义融合（部分实现）

- **`QkvLinearFusionPass`** `[已实现]`：以权重 `TransformerWeightRole`（`kAttentionQ`/`kAttentionK`/`kAttentionV`）为锚，匹配同一输入、同一 `decoder_layer_index` 的 Q/K/V 三个 `Linear`。校验节点 live、输出未被前置 pass 替换（`GetResolvedValue` 解析后与自身不同则跳过）、输出有 live consumer 或为 graph output、权重 dtype 与 `QuantizationSpec` 一致、in_features 可证相等后，通过 `AddSessionWeight` 创建 `QkvWeightBinding` 复合配方权重（spec `[q_out+k_out+v_out, in_features]`，语义为 concat(Q, K, V, axis=0)，split 边界由 `QkvLinearParams` 记录），并以 `SubgraphBuilder` 替换为单个 `OpType::kQkvLinear` 节点（三输出 q/k/v 依次接管原值）。任一条件不满足即跳过（安全保守）；同组重复角色使组 `ambiguous`，整组跳过。融合后的图保持可 lowering（`LowerModelGraph` 为 schema 驱动，对 `kQkvLinear` 无需特判）。
- **`SiluMulFusionPass`** `[已实现]`：匹配 `gate -> silu -> mul(up)`，支持 Mul 输入反向，检查 `silu_out` 单 consumer、非 graph output、`decoder_layer_index` 一致后合并为 `OpType::kSiluMul`。若 `silu_out` / `mul_out` 已被前置 pass（如 CF）替换（`GetResolvedValue` 解析后与自身不同），跳过融合——重新融合会以 fused kernel 形式重新引入已消除的 silu 计算，并使折叠常量成为死值。
- **Attention 不参与语义层 fusion**：attention 在语义层始终是 `kAttention` 单节点，causal masking 与 softmax 属于 attention 语义自身，不作子图提升；是否使用 fused / FlashAttention-like kernel 由 lowering / backend plan 决定。
- **`AddRmsNormFusionPass`** `[已实现，已注册 O2]`：以 `RmsNorm` 为锚，将 exact-shape residual `Add` 与后续 `RmsNorm` 融合为 `OpType::kAddRmsNorm`。输入固定为 `[branch_input, residual, weight]`，双输出 `output` / `new_residual` 分别接管原 RMSNorm / Add value identity，因此同时覆盖层内 `post_attention_add → post_attention_norm`、层间 `mlp_add(i) → input_norm(i+1)` 和末层 `mlp_add(last) → final_norm`；fused node 继承 RMSNorm 的 `decoder_layer_index`。broadcast Add、输出已被替换、dead chain 或同一 Add 输出被多个 RMSNorm 消费时保守跳过。`new_residual` 是普通 activation value，不建立 graph-level alias 或 must-alias 约束。注册位置：默认 O2 pipeline 中 `SiluMulFusionPass` 之后、`DeadCodeEliminationPass` 之前（与 QKV/SiluMul 模式不相交，无功能依赖；DCE 收尾原则不变）。**⚠️ 运行时 gate**：`AddRmsNormOp` 已实现通用 Operator dispatch，但 backend kernel 尚未实现——默认编译路径当前会生成该 fused op，编译/lowering 成功，但运行时 `Prepare`（`ResolveKernelInfo(kAddRmsNorm)` 无注册 kernel）会失败。kernel 落地（含 `WeightPrepackPlanner`/`FindPackedWeights` 的 `kAddRmsNorm` prepack 键）之前，fused 图不可端到端执行。

每个真实 pass 至少需要覆盖匹配成功、匹配失败、安全跳过、非法输入四类测试；fusion 后的图必须通过 `Validate()`，并保持可 lowering。

### 16.5 TVM / Relax 参考原则与非目标

`ModelGraph` rewrite 设计可以参考 Apache TVM / Relax 的工程思想，但不能照搬 TVM 的对象模型或编译栈。AetherMind 的目标是借鉴成熟的 pass 组织与 pattern matching 思路，同时保持 ID-based、扁平、轻量的推理引擎 IR。

建议参考的方向：

1. **遍历与 Mutation 收集**：pass 在 `GraphPass::Run` 内按拓扑序遍历 graph，识别可重写节点或子图，并产生 `GraphMutation`。遍历逻辑直接内联在 pass 中，不引入额外的 mutator 基类——per-node 遍历足够简单，手写循环比继承层次更清晰。子图构造可使用 `SubgraphBuilder` 封装 virtual value 分配和 `RewriteOutputBinding` 绑定。
2. **Dataflow Pattern Matching**：可以实现 TVM DFPattern-inspired 的极简结构匹配器，用于匹配 `OpType`、输入输出连接、payload 类型和基础 attrs。复杂 shape / params predicate 应等 typed `OpParams` 落地后再扩展。
3. **Semantic Pass 与 Lowering Pass 分层**：target-independent 的数学等价变换、canonicalize、dead value detection、fusion marking 属于 semantic pass；kernel dispatch、packed weight、workspace、buffer aliasing、device/resource binding 属于 lowering / execution planning。
4. **显式 StateValue**：KV cache、decode state 等跨 step 状态通过显式 state input / output 表达，避免隐藏副作用。

必须避免的方向：

1. 不引入 TVM `ObjectRef` / AST pointer ownership 模型；`ModelGraph` 保持 `GraphNodeId` / `GraphValueId` + `std::vector` 存储。
2. 不引入 TVM TIR / TensorIR 级别 kernel generation；AetherMind lowering 只做 kernel dispatch / fused kernel selection，不做运行时代码生成。
3. 不引入完整 Relax/Relay type inference 系统；仅实现 LLM 推理所需的符号 shape、dtype、schema validation 与约束传播。
4. 不实现完整 TVM DFPattern 语言；第一版只做保守的结构匹配。

关键约束：pass 在 transaction 内读取节点时必须通过 `GraphRewriteSession` 的 virtual view，例如 `GetNodeView()`，以获得已解析 replacement、过滤 tombstone 后的最新视图；不能直接读取原始 `ModelGraph::GetNode()` 后自行解释 inputs。

### 16.6 Graph Compiler API

compiler 模块提供两个显式的图编译阶段入口：`OptimizeModelGraph` 组装 graph-owned semantic passes，`LowerModelGraph` 产生 compiler-owned artifact。`ModelCompiler` 是 production 模型路径的组合入口。

#### `OptimizeModelGraph`

```cpp
// include/aethermind/compiler/semantic_optimization_pipeline.h
AM_NODISCARD StatusOr<ModelGraph> OptimizeModelGraph(
        const ModelGraph& graph,
        PassContext context = {});
```

应用基于 `context.opt_level` 的确定性 pass pipeline（见 16.4 默认 pipeline 表），返回新 `ModelGraph`。源图永不修改。所有 `PassContext` 字段——feature flags、`checkpoint_every`、`const_eval_policy`——逐字转发给每个 pass。

#### 两阶段组合调用

```cpp
// include/aethermind/compiler/graph_lowering.h
AM_NODISCARD StatusOr<LoweredGraph> LowerModelGraph(
        const ModelGraph& graph,
        const GraphLoweringConfig& config = {});
```

调用方依次执行 `OptimizeModelGraph(graph, opt_ctx)` → `LowerModelGraph(optimized, lower_cfg)`，各阶段失败即返回错误 `Status`。两个阶段的配置分别由 `PassContext` 与 `GraphLoweringConfig` 独立提供（默认 O2 优化、CPU/scalar/plain/both target selector）。`KernelSelector` 是 base 纯数据契约，不使 compiler 查询 backend/kernel registry。

#### 所有权设计

返回的 `LoweredGraph` 自身已自包含，不依赖任何源图存活：

- 常量折叠产生 inline `ConstantValue`，lowering 后 `ConstantBinding`（`shared_ptr` 内联数据，共享所有权）保留在 `LoweredGraph::values()` 的 payload 中；
- 所有 `TensorSpec`、`ShapeConstraint`、`runtime_checks`、dtype 在 lowering 时值拷贝；
- `lowered.model_outputs()` 中的 `GraphValueId` 与调用方持有的优化图 `GetOutputs()` 中的 ID 一致（需要语义回查时，由调用方自行持有优化图）；
- standalone constant graph output 的内联字节经 `LoweredGraph::values()` 的 payload 保持可达，编译返回后无需回读图；
- `LoweredGraph` storage 私有且只暴露 const spans；`LowerModelGraph` 在 finalization 时验证 schema arity、value/spec bindings、selector dtype、model I/O 与声明式 state alias。该验证不重跑 `InferOperator`。

> 注：早期设计曾有 `CompiledModelGraph { optimized_graph, lowered }` 包装、`CompileModelGraph` 组合器与 `GraphCompileConfig` 聚合配置；随实现演进（执行路径不读图、数据值拷贝/共享所有权、组合逻辑仅为顺序调用），三者均已移除，收敛为两个显式阶段函数。

#### 模块所有权

- graph 保留 `ModelGraph`、GraphRewrite/GraphPassManager 与所有 backend-independent semantic pass；不得 include compiler/execution/backend/model。
- compiler 拥有 `OptimizeModelGraph` 的 pipeline composition、`LowerModelGraph`、`LoweredGraph`/`LoweredStepSpec`、`ModelCompiler` 与 `ModelCompileOptions`；不得 include execution/backend/runtime。
- execution 消费 finalized `LoweredGraph`，在内部把 state aliases 转为 `StateAliasPlan` 并进行 workspace/kernel planning；raw `ExecutionPlanNodeSpec` 是会重跑 `InferOperator` 的 untrusted adapter。

## 17. Lowering

DAG lowering（compiler）的职责：

1. 拓扑排序或生成合法执行 schedule；
2. 按 operator schema 顺序解释 `GraphNode.inputs` / `GraphNode.outputs`；
3. 解析 `WeightValue` 的逻辑权重引用；
4. 将 `GraphLoweringConfig` 的 base `KernelSelector` 与经验证的 dtype 写入 `LoweredStepSpec`；
5. 声明式记录 state must-alias；
6. 生成并验证 `LoweredGraph`。

kernel registry 查询、packed weight materialization、workspace planning、runtime buffer alias binding 与 `ExecutionPlan` 构建均属 execution/backend 边界，不属于 compiler lowering。

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

- `include/aethermind/graph/graph.h`（v1 时代为 `model/graph/model_graph.h`）；
- `src/graph/graph.cpp`（v1 时代为 `model/graph/model_graph.cpp`）；
- `src/model/model_graph_builder.cpp`（v1 时代为 `model/graph/model_graph_builder.cpp`）；
- `tests/unit/model/test_model_graph_builder.cpp`（v1 时代为 `model/graph/test_model_graph_builder.cpp`）；
- 任何遍历 `GraphNode.inputs` / `GraphNode.outputs` / `GraphNode.weights` / `GraphNode.workspace_requirement` 的 execution plan 代码。

产出旧字段到新字段的迁移表：

```text
GraphNode.inputs:  vector<TensorSpec>        -> vector<GraphValueId>
GraphNode.outputs: vector<TensorSpec>        -> vector<GraphValueId>
GraphNode.weights: vector<WeightBinding> -> WeightValue payload
TensorSpec on edge                         -> GraphValue.spec
WorkspaceRequirement on GraphNode          -> execution planning
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
- graph dump；
- serialization；
- round-trip validation；
- pattern matcher；
- semantic pass pipeline。

这些能力不应提前阻塞 M1 / M2 的核心结构迁移。

##### M4 实际完成范围与延迟项

已交付：

- `GraphRewriteSession`：含 `AllocateVirtualValue` / `ReplaceSubgraph` / `RemoveNode` / `RedirectInput` / `ReplaceValue` / `Apply` / `Commit`，支持不可变快照 + 事务式重写；`ReplaceValue` 内置环检测；`GetResolvedValue` 使用 `mutable` 路径压缩缓存；`Apply()` 内 `overloaded` visitor 在循环外构造，避免每次迭代重新构造；
- `GraphRewriteSession` 节点枚举 API：`IsNodeLive` / `GetTopologicalOrder` / `FindNodesByOpType`，让 pass 可以独立发现节点而非依赖外部传入 id。`IsNodeLive` 是节点活跃度的单一真理源，`GetNodeView` 复用其判断以避免逻辑漂移；
- `GraphRewriteSession` 值枚举 API：`IsValueLive` / `GetLiveValues` / `FindConsumers` / `HasLiveConsumers`，与节点枚举 API 对称，让 pass 可以独立发现值与消费者。`IsValueLive` 表达"结构性存在"语义（非 DCE-reachable），三个 live 条件覆盖外部值、live producer、active rewrite 的 `replaces` binding；`FindConsumers` 对查询值与节点输入都经 `GetResolvedValue` 解析后比较，使 `ReplaceValue` 透明反映，但**仅返回 live 原图节点**（replacement 节点无 GraphNodeId）；`HasLiveConsumers` 是 DCE 正确判定 API，同时扫描原图节点和 active replacement 的 inputs；virtual value 与越界 id 返回空 / false；
- `GraphRewriteSession` Commit 重构（Q-1）：`Commit()` 拆分为 4 个 const helper（`CopyExternalValues` / `EmitRewrite` / `EmitOriginalNode` / `MarkCommittedOutputs`）+ `ValueMap` 别名，降低单函数圈复杂度；
- `GraphRewriteSession` RedirectInput 条件简化（Q-2）：将嵌套条件 `active && exposes_node_view && old_nodes.size()==1 && old_nodes[0]==node && replacements.size()==1` 替换为 `existing.has_value() && IsNodeLive(node)`，注释说明 `IsNodeLive` 与 `has_value()` 的语义差异（前者包含 untouched 节点，后者排除），避免未触及节点触发 SIGSEGV；
- `GraphRewriteSession` 内部状态：用统一的 `RewriteEntry { old_nodes, replacements, active, exposes_node_view }` 表示 node/subgraph rewrite；`node_to_rewrite_` 指向 active rewrite；`RemoveNode` 是 `ReplaceSubgraph({node}, {})`；`RedirectInput` 通过 mirror `ReplacementNode` 暴露 node view；virtual value 仅允许在同一个 rewrite 内部作为 replacement 子图边；
- `GraphMutation`：typed variant（`SubgraphReplacement` / `NodeRemoval` / `InputRedirection` / `ValueReplacement`），由 `Apply()` 批量提交；
- `GraphPass` / `GraphPassManager`：含 `SetCheckpointEvery(N)` phase checkpoint，最后一个 pass 落在 checkpoint 边界时正确 materialize；`SetCheckpointEvery(0)` 禁用 checkpoint；`Run()` 避免初始图深拷贝；
- graph dump：`DumpGraph` / `DumpOpParams`，覆盖全部 payload kind、OpParams variant、QuantizationSpec；
- `OpParams` serialization / round-trip：`SerializeOpParams` / `ParseOpParams`，使用 `std::from_chars` 替代 `std::stod`；
- `QuantizationSpec`：类型定义 + `GraphValue::quantization` 字段 + `ModelGraph::SetQuantization` API + dump 输出；
- `ConstantValue` payload：`ConstantBinding` 用 `shared_ptr<const vector<byte>>` 实现 interning，避免 `Commit` 深拷贝；
- lowering 不单独索引常量输入：`ConstantValue` 的 binding 保留在 `LoweredGraph::values` 的 payload 中，后端按真实查询模式经 value id 延迟解析。

延迟项（不阻塞 M4 验收，留待后续 milestone 或 use case 驱动）：

- `PassContext` dump 配置：`PassContext` 已引入基础字段（opt_level、checkpoint_every、fusion flags），dump 配置字段（dump_after_checkpoint、dump_dir）待引入；
- 完整图 serialization：当前仅 `OpParams` 序列化完成，图级序列化留待 use case 出现；
- round-trip validation：依赖完整图序列化；
- pattern matcher：待 graph rewrite 用例驱动；
- `QuantizationSpec` lowering 集成：当前仅记录语义 scheme，不参与 dtype 推断 / packed weight 生成；
- `ConstantBinding` 外部存储解析：`inline_data` 已 intern，但大型常量通过 `name` 引用外部存储的解析机制未实现；
- 算子注册 `kConstant` 输入端口：当前无算子声明 `kConstant` 输入端口；lowering visit 分支已就位，待 use case（如常量 attention mask）驱动时注册。

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
8. 不因起步范围较小而过早实现完整 compiler framework；
9. 不把 `GraphPass::Run()` 的错误通道降级为 `bool`；pass 发现非法图结构必须返回 `Status`；
10. 不在 semantic pass 中绕过 `GraphRewriteSession::GetNodeView()` 直接解释原始 `ModelGraph` inputs（此规则由接口设计强制：`GraphPass::Run()` 只接收 `GraphRewriteSession&`，不接收 `const ModelGraph&`，pass 无法直接访问原图）；
11. 不让 debug dump / instrumentation 改变 pass 匹配结果或 checkpoint 之外的 materialization 语义。

## 21. 总结

`ModelGraph` 应是稳定的 semantic use-def DAG：

- `GraphNode` 是语义算子；
- `GraphValue` 是数据流边；
- input / activation / weight / constant / state / resource 都统一为 value；
- shape、quantization、weight binding、state binding 属于语义图；
- workspace、device、ISA、packed weight、kernel selector、distributed placement 属于 lowering / planning / backend；
- `ExecutionPlan` 是某次 backend / execution mode / runtime context 下从 `ModelGraph` 派生出的可执行计划。

这样可以支撑 CPU Llama 顺序执行、fusion、多后端、KV cache、continuous batching、服务化和分布式推理。
