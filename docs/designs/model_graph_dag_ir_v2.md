# ModelGraph 语义 DAG IR 设计方案

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
- 数据流 use-def：node inputs / outputs、producer / consumers；
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
    ModelWeightBinding binding{};
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
        ModelInputValue,
        ActivationValue,
        WeightValue,
        ConstantValue,
        StateValue,
        ResourceValue>;
```

语义说明：

- `ModelInputValue`：请求输入，例如 `token_ids`、position ids、attention mask；
- `ActivationValue`：普通中间激活；
- `WeightValue`：模型权重的逻辑引用，携带 `ModelWeightBinding`，不拥有真实权重 payload；
- `ConstantValue`：编译期常量，携带 `ConstantBinding`，例如标量、固定 mask、RoPE table 描述；
- `StateValue`：跨 step 持久状态，携带 `StateBinding`，例如 KV cache、decode state、streaming state；
- `ResourceValue`：外部 runtime resource，携带 `ResourceBinding`，例如 paged attention handle、外部分配 buffer handle。

KV cache 不应伪装成普通 activation。它的生命周期跨 token / request step，serving、paged attention、continuous batching 都会依赖显式 state 语义。

若需要序列化 / debug 输出，用 helper 将 variant index 映射为字符串或显式 enum；不要让序列化 tag 反向侵入核心类型设计。

## 6. GraphValue

```cpp
struct GraphValue {
    GraphValuePayload payload{ActivationValue{}};

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

    std::vector<GraphNodeId> consumers{};

    QuantizationSpec quantization{};

    std::string debug_name{};
};
```

`payload` 中的 binding 类型是逻辑引用，不是物理存储所有权：

- `ModelWeightBinding` 指向模型权重角色、layer、checkpoint key 或 weight id；
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

struct StateBinding {
    std::string name{};
};

struct ResourceBinding {
    std::string name{};
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
- `StateBinding` 是 KV cache、decode state、streaming state 的逻辑身份，不描述具体 cache layout；
- `ResourceBinding` 是 runtime resource 的逻辑身份，不持有设备句柄；
- `QuantizationSpec` 描述模型语义量化，例如 int4/int8、group size、scale dtype、zero point policy，不描述 packed weight 格式。

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

    // Compatibility path: std::any matches the OperatorRegistry params path.
    // Preferred direction: typed OpParams variant or schema-validated parameter object.
    std::any op_params{};

    // Extension metadata only. Registered operators should prefer op_params.
    ModelGraphAttrs attrs{};
};
```

`GraphNode` 不应包含：

```cpp
std::vector<ModelWeightBinding> weights{};
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

### 7.2 OperatorSchema 类型草案

`OperatorSchema` 描述 operator 的语义 ABI：输入输出端口顺序、端口名称、期望 payload 类型和该端口是否参与 tensor spec validation。它用于 graph validation 和 lowering，不表达 backend kernel 选择。

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

可以保留 `std::any op_params`，因为它兼容现有 operator registry 和各 operator 的 `Params` 类型。

建议收敛为 typed params：

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
- 避免 `std::any_cast` 失败只在运行期暴露；
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

    GraphValueId AddInput(TensorSpec spec, std::string name);

    GraphValueId AddWeight(
            TensorSpec spec,
            ModelWeightBinding binding,
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

    GraphNodeId AddNode(GraphNode node);

    GraphValueId AddActivation(
            TensorSpec spec,
            GraphNodeId producer,
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

`AddNode()` 应统一维护 input value 的 `consumers`。如果允许 graph rewrite pass 修改节点输入输出，必须提供受控 mutation API 来同步维护 producer / consumer 信息。

## 10. Graph Rewrite API

`ModelGraph` 默认是 immutable semantic graph。Graph pass 不应直接修改 `nodes_`、`values_`、`inputs_`、`outputs_`，而应通过明确的 rewrite API 表达变更，避免 use-def 信息失配。

推荐两种 rewrite 模式。

### 10.1 Immutable rewrite 模式

这是默认模式。Pass 读取 `ModelGraph`，输出一组 mutation patch，由 `ModelGraphRewriter` 统一应用并生成新的 `ModelGraph`。

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

class ModelGraphRewriter {
public:
    static StatusOr<ModelGraph> Apply(
            const ModelGraph& graph,
            std::span<const GraphMutation> mutations);
};
```

`ModelGraphRewriter::Apply()` 负责：

1. 校验 mutation 引用的 node / value 是否存在；
2. 检查 mutation 之间是否冲突；
3. 重建或修补 producer / consumers；
4. 更新 graph inputs / outputs；
5. 应用后运行 `Validate()`。

这种模式的优点是 pass 易测试、graph 不会暴露中间非法状态、use-def 维护集中在 rewriter 内部。

### 10.2 Controlled mutable rewrite 模式

如果某些 pass 需要原地修改 graph，应通过受控 mutable API 完成，而不是暴露底层 `std::vector` 引用。

```cpp
class MutableModelGraph {
public:
    Status ReplaceNode(GraphNodeId node, GraphNode replacement);
    Status RemoveNode(GraphNodeId node);
    Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    Status ReplaceAllUses(GraphValueId old_value, GraphValueId new_value);

    Status ValidateUseDef() const;
};
```

这些 API 必须自动维护：

- `GraphValue::producer`；
- `GraphValue::consumers`；
- node inputs / outputs；
- graph inputs / outputs；
- dangling value / orphan node 检查。

除非有明确的复杂 rewrite 或性能需求，否则应优先使用 immutable rewrite 模式。

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

RoPE(v_q, v_k, state_position) → v_q_rope, v_k_rope

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
RoPE(v_q, v_k, state_position) → v_q_rope, v_k_rope
KVCacheUpdate(v_k_rope, v_v, state_kv_cache_in) → state_kv_cache_out
Attention(v_q_rope, state_kv_cache_out, mask_or_metadata) → v_attn
```

这使 prefill / decode / paged attention 的状态语义在 graph 中可见，但具体 cache layout、page table、device buffer 仍留给 lowering / runtime。

## 13. Validation 规则

`Validate()` 至少检查：

1. 所有 `GraphNodeId` / `GraphValueId` 有效；
2. 每个 node input / output 均引用有效 value；
3. 每个 `ActivationValue` 必须有 producer；
4. `ModelInputValue` / `WeightValue` / `ConstantValue` 默认不允许有 producer；
5. `StateValue` / `ResourceValue` 允许无 producer 的外部入口，也允许由节点产生更新后的 state/resource；
6. 每个 node output 对应 value 的 producer 必须是该 node；
7. 每个 node input 对应 value 的 consumers 必须包含该 node；
8. node 输入数量、输出数量、schema port 顺序、payload 类型必须符合 operator schema；
9. 输入输出 `TensorSpec` dtype / shape / rank 兼容；
10. dynamic shape 符号和约束一致；
11. 对普通 activation 数据流，DAG 必须无环；
12. 对 state/resource token，如需表达跨 step 状态更新，应通过显式 state input/output 建模，不能制造隐式环。

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

Semantic pass 默认采用 immutable rewrite 模式：读取 `ModelGraph`，输出 `GraphMutation` 列表，并由 `ModelGraphRewriter::Apply()` 生成新的 `ModelGraph`。受控 mutable rewrite API 仅用于复杂 pass 或性能敏感场景。

注意：semantic fusion rewrite 可以发生在 `ModelGraph`，但具体 fused kernel、workspace、launch config 仍属于 lowering / backend plan。

## 17. Lowering

DAG lowering 的职责：

1. 拓扑排序或生成合法执行 schedule；
2. 按 operator schema 顺序解释 `GraphNode.inputs` / `GraphNode.outputs`；
3. 解析 `WeightValue` 的逻辑权重引用；
4. 根据 backend / execution mode / device 选择 operator 或 kernel；
5. 生成 packed weight mapping；
6. 计算 workspace；
7. 插入必要的 runtime resource / communication / synchronization；
8. 生成 `ExecutionPlanNodeSpec`、`LoweredGraph` 或 `ExecutionPlan`。

Lowering 可以派生 execution-specific views，例如 runtime activation inputs、bound weight handles、state handles、resource handles。但这些派生视图必须保留原始 schema `port_index` 或 `port_name`，不能替代 `GraphNode.inputs` 本身。

`ExecutionPlanNodeSpec::input_specs` 用于 operator schema validation / shape inference，应按 schema port 顺序构造。对于声明为 tensor-like 的端口，weight / constant / state 也必须保留对应 `TensorSpec`，不能因为 payload 类型是 `WeightValue` 就过滤掉。

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

### Milestone 1：语义 DAG 骨架

1. 新增 `GraphNodeId`、`GraphValueId`、`GraphValuePayload`、`GraphValue`；
2. 将 `GraphNode.inputs` / `GraphNode.outputs` 改为 `GraphValueId`；
3. 将 weight 从 `GraphNode.weights` 迁移为 `WeightValue` payload；
4. 从 `GraphNode` 移除 `WorkspaceRequirement`；
5. 为 `ModelGraph` 增加 values / inputs / outputs；
6. 添加 producer / consumer validation。

### Milestone 2：Llama GraphBuilder

1. 重写 `BuildLlamaDense()`，显式创建 activation / weight / constant / state value；
2. 显式表达 residual、RoPE 多输出、KV cache state；
3. 添加 topology、shape、payload / binding 单元测试。

### Milestone 3：Lowering Bridge

1. 添加 Graph → `ExecutionPlanNodeSpec` 转换；
2. 保持 `ExecutionPlanBuilder` 接口尽量稳定；
3. 在 lowering 中解析 weight / constant / state；
4. workspace 继续由 operator / planner 计算。

### Milestone 4：参数与工具完善

1. 将 `std::any op_params` 收敛为 typed `OpParams`；
2. 增加 `QuantizationSpec`；
3. 增加 `StateBinding` 支撑 KV cache；
4. 增加 `GraphMutation` 与 `ModelGraphRewriter`；
5. 增加 semantic pass pipeline；
6. 增加 graph dump / serialization / validation tooling。

## 20. 设计红线

1. 不把 backend / device / ISA 放进 `ModelGraph`；
2. 不把 workspace / memory offset 放进 `GraphNode`；
3. 不把 packed weight 当成 `GraphValue` payload；
4. 不把 `attrs` 当成无类型参数垃圾桶；
5. 不依赖 `decoder_layer_index` 表达数据依赖；
6. 不在 graph builder 中根据已有 kernel 覆盖率裁剪拓扑；
7. 不因起步范围较小而过早实现完整 compiler framework。

## 21. 总结

`ModelGraph` 应是稳定的 semantic use-def DAG：

- `GraphNode` 是语义算子；
- `GraphValue` 是数据流边；
- input / activation / weight / constant / state / resource 都统一为 value；
- shape、quantization、weight binding、state binding 属于语义图；
- workspace、device、ISA、packed weight、kernel selector、distributed placement 属于 lowering / planning / backend；
- `ExecutionPlan` 是某次 backend / execution mode / runtime context 下从 `ModelGraph` 派生出的可执行计划。

这样可以支撑 CPU Llama 顺序执行、fusion、多后端、KV cache、continuous batching、服务化和分布式推理。
