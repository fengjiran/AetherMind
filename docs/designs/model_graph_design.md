# ModelGraph 模块设计方案

## 1. 背景

当前 `ModelLoader` / `ModelInstance` 已能承载 HF 配置、解析后的权重视图以及后端侧的 packed weights；`ExecutionPlanBuilder` 则消费 `std::vector<ExecutionPlanNodeSpec>` 并生成 `ExecutionPlan`。两者之间仍缺少一个生产级模块来回答：一个已加载的 Llama-family 模型应当按什么拓扑、什么 shape、什么算子参数和什么权重绑定生成执行计划节点。

目前 `ExecutionPlanNodeSpec` 的生产方主要存在于单元测试中，测试手工填充 `op_type`、dtype、selector、attrs 和 `op_params`。这适合作为 builder 的隔离测试，但不能作为真实模型加载后的执行路径。`execution_plan_builder.cpp` 中也已有 TODO，指出真实 input shape 应来自 model graph；`dispatch_design.md` 也要求 attrs 生命周期由 plan/model graph 持有。

因此需要一个轻量的 `ModelGraph` 模块承接 `ModelInstance` 的模型语义，并进一步 lowering 为 `ExecutionPlanNodeSpec`。

## 2. 设计目标

1. 为 Phase 1 dense Llama-family 模型提供生产级 `ExecutionPlanNodeSpec` 生成路径。
2. 将模型拓扑、shape 推导、权重绑定、operator attrs 生命周期从 `ModelLoader` 和 `ExecutionPlanBuilder` 中分离出来。
3. 保持 `ExecutionPlanBuilder` 的职责单一：消费完整 node specs、resolve kernel、构建 `ExecutionPlan`。
4. 保持 `ModelLoader` 的职责单一：读取模型文件、解析配置、resolve 权重、构建 `ModelInstance`。
5. 避免引入 Phase 1 不需要的完整 graph compiler、fusion pass、cost model 或通用动态图系统。

## 3. 非目标

Phase 1 的 `ModelGraph` 不承担以下职责；这些不是永久非目标，而是明确推迟到后续阶段的演进能力：

- 通用 DAG 图优化框架；
- operator fusion / rewrite pass；
- 多后端图分区；
- 动态 shape 推导框架；
- MoE、Speculative Decoding、PagedAttention、Multi-LoRA；
- 训练图、反向图或自动微分；
- runtime hot path 中的 kernel lookup 或动态分派。

这些能力应留给后续阶段，只有在出现跨 op fusion、多模型族拓扑差异、异构设备切分或 runtime-dependent shape planning 时再扩展。Phase 1 的设计必须为这些能力预留边界，但不能提前实现完整框架。

## 4. 模块定位

推荐的数据流如下：

```text
ModelLoader::Load()
    ↓
ModelInstance
  - HfModelConfig
  - ResolvedModelWeights
  - BackendSidecar / PackedWeights
    ↓
ModelGraphBuilder
    ↓
ModelGraph
  - Llama 拓扑
  - Tensor shape 元数据
  - weight handle / weight role
  - per-op attrs
    ↓
GraphToExecutionPlanLowering
    ↓
std::vector<ExecutionPlanNodeSpec>
    ↓
ExecutionPlanBuilder::Build(runtime, model_instance, specs)
    ↓
ExecutionPlan
```

`ModelGraph` 是 model 层和 execution 层之间的边界类型。它表达“模型是什么”和“每个节点需要哪些静态元数据”，但不表达“用哪个 kernel 执行”。kernel resolution 仍属于 `ExecutionPlanBuilder` / backend。

## 5. 职责边界

| 模块 | 负责 | 不负责 |
|------|------|--------|
| `ModelLoader` | 读取 HF 目录、解析 config、加载 safetensors、resolve 权重名到 `RawWeightView`、构建 `ModelInstance`、触发 prepack planner | 推导执行拓扑、构建 execution specs、选择 kernel |
| `ModelInstance` | 持有 config、resolved weights、backend sidecar，并作为 graph/build 阶段的生命周期根对象 | 实现 lowering 细节或执行逻辑 |
| `ModelGraphBuilder` | 根据 `HfModelConfig` 和 `ResolvedModelWeights` 构建 Llama-family graph | kernel resolve、workspace planning、执行 |
| `ModelGraph` | 持有不可变拓扑、shape、weight binding、attrs storage | 修改权重、管理 backend、执行 kernel |
| `GraphToExecutionPlanLowering` | 将 `ModelGraph` + backend/runtime 选择信息转换为 `ExecutionPlanNodeSpec` | 加载模型、resolve kernel、执行 plan |
| `ExecutionPlanBuilder` | 验证 specs、创建/prepare operator、resolve kernel、规划 workspace、生成 immutable `ExecutionPlan` | 理解 HF config、展开 Llama 层拓扑 |
| `Executor` / `LayerRunner` | 执行已构建 plan | 修改 graph、resolve kernel、读取模型文件 |

## 6. 建议目录结构

```text
include/aethermind/model/graph/
  model_graph.h
  model_graph_builder.h
  graph_to_execution_plan_lowering.h

src/model/graph/
  model_graph.cpp
  model_graph_builder.cpp
  graph_to_execution_plan_lowering.cpp

tests/unit/model/graph/
  test_model_graph_builder.cpp
  test_graph_to_execution_plan_lowering.cpp
```

`ExecutionPlanNodeSpec` 当前定义在 `execution_plan_builder.h` 中。短期可以保持不变，避免扩大迁移范围；中期建议移动到 `include/aethermind/execution/execution_plan_spec.h`，使 lowering 模块可以依赖 execution spec 类型而不依赖 builder 类声明。

## 7. 核心数据结构草案

### 7.1 Graph 节点 ID

```cpp
struct ModelGraphNodeId {
    uint32_t value = 0;
};
```

节点 ID 只在 `ModelGraph` 内部稳定有效，用于 shape、weight binding 和调试定位。Phase 1 可以使用拓扑数组下标实现，不需要复杂 handle table。

### 7.2 Tensor 描述

```cpp
struct ModelTensorSpec {
    DataType dtype{};
    ShapeAndStride shape{};
};
```

`ModelTensorSpec` 描述 graph 编译期可知的 tensor metadata。它不是 runtime tensor，也不拥有数据。

### 7.3 Weight Binding

```cpp
enum class ModelWeightRole : uint16_t {
    kTokenEmbedding,
    kAttentionQ,
    kAttentionK,
    kAttentionV,
    kAttentionO,
    kMlpGate,
    kMlpUp,
    kMlpDown,
    kInputNorm,
    kPostAttentionNorm,
    kFinalNorm,
    kLmHead,
};

struct ModelWeightBinding {
    ModelWeightRole role{};
    uint32_t layer_index = 0;
    const RawWeightView* raw_weight = nullptr;
};
```

`ModelWeightBinding` 不拥有权重数据。权重数据由 `ModelInstance::GetResolvedWeights()` 持有，binding 只保存稳定引用或可解析 key。若后续需要跨 vector reallocation 的稳定性，应改为 weight key / index，而不是裸指针。

### 7.4 Operator Attrs

```cpp
struct ModelGraphAttrs {
    std::vector<std::byte> bytes{};
};
```

attrs 由 `ModelGraph` 持有，生命周期覆盖 lowering 阶段。lowering 到 `ExecutionPlanNodeSpec` 时，可以将 attrs 复制进 spec 或确保 spec/plan 持有独立副本。执行期不能依赖临时栈内 attrs。

### 7.5 Graph Node

```cpp
struct ModelGraphNode {
    OpType op_type = OpType::kUnknown;
    uint32_t layer_index = 0;
    std::vector<ModelTensorSpec> inputs{};
    std::vector<ModelTensorSpec> outputs{};
    std::vector<ModelWeightBinding> weights{};
    ModelGraphAttrs attrs{};
    std::any op_params{};
    WorkspaceRequirement workspace_requirement{};
};
```

`ModelGraphNode` 保存模型语义层面的节点信息。它不包含 `KernelSelector`，因为 selector 中的 device、ISA、phase 是 lowering 时结合 runtime/backend 决策产生的。

### 7.6 ModelGraph

```cpp
class ModelGraph {
public:
    AM_NODISCARD std::span<const ModelGraphNode> nodes() const noexcept;
    AM_NODISCARD const HfModelConfig& config() const noexcept;

private:
    HfModelConfig config_{};
    std::vector<ModelGraphNode> nodes_{};
};
```

Phase 1 的 `ModelGraph` 构建后应视为 immutable。这样可以让后续 plan build、debug dump 和可能的多 phase lowering 共享同一个 graph，而不引入同步复杂度。

## 8. 构建接口设计

### 8.1 ModelGraphBuilder

```cpp
class ModelGraphBuilder {
public:
    AM_NODISCARD static StatusOr<ModelGraph> BuildLlamaDense(
            const HfModelConfig& config,
            const ResolvedModelWeights& weights);
};
```

输入只来自 model 层已有产物。`BuildLlamaDense` 应验证：

- `hidden_size`、`intermediate_size`、`num_hidden_layers` 等关键配置有效；
- required weights 全部存在；
- layer 数与 resolved weights 数量一致；
- RMSNorm、RoPE、attention、MLP 所需 attrs 可从 config 推导；
- shape 推导结果与权重 shape 兼容。

### 8.2 Lowering Options

```cpp
struct GraphLoweringOptions {
    DeviceType device_type = DeviceType::kCPU;
    DataType activation_dtype{};
    DataType weight_dtype{};
    WeightFormat preferred_weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
};
```

`GraphLoweringOptions` 表达 execution/backend 相关选择。它不应进入 `ModelGraphBuilder`，否则 graph 构建会过早绑定到某个 backend。

### 8.3 GraphToExecutionPlanLowering

```cpp
class GraphToExecutionPlanLowering {
public:
    AM_NODISCARD static StatusOr<std::vector<ExecutionPlanNodeSpec>> Lower(
            const ModelGraph& graph,
            const GraphLoweringOptions& options,
            const ModelInstance& model_instance);
};
```

lowering 负责：

1. 为每个 graph node 生成 `ExecutionPlanNodeSpec`；
2. 填充 `op_type`、`device_type`、dtype、weight format、ISA、phase；
3. 将 `ModelGraph` 中的 attrs 复制或转移为 spec 可安全引用的数据；
4. 根据 `ModelInstance` 的 backend sidecar 判断 packed weights 是否可用；
5. 填充 `op_params`，例如 RMSNorm epsilon、Embedding vocab 信息等；
6. 填充 `workspace_requirement`。

## 9. Llama-family Phase 1 拓扑

Phase 1 可以将 Llama dense 拓扑展开为固定序列：

```text
Embedding
for layer in [0, num_hidden_layers):
    RMSNorm(input_norm)
    Attention / MatMul(QKV) / RoPE / MatMul(O)   // 具体拆分取决于已实现算子粒度
    Add(residual)
    RMSNorm(post_attention_norm)
    Linear(gate)
    Linear(up)
    SiluMul
    Linear(down)
    Add(residual)
RMSNorm(final_norm)
Linear(lm_head)
Argmax
```

由于当前算子体系仍在演进，`ModelGraphBuilder` 不应假设必须存在 fused attention 或 fused MLP。它应根据仓库中已实现的 `OpType` 粒度展开节点，并允许未来将多个 graph node lowering 到一个 fused execution node。

## 10. Shape 推导策略

Phase 1 最小 shape 推导只需要覆盖单请求、同步执行场景：

- token input: `[seq_len]` 或 `[batch=1, seq_len]`；
- embedding output: `[seq_len, hidden_size]`；
- hidden states: `[seq_len, hidden_size]`；
- attention projection weights: `[hidden_size, hidden_size]` 或模型文件约定 shape；
- MLP gate/up: `[intermediate_size, hidden_size]`；
- MLP down: `[hidden_size, intermediate_size]`；
- logits: `[seq_len, vocab_size]`；
- argmax output: `[seq_len]` 或最后 token id。

Prefill 与 Decode 的主要差异应通过 `ExecPhase` 和 runtime sequence metadata 进入 lowering，而不是让 `ExecutionPlanBuilder` 推导。对于 Phase 1，如果真实 runtime shape 尚未完全接入，可以先在 `ExecutionPlanNodeSpec` 中预留 shape 字段，并在 graph tests 中验证静态维度推导。

## 11. Attrs 与 op_params 生命周期

attrs / `op_params` 是当前架构最容易出现悬空引用的位置，必须明确规则：

1. `ModelGraph` 持有从 config 派生的 attrs 原始存储；
2. lowering 生成 `ExecutionPlanNodeSpec` 时，不允许让 spec 指向临时栈内对象；
3. 如果 `ExecutionPlanNodeSpec::attrs` 仍为 `std::span<const std::byte>`，则需要有一个与 specs 同生命周期的 owner；
4. 更推荐中期将 `ExecutionPlanNodeSpec::attrs` 改为 owned `std::vector<std::byte>`，与近期 `ResolvedKernel::attrs` ownership 修复保持一致；
5. `op_params` 应只承载小型、可复制、类型明确的 operator 参数对象；
6. 大块权重数据不应进入 attrs 或 `op_params`，应通过 `ModelInstance` / weight binding / packed sidecar 管理。

## 12. 与 packed weights 的关系

`ModelGraph` 只描述某个节点需要什么 weight role，不直接持有 packed weight storage。packed weights 属于 backend sidecar：

```text
ModelGraphNode.weights
    ↓ role/layer/op_type/selector
ModelInstance::FindPackedWeights(op_type, selector)
    ↓
ExecutionStep::packed_weights
```

这样可以保持两个约束：

- graph 不绑定具体 backend artifact；
- `ExecutionPlanBuilder` 仍通过 `ModelInstance` 查找 packed weights，维持现有 `packed_weights` 设计。

## 13. 错误处理

`ModelGraphBuilder` 和 lowering 都必须使用 `Status` / `StatusOr`，不抛异常作为常规错误路径。

建议错误分类：

| 场景 | StatusCode |
|------|------------|
| 缺少必需 config 字段或非法超参数 | `kInvalidArgument` |
| 缺少必需权重 | `kNotFound` |
| 权重 shape 与 config 不匹配 | `kInvalidArgument` |
| 请求 unsupported model family | `kFailedPrecondition` 或 `kInvalidArgument` |
| packed weight 被请求但 sidecar 缺失 | `kNotFound` |
| attrs / specs 分配失败 | `kResourceExhausted` |

## 14. 测试策略

### 14.1 Graph Builder Tests

- 给定最小 Llama config，生成正确节点数量；
- layer loop 按 `num_hidden_layers` 展开；
- 缺少 embedding / qkv / mlp / norm 权重时报错；
- RMSNorm epsilon、RoPE 参数等 attrs 来自 config；
- 权重 shape 与 config 不匹配时报错。

### 14.2 Lowering Tests

- lowering 后每个 `ExecutionPlanNodeSpec.op_type` 正确；
- selector 字段来自 `GraphLoweringOptions`；
- attrs / `op_params` 生命周期安全，不依赖临时对象；
- packed/plain weight format 分支正确；
- prefill/decode phase 能生成不同 selector。

### 14.3 Builder Integration Tests

- `ModelGraphBuilder -> Lower -> ExecutionPlanBuilder::Build` 可构建 plan；
- build 阶段无 runtime registry lookup 热路径残留；
- `ExecutionStep::packed_weights` 与 sidecar 中 packed storage 对应；
- 已有手工 `ExecutionPlanNodeSpec` 单元测试保留，继续作为 builder 的隔离测试。

## 15. 分阶段实施计划

### Milestone 1：最小数据结构与目录落地

1. 新增 `include/aethermind/model/graph/` 与 `src/model/graph/`；
2. 定义 `ModelGraphNode`、`ModelGraph`、`ModelGraphBuilder`；
3. 只支持 dense Llama config 的静态拓扑展开；
4. 添加 graph builder 单元测试。

### Milestone 2：Lowering 到 ExecutionPlanNodeSpec

1. 定义 `GraphLoweringOptions`；
2. 实现 `GraphToExecutionPlanLowering::Lower`；
3. 将 attrs / `op_params` 安全传入 specs；
4. 添加 lowering 单元测试。

### Milestone 3：接入 ModelInstance

1. `ModelInstance` 可选择性持有 immutable `ModelGraph`；
2. `ModelLoader` 完成加载后调用 `ModelGraphBuilder`；
3. 保持旧接口兼容，允许测试继续手工构造 specs。

### Milestone 4：接入端到端 plan build

1. 新增 `BuildExecutionPlanFromModelGraph` 或等价 helper；
2. 端到端路径变为 `ModelInstance -> ModelGraph -> specs -> ExecutionPlan`；
3. 清理 `execution_plan_builder.cpp` 中关于 shape plumbing 的 TODO。

## 16. 设计风险与缓解

| 风险 | 缓解 |
|------|------|
| 过早设计成通用 graph compiler | Phase 1 只支持 Llama dense 顺序拓扑，不实现 pass framework |
| attrs 生命周期再次悬空 | specs 使用 owned storage，或提供明确 owner 对象覆盖 plan build 生命周期 |
| ModelLoader 膨胀 | Loader 只调用 builder，不内联拓扑逻辑 |
| ExecutionPlanBuilder 混入模型语义 | Builder 只消费 specs，不读取 HF config |
| shape 体系不成熟 | 先覆盖静态维度与单请求场景，动态 runtime shape 后续补充 |
| packed weights 与 graph 双重所有权 | graph 只保存 role/key，storage 仍由 `ModelInstance` / sidecar 持有 |

## 17. 后续演进路线

Phase 1 的 `ModelGraph` 应保持轻量，但接口边界必须支持后续扩展。推荐按以下顺序演进，避免在当前阶段一次性引入完整 compiler 复杂度。

### 17.1 通用图 IR

触发条件：需要支持 Llama 以外的模型族，例如不同 decoder-only 架构、encoder-decoder、视觉语言模型或多模态分支。

演进方向：

1. 将当前顺序 `std::vector<ModelGraphNode>` 扩展为显式 edge-based IR；
2. 为 tensor value 引入稳定 ID，例如 `GraphValueId`；
3. 区分 node、value、weight、constant、runtime input、runtime output；
4. 将 Llama dense 拓扑从 `BuildLlamaDense` 迁移为一个 model-family producer；
5. 保持 `GraphToExecutionPlanLowering` 作为 IR 到 execution spec 的唯一出口。

Phase 1 不应提前实现通用 DAG，但 `ModelGraphNodeId`、weight role、attrs owner 等设计应避免阻碍未来迁移。

### 17.2 算子融合与图优化 pass

触发条件：Phase 2 需要减少 kernel launch / function call 次数，或需要融合 RMSNorm、Linear、Activation、Residual 等相邻算子以提升吞吐。

演进方向：

1. 引入只读 analysis pass，例如 shape analysis、layout analysis、alias analysis；
2. 再引入 rewrite pass，例如 `RmsNorm + Linear`、`Silu + Mul`、`Q/K/V projection` fusion；
3. pass 输出仍应是新的 immutable graph，而不是原地修改旧 graph；
4. lowering 根据最终 graph 生成 `ExecutionPlanNodeSpec`；
5. fused op 必须通过 `OpType` / operator registry / kernel registry 的正式路径接入，不能在 lowering 中硬编码函数指针。

Phase 1 只需要把 graph 和 lowering 的职责拆开，为后续插入 pass pipeline 预留位置。

### 17.3 动态 shape 推导框架

触发条件：支持 batch、变长序列、PagedAttention、prefix cache、continuous batching 或多请求调度。

演进方向：

1. 将当前 arithmetic shape 推导升级为 shape constraint system；
2. 区分 compile-time static dim、runtime bounded dim 和 fully dynamic dim；
3. 让 `ExecutionPlanNodeSpec` 携带足够的 input/output shape metadata，供 `Operator::CheckInputSpecs` 在 plan build 阶段执行；
4. 将 workspace requirement 从固定值升级为 shape-dependent requirement；
5. Decode hot path 不做复杂 shape 推导，只消费预先规划好的 bounded layout。

Phase 1 的 Llama 单请求路径可以继续使用直接算术推导，例如 `hidden_size`、`intermediate_size`、`num_attention_heads`、`head_dim`。

### 17.4 多后端图分区

触发条件：引入 GPU/CUDA backend、CPU+GPU 混合执行、不同层或不同 op 放置到不同设备。

演进方向：

1. 在 graph 层引入 device placement annotation；
2. 增加 backend capability analysis，判断每个 op 的合法 backend 集合；
3. 增加 partition pass，将 graph 切成 backend-specific subgraphs；
4. lowering 为每个 partition 生成对应 `ExecutionPlanNodeSpec`；
5. 明确跨设备 tensor transfer node 与同步边界；
6. `ExecutionPlanBuilder` 仍只消费已经带有 `DeviceType` / selector 的 specs，不负责做 partition 决策。

Phase 1 只有 CPU backend，因此 `GraphLoweringOptions.device_type` 固定为 CPU 即可，但保留字段是合理的。

### 17.5 演进约束

无论后续引入哪种能力，都应保持以下约束：

1. `ModelLoader` 不参与图优化；
2. `ExecutionPlanBuilder` 不读取 HF config、不展开模型拓扑；
3. `ModelGraph` / lowering 不执行 kernel resolve；
4. attrs、shape、weight binding 的生命周期必须清晰；
5. runtime hot path 不做 registry lookup、graph rewrite 或复杂 shape 推导。

## 18. 推荐结论

需要新增 `ModelGraph` 模块，但必须保持轻量。Phase 1 的正确抽象不是“完整图编译器”，而是“Llama 模型拓扑与静态元数据的不可变表示”。它的核心价值是补齐当前架构中 `ModelInstance` 到 `ExecutionPlanNodeSpec` 的生产级桥梁，同时保持 `ModelLoader`、`ExecutionPlanBuilder`、`Executor` 的职责边界清晰。

短期最优路径是：先实现 `ModelGraphBuilder::BuildLlamaDense` 和 `GraphToExecutionPlanLowering::Lower`，让它们产出当前 `ExecutionPlanBuilder` 已能消费的 specs。等 shape、attrs ownership 和端到端执行稳定后，再考虑是否将 `ExecutionPlanNodeSpec` 移出 builder header，并扩展到更丰富的 graph IR。
