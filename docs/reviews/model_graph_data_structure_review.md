# ModelGraph 数据结构设计全面审核报告

## 一、审核范围

本报告审核 `docs/designs/model_graph_design.md` 中定义的核心数据结构及其与现有代码库的兼容性。审核基于以下已验证的源文件：

- `include/aethermind/execution/execution_plan_builder.h` — lowering 的输出目标（ExecutionPlanNodeSpec）
- `include/aethermind/execution/execution_plan.h` — 最终执行节点（ExecutionStep）
- `include/aethermind/model/model_instance.h` — 权重查找接口
- `include/aethermind/operators/op_type.h` — 算子类型枚举
- `include/aethermind/shape_inference/shape_symbol.h` — 形状表达
- `include/aethermind/backend/kernel_selector.h` — kernel 选择器
- `include/aethermind/runtime/workspace.h` — workspace 需求
- `include/aethermind/model/formats/hf/hf_model_config.h` — 模型配置
- `include/aethermind/model/resolved_model_weights.h` — 解析后权重

---

## 二、数据结构合理性评估

### 2.1 ModelGraphNode ↔ ExecutionPlanNodeSpec 字段映射

| ModelGraphNode 字段 | ExecutionPlanNodeSpec 字段 | 映射方式 | 评估 |
|---------------------|---------------------------|---------|------|
| `op_type` | `op_type` | 直接复制 | ✅ |
| `inputs` (vector\<TensorSpec\>) | `input_specs` (vector\<TensorSpec\>) | 直接复制 | ✅ 复用同一类型 |
| `attrs` (ModelGraphAttrs) | `attrs` (vector\<byte\>) | 复制 bytes | ✅ owned storage |
| `op_params` (std::any) | `op_params` (std::any) | 移动/复制 | ✅ |
| `workspace_requirement` | `workspace_requirement` | 直接复制 | ⚠️ 见 P1-4 |
| `outputs` (vector\<TensorSpec\>) | **不存在** | 丢弃 | ⚠️ 见 P1-2 |
| `weights` (vector\<ModelWeightBinding\>) | **不存在** | lowering 解析为 packed_weights | ✅ |
| `layer_index` (uint32_t) | **不存在** | 用于权重解析，不传入 spec | ✅ |
| — | `device_type` | 来自 GraphLoweringOptions | ✅ |
| — | `activation_dtype` | 来自 GraphLoweringOptions | ✅ |
| — | `weight_dtype` | 来自 GraphLoweringOptions | ✅ |
| — | `weight_format` | 来自 GraphLoweringOptions | ✅ |
| — | `isa` | 来自 GraphLoweringOptions | ✅ |
| — | `phase` | 来自 GraphLoweringOptions | ✅ |

**结论**：字段映射基本完整，6 个 KernelSelector 字段正确地从 GraphLoweringOptions 填充而非存储在 ModelGraphNode 中。

### 2.2 ModelWeightBinding 解析可行性

`(role, layer_index)` → `RawWeightView` 的映射逻辑：

| ModelWeightRole | ResolvedModelWeights 路径 |
|----------------|-------------------------|
| `kTokenEmbedding` | `.embed_tokens` |
| `kInputNorm` + layer=i | `.layers[i].norm.input_rmsnorm` |
| `kPostAttentionNorm` + layer=i | `.layers[i].norm.post_attn_rmsnorm` |
| `kFinalNorm` | `.final_norm` |
| `kAttentionQ` + layer=i | `.layers[i].attn.q_proj` |
| `kAttentionK` + layer=i | `.layers[i].attn.k_proj` |
| `kAttentionV` + layer=i | `.layers[i].attn.v_proj` |
| `kAttentionO` + layer=i | `.layers[i].attn.o_proj` |
| `kMlpGate` + layer=i | `.layers[i].mlp.gate_proj` |
| `kMlpUp` + layer=i | `.layers[i].mlp.up_proj` |
| `kMlpDown` + layer=i | `.layers[i].mlp.down_proj` |
| `kLmHead` | `.lm_head.value()` |

映射关系明确且完整，`ResolvedModelWeights` 的结构天然支持 `(role, layer_index)` 查找。

### 2.3 GraphLoweringOptions → KernelSelector 映射

| GraphLoweringOptions 字段 | KernelSelector 字段 | 映射 |
|--------------------------|-------------------|------|
| `device_type` | `device_type` | 直接 |
| `activation_dtype` | `activation_dtype` | 直接 |
| `weight_dtype` | `weight_dtype` | 直接 |
| `preferred_weight_format` | `weight_format` | ⚠️ "preferred" 语义未定义 |
| `isa` | `isa` | 直接 |
| `phase` | `phase` | 直接 |

---

## 三、设计亮点

### 1. ModelGraphNode 不含 KernelSelector

§7.5 明确声明 "不包含 KernelSelector，因为 selector 中的 device、ISA、phase 是 lowering 时结合 runtime/backend 决策产生的"。这是正确的关注点分离——同一个 ModelGraph 可以搭配不同的 GraphLoweringOptions 生成不同后端的 spec，无需重建 graph。

### 2. ModelWeightBinding 使用查找键而非裸指针

§7.3 的 `(role, layer_index)` 设计消除了悬垂指针风险，且与 `ResolvedModelWeights` 的层级结构天然对齐。

### 3. 允许包含未实现 OpType 的节点

§9 明确声明 "graph builder 的职责是完整表达模型拓扑与静态元数据，不以当前 execution backend 的可执行覆盖率裁剪拓扑"。这避免了"先有鸡还是先有蛋"的问题——graph 可以先稳定，算子逐步补齐。

### 4. GraphLoweringShapeContext 分离请求级 shape

§8.3 将 `seq_len` 等请求级参数从 ModelGraph 和 GraphLoweringOptions 中分离，确保同一个 graph 可以复用于不同请求。

### 5. attrs 与 op_params 互斥语义

§7.5 和 §11 明确 "registered operator 节点通过 op_params 传参，attrs 只用于 raw kernel fallback"，避免了两条路径的参数冲突。

---

## 四、问题缺陷

### 🔴 P1-1：无显式数据流边 — 残差连接的 input_specs 填充依赖隐式拓扑知识

§9.2 原则 4 声明 "残差连接通过 lowering 的 input_specs 表达"，但 **lowering 如何知道 kAdd 的第二个输入来自哪个前驱节点的输出？**

当前 `ModelGraphNode.inputs` 只存储 `TensorSpec`（shape 信息），不存储数据流来源。在纯顺序拓扑中，每个节点的输入隐式来自前一个节点的输出，但 **残差连接打破了这一假设**：

```
Linear(O) → Add(residual)    ← Add 的第二个输入来自哪里？
```

对于 Llama 的固定拓扑，lowering 可以硬编码 "第 10 个节点（Add）的第二个输入来自第 1 个节点（RMSNorm）的输出"。但这意味着 **lowering 逻辑与拓扑结构强耦合**，任何拓扑变更都需要同步修改 lowering。

**影响**：Phase 1 可行（固定拓扑），但违反了 "graph 与 lowering 解耦" 的设计原则。

**改进方案**：在 `ModelGraphNode.inputs` 中引入来源索引：

```cpp
struct InputSource {
    size_t node_index = 0;    // 来源节点的数组下标
    size_t output_index = 0;  // 来源节点的第几个输出
};

struct ModelGraphNode {
    // ...
    std::vector<TensorSpec> inputs{};
    std::vector<InputSource> input_sources{};  // 与 inputs 一一对应
};
```

这样 lowering 只需读取 `input_sources` 即可确定数据流，无需硬编码拓扑知识。对于 Phase 1 的顺序拓扑，`input_sources[i] = {node_index - 1, 0}`（前一个节点的第一个输出），残差节点例外。

### 🔴 P1-2：ModelGraphNode.outputs 在 lowering 后被丢弃，形状一致性无法验证

`ModelGraphNode.outputs` 携带 graph builder 推导的输出形状，但 `ExecutionPlanNodeSpec` 没有 outputs 字段。lowering 时 outputs 被丢弃，`ExecutionPlanBuilder` 通过 `Operator::InferOutputShapes` 重新推导输出形状。

**风险**：如果 graph builder 的形状推导与 operator 的形状推导不一致（例如 graph 认为 output 是 `[seq_len, hidden_size]`，但 operator 推导为 `[seq_len, hidden_size + 1]`），**没有任何机制检测这一不一致**。后续节点的 input_specs 基于错误的形状，可能导致运行时形状约束违反或内存越界。

**改进方案**：在 lowering 时将 graph 的 output shapes 与 operator 的 InferOutputShapes 结果做交叉验证：

```cpp
// lowering 伪代码
for (const auto& node : graph.GetNodes()) {
    auto spec = MakeSpec(node, options, shape_context);
    // 交叉验证：graph 推导的输出形状 vs operator 推导的输出形状
    if (node.outputs != inferred_outputs) {
        return Status::Internal("Shape mismatch between graph and operator inference");
    }
}
```

或者更轻量的方案：在 `ExecutionPlanNodeSpec` 中添加 `expected_output_specs`，`ExecutionPlanBuilder` 在 `InferOutputShapes` 后与之比较。

### 🟡 P1-3：ModelWeightBinding 解析逻辑未指定归属

§8.4 lowering 职责第 4 条声明 "根据 ModelWeightBinding 的 (role, layer_index) 从 ModelInstance::GetResolvedWeights() 解析实际权重引用"，但 **解析函数的归属未明确**：

- 是 `GraphToExecutionPlanLowering` 的内部实现细节？
- 还是 `ModelInstance` 或 `ResolvedModelWeights` 应提供公共 API？

当前 `ModelInstance` 只提供 `FindPackedWeights(op_type, selector)` 和 `GetResolvedWeights()`，没有 `(role, layer_index) → RawWeightView` 的查找接口。如果解析逻辑放在 lowering 中，每个 lowering 实现都要重复编写映射表。

**改进方案**：在 `ResolvedModelWeights` 或 `ModelInstance` 中提供查找接口：

```cpp
// resolved_model_weights.h
AM_NODISCARD const RawWeightView* FindWeight(ModelWeightRole role, uint32_t layer_index) const noexcept;
```

这样 lowering 只需调用 `model_instance.GetResolvedWeights().FindWeight(binding.role, binding.layer_index)`。

### 🟡 P1-4：ModelGraphNode.workspace_requirement 在 graph 构建时无法准确填充

`ModelGraphNode.workspace_requirement` 需要 kernel 级别的信息（如中间缓冲区大小），但 graph 构建时 **尚未 resolve kernel**。当前只有 `RmsNormOp` override 了 `ComputeWorkspaceRequirement`（返回零），但未来 `kAttention` 等算子可能需要显著的 workspace。

**影响**：graph builder 填充的 workspace_requirement 可能不准确，lowering 或 ExecutionPlanBuilder 需要覆盖。

**改进方案**：将 `workspace_requirement` 从 `ModelGraphNode` 中移除，改为在 lowering 或 ExecutionPlanBuilder 阶段由 `Operator::ComputeWorkspaceRequirement` 动态计算。Graph 层只负责拓扑和形状，workspace 规划属于 execution 层。

### 🟠 P2-1：GraphLoweringOptions.preferred_weight_format 的 "preferred" 语义未定义

`preferred_weight_format` 中的 "preferred" 暗示这是一个建议而非强制要求——如果请求 packed format 但 packed weights 不存在，是否应该 fallback 到 plain？当前设计没有描述 fallback 行为。

**改进方案**：将 `preferred_weight_format` 改为 `weight_format`（与 `KernelSelector` 字段名一致），并在 lowering 中明确：如果请求 packed 但 `ModelInstance::FindPackedWeights` 返回 nullptr，则降级为 plain 并在 spec 中反映。

### 🟠 P2-2：attrs 与 op_params 互斥语义未类型强制

§7.5 和 §11 声明 "registered operator 节点通过 op_params 传参，attrs 只用于 raw kernel fallback"，但 `ModelGraphNode` 的类型定义允许两者同时非空。如果 builder 错误地同时填充了 attrs 和 op_params，lowering 和 ExecutionPlanBuilder 的行为未定义。

**改进方案**：添加运行时校验（在 lowering 中）：

```cpp
if (!node.attrs.bytes.empty() && node.op_params.has_value()) {
    return Status::InvalidArgument(
        "ModelGraphNode cannot have both attrs and op_params");
}
```

### 🟠 P2-3：ModelGraph 的不可变性未类型强制

§7.6 声明 "构建后应视为 immutable"，但 `nodes_` 是 private 可变成员。`ModelGraphBuilder::BuildLlamaDense` 返回 `StatusOr<ModelGraph>`（值类型），调用者拿到的是全新对象，无法通过旧引用修改。但 `ModelGraph` 本身没有 `const` 保护。

**当前实际风险**：低。`BuildLlamaDense` 返回值语义，调用者可以立即存入 `const ModelGraph`。但如果 `ModelInstance` 持有 `ModelGraph` 成员，需要确保不暴露 mutable 访问。

**改进方案**：`ModelGraph` 的所有修改方法（如 `AddNode`）设为 private + `friend ModelGraphBuilder`，公开接口全部为 const。

### 🟠 P2-4：GraphLoweringShapeContext 只支持单一输入

§8.3 的 `GraphLoweringShapeContext` 只包含 `token_ids` 一个字段。对于 Llama 模型，这是足够的（唯一的外部输入是 token IDs）。但如果未来需要 KV cache 的 shape 信息（如 `num_kv_slots`、`kv_cache_shape`），当前结构无法扩展。

**改进方案**：Phase 1 保持当前设计，但将 `GraphLoweringShapeContext` 设计为可扩展结构（如添加 `std::unordered_map<std::string, TensorSpec> extra_inputs` 字段），或明确声明 "Phase 2 引入 KV cache 时需扩展此结构"。

### 🟢 P3-1：ModelGraph 缺少调试/诊断接口

当前 `ModelGraph` 只有 `GetNodes()` 和 `GetConfig()` 两个查询接口，缺少：
- `DebugDump()` — 打印完整 graph 结构
- `NodeCount()` — 快速获取节点数
- `FindNodesByType(OpType)` — 按类型查找节点

这些对调试和测试有价值，但不影响核心功能。

### 🟢 P3-2：ModelWeightRole 枚举缺少 kBias 角色

当前 `ModelWeightRole` 只覆盖了权重角色，没有 `kBias`（线性层偏置）。`HfModelConfig::attention_bias` 和 `mlp_bias` 字段表明部分模型使用偏置，但 `ModelWeightRole` 未预留。

**影响**：Phase 1 Llama 模型无偏置，不影响。但 `HfModelConfig` 已预留了 bias 字段，建议在 `ModelWeightRole` 中同步预留。

---

## 五、数据流转逻辑正确性验证

### 5.1 端到端数据流追踪

以 "Input RMSNorm, layer=0" 为例，追踪从 graph 构建到 kernel 执行的完整数据流：

```
1. ModelGraphBuilder::BuildLlamaDense
   → ModelGraphNode{
       op_type = kRmsNorm,
       layer_index = 0,
       inputs = [TensorSpec{float32, [S, H]}, TensorSpec{float32, [H]}],
       outputs = [TensorSpec{float32, [S, H]}],
       weights = [ModelWeightBinding{kInputNorm, 0}],
       attrs = {},
       op_params = RmsNormOp::Params{eps = 1e-6},
       workspace_requirement = {},
     }

2. GraphToExecutionPlanLowering::Lower
   → ExecutionPlanNodeSpec{
       op_type = kRmsNorm,
       device_type = kCPU,           // from options
       activation_dtype = float32,   // from options
       weight_dtype = float32,       // from options
       weight_format = kPlain,       // from options
       isa = kAVX2,                  // from options
       phase = kBoth,                // from options
       workspace_requirement = {},
       input_specs = [TensorSpec{float32, [S, H]}, TensorSpec{float32, [H]}],
       attrs = {},                   // empty for registered operator
       op_params = RmsNormOp::Params{eps = 1e-6},
     }
   + packed_weights = model_instance.FindPackedWeights(kRmsNorm, selector)

3. ExecutionPlanBuilder::Build
   → OperatorRegistry::Create(kRmsNorm, op_params) → RmsNormOp
   → op->ValidateParams() → Ok
   → op->CheckInputSpecs(input_specs) → Ok
   → op->InferOutputShapes(input_specs) → InferenceResult{outputs, runtime_checks}
   → op->Prepare(op_ctx) → resolve kernel → CpuRmsNormKernelEntry_FP32_AVX2
   → ExecutionStep{selector, op, packed_weights, workspace_req, output_specs, runtime_checks}

4. LayerRunner::RunStep
   → ValidateShapeConstraints(runtime_checks, inputs, outputs) → Ok
   → op->Run(ctx, bindings, step_index)
     → ctx.kernel_params = &params
     → resolved_kernel_.fn(ctx) → CpuRmsNormKernelEntry_FP32_AVX2(ctx)
```

**验证结果**：数据流完整，无断裂点。关键观察：
- `op_params` 从 graph → spec → OperatorRegistry → RmsNormOp 正确传递
- `attrs` 在 registered operator 路径为空，正确
- `packed_weights` 通过 `ModelInstance::FindPackedWeights` 独立解析，不经过 spec
- `input_specs` 中的符号维度 `S`（seq_len）在运行时通过 `RuntimeBindingContext` 绑定具体张量

### 5.2 权重解析路径追踪

以 "Q Projection, layer=3" 为例：

```
1. ModelGraphNode.weights = [ModelWeightBinding{kAttentionQ, 3}]

2. Lowering 解析:
   model_instance.GetResolvedWeights().layers[3].attn.q_proj → RawWeightView

3. Lowering 查找 packed weights:
   model_instance.FindPackedWeights(kLinear, selector) → PackedWeights*

4. ExecutionStep.packed_weights = packed_weights_ptr
```

**验证结果**：路径正确。但步骤 2 的解析函数尚未在 `ResolvedModelWeights` 中定义（见 P1-3）。

---

## 六、异常处理机制评估

| 场景 | 处理方式 | 评估 |
|------|---------|------|
| Config 字段非法 | `Status::kInvalidArgument` | ✅ |
| 缺少必需权重 | `Status::kNotFound` | ✅ |
| 权重 shape 不匹配 | `Status::kInvalidArgument` | ✅ |
| 不支持的模型族 | `Status::kFailedPrecondition` | ✅ |
| Packed weights 缺失 | `Status::kNotFound` | ✅ |
| 内存分配失败 | `std::bad_alloc`（不捕获） | ✅ 如实反映 |
| attrs + op_params 同时非空 | **未处理** | ⚠️ 见 P2-2 |
| Graph 输出形状与 Operator 推导不一致 | **未检测** | ⚠️ 见 P1-2 |
| 未实现 OpType 的节点 | lowering 生成 spec，ExecutionPlanBuilder 返回错误 | ✅ |

---

## 七、性能影响评估

| 操作 | 开销 | 说明 |
|------|------|------|
| ModelGraph 构建 | O(N_nodes) | 516 节点（Llama-2-7B），< 1ms |
| Lowering | O(N_nodes) | 纯数据转换 + 权重查找，< 1ms |
| attrs 复制 | O(total_attrs_bytes) | 每个 RMSNorm 节点 4 字节（epsilon），总量 < 1KB |
| op_params 复制 | O(N_nodes × sizeof(Params)) | 每个 Params < 64 字节，总量 < 33KB |
| 执行期 | **零开销** | ModelGraph 不参与执行路径 |

**内存占用**：`ModelGraph` 自身 ~516 × ~200B = ~103KB，加上 `HfModelConfig` 拷贝 ~1KB，总计 ~104KB。可忽略。

---

## 八、与系统其他模块的兼容性

| 模块 | 兼容性 | 说明 |
|------|--------|------|
| ModelLoader | ✅ | 只调用 builder，不内联拓扑逻辑 |
| ModelInstance | ✅ | 可选择性持有 ModelGraph |
| ExecutionPlanBuilder | ✅ | 消费 lowering 输出的 specs，无需修改 |
| OperatorRegistry | ✅ | op_params 透传 |
| KernelRegistry | ✅ | 不直接交互，通过 ExecutionPlanBuilder 间接使用 |
| LayerRunner / Executor | ✅ | 完全不感知 ModelGraph |
| Shape Constraint 系统 | ✅ | runtime_checks 由 Operator 生成，不经过 graph |

---

## 九、改进建议汇总

| 优先级 | 问题 | 建议 | 影响 |
|--------|------|------|------|
| 🔴 P1-1 | 无显式数据流边 | 在 ModelGraphNode 中添加 `input_sources` 字段 | 解耦 lowering 与拓扑 |
| 🔴 P1-2 | 输出形状一致性无法验证 | 在 lowering 或 ExecutionPlanBuilder 中添加交叉验证 | 防止形状不一致导致的运行时错误 |
| 🟡 P1-3 | 权重解析逻辑归属不明 | 在 `ResolvedModelWeights` 中提供 `FindWeight(role, layer_index)` API | 避免重复实现映射表 |
| 🟡 P1-4 | workspace_requirement 在 graph 层无法准确填充 | 从 ModelGraphNode 中移除，改由 lowering/ExecutionPlanBuilder 动态计算 | 消除不准确数据 |
| 🟠 P2-1 | preferred_weight_format 语义模糊 | 改为 `weight_format`，lowering 中实现 fallback 逻辑 | 语义清晰 |
| 🟠 P2-2 | attrs 与 op_params 互斥未强制 | lowering 中添加运行时校验 | 防止误用 |
| 🟠 P2-3 | ModelGraph 不可变性未类型强制 | 修改方法设为 private + friend builder | 防止误修改 |
| 🟠 P2-4 | GraphLoweringShapeContext 扩展性 | 文档中声明 Phase 2 扩展计划 | 前瞻性 |
| 🟢 P3-1 | 缺少调试接口 | 添加 DebugDump/NodeCount/FindNodesByType | 开发体验 |
| 🟢 P3-2 | ModelWeightRole 缺少 kBias | 预留 kBias 角色 | 与 HfModelConfig 对齐 |

---

## 十、总结

ModelGraph 数据结构设计整体合理，核心决策——**ModelGraphNode 不含 KernelSelector**、**ModelWeightBinding 使用查找键**、**允许未实现 OpType**、**GraphLoweringShapeContext 分离请求级 shape**——都是正确的架构选择。与现有代码库的兼容性良好，字段映射完整，端到端数据流验证通过。

最关键的缺陷是 **无显式数据流边**（P1-1），这导致 lowering 必须硬编码拓扑知识来处理残差连接，违反了 graph 与 lowering 解耦的设计原则。建议通过 `input_sources` 字段解决，实现成本极低（Phase 1 顺序拓扑的 input_sources 可由 builder 自动推导），但收益显著——lowering 逻辑与拓扑完全解耦。

次关键的缺陷是 **输出形状一致性无法验证**（P1-2），这可能导致 graph 推导与 operator 推导的形状不一致在运行时才暴露。建议在 lowering 中添加交叉验证，或至少在 `ExecutionPlanNodeSpec` 中携带 expected output specs 供 ExecutionPlanBuilder 校验。

其余 P2/P3 问题属于设计改进，建议在实施 Milestone 1 时一并处理。
