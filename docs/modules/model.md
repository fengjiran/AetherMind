# model 模块设计

model 是模型前端与权重语义层：把 HuggingFace 模型目录转换为 `LoadedModel`，再按模型家族构建 backend-independent 的 `ModelGraph`；同时提供角色到 raw weight 的解析、打包请求/产物集合，以及显式 FP32 权重转换。分层与依赖方向见 [architecture.md](../architecture.md)，下游编译见 [compiler.md](compiler.md)，执行准备见 [inference.md](inference.md)。

## 1. 职责与边界

### 1.1 负责

| 职责 | 具体内容 |
|---|---|
| HF 目录 I/O | `HfDirectoryReader` 检测单文件或分片 safetensors 布局，解析 `config.json`，加载 raw tensor 视图 |
| 配置与权重校验 | `HfModelValidator` 三阶段校验：`ValidateConfig`、`ValidateWeightSet`、`ValidateResolvedModel` |
| 逻辑权重解析 | `hf::ResolveWeights` 按 HF tensor 名建立 `ResolvedModelWeights`；缺省 `lm_head` 在 tied 场景保持可选 |
| 加载产物 | `LoadedModel` 持有 config 与 resolved raw weights，构造后只读；`ModelLoader::Load` 返回 `unique_ptr<LoadedModel>` |
| 家族识别与图构建 | `ParseModelArchitecture` 是家族判定唯一权威；`BuildModelGraph` 是 HF → semantic graph 唯一 dispatch 入口，当前路由 `BuildLlamaDense` |
| RoPE 规范化 | `BuildLlamaDense` 通过共享件 `MakeRoPEParams` 把 HF 原始字段规范化为 typed `RoPEAlgorithmParams`，拒绝未知算法 |
| 角色绑定 | `ResolveWeightBinding` 把结构化 `WeightBinding` 解析为 `RawWeightView`，是 tied lm-head 回退的唯一权威 |
| 打包请求与产物 | 定义 `WeightPackingRequest`、`WeightArtifactKey`、`PackedWeightCollection`；`PrepackWeightRequests` 通过 `Backend::PackWeights` 执行批量打包 |
| 显式 FP32 转换 | `ConvertRawModelWeightsToFp32` 在构图前按显式策略生成自持 FP32 `RawStorage` 的新 `LoadedModel` |
| 存储核算 | `ComputeRawWeightStorageUsage` 按 owner identity 统计 backing，`ComputeRawWeightViewStorageUsage` 统计显式视图集合 |

### 1.2 不得负责

| 不负责 | 归属 |
|---|---|
| `ModelCompiler`、语义优化与 lowering | compiler |
| kernel resolve、`ExecutionPlan`、workspace 规划 | execution / backend |
| 具体打包布局、composite 物化、对齐与产物分配 | backend（`PackingRecipe` / `Backend::PackWeights`） |
| Runtime 资源装配、Session、Generate、KV 物理存储 | runtime / inference / kv_cache |
| 模型输入绑定与请求级 workspace | inference（`InferenceSession`） |
| 加载期静默精度转换 | 不存在的隐式行为；只能显式调用 `ConvertRawModelWeightsToFp32` |

边界规则：

- `ModelLoader` 是无状态 facade，只做 HF I/O、校验与逻辑解析；不调用 graph、backend 或 prepack。
- graph 模块只拥有 IR 与 semantic passes；`BuildModelGraph` / per-family builder 位于 model，不放在 graph。
- model 的权重组装组件可依赖 backend 的纯数据契约（`PackedWeight` / `PackingRecipe`），但打包执行必须经 `Backend::PackWeights`，不得 include CPU 具体 prepacker 实现。
- `LoadedModel` 不含任何 backend artifact；图构建消费其 config 与 resolved weights，不反向修改。

## 2. 关键数据结构与所有权

### 2.1 加载侧

| 类型 | 关键内容 | 所有权与生命周期 |
|---|---|---|
| `HfModelConfig` | `model_type`、`architectures`、hidden/intermediate/layer/head/vocab/context 维度、`head_dim`、`sliding_window`、`rms_norm_eps`、`hidden_act`、tie/bias 标志、dtype hint、`HfRopeConfig` | 按值解析；`LoadedModel` 按值拥有；构造后只读 |
| `HfRopeConfig` / `HfRoPEAlgorithm` | RoPE theta、factor、original context、beta、scaling 因子、short/long factors、算法枚举 | 原始 HF 表示；语义映射归 `MakeRoPEParams` |
| `HfDirectoryDescriptor` | `HfDirectoryLayout` 与 config/safetensors/index 路径 | 由 `InspectDirectory` 产出；`HfDirectoryReader` 按值持有 |
| `RawWeightView` | `data`、`bytes`、`dtype`、`shape`、`shared_ptr<const RawStorage>`、`is_contiguous` | 视图借用 backing；`storage` 共享所有权；不复制 payload |
| `RawStorage` | `RawStorageKind{kMapped,kOwned,kExternal}` 与 `footprint()` | 可由 mmap、自持 Buffer 或外部 owner 实现；footprint 在 owner 生命周期内不变 |
| `ResolvedModelWeights` | `embed_tokens`、`final_norm`、可选 `lm_head`、`vector<DecoderLayerRawWeights>` | 按值持有视图；每个 layer 含 norm/attn/mlp 角色；`NumLayers()` |
| `LoadedModel` | `HfModelConfig config_`、`ResolvedModelWeights resolved_weights_`、可选 `RawWeightFp32ConversionUsage` | 按值拥有 config 与 resolved views；可选 usage 仅为观测元数据；整体只读 |
| `ModelValidationOptions` | `strict_tensor_names`、`allow_unknown_tensors`、`allow_rope_scaling`、`allow_bias`、量化/adapter 开关、tied lm-head 与 uniform linear dtype 策略 | 调用方按值传入；默认接受常见 HF 导出差异，拒绝 bias/量化/adapter |

`ResolvedModelWeights` 的每个 `RawWeightView` 共享 `RawStorage`；`LoadedModel` 销毁即释放其 config、views 与自身持有的 backing 所有权，若其他视图仍共享 backing 则由 `shared_ptr` 延长。`LoadedModel` 的生命周期由 `LoweredModelArtifact` 的 `unique_ptr` 延长，见 [compiler.md](compiler.md)。`ComputeRawWeightStorageUsage` 按共享 owner identity 只计一次 backing extent，合并同一 owner 内重叠的视图范围；未知 footprint 置 `backing_usage_complete=false`。

### 2.2 图构建侧

| 类型 | 关键内容 | 所有权/约束 |
|---|---|---|
| `ModelArchitecture` | `kLlamaDense`、保留的 Qwen/MoE 枚举、`kUnknown` | 纯值类型；`ParseModelArchitecture` 只把 `model_type == "llama"` 识别为 `kLlamaDense` |
| `ModelGraphBuildOptions` | `activation_dtype`、`kv_dtype` | 调用方按值传入；未定义字段沿用未显式指定时的默认策略；不转换权重 payload |
| `ModelGraph` | graph 模块的语义 DAG | `BuildModelGraph` 按值返回；不持有 HF config 或 backend artifact |

`ModelGraphBuildOptions` 的精确语义：未定义 `activation_dtype` 时，activation 默认取 checkpoint dtype hint（缺失则 Float32）；未定义 `kv_dtype` 时默认取 activation dtype；显式 `activation_dtype == Float32` 会把 Embedding 输出设为该 dtype，并作为 activation 路径的默认值；显式 activation 与推导出的 Embedding 输出不一致时构图失败。显式 dtype 只改变语义 spec，不改写权重字节。

### 2.3 权重绑定、打包与存储

| 类型 | 关键内容 | 所有权/语义 |
|---|---|---|
| `WeightBinding` | 结构化角色 + 可选 `decoder_layer_index` | graph 纯数据类型；不拥有权重 |
| `ResolveWeightBinding` | 返回 `const RawWeightView*` | 借用 `resolved`；tied `lm_head` 回退 `embed_tokens`；composite/roleless/`kMoERouter`/越界 layer 返回 `nullptr`，调用方必须视为硬错误 |
| `WeightPackingRequest` | `op_type`、`source_id`、`value_index`、`binding`、`components`（打包前置要求非空）、`selector`、`recipe` | `components` 按逻辑绑定顺序借用共享 backing；请求由 compiler 生成，准备期填充 recipe |
| `WeightArtifactKey` | `source_id`、`value_index`、`binding`、`selector`、`recipe` | 完整身份；与 `PackedWeight` 产物精确匹配 |
| `PackedWeightCollection` | `vector<pair<WeightArtifactKey, shared_ptr<const PackedWeight>>>` + source/plan 状态 | 拥有产物共享引用；`Insert` / `Find` 按完整 key；首个成功 `Insert` 绑定 source；`total_storage_bytes()` 按 unique `BufferImpl` 去重；非线程安全，调用方需串行化 |
| `PackedWeight` | `op_type`、`selector`、`storage()`、`recipe()`、`logical_dtype()`、`logical_shape()` | backend 定义格式；collection 与 execution plan 共享所有权 |

`PackedWeightCollection` 的单一 source 不变量：collection 只能属于一个 `LoweredGraph::artifact_id()`；`SetSourceId` 之后拒绝不同 source，首次 `Insert` 也会冻结 source。`Insert` 校验非空产物、selector 与 artifact 一致、recipe 为已知显式布局且与产物一致、storage 对齐不低于 recipe、storage 字节数不小于逻辑权重大小；重复 key 返回 `AlreadyExists`。`Find` 没有 recipe-agnostic 查询面。

布局身份统一使用 backend 纯数据枚举 `PackingLayout`：`kNone` 表示未指定，生产请求与产物要求已知非空布局；改变 tile、panel 顺序或 padding 必须新增布局 ID。命名上 `PackedWeight*` 表示产物及其视图/存储/校验，`WeightPacking*` 表示请求与过程，`Prepack` 表示模型准备期、执行前的打包时机。

`PrepackWeightRequests` 要求整批共享 `source_id`，每个请求有已知且非零对齐的 recipe、非空 components；components 必须是 valid、contiguous、自然对齐的 row-major 视图。它先调用 `Backend::GetWeightPackingRequirements` 做容量预检（有限 cap 时 Unimplemented 直接失败；无 cap 时允许缺失并标记 `storage_plan_complete=false`），再对每个请求调用 `Backend::PackWeights`，校验 recipe 与 storage 需求，最后 `Insert`。allocator 仅本次调用借用，产物 Buffer 生命周期独立于它。失败全有或全无，空批返回未绑定空 collection。

### 2.4 FP32 转换

| 类型 | 关键内容 | 所有权 |
|---|---|---|
| `RawWeightFp32ConversionScope` | `kAllRequiredWeights` / `kNormWeightsOnly` | 选择被加宽的逻辑角色 |
| `RawWeightFp32ConversionOptions` | owned payload cap、非空 FP16/BF16 回调、scope | 调用方按值传入；回调仅本次调用借用 |
| `RawWeightFp32ConversionUsage` | source/converted usage、转换前后字节、峰值 owned bytes、完整性标志 | 仅观测元数据；不保留 source owner |
| `ConvertRawModelWeightsToFp32` | 返回新的 `unique_ptr<LoadedModel>` | 新模型拥有新 `LoadedModel` 与转换后 `RawStorage`；源模型不变 |

转换只替换被选中的 F16/BF16 角色：F32 与未选中视图继续共享原 owner；同一 selected source 视图只转换一次并共享一个转换 owner；`kNormWeightsOnly` 只选 `final_norm` 与每层 `input_rmsnorm`、`post_attn_rmsnorm`。config（包括 dtype hint）保持不变，缺失的可选 `lm_head` 仍缺失。所有源视图校验、尺寸核算与 cap 检查都在分配 payload 之前；失败不修改输入模型。

## 3. 接口与代码位置

| 入口 | 声明 | 实现 | 行为与错误语义 |
|---|---|---|---|
| `ModelLoader::Load` | [model_loader.h](../../include/aethermind/model/model_loader.h) | [model_loader.cpp](../../src/model/model_loader.cpp) | 顺序执行打开目录、解析 config、校验 config、加载 raw table、校验 weight set、resolve、校验 resolved、构造 `LoadedModel`；任一步失败返回对应 `Status` |
| `HfDirectoryReader::Open` / `InspectDirectory` | [hf_directory_reader.h](../../include/aethermind/model/formats/hf/hf_directory_reader.h) | [hf_directory_reader.cpp](../../src/model/formats/hf/hf_directory_reader.cpp) | 校验 root/config 与单一 safetensors 布局；单文件与分片同时存在时拒绝；分片索引必须与每个 shard 一致 |
| `HfDirectoryReader::ParseConfig` | 同上 | 同上 | 解析 config 子集，未知字段跳过；缺失 `num_key_value_heads` 时取 `num_attention_heads`；`sliding_window: null` 视为全注意力 |
| `HfDirectoryReader::LoadRawWeightTable` | 同上 | 同上 | 单文件直接读取；分片按 index 加载，要求每个 indexed tensor 恰好出现一次、shard 不含未列出 tensor、shard 路径不逃出模型目录 |
| `HfModelValidator::ValidateConfig` | [hf_model_validator.h](../../include/aethermind/model/formats/hf/hf_model_validator.h) | [hf_model_validator.cpp](../../src/model/formats/hf/hf_model_validator.cpp) | 校验正维度、`model_type` 存在、存在且非 null 的 `sliding_window` 不被支持、rms eps/rope theta、head 整除与 GQA、`hidden_act == "silu"`、bias/量化/adapter 策略、RoPE scaling 结构完整性、dense float dtype hint；家族支持性留给 `ParseModelArchitecture` |
| `HfModelValidator::ValidateWeightSet` | 同上 | 同上 | 校验必需 tensor、rank（norm 1、其余 2）、dtype、uniform linear dtype、可选 `lm_head` 与 tied 规则、未知 tensor 策略 |
| `HfModelValidator::ValidateResolvedModel` | 同上 | 同上 | 校验 layer 数与所有角色 shape（embedding `[vocab,hidden]`、q/o `[hidden,hidden]`、k/v `[kv_hidden,hidden]`、gate/up `[intermediate,hidden]`、down `[hidden,intermediate]` 等）与 dtype |
| `hf::ResolveWeights` | [hf_tensor_resolver.h](../../include/aethermind/model/formats/hf/hf_tensor_resolver.h) | [hf_tensor_resolver.cpp](../../src/model/formats/hf/hf_tensor_resolver.cpp) | 按固定 HF 名建立 resolved views；缺必需 tensor 返回 `InvalidArgument`；`lm_head` 可选 |
| `LoadedModel` | [loaded_model.h](../../include/aethermind/model/loaded_model.h) | [loaded_model.cpp](../../src/model/loaded_model.cpp) | `GetConfig` / `GetResolvedWeights` / `GetWeightFp32ConversionUsage` 均返回只读引用 |
| `ParseModelArchitecture` | [model_architecture.h](../../include/aethermind/model/model_architecture.h) | [model_architecture.cpp](../../src/model/model_architecture.cpp) | 唯一家族判定；未识别 spelling 返回 `kUnknown` |
| `BuildModelGraph` | [build_model_graph.h](../../include/aethermind/model/build_model_graph.h) | [build_model_graph.cpp](../../src/model/build_model_graph.cpp) | 按 `ParseModelArchitecture` dispatch；未知家族在读取权重前返回 `InvalidArgument` |
| `BuildLlamaDense` | [llama_dense_graph_builder.h](../../include/aethermind/model/llama_dense_graph_builder.h) | [llama_dense_graph_builder.cpp](../../src/model/llama_dense_graph_builder.cpp) | 校验 dtype override 与 config/resolved，构建 embedding、逐层 attention/MLP、final norm、lm-head、Argmax；RoPE 失败不产生部分图 |
| `detail::MakeRoPEParams` | [rope_params_builder.h](../../src/model/rope_params_builder.h) | [rope_params_builder.cpp](../../src/model/rope_params_builder.cpp) | 把 HF 字段规范化为一种 typed RoPE 变体；`kSu` 映射 `LongRoPE`；未知/不完整配置返回 `InvalidArgument` |
| `ResolveWeightBinding` | [weight_binding_resolver.h](../../include/aethermind/model/weight/weight_binding_resolver.h) | [weight_binding_resolver.cpp](../../src/model/weight/weight_binding_resolver.cpp) | 结构化角色 → 借用 view；tied lm-head 回退；`nullptr` 表示不可解析 |
| `PrepackWeightRequests` | [packed_weight_collection.h](../../include/aethermind/model/weight/packed_weight_collection.h) | [packed_weight_collection.cpp](../../src/model/weight/packed_weight_collection.cpp) | 批量预检 + `Backend::PackWeights` + `Insert`；支持可选 batch payload cap；失败全有或全无 |
| `ConvertRawModelWeightsToFp32` | [raw_weight_fp32_conversion.h](../../include/aethermind/model/weight/raw_weight_fp32_conversion.h) | [raw_weight_fp32_conversion.cpp](../../src/model/weight/raw_weight_fp32_conversion.cpp) | 显式转换入口；要求 CPU allocator、非空回调、有效 scalar F16/BF16/F32 源视图；返回新 `LoadedModel` |
| `ComputeRawWeightStorageUsage` | [raw_weight_storage_usage.h](../../include/aethermind/model/weight/raw_weight_storage_usage.h) | [raw_weight_storage_usage.cpp](../../src/model/weight/raw_weight_storage_usage.cpp) | 按 owner 统计 mapped/owned/external；支持 dtype 过滤与显式视图集合；不读取 payload |

`WeightPackingRequest` / `WeightArtifactKey` 的请求构建入口 `BuildWeightPackingRequests` 归 compiler；model 只提供请求类型与批量执行。`ResolveWeightPackingRequests`（准备期 recipe 注入与 consumer 合并）定义在 inference，不归 model。

## 4. 主流程/时序

### 4.1 Load

```text
ModelLoader::Load(model_dir)
  → HfDirectoryReader::Open / InspectDirectory       // 目录与布局
  → ParseConfig                                      // HfModelConfig
  → HfModelValidator::ValidateConfig
  → LoadRawWeightTable                               // RawWeightTable
  → HfModelValidator::ValidateWeightSet
  → hf::ResolveWeights                               // ResolvedModelWeights
  → HfModelValidator::ValidateResolvedModel
  → make_unique<LoadedModel>(config, resolved)
```

每步失败立即返回，不继续流水线；失败值不含部分构造产物。加载为一次性路径，时间与目录文件数、tensor 数量线性相关。

### 4.2 家族识别与 Llama dense 构图

1. `BuildModelGraph` 调用 `ParseModelArchitecture`；只有 `kLlamaDense` 路由到 `BuildLlamaDense`，其余返回 `InvalidArgument`。
2. `BuildLlamaDense` 先校验 `ModelGraphBuildOptions` 的显式 dtype（未定义或 scalar F32/F16/BF16），把 `num_hidden_layers` narrow 到 `uint32_t`，再调用 `ValidateConfig` 与 `ValidateResolvedModel`。
3. 解析 activation/KV dtype、`head_dim`（缺省 `hidden_size / num_attention_heads`），并在任何 graph mutation 之前调用 `MakeRoPEParams`。
4. 追加 `token_ids`、`position_ids` 两个 `Int64` rank-1 输入；Embedding 使用 `embed_tokens` 的 shape/dtype。
5. 每层先添加 K/V cache state，再构建 attention block（RmsNorm、Q/K/V Linear、RoPE、KV cache update、Attention、O Linear、residual Add）与 MLP block（RmsNorm、Gate/Up Linear、SiluMul、Down Linear、residual Add）。
6. 追加 final RMSNorm、lm-head Linear（`ResolveWeightBinding` 负责 tied 回退）、Argmax，`MarkOutput` 后 `graph.Validate()` 返回。

`BuildLlamaDense` 的节点计数由内部断言固定（attention block 9 个、MLP block 6 个）；增加/删除节点必须同步维护。输入顺序是 token_ids 在前、position_ids 在后，下游 Session 依赖该顺序合同。

### 4.3 权重准备生命周期

```text
HF tensor 名
  → resolve   : hf::ResolveWeights → ResolvedModelWeights
  → bind      : ResolveWeightBinding（结构化角色，tied lm_head 回退）
  → request   : compiler 的 BuildWeightPackingRequests（读 LoweredGraph）
  → pack      : inference 注入 recipe → model 的 PrepackWeightRequests → Backend::PackWeights
  → store     : PackedWeightCollection（完整 WeightArtifactKey）
  → specialize: execution 的 PrepareExecutionBindings
```

边界固定为：请求构建归 compiler，recipe 查询/合并归 inference，打包执行归 model 的 `PrepackWeightRequests`，物理布局与 composite 物化归 backend，执行期只读消费归 execution。`WeightPackingRequest.components` 为空、布局未指定或 source 混合都会在触达 backend 前失败。

### 4.4 显式 FP32 转换

1. `CollectViews` 收集 embed、final norm、可选 lm-head 与每层全部角色；每个源视图先做 `IsValid`、scalar dtype、`ValidateRawWeightView`、contiguity 校验。
2. F32 视图或 `kNormWeightsOnly` 下的非 Norm 视图进入 retained 集合；其余 F16/BF16 视图按 `(owner identity, data, bytes, bfloat, shape)` 去重。
3. 统计 converted source/weight bytes 与 retained owned bytes，得到最终 owned payload；有限 cap 先校验，失败在分配 payload 前返回。
4. 为每个唯一 source 调用对应回调分配并写入 FP32 Buffer，构造共享 `ConvertedRawStorage`，把同一 source 的所有目标 view 改为新 data/bytes/Float32/storage。
5. 重新计算 usage，返回 `make_unique<LoadedModel>(原 config, 新 resolved, usage)`；源 `LoadedModel` 不变。

## 5. 关键取舍

| 决策 | 选择 | 原因 | 被否方案 |
|---|---|---|---|
| 加载期权重表示 | `RawWeightView` 借用 backing，不复制 payload | 加载内存峰值低；backing 共享一次，视图可跨角色复用 | 加载时深拷贝所有权重；会增加峰值与拷贝成本 |
| 校验默认值 | 默认接受 extra tensors、缺失 tied lm_head、rope_scaling 字段 | 兼容常见 HF 导出；严格性由 `ModelValidationOptions` 显式收紧 | 默认 strict；会让大量合法目录无法加载 |
| 家族识别位置 | `ParseModelArchitecture` 单点判定，`BuildModelGraph` dispatch | 避免 loader 与 per-family builder 各自解释 HF spelling；未知家族在读权重前拒绝 | 在 `HfModelValidator` 里做家族 gate；会让加载与构图拥有两个权威 |
| RoPE 语义映射 | per-family builder 的共享件 `MakeRoPEParams` 独占，loader 只做结构校验 | 语义算法属于模型家族，不属于文件格式；失败可在构图前拒绝且不产生部分图 | loader 直接把 HF 字符串映射为语义算法；会把模型家族知识泄入 I/O 层 |
| dtype override | `ModelGraphBuildOptions` 只改 semantic spec，不转换 payload | 构图与权重字节解耦；精度策略由调用方显式选择 | 在 graph build 中顺带转换/量化权重；会隐藏分配与失败路径 |
| 打包职责 | model 只组装请求/产物，`Backend::PackWeights` 是唯一入口 | 布局、composite 物化、对齐与分配归 backend；model 不依赖具体 prepacker | model 直接实例化 CPU prepacker；会越界并锁死 backend |
| 打包身份 | 完整 `WeightArtifactKey` + 精确 `Find` | 同 selector 的 Q/K/V、gate/up、不同 layer 不再碰撞；recipe 区分布局变体 | 只按 `(OpType, selector)` 查询；会返回错误权重 |
| 转换时机 | `ConvertRawModelWeightsToFp32` 显式、构图前执行 | 保留源 config/hint，失败不修改输入；Loader/Compiler 不产生隐式转换 | 在 Loader 或 Compiler 内自动转换；会让“加载”产生未声明的精度语义 |
| 转换所有权 | 新 `LoadedModel` 自持转换 Buffer；未选中视图保留原 owner | 源模型可继续使用；转换 owner 生命周期与目标模型一致 | 就地修改源视图；会破坏调用方持有的原模型 |

## 6. 测试与验证入口

构建并运行聚焦测试：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='ModelLoader_*.*:ModelLoaderSlidingWindow.*:HfSlidingWindowValidatorTest.*:GraphSlidingWindow.*:LoadedModel.*:ModelArchitecture.*:BuildModelGraph.*:LlamaDenseGraphBuilder.*:ModelGraphBuildOptions.*:WeightBindingResolver.*:PackedWeightCollection*.*:WeightPacking*.*:RawWeightFp32*.*:RawWeightStorageUsage.*'
```

| 测试套件 / 过滤 | 文件 | 覆盖点 |
|---|---|---|
| `ModelLoader_PipelineTest.*`、`ModelLoaderSlidingWindow.*` | [test_model_loader.cpp](../../tests/unit/model/test_model_loader.cpp) | 单文件/分片目录到 `LoadedModel`、缺失权重、sliding window 在读取权重前拒绝、家族支持性延迟到构图 |
| `ModelLoader_HfDirectoryReaderTest.*` | [test_hf_directory_reader.cpp](../../tests/unit/model/test_hf_directory_reader.cpp) | 布局检测、单文件/分片加载、冲突布局、缺失/越界 shard、index 与 shard 一致性、符号链接拒绝 |
| `ModelLoader_HfConfigTest.*` | [test_hf_config_parser.cpp](../../tests/unit/model/test_hf_config_parser.cpp) | config 解析、可选字段默认、rope theta 别名、未知字段跳过 |
| `ModelLoader_HfModelValidatorTest.*`、`HfSlidingWindowValidatorTest.*` | [test_hf_model_validator.cpp](../../tests/unit/model/test_hf_model_validator.cpp) | 正/负 config、head/GQA 约束、RoPE scaling 结构、dtype hint、bias/量化拒绝、权重集合校验 |
| `ModelLoader_HfWeightResolverTest.*` | [test_hf_tensor_resolver.cpp](../../tests/unit/model/test_hf_tensor_resolver.cpp) | 单/多层解析、缺 layer 权重、可选 lm_head、超大 layer 数 |
| `ModelLoader_HfSafetensorsFileTest.*`、`ModelLoader_HfSafetensorsIndexTest.*` | [test_hf_safetensors_file.cpp](../../tests/unit/model/test_hf_safetensors_file.cpp)、[test_hf_safetensors_index.cpp](../../tests/unit/model/test_hf_safetensors_index.cpp) | safetensors header/offset/dtype/shape 校验、空 tensor、index 元数据 |
| `LoadedModel.*` | [test_loaded_model.cpp](../../tests/unit/model/test_loaded_model.cpp) | config/resolved 存储、view backing 保活、HF resolver 产物 |
| `ModelArchitecture.*`、`BuildModelGraph.*`、`GraphSlidingWindow.*` | [test_model_architecture.cpp](../../tests/unit/model/test_model_architecture.cpp) | 家族识别、未知家族拒绝、dispatch 在读权重前拒绝 |
| `LlamaDenseGraphBuilder.*`、`ModelGraphBuildOptions.*` | [test_llama_dense_graph_builder.cpp](../../tests/unit/model/test_llama_dense_graph_builder.cpp) | 完整拓扑、K/V state、RoPE 全算法映射、dtype override、非法配置、debug name |
| `WeightBindingResolver.*` | [test_weight_binding_resolver.cpp](../../tests/unit/model/weight/test_weight_binding_resolver.cpp) | 全局/layer 角色、tied/untied lm-head、越界/缺失 layer、composite/roleless/MoE 返回 `nullptr` |
| `PackedWeightCollection*.*`、`WeightPacking.*` | [test_packed_weight_collection.cpp](../../tests/unit/model/weight/test_packed_weight_collection.cpp)、[test_weight_packing.cpp](../../tests/unit/model/weight/test_weight_packing.cpp) | source/recipe/selector/对齐/字节数校验、重复 key、生命周期、批量预检、composite components、stride/alignment 失败 |
| `RawWeightFp32Conversion*.*`、`RawWeightFp32BatchConversion.*` | [test_raw_weight_fp32_conversion.cpp](../../tests/unit/model/test_raw_weight_fp32_conversion.cpp) | FP16/BF16 全 bit pattern、回调复用、cap 先于分配、owner 生命周期、partial failure 可重试、源 config/hint 保留 |
| `RawWeightStorageUsage.*` | [test_raw_weight_storage_usage.cpp](../../tests/unit/model/test_raw_weight_storage_usage.cpp) | owner 去重、重叠视图合并、dtype 过滤、未知 footprint 的完整性标志 |

测试 fixture 通过 `AETHERMIND_TEST_MODELS_DIR` 提供 tiny Llama 目录；真实模型集成用例在 [test_hf_real_model_integration.cpp](../../tests/unit/model/test_hf_real_model_integration.cpp)。