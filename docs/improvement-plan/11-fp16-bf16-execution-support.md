# FP16 / BF16 执行支持方案

- **状态**: In Progress
- **版本**: 1.8
- **日期**: 2026-11-11
- **产品边界**: [当前产品 PRD](../products/aethermind_prd.md)；CPU-only、单进程、同步、单请求、Token IDs I/O、Llama Dense、静态连续 KV
- **架构边界**: [AGENTS.md](../../AGENTS.md) §2.1
- **关联计划**: [06 系统能力路线图](06-system-capability-evolution-roadmap.md)、[07 模型准备入口](07-executable-model-preparation.md)、[08 KV/Attention 演进](08-kv-cache-and-attention-capability-evolution.md)、[09 模型家族扩展](09-qwen-and-moe-architecture-support.md)、[10 Runtime 组装入口](10-upper-level-runtime-assembly.md)
- **算子工作流**: [算子开发与优化工作流](../guides/operator-development-workflow.md)
- **关联模块**: dtypes / model / operators / graph / compiler / backend / execution / kv_cache / runtime / inference

本文是阶段化提案。P1 的显式 FP32 compatibility 已完成限定实现和 tiny checkpoint 验证，见 [FP32 兼容推理指南](../guides/fp32-compatible-inference.md) 与 [P1 验证报告](../tests/fp32_compatibility_validation_2026-10-29.md)。P2 首批 CPU/O1/plain 与 O2/identity-packed、W16/A32/KV32 已完成限定验收，入口见 [W16 指南](../guides/w16-fp32-inference.md)，当前工作树测试与 sanitizer 范围见 [P2 验收记录](../tests/w16_fp32_execution_validation_2026-11-11.md)。P3–P4 仍为未来工作。

P2 的转换基础已包括 [FP16 批量转换](../operators/fp16_conversion/README.md) 的 scalar、F16C widening、AVX2 narrowing，以及 [BF16 批量转换](../operators/bf16_conversion/README.md) 的双向 scalar/AVX2、checked byte spans、prepared Backend 函数表和 P1 callback。下文冻结 P2 的准入、Embedding、Norm-only 和 kernel args 合同，P2 能力状态由各批次退出门禁判定。

## 1. 现状分析

### 1.1 问题与推荐路线

“支持 FP16/BF16”需要分别回答：checkpoint 能否读取、图上 dtype 是否正确、所有 kernel 能否 resolve、数值是否符合合同、运行内存与性能是否达到目标。当前 P1 已在限定 tiny checkpoint 上闭环完整生成链；这不代表 native 16-bit 执行、任意 checkpoint 或后续阶段已支持。

推荐顺序：

```text
P0 数值与 dtype 合同
  → P1 F16/BF16 checkpoint 显式转换为 FP32 的兼容执行
  → P2 权重保留 16-bit，activation / 累加 / KV / logits 使用 FP32
  → P3 独立引入 16-bit KV storage
  → P4 16-bit activation 与对应硬件路径
```

P1 建立可执行基线，不是低内存方案；P2 是首个保留 16-bit 权重存储的交付目标。P2 的转换/SIMD 优化可以在该阶段获得 correctness baseline 后开展，不要求等待 P4。P3/P4 不作为 P2 的前置。

### 1.2 已核对事实

| 已核对事实 | 源码依据 | 对方案的约束 |
|---|---|---|
| `DataType`、`Half`、`BFloat16` 与 F16/BF16 safetensors 读取已存在；validator 接受 F32/F16/BF16 权重 | [DataType](../../include/aethermind/dtypes/data_type.h)、[HF dtype 解析](../../src/model/formats/hf/hf_utils.cpp)、[HF validator](../../src/model/formats/hf/hf_model_validator.cpp) | 不重复建设类型枚举或文件格式入口；加载不等于可执行 |
| 多数 Llama 算子语义允许 reduced dtype；`Add` 白名单有 BF16、缺 F16 | [Linear 合同](../../include/aethermind/operators/ops/linear_op.h)、[Add 合同](../../include/aethermind/operators/ops/add_op.h) | 补齐 F16 Add 时同步 semantic inference、constant folding、CPU 注册及测试 |
| `BuildLlamaDense` 从 `weight_dtype_hint` 得出 activation/KV dtype；Embedding 输出实际跟随权重 dtype | [Llama builder](../../src/model/llama_dense_graph_builder.cpp)、[Embedding inference](../../src/operators/ops/embedding_op.cpp) | hint 缺失/auto 或与实际权重不同，可能导致 KV 与 activation 不一致；需要显式执行精度权威 |
| Linear/RMSNorm 语义分别校验 activation 与 weight，输出跟随 activation；Attention/KVUpdate 要求 activation 与 cache 同 dtype | [Linear inference](../../src/operators/ops/linear_op.cpp)、[RMSNorm inference](../../src/operators/ops/rmsnorm_op.cpp)、[Attention inference](../../src/operators/ops/attention_op.cpp)、[KVUpdate inference](../../src/operators/ops/kvcache_update_op.cpp) | W16/A32 有现成扩展基础；mixed KV 必须先扩展语义 |
| Llama 所需 CPU kernel 尚未形成 F16/BF16 完整执行链；例如 Linear 注册为 F32/F32，BF16 Add 是局部已有能力 | [Linear entry](../../src/backend/cpu/kernels/linear/linear_entry.cpp)、[Add entry](../../src/backend/cpu/kernels/add/add_entry.cpp) | 按算子与完整 production path 分别验收，不将局部支持推广为整模型支持 |
| `KernelSelector` 严格匹配 activation/weight dtype；`DeriveSelectorDTypes` 忽略 state，未独立表达 output dtype | [selector](../../include/aethermind/base/kernel_selector.h)、[dtype 派生](../../src/operators/operator_inference.cpp) | W16/A32 可复用；mixed KV 和后续 mixed output 才扩展匹配身份 |
| identity packing 保留源 dtype；现有 B-panel 只有 `kCpuBPanelF32Kc512Nr16`，packer/validator 限定 F32 | [CPU packing](../../src/backend/cpu/cpu_backend.cpp)、[recipe](../../include/aethermind/backend/packing_recipe.h)、[packed validation](../../src/backend/cpu/packed_weight_validation.cpp) | 16-bit B-panel 需要独立物理布局身份，不复用 F32 ID |
| `Half(float)` 软件标量转换使用固定 RNE + gradual underflow，双向转换 quiet NaN；BF16 转换使用 RNE 并保留 sign 与截断后的 NaN payload | [Half 转换](../../src/dtypes/half.cpp)、[BF16 转换](../../src/dtypes/bfloat16.cpp) | Half 的 FTZ 行为已独立迁移；软件转换的合同和测试不代表 native 16-bit kernel 已支持 |
| CPU capability 已检测 F16C、AVX512BF16、AMX_BF16 等，尚无对应完整 kernel 链；现有编译配置分别隔离 FP32 AVX2/FMA、FP16 F16C widening、FP16/BF16 AVX2 integer conversion | [CPU capability](../../include/aethermind/backend/cpu/cpu_capabilities.h)、[检测](../../src/backend/cpu/cpu_info.cpp)、[库 CMake](../../src/CMakeLists.txt) | feature detection 不是执行支持；新增 ISA 按 translation unit 隔离 |

### 1.3 范围与非目标

首批仅覆盖当前 Llama Dense 的完整推理链；其他模型家族按照 [09 号提案](09-qwen-and-moe-architecture-support.md) 分别验收，不因共享 dtype 获得自动支持。当前 loader 默认要求 linear 权重统一 dtype；P1 normalizer 在加载之后执行，不绕过或放宽该校验。Norm 与矩阵权重可以使用已批准的不同 dtype。

本提案不引入 GPU、服务化、并发请求、Paged KV、量化格式、自动精度搜索、通用 precision optimizer 或第二套 Graph IR。FP16/BF16 的浮点精度转换不承担 INT8/INT4 的 scale/zero-point/group 语义。

## 2. 目标架构

### 2.1 精度矩阵与不变量

下表区分已验收的 P1 范围与后续目标。W 表示矩阵权重存储，A 表示 activation；首批 residual 跟随 A。

| 模式 | Checkpoint 输入 | 执行矩阵权重 W | A / residual | KV storage | 累加 / logits | 状态 |
|---|---|---|---|---|---|---|
| FP32 兼容执行 | loader 已接受的 F32/F16/BF16 | 显式转换后的 F32 | F32 | F32 | 沿用既有 kernel / F32 | P1 C++ O1/plain 限定验收通过 |
| FP16 权重执行 | uniform F16 linear weights | F16 | F32 | F32 | F32 / F32 | P2 首批限定验收通过（O1/plain + O2/packed identity） |
| BF16 权重执行 | uniform BF16 linear weights | BF16 | F32 | F32 | F32 / F32 | P2 首批限定验收通过（O1/plain + O2/packed identity） |
| 16-bit KV | 已验收的 P2 权重组合 | F16/BF16 | F32 | 显式选择 F16 或 BF16，K/V 同型 | F32 / F32 | P3 待实施 |
| 16-bit activation | 后续按算子签名批准 | 按 preset 决定 | 显式边界；首版保留 FP32 residual 作为候选 | 独立选择已验收组合 | 至少 F32 / F32 | P4 待设计与实施 |

P2 要求 Embedding、linear 和 lm_head 的实际权重 dtype 与所选 F16/BF16 preset 一致。Norm 源权重允许 scalar F32/F16/BF16，准备后统一为 F32，矩阵权重保留目标 16-bit 格式。缺少独立 lm_head 时沿用 resolver 的 embedding fallback。显式 lm_head 验证实际 dtype，并保留其既有 storage 关系。

**P2 source hint 合同（当前实现）。** 在 [model_execution_policy.cpp](../../src/inference/model_execution_policy.cpp) 的新 policy 分支接受下表组合，先完成 metadata 准入，再分配转换 payload。

| Hint | P2 处理 |
|---|---|
| 缺省，或 undefined 且 name 为空 / `auto` | 接受，执行 dtype 由 policy 与实际权重决定 |
| scalar F32、F16、BF16，且与实际矩阵 dtype 一致 | 接受并保留原配置 |
| scalar F32、F16、BF16，与实际矩阵 dtype 不一致 | 接受并原样保留为 provenance / diagnostic，执行 activation/KV 显式为 F32 |
| 未识别的命名 hint，或上述集合以外的 dtype / lanes | 拒绝 |

所有矩阵角色的实际 dtype 仍须匹配所选 policy。即使 hint 匹配 policy，错误或混合的矩阵 dtype 也须拒绝。准入测试分别覆盖 known-hint mismatch 的接受与 actual-dtype mismatch 的拒绝。现有 Loader 已允许已识别 hint 与实际 dtype 不一致，P2 可沿用该读取合同。

必须保持：

1. TensorSpec、实际 bytes、TensorView、packed logical dtype、selector 与 kernel 参数一致。修改 dtype 标签不能代替数据转换。
2. checkpoint 源 dtype 以各 tensor 的实际 metadata 为准；执行 activation/KV dtype 以经解析的 graph-build options 为准，`weight_dtype_hint` 只保留来源信息。
3. 不隐式进行 BF16→F16、F32→16 或 F16↔BF16 转换；首次 P2 不承诺任意 mixed linear dtype。额外转换必须成为明确策略并单独验收。
4. 累加精度是算子/kernel family 的固定数值合同，首批不开放任意 accumulator dtype 配置。
5. 不支持的 tuple 在编译或准备阶段失败，错误携带 op、端口、实际/要求 dtype、preset 和 phase。ISA fallback 只能替换实现，不能改变精度合同。
6. 相同 preset 的 Prefill/Decode 使用一致的权重与 KV 表示；阶段特化只能在同一数值合同下选择 kernel。

### 2.2 职责、数据结构与拟新增接口

下表的 P1/P2 model/compiler/inference、Embedding mixed output 与 W16 backend 已实现，后续 mixed KV 扩展仍是提案。C++ API 不承诺冻结 binary ABI。

| 所在层 | 数据/接口方向 | 职责 |
|---|---|---|
| inference | `WeightPolicy` / `ModelLoadOptions` 接入显式 C++ 准备入口 | P1 选择 O1/plain + compatibility，协调 Runtime CPU allocator、normalizer、FP32 graph options 与已有 Prepare；不实现转换算法或新 kernel |
| model | `ModelGraphBuildOptions` 与 `ConvertRawModelWeightsToFp32` | 构图前完成真实逻辑权重转换；`BuildModelGraph` 继续是模型到 semantic graph 的唯一入口 |
| compiler | `ModelCompileOptions.graph_build` 传给 `BuildModelGraph` | 编排 Build→Optimize→Lower；不分配/转换权重，不访问 backend registry |
| operators | `EmbeddingParams.output_dtype` 及其 inference/serde；后续按需增加 Cast | 定义 gather+widening、mixed KV、转换边界的可见语义，不查询 ISA |
| backend | 新增对应 dtype tuple 的 kernel、params builder、recipe/validator | 实现 weight widening、FP32 累加、物理 packing 与 ISA 选择 |
| execution | 既有 plan/binding/prepare 路径；P3 扩展 state dtype 身份校验 | 冻结 kernel/资源，校验 TensorSpec 与物理 binding；维持完整 plan 的 KV transaction |
| kv_cache / runtime | 使用解析后的 `kv_dtype`，沿用 layout/storage/allocator | 管理物理字节与布局，不解析 HF hint，不判断算子数值语义 |

不为本提案新增 owning Engine facade。当前统一使用 `ExecutableModel::Load(runtime, path, ModelLoadOptions)`，按 `WeightPolicy` 选择源权重准入及可选转换；artifact 准备统一使用 `ExecutableModel::Prepare`。旧创建自由函数与转发 overload 已删除，C++ 调用方需迁移；直接 `ModelCompiler::Compile` / `LoadAndCompile` 仍是低层编译入口，不自动触发转换。低层调用方若绕过 normalizer，必须构造真实一致的 specs，未覆盖的 kernel tuple 在 prepare 时明确拒绝，不能静默取得兼容模式的承诺。

新增 Embedding 字段的缺省语义须保留既有“输出跟随权重”；W16/A32 preset 显式选 F32 输出。字段迁移覆盖 OpParams、graph builder、inference、序列化/反序列化、constant evaluator（如适用）、kernel 参数和测试。通用 Cast 延后到 P4 真正需要图上 dtype transition 时再引入。

**P2 Embedding 输出与 builder 合同（当前实现）。** `EmbeddingParams::output_dtype` 的唯一解析规则为 `output_dtype.IsUndefined() ? weight_spec.dtype : output_dtype`，`InferEmbedding` 用该结果生成 output spec。首批显式 output dtype 支持 scalar F32，undefined 路径继续覆盖原有 F32/F16/BF16 同型输出。

[llama_dense_graph_builder.cpp](../../src/model/llama_dense_graph_builder.cpp) 中当前比较 `options.activation_dtype` 与 `weights.embed_tokens.dtype` 的 guard 必须替换。P2 构图先给 Embedding 设置 `output_dtype=F32`，再将显式 activation override 与推导后的 Embedding **output spec dtype** 比较。output dtype 未设置时，该 spec 跟随 weight。回归须验证旧缺省调用的跟随规则，以及 W16 weight + 显式 F32 activation/output 能成功构图且输出 spec 一致。

**P2 Embedding serde 合同（当前实现）。** 修改 [op_params_serde.cpp](../../src/operators/op_params_serde.cpp) 的序列化与解析分支。

| 参数状态 / 文本 | Canonical serialize / parse |
|---|---|
| `output_dtype` 为 undefined | 仍序列化为裸 `Embedding`，与旧输出逐字节一致 |
| 显式 scalar F32 | 序列化为 `Embedding output_dtype=float32` |
| 裸 `Embedding`，0 字段 | 解析为 `EmbeddingParams{}` |
| `Embedding output_dtype=float32`，1 字段 | 解析为显式 scalar F32 |
| 未知 key、重复 key、空值、额外字段、首批集合以外的显式 dtype | 返回 `InvalidArgument` |

解析只接受 0 或 1 个字段，重复 key 继续由已有 `ParseFields` 拒绝。对应回归包括两种 canonical 文本的 round trip、旧裸格式的逐字节断言，以及各非法字段分支。

### 2.3 控制与数据流

```mermaid
flowchart TD
    A[ModelLoader::Load 保留原始 dtype] --> B[inference 解析 precision preset]
    B --> C[model 显式逻辑权重转换]
    B --> D[graph-build options 与 Runtime KV 配置]
    C --> E[LoadedModel 拥有最终逻辑权重]
    E --> F[ModelCompiler::Compile]
    D --> F
    F --> G[BuildModelGraph / Optimize / Lower]
    G --> H[LoweredModelArtifact]
    H --> I[ExecutableModel::Prepare]
    D --> I
    I --> J[Backend::PackWeights / kernel resolve / bindings]
    J --> K[ExecutableModel]
    K --> L[InferenceSession::Generate]
```

P1 转换全部非 F32 必需权重；P2 只执行合同要求的 Norm 等小权重转换，保留矩阵权重。转换完成后才构图，避免已冻结 TensorSpec 与实际存储不符。两种模式都只加载一次，不重新读取同一模型目录。

与 [10 号 Runtime 组装方案](10-upper-level-runtime-assembly.md) 集成时，先从已验证 config 和 preset 派生 KV dtype、capacity 及 RuntimeOptions；将 Runtime 放到最终地址，再通过其 allocator 完成需要的逻辑权重分配、Compile、Prepare、CreateSession、Generate。现有首版 FP32-only 入口合同在相应阶段验收后才扩展。

### 2.4 Ownership、分配与失败边界

```text
normalizer 生成新的 immutable RawStorage（拥有转换 buffer）
  → RawWeightView.storage
  → 新 LoadedModel 的 ResolvedModelWeights
  → LoweredModelArtifact.loaded_model
  → ExecutableModel.artifact_
       └─ immutable bindings 借用上述稳定 bytes 与自持 metadata
```

normalizer 创建新的 `LoadedModel`，不 `const_cast` 修改只读权重，不覆写 checkpoint mmap，也不改写原始 hint 伪装来源。无需给 ExecutableModel 增设平行的 converted-weight owner。沿用 [RawStorage/RawWeightView](../../include/aethermind/model/raw_weight.h)、[LoadedModel](../../include/aethermind/model/loaded_model.h) 和 [ExecutableModel](../../include/aethermind/inference/executable_model.h) 的生命周期路径。

- payload 经传入的 `Allocator&` 分配，以现有 Buffer/MemoryHandle 的正确释放合同封装进 source-private RawStorage 实现；不保存仅供分配调用的借用 allocator 引用，不更改 Allocator 接口。
- 每个输出 view 携带真实 dtype、shape、bytes 与 owner，并通过 `ValidateRawWeightView`；所有 shape×element-size、总字节与对齐计算检查溢出。
- 源数据不假定满足 SIMD alignment 或已构造为 `Half*`；使用满足 alignment、object lifetime 和 strict-aliasing 规则的读取，必要时经 `memcpy`/bit conversion。
- 按来源 storage、range 和逻辑 tensor identity 去重转换，保留 tied embedding/lm_head 的同源 alias；不能仅凭内容相同合并不同 tensor，也不能把同一权重转换两份。
- F32 来源可继续共享原 owner；不因兼容模式无条件再复制。RawStorage 只读性由封装与调用合同保证，`const Buffer` 本身不能证明没有其他可写 alias。
- 部分转换、metadata 或分配失败时 RAII 清理并返回原状态码及阶段上下文，不发布半成品模型；已存在的外部模型与 plan 不被替换。
- Runtime 地址稳定且长于模型/Session；model 长于所有借用权重的 prepared bindings。请求 workspace 仍由 Generate 分配，planner 只规划偏移。

**P2 Norm-only 预算合同（当前实现）。** 在 [RawWeightFp32ConversionOptions](../../include/aethermind/model/weight/raw_weight_fp32_conversion.h) 增加显式 scope，默认覆盖 P1 全量转换，P2 选择 Norm-only。所有角色先验证 metadata，转换计划仅收集 scope 选中且非 F32 的 view。retained 统计从转换后仍保留的 source views 构建，复用既有 ownership-identity 聚合与 range union 逻辑。

```text
final_owned = unique surviving source owners 的 owned backing
            + unique converted Norm F32 payload
peak_owned  = unique all source owners 的 owned backing
            + unique converted Norm F32 payload
```

Norm 与矩阵共享 backing 时，Norm 创建 F32 storage，矩阵保留原 view，原 backing 按 owner identity 计一次完整 footprint。Mapped/external 分类单列。有限 cap 要求 retained backing 信息完整，包含保留的 W16 backing。

同一实现批次必须更新转换器 public header 中的 `Unknown retained F32 backing...` 注释和 `retained Float32 backing metadata` 错误消息，使二者描述 retained backing（含 W16）。P1 与 P2 的固定账本验收见 §6.2。

### 2.5 Kernel、selector 与 packing

**P2：复用已有 act/weight selector。** Linear 使用 `{act=F32, weight=F16}` 或 `{act=F32, weight=BF16}`，输出 F32；Embedding 从显式 activation output 得出 act dtype。reference 读取 16-bit weight 并提升为 float，用 FP32 累加，直接写回 F32。不得用 reduced wrapper 的 `operator+=` 作为归约，也不在每个 token 展开整份 FP32 权重。

**P2 W16 Linear args 合同（当前实现）。** 现有 [LinearF32KernelArgs](../../src/backend/cpu/kernels/linear/linear_internal.h) 持有 `const float* weight`。新增 CPU-private `LinearF32W16KernelArgs`，由注册的 params builder 一次构造。

| 字段 / 行为 | 合同 |
|---|---|
| `input` / `output` | `const float*` / `float*`，activation/output 为 F32 |
| `weight` | `const std::byte*`，用 `memcpy` 读取 16-bit 元素 |
| `weight_dtype` | 实际 scalar F16 或 BF16，须与 weight TensorView 和 selector 一致 |
| 所有 row/col stride | 按对应张量的**元素数**计量，默认 col stride 为 1 |
| weight 地址计算 | 将逻辑元素 offset 显式乘以 2 转为 byte offset，binding-time 校验最大 offset 和字节范围溢出 |
| shape / layout / alias | 复用现有 TensorView、`AnalyzeRowwiseView` 与范围/alias 校验，沿用 rank flatten、空输出和零 K 合同 |

stride 回归使用相同物理 storage 的不同合法 row/col stride、非对齐 W16 source、尾部和溢出 metadata，确认 prepared args 与实际读写范围一致。

Prefill 可在已规划 scratch 中按 tile widening 并复用，Decode 可直接流式读取 16-bit weight；具体 blocking/threshold 属算子专项，须经 profiling 与 benchmark 决定。新增需求沿用 `WorkspaceRequirement`→plan→request allocation→binding，不在 kernel 中申请堆内存。

先验收 O1/plain；随后为 QKVLinear、GateUpLinear 等 fused consumer 补齐 mixed-dtype reference、identity packing 和绑定，再开放明确的 O2/packed preset。不能只增加 Linear 注册就宣称默认 O2/plain 可执行。

identity 布局可在合同允许时继续保存 16-bit 元素；如需新的 B-panel，新增包含 dtype、blocking、panel 顺序或 padding 特征的 `PackingLayout`，同步 byte formula、overflow、packer、descriptor、validator、artifact key 与 consumer。物理格式改变不得沿用 `kCpuBPanelF32Kc512Nr16`，也不预先照搬其 tile 参数。

**P3：最小扩展 state dtype。** 首批限定所有相关 K/V state ports 同 dtype，拟将 `state_dtype` 加入 KernelSelector；无 state 的 step 使用唯一的 undefined 表示，按严格相等匹配，不作为 wildcard。同步修改 dtype 派生、hash/equality/ToString、registry、lowering、untrusted plan 重校验与包含 selector 的 packing key。state dtype 来自真实 state specs，不借用 `weight_dtype` 字段。

KVUpdate 参数准备必须同时核对 state TensorSpec、selector、KV view/layout 与 binding dtype；保留现有 execution 校验。Attention 明确允许 `Q=F32, K/V=F16或BF16, output=F32`，KVUpdate 明确允许 `K/V activation=F32 → cache16`，在读写边界转换。

P4 如出现 input/output dtype 不同或同角色的多端口异型，需要按实际 schema port/role 扩展签名；单一 `result_dtype` 不能覆盖任意多输出异型情况。该合同在 P4 冻结并全链路迁移，P2 不提前建设通用端口签名系统。

### 2.6 数值与 ISA 合同

| 项目 | 推荐合同与限制 |
|---|---|
| Widening | F16/BF16 的有限值提升到 F32 精确表示，包括可表示的 subnormal；零符号保留；NaN 按已冻结的 bit-copy/quieting 规则处理 |
| Narrowing | Half 软件标量转换已采用固定 RNE + gradual underflow，保留 signed zero，overflow→Inf，NaN quiet 并保留可容纳的 payload，不设置浮点异常标志；FTZ 迁移见 [公共 API](../api/public-api.md)。后续 narrowing/SIMD 路径须匹配该合同；P1 的纯 widening 不依赖此次迁移 |
| GEMM/归约 | P1 沿用已有 kernel 的数值合同，当前 FP32 Linear reference 使用 double 累加。P2 新增 W16 Linear 固定 F32 累加，显式规定 reference 的累加顺序与 FMA contraction 策略，以独立 FP64 oracle、约定误差和 near-tie 规则验收 |
| Norm/Attention | 平方和、dot、stable softmax 的 max/exp/sum 与 value accumulation 至少 F32；RoPE 保留其既有更高精度频率/角度合同，不能因 dtype 扩展缩窄 |
| Fusion | 先规定 logical op 边界的 rounding/cast placement；融合不能移除要求的 narrowing 或额外引入舍入。浮点运算顺序变化按数值门禁审查，不要求所有实现逐位一致 |
| 硬件转换 | FP16 批量转换已具备 F16C widening 与 AVX2 整数 narrowing，并接入 P1 的 F16 权重准备，见 [合同与证据](../operators/fp16_conversion/README.md)。BF16 已有双向 scalar/AVX2 batch 与 P1 callback，见 [BF16 合同](../operators/bf16_conversion/README.md)。新增候选须遵守相同 subnormal/NaN 和异常合同 |
| Native BF16 | `VDPBF16PS` 乘数为 BF16、累加为 FP32；不能把 W16/A32 的 activation 偷转 BF16 以复用它。此类路径属于经过批准的 A16 模式 |
| Native FP16 | 有 FP16 算术指令不等于满足 FP32 累加；任何候选必须证明符合该 preset 的数值合同 |
| Dispatch/build | 独立 TU 与编译能力检查，使用实际 CPU/OS effective features，保留 scalar fallback；不全局添加高 ISA 或 fast-math 选项，不按 CPU 品牌猜测支持 |

P2 reference 的数值门禁固定如下。Embedding 与已冻结的 widening 位结果一致。W16 Linear 按 K 升序，每次先舍入 F32 product，再舍入 F32 add，reference TU 禁用 FMA contraction。有限且无溢出的单算子对照采用独立 double oracle，按 `gamma(2K) * sum(abs(A[k] * W[k]))` 加绝对误差下限检查，其中 `gamma(n)=n*u/(1-n*u)`、`u=2^-24`，并单列 subnormal、NaN/Inf 与溢出分类。

首批 tiny checkpoint 模型轨迹采用固定门禁 `abs_error <= 1e-5 + 1e-4 * abs(reference)`，逐 phase 记录最大误差。top-1 margin 大于两倍该步 logits 最大绝对误差时要求 argmax 一致，near-tie 按此 margin 规则分类。数值预算在调整算法前固定，后续改变 FMA/归约顺序的实现独立验收。

FP16 与 BF16 的指数范围和有效精度不同，不能按“都是 16-bit”互换。ISA 的可用性、吞吐与 workload 共同决定选择；不预设 BF16 总比 FP16 快，也不预设 16-bit 总比 FP32 快。硬件机制来源见 §7。

## 3. 方案与备选

| 方案 | 收益与代价 | 裁决 |
|---|---|---|
| 先显式转 F32，再建设 W16/A32 | 先复用完整 CPU 基线；兼容路径启动/内存成本较高，不能作为最终低内存路径 | 推荐分阶段实施；两个模式分别验收 |
| 一次性把所有 tensor 改为 F16/BF16 | 表面配置统一，但叠加 Norm/Softmax、残差、KV、logits、fusion rounding 与 ISA 风险 | 延后至 P4 按边界推进 |
| Loader 隐式转换 | 调用简短，但混淆读取事实与执行策略，掩盖峰值内存和来源 dtype | 不采用，显式 model helper 由 inference 编排 |
| 只修改 hint/selector/spec | 无真实转换，导致 dtype 与 bytes 不一致，可能触发错误读取或错误数值 | 禁止 |
| 直接复用 BF16 dot 执行 A32/W16 | 可调用现有硬件指令，但会额外量化 activation | 不作为等价 fallback；单列 A16 preset |
| 预先建立通用 precision pass/任意 dtype 笛卡尔积 | 扩展空间大，但增加无消费者抽象与难以覆盖的组合 | 先固定 presets；Cast 和异型端口签名按真实需求引入 |
| 首批仅依赖外部低精度算子库 | 可减少 kernel 开发，但新增依赖、dispatch、布局与数值集成合同 | 当前不作为前置；后续若收益明确，仍经 Backend 契约接入 |

## 4. 实施步骤

P1 的限定实现与验收已完成。P0 中 graph options 和 widening 数值合同已随 P1 落地，Half 软件标量 narrowing 已完成 FTZ→gradual underflow 迁移，FP16 F16C/AVX2 与 BF16 AVX2 批量转换已经接入当前源码。P2 已按以下首批合同实施，activation/residual/Add 全程使用 F32，F16 Add 按后续 reduced-activation consumer 的需要建设。P2–P4 的实现状态按退出门禁更新。各算子依照 O0–O4 执行。

### 4.1 P0：冻结合同与补齐基础语义

1. 固定 §2.1 的首批 presets、source/activation/KV 权威与明确拒绝策略。
2. 引入 graph-build options，经 ModelCompileOptions 传到唯一构图入口；覆盖 hint 缺失、auto、与实际权重不同的情况，不改 checkpoint 来源 metadata。
3. F16 Add 的 shared dtype contract、CPU reference 注册、constant folding 与相关测试按后续 A16 consumer 的需要联动补齐。P2 residual/Add 使用既有 F32 路径。
4. Half 软件标量转换已明确 rounding、NaN/Inf、subnormal 和 FTZ 迁移范围，并有独立回归覆盖；任何新 narrowing/SIMD 路径开放前仍须验证其数值和异常行为匹配同一合同。

退出条件：本批次 consumer 所需的语义与 dtype 身份一致，非法组合可诊断。F16 Add 的 producer/consumer/evaluator/CPU reference 联动验收归入使用 F16 activation 的批次，P2 按 F32 Add 合同验收。

### 4.2 P1：显式 FP32 兼容执行（限定验收通过）

1. model 实现独立 normalizer，基于每个真实 RawWeightView 转换；通过 RawStorage 拥有结果并去重 tied tensor。
2. inference 的显式模式串联 Load→Normalize→Compile→Prepare；低层直接 Compile 不偷偷触发转换。
3. FP32 graph/KV/logits 配置覆盖原始 hint，保留来源信息；复用真实 CpuBackend 和 O1/plain。
4. 在结果诊断中明确该模式已将执行权重展开为 F32；统计转换字节、保留 backing 与准备期内存。

退出条件：F16/BF16 tiny Llama checkpoint 分别完成 Prefill、至少两次 Decode 与 Generate，匹配从同一 checkpoint 解码得到的独立 FP32 reference；生命周期、tie alias 和失败清理通过。

### 4.3 P2：16-bit 权重、FP32 activation

首批步骤 1–9 已完成 CPU/O1/plain 与 O2/identity-packed、W16/A32、KV32。步骤 8 已记录 reference baseline，并接入经 feature policy 选择的 Embedding F16C/AVX2 行级 widening。Linear reference 的后续优化、Decode GEMV/Prefill 分块和新 W16 B-panel 依据各自 profiling 证据推进。

1. **冻结支持矩阵与数值合同。** 落实 §2.1 的 hint 白名单、不一致策略与按角色区分的 actual dtype 准入，明确 P2 F32 累加、reference 的 FMA contraction 策略、误差预算及 near-tie 规则。同步维护本提案版本与变更记录。
2. **扩展 inference 准入。** 在 [model_execution_policy.h](../../include/aethermind/inference/model_execution_policy.h) 增加 F16/BF16 两个 W16 策略，矩阵严格匹配所选类型，Norm 源接受 F32/F16/BF16。验证 unknown policy、target、preset、known-hint mismatch 和 actual-dtype mismatch 的各分支。
3. **完成 Norm-only transformation 与预算。** 在 [raw_weight_fp32_conversion.h](../../include/aethermind/model/weight/raw_weight_fp32_conversion.h) 的 options 增加 `RawWeightFp32ConversionScope::{kAllRequiredWeights,kNormWeightsOnly}`。转换器按角色选择 FinalNorm、InputNorm、PostAttentionNorm，复用两套 widening callback 与 SourceKey 去重。依 §2.4 重做 retained 选择，更新 cap 注释/错误文案，并通过 §6.2 的 P1/P2 独立账本和失败回滚测试。
4. **完成 Embedding 语义和 builder/serde 迁移。** 修改 [op_params.h](../../include/aethermind/operators/op_params.h)、[embedding_op.cpp](../../src/operators/ops/embedding_op.cpp)、[graph_op_builder.cpp](../../src/graph/graph_op_builder.cpp)、[llama_dense_graph_builder.cpp](../../src/model/llama_dense_graph_builder.cpp) 与 [op_params_serde.cpp](../../src/operators/op_params_serde.cpp)。尤其要将 builder 的 activation-vs-weight guard 改为 activation-vs-inferred-output 比较，按 §2.2 保持裸 serde 的逐字节兼容并扩展 0/1 字段解析。退出门禁包括 operator inference、serde、graph-build 和 selector dtype 派生回归。
5. **实现 O1/plain reference kernel。** Embedding 按选中 W16 row 写 F32 output，Linear/LM head 使用 §2.5 的专用 W16 args 和 F32 累加。分别注册 `{A=F32,W=F16}`、`{A=F32,W=BF16}`，覆盖 Prefill/Decode、合法 stride、非对齐、尾部、零 M/N/K、alias 和执行期零分配。
6. **接通公开 Load 链。** [executable_model.cpp](../../src/inference/executable_model.cpp) 编排 source 准入→Backend widening 选择→Norm-only conversion→prepared 角色校验→Compile(A32/KV32)→Prepare→Session。转换后校验 Norm=F32、矩阵=W16，复用原有 owner 链，分别闭环两种 policy。
7. **完成 P2-O1 模型级验收。** 对同一 reduced checkpoint 比较 P2 与 P1，按 §6 的 logits abs/rel、top-k、top-1 margin 验证 Prefill、多次 Decode 和 Generate。覆盖 MHA/GQA、tied/untied、EOS/容量、OOM 重试、owner lifetime、持久矩阵两字节存储与 Decode 稳态零分配，执行相关 sanitizer profile。
8. **采集 O1 baseline 并优化。** 记录 registered/prepared、production ExecutionStep 与端到端准备/生成指标，新增 scratch 经现有 workspace 规划链提供。**当前状态**为 [Linear baseline](../operators/linear/benchmarks/desktop-54h5mmi-linear.md)、[Embedding 行级 SIMD 对照](../operators/embedding/benchmarks/desktop-54h5mmi-embedding.md) 和 [模型级记录](../tests/benchmarks/models/desktop-54h5mmi-model.md) 已完成。Embedding 复用既有 F16C/AVX2 converter，完整位模式、feature-policy fallback、O1/O2 轨迹与零分配已回归。Linear 后续优化由其 baseline 和 profiling 决定。
9. **完成 P2-O2/packed。** 按预期 dtype 和 itemsize 验证 identity packed weight，补齐 W16 Embedding/Linear、QKV/GateUp composite、reference、参数准备与绑定，回归 AddRmsNorm 等实际 fusion。完整 O2 图通过后，再依据性能证据决定是否新增 W16 B-panel 及其独立布局身份。**状态：identity 批次已完成**，O2 preset 对两个 W16 policy 开放；W16 B-panel 评估为延后——baseline 证明瓶颈在 scalar widening 而非布局，identity 已保持 2 字节 composite payload，新布局须待 widening 优化后的独立证据（决策记录见 Linear 基准记录）。

步骤 1 完成后，步骤 2/3/4 可以并行推进。步骤 5/6 汇合语义、权重与 backend 实现，步骤 7 是 O1 批次退出门禁，步骤 9 另有 O2 退出门禁。

退出条件为两个 W16/A32 policy 均完成生成，持久矩阵保留 16-bit，plain/packed 支持范围分别记录，Decode 稳态分配合同通过。**首批达成**范围包括 tiny reduced checkpoint 的 O1/O2 Prefill/Decode/Generate、EOS、容量边界、失败重试和 owner lifetime，模型账本确认矩阵两字节且转换字节仅来自 Norm。当前工作树的 ASan/UBSan 与独立 Release malloc-family 门禁见 [验收记录](../tests/w16_fp32_execution_validation_2026-11-11.md)，其中明确 LSan 的环境限制。

### 4.4 P3：独立 16-bit KV

1. 在 operators 冻结 FP32 activation 与 KV16 的读写转换、输出 alias 和数值合同；K/V 同型。
2. 迁移 state dtype selector 身份和全部派生/重校验/registry/key；KVCacheLayout 按实际元素字节、alignment/padding 规划容量。
3. KVUpdate 写入时 narrow，Attention 读取时 widen，保持 FP32 attention reduction；Runtime 配置与图 state 精度一致。
4. 复核完整 plan 成功后才 commit watermark；失败写入不发布，stale view、owner/generation 与现有单 reservation 合同保持。

退出条件：Prefill/Decode、capacity 边界、长上下文误差与真实 byte accounting 通过；不能仅凭 Manager 能分配两字节元素就声明 KV16 可执行。

### 4.5 P4：16-bit activation 与 native compute

1. 以 P2/P3 数值与 profiling 证据选择首个具体 A16 preset，逐端口定义 residual、Norm、RoPE、Attention、MLP 与 logits 的存储/输出 dtype。
2. 对真实 graph transition 增加 typed Cast，覆盖 inference/serde/evaluator/kernel；最终 logits 保持 F32，不能先窄化 logits 再 widening 冒充保留精度。
3. 对 mixed input/output/multi-output 签名迁移 selector 与 prepared 合同，重新审查所有 fusion 的舍入边界。
4. 验证 AVX512BF16、适用的 FP16 或 AArch64 路径；缺失 capability 时使用同合同的实现，不隐式改变 preset。AMX 等需要额外 OS/resource 合同的实现单独准入。

退出条件：该具体 preset 的完整算子、phase、ISA/fallback 与模型级证据通过后才标记支持；不泛化为所有 dtype 或平台组合。

## 5. 风险与依赖

| 风险/依赖 | 处理与准入条件 |
|---|---|
| P1 峰值内存与启动成本 | F32 转换权重、原 checkpoint backing 和 packed artifact 可能共存；统计实际 owner 存活与 RSS，不以文件映射大小直接推算驻留内存。资源不足可恢复失败 |
| 所有权/allocator lifetime | 转换产物自持 owner、正确释放 provenance、tie 去重与稳定 metadata；不借用 normalizer 临时 buffer，不向热路径传 allocator 引用 |
| 数值兼容 | Half FTZ→gradual underflow 是可观察行为变更，独立迁移；FP16/BF16 和不同 fusion/ISA 的误差分别验收 |
| 混合 checkpoint | 延续 loader 的 uniform-linear 默认合同；norm 单独处理。扩大接纳范围需先改验证政策及 fused composite 合同，不让 normalizer 成为绕过校验的入口 |
| 默认优化组合 | 按 O1/plain、O2/packed 明确验收，不把 precision 选项与任意 optimization level 自动组合；未 resolve 的 kernel 保留原始错误 |
| Selector/layout 漏迁移 | state/output 身份变化需覆盖 hash、equality、诊断、artifact key 与 plan 校验；新 16-bit panel 格式有独立 ID |
| KV 累积误差/失败发布 | 量化后的 K/V 在长上下文重复读取，需独立质量预算；维持全 plan transaction 与未提交区间不可读 |
| 并发 | 本提案不新增线程或仲裁；准备、Manager 变更和 Execute 继续外部串行化。涉及 shared ownership/同步实现时，补充相应 sanitizer/TSAN profile |
| 内存与性能推广 | W 从 4 bytes 改为 2 bytes 仅使该部分逻辑 payload 减半；KV、activation、workspace、padding、mmap 与 packed 副本分别测量，不承诺整进程内存或延迟减半 |
| 关联提案 | 08 负责 KV 生命周期/资源，本文补充精度执行合同；10 的入口在对应 dtype 验收后扩展；09 的模型家族/MoE 和专家布局需要独立接入证据，不构成本提案首批前置 |

## 6. 验收标准

### 6.1 数值与语义

- conversion 覆盖正负零、所有类别的 subnormal/normal、边界邻值、RNE ties、最大有限值、overflow、Inf、qNaN/sNaN；对 65,536 种 16-bit 输入做 widening 分类/有限值正确性检查，NaN 按声明合同验证。
- narrow 与 SIMD 路径对拍独立 reference，不能只调用同一个 Half/BFloat16 helper 生成 expected；P1 纯 widening 的测试不以尚未迁移的 narrow round-trip 为 oracle。
- 每个算子记录支持的 input/weight/output/state tuple、累加 dtype、stride/alias/tail/zero-size/overflow；合法 tuple 的 semantic inference、constant folding（如适用）与 kernel 数值一致。
- W16/A32 增加能区分“保留 A32”和“先将 A32 转 BF16/FP16”的反例，防止优化偷偷改变乘数精度。
- P4 fusion 与 Cast placement 单独验收；与 unfused reference 对拍时按照冻结的中间舍入规则解释差异。

### 6.2 所有权、绑定与失败

- P1 默认全量 scope 的精确字节断言按测试名称保留，包括 [test_raw_weight_fp32_conversion.cpp](../../tests/unit/model/test_raw_weight_fp32_conversion.cpp) 的 `EnforcesFinalOwnedBudgetBeforeAnyPayloadAllocation`（cap 15 拒绝、16 接受，peak 24，final owned 16）和 `CountsRetainedFloat32BackingAlongsideConvertedPayload`（cap 39 拒绝、40 接受，final owned 40）。
- P2 单列 Norm-only 账本。一个最小 owned fixture 让全部矩阵角色共用 4 个 W16 元素的 8-byte owner，待转换 Norm 角色共用 2 个 W16 元素的 4-byte owner，已有 F32 Norm 使用 8-byte owner。断言唯一 converted source=4 bytes、converted payload=8 bytes、final owned=24 bytes、peak owned=28 bytes，cap 23 拒绝、24 接受，持久矩阵 itemsize=2 且原 view 保留。
- 共享 backing fixture 将上述矩阵与待转换 Norm 放入同一个 12-byte owner，已有 F32 Norm 仍有独立 8-byte owner。断言该 shared owner 按完整 footprint 计一次，final owned=28 bytes、peak owned=28 bytes，cap 27 拒绝、28 接受。Norm 间相同 source 仅转换一次，矩阵继续引用 source owner。
- 两种 W16 dtype 分别覆盖上述账本、未知 retained backing + 有限 cap 的拒绝、未知 source-only backing 的 peak completeness，以及 payload 分配前 callback/allocator 调用为零的失败断言。
- 正确释放 normalizer 输入、局部 helper 和请求对象后，ExecutableModel/plan 仍有有效 bytes 与 metadata；最后一个 owner 销毁才释放转换存储，无泄漏或重复释放。
- tied embedding/lm_head 在转换后保留共享关系；相同内容的不同 tensor 不被误合并；partial failure/OOM 不发布半成品。
- dtype、bytes、alignment、recipe 或 state binding 故意不一致时在执行前失败；不能通过修改 selector 绕过 validation。
- P3 检查 graph state、selector、view/layout 与 binding 一致；完整 plan 失败不推进 watermark，epoch/owner/capacity 合同不回归。
- 用相关 ASAN/UBSAN profile 检查存储与转换；涉及同步/shared-owner 代码变化时补 TSAN，不以纯文档检查替代运行证据。

### 6.3 Production 与模型质量

- 使用真实 CpuBackend，覆盖 `Load → Normalize（按 preset）→ Compile → Prepare → CreateSession → Generate`，不以 fake backend 或手工绕开 binding 的单测代替。
- F16/BF16 分别覆盖 hint 缺失/auto/不一致、tied/untied lm_head、合法 Norm dtype、plain/packed、Prefill、多次 Decode、EOS 和容量失败。
- 比较基于同一 checkpoint 数据解码的 FP32/FP64 oracle；按算子、phase 与 sequence length 记录 max abs/rel error、logits/top-k、必要的模型质量指标，并在实现准入前固定误差预算。
- 不把低精度结果与原始高精度 checkpoint 的逐 token 完全一致作为通用要求；near-tie 按 top-1 margin/约定处理，固定 golden 可要求指定 token 序列。未经评审不临时放宽容差。
- 相同平台与相同 preset 的确定性，以及现有 Decode 稳态零分配合同必须有运行证据；不将其扩大为所有 Generate 初始化路径零分配。

### 6.4 性能、资源与发布状态

- 分别记录转换/packing/准备耗时、峰值与稳态 RSS、持久权重/packed/KV/activation/workspace 字节、Prefill TTFT 和 Decode tok/s。
- 覆盖 Prefill/Decode、hot/streaming、代表性维度与 context depth；同机同配置比较。默认 priority 提升需要 registered/prepared、production ExecutionStep 与端到端证据，microbenchmark 只作诊断。
- 证据按 [算子工作流](../guides/operator-development-workflow.md) 写入对应算子机器记录及 raw artifact，不在本提案填入未经验证的加速倍数或占位 benchmark 文件。
- 每个 preset、phase、layout、ISA/fallback 单独标记状态；只有实际通过相应门禁后，才同步更新模块设计、算子合同/API 文档、产品 capability 和行为变更记录。该 Draft 不修改 PRD 的已交付状态。

## 7. 一手资料

以下资料用于数值与指令机制，不作为 AetherMind 已实现或性能收益的证据：

- [Intel BF16 格式与 AVX512_BF16 指令说明](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html)：BF16 指数/有效精度、BF16 乘数与 FP32 dot accumulation。
- [Intel AVX-512 FP16 技术指南](https://builders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide)：FP16 转换与原生算术能力；具体指令数值需按合同验证。
- [GCC Half-Precision 文档](https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html)：语言类型支持与硬件代码生成是不同能力；中间精度及编译选项可能影响结果。

## 8. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-20 | 1.0 | 建立 Draft：FP32 兼容路径、W16/A32、独立 KV16 与 A16 阶段；明确职责、所有权、数值/ISA、selector/layout 迁移和验收门禁 |
| 2026-10-29 | 1.1 | P1 转为 In Progress：记录显式 F16/BF16→FP32 normalizer、FP32 graph policy 和 C++ O1/plain 编排；限定模型与 sanitizer 证据另记验证报告；P2–P4 仍为未来工作 |
| 2026-11-09 | 1.2 | Half narrowing 迁移为 RNE + gradual underflow，记录 quiet NaN 与 reduced-float 逐位取负合同 |
| 2026-11-10 | 1.3 | 补充 FP16 checked batch、F16C widening、AVX2 integer narrowing 及 P1 Backend callback 接入 |
| 2026-11-10 | 1.4 | 补充 BF16 双向 scalar batch、prepared 函数表和 P1 callback，记录当时已完成的转换基础 |
| 2026-11-11 | 1.5 | 对齐 BF16 AVX2 现状，冻结 P2 hint、Embedding output/builder/serde、retained backing 与 W16 args 合同，保留 P1 精确预算并增加 P2 账本验收，细化九步实施顺序 |
| 2026-11-11 | 1.6 | 实施并验收 P2 CPU O1/plain 与 O2/identity-packed，固定 F32 reference 数值门禁，接入 Norm-only/retained 账本、Embedding mixed output/行级 SIMD 和 W16 Linear/QKV/GateUp，记录生产路径与性能范围 |
| 2026-11-11 | 1.7 | P2 首批完成：O1/plain 与 O2/packed（identity）两个 W16 policy 限定验收通过（4675 项单测、ASan/UBSan/LSan、模型级账本与 EOS/容量/重试覆盖）；记录 O1 kernel/生产路径 baseline 与优化顺序决策；W16 B-panel 经证据评估延后；修复准入诊断消息合同（required Float32 显示名） |
| 2026-11-11 | 1.8 | 接入 Embedding F16C/AVX2 行级 widening，补充 exact selector、空输出和 legacy BF16 override 回归，扩展两种 policy 的三种 Norm source dtype 与 O1/O2 零分配检查，按当前工作树重新记录 Release、ASan/UBSan 和 LSan 环境范围 |
