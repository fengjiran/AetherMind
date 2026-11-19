# FP16 / BF16 执行支持方案

- **状态**: In Progress
- **版本**: 2.13
- **日期**: 2026-11-16
- **产品边界**: [当前产品 PRD](../product.md)；CPU-only、单进程、同步、单请求、Token IDs I/O、Llama Dense、静态连续 KV
- **架构边界**: [AGENTS.md](../../AGENTS.md) §2.1
- **关联计划**: [06 系统能力路线图](06-system-capability-evolution-roadmap.md)、[07 模型准备入口](07-executable-model-preparation.md)、[08 KV/Attention 演进](08-kv-cache-and-attention-capability-evolution.md)、[09 模型家族扩展](09-qwen-and-moe-architecture-support.md)、[10 Runtime 组装入口](10-upper-level-runtime-assembly.md)
- **算子工作流**: [算子开发与优化工作流](../guides/operator-development-workflow.md)
- **关联模块**: dtypes / model / operators / graph / compiler / backend / execution / kv_cache / runtime / inference

本文是阶段化提案。P1 的显式 FP32 compatibility 已完成限定实现和 tiny checkpoint 验证，见 [FP32 兼容推理指南](../guides/fp32-compatible-inference.md) 与 [P1 验证报告](../archive/tests/fp32_compatibility_validation_2026-10-29.md)。P2 首批 CPU/O1/plain 与 O2/identity-packed、W16/A32/KV32 已完成限定验收，入口见 [W16 指南](../guides/w16-fp32-inference.md)，源码身份与 sanitizer 范围见 [P2 验收记录](../archive/tests/w16_fp32_execution_validation_2026-11-11.md)。P3 步骤 1–10 核心已完成：首批合同（§2.7–2.8）冻结与独立校准、mixed-KV 语义、`state_dtype` 身份迁移、布局账本、KV16 读写 reference、plan 事务、显式 `kv_dtype` 配置链、reduced fixture 四组合模型验收、§2.8 冻结预算校准 fixture（2048）正式生产验收、ASan/UBSan、prepared-operator baseline 与文档收口。KV16 在 CPU O1/O2 上端到端可执行并通过冻结预算验收；production ExecutionStep/模型级 baseline 与 SIMD 优化为后续 refinement；C ABI/CLI 保持 strict F32。P4 步骤 1–6 已落地（§4.5）：ISA 探针与 `kAvx512Fp16` detection、独立校准与冻结预算（含校准器与冻结放置对齐的修正）、Cast/`output_dtype`/`activation_dtype` 基础设施、A16 kernel 层（O1 reference 全算子集 + A16 Linear/RmsNorm 的 AVX2 widening fallback 与 AVX-512 native）、BF16 全同型模型级验收（reduced fixture + 2048 near-limit 校准 fixture、ASan/UBSan、零分配、两次独立进程重复性、负例），以及经官方 runner 校验的 baseline（A16 preset 本机 decode 3.8× / prefill 5.1×，但归因于 AVX2 kernel 路径；无 AVX-512 硬件，native 收益未验证）。FP16 A16 的模型级执行已按 P4-B 在 CPU O1/O2 完成限定验收，见 §4.6。新硬件 native 数值证据（步骤 7）待执行。

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
| `BuildLlamaDense` 接受显式 activation/KV dtype；W16 Load 指定 A32/KV32，Embedding 显式输出 F32 | [Llama builder](../../src/model/llama_dense_graph_builder.cpp)、[Load](../../src/inference/executable_model.cpp)、[Embedding inference](../../src/operators/ops/embedding_op.cpp) | P3 复用 graph-build options；公开 KV 精度选项必须与 Runtime 物理布局一致，不从源 hint 推测 |
| Linear/RMSNorm 语义分别校验 activation 与 weight，输出跟随 activation；Attention/KVUpdate 要求 K/V cache 同 dtype，activation 可为 F32 或与 cache 同型 | [Linear inference](../../src/operators/ops/linear_op.cpp)、[RMSNorm inference](../../src/operators/ops/rmsnorm_op.cpp)、[Attention inference](../../src/operators/ops/attention_op.cpp)、[KVUpdate inference](../../src/operators/ops/kvcache_update_op.cpp) | W16/A32 有现成扩展基础；mixed KV 语义已随 P3 扩展，新增 mixed 方向须按角色重新准入 |
| P2 W16/A32/KV32 的 plain/identity-packed 链已限定验收；P3 已补齐 CPU KV16 读写 reference、模型验收与校准 fixture | [P2 验收](../archive/tests/w16_fp32_execution_validation_2026-11-11.md)、[P3 验收](../archive/tests/kv16_execution_validation_2026-11-12.md)、[Attention entry](../../src/backend/cpu/kernels/attention/attention_entry.cpp)、[KVUpdate entry](../../src/backend/cpu/kernels/kvcache_update/kvcache_update_entry.cpp) | KV16 支持以独立读写 reference 和完整模型证据为准，不因 Manager 接受两字节元素就获得执行支持 |
| `KernelSelector` 严格匹配 activation/weight/state dtype；`DeriveSelectorDTypes` 从 state 端口派生 state dtype，尚未独立表达 output dtype | [selector](../../include/aethermind/base/kernel_selector.h)、[dtype 派生](../../src/operators/operator_inference.cpp) | mixed KV 已用 `state_dtype` 表达身份；后续 mixed output 才扩展 output 匹配身份 |
| identity packing 保留源 dtype；现有 B-panel 只有 `kCpuBPanelF32Kc512Nr16`，packer/validator 限定 F32 | [CPU packing](../../src/backend/cpu/cpu_backend.cpp)、[recipe](../../include/aethermind/backend/packing_recipe.h)、[packed validation](../../src/backend/cpu/packed_weight_validation.cpp) | 16-bit B-panel 需要独立物理布局身份，不复用 F32 ID |
| `Half(float)` 软件标量转换使用固定 RNE + gradual underflow，双向转换 quiet NaN；BF16 转换使用 RNE 并保留 sign 与截断后的 NaN payload | [Half 转换](../../src/dtypes/half.cpp)、[BF16 转换](../../src/dtypes/bfloat16.cpp) | Half 的 FTZ 行为已独立迁移；软件转换的合同和测试不代表 native 16-bit kernel 已支持 |
| CPU capability 已检测 F16C、AVX512BF16、AVX512FP16、AMX_BF16 等，尚无对应完整 kernel 链；现有编译配置分别隔离 FP32 AVX2/FMA、FP16 F16C widening、FP16/BF16 AVX2 integer conversion，并已加入 AVX512-BF16/FP16 编译器探针 | [CPU capability](../../include/aethermind/backend/cpu/cpu_capabilities.h)、[检测](../../src/backend/cpu/cpu_info.cpp)、[库 CMake](../../src/CMakeLists.txt)、[ISA CMake](../../cmake/AetherMindCpuIsa.cmake) | feature detection 与编译探针不是执行支持；新增 ISA 按 translation unit 隔离 |

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
| 16-bit KV | 已验收的 P2 权重组合 | F16/BF16 | F32 | 显式选择 F16 或 BF16，K/V 同型 | F32 / F32 | P3 CPU O1/O2 限定验收通过（KV16 对同 W16 KV32） |
| 16-bit activation | 后续按算子签名批准 | 按 preset 决定 | 显式边界；BF16 preset 全 A16 residual，FP16 preset 保留 F32 residual | 独立选择已验收组合 | 至少 F32 / F32 | P4 首批设计冻结（§4.5），ISA 基础已落地，实现进行中 |

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
3. 不按 hint 或 ISA 隐式改变精度；P2 不进行 F32→16 activation 转换，也不承诺任意 mixed linear dtype。P3 仅在显式选定的 KV 写入边界执行 F32→16，在读取边界 widening；不因 W/KV 类型不同而转换矩阵权重。其他转换须有明确策略并单独验收。
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

**P2 W16 Linear args 合同（当前实现）。** 现有 [LinearA32W32KernelArgs](../../src/backend/cpu/kernels/linear/linear_internal.h) 持有 `const float* weight`。新增 CPU-private `LinearA32W16KernelArgs`，由注册的 params builder 一次构造。

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

### 2.7 P3 首批执行合同（步骤 1 冻结；实现与验收见 §4.4）

本节定义并冻结实现必须满足的合同；当前实现与限定验收状态见 §4.4 和 [P3 验收记录](../archive/tests/kv16_execution_validation_2026-11-12.md)。算子细节分别归 [KVCacheUpdate](../operators/kv_cache_update/README.md) 和 [Attention](../operators/attention/README.md)。

| 矩阵 W | activation / residual / logits | 所有层 K/V storage | 首批 preset 与状态 |
|---|---|---|---|
| F16 | F32 | F16 | O1/plain 与 O2/identity-packed 已接通 |
| F16 | F32 | BF16 | 同上 |
| BF16 | F32 | F16 | 同上 |
| BF16 | F32 | BF16 | 同上 |

W 与 KV 精度独立选择；同一 Runtime 的全部层 K/V 必须使用同一 scalar dtype。Norm 保持 P2 的准备后 F32，checkpoint hint 继续只作 provenance。KV32 保持既有 P1/P2 合同。首批 KV16 只对上述两个 W16 policy 开放；F32 矩阵与 KV16 的模型组合另行准入。C ABI 与产品 CLI 保持 strict FP32。

**配置合同（当前实现）。** 在 `CpuRuntimeAssemblyOptions` 和 `ModelLoadOptions` 末尾分别追加 `DataType kv_dtype = DataType::Float32()`，不扩张 `WeightPolicy` 来编码 KV 精度。通用 `MakeKVCacheOptions(config, context_tokens, kv_dtype)` 接受 scalar F32/F16/BF16，拒绝 undefined、scalable、其他类型及非法 lanes；现有 `MakeFp32KVCacheOptions` 保留为 F32 wrapper。显式选项分别驱动 RuntimeOptions 与 ModelGraphBuildOptions；KV16 Load 在权重转换和 packing 前验证已有 Manager 的实际 dtype 与请求相等，不修改或重建 Runtime。低层 Compile→Prepare/binding 路径也必须验证真实 state specs 与已有物理布局一致。新增 C++ 字段改变结构布局，消费者需重新构建。

**唯一转换边界。** K 在既有 F32 RoPE 完成后窄化一次，V 从 F32 projection 输出窄化一次。KVUpdate 的 state 输入/输出 alias 同一实际 cache plane；Attention 必须从 cache16 读取，再 widening 到 F32。Prefill、Decode 及当前 query 可见的当步 K/V 都经过该边界，不能在 Prefill 直接旁路读取未窄化 activation。持久 cache 不保留平行 F32 数据源；未来 tile scratch 必须走既有 workspace 规划，不能改变该语义。

```text
K32 → 既有 F32 RoPE → RNE narrow → K cache16 ┐
V32 ─────────────────→ RNE narrow → V cache16 ├→ widen → F32 Attention(Q32) → output32
                                           ┘
```

**reference 算术。** KV 转换完全复用已有固定 RNE、gradual underflow、signed zero、overflow→signed Inf、quiet NaN 且保留 sign/可容纳 payload 的合同，转换自身不改变浮点环境。Attention 使用 F32 score、scale、max、`exp`、denominator、value accumulation 和除法；按 head dimension 升序进行独立 F32 product/add，按可见 token 升序执行 max 与 softmax 两遍扫描、累加 weighted V，最后除以 denominator。新 reference TU 禁止 FMA contraction 和 fast-math，不改变已交付 F32 kernel 的数值合同。RoPE 保持既有更高精度频率/旋转计算与最终 F32 写回。

数值误差门禁使用 FE_TONEAREST、gradual-underflow 的 F32 环境（x86 FTZ/DAZ 关闭）。Attention 不负责修改/恢复调用方 FP 控制，不承诺算术异常 flags 不变；转换的 FP 环境保证不能推广为 Attention 保证。NaN/Inf 或 F32 中间溢出不触发额外 payload 预扫描、clamp 或有限结果补救；按固定算法的 F32/`std::exp` 传播检查分类，Attention 算术不承诺 NaN payload。含 +Inf 或全 -Inf 的 logits 不承诺有限 softmax，不能用零向量替代 NaN 来通过测试。模型误差/排名门禁要求输入与所比较的全部中间/输出有限。

**state 身份与物理绑定。** `state_dtype` 只从完整 schema 的实际 state 输入/输出 specs 推导，不受 `contributes_tensor_spec=false` 过滤；各 state port 同型，无 state 时为唯一 undefined，严格匹配。同步覆盖 lowering、lowered artifact 验证、registry、hash/equality/ToString、untrusted 重校验与包含 selector 的 artifact key。无 weight 的算子沿用 weight dtype fallback，不借用该字段表达 KV。当前 untrusted stateful node 限制保持。

F32 activation/query/output 使用元素单位正 stride 与实际 float 自然对齐；KV binding 的 stride 是字节单位，按真实两字节元素验证范围与 overflow，不沿用 `sizeof(float)`。生产基址仍遵守 layout alignment，byte-safe KV 读写不新增 float 对齐要求。KV padding 不参与转换/Attention。KVUpdate 两个只读 activation 可以共享 storage，但不得覆盖任一写入窗口；K/V destinations 互不重叠。Attention output 与 Q、可见 K/V 均不重叠。动态 KV 指针、append/read 区间和 query position 只借用于本次调用，不写入 prepared 参数。

**失败与资源。** 先校验本次访问所需 metadata、dtype、geometry、alias 与可表示范围，再读写。完整 plan 和所有层写入成功后才提交 watermark；失败允许未提交窗口的 bytes 已改变，但不得发布该窗口或破坏已提交历史，重试重写完整 append range。Manager 的 generation/reservation/capacity 与外部串行化合同保持。逻辑两字节 payload 与 padded physical bytes 分别记账，不要求所有 head_dim/alignment 下总 KV 恰好减半。

### 2.8 P3 数值门禁与独立校准

验收分成三个独立层次：

1. **转换位合同。** 以独立可表示值/nearest-even oracle 检查写入的每个 16-bit pattern 与 widening 位结果，覆盖全部 65,536 个源模式、F32 midpoint 邻点、零/subnormal/overflow/NaN。不得用被测 Half/BFloat16 helper 生成 expected。NaN 算术的 payload 不属于此位合同。
2. **KV16 实现正确性。** 独立 reference 必须在每层 post-RoPE K 和原始 V 的写入位置窄化再 widening，包括 Prefill 当前 token；保留相同 W16 checkpoint 与 Norm 数据。比较每阶段 Attention/模型 logits、各层 K/V、watermark，固定数值容差与 near-tie 规则后再实现 kernel。
3. **KV16 相对 KV32 的误差。** 同一 W16 checkpoint、同一 teacher-forced token/position 轨迹，对比 KV16 与 KV32。不能把权重精度误差并入 KV 误差，不能只比较最终 greedy token，也不能直接沿用 P2 容差。closed-loop Generate、EOS、重复请求另列测试。

校准必须覆盖 MHA tied/untied、GQA/MQA tail、短 Prefill+至少两次 Decode，以及 2046-token Prefill+两次 Decode 至 2048 的 near-limit case；四种 W×KV 组合分别记录。校准使用独立于 AetherMind 的算法，并先通过既有 FP32 HF golden 交叉验证。本次校准从固定 F32 source 的全部矩阵独立 RNE 派生数学 W16，Norm 保留原 F32；它不验证实际序列化 W16 checkpoint 的加载。步骤 9 必须消费真实 reduced checkpoint、记录 payload identity，并对相同 payload 生成 reference。生成的脚本/编译参数、输入与 oracle hash、环境和全部测量保存在 raw；校准不验证 KV16 production path、实际两字节分配、O1/O2 kernel 或性能，不更新 accepted FP32 assets/checksums。

**固定 finite 模型预算（已冻结，2026-11-12）。** 下表适用于本批 5 个 fixture、最长 2048-token 的 Prefill/两次 Decode、两种矩阵 dtype；每个 scalar 均须满足 `abs(actual-reference) <= atol + rtol * abs(reference)`，同时拒绝非有限比较值。K/V 以 decoded F32 数值比较。实现正确性对照固定量化边界、采用 FP64 归约后写回 F32 的独立 reference；质量对照采用同 W16 的 F32 计算/KV32 reference。

| 对照层次 | KV dtype | logits atol | logits rtol | K/V atol | K/V rtol |
|---|---|---:|---:|---:|---:|
| 实现正确性 | F16 | `1e-6` | `2e-5` | `2e-4` | `2e-3` |
| 实现正确性 | BF16 | `1e-6` | `2e-5` | `1.5e-3` | `2e-3` |
| 相对 KV32 的误差 | F16 | `1e-3` | `2e-3` | `4e-4` | `4e-3` |
| 相对 KV32 的误差 | BF16 | `5e-3` | `1e-2` | `2e-3` | `2e-2` |

不同归约的上游 F32 微差可能跨 KV16 midpoint，使 cache 数值相差一个离散间隔；因此模型 K/V 不要求逐位相等，**单次转换在相同输入位模式下仍必须精确匹配独立位 oracle**。模型容差不能用于放宽 converter 位合同。误差预算按 phase、context、W/KV 组合分别报告最大绝对/相对误差和最大归一化误差，不能只报告合并平均数。

**固定排名规则。** 对相应对照层次，先从 reference logits 和本表预算计算每行 `epsilon = max_i(atol + rtol * abs(reference[i]))`。top-1 margin（第一与第二名差）及 top-4 boundary margin（第四与第五名差）分别大于 `2*epsilon` 时，该行对应的排名集合必须一致；否则作为 near-tie 记录，仍须通过全部 scalar 数值门禁。classification 绑定 reference identity、预算和 row，在检查候选前固定，不使用候选自身实测误差重新分类。P2 的历史 margin 规则不在本步迁移。位置、watermark、logit row identity 必须精确一致；固定 HF golden 自检还要求全部已保存 token IDs 一致。独立模型的 short closed-loop greedy 结果单独记录，不能用 teacher-forcing 代替 production Generate/EOS 验收。

预算在新的 production kernel 实现前冻结。校准结果、重复性和比较器 negative checks 完整记录后，才将步骤 1 标为完成；候选失败不能通过临时放宽阈值、改 reference 或动态重分类绕过。更大 context、不同模型/geometry 或更改数值算法需新增独立校准和明确范围，不能由本批预算推导通用模型质量保证。

**预算冻结依据（2026-11-12，步骤 1 完成）。** 上表预算已据独立校准冻结。校准器 [calibrate_kv16_reference.py](../../tools/calibrate_kv16_reference.py)（sha256 `0c1b0be328c4dcb2453631ec27b125138119dad97df652cc48eccffb8affaba6`）以 pinned HF Llama 实现与整数 RNE oracle 独立于 AetherMind：转换器有限域逐位匹配独立 oracle（F16 63,488 / BF16 65,280 个模式、各约 9.7 万 midpoint 邻点、20 万随机模式），比较器 11 项 negative checks 通过；FP32 模式对 5 个 accepted golden case 逐位复现（max_abs=0，含 2046→2048 near-limit）。四种 W×KV 组合 × 5 fixture 共 20 次运行全部通过两层预算：实现正确性最大归一化误差 0.298（logits max_abs 2.83e-7、K/V max_abs 4.88e-4），相对 KV32 最大归一化误差 0.567（logits max_abs 3.58e-3、K/V max_abs 4.10e-3）；near-limit 出现预测的 KV16 midpoint 跨越（impl K/V 非零），预算均吸收，质量层 near-tie 分类被实际触发且不误判。两次独立进程校准的 65 个规范产物逐位一致。raw 证据（脚本/环境/输入与 oracle hash/全部测量/重复性报告）见 gitignored `benchmark-results/models/kv16-calibration/20261112T053100Z_592df0e4-dirty_desktop-54h5mmi_kv16-step1/`。校准不验证 production path、实际两字节分配、O1/O2 kernel 或性能；torch bf16 narrowing 的 NaN 规范化不在有限校准域内，NaN 位合同归步骤 5/6 的生产转换器验收，真实序列化 W16 checkpoint 加载归步骤 9。

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

首批步骤 1–9 已完成 CPU/O1/plain 与 O2/identity-packed、W16/A32、KV32。步骤 8 已记录 reference baseline，并接入经 feature policy 选择的 Embedding F16C/AVX2 行级 widening。CPU-02 已接入 W16/A32 SIMD/Decode/small-M/Prefill blocking 与 plain/identity 生产路径，见 [Linear 方案](../operators/linear/linear-optimization.md) 及其机器证据；reference 独立保留，新 W16 B-panel 仍须后续 profiling。

1. **冻结支持矩阵与数值合同。** 落实 §2.1 的 hint 白名单、不一致策略与按角色区分的 actual dtype 准入，明确 P2 F32 累加、reference 的 FMA contraction 策略、误差预算及 near-tie 规则。同步维护本提案版本与变更记录。
2. **扩展 inference 准入。** 在 [model_execution_policy.h](../../include/aethermind/inference/model_execution_policy.h) 增加 F16/BF16 两个 W16 策略，矩阵严格匹配所选类型，Norm 源接受 F32/F16/BF16。验证 unknown policy、target、preset、known-hint mismatch 和 actual-dtype mismatch 的各分支。
3. **完成 Norm-only transformation 与预算。** 在 [raw_weight_fp32_conversion.h](../../include/aethermind/model/weight/raw_weight_fp32_conversion.h) 的 options 增加 `RawWeightFp32ConversionScope::{kAllRequiredWeights,kNormWeightsOnly}`。转换器按角色选择 FinalNorm、InputNorm、PostAttentionNorm，复用两套 widening callback 与 SourceKey 去重。依 §2.4 重做 retained 选择，更新 cap 注释/错误文案，并通过 §6.2 的 P1/P2 独立账本和失败回滚测试。
4. **完成 Embedding 语义和 builder/serde 迁移。** 修改 [op_params.h](../../include/aethermind/operators/op_params.h)、[embedding_op.cpp](../../src/operators/ops/embedding_op.cpp)、[graph_op_builder.cpp](../../src/graph/graph_op_builder.cpp)、[llama_dense_graph_builder.cpp](../../src/model/llama_dense_graph_builder.cpp) 与 [op_params_serde.cpp](../../src/operators/op_params_serde.cpp)。尤其要将 builder 的 activation-vs-weight guard 改为 activation-vs-inferred-output 比较，按 §2.2 保持裸 serde 的逐字节兼容并扩展 0/1 字段解析。退出门禁包括 operator inference、serde、graph-build 和 selector dtype 派生回归。
5. **实现 O1/plain reference kernel。** Embedding 按选中 W16 row 写 F32 output，Linear/LM head 使用 §2.5 的专用 W16 args 和 F32 累加。分别注册 `{A=F32,W=F16}`、`{A=F32,W=BF16}`，覆盖 Prefill/Decode、合法 stride、非对齐、尾部、零 M/N/K、alias 和执行期零分配。
6. **接通公开 Load 链。** [executable_model.cpp](../../src/inference/executable_model.cpp) 编排 source 准入→Backend widening 选择→Norm-only conversion→prepared 角色校验→Compile(A32/KV32)→Prepare→Session。转换后校验 Norm=F32、矩阵=W16，复用原有 owner 链，分别闭环两种 policy。
7. **完成 P2-O1 模型级验收。** 对同一 reduced checkpoint 比较 P2 与 P1，按 §6 的 logits abs/rel、top-k、top-1 margin 验证 Prefill、多次 Decode 和 Generate。覆盖 MHA/GQA、tied/untied、EOS/容量、OOM 重试、owner lifetime、持久矩阵两字节存储与 Decode 稳态零分配，执行相关 sanitizer profile。
8. **采集 O1 baseline 并优化。** 记录 registered/prepared、production ExecutionStep 与端到端准备/生成指标，新增 scratch 经现有 workspace 规划链提供。**当前状态**为 [Linear baseline](../operators/linear/benchmarks/desktop-54h5mmi-linear.md)、[Embedding 行级 SIMD 对照](../operators/embedding/benchmarks/desktop-54h5mmi-embedding.md) 和 [模型级记录](../archive/tests/benchmarks/models/desktop-54h5mmi-model.md) 已完成。Embedding 复用既有 F16C/AVX2 converter，完整位模式、feature-policy fallback、O1/O2 轨迹与零分配已回归。Linear 后续优化由其 baseline 和 profiling 决定。
9. **完成 P2-O2/packed。** 按预期 dtype 和 itemsize 验证 identity packed weight，补齐 W16 Embedding/Linear、QKV/GateUp composite、reference、参数准备与绑定，回归 AddRmsNorm 等实际 fusion。完整 O2 图通过后，再依据性能证据决定是否新增 W16 B-panel 及其独立布局身份。**状态：identity 批次已完成**，O2 preset 对两个 W16 policy 开放；W16 B-panel 评估为延后——baseline 证明瓶颈在 scalar widening 而非布局，identity 已保持 2 字节 composite payload，新布局须待 widening 优化后的独立证据（决策记录见 Linear 基准记录）。

步骤 1 完成后，步骤 2/3/4 可以并行推进。步骤 5/6 汇合语义、权重与 backend 实现，步骤 7 是 O1 批次退出门禁，步骤 9 另有 O2 退出门禁。

退出条件为两个 W16/A32 policy 均完成生成，持久矩阵保留 16-bit，plain/packed 支持范围分别记录，Decode 稳态分配合同通过。**首批达成**范围包括 tiny reduced checkpoint 的 O1/O2 Prefill/Decode/Generate、EOS、容量边界、失败重试和 owner lifetime，模型账本确认矩阵两字节且转换字节仅来自 Norm。当前工作树的 ASan/UBSan 与独立 Release malloc-family 门禁见 [验收记录](../archive/tests/w16_fp32_execution_validation_2026-11-11.md)，其中明确 LSan 的环境限制。

### 4.4 P3：独立 16-bit KV

步骤 1 的合同见 §2.7–2.8；它的完成与后续 KV16 可执行状态分开记录。步骤 1–10 核心已完成：合同冻结与独立校准、mixed-KV 语义、state_dtype 身份迁移、布局账本、KV16 读写 reference、plan 事务、kv_dtype 配置链、reduced fixture 四组合模型验收、§2.8 冻结预算的 2048 near-limit 正式验收、ASan/UBSan 与 prepared-operator baseline。production ExecutionStep/模型级 baseline 和 SIMD 优化为后续 refinement，见 [P3 验收记录](../archive/tests/kv16_execution_validation_2026-11-12.md)。

| 步骤 | 交付与门禁 | 当前状态 |
|---|---|---|
| 1. 冻结合同 | 四种 W×KV 组合、O1→O2 顺序、配置、转换位置、reference 数值/alias/失败合同，以及独立校准后的固定数值预算 | 完成：合同冻结、独立校准与重复性通过、预算冻结（§2.8） |
| 2. 算子语义 | 扩展 Attention A32/KV16 和 KVUpdate A32→KV16；保留原同型合同，state 输出保持原 cache alias | 完成：mixed inference 接受/拒绝 + 原有组合回归；KV16 kernel 已随步骤 5/6 接通 |
| 3. state 身份 | selector/derivation/lowering/lowered validation/registry/key/untrusted 校验完整迁移；无 state 严格使用 undefined | 完成：全链路迁移 + 严格匹配/篡改拒绝回归 |
| 4. 布局与参数 | 复用 dtype-aware KVCacheLayout，补齐两字节账本、padding/cap/overflow/OOM；动态指针与位置不进入 prepared args | 完成：dtype-aware 布局复核 + 两字节账本回归（lifetime 保持） |
| 5. KVUpdate reference | 两种 narrow 写入、访问范围和 alias 校验、padding/历史保护、真实 prepared 路径零分配 | 完成：F16/BF16 窄化写入 + 已知位模式 oracle、padding/history/alias/stride、dtype 拒绝、零分配 |
| 6. Attention reference | 两种 cache widening、F32 固定归约、GQA/MQA/causal/read visibility，独立 oracle 与特殊值分类 | 完成：F16/BF16 widening + F32 固定归约（FMA off）+ double oracle、GQA/MQA/decode、dtype 拒绝 |
| 7. plan 事务 | 每层写入证明、完整 plan 成功后 commit、失败不发布与重试；复核 generation/capacity/stale view | 完成：LayerRunner dtype-agnostic 无需改；KV16 prefill/decode/多层/缺层不提交/update-then-attention/后续失败不发布回归（F16/BF16） |
| 8. 公开 C++ 链 | kv_dtype 配置统一驱动 Runtime/graph，Load 早期一致性校验，低层 Prepare/binding 不可绕过 | 完成：kv_dtype 驱动 Runtime/graph + Load 早期 Manager dtype 校验 + MakeKVCacheOptions 通用/F32 wrapper；KV16 Load/Generate 闭环与拒绝回归；C ABI strict F32 保持 |
| 9. 模型验收 | 先 O1 再 O2；四种组合的 Prefill/多次 Decode/Generate、近容量长上下文、预算、EOS/失败/零分配 | 完成（reduced fixture）：四组合 O1/O2 + 近容量 14→16 + GQA tied，KV16 vs KV32 实测容差 + 固定排名规则 + Generate/重复 + Decode 模型级零分配（F16/BF16×O1/O2）；§2.8 冻结预算的校准 fixture（2048）正式验收已通过（见步骤 10） |
| 10. baseline 与收口 | prepared/production/model 资源与性能证据、适用 sanitizer/fallback、API/指南/验收状态；SIMD 根据证据后续推进 | 完成（核心）：文档同步；ASan/UBSan 通过；§2.8 冻结预算校准 fixture（2048）正式验收通过；KVUpdate/Attention prepared-operator baseline 采集（raw + 算子记录）。production ExecutionStep/模型级 baseline 与 SIMD 优化为后续 refinement |

依赖为 `1 → 2/3/4 → 5/6 → 7`，配置接口可并行设计，集成后完成 `8 → 9(O1→O2) → 10`。新 reference 实现前必须先完成步骤 1 的预算校准；校准器不实现或替代 production kernel。

退出条件：Prefill/Decode、capacity 边界、长上下文误差与真实 byte accounting 通过；不能仅凭 Manager 能分配两字节元素就声明 KV16 可执行。

### 4.5 P4：16-bit activation 与 native compute（首批设计已冻结）

本节记录 P4-A 的首批合同和验收快照。本节所述 FP16 activation 的 `Unimplemented` 仅适用于当时的 P4-A。FP16 A16 的现行准入与独立验收见 [P4-B](#46-p4-bfp16-计算与-fp32-residual)。

P4 首批（P4-A）冻结 A16 合同，并把 native ISA kernel 纳入主线实现（无单独准入阶段）；C ABI/CLI 保持 strict F32。本机无 AVX-512/AMX（仅 avx2/f16c/fma），native 代码经"编译器探针 + per-TU 隔离 + runtime `effective_features` 过滤"机制随主线交付：新硬件到位后跑同一套 P4 验收套件补齐硬件数值证据；逐 ISA 状态按 §6.4 分别标记，属状态记账而非准入壁垒。BF16 先行理由为指数域安全（activation/residual 动态范围）与 BF16 checkpoint 生态，非 `VDPBF16PS` 成熟度（那是未来硬件理由）。

**支持矩阵与配置。** 首批仅 2 种同型三元组（A=W=KV）：F16、BF16；4 是 P3 的 W×KV 交叉数，不适用。P4-A 的**可执行** preset 限定为 BF16 全同型；F16 的 A=W=KV kernel 已随本批实现并通过 kernel 级验收，但模型级执行要求 residual 保持 F32（P4-B）并需要 residual 边界的 Cast kernel，故 `Load` 对 F16 activation 显式返回 `Unimplemented`。KV 维度显式声明：首批 KV=A16 同型，A16×KV32（KVUpdate widening 新 mixed 方向，P3 未覆盖）不在首批。BF16 preset 为全 A16 同型（A=W=KV=residual=BF16，对齐业界 W16A16 主流）：残差流跨层 BF16 持久，Add 结果即新的跨层 hidden、无 F32 副本，残差 Cast 消除。FP16 preset（P4-B）residual 仍强制 F32（跨层累积溢出风险），全 F16 residual 须独立校准后另评。A16 preset 仅与同 dtype W16 policy、CPU、O1/plain、O2/packed 组合；`ModelLoadOptions.activation_dtype` 独立字段驱动，不扩张 `WeightPolicy`（§2.7 先例）。既有 policy 枚举名含 `Fp32Activation` 为历史命名：A16 复用时诊断按实际 `activation_dtype` 生成，不按枚举名声称 F32，enum 不重命名。替换 [executable_model.cpp](../../src/inference/executable_model.cpp) 的 activation 硬编码。

**逐端口 rounding placement（冻结合同）。** 以 BF16 preset 为例；每个窄化边界进入校准模型。

| 位置 | I/O dtype | 舍入 |
|---|---|---|
| Embedding | W16 → A16 | 0 次（同型 gather，undefined 跟随 weight 路径，无白名单/serde/builder 改动） |
| RMSNorm | A16 I/O，F32 累加 | 输出窄化 1 次 |
| QKV/GateUp/Down Linear | A16×W16→A16 | 输出窄化 1 次（Q/K/V 各） |
| RoPE（Q/K） | A16 I/O，F32 三角/旋转 | 输出再窄化 1 次 → **Q/K 合计 2 次舍入**（区别于 P3 §2.7 的 K 单次窄化，显式冻结） |
| KVUpdate | A16 → A16 cache | **直写 copy，0 次**（禁用 P3 narrow kernel，防双重窄化） |
| Attention | A16 QKV，F32 score/softmax/累加 | 输出窄化 1 次 |
| Add residual | BF16+BF16→BF16（同型，无 Cast） | 0 次；中间精度 = widen→F32 add→RNE narrow（新增合同点 1） |
| SiLU/SiluMul | A16 | 各 1 次 |
| LMHead Linear | A16×W16 → **F32** | 0 次（禁止先窄化 logits 再 widening，§4.5 原步骤 2） |
| Softmax/Argmax | F32 | 不变 |

residual Add 零新增：BF16 语义白名单与 CPU reference 已存在（[add_reference.cpp](../../src/backend/cpu/kernels/add/add_reference.cpp) 的 `ExecuteTyped<BFloat16>`），[BuildAddArgs](../../src/backend/cpu/kernels/add/add_entry.cpp) 的 lhs/rhs/output 同型校验直接成立。F16 Add 仅 P4-B 选 F16 residual 时需要（P0 步骤 3 预留），不在首批。

**新增冻结合同点。** ① residual Add 中间精度：`BFloat16 operator+` 为 widen→F32 add（一次 F32 舍入）→RNE narrow（一次 BF16 舍入），已由 [bfloat16.cpp](../../src/dtypes/bfloat16.cpp) 与 Add reference 共同成立，校准 reference 必须与之逐位一致；② 残差流跨层 BF16 持久：Add 结果即新 hidden，无 F32 副本，这是全 A16 的定义；③ K 双舍入、Attention 输出舍入、LMHead F32 输出边界不变。

**端口签名合同。** [KernelSelector](../../include/aethermind/base/kernel_selector.h) 增加 `output_dtype`：由 `DeriveSelectorDTypes` 从真实 output specs 派生 concrete 值、严格相等匹配，无 undefined wildcard——"输出跟随 activation"是 `InferXxx` 的推导规则，不是匹配规则；每个 step 都有输出，selector 未派生前不参与匹配。迁移面覆盖 hash/equality/ToString、registry、lowering、lowered 验证、untrusted 重校验与含 selector 的 artifact key；全部既有 descriptor 与测试 selector 补 concrete output_dtype（对齐 P3 补 `state_dtype` 的回归量）。单字段适用性：盘点 [op_type.h](../../include/aethermind/operators/op_type.h) 与 Llama 图实际算子集，多输出算子仅 kAddRmsNorm（builder 直接发射，见 [llama_dense_graph_builder.cpp](../../src/model/llama_dense_graph_builder.cpp)）；全 A16 residual 下其双输出同为 BF16、单字段可覆盖——首批以未融合 Add + RmsNorm 为保守基线（A16 构图分支），AddRmsNorm 的 A16 融合按 O4 证据另行批准，异构多输出情形仍归后续批次。Attention/KVUpdate 同型语义已接受（白名单含 F16/BF16），首批只补 kernel，无语义改动。

**基础设施（步骤 3 已完成）。** typed Cast 算子已落地：`OpType::kCast` + `CastParams` + `InferCast`（仅 F32/F16/BF16 六个跨型方向，同型显式拒绝，输出形状恒等）、schema 标记 `CompileTimeEvaluable`、serde canonical `Cast target_dtype=float32|float16|bfloat16`、`AddCast` builder、以及 flat/strided 常量折叠 evaluator（经 float 中转复用冻结转换位合同）；按首批范围约定，backend kernel 尚未注册（P4-A 图流无 Cast 消费者）。[KernelSelector](../../include/aethermind/base/kernel_selector.h) 的 `output_dtype` 已在 base/operators/compiler/execution 全链路完成严格派生与校验（无 wildcard，state-only 步骤为 undefined），全部既有 descriptor 与测试 selector 已补 concrete 值。[Linear](../../src/operators/ops/linear_op.cpp) 增加 `LinearParams.output_dtype` override（undefined=跟随 activation；serde 裸 `Linear` 保持逐字节兼容），A16 构图时 LMHead 显式输出 F32、F32 preset 仍走 undefined。[ModelLoadOptions](../../include/aethermind/inference/executable_model.h) 增加 `activation_dtype`（默认 F32）并在 Load 早期校验：标量白名单、A16 仅同 dtype retained-W16 policy、CPU O1/plain 或 O2/packed；诊断按实际 dtype 生成，policy 枚举名不参与精度声明。

**校准先行。** 独立 reference 建模整条量化数据流（每个边界窄化含 Q/K 双舍入、**BF16 residual 跨层累积**——残差流每层量化、预算须吸收跨层残差舍入，比 F32-residual 版更严、F32 累加、LMHead 直出 F32），非 KV 量化器扩展；预算全新冻结、不复用 P3 数值；先于一切 production kernel（含 native reference）。回退（AVX2 widen，K 升序）与 native（`VDPBF16PS` 内部分组累加）顺序不同：二者各自对 FP64 oracle 独立满足冻结预算，不要求相互逐位一致；ISA kernel 独立冻结预算，不能只引用 reference 的 `gamma(2K)` 门禁。F16/BF16 有效数乘积至多需要 22/16 bit，但指数范围仍可能导致 F32 underflow/overflow；不能仅凭有效位宽证明 native BF16 与 widening 等价。

**A16 冻结数值预算（2026-11-13，步骤 2 完成）。** 下表覆盖 5 个 fixture（含 2046→2048 near-limit）× 2 种同型组合；每个 scalar 须满足 `abs(actual-reference) <= atol + rtol * abs(reference)`，并拒绝非有限比较值；K/V 以 F32 数值比较。

| 对照层次 | A16 dtype | logits atol | logits rtol | K/V atol | K/V rtol |
|---|---|---:|---:|---:|---:|
| 实现正确性（vs F64 同构参考） | F16 | `1e-4` | `1e-3` | `2e-4` | `2e-3` |
| 实现正确性（vs F64 同构参考） | BF16 | `2e-4` | `2e-3` | `2e-3` | `2e-2` |
| 相对 W16/A32（KV32） | F16 | `2e-3` | `6e-3` | `1e-3` | `1e-2` |
| 相对 W16/A32（KV32） | BF16 | `1e-2` | `3e-2` | `5e-3` | `3e-2` |

预算冻结依据（2026-11-13 修正版）：校准器 [calibrate_a16_reference.py](../../tools/calibrate_a16_reference.py)（sha256 `11adc1b141ae5e4da6f6132aebbdf4c42be3f995e35831dc55486d3d50ce6d74`）独立于 AetherMind，从零实现 Llama 前向并在每个冻结边界显式量化；先与 accepted FP32 HF golden 交叉验证（5/5 通过，logits max_abs 8.94e-8、K/V 逐位一致、greedy token 全一致），再以 F32 量化数据流对 F64 同构参考测实现正确性、对同 W16 checkpoint 的 HF F32/KV32 测质量。**修正记录**：首版校准器在 RMSNorm（把归一化值先窄化再加 F32 weight）、SiluMul（拆成 SiLU 与 Mul 两次窄化）、RoPE（先窄化 cos/sin）和 Attention（16-bit 精度链）上镜像 HF 混合精度写法，与本文冻结表不一致，导致 production kernel 无法落进 impl 预算；已改为冻结放置（RMSNorm F32 weight + 单次输出窄化、SiluMul 单算子、F32 cos/sin 与 F32 score/softmax/累加），并据 2046-token prefill 的 F32-vs-F64 累积顺序漂移把 F16 impl logits 预算由 `2e-5`/`2e-4` 重冻结为 `1e-4`/`1e-3`（实测 1.72× 旧预算，新预算下归一化 0.69）。最终实测最大归一化误差：impl logits ≤0.002、K/V ≤0.376；quality logits ≤0.474、K/V ≤0.684，全部低于 1；两次独立进程的 35 个规范产物逐位一致。raw 证据见 gitignored `benchmark-results/models/a16-calibration/contractfix-a` 与 `contractfix-b`（首版 `run-a`/`run-b` 已被本次修正取代）。校准不验证 production kernel、实际内存布局或性能；activation 量化在参考实现内完成，不对应任何序列化资产；消费真实 W16 checkpoint 仅用于质量对照。ISA kernel 须按同一预算独立验收。

**kernel 分层与 ISA。** 同一冻结合同下：O1 scalar reference（禁 FMA，沿用 -ffp-contract=off TU 隔离）→ O2 native（GEMM `VDPBF16PS` 微内核优先 → RMSNorm/元素级 AVX512-FP16 → Attention QK^T）+ AVX2 widen fallback，同批实现。CMake 探针 `AETHERMIND_HAS_AVX512BF16/AVX512FP16` 与 per-TU configure 函数按 [AetherMindCpuIsa.cmake](../../cmake/AetherMindCpuIsa.cmake) 现有模式扩展（编译器能力探针，非硬件），consumer 宏照搬 [src/CMakeLists.txt](../../src/CMakeLists.txt)；`CpuFeature::kAvx512Fp16` 已接入 CPUID leaf7 EDX[23] + AVX-512 XSAVE 门控（对齐 kAvx512Bf16；本机两 feature 均不满足，native 永不选中）。native descriptor 声明 required features，CpuBackend 按 `effective_features` 过滤：无硬件时 native 永不选中、fallback 承接，不隐式改变 preset（§2.6）。无硬件验证分层：编译/探针与 per-TU 隔离、dispatch 正确性（合成 CpuCapabilities 快照单测：native 仅 feature 齐备时选中、policy 只能缩减）、算法数值设计（scalar 仿真 native 累加分组合同对拍 oracle）本机可做；硅片数值证据待硬件首跑同一套件补齐。

**实施顺序。** ① 合同冻结（preset/rounding placement/端口签名/2 组合矩阵）+ ISA 基础（探针/detection/configure）并行 → ② 独立校准冻结预算（已完成，见上）→ ③ Cast 基础设施 + `output_dtype` 迁移 + Linear mixed-output + `activation_dtype` 配置链（已完成）→ ④ O1 reference + O2 native + AVX2 fallback 同批（已完成，见下）→ ⑤ 本机验收（tiny + 校准 fixture 2048 near-limit 正式验收、ASan/UBSan、零分配、重复性、负例）（已完成，见下）→ ⑥ baseline（含 vs W16/A32 对照，诚实记录本机无 native 收益）+ 文档收口（已完成，见下）→ ⑦ 新硬件首跑同一套件补 native 数值证据、更新逐 ISA 状态。

**步骤 4 实现记录（2026-11-13）。** O1 reference 覆盖 Llama 图实际算子集：Embedding（同型 gather，0 次舍入）、RmsNorm（A16 激活 + F32 weight，F32 方差，输出 1 次窄化）、Linear（A16×W16，LMHead 直出 F32）、RoPE（F32 三角/旋转，1 次窄化）、KVCacheUpdate（同型直写 0 次）、Attention（F32 score/softmax/累加，1 次窄化）、SiluMul（1 次窄化）；residual Add 复用既有 BF16 reference，Argmax 保持 F32。Attention 的 16-bit 输出无法承载 KV16 reference 的 F32 累加缓冲，改用固定 64 元素栈块按 head_dim 分块并重算 logits，换取零堆分配与"输出只窄化一次"。ISA 层：A16 Linear 注册 AVX2/FMA widening fallback（F16C 加宽 F16、精确左移加宽 BF16、FMA 累加，priority 20）与 AVX-512 native（BF16 `VDPBF16PS` 分组累加、F16 native 加宽 + FMA，priority 30）；A16 RmsNorm 注册 AVX2/FMA widening fallback 与 AVX-512 native（BF16 复用固定 RNE/gradual-underflow 窄化、F16 用 `VCVTPS2PH` 窄化）。所有 ISA descriptor 声明显式 `cpu_requirements`，`CpuBackend` 仅按 `effective_features` 选中；非单位列 stride 在 kernel 内回退 scalar reference，binding 不因布局失败。合成 `CpuCapabilities` 单测覆盖"仅 AVX-512F 不选 native""BF16-only 不选 FP16 native"与 fallback 选择；AVX2 路径在本机执行并与 FP64 oracle 对拍。**逐 ISA 状态**：scalar reference 与 AVX2 fallback = 已实现且本机数值验证；AVX-512 BF16/FP16 native = 已实现、编译与 synth dispatch 已验证、硬件数值证据待首跑采集。SiluMul/Attention 的 ISA 专用路径未在本批实现（exp/归约占总成本，且需独立 baseline），随步骤 6 profiling 决定。

**native BF16 subnormal 修复（2026-11-16）。** `VDPBF16PS` 固定 DAZ/FTZ，Linear 因而先对 vector body 做整数安全域检查：非零乘数对须为 finite normal，biased exponent 和在 `[142, 379]`；下界保证每个乘积及 F32 部分和均为 `FLT_MIN` 的整数倍，覆盖 normal 乘数相消后产生 subnormal 的情况。超出安全域时重算整个 invocation，使用既有 scalar reference；scalar tail 继续沿用 F32 运算。RMSNorm 保留向量 F32 计算，将 16-lane 结果经固定栈缓冲交给既有整数 RNE narrowing，避免 `VCVTNEPS2BF16` 把 subnormal 清零。新增向量主体/tail、normal product underflow、相消、signed zero 与 RNE/normal 边界回归；本机覆盖 reference/AVX2 和 portable 安全域，AVX-512 硅片数值证据仍待对应硬件。该修复不修改冻结预算；Release 相关 77 项通过（含 2048 near-limit），ASan/UBSan 相关 75 项通过、1 项 interposer 用例跳过（Release 已通过），sanitizer 本轮排除 near-limit 且未启用 LSan。安全域预扫描及 narrowing 的性能成本尚未在 native 硬件测量。

**步骤 5 验收记录（2026-11-13）。** P4-A 可执行 preset 为 **BF16 全同型**（A=W=KV=residual）。当时 FP16 A16 仅在 kernel 层可用，`Load` 对 F16 activation 返回 `Unimplemented`。[P4-B](#46-p4-bfp16-计算与-fp32-residual) 后续补齐 FP32 residual 的 Cast 边界并开放模型执行。P4-A 模型级验收分两层：

- **reduced fixture**（`test_a16_model_acceptance.cpp`，BF16）：O1/O2 × Prefill + 2 Decode 与同 W16/A32/KV32 对照，含 near-capacity（14→16）、GQA+tied、Generate 两次重复一致、Decode 稳态 malloc-family 全零（2 preset）。实测最大归一化误差（日志 properties）：标准 0.30/0.22/0.38、near-capacity 0.57/0.41/0.89、GQA+tied 0.17/0.27/0.05，均低于 1。另以"真 A16"负例断言 A16 与 F32 轨迹非逐位相同（防止静默执行 F32 kernel），并断言 Runtime KV dtype 与 activation 不一致时 `Load` 早拒。
- **校准 fixture**（`test_a16_calibrated_acceptance.cpp` + `tools/prepare_a16_reference.py` 导入 `impl_f64_oracle`）：5 个数值 case（含 2046→2048 near-limit）× BF16 × O1/O2，对 F64 同构 oracle 满足 §4.5 impl 预算，near-limit 额外与同 W16/A32 实时运行满足 quality 预算与固定排名规则。fixture 由 `test_a16_reference_identities.h` 逐文件校验 bytes + FNV-1a64。

其余门禁：A16 全套件在 ASan+UBSan 下通过（0 AddressSanitizer/UBSan/LeakSanitizer 报告）；两次独立进程运行规范化 `max_normalized` properties 297 项逐位一致（缺失 0、差异 0）；O2 构图经 `ModelCompiler` 对 A16 关闭 QKV/GateUp/AddRmsNorm 融合（未融合保守基线，与 §4.5"AddRmsNorm A16 融合按 O4 另行批准"一致），因此 A16 的 O1/O2 差异仅来自 identity-packed 权重。

**步骤 6 baseline（2026-11-13）。** 环境：Intel Core Ultra 9 285H（WSL2，Release，单请求同步，未启用线程池），`effective_features` 含 AVX2/FMA/F16C、**无 AVX-512**，因此本机不执行任何 native kernel。负载：校准 fixture `mha-near-limit` 的 BF16 W16 checkpoint（hidden 16、2 层、4 heads、head_dim 4、vocab 32000），prompt 128、`max_new_tokens 3`、warmup 1、repetitions 15，官方 runner 产出并通过 `verify_model_benchmark.py` 校验（`performance_claim: none`）。测量口径为 `internal.*`（prefill/decode 仅计 `Executor::Execute`）：

| 配置 | prefill 中位 | decode 总中位（2 步） | Generate tokens/s 中位 | first-token 中位 |
|---|---:|---:|---:|---:|
| BF16 A16 / O1 plain（keep-bf16, KV bf16） | 46.26 ms | 0.964 ms | 60.05 | 48.96 ms |
| W16/A32 / O1 plain（keep-bf16, KV f32） | 234.86 ms | 3.643 ms | 12.34 | 237.39 ms |

同进程内存账本（两份报告，均 15 次）：raw weights 2,064,704 B（相同）、KV capacity 133,120 B（相同）、workspace 0、peak RSS 25.99 MB vs 26.19 MB、prefill activation 16,589,824 B vs 16,794,624 B。

**结论与归因（必须如实解读）**：
- 本机 A16 preset 端到端约快 3.8×（decode）/ 5.1×（prefill），但差异**主要来自 kernel 路径而非数据宽度**：A16 Linear/RmsNorm 在本机选中 AVX2/FMA widening 实现，而 W16/A32 只有 scalar reference；两侧权重都是 16-bit W16，读取字节数相同。native AVX-512 未参与，本机不能证明 native 收益。
- 内存侧本 fixture 无可测收益：head_dim 4 时 KV 每 token 行被 64 B 对齐支配（F32 16 B 与 BF16 8 B 都填充到 64 B），且 prefill 被 `[128, 32000]` F32 logits 缓冲主导。按同一公式，head_dim ≥ 32 时 KV 才出现 2× 收益；hidden 很小（16）时 O2 identity packing 也不产生收益。
- 因此本 baseline 支持"发布 preset 在本机可用且更快"，**不支持**"A16 相比 F32 普遍更快"或"native AVX-512 已验证收益"的结论。跨机/跨 preset 比较仍须同一 runner、同一 workload 身份与至少 4 次独立进程分散度（见 `tests/benchmark/models/README.md`）。
- 工具链同步：P4-A 的 `aethermind_model_benchmark`、`run_model_benchmark.py`、`verify_model_benchmark.py` 新增 `activation_dtype` 身份字段（schema 2 可选，历史报告按 F32 解释）。当时工具校验 A16 的同型权重和 KV，并拒绝 FP16 activation。FP16 的后续执行范围见 [P4-B](#46-p4-bfp16-计算与-fp32-residual)。
- ISA 专用 SiluMul/Attention 路径的步骤 6 决策：本批不新增。当前 fixture（hidden 16）的 decode 由 LMHead/Linear 主导，未单独测量这两类算子的占比；在缺少 AVX-512 硅片数值证据的前提下，保留 scalar reference 与 AVX2 Linear/RmsNorm，待真实模型 profile 后按 O4 门禁另行评估。

**退出条件。** 2 组合 × O1/O2 的 Prefill/多次 Decode/Generate、EOS/容量/OOM 重试、冻结预算 + 固定排名规则（含 2046→2048 near-limit 的校准 fixture 正式生产验收）、Decode 稳态零分配、确定性/重复性（两次独立进程逐位一致）与负例（真 A16 vs 贴标签 A32、ISA fallback 不改精度合同、selector 篡改拒绝、Cast folding 复用冻结位合同、fusion 不得移除必需窄化）全部通过；逐 ISA 状态单独记录后才标记对应支持（"已实现；编译/dispatch/仿真已验证；硬件数值证据待采集"为状态记账，非准入壁垒）；不泛化为所有 dtype 或平台组合。

### 4.6 P4-B：FP16 计算与 FP32 residual

CPU-06 首段已在 CPU O1/plain 和 O2/identity-packed 打通 C++ `Load → Compile → Prepare → Prefill/Decode → Generate`。FP16 权重、计算分支和 KV 同型。跨层 residual、RMSNorm 输入输出和权重为 FP32。沿用 A16 的未融合图。CLI 和 C ABI 使用既有 strict FP32 准入。

Embedding 从 FP16 直接拓宽到 FP32。RMSNorm 在 FP32 residual 上执行，输出在进入 Q/K/V、gate/up 或 LM head 前窄化。Attention 输出投影和 MLP down projection 先写出 FP16，再拓宽并与 residual 作 FP32 Add。K 保留两次窄化（投影和 RoPE），KVUpdate 直接写 FP16 cache。LM head 直接写 F32 logits。动态 Cast 注册所需的两个方向，绑定时验证范围、输出 injectivity 和输入输出互斥。详见 [Cast](../operators/cast/README.md)。

独立 oracle 由 `tools/calibrate_p4b_reference.py` 生成。使用既有冻结的五组 FP16 checkpoint，FP64 oracle 保留同一 FP32 residual 和 FP32 RMSNorm 写回边界，另以 FP32 算术轨迹校准。生产测试比较独立 oracle，并在 near-limit 与同源 W16/A32/KV32 比较质量。预算为逐元素 `atol + rtol × abs(ref)`。

| 比较 | Logits (atol, rtol) | K/V (atol, rtol) |
|---|---:|---:|
| P4-B 生产与独立 oracle | 1e-4, 1e-3 | 2e-4, 2e-3 |
| P4-B 生产与 W16/A32/KV32 | 2e-3, 6e-3 | 1e-3, 1e-2 |

`P4BCalibratedAcceptance` 在两个 preset 检查 MHA、GQA、MQA、tied 权重、logits 排名、KV 水位和公开 greedy token。2046-token Prefill 后的两步 Decode 到达 2048。两次独立 oracle 生成的文件身份一致。Release 的 malloc-family 门禁覆盖两个 preset 的 prepared Decode。见 [本机记录](../operators/cast/benchmarks/desktop-54h5mmi-cast.md)。

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
| A16 首批性能 | P4-A 的 BF16 全同型图不含 Cast；本机 baseline（§4.5 步骤 6）实测 A16 preset decode 3.8×、prefill 5.1× 快于 W16/A32，但归因于 A16 已具备 AVX2 widening Linear/RmsNorm 而 W16/A32 仅 scalar reference，**不能归因于数据宽度**；无 AVX-512 硬件，native 收益未验证。是否默认开放由真实模型 profiling 与跨机型 baseline 决策，性能故事随 native 硬件兑现 |
| 全 A16 residual 累积误差 | BF16 residual 跨层舍入累积（8-bit 有效数）须独立校准证明；校准 reference 精确建模残差每层量化；FP16 全 residual 溢出风险留 P4-B（其 residual 强制 F32）。步骤 5 已由 2046→2048 near-limit 校准 fixture 的 impl/quality 预算正式验收覆盖 |
| Native 在未验证硬件自动 eligible | 新硬件到位后 native 按 `effective_features` 自动选中；首跑同一套 P4 验收套件补齐硬件数值证据后再更新逐 ISA 状态；过渡期可用 `CpuFeaturePolicy.disabled_features` 临时禁用（可选杠杆，非准入门） |
| Native intrinsics 正确性 | 合同先行 + scalar 仿真 native 累加分组对拍 + 独立 TU 隔离；native 代码不影响 fallback 构建与本机验收 |
| AVX512 downclock / 256↔512 切换 | 留待硬件实测；512 位限 GEMM 微内核的策略不变 |
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

CPU-02/03 的后续状态：W16/A32 矩阵已不再统一落在 scalar reference，FP32 亦已接入生产 optimized driver。§4.5 步骤 6 的旧 A16/W16 baseline 保留原源码/机器范围，不应将其比值或归因套用于当前 W16 路径；当前同合同 A/B 见 [CPU 算子路线图](../operators/cpu-operator-evolution.md)。

- 分别记录转换/packing/准备耗时、峰值与稳态 RSS、持久权重/packed/KV/activation/workspace 字节、Prefill TTFT 和 Decode tok/s。
- 覆盖 Prefill/Decode、hot/streaming、代表性维度与 context depth；同机同配置比较。默认 priority 提升需要 registered/prepared、production ExecutionStep 与端到端证据，microbenchmark 只作诊断。
- 证据按 [算子工作流](../guides/operator-development-workflow.md) 写入对应算子机器记录及 raw artifact，不在本提案填入未经验证的加速倍数或占位 benchmark 文件。
- 每个 preset、phase、layout、ISA/fallback 单独标记状态；只有实际通过相应门禁后，才同步更新模块设计、算子合同/API 文档、产品 capability 和行为变更记录。该 Draft 不修改 PRD 的已交付状态。

**A16 逐 ISA 状态。** 下表区分当前选核行为与验证范围；沿用 §4.5 的自动 eligibility 策略，验证状态不构成运行时禁用条件。对应 descriptor 已编译、selector 匹配且 effective features 满足时，Linear/RMSNorm 会优先选择 native，具体 priority 见 [backend 选核规则](../handbook/核心模块设计/后端系统%20%28Backend%29/后端系统%20%28Backend%29.md#cpu-后端选择算法)。合成 capabilities 只用于选核测试，不能用于在缺少指令的机器上执行 native entry。

| 路径 | 编译 / dispatch 证据 | 真实硬件数值 / 模型证据 | 性能状态 |
|---|---|---|---|
| reference / AVX2 widening | 主线构建；[Linear dispatch](../../tests/unit/backend/cpu/kernels/test_cpu_linear_a16_dispatch.cpp) 与 [RMSNorm](../../tests/unit/backend/cpu/kernels/test_cpu_rmsnorm_a16_kernel.cpp) 回归；真实本机选核见 [Linear 记录](../operators/linear/benchmarks/desktop-54h5mmi-linear.md#a16-实际选核身份与验证边界核对) / [RMSNorm 记录](../operators/rmsnorm/benchmarks/desktop-54h5mmi-rmsnorm.md#a16-实际选核身份与验证边界核对) | §4.5 步骤 4–5 的 kernel 验收与 BF16 O1/O2 模型/2048 near-limit 验收；FP16 仅 kernel 层 | §4.5 步骤 6 的限定 BF16 fixture baseline，不推导跨机器收益 |
| AVX-512 BF16 Linear/RMSNorm | per-TU 编译与 descriptor 注册；Linear 合成特征 dispatch 已覆盖；BF16 安全域/窄化的 portable 回归已覆盖 | **not-run：缺少 AVX-512 BF16 目标硬件**；portable 安全域证明不替代 native 指令执行 | **not-run**：native latency、保护预扫描、fallback 占比与降频成本待测 |
| AVX-512 FP16 Linear/RMSNorm | per-TU 编译与 descriptor 注册；Linear 合成特征 dispatch 已覆盖 | **not-run：缺少 AVX-512 FP16 目标硬件**；高层 FP16 A16 已完成 §4.6 的本机验收 | **not-run**：kernel 性能待测；目标硬件模型性能待测 |

目标硬件补验须保留真实 `DetectCpuCapabilities` 与 feature policy、完整 selector、实际 `ResolvedKernel::name` 和 exact recipe，分别覆盖 plain/identity-packed、Prefill/Decode、A16/F32 output（Linear）、tail/stride/fallback 和特殊值。先完成 Linear/RMSNorm 的 native 执行与冻结预算对拍，再在 BF16 硬件上运行 §4.5 的 BF16 模型/near-limit 验收；FP16 目标硬件复用 §4.6 的模型与 near-limit 验收。性能另做同合同、同机交错 A/B，包含 prepared、ExecutionPlan 与可执行模型路径及 native 数值保护成本。原始证据与结果按算子工作流入对应机器记录，并由 CPU-01 台账链接；没有目标硬件的项目保留 not-run。

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
| 2026-11-12 | 1.9 | 开展 P3 步骤 1：冻结四种 W16×KV16 支持矩阵、显式配置、post-RoPE narrow/cache widen、reference 算术与 FP 环境、state/alias/事务边界，建立 Attention/KVUpdate Draft 合同；数值预算由独立校准后冻结，后续语义与 kernel 仍待实施 |
| 2026-11-12 | 2.0 | P3 步骤 1 完成：独立校准器（整数 RNE oracle + pinned HF reference + 量化 cache 边界 + FP64 写回 F32）通过转换器位合同、比较器 negative checks 与 FP32 golden 逐位交叉验证；四种 W16×KV16 组合 × 5 fixture 全部通过两层预算（impl 归一化 ≤0.298、quality ≤0.567），两次独立运行 65 个产物逐位重复；据校准冻结 §2.8 数值预算并记录依据；步骤 2–10 仍待实施 |
| 2026-11-12 | 2.1 | P3 步骤 2–4 完成：Attention/KVUpdate 扩展 A32→KV16 mixed 语义（保留同型合同、state 输出 alias）；`state_dtype` 加入 KernelSelector 并迁移 derivation/lowering/lowered 验证/untrusted 重校验/registry/artifact key，F32 descriptor 补 state_dtype、无 state 严格 undefined；复核 dtype-aware KVCacheLayout 两字节账本（逻辑减半/物理未必减半）。335 项迁移回归 + 58 项 F32/W16 端到端模型生成 + execution/lowering 套件通过，clang-format 干净；KV16 kernel（步骤 5/6）及后续仍待实施 |
| 2026-11-12 | 2.2 | P3 步骤 5–6 完成：新增 CPU KV16 KVUpdate 写入 reference（F32 activation 经固定 RNE 标量窄化为 F16/BF16，byte-safe 写入 logical head_dim，复用 F32 params builder）与 Attention 读取 reference（cache16 widening、F32 固定归约、独立 TU 禁 FMA contraction）；注册 {act=F32, state=F16/BF16} 四个 kernel。独立 oracle 验收：KVUpdate 已知位模式/RNE midpoint/特殊值/padding/history/stride/alias 重叠拒绝/dtype 拒绝/prepared 零分配，Attention double 精度 oracle 覆盖 prefill/GQA/MQA/decode/dtype 拒绝。KV16 + F32 kernel + 58 项端到端模型 + execution/kv_cache/registry 回归通过；plan 事务（7）、配置链（8）、模型验收（9/10）仍待实施 |
| 2026-11-12 | 2.3 | P3 步骤 7–8 完成：LayerRunner 事务 dtype-agnostic 无需改，补 KV16 prefill/decode/多层/缺层不提交/update-then-attention/后续失败不发布回归（F16/BF16）；`CpuRuntimeAssemblyOptions`/`ModelLoadOptions` 追加 kv_dtype，新增通用 MakeKVCacheOptions（标量 F32/F16/BF16 准入）+ F32 wrapper，MakeCpuRuntimeOptions 与 Load 由其驱动，Load 在转换/packing 前校验 Runtime Manager dtype 一致；KV16 仅对两个 W16 policy 开放。KV16 Load→Generate 限定 fixture 闭环（含跨 W×KV）、不一致/非法配置早拒、C ABI strict F32 保持（51 项 C ABI 含 KV 校验 fixture 修正）。综合回归通过；四组合模型验收（9）与 baseline/收口（10）仍待实施 |
| 2026-11-12 | 2.4 | P3 步骤 9 完成（reduced fixture）：泛化 CaptureStage 按 kv_dtype widening K/V（F32 identity，golden 不变），新增 KV16 四组合模型验收（O1→O2，KV16 vs 同 W16 KV32，固定排名规则 + Generate/重复）、近容量 14→16 长上下文与 GQA tied。实测 reduced fixture 容差（区别于 §2.8 校准 fixture 冻结预算：BF16 V 尺度需独立校准，max normalized logits≤0.37/keys≤0.24/values≤0.73）。步骤 10 文档收口进行中；§2.8 冻结预算在校准数值 fixture（2048）的正式生产验收、KV16 Decode 模型级零分配、baseline 与 sanitizer/fallback 证据为后续精确项 |
| 2026-11-12 | 2.5 | P3 步骤 9/10 推进：补 KV16 Decode 模型级零分配证明（F16/BF16×O1/O2，binding 后 malloc-family 全零）；KV16 全套件（kernel/事务/Load 链/模型验收）在 ASan/UBSan 下通过（29 过/2 跳过零分配 interposer，0 内存错误/UB/泄漏），证据存 gitignored benchmark-results/models/kv16-step10/。KV16 reference 为纯 scalar、无 ISA 依赖（fallback 即 reference）。文档同步完成；prepared/production/model baseline 与 §2.8 校准 fixture（2048）正式生产验收为后续精确项 |
| 2026-11-12 | 2.6 | P3 步骤 10 核心完成：新增 tools/derive_w16_numerical_fixture.py（逐元素独立 RNE oracle 校验，修正 int16→uint32 符号扩展）派生 W16 校准 fixture；KV16 生产路径在 mha-near-limit（2048，vocab 32000）四组合×O1/O2 通过 §2.8 冻结预算正式验收（KV16 vs 同 W16 KV32，2046 prefill+2 decode，固定排名规则）；采集 KVUpdate/Attention prepared-operator baseline（KVUpdate decode ~13.5µs/prefill ~1.6ms，Attention scalar decode ~24ms，raw + desktop-54h5mmi 算子记录）。P3 步骤 1–10 核心闭环；production ExecutionStep/模型级 baseline 与 SIMD 优化为后续 refinement |
| 2026-11-13 | 2.7 | P4 首批设计冻结（§4.5）：2 同型三元组支持矩阵与逐端口 rounding placement（Q/K 双舍入、KVUpdate 直写 0 次、LMHead 直出 F32、BF16 全 A16 residual 同型 Add 无 Cast，冻结 Add 中间精度 = widen→F32 add→RNE narrow 与残差跨层 BF16 持久）；selector `output_dtype` 严格派生合同（全 A16 下 AddRmsNorm 双输出同型可覆盖，首批以未融合 Add+RmsNorm 为基线）；校准先于一切 production kernel、预算全新冻结且须吸收 BF16 残差跨层累积；native ISA kernel 纳入主线（编译探针/per-TU 隔离/dispatch/仿真本机可验，硬件证据首跑同一套件补齐），O1 scalar + O2 native + AVX2 widen fallback 同批实现；补 A16 性能不预设、未验证硬件自动 eligible、intrinsics 与 downclock 风险。ISA 基础同步落地：AVX512-BF16/FP16 编译探针与 per-TU 配置函数、`CpuFeature::kAvx512Fp16` 的 CPUID leaf7 EDX[23] + AVX-512 XSAVE 门控 detection |
| 2026-11-13 | 2.8 | P4 步骤 2 完成：新增 [calibrate_a16_reference.py](../../tools/calibrate_a16_reference.py) 独立校准器（从零 Llama 前向 + 逐边界显式 A16 量化；先对 accepted FP32 HF golden 5/5 交叉验证，logits max_abs 8.94e-8、K/V 逐位一致、greedy token 全一致；再以 F32 量化数据流对 F64 同构参考测实现正确性、对同 W16 checkpoint 的 F32/KV32 测质量），覆盖 5 fixture（含 2046→2048 near-limit）× 2 同型组合。据实测最大归一化误差（impl logits ≤0.002、K/V ≤0.376；quality logits ≤0.441、K/V ≤0.562）冻结 §4.5 A16 四层预算；两次独立进程 35 个规范产物逐位一致；raw 证据存 gitignored `benchmark-results/models/a16-calibration/run-a|run-b`（首版证据已被 2.11 的合同修正取代） |
| 2026-11-13 | 2.9 | P4 步骤 3 完成：新增 `kCast`/`CastParams`/`InferCast`（六个跨型方向、同型拒绝）、schema `CompileTimeEvaluable`、serde `Cast target_dtype=...`、`AddCast` 与 flat/strided 常量折叠 evaluator（经 float 复用冻结转换位合同；backend kernel 按首批范围暂不注册）；`KernelSelector.output_dtype` 在 base/operators/compiler/execution 全链路严格派生与校验（无 wildcard，state-only 为 undefined），全部 descriptor 与测试 selector 补 concrete 值并新增 selector 字段回归；`LinearParams.output_dtype` override 落地（裸 `Linear` serde 逐字节兼容），A16 构图 LMHead 显式 F32；`ModelLoadOptions.activation_dtype` 与 Load 早拒（同 dtype W16、CPU O1/O2、标量白名单）落地。全量单测 4776 通过（CliMatrix 集成门禁需独立 CLI 环境，未在本轮运行），clang-format 干净 |
| 2026-11-13 | 2.10 | P4 步骤 4 完成：A16 kernel 层落地。O1 scalar reference 覆盖 Llama 图算子集（Embedding 同型 gather 0 次舍入、RmsNorm F32 weight + 1 次窄化、Linear A16×W16 含 LMHead F32 直出、RoPE 1 次窄化、KVCacheUpdate 同型直写 0 次、Attention 固定 64 元素栈块分块累加 + 1 次窄化、SiluMul 1 次窄化），全部置于 -ffp-contract=off 的 reference TU；ISA 层为 A16 Linear 与 RmsNorm 增加 AVX2/FMA widening fallback（priority 20）与 AVX-512 native（BF16 `VDPBF16PS`/`VCVTNEPS2BF16`，F16 native 加宽与 `VCVTPS2PH`，priority 30），per-TU ISA 隔离 + `cpu_requirements` 过滤，非单位列 stride 回退 scalar。新增 36 个 A16 单测（Linear/Embedding/RmsNorm/SiluMul/RoPE/KVCacheUpdate/Attention reference 套件 + Linear dispatch 与 AVX2 数值对拍），并将既有 SiluMul "F16 不 resolve" 断言更新为解析 A16 reference；合成 `CpuCapabilities` 覆盖 native 仅在 feature 齐备时选中；全量单测 4812 通过（-CliMatrix/*），本机 AVX2 路径与 FP64 oracle 对拍通过，AVX-512 数值证据待硬件首跑。SiluMul/Attention ISA 专用路径留待步骤 6 profiling 决策；顶层状态同步更新 |
| 2026-11-13 | 2.11 | P4 步骤 5 完成（BF16 全同型模型级验收）并修正校准器：`calibrate_a16_reference.py` 的 RMSNorm/SiluMul/RoPE/Attention 改为严格冻结放置（RMSNorm F32 weight + 单次输出窄化、SiluMul 单算子、F32 cos/sin、F32 score/softmax/累加），据 2046-token F32-vs-F64 累积漂移把 F16 impl logits 预算重冻结为 `1e-4`/`1e-3`；两次独立进程 35 个产物逐位一致，最终实测 impl logits ≤0.002 / K/V ≤0.376、quality logits ≤0.474 / K/V ≤0.684。新增 `tools/prepare_a16_reference.py`（校验 manifest tensor payload、冻结预算与全部 pass 标记后导入 `impl_f64_oracle`，产出 `.models/a16-reference` + `a16_reference_identities.h`）与 `test_a16_calibrated_acceptance.cpp`（5 case × O1/O2，对 F64 oracle 满足 impl 预算，near-limit 另与实时 W16/A32 满足 quality 预算与排名规则）；`test_a16_model_acceptance.cpp` 覆盖 reduced fixture O1/O2、near-capacity、GQA+tied、Generate 重复、Decode 零分配与真 A16/早拒负例；A16 全套件在 ASan+UBSan 下通过，两次独立进程 297 项归一化指标逐位一致。P4-A 可执行 preset 限定 BF16：F16 activation 在 `Load` 显式返回 `Unimplemented`（P4-B 需 F32 residual Cast），A16 构图关闭 QKV/GateUp/AddRmsNorm 融合（未融合保守基线） |
| 2026-11-13 | 2.12 | P4 步骤 6 完成（baseline + 文档收口）：`aethermind_model_benchmark`、`run_model_benchmark.py`、`verify_model_benchmark.py` 增加 `activation_dtype` 身份字段与 A16 一致性校验（schema 2 可选，历史报告按 F32 解释），两份 BF16 A16 / W16-A32 对照报告经官方 runner 采集并通过 `verify_model_benchmark.py`（15 reps，prompt 128，near-limit 校准 checkpoint）：prefill 46.26 ms vs 234.86 ms、decode 总 0.964 ms vs 3.643 ms、Generate 60.05 vs 12.34 tokens/s；内存账本 raw weights/KV capacity/workspace 相同，peak RSS 25.99 vs 26.19 MB。结论明确归因：差异来自 A16 已具备 AVX2 widening Linear/RmsNorm 而 W16/A32 仅 scalar reference，且本机无 AVX-512，native 收益未验证；head_dim 4 时 KV 行被 64 B 对齐支配、hidden 16 时 O2 identity packing 无收益。据此决定本批不新增 SiluMul/Attention ISA 专用路径。顶层状态与 §4.5 步骤序列同步更新 |
| 2026-11-16 | 2.13 | 修复 native BF16 的 subnormal 合同：Linear 为 `VDPBF16PS` 增加可证明覆盖 cancellation 的保守安全域与 reference fallback；RMSNorm 复用固定 RNE/gradual-underflow narrowing；新增 prepared/registered 与安全域回归，冻结预算不变，native 硬件与性能证据仍单列待验 |
