# ExecutableModel 模块设计

- **状态**: Current（描述已验证实现；只写仓库事实）
- **版本**: 2.0
- **日期**: 2026-09-23
- **最近更新**: 2026-11-12
- **来源提案**: [07 号提案：ExecutableModel 生产准备入口方案](../../improvement-plan/07-executable-model-preparation.md)（Implemented）
- **关联代码**: [include/aethermind/inference/](../../../include/aethermind/inference/)（`executable_model.h`、`weight_binding_metadata.h`）/[src/inference/](../../../src/inference/)
- **上游依赖**: compiler（`LoweredModelArtifact`、`BuildWeightPackingRequests`）、model（`LoadedModel`/`ResolvedModelWeights`、`ResolveWeightBinding`、`PrepackWeightRequests`、`PackedWeightCollection`）、execution（`ExecutionPlanBuilder`、`ComputeExternalReadRequirements`、`ExternalTensorBindings`）、runtime（`Runtime` 提供 backends/allocator）、graph/operators 纯数据 payload 契约（`WeightValue`/`ConstantValue`）
- **下游消费者**: [`InferenceSession`](02-inference-session.md)（同步 greedy `Generate`，已落地；[01 号计划](../../improvement-plan/01-inference-session-generate-readiness.md) M5）
- **关联测试**: [tests/unit/inference/test_executable_model.cpp](../../../tests/unit/inference/test_executable_model.cpp)、[test_weight_binding_metadata.cpp](../../../tests/unit/inference/test_weight_binding_metadata.cpp)、[test_load_prepare_executable_model.cpp](../../../tests/unit/inference/test_load_prepare_executable_model.cpp)；权重解析权威测试见 [tests/unit/model/weight/test_weight_binding_resolver.cpp](../../../tests/unit/model/weight/test_weight_binding_resolver.cpp)（`WeightBindingResolver` 套件），需求查询测试见 [test_execution_bindings.cpp](../../../tests/unit/execution/test_execution_bindings.cpp)

## 1. 背景与目标

`ExecutableModel::Prepare` 是编译与执行之间唯一的 artifact 准备入口：`LoweredModelArtifact → ExecutableModel`，负责把 packed 权重、外部绑定与 plan 组装起来。`ExecutableModel::Load(runtime, model_dir, options)` 是面向 HF 模型目录的目标感知编排入口，执行加载、源权重策略、可选转换与编译，再调用 `Prepare`。

目标：

- **单一入口**：调用方（如 `InferenceSession`）只持有 `ExecutableModel`，不读取 `LoweredGraph`/`LoweredModelArtifact`，不解析 `TransformerWeightRole`，不解释 packing 决策。
- **准备期失败优于执行期错误数值**：权重无法解析、常量无 inline 数据、phase 自相矛盾等在 prepare 期以可定位的 `Status` 失败，不推进到执行期。
- **绑定与需求同源**：不可变权重绑定表由 execution 层的唯一需求查询 `ComputeExternalReadRequirements` 推导，二者不可能分叉。

## 2. 职责与边界

- **提供**：`ExecutableModel::Load(runtime, model_dir, options)` 文件到可执行模型的编排、`ExecutableModel::Prepare(runtime, artifact)` artifact 准备、`ExecutableModel::plan(phase)`、`ExecutableModel::immutable_weight_bindings(phase)`、`artifact_id()`、`phase()`。
- **请求**：`Runtime`（backend 解析与 workspace 规划）、compiler 的 request builder、model 的 resolver/prepack planner/store、execution 的 plan builder 与需求查询。
- **所有权**：`ExecutableModel` 按值持有 artifact、`PackedWeightCollection`、`WeightBindingMetadata`、`ExternalTensorBindings`、`ExecutionPlan`；绑定表只借用其中数据，不复制权重。
- **明确不做**：不承担算子语义、kernel resolve 算法、weight materialization 算法（packing recipe）或 KV 物理存储；不提供 model inputs 绑定（由 Session 按 phase 追加）；不暴露 packed store 访问器。
- **生命周期**：`ExecutableModel::Load` 借用调用方的 `Runtime`，不创建或拥有 Runtime/Session；Runtime 必须保持原地址并长于返回的 `ExecutableModel`。整体顺序为 `Runtime` > `ExecutableModel` > `InferenceSession` > `ExecutionContext`；`PreparedExecutionBindings` 借用模型数据指针，必须先于 `ExecutableModel` 释放。

### 2.1 上层共享 FP32 策略

[model_execution_policy.h](../../../include/aethermind/inference/model_execution_policy.h) / [实现](../../../src/inference/model_execution_policy.cpp) 提供 `ValidateSourceWeights(const LoadedModel&, WeightPolicy)` 和 `MakeFp32KVCacheOptions(const HfModelConfig&, size_t)`：前者只读 hint 与全部必需 resolved roles，后者返回已有 `KVCacheRuntimeOptions` owning 值，校验容量、正数/安全转换及逻辑 head_dim。`WeightPolicy` 在该策略头中定义，策略头不再依赖 `ExecutableModel` 的完整定义。策略不读/转换 payload、不分配 KV、不访问 backend、不保留借用；诊断 OOM 返回空 message 的 ResourceExhausted。

该产品准入策略由上层入口显式调用，不改变通用 Loader/Compile/Prepare 的精度支持契约。M5 已移除应用中的实现副本，CLI 与 M6 C adapter 消费同一公共头。Runtime/model KV 跨对象比较和已编译图所有层 state dtype 扫描属于 adapter 私有 helper，不由此共享配置映射代替。完整边界见 [Runtime C ABI 合同](../../api/runtime-c-abi.md)。

`inference_options.h` 中的 `CompilePreset` 与 `MakeCompileOptions(preset, target)` 只映射优化级别、packing 策略与显式目标，独立于 backend 可用性；CLI/C adapter 传 CPU。CPU 资源与 ISA 相关映射继续使用 `CpuRuntimeAssemblyOptions`/`MakeCpuRuntimeOptions`。这些声明继续在现有文件内维护。


## 3. 关键数据结构

### 3.1 ExecutableModel 成员与销毁契约

| 成员（声明序） | 含义 | 备注 |
|---|---|---|
| `artifact_` | `LoweredModelArtifact`（间接持有 `LoadedModel`/`ResolvedModelWeights`/`RawStorage`） | 最先声明、最后销毁，保证绑定借用的权重 backing 与常量 `inline_data` 全程有效 |
| `packed_weights_` | `PackedWeightCollection` | 与 plan 之间通过 `shared_ptr` 共享所有权；保留它是为了维持 `source_id` 与 artifact 的对账能力 |
| `binding_metadata_` | `WeightBindingMetadata` | 被绑定表借用的 shape/stride 元数据 |
| `bindings_` | `ExternalTensorBindings`（只含 weight/constant） | `TensorView` 借用 `binding_metadata_` 的元数据与 artifact 的数据 |
| `plan_` | `ExecutionPlan` | 构建后自持，不借用 `LoweredGraph`；packed artifact 以 `shared_ptr` 持有 |
| `phase_` | artifact 的唯一 `ExecPhase` | 由 plan 各 step 的 `selector.phase` 推导 |

成员声明顺序即销毁顺序的逆序，是头文件写明的 teardown contract。

### 3.2 WeightBindingMetadata

`std::vector<Entry>`，`Entry{std::vector<int64_t> shape, strides}`：

- 元数据指针稳定性由两个独立机制保证——(a) 外层 vector 扩容时 `Entry` 被 move 构造，但内层 vector 的 `data()` 跨 move 不变；(b) `ExecutableModel`/元数据对象整体移动时外层缓冲指针直接移交。
- 因此**禁止**把 shape/stride 换成内联存储（`std::array`、small-buffer）——任何一次扩容或移动都会让已发出的 `IntArrayView` 悬垂。
- stride 按行主序紧凑推导（权重连续性由 HF 校验器与 prepack 路径保证）；`alignment` 统一以 0（未指定）交付，因为 `RawWeightView` 不携带对齐信息。
- `Add` 返回 `StatusOr<TensorView>`，检查负维度、与 `TensorView` 一致的正序 `uint64_t` 元素计数以及每一级逆序 `int64_t` 乘积；负维度返回 `InvalidArgument`，溢出返回 `Overflow`。即使总元素数为零，后缀乘积也可能不可表示，不能用总字节数校验代替 stride 校验。
- shape/stride 完整构造成功后才发布元数据 entry；失败保持原有 entries 和已发出的 views 有效，函数内部的分配失败返回空消息 `ResourceExhausted`。此 C++ 返回类型变更要求直接调用方检查状态并重新编译。

## 4. 并发模型

准备是冷路径操作；`ExecutableModel` 构造后只读，多线程并发读取 plan/绑定表安全，但**构造与访问不并发**（头文件将并发序列化的责任交给调用方）。

## 5. 接口定义

```cpp
struct ModelLimits {
    size_t max_model_payload_bytes = 0;
    size_t max_packed_weight_bytes = 0;
};

enum class WeightPolicy : uint8_t {
    kRequireFp32, kConvertToFp32,
    kKeepFp16WeightsFp32Activation, kKeepBf16WeightsFp32Activation
};
struct ModelLoadOptions {
    CompilePreset preset = CompilePreset::kO1Plain;
    DeviceType target = DeviceType::kCPU;
    WeightPolicy weights = WeightPolicy::kRequireFp32;
    ModelLimits limits{};
    DataType kv_dtype = DataType::Float32();
};

class ExecutableModel {
public:
    static StatusOr<ExecutableModel> Load(
            Runtime&, const std::filesystem::path&, const ModelLoadOptions&);
    static StatusOr<ExecutableModel> Prepare(
            Runtime&, LoweredModelArtifact, const ModelLimits& = {});
    StatusOr<const ExecutionPlan*> plan(ExecPhase) const noexcept;
    StatusOr<const ExternalTensorBindings*> immutable_weight_bindings(ExecPhase) const noexcept;
    const ModelResourceUsage& resource_usage() const noexcept;
};
```

- 两个 phase 访问器返回 `StatusOr<const T*>`（借用指针），使 phase 不匹配可表达为失败而非静默复用；诊断字符串分配失败时返回空消息 `ResourceExhausted`，保持 `noexcept` 合同。
- 只可移动构造、不可赋值：就地替换一个已 prepare 的模型会作废此前交出的 `plan()`/`immutable_weight_bindings()` 借用指针与派生的 `PreparedExecutionBindings`，且逐成员赋值会先释放 artifact 再替换借用它的绑定表（违反 §3.1 的逆序销毁契约）；换模型应构造新对象。
- 返回路径只依赖 `StatusOr` 的 nothrow 移动构造前提（`StatusOr` 不再要求 `T` 可赋值，其赋值运算符对这类 `T` 变为 deleted）。
- 不暴露 packed store 与 artifact 访问器。

## 6. 算法与流程

`ExecutableModel::Prepare` 八步：

1. 取 `artifact.loaded_model->GetResolvedWeights()`（缺失 `loaded_model` 即拒绝）。
2. `BuildWeightPackingRequests(artifact.graph, resolved)`：graph-driven，跳过非 `kPacked` step，按 `(value_index, selector)` 去重。
3. 取 backend（`runtime.GetBackend`，按首个 request 的 device；plain 图无请求时取首个 lowered step 的 device，空图才默认 CPU）；请求先经 `ResolveWeightPackingRequests` 注入 descriptor recipe、收敛共享消费者，再由 `PrepackWeightRequests(backend, requests)` 创建并返回绑定批内 source 的 store（`AM_ASSIGN_OR_RETURN`；失败全有或全无，空批为未绑定空 store）；composite 物化、对齐与分配由 backend 的 `Backend::PackWeights` 落实，产物 recipe 须等于 request recipe。
4. `ExecutionPlanBuilder::Build(runtime, store, artifact.graph)`。
5. `ComputeExternalReadRequirements(plan)`：execution 层的唯一需求权威（packed 裁剪掉权重端口后自然不进入需求集合）。
6. 对每个"被需求且非 `kModelInput`"的值按下标 i 物化绑定：
   - 结构化身份只能取自 `artifact.graph.values()[i].payload`（`ExecutionValueDesc` 不含 payload），下标同一性由 `PrepareTrustedGraph` 1:1 push 保证；
   - `kWeight` → 取 `WeightValue` 的 `binding` 字段交给 `ResolveWeightBinding` 解析（model 层单一权威，含 tied lm-head 回退）；
   - `kConstant` → `ConstantValue::inline_data` 按值 spec 包成 `TensorView`，并校验 inline 字节数与形状推导一致；
   - shape/stride 数组写入 `binding_metadata_` 并被 `TensorView` 借用。
7. 双向对账：`{required 且非 kModelInput}` 与 `{已生成绑定}` 必须完全相等，任一方向不匹配即 `kInternal`。
8. 组装 `ExecutableModel`（成员按 §3.1 顺序，失败路径不泄漏半成品对象）。

### 6.1 phase 推导与查询

- phase 的权威来源是 **plan step 的 `selector.phase`**（lowering 期的 `GraphLoweringConfig.selector` 只是默认值，完成后不再可查）。
- step 间 phase 不一致的 artifact 在 prepare 期直接拒绝（`kFailedPrecondition`，"mixes execution phases"）。
- 内部只构建一个 plan；`plan(phase)` 复用 base 层 `PhaseMatches(candidate, request)`（`candidate == request || candidate == kBoth`）：
  - artifact 为 `kBoth` → `kPrefill`/`kDecode`/`kBoth` 查询一律返回同一 plan；
  - artifact 为单 phase → 不匹配查询返回 `kFailedPrecondition`，不静默复用。

### 6.2 目录加载编排与验证范围

- `ExecutableModel::Load` 依次调用 `ModelLoader::Load`、`ValidateSourceWeights`、可选 `ConvertRawModelWeightsToFp32`、`ModelCompiler::Compile`，最后调用 `Prepare`。转换先于构图，并在编译前释放只供转换使用的 source owner。
- `ModelLoadOptions` 必须显式传入；`{}` 选择 CPU、O1/plain、strict FP32 和无限额。`kConvertToFp32` 允许 F32/F16/BF16 源权重，转换后 weights、activation、KV 与 logits 全为 F32，目前仅允许 CPU O1/plain；CLI/C ABI 继续 CPU/strict FP32。
- `kKeepFp16WeightsFp32Activation` / `kKeepBf16WeightsFp32Activation` 在 CPU O1/plain 或 O2/identity-packed 保留匹配的 W16 矩阵，仅将 Norm 转为 F32。转换前按角色准入，转换后重新验证矩阵 dtype 和 F32 Norm。retained 账本按 surviving views 的 owner identity 去重，完整保留共享 shard 的 footprint。Embedding output、activation 和 logits 均为 F32，W16 Linear 使用升序 F32 分离乘加。KV 默认 F32；显式 `kv_dtype` 可为 F16/BF16，与矩阵精度独立。Load 在转换与 packing 前验证既有 Manager 与请求 dtype 一致；后续 state specs、selector、view/layout 与 binding 仍逐层一致。支持矩阵与入口见 [W16 指南](../../guides/w16-fp32-inference.md)，限定验收见 [P3 记录](../../tests/kv16_execution_validation_2026-11-12.md)。
- 使用任意 `ModelCompileOptions` 的调用方显式组合 Load/Compile/Prepare；旧目录包装函数、CPU 专用名称及转发 overload 已删除。C++ 调用方须迁移并重新编译，C ABI 合同不变。
- 本文件的实现仍集中在 `src/inference/executable_model.cpp`：私有 helper 处理资源核算、packing 限额、权重打包、bindings 与权重策略；对象方法和两个 factory 分别呈现主流程。`WeightBindingMetadata` 与资源观察类型维持各自所有权和计量合同。
- `PreparationPaths/ModelLoadPath` 覆盖 O1/plain 与 O2/packed 的 HF 文件到三 token Generate；只验证 token 数量、范围与重复请求一致性。`CompatibleCheckpoints/Fp32CompatibleInference` 覆盖显式 FP32 转换与独立 FP32 checkpoint 的轨迹、KV 和 Generate 对照。数值与性能承诺以各自验证记录为准。
- `ModelLoadOptions::target` 显式进入 compile options 和 lowered selectors；CUDA/CANN 仅有配置/lowering 与缺 backend 失败路径的验证，不能据此宣称对应设备可执行。未知 target 在加载前失败，FP32 转换对非 CPU 目标在 payload 分配前返回 Unimplemented。
- Runtime 必须保持地址稳定且长于模型/session。CLI 保持先编译、后构建 Runtime；C adapter 保持 Prepare 前的 Runtime/model KV 一致性检查。

## 7. 边界条件与错误处理

| 情形 | 错误码 | 备注 |
|---|---|---|
| artifact 无 `loaded_model` | `kInvalidArgument` | 首个检查 |
| 权重解析返回 `nullptr`（`kMoERouter`、越界/缺失 layer index、composite/roleless） | `kFailedPrecondition` | 消息含 value index 与 `role=<label>, layer=<n>` |
| 权重视图无效/不连续 | `kFailedPrecondition` | 经 `ValidateRawWeightView` + 连续性复查 |
| 常量无 `inline_data`、非静态形状、负维度、形状乘积溢出、字节数与形状不符 | `kFailedPrecondition` | 每条均含 value index |
| step 间 phase 不一致 | `kFailedPrecondition` | prepare 期拒绝 |
| phase 查询不匹配 | `kFailedPrecondition` | 两个访问器一致；诊断分配失败返回空消息 `kResourceExhausted` |
| 绑定 shape/stride 的逆序乘积不可表示 | `kOverflow` | 包括总元素数为零但后缀乘积溢出的形状；不发布部分模型 |
| plan 值下标越出 artifact 值表 / payload 类别不符 | `kInternal` | 值索引同一性被破坏的内部错误 |
| 第 7 步对账不相等 | `kInternal` | 不进入执行期 |

model inputs 不进绑定表：token/position 由 Session 按 phase 追加，重复 id 由 `PrepareExecutionBindings` 的 `InvalidArgument` 校验兜底。

## 8. 风险与权衡

| 风险 | 现状与处理 |
|---|---|
| 值索引同一性是隐式约定 | 已由 plan builder 的 1:1 push 与专项测试固化；prepare 对越界索引返回 `kInternal` |
| shape/stride 借用被后续改动破坏 | §3.2 明确禁止内联缓冲；移动与扩容后的有效性有回归测试 |
| 单 plan 阻碍后续 phase 拆分 | 对外保持按 phase 查询的形状，未来拆分双 plan 不改调用方 |
| 完整 packed Llama 的可解析性依赖描述符覆盖 | `enable_packed_weights=true` 会把所有含权重的 step 标为 packed，因此每个含权重算子都必须有 packed 描述符。`kEmbedding`/`kRmsNorm`/`kLinear` 的 packed identity 变体已补齐，完整 Llama 的 packed 配置现可 prepare（`ExecutableModel.PackedLoweringPreparesAllWeightConsumers`）；新增含权重算子时若漏掉 packed 变体，会在 kernel resolve 期以 `kNotFound` 失败 |
| packed recipe 选举 | `cpu_bpanel_f32_kc512_nr16` 候选与 identity 同 priority，当前选举仍落 identity；升为默认需 benchmark 证据，见 [GEMM packed weight 提案](../../operators/gemm/cpu-gemm-packed-weight.md) |
| phase 匹配语义 | 单 phase artifact 对 `kBoth` 查询返回错误（不声称覆盖两个 phase），与 `PhaseMatches` 一致 |

## 9. 测试要点

2026-11-05 原文件内重构验证：Release 的库、单测、benchmark 和独立 CLI 目标均构建成功；核心聚焦测试 68/68、数值轨迹与 CLI/C ABI 集成测试 223/223 通过，无 skip。覆盖模型移动/绑定、资源边界与失败重试、FP32 转换、O1/plain 与 O2/packed 的 Prefill→Decode；代码格式与 diff whitespace 检查通过。此次接口整理要求 C++ 调用方迁移并重新编译，C ABI 合同保持不变。

同日 preset/target 分离验证：Release 库、单测、benchmark 和独立 CLI 均构建成功；核心与准入测试 92/92、数值轨迹及 CLI/C ABI 集成测试 231/231 通过，无 skip。新增 CUDA/CANN target 经编译/lowering 保留、plain/packed 无 backend 时报告目标设备、非法 target 拒绝，以及非 CPU FP32 转换在分配前拒绝。该证据不代表 CUDA/CANN kernel 或推理已实现。

- artifact 准备：真实 `ModelCompiler` artifact（O1 未融合 tiny GQA Llama）→ `ExecutableModel::Prepare`，权重自动绑定并与需求集合双向对账。文件编排：HF safetensors tiny Llama（O1 + plain）→ `ExecutableModel::Load` → `InferenceSession::Generate`，请求 3 个新 token 以覆盖 Prefill 与 Decode。
- tied/untied lm-head：绑定 `data()` 共享/独立。
- 多层（≥2 layer）：每个权重绑定到自己的 backing，不串层。
- 常量：物化成功、无 inline 数据、字节数不符三条路径。
- packed：`AddRmsNorm` 子图权重不进绑定表且 `step.packed_weights` 非空；完整 Llama 在 `enable_packed_weights=true` 下所有含权重 step 均可解析（`PackedLoweringPreparesAllWeightConsumers`）。
- phase：`kBoth` artifact 三查询同 plan；单 phase artifact 拒绝不匹配查询；混合 phase artifact prepare 期拒绝。glibc interposer 可用时，逐个注入两个访问器错误路径上的每次分配失败，验证空消息 `ResourceExhausted` 及随后查询仍正常。
- 元数据稳定性：移动后、扩容后已发出的 `TensorView` 仍有效；绑定表先于模型释放的 teardown 顺序在 ASAN/TSAN 下验证；`Add` 对形状/stride 溢出返回 `Overflow`；负维度或失败添加不破坏旧 views，正常空张量仍可准备。通过真实 lowering 的 `[0, INT64_MAX, 2]` 空常量在 Prepare 返回错误且 Runtime 可重试。
- 契约钉住：`IsMoveConstructibleButNotAssignable` 断言只可移动构造、不可赋值/复制。

## 10. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-23 | 1.0 | 从 [07 号提案](../../improvement-plan/07-executable-model-preparation.md) 落地实现承接：入口与 API、所有权与销毁契约、八步准备流程、phase 合同、错误码表、测试要点 |
| 2026-09-24 | 1.1 | 修正两处过期断言：下游消费者 `InferenceSession` 已落地（改指 [02-inference-session.md](02-inference-session.md)）；`kEmbedding`/`kRmsNorm`/`kLinear` 的 packed identity 变体已补齐，完整 Llama 的 packed 配置现可 prepare，原"完整 packed Llama 不可解析"风险行改为描述符覆盖依赖，并新增 packed recipe 选举一行 |
| 2026-09-26 | 1.2 | 记录 `ExecutableModel::Load` 的职责、显式编译配置与 Runtime 生命周期；区分 O1+plain 文件到 Generate 的覆盖范围和 O2+packed 的 prepare 证据 |
| 2026-10-05 | 1.3 | `PrepackWeightRequests` 签名改为返回 `StatusOr<PackedWeightStore>`：store 由函数创建并绑定批内 source，失败全有或全无（不再存在部分打包的调用方 store）；生产准备不再显式 `SetSourceId`，§6 第 3 步同步 |
| 2026-10-12 | 1.4 | `WeightBindingStorage` 更名为 `WeightBindingMetadata`（文件同步为 `weight_binding_metadata.{h,cpp}`、测试为 `test_weight_binding_metadata.cpp`）：原名暗示它拥有权重数据，实际只拥有绑定所借用的 shape/stride 数组；`ExecutableModel` 成员与准备期局部变量随之改为 `binding_metadata_`/`binding_metadata`。同时记录 stride 累加改用 `CheckOverflowMul` + `AM_CHECK`（§3.2），消除 `int64_t` 有符号溢出 UB，新增死亡测试 `ShapeProductOverflowDeath`（该文件 8 例 → 9 例） |
| 2026-10-22 | 1.5 | M5 将 FP32 产品准入和模型 KV 配置映射的唯一实现迁入 inference 公共 model_execution_policy.h；明确其与通用 Prepare、C adapter 跨对象比较的职责边界。 |
| 2026-10-22 | 1.6 | M6 C adapter 与 CLI 共用 FP32/KV 策略及新增 CPU preset 接口；跨对象比较仍归 adapter。 |

| 2026-11-05 | 1.7 | 原文件内收敛为 `ExecutableModel::Load/Prepare`，统一 options 与源权重策略，整理资源核算、packing、bindings 私有步骤；删除旧创建自由函数 |

| 2026-11-05 | 1.8 | `CompilePreset` 与显式 target 分开映射；ModelLoadOptions 增加 target，plain 准备沿 lowered target 解析资源；CPU 转换准入与 CLI/C ABI 范围保持明确 |

| 2026-11-06 | 1.9 | phase 查询诊断 OOM 转为 Status；WeightBindingMetadata::Add 改为可恢复的 stride 校验并在成功后发布 entry；WeightPolicy 归入现有策略头，去除策略对完整模型的依赖 |
| 2026-11-12 | 2.0 | 同步 W16 policy 与显式 KV16 options、Runtime/graph/state dtype 一致性，以及独立 oracle/完整 baseline 的限定验收入口 |
