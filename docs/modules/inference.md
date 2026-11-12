# inference 模块设计

inference 位于依赖最上层，负责把编译产物准备成可执行模型，并完成同步单请求的生成编排。它聚合 execution、compiler、model、runtime 与 kv_cache 的能力，但只做编排、产品准入与所有权聚合。分层与依赖方向见 [architecture.md](../architecture.md)，execution 契约见 [execution.md](execution.md)，编译产物见 [compiler.md](compiler.md)，模型前端见 [model.md](model.md)。

## 1. 职责与边界

### 1.1 负责

| 职责 | 具体内容 |
|---|---|
| 目标感知模型准备 | `ExecutableModel::Load` 按 `ModelLoadOptions` 编排 Load → 可选权重策略 → Compile → `Prepare`；`ExecutableModel::Prepare` 是编译产物到可执行模型的唯一生产入口 |
| 所有权聚合 | `ExecutableModel` 按值拥有 artifact、packed weights、绑定元数据、immutable bindings 与 phase plan；绑定表只借用这些数据 |
| 权重绑定元数据 | `WeightBindingMetadata` 拥有 immutable binding 所借用的 shape/stride 数组，保证多次 `PrepareExecutionBindings` 后视图仍有效 |
| phase 合同 | 从 plan step 的 `selector.phase` 推导 artifact 单一 phase；`plan(phase)` / `immutable_weight_bindings(phase)` 在 phase 不匹配时失败而非静默复用 |
| 同步生成 | `InferenceSession::Create` 校验 plan I/O 合同；`Generate` 执行 Prefill → Decode loop → Argmax/stop，并在返回前释放 KV reservation |
| KV 预约与 workspace | 按 `prompt_len + (max_new_tokens - 1)` 预约；一块 workspace 取两 phase 最大值；Decode 稳态不重新分配 |
| 资源核算 | `ModelResourceUsage` / `RequestResourceUsage` 汇总模型与请求 payload；可选 `ModelLimits` / `InferenceSessionOptions` cap 在分配前拒绝 |
| 共享产品准入 | `ValidateSourceWeights`、`MakeKVCacheOptions` / `MakeFp32KVCacheOptions` 提供只读的 FP32/W16 准入与模型 KV 配置映射 |
| 编译与 CPU Runtime 映射 | `inference_options.h` 的 `CompilePreset`、`MakeCompileOptions(preset, target)`、`MakeCpuRuntimeOptions(config, options)` |
| C ABI / CLI 复用面 | CLI 与 C adapter 消费同一 `ExecutableModel` / `InferenceSession` / 共享策略；本模块不复制其错误映射或宿主生命周期管理 |

### 1.2 不得负责

| 不负责 | 归属 |
|---|---|
| 算子语义、shape 推导、kernel resolve | operators / execution / backend |
| `ExecutionPlan` 构建、state alias、workspace 规划 | execution |
| 权重 packing 算法、composite 物化、物理 layout | model / backend（布局权威在 backend） |
| KV 物理 layout、storage、reservation 状态机 | kv_cache（`KVCacheManager` / `KVCacheSessionView`） |
| Runtime 装配、allocator/backend registry 生命周期 | runtime |
| C ABI 的 opaque handle 链、Status/异常/error 适配、buffer 合同 | C runtime adapter |
| Runtime 与已编译模型的跨对象 KV 一致性校验 | C adapter 私有 helper |
| 采样策略（temperature/top-k/top-p）、跨调用 KV 复用、并发 Generate | 当前产品范围之外 |

边界规则：

- 依赖方向为 `inference → execution + compiler + model + runtime + kv_cache`，并可依赖 graph/operators 的纯数据 `WeightValue` / `ConstantValue` / `WeightBinding` 与 base 的 `TensorView` / `KernelSelector`；任何下层不得依赖 inference。
- `ExecutableModel::Load` 不创建 Runtime、不拥有 Session；Runtime 必须保持地址稳定并长于模型与所有派生 bindings。
- `InferenceSession` 不读取 `LoweredGraph` / `LoweredModelArtifact`，不解析 `TransformerWeightRole`，不解释 packing 决策；这些工作全部在准备期完成。
- `MakeCpuRuntimeOptions` 只映射 `RuntimeOptions`，不分配 Buffer、不创建 Runtime；`MakeCompileOptions` 不检查 backend/kernel 是否可用。
- 每个 `Generate` 从新的 Prefill 开始，不缓存跨请求状态；当前唯一采样策略是 Argmax。
- `Generate` 非线程安全；同一 Runtime 上的所有请求（包括不同 Session）必须由调用方串行化，且请求期间不得修改 Runtime 的 KV/backend/allocator 资源。

## 2. 关键数据结构与所有权

### 2.1 `ExecutableModel`

| 成员（声明序） | 含义 | 所有权/生命周期 |
|---|---|---|
| `runtime_` | 借用的 `Runtime*` | 非拥有；必须地址稳定且长于模型；resolved kernel 借用其 backend 状态 |
| `artifact_` | `LoweredModelArtifact` | 按值拥有；间接持有 `LoadedModel`、resolved weights 与 raw backing；最先声明、最后销毁 |
| `packed_weights_` | `PackedWeightCollection` | 按值拥有；plan step 以 `shared_ptr` 共享 artifact，collection 销毁后 plan 仍可执行 |
| `binding_metadata_` | `WeightBindingMetadata` | 拥有绑定表借用的 shape/stride 数组 |
| `bindings_` | `ExternalTensorBindings` | 仅含 weight/constant 的 read-only 绑定；`TensorView` 借用 metadata 与 artifact 数据 |
| `plan_` | `ExecutionPlan` | 按值拥有；不借用 `LoweredGraph`；packed artifact 以 `shared_ptr` 持有 |
| `phase_` | `ExecPhase` | 由 plan 各 step 的 `selector.phase` 推导；混合 phase 在准备期拒绝 |
| `context_limit_` | 已验证的 HF `max_position_embeddings` | 值类型，Session 不回读 artifact |
| `vocab_size_` | 已验证的 HF `vocab_size` | 值类型，Session 不回读 artifact |
| `resource_usage_` | `ModelResourceUsage` | 准备期冻结的观测值 |

`ExecutableModel` 只可移动构造、不可赋值：就地替换已 prepare 的模型会作废已交出的 `plan()` / `immutable_weight_bindings()` 借用指针与派生 bindings；逐成员赋值还会先释放 artifact 再替换借用它的绑定表。成员声明顺序是 teardown contract，析构按逆序释放 plan、bindings、metadata、artifact。

### 2.2 `WeightBindingMetadata`

`WeightBindingMetadata` 是 `std::vector<Entry>`，`Entry{std::vector<int64_t> shape, strides}`。`Add(data, dtype, shape)` 按紧凑 row-major 推导 stride，检查负维度、前向元素数 `uint64_t` 溢出与逆序 suffix `int64_t` 溢出，成功后追加 entry 并返回借用该 entry 的 `TensorView`。该类型只可移动、不可复制，非线程安全，调用方需串行化访问。

稳定性契约：

- 后续 `Add` 扩容时外层 `Entry` 被 move 构造，但内层 `std::vector` 的堆缓冲不迁移，已发出视图仍有效。
- 整个对象移动时外层缓冲指针直接移交，已发出视图仍有效。
- 视图随对象销毁而失效，因此 `WeightBindingMetadata` 必须长于所有由它构造的绑定表与 `PreparedExecutionBindings`。
- **禁止**把 shape/stride 换成 `std::array` 或 small-buffer 内联存储；扩容或移动会使 `IntArrayView` 悬垂。

`Add` 以 `dtype.natural_alignment()` 检查 data 地址并记录在返回的 `TensorView` 中；这是标量 lane 的自然对齐要求，不承诺 SIMD 对齐。负维度返回 `InvalidArgument`，形状/stride 不可表示返回 `Overflow`，元数据分配失败返回空消息 `ResourceExhausted`；失败不发布 entry，也不破坏既有视图。

### 2.3 准备选项与产品策略

| 类型 | 关键内容 | 语义 |
|---|---|---|
| `ModelLimits` | `max_model_payload_bytes`、`max_packed_weight_bytes` | 零表示不限；model cap 要求 raw backing footprint 完整 |
| `ModelLoadOptions` | `preset`、`target`、`weights`、`limits`、`kv_dtype` | `target` 进入 compile options 与 lowered selectors；`kv_dtype` 为 F32/F16/BF16 |
| `WeightPolicy` | `kRequireFp32`、`kConvertToFp32`、`kKeepFp16WeightsFp32Activation`、`kKeepBf16WeightsFp32Activation` | strict FP32、显式全量转换、保留 W16 矩阵仅加宽 Norm |
| `CompilePreset` | `kO1Plain`、`kO2Packed` | 映射 `opt_level` 与 `enable_packed_weights` |
| `CpuRuntimeAssemblyOptions` | `context_tokens`、`kv_max_bytes`、`workspace_limit_bytes`、`cpu_feature_policy`、`kv_dtype` | 用户设置；`MakeCpuRuntimeOptions` 与模型几何合并 |

`ModelLoadOptions{}` 的默认值为 CPU、O1/plain、strict FP32、无限制、KV F32。`kConvertToFp32` 允许 F32/F16/BF16 源权重，转换后权重、activation、KV 与 logits 均为 F32，当前仅允许 CPU O1/plain。两个 retained-W16 policy 在 CPU O1/plain 或 O2/packed 下保留匹配的 F16/BF16 矩阵，只把 Norm 加宽为 F32；KV 可显式选择 F16/BF16，但必须与 Runtime 的物理 KV dtype 一致，且仅 retained-W16 policy 准入。

### 2.4 `InferenceSession` 与请求级资源

| 类型/成员 | 含义 | 所有权 |
|---|---|---|
| `GenerationConfig` | `max_new_tokens`、可选 `eos_token_id` | 值类型；无采样参数 |
| `InferenceSessionOptions` | `max_request_payload_bytes` | 值类型；零表示不限 |
| `prefill_contract_` / `decode_contract_` | `inference::detail::PhaseContract{token_input, position_input, token_output}` | 两份 value id 合同，Decode id 可不同于 Prefill |
| `runtime_` | 借用的 `Runtime*` | 非拥有；被移动过的 Session 由空 `model_` 触发失败 |
| `model_` | `shared_ptr<const ExecutableModel>` | 共享模型所有权，保活权重 backing 与 metadata |
| `max_request_payload_bytes_` | 请求 payload cap | 值类型；参与 `Generate` 的预算分支 |

`PhaseContract` 定义在 `inference_session.h` 的 `aethermind::inference::detail` 命名空间；`InferenceSession::PhaseContract` 是别名，`ValidatePlanContract` 直接返回同一类型。定义放在会话头的唯一原因是 Session 需要按值成员完整类型，而 `inference_internal.h` 位于 `src/` 下不能被公共头包含。

`RequestResources`（`Generate` 内局部 RAII）拥有：

| 成员 | 含义 |
|---|---|
| `prefill_tokens` / `prefill_positions` | Prefill 输入 `vector<int64_t>`，地址在请求内稳定 |
| `decode_tokens` / `decode_positions` | 长度 1 的稳定 Decode 输入 buffer，循环内只改内容 |
| `generated_tokens` | 结果序列；循环前 `reserve(max_new_tokens)` |
| `storage_` | 拥有 workspace 的 `Buffer`；为空时没有 workspace |
| `workspace_arena_` | `SessionWorkspaceArena`，借用 `storage_` |
| `cache_view_` / `has_reservation_` | KV 预约视图与标志 |
| `context_` | `std::optional<ExecutionContext>`，同一时刻至多一个 |

清理顺序为 `~RequestResources` → `Close()` → `ClearContext()`（销毁 context，由 RAII 释放其 owned bindings）→ 若持有预约则 `ReleaseSession(cache_view_)`。`Finish()` 先 `Close()`，成功后移出 `generated_tokens`。`Close()` 只在 `ReleaseSession` 成功后清除 `has_reservation_`；`Finish()` 传播释放错误，析构忽略返回值，因此提前返回或异常展开期间的释放错误不保证向调用方报告。Prefill 结束后显式 `ClearContext()`，使 Decode context 成为当时唯一 context；请求级 KV 句柄与预约标志独立于 context，负责准备失败与两阶段之间的清理。

`SessionWorkspaceArena` 把调用方自有的一块对齐 buffer 适配为 `WorkspaceArena`：`Bind` 校验 `requirement.offset/bytes` 落在 buffer 内且 `base + offset` 满足 `requirement.alignment`，越界或未对齐返回空绑定；`Reset()` 是 no-op。workspace 尺寸与对齐取 Prefill 与 Decode 两个 plan 的最大值，使一块 buffer 服务两阶段。

### 2.5 资源观测

| 类型 | 关键字段 | 语义 |
|---|---|---|
| `ModelResourceUsage` | `weights`、`inline_constant_bytes`、`packed_weight_bytes`、`managed_payload_bytes`、`packing_plan_complete`、可选 `weight_fp32_conversion` | 模型级可见 payload；`managed_payload_bytes` 为 known owned raw backing + inline constants + unique packed storage；不完整时是部分观测 |
| `RequestResourceUsage` | `workspace_bytes`、`prefill_activation_bytes`、`decode_activation_bytes`、`peak_payload_bytes`、`kv_reserved_bytes`、`model_managed_payload_bytes`、`peak_with_persistent_payload_bytes`、`model_payload_complete` | 请求级具体 Buffer 需求；不含 allocator 元数据、mapped backing 与 OS residency |

`RequestUsage` 的算术：`peak_payload_bytes = workspace + max(prefill_activation, decode_activation)`；`kv_reserved_bytes` 取 Manager 的 `total_bytes()`（物理 KV 总量，不是本次 reservation 水位）；`peak_with_persistent_payload_bytes = kv_reserved_bytes + model_managed_payload_bytes + peak_payload_bytes`。所有加法经 checked add，溢出返回 `Overflow`。

## 3. 接口与代码位置

| 入口 | 声明 | 实现 | 行为与错误语义 |
|---|---|---|---|
| `ExecutableModel::Load` | [executable_model.h](../../include/aethermind/inference/executable_model.h) | [executable_model.cpp](../../src/inference/executable_model.cpp) | 校验 policy/preset/target/KV dtype，设置 graph build dtype，调用 `ModelLoader::Load`、`ValidateSourceWeights`、可选转换、`ModelCompiler::Compile`、`Prepare`；阶段失败保留错误码并加阶段前缀 |
| `ExecutableModel::Prepare` | 同上 | 同上 | 校验 artifact/config，核算与限制 payload，打包权重，构建 plan，推导 phase，物化 binding；失败不发布部分模型；`std::bad_alloc` → 空消息 `ResourceExhausted` |
| `ExecutableModel::plan` / `immutable_weight_bindings` | 同上 | 同上 | 返回借用指针；phase 不匹配 `FailedPrecondition`；诊断分配失败返回空消息 `ResourceExhausted` |
| `artifact_id` / `phase` / `context_limit` / `vocab_size` / `IsPreparedFor` / `resource_usage` | 同上 | 同上 | 只读查询；`IsPreparedFor` 按 Runtime 地址比较 |
| `WeightBindingMetadata::Add` | [weight_binding_metadata.h](../../include/aethermind/inference/weight_binding_metadata.h) | [weight_binding_metadata.cpp](../../src/inference/weight_binding_metadata.cpp) | 返回借用视图；负维度/未对齐 `InvalidArgument`，溢出 `Overflow`，分配失败 `ResourceExhausted` |
| `ValidateSourceWeights` | [model_execution_policy.h](../../include/aethermind/inference/model_execution_policy.h) | [model_execution_policy.cpp](../../src/inference/model_execution_policy.cpp) | 校验 dtype hint 与全部必需角色；不读/不转换 payload；不支持精度返回 `Unimplemented` |
| `MakeKVCacheOptions` / `MakeFp32KVCacheOptions` | 同上 | 同上 | 从已验证 config 与容量生成 owning `KVCacheRuntimeOptions`，含 `enable_manager=true`、`alignment=64`；校验容量、维度、head_dim 与 dtype；不分配 storage |
| `MakeCompileOptions` | [inference_options.h](../../include/aethermind/inference/inference_options.h) | [inference_options.cpp](../../src/inference/inference_options.cpp) | preset → `opt_level` / `enable_packed_weights`；显式 target 写入 selector；未知 preset/target 返回 `InvalidArgument` |
| `MakeCpuRuntimeOptions` | 同上 | 同上 | 生成 CPU-only `RuntimeOptions`；合并 `MakeKVCacheOptions`、`kv_max_bytes`、workspace 上限与 CPU feature policy；不创建 Runtime |
| `InferenceSession::Create` | [inference_session.h](../../include/aethermind/inference/inference_session.h) | [inference_session.cpp](../../src/inference/inference_session.cpp) | 含 `InferenceSessionOptions` 重载；拒绝空模型、Runtime 不匹配、无有效 token 上限；取 Prefill/Decode plan 并各校验一次 I/O 合同 |
| `InferenceSession::PlanResources` | 同上 | 同上 | 只规划、不预约、不分配 tensor Buffer；校验几何/生命周期但不应用 cap、不校验 prompt token 值 |
| `InferenceSession::Generate` | 同上 | 同上 | 同步贪婪生成；空 prompt 非法；`max_new_tokens == 0` 返回空结果且不预约；返回含 Prefill 首 token 与触发停止的 EOS |
| `internal::ValidatePlanContract` | [inference_internal.h](../../src/inference/inference_internal.h) | [inference_session.cpp](../../src/inference/inference_session.cpp) | 2 输入/1 输出、Int64 rank-1、CPU step、唯一 Argmax 输出；结构错误 `Unimplemented`，id/kind 错误 `InvalidArgument` |
| `internal::RunPrefill` | 同上 | 同上 | 校验 prompt/vocab/KV 前置条件，重置 workspace，执行完整 plan，校验 prompt watermark，读取最后位置首 token |
| `internal::RunDecodeLoop` | 同上 | 同上 | 循环执行单 token plan，校验每步恰好提交一个 KV position，Argmax 后 push/EOS；不重新准备 bindings |
| `internal::ResolveWeightPackingRequests` | 同上 | [executable_model.cpp](../../src/inference/executable_model.cpp) | 经 `Backend::GetPackingRecipe` 注入 recipe；拒绝跨 device；合并相同 `(value_index, selector)` 且 op/binding/recipe 兼容的 consumer |

`plan` 与 `immutable_weight_bindings` 的 phase 语义由 base `PhaseMatches(candidate, request)` 决定：`candidate == request || candidate == kBoth`。artifact 为 `kBoth` 时 Prefill/Decode/Both 查询返回同一 plan；单 phase artifact 对不匹配查询返回 `FailedPrecondition`。空 plan 的推导 phase 为 `kBoth`。

## 4. 主流程/时序

### 4.1 `ExecutableModel::Load`

1. 校验 `options.weights` 是已知 policy；`MakeCompileOptions(options.preset, options.target)`。
2. policy 与 preset/target 组合准入：`kConvertToFp32` 仅 CPU O1/plain；两个 retained-W16 policy 仅 CPU O1/plain 或 O2/packed。
3. 校验 `kv_dtype` 为 scalar F32/F16/BF16；KV16 仅 retained-W16 policy 允许。若 Runtime 已有 `KVCacheManager`，要求其 `layout().kv_dtype` 与请求一致；KV16 且无 manager 返回 `FailedPrecondition`。
4. 设置 `compile_options.graph_build.activation_dtype = Float32`、`kv_dtype = options.kv_dtype`。
5. `ModelLoader::Load`；失败加 `ExecutableModel::Load loading failed:` 前缀。
6. `ApplyWeightPolicy`：`ValidateSourceWeights`；非 strict policy 时取 CPU allocator/backend 的 host 转换回调，调用 `ConvertRawModelWeightsToFp32`（全量或仅 Norm，owned cap 取 `limits.max_model_payload_bytes`）；随后统一调用 `ValidatePreparedWeights`；转换失败加 `Weight FP32 conversion failed:` 前缀。
7. `ModelCompiler::Compile(std::move(normalized), compile_options)`；失败加 `compilation failed:` 前缀。
8. `Prepare(runtime, std::move(artifact), options.limits)`；失败加 `preparation failed:` 前缀。

### 4.2 `ExecutableModel::Prepare`

1. 要求 `artifact.loaded_model != nullptr`，否则 `InvalidArgument`。
2. 要求 config 的 `vocab_size` 与 `max_position_embeddings` 为正且可表示为 `size_t`；否则 `InvalidArgument` / `Overflow`。
3. `CollectModelUsage`：`ComputeRawWeightStorageUsage` 统计 raw backing（无 cap 时允许不完整观测）；按 `inline_data` 的 vector 指针去重累计常量字节；得到 `managed_payload_bytes`。
4. `PackingLimit`：显式 packed cap 与 model cap 取较小约束；model cap 要求 raw footprint 完整，否则 `Unimplemented`；超出 model cap 返回 `ResourceExhausted`。
5. `PackModelWeights`：`BuildWeightPackingRequests`；device 取首个 request 的 selector，无 request 时取首个 lowered step，空图才默认 CPU；`runtime.GetBackend(device)`；`ResolveWeightPackingRequests`；`runtime.TryGetAllocator(Device(device))`；`PrepackWeightRequests(..., packed_cap)`。
6. 累加 `packed_weight_bytes` 与 `managed_payload_bytes`，记录 `packing_plan_complete`。
7. `ExecutionPlanBuilder::Build(runtime, packed_weights, artifact.graph)`；plan 自己持有 packed artifact 的 `shared_ptr`。
8. `ResolveArtifactPhase`：以首个 step 的 `selector.phase` 为准；任一步 phase 不同即 `FailedPrecondition`。
9. `BuildModelBindings`：`ComputeExternalReadRequirements(plan)` 是唯一需求权威；跳过 `kModelInput`；对 `kWeight` 取 artifact value 的 `WeightValue.binding` 交给 `ResolveWeightBinding`，对 `kConstant` 校验 `inline_data`、静态 shape、维度、乘积与字节数后物化；shape/stride 写入 `binding_metadata_` 并被 `TensorView` 借用。
10. 值下标同一性是准备期前提：plan value index 必须与 artifact value index 1:1 对应；越界或 payload 类别不符返回 `Internal`。
11. 双向对账 `{required 且非 kModelInput}` 与 `{已生成绑定}`；任一方向不等即 `Internal`。
12. 按成员声明顺序构造 `ExecutableModel`；失败路径不泄漏半成品。

### 4.3 `InferenceSession::Create` 与 plan I/O 合同

`Create` 依次检查：模型非空（`InvalidArgument`）、`model->IsPreparedFor(runtime)`（`FailedPrecondition`）、`context_limit()` / `vocab_size()` 非零（`FailedPrecondition`）、取 `kPrefill` 与 `kDecode` plan，再对两份 plan 各执行一次 `ValidatePlanContract`。允许 Runtime 没有 `KVCacheManager`；该缺失只在请求正数上限时报告。

`ValidatePlanContract` 的合同：

| 检查 | 失败码 |
|---|---|
| `model_inputs().size() == 2` 且 `model_outputs().size() == 1` | `Unimplemented` |
| 两个输入 id 互异、在值表范围内、`kind == kModelInput` | `InvalidArgument` |
| 输入 dtype `Int64`、rank 1 | `Unimplemented` |
| 唯一输出 `kind == kActivation`、`Int64`、rank 1 | `Unimplemented` |
| 每个 step 的 `selector.device_type == kCPU` | `Unimplemented` |
| 全 plan 恰好一个 `kArgmax`，且其唯一输出就是模型输出 | `Unimplemented` |

输入角色是位置合同：`model_inputs()[0]` 绑定 token_ids，`[1]` 绑定 position_ids，不依赖 debug name。两个输入 dtype/rank 相同，互换无法在此识别；新增家族或改变输入顺序时必须同步维护该合同。

### 4.4 `Generate`

```text
moved-from / 空 prompt / prompt 与 EOS 词表 / prompt ≤ context_limit
  → max_new_tokens == 0：返回空结果（不取 KV manager、不执行）
  → future_kv_appends = N - 1；checked add 求 prompt + appends，再比 context_limit
  → 取 KVCacheManager 与 CPU allocator provider（缺失即 FailedPrecondition）
  → 取 prefill/decode plan 与两份 immutable bindings
  → workspace_bytes = max(两 phase workspace bytes)
  → 若有 request cap 或 Runtime workspace limit：先 ValidateReservation
  → 若 workspace limit 非零且超出：ResourceExhausted
  → RequestResources：有 request cap 时先 PlanBindings + 预算校验 + Reserve；无 request cap 时直接 Reserve
  → 填 prefill token/position（position = 0..prompt_len-1），reserve 输出 capacity
  → InitializeWorkspace(max(两 phase bytes), max(两 phase alignment))
  → PrepareContext(Prefill) 或复用已规划的 layout → RunPrefill
  → push 首 token → ClearContext()
  → 已达 N 或首 token 即 EOS：Finish()
  → 写 decode buffer（token = 首 token，position = prompt_len）
  → PrepareContext(Decode) 一次 → RunDecodeLoop → Finish()
```

`RunPrefill` 的固定合同：

1. `prompt_length` 在 `[1, INT64_MAX]`，`vocab_size > 0`，否则 `InvalidArgument`。
2. `ExecutionContext` 必须带有效 `KVCacheSessionView`，lifecycle 为 `AwaitingPrefill`、当前水位为 0、reservation 的 prompt length 等于 `prompt_length`，否则 `FailedPrecondition`。
3. 重置 workspace（存在时），`Executor::Execute`，原样传播执行错误；成功后要求 committed watermark 等于 `prompt_length`，否则 `Internal`。
4. `ReadOutputToken` 读取输出 tensor 最后位置，返回首 token；输出视图必须是有效 `Int64` rank-1、长度等于期望、连续，token 必须可收窄到 `uint32_t` 且小于 vocab，否则 `Internal` / `OutOfRange`。

`RunPrefill` 不接收 `generated_tokens`、`PhaseContract` 或请求级输入 buffer；这些由 `Generate` 准备。若 plan 已提交 KV、随后 output 校验失败，helper 不回滚 watermark；`RequestResources` 会清除 context 并释放整个请求 reservation。

### 4.5 Decode 稳态循环

```text
while generated_tokens.size() < max_new_tokens:
  若上一个 token 是 EOS → Ok（不再执行）
  *input_token = 上一个 token                 // 只写内容，地址不变
  workspace->Reset()（存在时）
  Executor::Execute(decode_plan, context)
  checked add 求 expected = *next_pos + 1
  校验 current_pos() == expected，否则 Internal
  ReadOutputToken(长度 1) → push
  若该 token 是 EOS → Ok
  若还会继续生成：*next_pos 自增（Int64 上限保护）
```

循环内不得变化：plan identity、输入/输出地址、shape/stride/dtype、prepared kernel params、workspace 基址、packed artifact identity。`next_position` 只在还会继续生成时自增，最后一次迭代不会把 position 推到未使用的位置。

### 4.6 KV 预约算术与零分配

Prefill 预测出的首 token 尚未写入 KV，因此生成上限 N 只需要 `N - 1` 次 Decode append：预约量为 `prompt_len + (N - 1)`。`KVCacheManager::ReserveForSession(prompt_len, future_kv_appends)` 的第二个参数就是“prompt 之后最多追加的 KV positions”；所有加法经 checked add，溢出返回 `Overflow` 且发生在预约之前。`ReserveForSession` 同时校验未初始化、已有 reservation、空 prompt、物理容量与 generation 耗尽；`ValidateReservation` 只做同一检查，不改变状态。

Decode 稳态的零分配前提在循环前全部满足：`generated_tokens` 已 `reserve(max_new_tokens)`；Decode token/position buffer 长度为 1；workspace 已按两 phase 最大值分配；Decode context 已 prepare 一次；`ExecutionContext` 的 KV layer transaction scratch 已在 `Create` 时按 KV 几何预分配。循环内只改输入内容与 KV commit 位置，不调用 `PrepareExecutionBindings`、不重建绑定、不扩容输出。

关键错误语义：

| 情形 | 错误码 |
|---|---|
| `Prepare` 缺 `loaded_model`、config 的 vocab/context 非正 | `InvalidArgument` |
| 权重不可解析、常量缺 inline/形状或字节不符、phase 混合 | `FailedPrecondition` |
| 绑定需求与生成绑定双向对账不一致、plan/artifact 值下标错位 | `Internal` |
| `Generate` 的 Session 已移动/无 Runtime、缺 KVCacheManager 或 CPU allocator | `FailedPrecondition` |
| 空 prompt、tokens/positions 长度不等 | `InvalidArgument` |
| prompt/EOS 超 vocab、prompt 或 prompt+appends 超 `context_limit` | `OutOfRange` |
| `max_new_tokens == 0` | Ok（空结果，不预约、不执行） |
| token 总数或 KV 位置加法溢出 | `Overflow` |
| plan I/O 合同不符（数量、dtype、rank、非 CPU step、Argmax 拓扑） | `Unimplemented` |
| plan 值 id 越界/重复、输入 kind 不符 | `InvalidArgument` |
| workspace alignment 非法、workspace 超限、activation/请求分配失败 | `InvalidArgument` / `ResourceExhausted` |
| Prefill 未提交完整 prompt、Decode 未恰好提交一个 position、输出视图不符 | `Internal` |
| Argmax token 为负、超 vocab 或不可收窄为 `uint32_t` | `OutOfRange` |
| 传播至 `Generate` 边界的 `std::bad_alloc` | `ResourceExhausted`（空消息） |

### 4.7 C ABI / CLI 关系与所有权

| 入口 | 路径 | 关键事实 |
|---|---|---|
| C++ 目录入口 | `ExecutableModel::Load` | 一次性完成 Load → 可选转换 → Compile → Prepare；CLI 与 C ABI 不使用该入口，而是走分阶段路径 |
| CLI | [inference_runner.cpp](../../apps/inference/inference_runner.cpp) | `ModelLoader::Load` → `ValidateSourceWeights(kRequireFp32)` → `MakeCpuRuntimeOptions` / `MakeCompileOptions(CPU)` → `ModelCompiler::Compile` → `RuntimeBuilder::TryBuild` → `ExecutableModel::Prepare` → `InferenceSession::Create` → `Generate`；Runtime 在 Prepare 前构建并保持同一地址 |
| C ABI | [c_api_runtime.cpp](../../src/c_api_runtime.cpp) | `am_runtime_create` 拥有 `RuntimeState{Runtime, preset}`；`am_model_load` 执行 `ModelLoader::Load` → `ValidateSourceWeights(kRequireFp32)` → `MakeCompileOptions(preset, CPU)` → `Compile` → 私有 `ValidateRuntimeModelKV` → `Prepare`；`ModelState` 以 `shared_ptr` 同时保活 Runtime 与 model |
| C ABI session | 同上 | `am_session` 持有 `ModelState`、`GenerationConfig` 与 `InferenceSession`；`am_session_generate` 要求 output capacity ≥ `max_new_tokens`，调用 `Generate` 后拷贝 token IDs 并映射 Status |

CLI 与 C ABI 继续 strict FP32：CLI 显式 `ValidateSourceWeights(kRequireFp32)`；C ABI 的 `am_config_t.kv_dtype` 只接受 `AM_KV_DTYPE_FLOAT32`，并在模型加载时调用 `ValidateRuntimeModelKV` 校验 KV manager 已初始化、Runtime/model 的 layer/kv-head/head_dim 一致、每层 K/V state 根齐全且 state dtype 为 F32。retained-W16/KV16 只通过 C++ `ModelLoadOptions` 暴露，C ABI 不承诺；C ABI 合同见 [runtime-c-abi.md](../api/runtime-c-abi.md)。

所有权链：`Runtime > ExecutableModel > InferenceSession > RequestResources/ExecutionContext > PreparedExecutionBindings`。C ABI 的 `ModelState` 共享 Runtime 与 model，`am_session` 再共享 `ModelState`，因此句柄销毁顺序不会让 model 先于 Runtime 释放；C++ 调用方必须自行保证同一顺序。`ExecutableModel` 的 resolved kernel 函数指针借用 backend 状态，Runtime 不得移动。

## 5. 关键取舍

| 决策 | 选择 | 原因 | 被否方案 |
|---|---|---|---|
| 准备入口 | `ExecutableModel::Prepare` 是唯一生产入口，`Load` 只做目录编排 | 调用方只持有可执行模型，不接触 `LoweredGraph` / role / packing；失败集中在冷路径 | 暴露自由函数或多套目录包装入口；会让所有权与校验分叉 |
| phase plan | 内部只构建一个 plan，按 `PhaseMatches` 提供 Prefill/Decode 查询 | 当前 Llama 两 phase 共享同一计划；查询仍拒绝不匹配 artifact，避免静默复用 | 立即构建两份 phase plan；增加准备成本且当前无收益 |
| 失败时机 | 权重不可解析、常量无 inline、phase 混合等在 prepare 期失败 | 避免把结构错误推迟到执行期变成错误数值 | 延迟到首次 `PrepareExecutionBindings` / `Execute`；错误定位更差 |
| 绑定需求 | `ComputeExternalReadRequirements` 是唯一需求权威，准备期双向对账 | 绑定表与实际 kernel 需求不可能分叉；packed weight 自然不进绑定表 | Session/prepare 各自推导需求；会漏绑或重复绑定 |
| 绑定元数据 | `WeightBindingMetadata` 独立拥有 shape/stride 数组 | 一份 immutable 表可被 Prefill/Decode 多次 specialize；视图跨扩容/移动稳定 | 每次提交内联 `std::array` 或 small-buffer；扩容/移动会使视图悬垂 |
| 模型输入 | 不进 immutable 绑定表，由 Session 按 phase 追加 | 同一模型可服务不同请求；token/position 地址归请求所有 | 把 Session 输入存进模型；会引入跨请求状态与并发问题 |
| 请求资源 | 每请求一个 `RequestResources`，Prefill 后清 context，Decode 只 prepare 一次 | KV reservation、workspace 与输出 capacity 归请求；Decode 稳态零分配 | 每步重新 `PrepareExecutionBindings`；会产生分配与性能抖动 |
| KV 预约 | `prompt_len + (N - 1)` | Prefill 首 token 不在 KV 中；少预约一个位置，恰好覆盖 Decode append | `prompt + N`；会浪费一个物理 KV 位置并使容量边界更保守 |
| workspace | 取 Prefill/Decode 最大值，一块 buffer 服务两阶段 | 避免两套分配；`SessionWorkspaceArena` 只借用，不拥有 | 每 phase 单独分配；增加分配与峰值 |
| 共享策略 | `ValidateSourceWeights` / `MakeKVCacheOptions` / `CompilePreset` 由 CLI、C ABI、C++ 共用 | 产品准入与映射只有一份实现；避免入口间漂移 | CLI/C ABI 各自复制 FP32/KV 映射；会产生不一致 |
| C ABI 精度 | C ABI 继续 strict FP32，W16/KV16 仅 C++ | C ABI 需要稳定合同；W16 仅由 C++ 准备路径支持 | 立即把 W16 暴露到 C ABI；会把未承诺的策略固化进 ABI |
| context 上限 | `context_limit = max_position_embeddings` | 保守、来自已验证 HF config；不承诺 RoPE 外推 | 使用 RoPE scaling 推导更大窗口；当前无执行合同 |
| 内部 helper 暴露 | `ValidatePlanContract` / `RunDecodeLoop` 放 `inference_internal.h` | 聚焦测试可用合成 plan 覆盖失败分支；malloc 计数窗口只包住循环本体 | 放进公共头；会扩大稳定 API 面 |

## 6. 测试与验证入口

构建并运行聚焦测试：

```bash
cmake --build build --target aethermind_unit_tests -j
./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='ExecutableModel.*:ModelLoadPath.*:ModelLoad.*:WeightBindingMetadata.*:Fp32CompatibilityPolicy.*:W16*.*:Prefill.*:PrefillInvalidToken.*:TinyLlamaExecution.*:PlanContractValidation.*:DirectPrefillDecode.*:InferenceResourceLimits.*:ModelResourceLimits.*:CpuRuntimeAssembly.*:CpuRuntimeConfiguration.*:Fp32KVCacheOptions.*:PackingRequestResolution.*:KV16ModelAcceptance.*:KV16CalibratedAcceptance.*'
```

| 测试套件 / 过滤 | 文件 | 覆盖点 |
|---|---|---|
| `ExecutableModel.*` | [test_executable_model.cpp](../../tests/unit/inference/test_executable_model.cpp) | plain artifact 准备、绑定与需求双向对账、tied/untied lm-head、多层独立 backing、常量物化/失败、packed 子图、phase 查询、移动/不可赋值、teardown 顺序 |
| `ModelLoadPath.*`、`ModelLoad.*` | [test_load_prepare_executable_model.cpp](../../tests/unit/inference/test_load_prepare_executable_model.cpp) | HF 目录 O1/plain 与 O2/packed 到三 token Generate；错误码保留与前缀；非 CPU target 透传；缺 backend 报告 |
| `WeightBindingMetadata.*` | [test_weight_binding_metadata.cpp](../../tests/unit/inference/test_weight_binding_metadata.cpp) | 借用视图、紧凑 row-major stride、标量对齐、rank-0/空张量、溢出、失败保留既有视图、扩容/移动稳定性 |
| `Fp32CompatibilityPolicy.*` | [test_fp32_compatibility_policy.cpp](../../tests/unit/inference/test_fp32_compatibility_policy.cpp) | 混合 scalar checkpoint、auto hint、strict FP32、缺角色/layer 不匹配、非 scalar/向量 dtype 拒绝 |
| `W16WeightPolicy.*`、`W16PreparedWeightPolicy.*`、`W16Inference.*`、`W16Preparation.*` | [test_w16_weight_policy.cpp](../../tests/unit/inference/test_w16_weight_policy.cpp)、[test_w16_inference.cpp](../../tests/unit/inference/test_w16_inference.cpp) | 仅 Norm 加宽、矩阵 dtype 准入、prepared 校验、W16 与 FP32 轨迹/Generate 一致、O2 packed 数值一致、Decode 零 malloc、源作用域释放后权重保活 |
| `Fp32KVCacheOptions.*` | [test_model_execution_policy.cpp](../../tests/unit/inference/test_model_execution_policy.cpp) | KV 几何映射、显式/推导 head_dim、容量边界、错误维度、独立于 config 生命周期 |
| `CpuRuntimeAssembly.*`、`CpuRuntimeConfiguration.*` | [test_cpu_inference_options.cpp](../../tests/unit/inference/test_cpu_inference_options.cpp) | 仅 CPU/F32 资源选择、几何映射、零字节 cap、KV cap 先于 payload、feature policy 到达 backend、builder 使用 owning snapshot |
| `PackingRequestResolution.*` | [test_weight_packing_request_resolution.cpp](../../tests/unit/inference/test_weight_packing_request_resolution.cpp) | 同 recipe consumer 合并、recipe 冲突拒绝、不同 weight value 保持分离 |
| `Prefill.*`、`PrefillInvalidToken.*` | [test_prefill.cpp](../../tests/unit/inference/test_prefill.cpp) | 最后位置预测、参数/重复 Prefill 拒绝、执行错误不提交、缺 append 后置水位错误、提交后输出失败保留 watermark |
| `TinyLlamaExecution.*`、`PlanContractValidation.*`、`DirectPrefillDecode.*` | [test_direct_prefill_decode.cpp](../../tests/unit/inference/test_direct_prefill_decode.cpp) | 真实 tiny tied-GQA Llama 的 scalar oracle 对照、两次调用重新 Prefill、N=1/EOS/精确 KV 容量、上下文与溢出拒绝、分配失败可重试、Runtime 不匹配、Decode 循环 malloc 计数、plan 合同全部失败分支 |
| `InferenceResourceLimits.*`、`ModelResourceLimits.*` | [test_inference_resource_limits.cpp](../../tests/unit/inference/test_inference_resource_limits.cpp) | 规划不分配 payload、预算在预约前拒绝、workspace limit、零生成、容量/占用优先、model/packed cap、inline constant 去重、共享 owner 计费 |
| `KV16ModelAcceptance.*`、`KV16CalibratedAcceptance.*` | [test_kv16_model_acceptance.cpp](../../tests/unit/inference/test_kv16_model_acceptance.cpp)、[test_kv16_calibrated_acceptance.cpp](../../tests/unit/inference/test_kv16_calibrated_acceptance.cpp) | KV16 组合与 KV32 的限定容差、长上下文、GQA tied 几何、Decode 无 malloc、独立量化 oracle |
| CLI / C ABI 集成 | `CliMatrix/InferenceCliIntegration.*` | 需要 `AETHERMIND_INFER_BINARY` 指向独立 CLI；缺 CLI、fixture 或平台条件会显式失败 |

`TinyLlamaExecution.*` 的四项分配失败测试与 `SharedDecodeLoopHasNoMallocFamilyCallsAfterPreparation` 需要 glibc/Linux malloc interposer；`ExecutableModel.PreparedBindingsAreReleasedBeforeTheModel` 在 ASAN/TSAN 下验证 teardown 顺序。KV/W16 数值验收的容差与证据边界见 [W16 指南](../guides/w16-fp32-inference.md)。