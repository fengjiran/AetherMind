# backend 模块设计

> 本文描述已验证实现。分层与依赖方向见 [architecture.md](../architecture.md)，运行时资源装配见 [runtime 模块设计](runtime.md)，plan 构建与逐 step 执行见 [execution 模块设计](execution.md)，跨层纯数据契约见 [foundation 模块设计](foundation.md)，产品范围见 [product.md](../product.md)。

## 1. 职责与边界

backend 是**设备族执行能力与 kernel 目录层**：它登记 kernel descriptor，在 plan-build 阶段按 `KernelSelector` 与 CPU 有效特征集解析出冻结的 `ResolvedKernel`，并提供权重打包、CPU 能力探测与 workspace arena 实现。它不拥有模型、执行计划或 KV 存储，也不控制执行循环。

| 负责 | 说明 |
|---|---|
| kernel 注册目录 | `KernelRegistry` 保存 `KernelDef`，按 `OpType` 建桶索引，`Freeze()` 后只读 |
| kernel 解析 | `Backend::PrepareKernel` 是唯一正式入口；CPU 实现按结构匹配 + `effective_features` 过滤 + priority 选择 |
| descriptor 校验 | `ValidateKernelDef` 校验 op、入口、名称、device、CPU 特征声明、packing recipe 与 params 契约 |
| CPU 能力模型 | `CpuFeature` / `CpuFeatureSet` / `CpuCapabilities` / `CpuFeaturePolicy`；`DetectCpuCapabilities` 探测 hardware/usable/effective 三层快照 |
| 权重打包 | `Backend::GetPackingRecipe` / `GetWeightPackingRequirements` / `PackWeights`；CPU 实现 identity 与 B-panel 布局 |
| 打包布局身份 | `PackingLayout` 枚举与 `PackingRecipe`；CPU 布局公式集中在 `cpu_packed_weight_layout.h/.cpp` |
| 执行期窄契约 | `KernelContext`、`KernelFunc`、`KernelParamsBuilder`、`KernelMetadataBuilder`；kernel 只读取窄句柄 |
| CPU 执行辅助 | `CpuWorkspaceArena`（预分配 arena 的绑定视图）、FP16/BF16 host conversion 函数表选择 |
| backend 装配 | `BackendFactory` / `BackendRegistry`；实例由 `Runtime` 持有 |

**不得负责什么**：

| 不负责 | 实际归属 |
|---|---|
| 模型加载、HF I/O、模型家族判定、语义图构建 | model |
| 图改写、常量折叠、融合、lowering | graph / compiler |
| `ExecutionPlan` / `StateAliasPlan` / `PreparedExecutionBindings` / 执行循环 / Argmax | execution / inference |
| KV 物理 layout、storage、reservation、commit watermark | kv_cache（`KVCacheManager` / `KVCacheSessionView`）；kernel 只消费 base 的窄 KV binding |
| Runtime 装配、allocator/backend registry 生命周期、workspace offset 规划 | runtime / execution |
| 模型语义、token/position 语义、tokenizer、C ABI | inference / C adapter |
| 权重打包时机与 recipe 选择策略 | inference/model 的准备流程；backend 只提供候选与执行打包 |
| 热路径 registry 查询、哈希/字符串查找、对象虚调用 | 无；plan-build 后只调用冻结函数指针 |

边界规则：

- backend 只依赖 base、dtypes、memory、operators 的纯数据契约（`OpType`、`OpParams`、`kXxxSupportedDTypes`/`IsXxxSupportedDType`）与 backend-independent 的 `rope_frequency_resolver.h`；**不得依赖 graph、compiler、model、execution、inference**。
- `execution` 与 `model` 可直接 include backend 的纯数据头（`ResolvedKernel`、`PackedWeight`、`PackingRecipe`），但 backend 不反向依赖它们。
- `KernelRegistry::Global()` 是 process-wide descriptor catalog，不是 `Runtime` 的资源 owner；`BackendRegistry` 才由 `Runtime` 持有。
- kernel 不得持有 `Runtime*`、`InferenceSession*`、`ExecutionPlan*` 等宽对象；`KernelContext` 不暴露后端专属类型。
- backend 不长期拥有 `Tensor` / `Buffer` 生命周期，也不持有 `PackedWeight` 产物；`PackWeights` 返回的产物连同其 `Buffer` 移交给 `PackedWeightCollection` 或调用方。
- 当前 CPU kernel 均未声明非零 `ResolvedKernel::workspace_requirement`；workspace 需求字段为默认空值（`bytes == 0`、`alignment == 64`、`offset == 0`）。地址绑定与容量管理不属于 backend。

## 2. 关键数据结构与所有权

### 2.1 选择、注册与解析

| 类型 | 关键内容 | 所有权/不变量 |
|---|---|---|
| `KernelSelector` | `device_type`、`act_dtype`、`weight_dtype`、`state_dtype`、`weight_format`、`phase` | base 层纯数据；`state_dtype` 按严格相等匹配，未定义值不是通配符；`PhaseMatches(candidate, request)` 允许候选 `kBoth` 服务 `kPrefill`/`kDecode` |
| `KernelDef` | `op_type`、`selector`、`packing_recipe`、`cpu_requirements`、`kernel_func`、`priority`、`params_size`、`params_builder`、`metadata_builder`、`name` | 可平凡拷贝的纯数据；`name` 为 `std::string_view`，必须借用静态存储；plain 必须没有 recipe，packed 必须携带已知非空布局与合法对齐 |
| `KernelFunc` | `Status (*)(const KernelContext&) noexcept` | 类型擦除入口；参数从 `kernel_params`/`attrs` 读取，packed weight、workspace、KV 与 stream 通过 `KernelContext` 的其他窄句柄提供 |
| `KernelParamsBuilder` | `Status (*)(const KernelParamsBuildContext&, void*) noexcept` | 每 step 在绑定准备期调用一次；params 必须在 `params_buffer` 中 placement-construct，且可平凡析构 |
| `KernelMetadataBuilder` | `Status (*)(const OpParams&, std::vector<std::byte>&)` | 在 plan-build 期由 `CpuBackend::PrepareKernel` 调用；结果复制进 `ResolvedKernel::attrs` |
| `ResolvedKernel` | `op_type`、`fn`、`attrs`、`name`、`params_builder`、`params_size`、`workspace_requirement`、`expected_packing_recipe` | 值类型；**拥有** `attrs` 字节；**借用** `name`；由 `ExecutionStep` 持有并重复调用 |
| `KernelContext` | `device_type`、`stream`、`workspace`、`workspace_binding`、`packed_weights`、`kernel_params`、`kv_append`、`kv_read`、`attrs` | 单次调用的窄句柄；workspace、params、packed weight、KV binding 均为借用；KV binding 只在本次同步调用内有效，禁止缓存 |
| `KernelRegistry` | `kernels_`、`buckets_`（`OpType` → 下标）、`registration_keys_`、`frozen_`、`mutex_` | 全局 function-local static singleton；`Register`/`Freeze` 加锁；`frozen_` 为 release/acquire 同步点；冻结后 `FindCandidates` 无锁读 |
| `RegistrationKey` | `op_type + selector + cpu_requirements` | 重复注册判定包含特征集；scalar 与 AVX2 变体可同 selector 共存 |
| `BackendFactory` / `BackendRegistry` | factory 按 `DeviceType` 注册；`backends_` 懒创建并缓存 | registry move-only、无内部锁；`SetFactory` 驱逐同 device 缓存实例 |

关键不变量：

- `ResolvedKernel::name` 借用 `KernelDef::name`，因此 `KernelRegistry`（或至少静态字符串）必须比 plan 活得久。
- `FindCandidates` 返回借用 registry 的 `const KernelDef*`；冻结后不再追加 descriptor，指针在 registry 生命周期内稳定。
- `KernelRegistry::Freeze()` 幂等；冻结后 `Register` 返回 `FailedPrecondition`，冻结前 `FindCandidates`/`FindByOpType` 返回 `FailedPrecondition`。
- `FindCandidates` 只做结构匹配，不按 `cpu_requirements` 过滤；过滤是 `CpuBackend` 的职责。
- `KernelDef::params_builder` 与 `params_size` 必须同时为空/零或同时非空/非零，且 `params_size <= kMaxKernelParamsSize`（512）。
- `KernelMetadataBuilder` 只生成不可变 attrs；执行期不读取 `OpParams`，也不分配 attrs。

### 2.2 Packed weight 与布局身份

| 类型 | 关键内容 | 所有权/语义 |
|---|---|---|
| `PackingLayout` | `kNone`、`kCpuIdentity`、`kCpuBPanelF32Kc512Nr16` | 每个非 `kNone` ID 对应不可变物理字节格式；改 tile、panel 顺序或 padding 必须新增 ID |
| `PackingRecipe` | `layout` + `alignment` | 精确身份匹配；默认 `kNone`/`0`；`PackingRecipe::operator==` 比较两个字段 |
| `PackedWeightView` | `data`、`nbytes`、`logical_dtype`、`logical_shape`、`recipe_layout`、`recipe_alignment`、`alignment` | 绑定期视图；payload 不暴露为 `TensorView`，logical metadata 只用于校验 |
| `PackedWeight` | `op_type()`、`selector()`、`storage()`、`recipe()`、`logical_dtype()`、`logical_shape()` | 抽象产物；`storage()` 为自有 `Buffer`；由 `PackedWeightCollection` 或调用方持有，backend 不持有 |

布局身份规则：

- `PackingRecipe::layout` 与 `PackedWeightView::recipe_layout` 共用 `PackingLayout`；`kNone` 表示未指定。生产 packed descriptor/request 必须携带已知非空布局，未知枚举值必须在边界拒绝。
- 布局名称按 dtype、blocking、必要的 panel 顺序或 padding 等具体格式特征区分，不使用 `V1`/`V2` 这类通用版本号；改变物理格式必须新增 ID。
- `ValidateKernelDef` 拒绝 packed descriptor 的 `kNone`/未知布局、`alignment < alignof(void*)` 或非 2 的幂；plain descriptor 只要声明了布局或非零 alignment 即拒绝。
- `ToString(PackingLayout)` 仅用于诊断，不参与身份匹配；alignment 与 ISA eligibility 按各自契约处理。
- `cpu_identity` 对齐 64 字节，存储连续逻辑字节；单组件接受任意 rank（含 scalar/rank-1），多组件按 axis 0 拼接。
- `cpu_bpanel_f32_kc512_nr16` 只接受 rank-2 FP32、相同 K 的组件；物理轴序为 `[K-panel][N-block][K-row][N-lane]`，逻辑 shape 为 `[ceil(K/512), ceil(N/16), 512, 16]`，N lane 连续，超出逻辑 K/N 的 padding 写 `+0.0F`，每个 panel block 起始 64 字节对齐。

### 2.3 CPU 能力

| 类型 | 关键内容 | 语义 |
|---|---|---|
| `CpuFeature` | x86：SSE4.1、AVX、AVX2、FMA、F16C、AVX512F/BW/VNNI/BF16、AVXVNNI、AMX TILE/INT8/BF16；AArch64：NEON、FP16、DotProd、I8MM、BF16、SVE、SVE2 | 扁平 append-only；特征互不隐含；`kCount` 为哨兵 |
| `CpuFeatureSet` | 两个 `uint64_t`，`kWordCount = 2` | 无堆分配、`constexpr`/`noexcept`；`Contains`、`ContainsAll`、`Intersect`、`Difference`、`Hash` |
| `CpuCapabilities` | `architecture`、`hardware_features`、`usable_features`、`effective_features`、`sve_vector_bytes` | 值类型快照；不变量 `hardware ⊇ usable ⊇ effective`；kernel 选择只看 `effective_features` |
| `CpuFeaturePolicy` | `disabled_features`、`required_features` | 只能削减，不能启用；required 必须留在 effective |

`CpuFeature` 枚举位置 append-only；新增特征必须同步扩展 `cpu_capabilities.cpp` 的名字表，否则 `ToString(CpuFeature)` 返回 `"Unknown"`。`sve_vector_bytes` 是创建快照线程的 SVE VL（字节），不参与 kernel 选择。

检测入口与平台门控：

| 平台 | 检测来源 | 进入 `usable` 的关键门控 |
|---|---|---|
| x86-64 | cpuid leaf 0/1/7、XCR0 | AVX 需 OSXSAVE + XCR0 bits 1-2；FMA/F16C/AVX2/AVXVNNI 随 AVX 状态；AVX512 需 XCR0 `0xE6`；AMX 需 `arch_prctl(ARCH_REQ_XCOMP_PERM)` 授权且 XCR0 bit17/18；`leaf0.EAX < 1` 或 `< 7` 提前返回 |
| AArch64 Linux | `getauxval(AT_HWCAP/AT_HWCAP2)` | NEON 默认可用；FP16/DotProd/I8MM/BF16 按 HWCAP 位；SVE 需 `prctl(PR_SVE_GET_VL) >= 0`，SVE2 仅在 SVE 已 usable 时可用 |
| AArch64 Apple | `sysctlbyname("hw.optional.arm.*")` | NEON 恒可用；SVE/SVE2 不检测 |
| 其他 AArch64 | 编译期 `__ARM_FEATURE_*` 宏 | 宏存在即记录 |
| 未支持平台 | 无检测器 | 返回空快照（不是错误）；空 policy 得到空 effective 集 |

硬件的 `hardware_features` 始终记录 cpuid/hwcaps 原始位；`usable_features` 才叠加 XCR0、AMX 权限、SVE VL 等 OS/进程门控。`DetectCpuCapabilities(policy)` 等价于 `ApplyCpuFeaturePolicy(DetectUsableCapabilities(), policy)`；未知平台返回空快照而非 `Status` 错误。

注入路径：`CpuBackendFactory(policy)` 在每次 `Create()` 时重新检测并返回 `Status`；`CpuBackend(policy)` 在构造时检测，失败 `AM_CHECK` 终止；`CpuBackend(const CpuCapabilities&)` 直接使用调用方快照，不再检测。`RuntimeOptions.backend.cpu_feature_policy` 是运行时正式装配入口。

### 2.4 所有权与生命周期

```text
KernelRegistry::Global()                 进程级；首个 CpuBackend 构造时 Freeze
Runtime
  └─ BackendRegistry
       └─ Backend (CpuBackend)            按值持有不可变 CpuCapabilities 快照
ExecutionPlan
  └─ ExecutionStep
       └─ ResolvedKernel                 attrs 自有；name 借用 KernelDef
PackedWeightCollection
  └─ PackedWeight → Buffer              产物归 collection；plan 共享 shared_ptr
ExecutionContext / LayerRunner
  └─ KernelContext                       借用 workspace/params/packed/KV binding
```

线程安全与并发边界：

- `KernelRegistry` 的 `Register`、`Freeze`、`DebugDump` 使用 `std::mutex`；冻结后 `FindCandidates`/`FindByOpType` 通过 acquire load 读取桶索引，无锁。
- `Backend` 的能力快照在构造后不可变；`Backend::PrepareKernel` 是 `const`，可在 plan-build 阶段被调用。
- `BackendRegistry` 与 `Runtime` 无内部锁；装配、查询与借用必须由调用方串行化。替换 factory 会驱逐缓存 backend，不得在已有 backend 被借用时执行。
- kernel 函数不得保存 `KernelContext` 或其借用指针；backend 不提供 kernel 间同步，当前执行模型为单请求同步、允许算子内部并行。
- `AM_REGISTER_KERNEL` 的静态注册在静态初始化期执行；注册失败通过 `AM_CHECK` 终止进程，而不是向调用方返回 `Status`。

## 3. 接口与代码位置

| 接口 | 语义要点 | 代码位置 |
|---|---|---|
| `Backend::device_type()` | 返回后端处理的 `DeviceType` | [backend.h](../../include/aethermind/backend/backend.h) |
| `Backend::PrepareKernel(op_type, selector, params)` | 唯一正式 kernel 解析入口；冻结 metadata、params 契约、workspace requirement 与 packed recipe | 同上 / [cpu_backend.cpp](../../src/backend/cpu/cpu_backend.cpp) |
| `Backend::GetPackingRecipe(op_type, selector)` | 模型准备期查询；必须复用与 `PrepareKernel` 相同的 eligibility/priority 策略 | 同上 |
| `Backend::GetWeightPackingRequirements(op, components, selector, recipe)` | 只读 metadata 的存储需求查询；不读 payload、不分配；结果不含 allocator metadata 与容量取整 | 同上 |
| `Backend::PackWeights(op, components, selector, recipe, allocator)` | 唯一打包入口；`components` 按逻辑绑定顺序；默认实现返回 `Unimplemented` | 同上 |
| `Backend::GetHostFp16Conversions()` / `GetHostBf16Conversions()` | host 转换函数表；默认 scalar；`CpuBackend` 按 effective features 选择 F16C/AVX2 | 同上 |
| `Backend::TryGetKernelRegistryForDebug()` | 仅调试/自省；生产执行不得使用 | 同上 |
| `BackendFactory::Create()` | 通过 `Status` 通道创建 backend；检测/初始化失败可传播 | [backend_factory.h](../../include/aethermind/backend/backend_factory.h) |
| `BackendRegistry::SetFactory` / `GetBackend` | 注册 factory；按 device 懒创建并缓存 backend | [backend_registry.h](../../include/aethermind/backend/backend_registry.h) / [backend_registry.cpp](../../src/backend/backend_registry.cpp) |
| `KernelRegistry::Global` / `Register` / `Freeze` / `FindCandidates` / `FindByOpType` / `DebugDump` | 全局注册目录；结构匹配与只读索引 | [kernel_registry.h](../../include/aethermind/backend/kernel_registry.h) / [kernel_registry.cpp](../../src/backend/kernel_registry.cpp) |
| `ValidateKernelDef(def)` | descriptor 不变量校验；返回 `InvalidArgument` | [kernel_def.h](../../include/aethermind/backend/kernel_def.h) |
| `AM_REGISTER_KERNEL(unique_name, ...)` | 静态自注册宏；失败 `AM_CHECK` | [kernel_static_registration.h](../../include/aethermind/backend/kernel_static_registration.h) |
| `KernelFunc` / `KernelParamsBuilder` / `KernelMetadataBuilder` / `kMaxKernelParamsSize` | 类型擦除 kernel ABI；params 上限 512 字节 | [kernel_types.h](../../include/aethermind/backend/kernel_types.h) |
| `KernelContext` | 单次调用窄资源 | [kernel_context.h](../../include/aethermind/backend/kernel_context.h) |
| `ResolvedKernel` | plan 冻结的 kernel 句柄 | [resolved_kernel.h](../../include/aethermind/backend/resolved_kernel.h) |
| `PackingLayout` / `PackingRecipe` / `IsValidPackingLayout` / `ToString` | 布局身份与诊断 | [packing_recipe.h](../../include/aethermind/backend/packing_recipe.h) |
| `PackedWeight` | 打包产物抽象接口 | [packed_weight.h](../../include/aethermind/backend/packed_weight.h) |
| `CpuBackend` / `CpuBackendFactory` / `cpu_capabilities()` | CPU 解析、能力快照、policy 注入 | [cpu_backend.h](../../include/aethermind/backend/cpu/cpu_backend.h) / [cpu_backend.cpp](../../src/backend/cpu/cpu_backend.cpp) |
| `DetectCpuCapabilities` / `ApplyCpuFeaturePolicy` | 能力探测与策略应用 | [cpu_info.h](../../include/aethermind/backend/cpu/cpu_info.h) / [cpu_info.cpp](../../src/backend/cpu/cpu_info.cpp) |
| `CpuFeature` / `CpuFeatureSet` / `CpuCapabilities` / `CpuFeaturePolicy` | CPU 能力数据模型 | [cpu_capabilities.h](../../include/aethermind/backend/cpu/cpu_capabilities.h) / [cpu_capabilities.cpp](../../src/backend/cpu/cpu_capabilities.cpp) |
| `CpuIdentityPackingRecipe` / `CpuBPanelF32Kc512Nr16Recipe` / `CpuBPanelF32Kc512Nr16PackedByteSize` | CPU 布局契约与尺寸公式 | [cpu_packed_weight_layout.h](../../include/aethermind/backend/cpu/cpu_packed_weight_layout.h) / [cpu_packed_weight_layout.cpp](../../src/backend/cpu/cpu_packed_weight_layout.cpp) |
| `ValidateIdentityPackedWeight` / `ValidateBPanelF32PackedWeight` | 绑定期校验 recipe、alignment、逻辑 metadata 与存储大小 | [packed_weight_validation.h](../../include/aethermind/backend/cpu/packed_weight_validation.h) / [packed_weight_validation.cpp](../../src/backend/cpu/packed_weight_validation.cpp) |
| `CpuWorkspaceArena::Bind` / `Reset` | 预分配 arena 的切片绑定；`Reset` 为空操作 | [cpu_workspace_arena.h](../../include/aethermind/backend/cpu/cpu_workspace_arena.h) / [cpu_workspace_arena.cpp](../../src/backend/cpu/cpu_workspace_arena.cpp) |
| `KernelSelector` / `SelectorMatches` / `ExecPhase` / `WeightFormat` | 跨层纯数据选择契约 | [kernel_selector.h](../../include/aethermind/base/kernel_selector.h) / [kernel_attrs.h](../../include/aethermind/base/kernel_attrs.h) |
| `WorkspaceRequirement` / `WorkspaceBinding` / `WorkspaceArena` | 执行期 workspace 共享契约 | [workspace_types.h](../../include/aethermind/base/workspace_types.h) / [workspace_arena.h](../../include/aethermind/base/workspace_arena.h) |
| `KVCacheAppendBinding` / `KVCacheReadBinding` | kernel 的窄 KV 读写绑定 | [kv_cache_binding.h](../../include/aethermind/base/kv_cache_binding.h) |

## 4. 主流程/时序

### 4.1 注册与冻结

```text
kernel TU: AM_REGISTER_KERNEL(name, KernelDef{...})
  → 静态初始化 lambda
  → KernelRegistry::Global().Register(def)
       → ValidateKernelDef
       → RegistrationKey{op_type, selector, cpu_requirements} 去重
       → kernels_.push_back(def)
首个 CpuBackend 构造
  → KernelRegistry::Global().Freeze()
       → BuildBucketIndex()（OpType → 下标）
       → frozen_.store(true, release)
```

要点：

1. descriptor 的 `packing_recipe` 与 `selector.weight_format` 必须一致；packed descriptor 缺布局/对齐会在注册期失败。
2. `AM_REGISTER_KERNEL` 把注册失败视为启动错误，`AM_CHECK` 终止进程；不是可恢复的 `Status`。
3. 冻结是 process-wide 且幂等的；之后任何 `Register` 都返回 `FailedPrecondition`。因此 kernel 必须在首个 `CpuBackend` 构造前完成静态注册。
4. 测试或嵌入方若需要重新注册，必须在新进程内进行；当前实现没有 unfreeze/reset 入口。

### 4.2 plan-build 期 kernel resolve

```text
ExecutionPlanBuilder::Build
  → Runtime::GetBackend(selector.device_type)
  → Backend::PrepareKernel(op_type, selector, op_params)
       → CpuBackend 校验 selector.device_type == kCPU
       → KernelRegistry::Global().FindCandidates(op_type, selector)
            → 结构匹配 device/act/weight/state/weight_format/phase
       → effective_features.ContainsAll(candidate.cpu_requirements)
       → priority 取最大（相同 priority 先注册者胜）
       → 无候选：NotFound（携带 op、selector、effective_features）
       → packed selector：descriptor 必须有合法 packing_recipe
       → metadata_builder(params, attrs)
  → ExecutionPlanBuilder 校验 kernel.op_type、workspace offset/alignment
  → 冻结为 ExecutionStep 的 ResolvedKernel
```

选择顺序固定为：**结构匹配 → CPU 特征子集过滤 → priority 取舍**。`priority` 只表示“多个可用实现谁更优”，不承担表达能力匹配；CPU 指令集不是全序关系，因此不使用 `candidate.isa <= request.isa` 这类上限规则。priority 由各 descriptor 自行声明；相同 priority 时先注册者胜。

`GetPackingRecipe` 复用同一个 `ResolveEligibleDescriptor`，并要求 selector 为 CPU + `WeightFormat::kPacked`；缺少合法 recipe 返回 `Internal`。`GetWeightPackingRequirements` 与 `PackWeights` 会再次核对调用方传入的 recipe 等于 `GetPackingRecipe` 的结果，不等则返回 `InvalidArgument`。

### 4.3 权重打包

```text
模型准备
  → Backend::GetPackingRecipe(op, selector)
  → Backend::GetWeightPackingRequirements(op, components, selector, recipe)
       → 校验 recipe 与当前 descriptor 一致
       → 只计算 nbytes/alignment（不读 payload，不分配）
  → Backend::PackWeights(op, components, selector, recipe, allocator)
       → 再次校验 recipe
       → cpu::internal::PackWeightsWithRecipe
       → 返回 PackedWeight（storage 由 allocator 分配）
```

CPU 打包契约：

| 布局 | 组件要求 | 产物 |
|---|---|---|
| `cpu_identity` | 单组件任意有效 rank 且连续 row-major；多组件必须 rank-2、同 dtype、同 K，按调用顺序沿 axis 0 拼接 | 连续逻辑字节；单组件保留原 shape，多组件 shape 为 `[总行数, K]` |
| `cpu_bpanel_f32_kc512_nr16` | 所有组件 rank-2、FP32、同 K；行数累加 | 零填充 padded B-panel；shape 为 `[N, K]` 的逻辑 metadata，storage 按 panel 布局 |

- 单组件 identity 接受 scalar、rank-1 norm 权重与零尺寸权重；零尺寸允许空 data。
- 打包前先做 shape/stride/字节数溢出检查；溢出返回 `Overflow`，不读取 payload。
- 复合权重（Q/K/V、Gate/Up 等）可按消费方 recipe 走 identity 或 B-panel；backend 负责物化拼接与对齐，不改变调用方传入的组件顺序。
- 产物存储来自 `Allocator&`，backend 只在调用期间借用该引用，不保存；返回的 `Buffer` 在 allocator 销毁后仍然有效。
- `GetWeightPackingRequirements` 的结果排除了 allocator 自身的 metadata 与物理容量取整，调用方不得用它推断 allocator 的实际内部占用。
- 布局 ID 随 artifact 的 `recipe()` 返回，并被写入 `WeightArtifactKey`；execution 用 kernel 的 `expected_packing_recipe` 做精确查找，不存在“按 recipe 猜测”的路径。

### 4.4 执行期

```text
LayerRunner 逐 step
  → ExecutionContext::BindWorkspace(requirement) → WorkspaceBinding
  → 组装 KernelContext{workspace, packed_weights, kernel_params, kv_append, kv_read, attrs}
  → ResolvedKernel::fn(context)
```

- 执行期不访问 `Backend` 或 `KernelRegistry`，不做 selector 匹配、不构造 params、不分配内存。
- `packed_weights` 与 `kernel_params` 由 plan/`PreparedExecutionBindings` 持有，kernel 只借用；`attrs` 借用 `ResolvedKernel::attrs`。
- `kv_append` / `kv_read` 由 execution 按 plan 事务填充；kernel 不得缓存或跨调用保存。
- `CpuWorkspaceArena::Bind` 在 base 为空、`bytes == 0`、`alignment == 0`、`offset >= size`、`bytes > size - offset` 或地址不满足对齐时返回空 binding；它不做新分配，`Reset()` 不改变绑定寿命。调用方（execution）负责在触达 arena 前拒绝非法 alignment。

### 4.5 失败语义

| 场景 | 结果 |
|---|---|
| `Register` 遇到非法 descriptor | `InvalidArgument`（具体字段见 `ValidateKernelDef`） |
| 重复 `RegistrationKey` | `AlreadyExists` |
| freeze 后 `Register` | `FailedPrecondition` |
| freeze 前 `FindCandidates`/`FindByOpType` | `FailedPrecondition` |
| `FindCandidates` 收到 `kUnknown` op 或 `kUndefined` device | `InvalidArgument` |
| `PrepareKernel` 收到非 CPU selector | `InvalidArgument` |
| 无结构匹配或无特征匹配 | `NotFound`，消息携带 op、selector、`effective_features` |
| packed descriptor 缺少合法 recipe | `Internal` |
| `metadata_builder` 失败 | 原样传播其 `Status` |
| `GetPackingRecipe` 收到非 CPU 或非 packed selector | `InvalidArgument` |
| packing request 的 recipe 与当前选择不一致 | `InvalidArgument` |
| 打包 shape/stride/字节数溢出 | `Overflow`，在读取 payload 前返回 |
| 打包存储分配失败 | 原样传播 `Allocator::TryAllocate` 的 `Status`；`std::bad_alloc` 被捕获为 `ResourceExhausted`（空消息） |
| `Backend` 默认打包/查询实现 | `Unimplemented`，不做隐式 fallback |
| `BackendRegistry` 无对应 factory | `NotFound`；factory 错误原样传播；factory 返回空指针为 `Internal` |
| `SetFactory` 传 `kUndefined` 或空 factory | `AM_CHECK` 终止 |
| `ApplyCpuFeaturePolicy` 的 usable 不是 hardware 子集 | `InvalidArgument` |
| policy 的 required 不在 effective 中 | `FailedPrecondition`，消息携带缺失差集 |
| `CpuBackend(policy)` 检测失败 | `AM_CHECK` 终止；`CpuBackendFactory::Create` 返回 `Status` |

## 5. 关键取舍

| 选择 | 原因 | 被否方案 |
|---|---|---|
| `KernelRegistry` 为全局 singleton，`AM_REGISTER_KERNEL` 静态自注册 | 新增 kernel 只改自身 TU；descriptor 只有一个注册目录；跨 backend 通过 `KernelSelector.device_type` 硬匹配隔离 | per-backend registry；被否因为 descriptor 重复注册、跨 backend resolve 需额外桥接、添加 kernel 需改 backend 代码。代价是模块隔离较弱、依赖静态初始化顺序，且未使用 `used` 属性时存在链接器 GC 风险 |
| registry 只做结构匹配，CPU 特征过滤放在 `CpuBackend` | 指令集不是全序关系，`effective_features ⊇ cpu_requirements` 只能由持有能力快照的 backend 判定 | 把 ISA 放进 `KernelSelector` 并用序关系比较；被否因为 AVX512F/AMX/VNNI/FMA 相互正交，无法表达 |
| 三层能力快照 + 特征集合，而不是单层 bool / `IsaLevel` | 区分“硬件不支持”与“硬件支持但 OS 未启用”；kernel 只看 effective | 单层 capability 或 ISA 等级；被否因为 macOS AVX512、容器 XSAVE、AMX 权限等场景会产生 `#UD`/`SIGILL` |
| plan-build-time resolve，冻结 `ResolvedKernel::fn` | 热路径只有函数指针 direct call，无 registry 查找、无字符串/hash、无 backend 虚调用 | 执行期 dispatch table / graph executor；被否因为每次 token 都付出查找与失败后置成本 |
| `KernelContext` 只携带窄句柄 | kernel 不需要 `Runtime`/`Session`/plan；便于跨 backend 复用与测试 | 把 `Runtime*` 或整棵执行上下文下放给 kernel；被否因为破坏热路径、扩大耦合面 |
| attrs 在 plan-build 期由 `metadata_builder` 生成并冻结；params 在绑定准备期由 `params_builder` 生成一次 | 执行期不解析 typed `OpParams`，也不重复构造 params | 每次 kernel 调用读取语义参数/运行 builder；被否因为热路径分配与语义依赖 |
| 打包布局用显式 `PackingLayout` 枚举身份，不使用字符串或版本号 | 精确、可比较、无注册期字符串分配；改格式必须换 ID | `ToString` 名称参与匹配、`V1/V2` 版本号、prepacker 隐式决定布局；被否因为身份不稳定且容易误复用 |
| `PackWeights` 接收 `Allocator&` 并只借用 | packed storage 与其余受管存储使用同一分配策略；产物寿命独立于 allocator | backend 持有 allocator 或使用设备专属分配原语；被否因为破坏统一内存策略与生命周期边界 |
| `PackedWeight` 由 `PackedWeightCollection` 持有 | artifact 是准备期产物，可被多个 plan step 共享；collection 销毁后 plan 仍可执行 | backend 持有产物或 plan 持有裸指针；被否因为生命周期不确定 |
| `Backend` 的打包/recipe 查询默认返回 `Unimplemented` | 能力必须显式声明；没有“通用打包”假设 | 默认做一次 identity 拷贝；被否因为会掩盖 backend 不支持打包的事实，并制造错误的 recipe 语义 |
| `KernelDef::name` / `ResolvedKernel::name` 使用 `std::string_view` | descriptor 成为可平凡拷贝的纯数据，注册期不分配字符串 | 持有 `std::string`；被否因为注册期分配与拷贝成本，且 name 只用于诊断 |
| CPU 策略只能削减，`required` 不满足即失败 | policy 是部署约束，不是能力扩展；避免“声称支持但实际会 SIGILL” | 允许 policy 启用硬件不存在的特征；被否因为无法安全执行 |
| `CpuBackend` 构造函数在检测失败时 abort，`CpuBackendFactory` 返回 `Status` | 构造函数没有错误通道；factory 是正式装配入口，可向上传播 | 所有入口都 abort；被否因为 Runtime 装配需要可恢复错误 |
| 可选 AVX2/FMA kernel 放在独立 TU，编译选项与宏共同门控 | 保证不含 AVX2/FMA 的构建不会链接/注册对应入口；旧 CPU 只解析 scalar | 在同一个 TU 内用运行期分支编译所有 ISA；被否因为无法在旧机器上安全执行，且削弱编译期隔离 |
| workspace 只冻结 requirement，地址由 execution 绑定 | 满足 Decode 稳态零分配与 arena 复用；backend 不拥有 arena | 在 plan 中冻结绝对地址；被否因为 arena 生命周期与 session 绑定不归 backend |
| KV 物理布局归 kv_cache，backend 只消费窄 binding | 物理存储、reservation、commit 需要单一所有者；kernel 只读 `KVCacheLayerStorageBinding` 等句柄 | backend 定义并强制 KV 物理布局；当前代码没有该所有权，且会与 `KVCacheManager` 冲突 |

已知边界：

- `DotProductF32` / `DotProductF32MultiTarget` 在 kernel 内部使用进程级缓存的 `DetectCpuCapabilities()` 结果选择 AVX2/FMA 实现，**不读取某个 `CpuBackend` 的 policy**。因此 policy 只保证 descriptor 选择层面的回退，不保证所有 kernel 内部辅助函数都回退到 scalar。
- `CpuBackend` 构造会冻结全局 registry；测试中创建多个 backend 实例不会重置注册目录，注册只能在首次构造前完成。
- 当前 CPU kernel 没有非零 workspace 需求；`workspace_requirement` 字段已冻结进 `ResolvedKernel`，但实际 scratch 使用尚未出现。
- 空 policy 下，未支持平台不会让 `DetectCpuCapabilities` 失败，而是返回空快照；`CpuBackend` 仍可构造，只是所有带 `cpu_requirements` 的 kernel 都会被过滤，只有空需求 descriptor 可被解析。若 policy 的 `required_features` 非空，则在未支持平台上按 `FailedPrecondition` 失败。

## 6. 测试与验证入口

| 测试文件 | 覆盖重点 |
|---|---|
| [test_kernel_registry.cpp](../../tests/unit/backend/test_kernel_registry.cpp) | descriptor 校验、packed/plain recipe 规则、重复注册、freeze 前后语义、name 借用、并发 freeze 幂等、`DebugDump` |
| [test_kernel_registry_resolve.cpp](../../tests/unit/backend/test_kernel_registry_resolve.cpp) | `kBoth` 匹配、结构候选不按特征过滤、phase 不兼容返回空 |
| [test_kernel_selector.cpp](../../tests/unit/backend/test_kernel_selector.cpp) | selector 相等性、`state_dtype` 严格匹配、`ExecPhase`/`WeightFormat` 字符串化、hash 与相等一致 |
| [test_backend_registry.cpp](../../tests/unit/backend/test_backend_registry.cpp) | factory 注册/替换、懒创建与缓存、未注册 device、替换后驱逐缓存实例 |
| [test_no_hotpath_resolve.cpp](../../tests/unit/backend/test_no_hotpath_resolve.cpp) | plan-build 只 prepare 一次；执行期不再次查询 backend，只调用冻结 kernel |
| [test_backend_weight_packing.cpp](../../tests/unit/backend/test_backend_weight_packing.cpp) | 默认 `PackWeights` 返回 `Unimplemented` 且不查 recipe；显式 recipe/组件转发 |
| [test_cpu_backend.cpp](../../tests/unit/backend/cpu/test_cpu_backend.cpp) | CPU device、policy 限制 kernel 选择、线性 kernel resolve、recipe 选择与拒绝、RuntimeBuilder 注入 policy |
| [test_cpu_capabilities.cpp](../../tests/unit/backend/cpu/test_cpu_capabilities.cpp) | 默认快照、三个 `ToString`、名字表覆盖、越界回退、`Intersect`/`Difference` |
| [test_cpu_info.cpp](../../tests/unit/backend/cpu/test_cpu_info.cpp) | 检测不变量 `hardware ⊇ usable ⊇ effective`、policy 只削减、required 失败、非法快照拒绝、空 policy 幂等 |
| [test_cpu_resolve_kernel.cpp](../../tests/unit/backend/cpu/test_cpu_resolve_kernel.cpp) | 已注册 key 的 prepared kernel、metadata 校验、debug registry、冻结 kernel 可调用 |
| [test_cpu_isa_fallback.cpp](../../tests/unit/backend/cpu/test_cpu_isa_fallback.cpp) | 禁用 AVX2/FMA 后 RmsNorm 回退参考实现；packed Linear 保持 identity recipe；可选 descriptor 与编译期宏一致；factory 不能启用硬件缺失特征 |
| [test_cpu_workspace_arena.cpp](../../tests/unit/backend/cpu/test_cpu_workspace_arena.cpp) | 合法绑定、空 base、越界、未对齐、零字节、零 alignment、`Reset` 不改变绑定契约 |
| [test_cpu_weight_packing.cpp](../../tests/unit/backend/cpu/test_cpu_weight_packing.cpp) | identity 单组件/多组件、B-panel 拼接与零填充、recipe 拒绝、溢出先于读取、分配失败 |
| [test_cpu_weight_packing_requirements.cpp](../../tests/unit/backend/cpu/test_cpu_weight_packing_requirements.cpp) | 查询与真实存储一致、查询不读 payload、批量预算先检查、未知查询限制 |
| [test_packed_weight_validation.cpp](../../tests/unit/backend/cpu/test_packed_weight_validation.cpp) | identity/B-panel 的逻辑 metadata、recipe、alignment、字节大小与未知布局拒绝 |

```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter=KernelRegistry.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=KernelRegistryCandidates.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuPrepareKernel.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuBackend.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuCapabilities.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuInfo.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuIsaFallback*.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuWeightPacking*.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CpuWorkspaceArena.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=BackendWeightPacking.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=NoHotpathPrepare.*
```

并发相关改动至少运行 TSAN 构建下的 `KernelRegistry.ConcurrentFreezeIsIdempotent`。新增 kernel、布局或打包路径时，按 [算子开发工作流](../guides/operator-development-workflow.md) 补齐 reference/optimized 证据；测试命名与断言规范见 [测试编写指南](../guides/test_writing_guidelines.md)。
