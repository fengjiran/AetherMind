# AetherMind 公共 API 参考

> 本文件汇总 AetherMind 公共 API 的语义，与头文件 Doxygen 注释**同源同步**：修改头文件注释必须同步本文件，反之亦然（[文档系统规范](../guides/documentation-guide.md) §7.3）。
>
> 构建选项与运行配置以根 [README.md](../../README.md) 为单一事实源，本文件不复制其表格。C++ 已提供 `PrepareExecutableModel` 与 `InferenceSession::Generate`，M6 已接通 runtime/model/session C ABI。当前已验证范围为 CPU FP32 Llama Dense，不以 loader 接受范围推导执行支持；能力与产品目标见 [PRD](../products/aethermind_prd.md) 和 [10 号方案](../improvement-plan/10-upper-level-runtime-assembly.md)。

## 1. C ABI（include/c_api.h）

> Runtime/model/session 的 v1 合同由 M5 冻结、M6 实现，声明见 `include/c_api.h`，实现见 `src/c_api_runtime.cpp`。完整 [Runtime C ABI 合同](runtime-c-abi.md) 规定 version/config、owner、error 与 buffer 规则；[M6 报告](../tests/upper_level_runtime_assembly_m6_validation_2026-10-22.md) 记录纯 C 调用与验证边界。既有对象引用计数、错误处理和 traceback 原语保留。

| Runtime C 接口 | 返回与生命周期 |
|---|---|
| `am_get_abi_version()` | 返回 `(major << 16) \| minor`，当前为 `0x00010000`；不分配 |
| `am_runtime_create(config, error)` / `am_runtime_destroy(runtime)` | 显式 CPU/FP32 KV 几何与容量创建 Runtime，失败返回 NULL；destroy 释放调用方 handle，model/session owner 仍可保活资源 |
| `am_model_load(runtime, path, error)` / `am_model_unload(model)` | Load→共享 FP32 准入→Compile→唯一跨对象 KV 校验→Prepare；失败不发布 model，已有 Runtime/KV 不变 |
| `am_session_create(model, config, error)` / `am_session_destroy(session)` | 复制生成配置并保活 model/Runtime；非空 handle 仅释放一次，NULL destroy/unload 为 no-op |
| `am_session_generate(session, input, len, output, capacity, output_len, error)` | 返回 am_status_code；短 buffer 返回 N 上界且不执行，成功才复制完整新 IDs 并写实际长度，其他失败长度为零/输出不变。详见完整合同的错误优先级与 N=0 规则 |

所有创建的 error out 可省略，提供时入口先清空、不释放旧值；诊断 OOM 时可无 error 对象，Generate 仍保留失败码。共享 Runtime 的调用由宿主串行化；C wrapper 不更改宿主 logger 或增加并发支持。

### `int IncObjectRef(ObjectHandle obj_ptr)`

| 项 | 内容 |
|---|---|
| 简介 | 递增对象的强引用计数（对应 `@brief`） |
| 参数 | `obj_ptr`：指向对象头的句柄（对应 `@param`） |
| 返回值 | 新引用计数或错误码（对应 `@return`） |
| 语义 | 与 `DecObjectRef` 配对使用；引用计数归零时调用 `ObjectHeader::deleter`（对应 `@note`） |
| 前置条件 | `obj_ptr` 必须指向有效的 `ObjectHeader`（对应 `@pre`） |

### `int DecObjectRef(ObjectHandle obj_ptr)`

| 项 | 内容 |
|---|---|
| 简介 | 递减对象的强引用计数，归零时触发删除器（对应 `@brief`） |
| 参数 | `obj_ptr`：指向对象头的句柄（对应 `@param`） |
| 返回值 | 剩余引用计数或错误码（对应 `@return`） |
| 语义 | 引用计数归零后对象内存由 `deleter` 回收；不得再次访问（对应 `@note`） |
| 前置条件 | `obj_ptr` 必须指向有效且未被释放的对象（对应 `@pre`） |

### `am_error_handle am_error_create(am_status_code code, const char* message)`

| 项 | 内容 |
|---|---|
| 简介 | 创建带状态码与消息的错误对象（对应 `@brief`） |
| 参数 | `code`：`am_status_code` 枚举；`message`：复制的错误描述字符串，NULL 表示空字符串（对应 `@param`） |
| 返回值 | 错误句柄，必须由 `am_error_destroy` 释放；对象或 message 分配失败返回 NULL（对应 `@return`） |
| 语义 | 错误对象生命周期独立；句柄所有权转移给调用方；初始化失败自动释放部分构造对象，不传播异常（对应 `@note`） |
| 前置条件 | 非 NULL `message` 指向有效 NUL 结尾字符串（对应 `@pre`） |

### `void am_error_destroy(am_error_handle error)`

| 项 | 内容 |
|---|---|
| 简介 | 释放错误对象（对应 `@brief`） |
| 参数 | `error`：`am_error_create` 返回的句柄（对应 `@param`） |
| 返回值 | 无 |
| 语义 | `nullptr` 作为 no-op 接受（对应 `@note`） |
| 前置条件 | `error` 未被释放过（对应 `@pre`） |

### `am_status_code am_error_code(am_error_handle error)`

| 项 | 内容 |
|---|---|
| 简介 | 读取错误对象的状态码（对应 `@brief`） |
| 参数 | `error`：错误句柄（对应 `@param`） |
| 返回值 | `am_status_code` 枚举值（对应 `@return`） |

### `const char* am_error_message(am_error_handle error)`

| 项 | 内容 |
|---|---|
| 简介 | 读取错误对象的描述消息（对应 `@brief`） |
| 参数 | `error`：错误句柄（对应 `@param`） |
| 返回值 | 消息字符串指针，生命周期与错误对象一致（对应 `@return`） |
| 语义 | 返回值不得由调用方释放；错误对象销毁后指针失效（对应 `@note`） |

### `const char* AetherMindTraceback(filename, lineno, func, cross_aethermind_boundary)`

| 项 | 内容 |
|---|---|
| 简介 | 记录一条 traceback 上下文（对应 `@brief`） |
| 参数 | `filename`/`lineno`/`func`：调用位置；`cross_aethermind_boundary`：是否跨越库边界（默认 0）（对应 `@param`） |
| 返回值 | 追踪信息字符串（对应 `@return`） |

## 2. C++ 公开构建块（include/aethermind/）

### `ModelLoader::Load`（include/aethermind/model/model_loader.h）

| 项 | 内容 |
|---|---|
| 简介 | 加载并校验 HF 模型目录为 backend-independent 的 config 与逻辑原始权重视图（对应 `@brief`） |
| 参数 | `model_dir`：含 `config.json` 与 safetensors 的目录（对应 `@param`） |
| 返回值 | 持有 config 与 resolved weights 的 `unique_ptr<LoadedModel>`；失败经 `StatusOr` 上报（对应 `@return`） |
| 语义 | 不构建图、不 resolve kernel、不 prepack；所有失败通过 Status 返回（对应 `@note`） |
| 前置条件 | `model_dir` 为有效可读目录（对应 `@pre`） |

### 共享 FP32 执行策略（include/aethermind/inference/model_execution_policy.h）

| 接口 | 语义与所有权 |
|---|---|
| `Status ValidateFp32ExecutionInputs(const LoadedModel&)` | 检查 scalar Float32 hint 与全部必需 resolved weight roles，支持 tied lm-head fallback；只读 metadata，不读/转换 payload、不保留借用。精度不支持返回 Unimplemented，非法 metadata 返回相应错误，诊断 OOM 返回空 message 的 ResourceExhausted。家族/算子语义仍由 Compile 判定。 |
| `StatusOr<KVCacheRuntimeOptions> MakeFp32KVCacheOptions(const HfModelConfig&, size_t context_tokens)` | 将 Loader 已验证的模型几何与显式 capacity 映射为 owning KV 配置：CPU 首版 Float32、64-byte alignment、逻辑 head_dim；只有 absent head_dim=0 才按整除 fallback。校验正数、安全转换和模型 context 上限，不分配 KV、不访问 backend、不比较既有 Runtime/model。 |

M5 将这两项规则归入 inference，CLI 与 M6 C adapter 消费同一实现，没有旧应用 namespace alias 或兼容转发。跨对象 KV 相等与 compiled state dtype 校验仍属于 adapter。

### 共享 CPU preset（include/aethermind/inference/cpu_inference_options.h）

`CpuInferencePreset::kO1Plain` / `kO2Packed` 和 `MakeCpuCompileOptions(CpuInferencePreset)` 为 CLI/C adapter 共用的显式编译策略。返回 owning ModelCompileOptions：O1/plain 或 O2/packed、CPU、kBoth，activation/weight dtype 由 lowering 推导；未知枚举返回 InvalidArgument，诊断 OOM 返回空 message 的 ResourceExhausted。该函数不加载/准入模型、不 resolve kernel、不分配 storage，也不暗含性能优劣。

### `ModelCompiler`（include/aethermind/compiler/model_compiler.h）

| 接口 | 语义与所有权 |
|---|---|
| `Compile(std::unique_ptr<LoadedModel>, const ModelCompileOptions& = {})` | 返回 `StatusOr<LoweredModelArtifact>`。消费模型，依次构建语义图、执行 semantic passes、lowering；artifact 拥有 `LoadedModel` 及原始权重 backing，不依赖 Runtime，不 resolve kernel 或物化权重。 |
| `LoadAndCompile(const std::filesystem::path&, const ModelCompileOptions& = {})` | 加载目录后调用同一编译链，返回拥有模型的 artifact。 |

`ModelCompileOptions` 包含 semantic optimization 与 lowering 配置。默认 O2/plain 组合当前不能直接作为 CPU 可执行默认值；入口使用上述共享 preset 显式选择 O1/plain 或 O2/packed，编译为 `ExecPhase::kBoth`。RunOptions、参数解析与 Token IDs I/O 仍属应用私有实现。

### `RuntimeBuilder`（include/aethermind/runtime/runtime_builder.h）

| 接口 | 签名 | 语义要点 |
|---|---|---|
| `WithOptions` | `RuntimeBuilder& WithOptions(const RuntimeOptions& options)` | 设置运行时装配选项；返回 `*this` 支持链式调用 |
| `RegisterCustomAllocatorProvider` | `RuntimeBuilder& RegisterCustomAllocatorProvider(DeviceType, std::unique_ptr<AllocatorProvider>)` | 注册自定义分配器提供者；所有权转移 |
| `RegisterBackendFactory` | `RuntimeBuilder& RegisterBackendFactory(DeviceType, std::unique_ptr<BackendFactory>)` | 注册后端工厂；所有权转移 |
| `Build` | `Runtime Build()` | 装配 AllocatorRegistry + BackendRegistry + KVCacheManager 为 `Runtime`；初始化失败（含分配失败）时 abort |
| `TryBuild` | `StatusOr<Runtime> TryBuild()` | 与 `Build` 相同的装配，但把 KV 初始化失败与注册阶段的分配失败作为 Status 返回（分配失败为空 message 的 `kResourceExhausted`）。成功后消费 pending 注册；失败时两个 pending 列表只保留从未转移的项，builder 仍可复用而不会 abort，但已转入被丢弃 registry 的注册项不可恢复 |

### `PrepareExecutableModel` / `ExecutableModel`（include/aethermind/inference/executable_model.h）

| 接口 | 语义与所有权 |
|---|---|
| `StatusOr<ExecutableModel> PrepareExecutableModel(Runtime&, LoweredModelArtifact)` | 唯一生产准备入口；消费 artifact，编排权重物化、执行计划构建与 immutable weight/constant bindings。失败不发布部分准备的模型。 |
| `LoadAndPrepareExecutableModel(Runtime&, const std::filesystem::path&, const ModelCompileOptions&)` | 组合加载、编译、准备的便利入口，要求 Runtime 已存在且编译选项显式给出。CLI 为读取模型配置并在 Compile 成功后分配 KV，改用上述独立步骤。 |
| `ExecutableModel::plan(ExecPhase)` / `immutable_weight_bindings(ExecPhase)` | 返回模型拥有的 plan/binding table 的借用指针，phase 不匹配报错；Prefill/Decode 当前共享一个不可变 plan。binding table 不含请求 token/position 输入。 |
| `artifact_id()` / `phase()` / `context_limit()` / `vocab_size()` | 查询编译产物身份、phase、模型 context 上限与词表大小。 |
| `IsPreparedFor(const Runtime&)` | 检查模型 resolve 的 kernels 是否属于该 Runtime。 |

`ExecutableModel` 拥有 compiler artifact、packed artifacts、binding metadata、immutable bindings 与 plan，可 move-construct，不可复制或赋值。传给 Prepare 的 Runtime 从此必须保持地址稳定、不可移动，并长于模型和 Session；模型必须长于从其 plan/bindings 派生的 prepared bindings。底层权重 storage 由模型保活，resolved kernel 所借用的 backend 仍由 Runtime 保活。

### `InferenceSession` / `GenerationConfig`（include/aethermind/inference/inference_session.h）

| 接口 | 语义与所有权 |
|---|---|
| `static StatusOr<InferenceSession> Create(Runtime&, std::shared_ptr<const ExecutableModel>)` | 校验模型与 Runtime 身份、phase 及 plan I/O 合同；Session 借用 Runtime，共享拥有模型。无 KV manager 的 Runtime 可以创建 Session，正数生成请求时才报告资源缺失。 |
| `StatusOr<std::vector<uint32_t>> Generate(std::span<const uint32_t>, const GenerationConfig&) const` | 同步 greedy 生成，返回拥有的新 Token IDs，包含 Prefill 预测的第一个 token，以及生成出的 EOS；不返回 prompt。每次调用重新 Prefill，并在成功、执行失败或分配失败返回前释放 KV reservation。 |
| `GenerationConfig::max_new_tokens` | `size_t`，默认 0。正数 N 最多需要 N−1 次 Decode KV append；检查 `prompt.size() + N - 1` 溢出、模型 context 与物理 KV capacity。 |
| `GenerationConfig::eos_token_id` | `std::optional<uint32_t>`，默认无 EOS；生成该 token 后停止，结果包含它。 |

prompt 必须非空；prompt/EOS 必须在词表范围内，prompt 长度须满足模型 context。N=0 仍进行这些校验，随后返回空 vector，不预约 KV、不执行模型，也不要求 prompt 满足物理 KV capacity。request buffer、binding、workspace 或 activation 分配传播出的 `std::bad_alloc` 转为空 message 的 `ResourceExhausted`。

共享同一 Runtime 的所有 Generate 调用必须由宿主串行化，包括不同 Session；当前接口不承诺并发 Session。推荐销毁顺序为 Session → ExecutableModel → Runtime。阶段执行与资源细节见 [Session 设计](../designs/inference/02-inference-session.md)。

### `Allocator::TryAllocate`（include/aethermind/memory/allocator.h）

| 项 | 内容 |
|---|---|
| 签名 | `StatusOr<Buffer> TryAllocate(const AllocationRequest& request) noexcept` |
| 简介 | 统一的 owning storage 分配入口。基类以 NVI 实现：校验请求 → 调用派生类 protected `AllocateStorage` hook → 校验 ownership、device 和 alignment → 将原始 `MemoryHandle` 直接移入 `Buffer`。KV、packed weights、workspace 与 activation 均接受 Runtime 配置的 allocator 策略。 |
| 请求契约 | `AllocationRequest{nbytes, alignment}`。`nbytes` 为可见大小，布局 padding 由所属模块计入。零字节请求仍产生有效 owning `Buffer`，CPU 实际申请至少 1 byte。`alignment == 0` 使用默认策略；非零值须为 2 的幂，表示最低对齐。返回值可以更强，子视图需按自身 offset 确定对齐保证。CPU 会提升弱于 `sizeof(void*)` 的对齐。 |
| 生命周期 | `BufferImpl` 直接持有 allocator 返回的原始 `MemoryHandle`。自定义 deleter 及其 context 必须独立维持所需资源的生命周期，不能借用会提前销毁的 allocator。共享副本在最后一个 owner 释放时执行一次原始 deleter；Buffer 可晚于 allocator、Runtime 或被替换的 provider 销毁。 |
| 失败语义 | 非零且非 2 的幂的 alignment 返回 `kInvalidArgument`；hook 返回借用内存、错误 device、无效或不足的 alignment，或 CPU 指针不满足声明对齐时返回 `kInternal`。存储或 `BufferImpl` 元数据 OOM 返回空 message 的 `kResourceExhausted`，已有 owning storage 自动释放。`Allocate(size_t)` 转调本入口，将 `kResourceExhausted` 映射为 `std::bad_alloc`，其余错误映射为 `Error`。 |
| 扩展点 | `AllocatorProvider::CreateAllocator(Device)` 只创建对应设备的 allocator；`CPUAllocator(Device, CPUAllocatorPolicy)` 控制默认对齐、junk fill 和 THP。当前不附带分类标签、内存账本或额外释放包装，也不提供预算限制。 |

### `IsValidAlignment`（include/aethermind/base/alignment.h）

| 项 | 内容 |
|---|---|
| 签名 | `constexpr bool IsValidAlignment(size_t alignment) noexcept` |
| 简介 | 通用对齐合法性谓词；workspace、KV、allocator 等调用方直接使用 |
| 参数 | `alignment` 为待校验的字节对齐值 |
| 返回值 | 非零的 2 的幂返回 true（包括 1）；0 和非 2 的幂返回 false；不分配内存 |

### `AlignOffsetUp`（include/aethermind/base/alignment.h）

通用 offset 对齐接口定义于 [alignment.h](../../include/aethermind/base/alignment.h)，实现归 [alignment.cpp](../../src/base/alignment.cpp)。workspace、activation 和 kernel params 统一调用该接口；调用方需链接核心库。

| 项 | 内容 |
|---|---|
| 签名 | `StatusOr<size_t> AlignOffsetUp(size_t offset, size_t alignment) noexcept` |
| 简介 | 返回不小于 offset、满足 alignment 的最小对齐偏移 |
| 参数 | `offset` 为待对齐偏移；`alignment` 必须是非零的 2 的幂，1 合法 |
| 返回值 | 成功返回对齐偏移；非法 alignment 返回 `kInvalidArgument`，结果不可表示为 size_t 时返回 `kOverflow`；错误诊断分配失败返回空 message 的 `kResourceExhausted` |
| 语义 | 先检查 alignment 再进行算术；已对齐值保持不变，`AlignOffsetUp(SIZE_MAX, 1)` 成功；成功路径不分配内存 |

旧公开接口 `AlignWorkspaceOffset()` 和转发谓词 `IsValidWorkspaceAlignment()` 均已删除，没有兼容转发；旧调用方应直接包含 `aethermind/base/alignment.h`，分别改用 `AlignOffsetUp()` 和 `IsValidAlignment()`，重新编译、链接。Workspace planner 的布局、零字节和失败原子性合同保持不变。

### Workspace 数据契约（include/aethermind/base/workspace_types.h）

`WorkspaceLifetime`、`WorkspaceRequirement`、`WorkspaceBinding` 统一定义在 [workspace_types.h](../../include/aethermind/base/workspace_types.h)。原先包含 `base/workspace_binding.h` 的调用方需改为包含 `base/workspace_types.h`；类型名、字段布局与行为不变。

| 类型 | 语义与生命周期 |
|---|---|
| `WorkspaceLifetime` | 描述 scratch-memory 需求的生命周期范围 |
| `WorkspaceRequirement` | backend 提供 `bytes`、`alignment`、`lifetime`、`reusable`；planner 填写相对 workspace base 的 `offset`，不持有内存地址 |
| `WorkspaceBinding` | `WorkspaceArena::Bind` 产生的借用切片；`data` 已满足 requirement 对齐，`size` 不小于需求字节数；仅供单次 kernel 调用，不得保存到调用结束之后 |

资源接口 `WorkspaceArena` 仍定义于 [workspace_arena.h](../../include/aethermind/base/workspace_arena.h)，布局规划仍定义于 [workspace_planner.h](../../include/aethermind/runtime/workspace_planner.h)。

### `BoundValue`（include/aethermind/execution/execution_bindings.h）

`BoundValue` 是一个逻辑 execution value 的规范物理绑定，内部封装 `std::variant<std::monostate, TensorView, MutableTensorView>`。空状态表示没有普通 tensor binding，例如 state 值或仅由 packed artifact 提供的 weight；可写分支同时允许读取。该类型只借用 data/shape/stride 元数据，不拥有 tensor storage。

| 接口 | 语义与生命周期 |
|---|---|
| `BoundValue()` | 创建未绑定值 |
| `explicit BoundValue(TensorView view)` | 保存只读 descriptor；无效 view 规范化为未绑定值 |
| `explicit BoundValue(MutableTensorView view)` | 保存可写 descriptor；无效 view 规范化为未绑定值；合法零元素 view 可使用 null data，不等于未绑定 |
| `TensorView ReadView() const noexcept` | 按值返回只读 descriptor；可写分支从同一份 data/metadata 派生，空状态返回无效 view；不复制数据、不分配内存。返回值不借用 BoundValue 本体，但其底层 data 和 metadata 必须继续存活 |
| `const MutableTensorView* TryGetWritableView() const noexcept` | 仅可写分支返回指向内部 descriptor 的借用指针，其余返回 nullptr；指针不得跨 BoundValue 修改、移动或析构使用 |

原 `readable`、`writable`、`has_writable` 字段已移除，调用方需迁移到上述接口并重新编译，不能将新旧布局视为 ABI 兼容。`PrepareExecutionBindings` 仍在 cold path 构建 `StepTensorBinding.inputs/outputs` 和 prepared kernel params；执行器不在每次 kernel 调用时访问 variant。Session 读取 token 输出时使用 `ReadView()`。

### `Executor::Execute`（include/aethermind/execution/executor.h）

| 项 | 内容 |
|---|---|
| 简介 | 按序执行计划中的每一步（对应 `@brief`） |
| 参数 | `plan`：待执行计划；`context`：由 `ExecutionContext::Create` 产生的 prepared tensor bindings、workspace 和 KV view，调用期间必须保持有效（对应 `@param`） |
| 返回值 | 成功返回 `Status::Ok()`；失败返回首个失败步骤的错误（对应 `@return`） |
| 语义 | 同步单线程执行，委托 `LayerRunner`；不产生 Token IDs（对应 `@note`） |

### `InvokePreparedKernel`（include/aethermind/execution/kernel_invoker.h）

| 项 | 内容 |
|---|---|
| 签名 | `Status InvokePreparedKernel(const ResolvedKernel& kernel, const KernelContext& context) noexcept` |
| 简介 | 调用已经由 `PreparedExecutionBindings` 构建 params 的冻结 kernel |
| 参数 | `kernel`：backend resolve 后的不可变 kernel；`context`：调用前已完整填入 workspace binding、KV bindings、packed weights、attrs 和 `kernel_params` 的执行上下文 |
| 语义 | 通过 `AM_DCHECK` 检查 `params_builder`/`params_size` 与 `context.kernel_params` 的契约，然后直接调用 `kernel.fn`；不修改 context，不产生分配 |
| 生命周期 | `context.kernel_params` 是借用指针，存储仍由 `PreparedExecutionBindings` 拥有，必须覆盖调用期间 |
| 调用方迁移 | 直接调用该低层接口的 caller 必须在调用前预设 `KernelContext::kernel_params`，并按新签名重新编译 |

`BuildAndInvokeKernelForTesting` 仍保留原有签名和逐次构建 params 的行为，仅供测试与低层诊断使用，不属于生产执行路径。

### `PrepareExecutionBindings` / `ExecutionContext::Create`

| 接口 | 语义要点 |
|---|---|
| `PrepareExecutionBindings(const ExecutionPlan&, const ExternalTensorBindings&, Allocator&)` | cold path specialization：校验 external dtype/shape/stride、runtime constraints 与 kernel-specific layout/aliasing，分配 activation 并构造 prepared params。external tensor backing 由调用方借出，改变其地址、shape、stride、dtype 或 alias 前必须重新 prepare。 |
| `ExecutionContext::Create(const ExecutionPlan&, PreparedExecutionBindings, WorkspaceArena*, KVCacheSessionView)` | 消费 prepared bindings，校验 plan binding key、非零 workspace 的 arena presence 与 state aliases 的 KV view presence；不拥有 plan、arena 或 KV storage。 |
| `ExecutionContext::ResetKVLayerTransactionScratch() noexcept` | 将逐层 append 状态重置为 `KVAppendLayerState::kNotScheduled`，返回 `std::span<KVAppendLayerState>`；借用 context 在 `Create()` 中预分配的存储，仅供当前同步执行使用，重置不分配内存。 |
| `ExecutionContext::Clear()` | 释放 owned prepared bindings 并清除 borrowed handles；不 reset WorkspaceArena，也不 release KV reservation。 |

`KVAppendLayerState : uint8_t` 定义于 [execution_context.h](../../include/aethermind/execution/execution_context.h)，表示单次 plan 执行中的 `kNotScheduled`、`kScheduled`、`kWritten` 三种逐层 append 状态。context 拥有 `std::vector<KVAppendLayerState>`，`KVAppendTransaction::layer_states` 只借用其 span；`kWritten` 只表示本层写入成功，整计划成功后才发布 committed watermark。原来将 scratch 保存为 `std::span<uint8_t>` 或 `uint8_t*` 的调用方需迁移到枚举类型，并重新编译。

### `KVCacheSessionView::CommitAppend`（include/aethermind/kv_cache/kv_cache_session_view.h）

| 项 | 内容 |
|---|---|
| 签名 | `Status CommitAppend(size_t begin, size_t end) const noexcept` |
| 简介 | 发布一个已经由完整 append plan 写入完成的 KV 区间，并推进 manager-owned committed watermark/lifecycle |
| 前置条件 | 调用方必须先证明完整 append plan 成功，且每个 KV layer 都完成 `[begin, end)` 写入；该方法自身不证明 layer 写入完成 |
| 校验 | 验证 view generation、session lifecycle、`begin/end` 关系与 reservation/physical capacity；失败时 committed watermark 和 lifecycle 不变 |
| 语义 | `const` 只表示 borrowed view descriptor 不变；成功调用会修改其借用的 manager-owned session state。Prefill/Decode 的完整 plan transaction 仍由 `LayerRunner` 负责在所有 layer 成功后调用 |
| 生命周期 | View 借用 `KVCacheManager` 的 storage/session state；Manager 必须保持地址稳定并长于 view |

旧的 `CommitUntil` 与私有提交访问 header 不再是当前 API；不提供兼容别名或旧 header。

## 3. 通用 C++ 工具（include/utils/）

| 接口 | 合同 |
|---|---|
| `CheckNarrow(value, out)` | 检查标准 signed/unsigned 整数到目标整数类型的可表示性；`true` 表示失败且输出保持原值，`false` 表示已完成转换。目标须为无 cv 限定的标准整数类型，`bool`、普通字符类型、枚举及浮点数不参与重载。成功与失败均不分配，支持 constexpr。`out` 必须非空。 |
| `CheckOverflowAdd/Mul/Sub(a, b, out)` | 对非 bool、无 cv 的整数类型检查算术结果，`true` 表示溢出。始终写入按目标位宽取模的结果；溢出时不得将该值视为精确结果。`out` 必须非空。GCC/Clang 使用 builtin，其余编译器使用同合同的 portable 实现。 |
| `aethermind::hash<T>` / `get_hash(...)` | 定制优先级为 `T::hash`、enum underlying、`std::hash`；提供 tuple/pair/vector/complex 的显式复合 hasher。pair 按引用读取元素，不要求元素可复制。标准容器的复合键应显式传入 `aethermind::hash<Key>`，不再提供标准类型的通用 `std::hash` 偏特化。 |
| `utils::ParseFloat/ParseDouble(text, out)` | 完整解析 `from_chars` general 格式的字面量；拒绝空串、前导空白/加号、十六进制形式及尾随垃圾。保留合法的 inf/nan，由所属领域检查是否允许非有限值；失败保留输出。POSIX fallback 使用局部 C numeric locale，恢复线程原 rounding mode，locale/分配失败返回 false，不传播异常。 |
| `AM_CHECK` / `AM_DCHECK` | 可从任意命名空间调用；失败诊断保留 `Check failed` 前缀。`AM_DCHECK` 在 release 中不求值。头文件只提供断言，不传递包含 spdlog；日志调用方直接声明依赖。 |

`numeric_utils.h`、`variant_utils.h` 与 `qualified_name.h` 保留现有接口。未使用的 `lru.h`、`xxh3.h`、`bits.h`、`function.h`、`function_traits.h`、`registry.h` 及 `Function`/`Registry` 已删除；DataType 枚举与已有 dtype 行为保留。原公共 `memory_utils.h` 与 pattern getters 已移除，junk-fill 属于 CPUAllocator 的私有实现，不提供兼容转发。

## 4. 语义约束

- C ABI 句柄（`ObjectHandle`、`am_error_handle`）为不透明指针，不暴露内部布局；对象生命周期由引用计数/显式 destroy 管理。
- `am_status_code` 覆盖当前产品所需的错误类别（OK/CANCELLED/UNKNOWN/INVALID_ARGUMENT/.../UNAUTHENTICATED）。
- `Executor::Execute` 执行一次单计划；跨 Prefill/Decode 的请求编排由已实现的 `InferenceSession::Generate` 负责。
- `apps/inference/` 的 runner、参数校验与 I/O 为应用私有实现；FP32 准入、模型 KV 映射及 CPU preset 归 inference 共享接口，C ABI 适配有独立 C 调用验收。

## 5. 相关文档

- 头文件：`include/c_api.h`、`include/aethermind/model/model_loader.h`、`include/aethermind/compiler/model_compiler.h`、`include/aethermind/runtime/runtime_builder.h`、`include/aethermind/inference/executable_model.h`、`include/aethermind/inference/inference_session.h`、`include/aethermind/execution/executor.h`（Doxygen 注释为语义事实源）
- 架构总览：[docs/designs/architecture/architecture_overview.md](../designs/architecture/architecture_overview.md)（§一 API 服务层、§八 差距概要）
- ModelLoader 模块设计：[docs/designs/model/01-model-loader.md](../designs/model/01-model-loader.md)
- 注释规范：[docs/guides/cpp_comment_guidelines.md](../guides/cpp_comment_guidelines.md)
