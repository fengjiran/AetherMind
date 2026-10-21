# AetherMind 公共 API 参考

> 本文件汇总 AetherMind 公共 API 的语义，与头文件 Doxygen 注释**同源同步**：修改头文件注释必须同步本文件，反之亦然（[文档系统规范](../guides/documentation-guide.md) §7.3）。
>
> 构建选项与运行配置以根 [README.md](../../README.md) 为单一事实源，本文件不复制其表格。当前产品目标 API（`am_session_generate`、`Session::Generate`）尚未实现，见 [PRD](../products/aethermind_prd.md)，不在此列。

## 1. C ABI（include/c_api.h）

> 当前 C ABI 仅提供对象引用计数、错误处理与 traceback 原语；`am_session_generate` 为当前产品冻结目标（PRD 定义）。

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
| 参数 | `code`：`am_status_code` 枚举；`message`：错误描述字符串（对应 `@param`） |
| 返回值 | 错误句柄，必须由 `am_error_destroy` 释放（对应 `@return`） |
| 语义 | 错误对象生命周期独立；句柄所有权转移给调用方（对应 `@note`） |
| 前置条件 | `message` 指向有效 NUL 结尾字符串（对应 `@pre`） |

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

### `RuntimeBuilder`（include/aethermind/runtime/runtime_builder.h）

| 接口 | 签名 | 语义要点 |
|---|---|---|
| `WithOptions` | `RuntimeBuilder& WithOptions(const RuntimeOptions& options)` | 设置运行时装配选项；返回 `*this` 支持链式调用 |
| `RegisterCustomAllocatorProvider` | `RuntimeBuilder& RegisterCustomAllocatorProvider(DeviceType, std::unique_ptr<AllocatorProvider>)` | 注册自定义分配器提供者；所有权转移 |
| `RegisterBackendFactory` | `RuntimeBuilder& RegisterBackendFactory(DeviceType, std::unique_ptr<BackendFactory>)` | 注册后端工厂；所有权转移 |
| `Build` | `Runtime Build()` | 装配 AllocatorRegistry + BackendRegistry + KVCacheManager 为 `Runtime`；初始化失败（含分配失败）时 abort |
| `TryBuild` | `StatusOr<Runtime> TryBuild()` | 与 `Build` 相同的装配，但把 KV 初始化失败与注册阶段的分配失败作为 Status 返回（分配失败为空 message 的 `kResourceExhausted`）。成功后消费 pending 注册；失败时两个 pending 列表只保留从未转移的项，builder 仍可复用而不会 abort，但已转入被丢弃 registry 的注册项不可恢复 |

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

## 3. 语义约束

- C ABI 句柄（`ObjectHandle`、`am_error_handle`）为不透明指针，不暴露内部布局；对象生命周期由引用计数/显式 destroy 管理。
- `am_status_code` 覆盖当前产品所需的错误类别（OK/CANCELLED/UNKNOWN/INVALID_ARGUMENT/.../UNAUTHENTICATED）。
- `Executor::Execute` 是单计划一次性执行，不具备跨步骤状态管理（Prefill/Decode 状态机为当前产品目标，见[架构总览](../designs/architecture/architecture_overview.md) §八）。
- `Session::Generate` 与 `am_session_generate` 尚未实现；当前公开 API 形态为底层构建块（PRD 定义的生成入口为冻结目标）。

## 4. 相关文档

- 头文件：`include/c_api.h`、`include/aethermind/model/model_loader.h`、`include/aethermind/runtime/runtime_builder.h`、`include/aethermind/execution/executor.h`（Doxygen 注释为语义事实源）
- 架构总览：[docs/designs/architecture/architecture_overview.md](../designs/architecture/architecture_overview.md)（§一 API 服务层、§八 差距概要）
- ModelLoader 模块设计：[docs/designs/model/01-model-loader.md](../designs/model/01-model-loader.md)
- 注释规范：[docs/guides/cpp_comment_guidelines.md](../guides/cpp_comment_guidelines.md)
