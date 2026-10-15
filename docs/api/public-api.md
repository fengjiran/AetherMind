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
| 简介 | 受管 backing storage 的唯一分配入口。基类以 NVI 实现：校验请求 → 调用派生类的 protected `AllocateStorage` hook → 校验返回的 ownership、device 和 alignment → 完成计费记录及 `Buffer` 构造 → 提交计账。KV storage、packed weights、workspace 与 activation 四类存储都经此入口，派生类因此无需各自记账。 |
| 请求契约 | `AllocationRequest{nbytes, alignment, tag}`。`nbytes` 是 `Buffer` 的可见大小，布局 padding 由资源所属模块计入这个值。`alignment == 0` 使用 allocator 默认策略；非零值必须为 2 的幂，且表示**最低**对齐而非精确对齐——返回的 `Buffer` 可能声明更强对齐，调用方必须读取实际值。CPU 实现会把弱于 `sizeof(void*)` 的请求提升而不是拒绝。`tag` 为不透明分类标签，仅用于统计，不携带模型/graph/kernel 对象。 |
| 计账与生命周期 | 成功分配按 `max(nbytes, 1)` 计入共享 `AllocationState`（零字节 `Buffer` 仍向系统申请 1 byte）。只有完整构造成功后才提交成功计数和峰值。计费记录由 `MemoryHandle` 的 deleter 拥有并强持 `AllocationState`，不借用 `Allocator*` 或 `Runtime*`，因此 allocator 实例或 provider 被销毁后，存活的 `Buffer` 仍能安全释放并正确退还。共享同一 `BufferImpl` 的多个副本只在最终释放时退还一次；deleter 先释放底层存储再退还额度，且全程不分配、不抛出。Registry 会拒绝未转发其共享状态的 provider 产物，避免 Runtime 账本漏记。 |
| 失败语义 | 非零且非 2 的幂的 alignment 返回 `kInvalidArgument`，且校验先于任何分配；hook 返回借用内存、错误 device、无效或不足的 alignment、CPU 指针不满足声明对齐时返回 `kInternal`。系统分配失败或计账/Buffer 元数据分配失败返回空 message 的 `kResourceExhausted`；失败路径释放已取得的 owning storage，不改变存活量、成功次数、释放次数或峰值。hook 返回 `kResourceExhausted` 时增加系统失败计数。`Allocate(size_t)` 保留为 wrapper，内部转调本入口，把 `kResourceExhausted` 映射为 `std::bad_alloc`，其余映射为 `Error`，因此无法绕过计账。 |
| 统计边界 | `AllocationState` 只覆盖受管 backing storage 的申请量，不含 mmap raw weights、普通 STL 分配、`BufferImpl` 与计费记录本身、libc 内部开销。**这不是进程 RSS 上限，也不能单独证明满足 PRD 的总内存目标。** 当前只统计不拒绝，预算强制属于后续阶段。 |

### `Executor::Execute`（include/aethermind/execution/executor.h）

| 项 | 内容 |
|---|---|
| 简介 | 按序执行计划中的每一步（对应 `@brief`） |
| 参数 | `plan`：待执行计划；`context`：由 `ExecutionContext::Create` 产生的 prepared tensor bindings、workspace 和 KV view，调用期间必须保持有效（对应 `@param`） |
| 返回值 | 成功返回 `Status::Ok()`；失败返回首个失败步骤的错误（对应 `@return`） |
| 语义 | 同步单线程执行，委托 `LayerRunner`；不产生 Token IDs（对应 `@note`） |

### `PrepareExecutionBindings` / `ExecutionContext::Create`

| 接口 | 语义要点 |
|---|---|
| `PrepareExecutionBindings(const ExecutionPlan&, const ExternalTensorBindings&, Allocator&)` | cold path specialization：校验 external dtype/shape/stride、runtime constraints 与 kernel-specific layout/aliasing，分配 activation 并构造 prepared params。external tensor backing 由调用方借出，改变其地址、shape、stride、dtype 或 alias 前必须重新 prepare。 |
| `ExecutionContext::Create(const ExecutionPlan&, PreparedExecutionBindings, WorkspaceArena*, KVCacheSessionView)` | 消费 prepared bindings，校验 plan binding key、非零 workspace 的 arena presence 与 state aliases 的 KV view presence；不拥有 plan、arena 或 KV storage。 |
| `ExecutionContext::Clear()` | 释放 owned prepared bindings 并清除 borrowed handles；不 reset WorkspaceArena，也不 release KV reservation。 |

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
