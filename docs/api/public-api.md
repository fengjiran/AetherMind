# AetherMind 公共 API 参考

> 本文件汇总 AetherMind 公共 API 的语义，与头文件 Doxygen 注释**同源同步**：修改头文件注释必须同步本文件，反之亦然（[文档系统规范](../guides/documentation-guide.md) §7.3）。
>
> 构建选项与运行配置以根 [README.md](../../README.md) 为单一事实源，本文件不复制其表格。Phase 1 目标 API（`am_session_generate`、`Session::Generate`）尚未实现，见 [PRD](../products/aethermind_prd.md)，不在此列。

## 1. C ABI（include/c_api.h）

> 当前 C ABI 仅提供对象引用计数、错误处理与 traceback 原语；`am_session_generate` 为 Phase 1 冻结目标（PRD 定义）。

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
| `Build` | `Runtime Build()` | 装配 AllocatorRegistry + BackendRegistry + KVCacheManager 为 `Runtime` |

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
| `ExecutionContext::Create(const ExecutionPlan&, PreparedExecutionBindings, WorkspaceArena*, KVCacheView)` | 消费 prepared bindings，校验 plan binding key、非零 workspace 的 arena presence 与 state aliases 的 KV view presence；不拥有 plan、arena 或 KV storage。 |
| `ExecutionContext::Clear()` | 释放 owned prepared bindings 并清除 borrowed handles；不 reset WorkspaceArena，也不 release KV reservation。 |

## 3. 语义约束

- C ABI 句柄（`ObjectHandle`、`am_error_handle`）为不透明指针，不暴露内部布局；对象生命周期由引用计数/显式 destroy 管理。
- `am_status_code` 覆盖全部 Phase 1 错误类别（OK/CANCELLED/UNKNOWN/INVALID_ARGUMENT/.../UNAUTHENTICATED）。
- `Executor::Execute` 是单计划一次性执行，不具备跨步骤状态管理（Prefill/Decode 状态机为 Phase 1 目标，见[架构总览](../designs/architecture/architecture_overview.md) §八）。
- `Session::Generate` 与 `am_session_generate` 尚未实现；当前公开 API 形态为底层构建块（PRD 定义的生成入口为冻结目标）。

## 4. 相关文档

- 头文件：`include/c_api.h`、`include/aethermind/model/model_loader.h`、`include/aethermind/runtime/runtime_builder.h`、`include/aethermind/execution/executor.h`（Doxygen 注释为语义事实源）
- 架构总览：[docs/designs/architecture/architecture_overview.md](../designs/architecture/architecture_overview.md)（§一 API 服务层、§八 差距概要）
- ModelLoader 模块设计：[docs/designs/model/01-model-loader.md](../designs/model/01-model-loader.md)
- 注释规范：[docs/guides/cpp_comment_guidelines.md](../guides/cpp_comment_guidelines.md)
