## 1. 总体方案

AetherMind 没有引入外部配置文件格式（如 YAML/TOML/.env），而是采用三层纯 C++ 的结构化配置模型：

1. **编译期/构建期**：通过 `CompilePreset`（`kO1Plain` / `kO2Packed`）与 `DeviceType` 选择图优化级别、是否启用 packed weight、lowering selector；由顶层 `CMakePresets.json` 与 `cmake/*.cmake` 驱动。
2. **运行前装配期**：用户输入 `CpuRuntimeAssemblyOptions`，经 `MakeCpuRuntimeOptions()` 映射为完整的 `RuntimeOptions`。
3. **运行时装配**：`RuntimeBuilder::WithOptions()` → `TryBuild()` 执行选项校验、后端/分配器注册表组装、KV cache 物理布局规划与资源上限检查。
4. **进程级环境变量**：通过 `aethermind/base/env.{h,cpp}` 提供的 `set_env` / `get_env` / `check_env` 接口统一访问，并由 `RegisterEnvs` 单例在模块初始化时注入默认值。

文档 `docs/guides/runtime-configuration.md` 将这一流程描述为：

```
Load model → validate execution policy → MakeCpuRuntimeOptions
    → Compile → RuntimeBuilder::WithOptions → TryBuild
    → ExecutableModel::Prepare → InferenceSession::Create → Generate
```

该时序保证模型几何、KV 容量与预算在资源分配前一次确定。

## 2. 关键文件与包

- `include/aethermind/runtime/runtime_options.h`：定义 `RuntimeOptions` 及其六个子结构 `AllocatorRuntimeOptions`、`BackendRuntimeOptions`、`ExecutionRuntimeOptions`、`WorkspaceRuntimeOptions`、`TracingRuntimeOptions`、`KVCacheRuntimeOptions`。
- `include/aethermind/inference/inference_options.h`：声明 `CompilePreset`、`CpuRuntimeAssemblyOptions`、`MakeCompileOptions`、`MakeCpuRuntimeOptions`。
- `src/inference/inference_options.cpp`：实现从模型 metadata (`HfModelConfig`) + 用户输入到完整 `RuntimeOptions` 的映射；固定 CPU/FP32/KV geometry，显式设置所有六组字段。
- `src/runtime/runtime_builder.cpp`：`RuntimeBuilder::WithOptions` / `ValidateOptions` / `BuildAllocatorRegistry` / `BuildBackendRegistry` / `TryBuildKVCacheManager`，是运行时配置的最终验证与装配点。
- `include/aethermind/base/env.h` + `src/base/env.cpp`：跨平台环境变量封装，提供线程安全的 `get_env` / `set_env` / `has_env` / `check_env` 与 `RegisterEnvs` 全局注册器。
- `docs/guides/runtime-configuration.md`：运行前配置契约文档，约束调用方组合规则。
- `CMakePresets.json`、`cmake/*.cmake`：构建期配置入口。

## 3. 架构与约定

### 3.1 结构化 `RuntimeOptions`

`RuntimeOptions` 是值对象，不依赖模型，也不持有运行时状态；它仅描述 allocator/backend/execution/workspace/tracing/kv_cache 六大维度的开关与参数。`MakeCpuRuntimeOptions` 是唯一被应用入口复用的构造路径，它直接填充每个子结构的所有成员，而不是依赖隐式默认值——这使配置变更可审计。

### 3.2 分层装配

| 层级 | 类型 | 职责 |
|---|---|---|
| 用户输入 | `CpuRuntimeAssemblyOptions` | `context_tokens`、`kv_max_bytes`、`workspace_limit_bytes`、`cpu_feature_policy`、`kv_dtype` |
| 模型感知映射 | `MakeCpuRuntimeOptions(HfModelConfig, CpuRuntimeAssemblyOptions)` | 生成 FP32 KV geometry、固定 CPU-only backend/allocator、关闭未实现特性 |
| 运行时装配 | `RuntimeBuilder::WithOptions` → `TryBuild` | 校验选项、注册 registry、规划 KV layout、检查字节上限 |

### 3.3 环境变量机制

`env.cpp` 用 `std::shared_mutex` 保护对 `std::getenv` / `std::setenv` 的访问。`check_env` 把字符串解析为布尔：`"0"` → `false`，`"1"` → `true`，其他值返回空 `optional<bool>`。模块初始化时用 `RegisterEnvs::Global().SetEnv(...)` 注入两条进程级默认值：

- `THP_MEM_ALLOC_ENABLE=1`
- `TRACEBACK_LIMIT=512`

当前代码库中 `set_env` 仅在测试和 `env.cpp` 自身使用；业务逻辑通过 `check_env` 读取开关。

### 3.4 构建期配置

`CompilePreset` 枚举驱动 `MakeCompileOptions`：`kO1Plain` → opt_level=1、禁用 packed weight；`kO2Packed` → opt_level=2、启用 packed weight。目标设备 `DeviceType` 独立于后端可用性，仅用于 lowering selector。

## 4. 约定与约束

以下约束来自源码注释或 `ValidateOptions` 中的显式拒绝：

- **CPU/FP32 固定策略**：`MakeCpuRuntimeOptions` 始终设置 `backend.enable_cpu=true`、`enable_cuda=false`、`enable_cann=false`；GPU/CANN 仅在调用方自行注册 custom factory 后才允许开启（见 `ValidateOptions` 中对 `backend.enable_cuda` / `backend.enable_cann` 的检查）。来源：`src/runtime/runtime_builder.cpp`。
- **未实现选项拒绝**：`execution.enable_default_stream=false`、`execution.enable_graph_executor=true`、`workspace.enable_workspace_manager=true`、`tracing.enable_profiling=true`、`tracing.enable_memory_profiling=true` 均返回 `Status::Unimplemented`。来源：`src/runtime/runtime_builder.cpp::ValidateOptions`。
- **零容量无效**：`CpuRuntimeAssemblyOptions::context_tokens = 0` 被视为无效（`context_tokens` 须为正且不超过模型 context），参考 `inference_options.h` 注释与 `MakeCpuRuntimeOptions` 对 `MakeFp32KVCacheOptions` 的调用。来源：`include/aethermind/inference/inference_options.h`。
- **零字节上限表示无限制**：`kv_max_bytes`、`workspace_limit_bytes`、`kv_cache.max_bytes` 为零即不施加限制。来源：`docs/guides/runtime-configuration.md` 与 `runtime_options.h` 注释。
- **KV 容量不可超限**：`TryBuildKVCacheManager` 计算 `layout.TotalBytes()`，若 `max_bytes != 0 && required_bytes > max_bytes` 返回 `ResourceExhausted`。来源：`src/runtime/runtime_builder.cpp`。
- **`RuntimeOptions` 不可变于构建后**：文档明确指出“Changing this result after TryBuild does not reconfigure the existing Runtime”。来源：`include/aethermind/inference/inference_options.h`。
- **未知 compile preset 拒绝**：`MakeCompileOptions` 对非 `kO1Plain` / `kO2Packed` 的值返回 `InvalidArgument("Unknown compile preset")`。来源：`src/inference/inference_options.cpp`。
- **环境变量写入需覆盖标志**：`set_env(name, value, overwrite)` 第三个参数控制 `setenv` 的 overwrite 行为；`RegisterEnvs::SetEnv` 默认传入 `false`，避免覆盖已有值。来源：`src/base/env.cpp`。
- **C ABI 边界隔离**：C ABI v1 仍由 ABI config 显式提供 KV geometry，并在加载模型后校验 Runtime/model 一致性；其 CPU 默认值直接来自 `RuntimeOptions{}`，不复用 CLI/benchmark 的模型感知映射。来源：`docs/guides/runtime-configuration.md`。
