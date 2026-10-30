# CPU Runtime 运行前配置

状态：Current。本文定义当前 CPU、单模型、单请求范围内，在 `RuntimeBuilder::TryBuild()` 前集中派生和校验 Runtime 配置的方法。它不增加 GPU、调度、全局资源账本或新的 C ABI。

## 职责与时序

`RuntimeOptions` 是 Runtime 层的资源装配值对象；它不依赖模型。模型几何到 CPU/FP32 Runtime 配置的映射属于 inference 层：`MakeCpuRuntimeOptions` 读取已经通过 `ModelLoader` 结构校验的 `HfModelConfig`，生成不借用模型的独立 `RuntimeOptions`。

```text
Load model → validate execution policy → MakeCpuRuntimeOptions
    → Compile → RuntimeBuilder::WithOptions → TryBuild
    → PrepareExecutableModel → InferenceSession::Create → Generate
```

此顺序使模型配置、KV 容量和预算在资源分配前一次确定。`Runtime` 必须在 `PrepareExecutableModel` 前移动到最终地址，之后直到 model、session 销毁前都不能移动。

## 共享 CPU 映射

`CpuRuntimeAssemblyOptions` 与 `MakeCpuRuntimeOptions` 和编译 preset 共同声明于 `include/aethermind/inference/cpu_inference_options.h`，调用方无需再包含单独的 runtime options 头。

`CpuRuntimeAssemblyOptions` 是调用方设置的运行前输入：

| 字段 | 含义 |
|---|---|
| `context_tokens` | 必填的物理 KV 容量；须为正且不超过模型 context |
| `kv_max_bytes` | K/V 两个 plane 的可见 payload 上限；0 为无限制 |
| `workspace_limit_bytes` | 单请求 workspace payload 上限；0 为无限制 |
| `cpu_feature_policy` | CPU kernel 选择的 required/disabled feature policy |

`MakeCpuRuntimeOptions` 统一设置 CPU allocator/backend 默认值、FP32 KV geometry、两个 Runtime 阶段预算和 feature policy。它只读取 metadata；不读取权重 payload、不分配 Buffer、不创建 Runtime。`RuntimeBuilder` 负责真正的 KV layout 规划、物理字节上限检查及资源分配。

模型准备和请求的预算不属于 `RuntimeOptions`：分别使用 `ExecutableModelPrepareOptions` 与 `InferenceSessionOptions`，因为它们需要已编译的 artifact 或具体 prompt/generation。它们可以和这份 Runtime 配置一起在产品入口处保存，但仍在各自阶段应用。

```cpp
using namespace aethermind;

StatusOr<std::vector<uint32_t>> GenerateWithUnifiedConfiguration(
        const std::filesystem::path& model_dir,
        std::span<const uint32_t> prompt) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_dir));
    AM_RETURN_IF_ERROR(ValidateFp32ExecutionInputs(*loaded));

    CpuRuntimeAssemblyOptions runtime_input{
            .context_tokens = 4096,
            .kv_max_bytes = 512U * 1024U * 1024U,
            .workspace_limit_bytes = 8U * 1024U * 1024U,
            .cpu_feature_policy = {},
    };
    AM_ASSIGN_OR_RETURN(auto runtime_options,
                        MakeCpuRuntimeOptions(loaded->GetConfig(), runtime_input));
    AM_ASSIGN_OR_RETURN(auto compile_options,
                        MakeCpuCompileOptions(CpuInferencePreset::kO1Plain));
    AM_ASSIGN_OR_RETURN(auto artifact,
                        ModelCompiler::Compile(std::move(loaded), compile_options));

    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    AM_ASSIGN_OR_RETURN(Runtime runtime, builder.TryBuild());

    const ExecutableModelPrepareOptions prepare_options{
            .max_model_payload_bytes = 1024U * 1024U * 1024U,
            .max_packed_weight_bytes = 512U * 1024U * 1024U,
    };
    AM_ASSIGN_OR_RETURN(auto executable,
                        PrepareExecutableModel(runtime, std::move(artifact), prepare_options));
    auto model = std::make_shared<ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(
            runtime, model, {.max_request_payload_bytes = 16U * 1024U * 1024U}));
    return session.Generate(prompt, {.max_new_tokens = 32});
}
```

CLI 和 model benchmark 都复用此映射；它们的既有命令行参数和 benchmark 报告 schema 不因此增加字段。

## Builder 支持边界

`RuntimeBuilder::TryBuild()` 在组装 registry 或转移 custom registrations 前验证选项。以下非默认请求目前返回 `Unimplemented`：

- `execution.enable_default_stream = false`
- `execution.enable_graph_executor = true`
- `workspace.enable_workspace_manager = true`
- `tracing.enable_profiling = true` 或 `enable_memory_profiling = true`

`backend.enable_cuda` 与 `backend.enable_cann` 只有在 Builder 已持有同设备 custom `BackendFactory` 时才允许。它只允许外部后端接入，不提供内置 GPU backend，也不构成 GPU 推理支持。选项预检失败不会消费 pending registration；registry 组装开始后的失败仍遵循原有转移语义：已转移到丢弃 registry 的 registration 不会回到 Builder。 CPU `required_features` 等实际能力仍由 `CpuBackendFactory` 在惰性创建 backend 时检查；本次预检不提前创建 backend。

## C ABI 边界

C ABI v1 的生命周期是先 `am_runtime_create`、后 `am_model_load`。因此它仍由 ABI config 显式提供 KV geometry，并在加载模型后校验 Runtime/model 一致性；不能等待模型 metadata 后再创建 Runtime。其 CPU 默认值直接来自 `RuntimeOptions{}`，没有复制 CLI/benchmark 的模型感知映射。C ABI 的签名、配置 struct 和 strict FP32 准入均未改变。

验证结果见 [运行前 Runtime 配置验证](../tests/runtime_configuration_validation_2026-10-30.md)。
