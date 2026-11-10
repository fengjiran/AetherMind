# FP32 兼容推理指南

状态：Current。适用范围是 CPU Llama Dense 的 P1：将已由 loader 接受的 F16/BF16 checkpoint 权重显式拓宽为 FP32，再复用既有 FP32 O1/plain 执行链。实现和验证范围见 [11 号方案](../improvement-plan/11-fp16-bf16-execution-support.md) 与 [P1 验证报告](../tests/fp32_compatibility_validation_2026-10-29.md)。

## 使用入口

`Runtime` 必须在转换前创建并保持最终地址，因为转换 payload 使用它的 CPU allocator。如果 Runtime 已按模型几何配置，可使用显式 convenience 入口；该入口只加载一次：

```cpp
ModelLoadOptions options{
        .preset = CompilePreset::kO1Plain,
        .target = DeviceType::kCPU,
        .weights = WeightPolicy::kConvertToFp32,
        .limits = {.max_model_payload_bytes = 512U * 1024U * 1024U},
};
auto executable = ExecutableModel::Load(runtime, model_dir, options);
```

需要先读取 config 以组装 Runtime 时，可以组合公开的阶段入口，同样只加载一次。下面的例子显式传播相同的模型 cap，并在编译前释放只供转换使用的 source owner：

```cpp
using namespace aethermind;

StatusOr<std::vector<uint32_t>> GenerateCompatible(
        const std::filesystem::path& model_dir,
        std::span<const uint32_t> prompt, size_t context_tokens) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_dir));
    AM_RETURN_IF_ERROR(ValidateSourceWeights(*loaded, WeightPolicy::kConvertToFp32));
    AM_ASSIGN_OR_RETURN(auto runtime_options,
                        MakeCpuRuntimeOptions(
                                loaded->GetConfig(),
                                {.context_tokens = context_tokens}));
    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    AM_ASSIGN_OR_RETURN(Runtime runtime, builder.TryBuild());

    const ModelLimits resources{
            .max_model_payload_bytes = 512U * 1024U * 1024U};
    AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
    AM_ASSIGN_OR_RETURN(auto converted,
                        ConvertRawModelWeightsToFp32(
                                *loaded, *allocator,
                                {.max_owned_weight_bytes = resources.max_model_payload_bytes}));
    loaded = std::move(converted);
    AM_ASSIGN_OR_RETURN(auto compile_options,
                        MakeCompileOptions(CompilePreset::kO1Plain, DeviceType::kCPU));
    compile_options.graph_build = {.activation_dtype = DataType::Float32(),
                                   .kv_dtype = DataType::Float32()};
    AM_ASSIGN_OR_RETURN(auto artifact,
                        ModelCompiler::Compile(std::move(loaded), compile_options));
    AM_ASSIGN_OR_RETURN(auto executable,
                        ExecutableModel::Prepare(runtime, std::move(artifact), resources));
    auto model = std::make_shared<ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(runtime, model));
    return session.Generate(prompt, {.max_new_tokens = 3});
}
```

转换不是 `ModelCompiler::Compile`、`LoadAndCompile` 或普通 `ExecutableModel::Prepare` 的隐式行为。上例中的局部 `Runtime` 先于 model/session 创建，销毁顺序满足借用合同。

`MakeCpuRuntimeOptions` 是值映射，不读取 payload 或分配转换/KV storage；相关 budget、CPU feature policy 和 Builder 预检见 [运行前 Runtime 配置](runtime-configuration.md)。

`kConvertToFp32` 目前只允许 CPU 目标与 `CompilePreset::kO1Plain`。非 CPU 目标会在读取 checkpoint 或分配转换 payload 前返回 Unimplemented。未知 preset、O2/packed 和不受支持的 source dtype 在分配转换 payload 前失败。C ABI v1 和 `aethermind_infer` CLI 继续使用 strict FP32 source admission；它们不会因这个 C++ 入口 放宽输入。

## 转换和所有权

每个必需、连续、标量的实际权重 view 必须为 F32、F16 或 BF16。F16/BF16 会在准备前写入 Runtime CPU allocator 所分配的独立 `RawStorage`；graph 的 activation、KV state、weights 和最终 logits 均为 F32。配置的 `weight_dtype_hint` 保留为 checkpoint provenance，不能作为伪造的执行 dtype。

F32 view 零拷贝并继续共享原 owner。相同 source owner、起始地址、范围、dtype 和 shape 的 reduced view 只转换一次，因此 tied embedding/lm_head 继续共享转换 storage；内容相同但来源不同的 tensor 不合并。无用的 F16/BF16 source owner 不会被结果模型保活，仍被共享的 F32 owner 会保留。

转换不要求 source 对齐，也不将原始字节解释为对齐的 C++ reduced-float 对象。有限值（包括 subnormal）精确拓宽，保留 signed zero；NaN 保留 sign/payload 并在 FP32 结果置 quiet bit。F16 的批量 widening 可按 Runtime backend 的 effective features 使用 F16C，先以整数操作 quiet signaling NaN，保持不设置浮点异常的合同；BF16 使用独立的 portable scalar 批量转换。backend 禁用 F16C 或 AVX 时，F16 回退 scalar，不在 model 层重新检测 CPU。`ExecutableModel::Load` 分别查询 `GetHostFp16Conversions()` / `GetHostBf16Conversions()`，把 prepared widening 传入 `fp16_to_fp32` / `bf16_to_fp32`；每个唯一 reduced tensor 仅调用相应 callback 一次。独立调用 `ConvertRawModelWeightsToFp32` 默认使用 scalar，也可显式传入遵循同一合同的两个非空 callback。BF16 双向基础 API 与包装类型的 NaN 区别见 [转换合同](../operators/bf16_conversion/README.md)。

## 资源观察与限额

`ExecutableModel::resource_usage().weight_fp32_conversion` 提供 source/converted backing 分类、唯一转换 source/target bytes 以及 source-owned 与转换 payload 同时存活时的 visible peak。它不持有 source owner，也不等于 RSS、allocator rounded capacity、mmap resident pages 或后续 packed artifacts 的峰值。

`ModelLoadOptions::limits.max_model_payload_bytes` 在转换 payload 分配前检查最终 retained FP32 owned bytes 加将创建的 converted bytes；之后 `ExecutableModel::Prepare` 继续以同一 cap 检查 inline constants 和 packed weights。该 cap 不是原 checkpoint mapping、临时 RSS 或进程级多模型 budget。0 保持现有“无限制”语义。

## 能力边界

P1 不是原生 F16/BF16 推理：权重不会保留 16-bit，activation、FP32 accumulation、KV 和 logits 没有新的 mixed-dtype kernel。F16C 改善的是准备阶段的权重转换，不代表 token 吞吐或峰值内存改善；实测范围见 [转换基准](../operators/fp16_conversion/README.md)。16-bit KV、A16 activation、GPU 和调度仍属于其他阶段。

公开 C++ 创建接口统一为 `ExecutableModel::Load` 和 `ExecutableModel::Prepare`，旧自由函数已删除，调用方须迁移并重新编译。`Load` 接收 `ModelLoadOptions`；任意编译配置通过显式 Load/Compile 后调用 `Prepare`。C ABI v1 的签名与 strict 输入合同保持不变。
