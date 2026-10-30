# 资源预算与分配前准入

本文说明当前 CPU、单模型、单请求产品范围内的资源规划接口。它为可确定的 tensor payload 建立分阶段上限，不是进程 RSS、allocator capacity 或 Runtime 全局内存配额系统。

## 范围与单位

Runtime、Prepare 和 Session options 中的整数 limit 单位均为可见 payload bytes，`0` 表示 unlimited。低层 `PrepackWeightRequests` 的可选上限以 `std::nullopt` 表示无限制，显式 `0` 表示零 payload 上限，避免把恰好耗尽的模型剩余预算变成无限制。它们不包含 allocator metadata 或内部 capacity rounding、`std::vector`/prepared-kernel metadata、输入输出 token vector、mapped/external backing payload（字节另报）和 OS RSS。

| 阶段 | 接口 | 覆盖的 payload | 失败时间 |
|---|---|---|---|
| Runtime 构造 | `KVCacheRuntimeOptions::max_bytes` | layout padding 后的 K/V 两块 plane | 获取 KV allocator、分配 plane 前 |
| Request | `WorkspaceRuntimeOptions::default_workspace_limit_bytes` | Prefill/Decode 共用的最大 workspace | reservation 与 workspace 分配前 |
| Request | `InferenceSessionOptions::max_request_payload_bytes` | `workspace + max(prefill activation, decode activation)` | reservation、workspace 与 activation 分配前 |
| Model 准备 | `ExecutableModelPrepareOptions::max_packed_weight_bytes` | batch 查询的 packed storage 上界 | 任一 packed artifact 分配前 |
| Model 准备 | `ExecutableModelPrepareOptions::max_model_payload_bytes` | known owned raw backing、unique inline constants、packed payload | 准备阶段、packing 前；原始 backing 不完整时拒绝有限 cap |

`KVCacheManager::Init` 是低层物理存储接口，保留原有几何和 allocator 合同；它不会自动读取 Runtime 的 `max_bytes`。KV cap 只在 `RuntimeBuilder` 组装 Runtime 时生效。

## 查询、准入与分配

```text
KV geometry ── PlanKVCacheLayout ── KV cap ── RuntimeBuilder 分配 K/V

ExecutionPlan + concrete external shapes
  ── PlanExecutionBindings ── request caps ── MaterializeExecutionBindings
  ── Prefill arena，释放后再 materialize Decode arena

packing request + exact recipe
  ── Backend::GetWeightPackingRequirements ── packed/model caps ── PackWeights
```

`PlanExecutionBindings` 只验证 bindings 并冻结 concrete activation layout；它不分配 activation Buffer 或构造 kernel params。`MaterializeExecutionBindings` 只能消费兼容的规划对象，随后一次分配 arena 并完成 params preparation。`InferenceSession::PlanResources(prompt_len, config)` 同样不执行、不会 reservation KV 或分配 tensor Buffer；它报告本次 concrete request 的 workspace、两阶段 activation peak、当前 Runtime 的完整物理 KV payload 以及已准备模型的 payload 观察值。

Request 的两个 activation arena 不同时存活：Prefill context 清理后才构造 Decode context。因此 request peak 不是两者相加。`max_new_tokens == 0` 保持原合同：不会申请 KV、workspace 或 activation，也不因为任何 request budget 失败。

正数 token 请求仍先检查模型 context 与 KV reservation；容量不足或已有 reservation 的既有 `OutOfRange`/`FailedPrecondition` 不被 budget 的 `ResourceExhausted` 覆盖。

## 模型 backing 与 packing

`RawStorage::footprint()` 是可选只读 metadata。HF safetensors mmap 提供完整 mapping bytes；自定义 storage 未提供时，`ModelWeightStorageSummary::backing_complete` 为 false。摘要按 RawStorage ownership identity 合并重叠 view 区间，按 Buffer implementation 去重 packed payload。未知 backing 绝不按零计费：只有设置 `max_model_payload_bytes` 时，未知 source backing 才拒绝该模型 cap；仅设置 packed cap 仍可使用精确 packing query。

`Backend::GetWeightPackingRequirements` 只读取 component metadata，返回 exact recipe 所需的 visible bytes 和 alignment。有限 packed/model cap 要求此 query 成功并在 batch 内全部 artifact 分配前检查。分配前按 request 汇总需求上界，完成后按共享 Buffer implementation 去重实际值；当前 CPU 每个 request 创建独立产物，两者一致。旧 backend 在 unlimited 路径可返回 `Unimplemented`，此时 model 的 `packing_plan_complete` 为 false；不能把未知大小当作零。

`ExecutableModel::resource_usage()` 返回当前模型的观察值：raw backing 分类、inline constants、unique packed bytes、known managed payload 与 planning completeness。它不是多模型 Runtime 的 live-byte ledger。一个 Runtime 内并存的多个模型、独立 Session、allocator 外部使用和整个进程的 RSS 仍不受本接口控制。

## 使用示例

以下调用方只加载一次模型，并在 Runtime 固定地址后准备模型；所有借用对象在同一作用域内销毁。示例函数返回 owning Token IDs：

```cpp
using namespace aethermind;

StatusOr<std::vector<uint32_t>> GenerateWithCaps(
        const std::filesystem::path& model_dir,
        std::span<const uint32_t> prompt, size_t context_tokens) {
    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_dir));
    AM_RETURN_IF_ERROR(ValidateFp32ExecutionInputs(*loaded));
    AM_ASSIGN_OR_RETURN(auto runtime_options,
                        MakeCpuRuntimeOptions(
                                loaded->GetConfig(),
                                {.context_tokens = context_tokens,
                                 .kv_max_bytes = 64U * 1024U * 1024U,
                                 .workspace_limit_bytes = 2U * 1024U * 1024U}));
    AM_ASSIGN_OR_RETURN(auto compile_options, MakeCpuCompileOptions(CpuInferencePreset::kO1Plain));
    AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), compile_options));
    RuntimeBuilder builder;
    builder.WithOptions(runtime_options);
    AM_ASSIGN_OR_RETURN(Runtime runtime, builder.TryBuild());
    const ExecutableModelPrepareOptions prepare_options{
        .max_model_payload_bytes = 512U * 1024U * 1024U,
        .max_packed_weight_bytes = 256U * 1024U * 1024U,
    };
    AM_ASSIGN_OR_RETURN(auto executable,
                        PrepareExecutableModel(runtime, std::move(artifact), prepare_options));
    auto model = std::make_shared<ExecutableModel>(std::move(executable));
    AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(
        runtime, model, {.max_request_payload_bytes = 8U * 1024U * 1024U}));
    AM_ASSIGN_OR_RETURN(auto usage, session.PlanResources(prompt.size(), {.max_new_tokens = 32}));
    // usage is a cold-path requirements report; Generate performs admission again.
    return session.Generate(prompt, {.max_new_tokens = 32});
}
```

原有 `PrepareExecutableModel`、`LoadAndPrepareExecutableModel` 与 `InferenceSession::Create` overload 仍保留，默认均不启用新的可选 caps。C ABI v1 的 struct、函数和默认 FP32 行为未改变。

CPU Runtime 的完整配置映射和 Builder 的 option 预检见 [运行前 Runtime 配置](runtime-configuration.md)。

## 兼容与边界

`RawStorage`、`Backend`、`Runtime` 和 execution/inference 的 C++ public contracts 增加了资源查询或可选 options；所有直接使用这些 C++ headers 的 consumer 应随本次版本重新编译。此要求不改变 C ABI v1。

本机制不引入 Paged KV、全局 allocator interception、请求调度、并发 reservation 或性能承诺。它的作用是让已有的 static KV、prepared bindings 和 packing 在能确定 payload 时于分配前失败，并为后续 FP16/BF16 compatibility mode 提供准备期峰值的可检查边界。
