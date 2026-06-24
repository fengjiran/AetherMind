# AetherMind 图编译功能完整实现流程

> **文档版本**：v1.0
> **更新日期**：2026-05-30
> **适用范围**：Phase 1 — 桌面/服务器 CPU 本地推理运行时

---

## 目录

1. [概述](#1-概述)
2. [已实现与未实现的边界](#2-已实现与未实现的边界)
3. [阶段 1：运行时初始化](#3-阶段-1运行时初始化)
4. [阶段 2：模型加载](#4-阶段-2模型加载)
5. [阶段 3：执行计划构建](#5-阶段-3执行计划构建)
6. [阶段 4：执行](#6-阶段-4执行)
7. [完整函数调用序列总表](#7-完整函数调用序列总表)
8. [核心数据结构变换链](#8-核心数据结构变换链)
9. [未实现的设计路径（ModelGraph 模块）](#9-未实现的设计路径modelgraph-模块)
10. [关键技术点](#10-关键技术点)
11. [配置说明](#11-配置说明)
12. [测试方法](#12-测试方法)
13. [注意事项](#13-注意事项)

---

## 1. 概述

AetherMind 的图编译功能负责将 Hugging Face 格式的模型文件转换为可执行的推理计划。当前实现采用 **四阶段管线** 架构：

```
运行时初始化 → 模型加载 → 执行计划构建 → 执行
```

核心设计原则：

- **Plan-Build-Time Resolve**：所有 kernel 查找在构建期完成，执行期零间接调用
- **数据与逻辑分离**：`ModelInstance` 持有数据，`ExecutionPlanBuilder` 负责编译，`LayerRunner` 负责执行
- **Freeze-before-Resolve**：`KernelRegistry` 冻结后不可变，支持无锁读取
- **运行时形状校验**：kernel 执行前验证形状约束，防止内存越界

---

## 2. 已实现与未实现的边界

### 2.1 已实现模块

| 模块 | 核心文件 | 状态 |
|------|---------|------|
| RuntimeBuilder | `src/runtime/runtime_builder.cpp` | ✅ 已实现 |
| ModelLoader | `src/model/model_loader.cpp` | ✅ 已实现 |
| HfDirectoryReader | `src/model/formats/hf/hf_directory_reader.cpp` | ✅ 已实现 |
| HfModelValidator | `src/model/formats/hf/hf_model_validator.cpp` | ✅ 已实现 |
| HfWeightResolver | `src/model/formats/hf/hf_weight_resolver.cpp` | ✅ 已实现 |
| ModelInstanceBuilder | `src/model/model_instance_builder.cpp` | ✅ 已实现 |
| WeightPrepackPlanner | `src/model/weight_prepack_planner.cpp` | ✅ 已实现 |
| ExecutionPlanBuilder | `src/execution/execution_plan_builder.cpp` | ✅ 已实现 |
| Executor / LayerRunner | `src/execution/executor.cpp`, `layer_runner.cpp` | ✅ 已实现 |
| Operator 体系 | `src/operators/rmsnorm_op.cpp`, `embedding_op.cpp` | ✅ 已实现 |
| Kernel 体系 | `src/backend/cpu/kernels/rmsnorm/` | ✅ 已实现 |

### 2.2 未实现模块

| 模块 | 设计文档 | 状态 |
|------|---------|------|
| ModelGraph | `docs/designs/model_graph_design.md` | ❌ 未实现 |
| ModelGraphBuilder | 同上 | ❌ 未实现 |
| GraphToExecutionPlanLowering | 同上 | ❌ 未实现 |

**核心缺口**：`include/aethermind/model/graph/` 和 `src/model/graph/` 目录不存在。`ExecutionPlanBuilder::Build()` 所需的 `vector<ExecutionPlanNodeSpec>` 当前仅能在单元测试中手工构造，没有从已加载模型自动生成的生产级路径。

---

## 3. 阶段 1：运行时初始化

### 3.1 流程图

```
RuntimeBuilder
  ├─ .WithOptions(options)        → 配置运行时选项
  ├─ .RegisterBackendFactory(...)  → [可选] 注册自定义后端
  └─ .Build()                     → RuntimeContext
       ├─ BuildAllocatorRegistry()  → AllocatorRegistry
       ├─ BuildBackendRegistry()    → BackendRegistry
       └─ BuildKVCacheManager()     → KVCacheManager
```

### 3.2 函数调用序列

| # | 函数 | 参数 | 返回值 | 作用 |
|---|------|------|--------|------|
| 1.1 | `RuntimeBuilder::WithOptions` | `const RuntimeOptions&` | `RuntimeBuilder&` | 配置运行时选项 |
| 1.2 | `RuntimeBuilder::Build` | 无 | `RuntimeContext` | 构建完整运行时上下文 |
| 1.2a | `BuildAllocatorRegistry` | 无 | `AllocatorRegistry` | 注册内存分配器 |
| 1.2b | `BuildBackendRegistry` | 无 | `BackendRegistry` | 注册后端工厂 |
| 1.2c | `BuildKVCacheManager` | 无 | `KVCacheManager` | 初始化 KV cache 管理器 |

### 3.3 代码示例

```cpp
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/runtime/runtime_options.h"

// 构建运行时上下文
RuntimeOptions options;
options.allocator.enable_cpu = true;
options.backend.enable_cpu = true;
options.kv_cache.enable_manager = true;
options.kv_cache.num_layers = 32;
options.kv_cache.num_kv_heads = 32;
options.kv_cache.max_tokens = 4096;
options.kv_cache.head_dim = 128;

RuntimeContext runtime = RuntimeBuilder()
    .WithOptions(options)
    .Build();
```

### 3.4 关键技术点

- `AllocatorRegistry` 按 `DeviceType` 索引，支持 CPU/CUDA/CANN 三种分配器
- `BackendRegistry` 延迟创建后端实例（Factory 模式），首次 `GetBackend()` 时才实例化
- `KVCacheManager` 在 `enable_manager = false` 时返回空实例，无额外开销
- `RuntimeContext` 不可拷贝、可移动，生命周期通常覆盖整个推理会话

### 3.5 依赖关系

```
RuntimeOptions
  ├─ .allocator{enable_cpu, enable_cuda, enable_cann}
  │     → AllocatorRegistry
  │         ├─ CPUAllocatorProvider  (if enable_cpu)
  │         ├─ CUDAAllocatorProvider (if enable_cuda)
  │         └─ CANNAllocatorProvider (if enable_cann)
  ├─ .backend{enable_cpu}
  │     → BackendRegistry
  │         └─ CpuBackendFactory (if enable_cpu)
  └─ .kv_cache{enable_manager, num_layers, num_kv_heads, ...}
        → KVCacheManager (if enable_manager)

RuntimeContext = AllocatorRegistry + BackendRegistry + KVCacheManager
```

---

## 4. 阶段 2：模型加载

### 4.1 流程图

```
ModelLoadOptions{model_dir}
     │
     ▼
HfDirectoryReader::Open(model_dir)
  └─ InspectDirectory() → 检测目录布局
     │
     ├─ ParseConfig() → HfModelConfig
     │     └─ ValidateConfig() → 校验配置字段
     │
     └─ LoadRawWeightTable() → RawWeightTable
           └─ ValidateWeightSet() → 校验权重完整性
     │
     ▼
hf::ResolveWeights(config, weights) → ResolvedModelWeights
     └─ ValidateResolvedModel() → 校验解析结果
     │
     ▼
ModelInstanceBuilder::Create(config, resolved) → ModelInstance
     │
     ▼
WeightPrepackPlanner::BuildRequests(config, weights, backend, registry)
     │
     ▼
WeightPrepackPlanner::PrepackAndStore(model, requests)
     └─ CpuWeightPrepacker::Pack() → PackedWeights
     └─ ModelInstance::StorePackedWeights() → 存入 BackendSidecar
     │
     ▼
ModelInstance {config, resolved_weights, backend_sidecar{packed_weights}}
```

### 4.2 函数调用序列

| # | 函数 | 参数 | 返回值 | 作用 |
|---|------|------|--------|------|
| 2.1 | `ModelLoader::Load` | `ModelLoadOptions`, `const Backend&`, `const KernelRegistry&` | `StatusOr<unique_ptr<ModelInstance>>` | 模型加载总入口 |
| 2.2 | `HfDirectoryReader::Open` | `filesystem::path` | `StatusOr<HfDirectoryReader>` | 打开并检测 HF 目录布局 |
| 2.2a | `InspectDirectory` | `filesystem::path` | `StatusOr<HfDirectoryDescriptor>` | 验证目录结构 |
| 2.3 | `reader.ParseConfig` | 无 | `StatusOr<HfModelConfig>` | 解析 config.json |
| 2.4 | `HfModelValidator::ValidateConfig` | `const HfModelConfig&` | `Status` | 校验配置字段合法性 |
| 2.5 | `reader.LoadRawWeightTable` | 无 | `StatusOr<RawWeightTable>` | 加载 safetensors 权重 |
| 2.6 | `HfModelValidator::ValidateWeightSet` | `const HfModelConfig&`, `const RawWeightTable&` | `Status` | 校验权重完整性 |
| 2.7 | `hf::ResolveWeights` | `const HfModelConfig&`, `const RawWeightTable&` | `StatusOr<ResolvedModelWeights>` | 按层解析权重名 |
| 2.8 | `HfModelValidator::ValidateResolvedModel` | `const HfModelConfig&`, `const ResolvedModelWeights&` | `Status` | 校验解析后权重与配置一致性 |
| 2.9 | `ModelInstanceBuilder::Create` | `HfModelConfig`, `ResolvedModelWeights` | `StatusOr<unique_ptr<ModelInstance>>` | 构建 ModelInstance |
| 2.10 | `WeightPrepackPlanner::BuildRequests` | `const HfModelConfig&`, `const ResolvedModelWeights&`, `const Backend&`, `const KernelRegistry&` | `StatusOr<vector<Request>>` | 枚举需要预打包的权重 |
| 2.11 | `WeightPrepackPlanner::PrepackAndStore` | `ModelInstance&`, `const vector<Request>&` | `Status` | 执行预打包并存储 |

### 4.3 代码示例

```cpp
#include "aethermind/model/model_loader.h"
#include "aethermind/model/model_load_options.h"

// 加载模型
ModelLoadOptions options;
options.model_dir = "/path/to/llama-model";

auto& backend = *runtime.GetBackend(DeviceType::kCPU).value();
auto& registry = KernelRegistry::Global();

auto model = ModelLoader::Load(options, backend, registry);
if (!model.ok()) {
    // 处理加载错误
    return model.status();
}
// model.value() 是 std::unique_ptr<ModelInstance>
```

### 4.4 hf::ResolveWeights 权重名映射

按层枚举权重名，从 `RawWeightTable` 中查找并组织为结构化视图：

| 权重名模式 | 映射目标 | 必需 |
|-----------|---------|------|
| `model.embed_tokens.weight` | `embed_tokens` | 是 |
| `model.norm.weight` | `final_norm` | 是 |
| `lm_head.weight` | `lm_head` | 否（tie_word_embeddings 时可能缺失） |
| `model.layers.{i}.input_layernorm.weight` | `layers[i].norm.input_rmsnorm` | 是 |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers[i].norm.post_attn_rmsnorm` | 是 |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers[i].attn.q_proj` | 是 |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers[i].attn.k_proj` | 是 |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers[i].attn.v_proj` | 是 |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers[i].attn.o_proj` | 是 |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers[i].mlp.gate_proj` | 是 |
| `model.layers.{i}.mlp.up_proj.weight` | `layers[i].mlp.up_proj` | 是 |
| `model.layers.{i}.mlp.down_proj.weight` | `layers[i].mlp.down_proj` | 是 |

缺失必需权重时返回 `Status::InvalidArgument`。

### 4.5 WeightPrepackPlanner 预打包逻辑

**BuildRequests** 枚举策略：

- 每层 7 个线性投影权重：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- 可选 `lm_head`（如果存在）
- **排除**：`embed_tokens`, `input_rmsnorm`, `post_attn_rmsnorm`, `final_norm`（这些权重不需要预打包）
- 每个 Request 的 `op_type = OpType::kLinear`，`selector` 由 `MakePackedSelector(backend, dtype)` 生成

**PrepackAndStore** 执行逻辑：

1. 去重检查：同 `(op_type, selector)` 对只打包一次（因为所有 Linear 权重共享同一预打包策略）
2. 从 `RawWeightView` 构造 `TensorView`（含 strides 计算）
3. 调用 `CpuWeightPrepacker::Pack(op_type, view, selector)` 执行实际打包
4. 调用 `ModelInstance::StorePackedWeights()` 存入 `BackendSidecar`

### 4.6 关键技术点

- **HfDirectoryReader** 支持单文件和分片两种 safetensors 布局，混合布局会被拒绝
- **RawWeightView** 不拥有数据，通过 `shared_ptr<const RawStorage>` 引用底层内存映射
- **预打包去重**：所有 Linear 权重共享相同的 `(OpType::kLinear, MakePackedSelector(...))` 键，因此只执行一次打包操作
- **错误传播**：全链路使用 `StatusOr` + `AM_RETURN_IF_ERROR`，任何步骤失败都会立即返回

---

## 5. 阶段 3：执行计划构建

### 5.1 流程图

```
vector<ExecutionPlanNodeSpec>  ← ⚠️ 当前仅测试手工构造
     │
     ▼
ExecutionPlanBuilder::Build(runtime, model_instance, nodes)
     │
     ├─ PlanWorkspaceRequirements(requirements)
     │   → 全局 workspace 布局规划
     │
     └─ for each node:
         │
         ├─ runtime.GetBackend(device_type) → Backend&
         │
         ├─ CreateAndPrepareOperator(backend, node)
         │   ├─ MakeOperatorParamsForNode(node) → std::any
         │   │   ├─ node.op_params (优先)
         │   │   └─ OperatorRegistry::CreateDefaultParams(op_type) (fallback)
         │   │
         │   ├─ OperatorRegistry::Create(op_type, params) → unique_ptr<Operator>
         │   │   └─ CreateTypedOperator<OpClass>(params)
         │   │       ├─ any_cast<Params>(&params) → 成功 → make_unique<OpClass>
         │   │       └─ any_cast 失败 → Status::InvalidArgument
         │   │
         │   ├─ op->ValidateParams()     → 参数语义校验
         │   ├─ op->CheckInputSpecs()    → 输入 dtype/rank 校验
         │   ├─ op->InferOutputShapes()  → 形状推导 + runtime_checks 生成
         │   └─ op->Prepare(op_ctx)      → kernel resolve + 缓存
         │       └─ backend.ResolveKernelInfo(op_type, selector)
         │           └─ KernelRegistry::Resolve(op_type, selector)
         │               → 无锁查找 buckets_[op_type]
         │               → 最高 priority 的匹配 kernel
         │
         ├─ [if op == nullptr] → FunctionOperator fallback
         │   ├─ ResolveKernelForNode(backend, node)
         │   │   └─ backend.ResolveKernelInfo() → ResolvedKernel
         │   └─ make_shared<FunctionOperator>(resolved)
         │
         ├─ ResolvePackedWeightsForNode(model_instance, node)
         │   └─ model_instance->FindPackedWeights(op_type, selector)
         │
         └─ plan.AddStep(ExecutionStep{...})
```

### 5.2 函数调用序列

| # | 函数 | 参数 | 返回值 | 作用 |
|---|------|------|--------|------|
| 3.1 | `ExecutionPlanBuilder::Build` | `RuntimeContext&`, `vector<ExecutionPlanNodeSpec>&` | `StatusOr<ExecutionPlan>` | 无 ModelInstance 的构建入口 |
| 3.1' | `ExecutionPlanBuilder::Build` | `RuntimeContext&`, `const ModelInstance&`, `vector<ExecutionPlanNodeSpec>&` | `StatusOr<ExecutionPlan>` | 含 ModelInstance 的构建入口 |
| 3.2 | `BuildExecutionPlan` | `RuntimeContext&`, `const ModelInstance*`, `vector<ExecutionPlanNodeSpec>&` | `StatusOr<ExecutionPlan>` | 内部实现 |
| 3.3 | `PlanWorkspaceRequirements` | `span<const WorkspaceRequirement>` | `StatusOr<WorkspaceLayout>` | 全局 workspace 布局规划 |
| 3.4 | `runtime.GetBackend` | `DeviceType` | `StatusOr<Backend*>` | 获取后端实例 |
| 3.5 | `CreateAndPrepareOperator` | `Backend&`, `const ExecutionPlanNodeSpec&` | `StatusOr<PreparedOperator>` | 创建并准备算子 |
| 3.5a | `MakeOperatorParamsForNode` | `const ExecutionPlanNodeSpec&` | `std::any` | 获取算子参数 |
| 3.5b | `OperatorRegistry::Create` | `OpType`, `std::any` | `StatusOr<unique_ptr<Operator>>` | 从注册表创建算子 |
| 3.5c | `op->ValidateParams` | 无 | `Status` | 校验算子参数 |
| 3.5d | `op->CheckInputSpecs` | `span<const TensorSpec>` | `Status` | 校验输入张量规格 |
| 3.5e | `op->InferOutputShapes` | `span<const TensorSpec>` | `StatusOr<InferenceResult>` | 推断输出形状 |
| 3.5f | `op->Prepare` | `const OperatorContext&` | `Status` | resolve kernel 并缓存 |
| 3.6 | `ResolveKernelForNode` | `const Backend&`, `const ExecutionPlanNodeSpec&` | `StatusOr<ResolvedKernel>` | fallback：直接 resolve kernel |
| 3.7 | `FunctionOperator` 构造 | `OpType`, `KernelFunc`, `span<const byte>`, `const char*` | `shared_ptr<FunctionOperator>` | fallback：包装 raw kernel |
| 3.8 | `ResolvePackedWeightsForNode` | `const ModelInstance*`, `const ExecutionPlanNodeSpec&` | `StatusOr<const void*>` | 查找预打包权重 |
| 3.9 | `plan.AddStep` | `ExecutionStep` | `Status` | 添加执行步骤 |

### 5.3 代码示例

```cpp
#include "aethermind/execution/execution_plan_builder.h"

// 手工构造 ExecutionPlanNodeSpec（当前唯一方式）
std::vector<ExecutionPlanNodeSpec> nodes;

// 添加 Embedding 节点
nodes.push_back(ExecutionPlanNodeSpec{
    .op_type = OpType::kEmbedding,
    .device_type = DeviceType::kCPU,
    .activation_dtype = DataType::Float32(),
    .weight_dtype = DataType::Float32(),
    .weight_format = WeightFormat::kPlain,
    .isa = IsaLevel::kAVX2,
    .phase = ExecPhase::kBoth,
    .workspace_requirement = {},
    .input_specs = {TensorSpec{DataType::Float32(), SymbolicShape({-1})}},
    .attrs = {},
    .op_params = std::any(EmbeddingOp::Params{}),
});

// 添加 RmsNorm 节点
nodes.push_back(ExecutionPlanNodeSpec{
    .op_type = OpType::kRmsNorm,
    .device_type = DeviceType::kCPU,
    .activation_dtype = DataType::Float32(),
    .weight_dtype = DataType::Float32(),
    .weight_format = WeightFormat::kPlain,
    .isa = IsaLevel::kAVX2,
    .phase = ExecPhase::kBoth,
    .workspace_requirement = {},
    .input_specs = {
        TensorSpec{DataType::Float32(), SymbolicShape({-1, 4096})},
        TensorSpec{DataType::Float32(), SymbolicShape({4096})},
    },
    .attrs = {},
    .op_params = std::any(RmsNormOp::Params{.eps = 1e-6f}),
});

// 构建执行计划
auto plan = ExecutionPlanBuilder::Build(runtime, *model, nodes);
if (!plan.ok()) {
    // 处理构建错误
    return plan.status();
}
```

### 5.4 双路径分派逻辑

`CreateAndPrepareOperator` 返回 `PreparedOperator{op, inference}`，根据 `op` 是否为空分派到不同路径：

```
CreateAndPrepareOperator(backend, node)
     │
     ├─ OperatorRegistry::Create 成功
     │   → op != nullptr → 走 Operator 路径
     │   → op->ValidateParams → CheckInputSpecs → InferOutputShapes → Prepare
     │   → resolved_kernel_ 缓存在 Operator 内部
     │
     └─ OperatorRegistry::Create 返回 NotFound
         → op == nullptr → 走 FunctionOperator fallback 路径
              ├─ ResolveKernelForNode(backend, node)
              │   └─ backend.ResolveKernelInfo() → ResolvedKernel
              └─ make_shared<FunctionOperator>(resolved)
```

**Operator 路径**（推荐）：通过 `OperatorRegistry` 注册的算子，享受完整的生命周期管理（参数校验、形状推导、约束生成）。

**FunctionOperator 路径**（fallback）：未注册算子的降级路径，直接包装 `KernelFunc` 为 `Operator` 接口，跳过形状推导和约束生成。

### 5.5 关键技术点

- **Plan-Build-Time Resolve**：`op->Prepare()` 在构建期完成 kernel resolve，结果缓存在 `resolved_kernel_` 中，执行期直接调用 `resolved_kernel_.fn(ctx)`
- **Workspace 全局规划**：`PlanWorkspaceRequirements` 为所有步骤统一计算 workspace 偏移，执行期通过 `WorkspaceArena::Bind()` 做零分配切片
- **InferenceResult 传递**：`InferOutputShapes` 返回的 `runtime_checks` 随 `ExecutionStep` 传递到执行期
- **packed_weights 查找**：通过 `(op_type, selector)` 在 `ModelInstance::BackendSidecar` 中查找预打包权重

---

## 6. 阶段 4：执行

### 6.1 流程图

```
Executor::Execute(plan, bindings)
  └─ LayerRunner::Run(plan, bindings)
       └─ for each step:
            └─ RunStep(step_index, step, bindings)
                 │
                 ├─ bindings.BindWorkspace(requirement)
                 │   → WorkspaceBinding{offset, size, arena*}
                 │
                 ├─ BuildKernelContext(step, bindings)
                 │   → KernelContext{device_type, stream, workspace,
                 │                   packed_weights, kernel_params, attrs}
                 │
                 ├─ if (!step.runtime_checks.empty()):
                 │   ├─ bindings.GetStepTensorBinding(step_index)
                 │   │   → StepTensorBinding{inputs, outputs}
                 │   └─ ValidateShapeConstraints(checks, inputs, outputs)
                 │       └─ for each constraint:
                 │           └─ EvaluateShapeConstraint(constraint, inputs, outputs)
                 │               ├─ kSatisfied → continue
                 │               └─ kViolated  → return Status::InvalidArgument
                 │
                 └─ step.op->Run(ctx, bindings, step_index)
                      │
                      ├─ [Operator 路径]
                      │   ├─ ctx.kernel_params = &params_
                      │   └─ resolved_kernel_.fn(ctx)  ← 直接函数指针调用
                      │       └─ CpuRmsNormKernelEntry_FP32_AVX2(ctx)
                      │
                      └─ [FunctionOperator 路径]
                          └─ fn_(ctx)  ← 直接调用 KernelFunc
```

### 6.2 函数调用序列

| # | 函数 | 参数 | 返回值 | 作用 |
|---|------|------|--------|------|
| 4.1 | `Executor::Execute` | `const ExecutionPlan&`, `RuntimeBindingContext&` | `Status` | 执行总入口 |
| 4.2 | `LayerRunner::Run` | 同上 | `Status` | 顺序遍历步骤 |
| 4.3 | `RunStep` | `size_t`, `const ExecutionStep&`, `RuntimeBindingContext&` | `Status` | 单步执行 |
| 4.3a | `bindings.BindWorkspace` | `const WorkspaceRequirement&` | `StatusOr<WorkspaceBinding>` | 绑定 workspace 切片 |
| 4.3b | `BuildKernelContext` | `const ExecutionStep&`, `RuntimeBindingContext&` | `KernelContext` | 构建 kernel 上下文 |
| 4.3c | `ValidateShapeConstraints` | `span<ShapeConstraint>`, `span<TensorView>`, `span<MutableTensorView>` | `Status` | 运行时形状约束校验 |
| 4.3d | `step.op->Run` | `KernelContext&`, `RuntimeBindingContext&`, `size_t` | `Status` | 算子执行 |

### 6.3 代码示例

```cpp
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"

// 准备运行时绑定上下文
WorkspaceArena workspace_arena(runtime.GetAllocator(Device{DeviceType::kCPU, 0}));
RuntimeBindingContext bindings(&workspace_arena);

// 设置每步的张量绑定
for (size_t i = 0; i < plan->size(); ++i) {
    StepTensorBinding tensor_binding;
    tensor_binding.inputs = {input_tensor_view};   // 根据步骤设置
    tensor_binding.outputs = {output_tensor_view};  // 根据步骤设置
    bindings.SetStepTensorBinding(i, std::move(tensor_binding));
}

// 执行推理
auto status = Executor::Execute(*plan, bindings);
if (!status.ok()) {
    // 处理执行错误
}
```

### 6.4 关键技术点

- **热路径零间接调用**：`resolved_kernel_.fn(ctx)` 是直接的函数指针调用，无虚函数分派、无注册表查找
- **Workspace 零分配**：`BindWorkspace` 返回预计算偏移的切片，无堆分配
- **形状校验前置**：`ValidateShapeConstraints` 在 `op->Run()` 前执行，违反时直接返回错误，**不执行 kernel**
- **KernelContext 构造**：栈上构造，无堆分配；`kernel_params` 由 `Operator::Run` 内部设置

---

## 7. 完整函数调用序列总表

### 7.1 已实现函数（按调用顺序）

| 阶段 | # | 函数 | 文件 |
|------|---|------|------|
| 初始化 | 1 | `RuntimeBuilder::WithOptions` | `src/runtime/runtime_builder.cpp` |
| 初始化 | 2 | `RuntimeBuilder::Build` | `src/runtime/runtime_builder.cpp` |
| 初始化 | 2a | `BuildAllocatorRegistry` | `src/runtime/runtime_builder.cpp` |
| 初始化 | 2b | `BuildBackendRegistry` | `src/runtime/runtime_builder.cpp` |
| 初始化 | 2c | `BuildKVCacheManager` | `src/runtime/runtime_builder.cpp` |
| 加载 | 3 | `ModelLoader::Load` | `src/model/model_loader.cpp` |
| 加载 | 4 | `HfDirectoryReader::Open` | `src/model/formats/hf/hf_directory_reader.cpp` |
| 加载 | 5 | `reader.ParseConfig` | `src/model/formats/hf/hf_directory_reader.cpp` |
| 加载 | 6 | `HfModelValidator::ValidateConfig` | `src/model/formats/hf/hf_model_validator.cpp` |
| 加载 | 7 | `reader.LoadRawWeightTable` | `src/model/formats/hf/hf_directory_reader.cpp` |
| 加载 | 8 | `HfModelValidator::ValidateWeightSet` | `src/model/formats/hf/hf_model_validator.cpp` |
| 加载 | 9 | `hf::ResolveWeights` | `src/model/formats/hf/hf_weight_resolver.cpp` |
| 加载 | 10 | `HfModelValidator::ValidateResolvedModel` | `src/model/formats/hf/hf_model_validator.cpp` |
| 加载 | 11 | `ModelInstanceBuilder::Create` | `src/model/model_instance_builder.cpp` |
| 加载 | 12 | `WeightPrepackPlanner::BuildRequests` | `src/model/weight_prepack_planner.cpp` |
| 加载 | 13 | `WeightPrepackPlanner::PrepackAndStore` | `src/model/weight_prepack_planner.cpp` |
| 编译 | 14 | `ExecutionPlanBuilder::Build` | `src/execution/execution_plan_builder.cpp` |
| 编译 | 15 | `PlanWorkspaceRequirements` | `src/runtime/workspace.cpp` |
| 编译 | 16 | `runtime.GetBackend` | `src/runtime/runtime_context.cpp` |
| 编译 | 17 | `CreateAndPrepareOperator` | `src/execution/execution_plan_builder.cpp` |
| 编译 | 17a | `MakeOperatorParamsForNode` | `src/execution/execution_plan_builder.cpp` |
| 编译 | 17b | `OperatorRegistry::Create` | `src/operators/operator_registry.cpp` |
| 编译 | 17c | `op->ValidateParams` | 各算子实现 |
| 编译 | 17d | `op->CheckInputSpecs` | 各算子实现 |
| 编译 | 17e | `op->InferOutputShapes` | 各算子实现 |
| 编译 | 17f | `op->Prepare` | 各算子实现 |
| 编译 | 18 | `backend.ResolveKernelInfo` | `src/backend/cpu/cpu_backend.cpp` |
| 编译 | 19 | `KernelRegistry::Resolve` | `src/backend/kernel_registry.cpp` |
| 编译 | 20 | `ResolvePackedWeightsForNode` | `src/execution/execution_plan_builder.cpp` |
| 编译 | 21 | `plan.AddStep` | `src/execution/execution_plan.cpp` |
| 执行 | 22 | `Executor::Execute` | `src/execution/executor.cpp` |
| 执行 | 23 | `LayerRunner::Run` | `src/execution/layer_runner.cpp` |
| 执行 | 24 | `RunStep` | `src/execution/layer_runner.cpp` |
| 执行 | 25 | `bindings.BindWorkspace` | `src/execution/runtime_binding_context.cpp` |
| 执行 | 26 | `BuildKernelContext` | `src/execution/layer_runner.cpp` |
| 执行 | 27 | `ValidateShapeConstraints` | `src/shape_inference/shape_constraint_evaluator.cpp` |
| 执行 | 28 | `step.op->Run` | 各算子实现 |
| 执行 | 29 | `resolved_kernel_.fn` | 各 kernel 实现 |

### 7.2 未实现函数（设计文档规划）

| # | 函数 | 设计文档位置 | 作用 |
|---|------|-------------|------|
| U1 | `ModelGraphBuilder::BuildLlamaDense(config, weights)` | `model_graph_design.md` §8.1 | 构建 Llama 拓扑图 |
| U2 | `GraphToExecutionPlanLowering::Lower(graph, options, model_instance)` | `model_graph_design.md` §8.3 | 图降低为执行计划节点 |

---

## 8. 核心数据结构变换链

```
filesystem::path (model_dir)
     │
     ▼ [HfDirectoryReader]
HfDirectoryDescriptor {layout, config_path, safetensors_path, ...}
     │
     ▼ [ParseConfig]
HfModelConfig {hidden_size, num_hidden_layers, num_attention_heads, ...}
     │
     ▼ [LoadRawWeightTable]
RawWeightTable {unordered_map<string, RawWeightView>}
     │
     ▼ [hf::ResolveWeights]
ResolvedModelWeights {embed_tokens, final_norm, lm_head?, layers[]}
  每个 layer:
    ├─ norm: {input_rmsnorm, post_attn_rmsnorm}
    ├─ attn: {q_proj, k_proj, v_proj, o_proj}
    └─ mlp: {gate_proj, up_proj, down_proj}
     │
     ▼ [ModelInstanceBuilder::Create]
ModelInstance {config, resolved_weights, backend_sidecar}
     │
     ▼ [WeightPrepackPlanner]
ModelInstance {+ backend_sidecar.packed_weights[]}
     │
     │  ⚠️ 缺口：缺少 ModelGraph → ExecutionPlanNodeSpec 的生产级路径
     │
     ▼ [手工构造 / 未来 ModelGraph lowering]
vector<ExecutionPlanNodeSpec>
  每个 spec:
    ├─ op_type: OpType
    ├─ device_type, activation_dtype, weight_dtype, weight_format, isa, phase
    ├─ workspace_requirement: WorkspaceRequirement
    ├─ input_specs: vector<TensorSpec>
    ├─ attrs: span<const byte>
    └─ op_params: std::any
     │
     ▼ [ExecutionPlanBuilder::Build]
ExecutionPlan {steps[]}
  每个 ExecutionStep:
    ├─ selector: KernelSelector
    ├─ op: OperatorPtr (shared_ptr<const Operator>)
    ├─ packed_weights: const void*
    ├─ workspace_requirement: WorkspaceRequirement
    ├─ output_specs: vector<TensorSpec>
    └─ runtime_checks: vector<ShapeConstraint>
     │
     ▼ [Executor::Execute → LayerRunner::Run]
KernelContext {device_type, stream, workspace, packed_weights, kernel_params, attrs}
     │
     ▼ [Operator::Run → resolved_kernel_.fn(ctx)]
Status (执行结果)
```

---

## 9. 未实现的设计路径（ModelGraph 模块）

### 9.1 设计文档

详见 `docs/designs/model_graph_design.md`。

### 9.2 规划的完整流程

```
ModelInstance
  ├─ HfModelConfig
  ├─ ResolvedModelWeights
  └─ BackendSidecar / PackedWeights
     │
     ▼  [未实现 - M1]
ModelGraphBuilder::BuildLlamaDense(config, weights)
     │
     ▼  [未实现 - M1]
ModelGraph
  ├─ config_: HfModelConfig
  └─ nodes_: vector<ModelGraphNode>
       每个 ModelGraphNode:
         ├─ op_type: OpType
         ├─ layer_index: uint32_t
         ├─ inputs/outputs: vector<ModelTensorSpec>
         ├─ weights: vector<WeightBinding>
         ├─ attrs: ModelGraphAttrs (owned bytes)
         ├─ op_params: std::any
         └─ workspace_requirement: WorkspaceRequirement
     │
     ▼  [未实现 - M2]
GraphToExecutionPlanLowering::Lower(graph, options, model_instance)
     │
     ▼  [未实现 - M2]
vector<ExecutionPlanNodeSpec>
     │
     ▼  [已实现]
ExecutionPlanBuilder::Build(runtime, model_instance, specs)
```

### 9.3 Llama 拓扑展开

```
Embedding
for layer in [0, num_hidden_layers):
    RMSNorm(input_norm)
    Attention / MatMul(QKV) / RoPE / MatMul(O)
    Add(residual)
    RMSNorm(post_attention_norm)
    Linear(gate)
    Linear(up)
    SiluMul
    Linear(down)
    Add(residual)
RMSNorm(final_norm)
Linear(lm_head)
Argmax
```

### 9.4 实施里程碑

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| M1 | 定义 `ModelGraphNode`/`ModelGraph`/`ModelGraphBuilder` + 测试 | ❌ 未实现 |
| M2 | 实现 `GraphToExecutionPlanLowering::Lower` | ❌ 未实现 |
| M3 | 接入 `ModelInstance`，`ModelLoader` 完成后调用 Builder | ❌ 未实现 |
| M4 | 端到端路径 `ModelInstance → ModelGraph → specs → ExecutionPlan` | ❌ 未实现 |

### 9.5 目录结构规划

```
include/aethermind/model/graph/
  model_graph.h
  model_graph_builder.h
  graph_to_execution_plan_lowering.h

src/model/graph/
  model_graph.cpp
  model_graph_builder.cpp
  graph_to_execution_plan_lowering.cpp

tests/unit/model/graph/
  test_model_graph_builder.cpp
  test_graph_to_execution_plan_lowering.cpp
```

---

## 10. 关键技术点

### 10.1 Plan-Build-Time Resolve

所有 kernel 查找在 `ExecutionPlanBuilder::Build` 阶段完成，结果缓存在 `Operator::resolved_kernel_` 中。执行期 `LayerRunner::RunStep` 直接调用 `resolved_kernel_.fn(ctx)`，无注册表查找、无虚函数分派。

**性能影响**：热路径开销仅 ~5ns（一次函数指针间接调用），相比 kernel 执行时间（微秒到毫秒级）可忽略。

### 10.2 Freeze-before-Resolve

`KernelRegistry::Freeze()` 后，`buckets_` 和 `kernels_` 不可变。`Resolve()` 通过 `atomic<bool> frozen_` 的 acquire-release 语义实现无锁读取，消除了多线程推理的全局序列化点。

### 10.3 Workspace 全局规划

`PlanWorkspaceRequirements` 为所有步骤统一计算 workspace 偏移，避免每次 kernel 调用时的动态内存分配。执行期 `WorkspaceArena::Bind()` 返回预计算偏移的切片。

### 10.4 运行时形状校验

`ValidateShapeConstraints` 在 kernel 执行前验证形状约束，违反时直接返回错误，**不执行 kernel**。这防止了形状不匹配时的内存越界访问。

### 10.5 双路径分派

- **Operator 路径**：通过 `OperatorRegistry` 注册的算子，享受完整生命周期管理
- **FunctionOperator 路径**：未注册算子的降级路径，直接包装 `KernelFunc`

### 10.6 权重预打包

`WeightPrepackPlanner` 在模型加载时对所有 Linear 权重执行预打包（如 VNNI 重排），打包结果存入 `ModelInstance::BackendSidecar`。执行期通过 `(op_type, selector)` 查找预打包权重。

---

## 11. 配置说明

### 11.1 RuntimeOptions

```cpp
struct RuntimeOptions {
    struct {
        bool enable_cpu = true;
        bool enable_cuda = false;
        bool enable_cann = false;
    } allocator;

    struct {
        bool enable_cpu = true;
    } backend;

    struct KVCacheRuntimeOptions {
        bool enable_manager = false;
        size_t num_layers = 0;
        size_t num_kv_heads = 0;
        size_t max_tokens = 0;
        size_t head_dim = 0;
        DataType kv_dtype = DataType::Float32();
        size_t alignment = 64;
    } kv_cache;
};
```

### 11.2 ModelLoadOptions

```cpp
struct ModelLoadOptions {
    std::filesystem::path model_dir;
};
```

### 11.3 ExecutionPlanNodeSpec

```cpp
struct ExecutionPlanNodeSpec {
    OpType op_type = OpType::kUnknown;
    DeviceType device_type = DeviceType::kCPU;
    DataType activation_dtype{};
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
    WorkspaceRequirement workspace_requirement{};
    std::vector<TensorSpec> input_specs{};
    std::span<const std::byte> attrs{};
    std::any op_params{};
};
```

### 11.4 GraphLoweringOptions（规划中）

```cpp
struct GraphLoweringOptions {
    DeviceType device_type = DeviceType::kCPU;
    DataType activation_dtype{};
    DataType weight_dtype{};
    WeightFormat preferred_weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
};
```

---

## 12. 测试方法

### 12.1 构建与运行测试

```bash
# 构建单元测试
cmake --build build --target aethermind_unit_tests -j

# 运行模型加载相关测试
./build/tests/unit/aethermind_unit_tests --gtest_filter="ModelLoader.*"

# 运行执行计划构建相关测试
./build/tests/unit/aethermind_unit_tests --gtest_filter="ExecutionPlanBuilder.*"

# 运行执行器相关测试
./build/tests/unit/aethermind_unit_tests --gtest_filter="Executor.*:LayerRunner.*"

# 运行算子相关测试
./build/tests/unit/aethermind_unit_tests --gtest_filter="RmsNormOp.*:EmbeddingOp.*"

# 运行 KernelRegistry 相关测试
./build/tests/unit/aethermind_unit_tests --gtest_filter="KernelRegistry.*"

# 运行端到端测试
./build/tests/unit/aethermind_unit_tests --gtest_filter="*BackendPath*"
```

### 12.2 关键测试用例

| 测试文件 | 测试用例 | 验证内容 |
|---------|---------|---------|
| `test_model_loader.cpp` | `LoadValidModel` | 端到端模型加载 |
| `test_execution_plan_builder.cpp` | `BuildsWithRegisteredOperator` | Operator 路径构建 |
| `test_execution_plan_builder.cpp` | `BuildsWithFunctionOperator` | FunctionOperator fallback |
| `test_execution_plan_builder.cpp` | `PropagatesRuntimeChecks` | runtime_checks 传递 |
| `test_executor_backend_path.cpp` | `ExecuteRunsWhenRuntimeShapeConstraintIsSatisfied` | 约束满足时正常执行 |
| `test_executor_backend_path.cpp` | `ExecuteRejectsViolatedRuntimeShapeConstraintBeforeRun` | 约束违反时拒绝执行 |
| `test_kernel_registry.cpp` | `ConcurrentFreezeIsIdempotent` | 并发冻结幂等性 |

### 12.3 TSAN 变体测试

```bash
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j
./build-tsan/tests/unit/aethermind_unit_tests --gtest_filter="KernelRegistry.*:OperatorRegistry.*"
```

---

## 13. 注意事项

### 13.1 当前限制

1. **缺少 ModelGraph 模块**：`ExecutionPlanNodeSpec` 只能手工构造，没有从已加载模型自动生成的生产级路径
2. **仅支持 CPU 后端**：CUDA/CANN 后端尚未实现
3. **仅支持 Llama 家族**：`hf::ResolveWeights` 硬编码了 Llama 权重名映射
4. **仅支持 float32**：量化精度（int8/int4）尚未实现
5. **仅支持单请求推理**：batch 推理、continuous batching 尚未实现

### 13.2 生命周期注意事项

1. **`RuntimeContext` 必须在 `ModelLoader::Load` 之前构建**：因为 `Load` 需要 `Backend&` 和 `KernelRegistry&`
2. **`KernelRegistry::Global()` 必须在 `CpuBackend` 构造前完成所有注册**：`AM_REGISTER_KERNEL` 在静态初始化阶段执行，`Freeze()` 在 `CpuBackend` 构造时调用
3. **`ExecutionPlan` 构建后不可变**：`AddStep()` 为 private，仅 `ExecutionPlanBuilder` 可调用
4. **`ResolvedKernel::attrs` 拥有数据**：`std::vector<std::byte>` 拷贝语义，生命周期与 `Operator` 绑定
5. **`packed_weights` 指针生命周期与 `ModelInstance` 绑定**：`ExecutionStep::packed_weights` 指向 `BackendSidecar` 内部数据

### 13.3 线程安全注意事项

1. **`KernelRegistry::Resolve()` 冻结后无锁**：多线程并发安全，但冻结前不可调用
2. **`OperatorRegistry::Create()` 持锁**：锁粒度极小（先拷贝 factory 再解锁调用）
3. **`ExecutionPlan` 构建后只读**：多线程可安全共享
4. **`Operator::Run()` 非线程安全**：同一 `Operator` 实例不可在多线程中并发调用

### 13.4 错误处理注意事项

1. **全链路 `Status`/`StatusOr`**：无异常作为常规错误路径
2. **`AM_RETURN_IF_ERROR`**：错误立即传播，不吞没
3. **`AM_CHECK`**：程序错误（如 null pointer）直接 abort，不返回 Status
4. **`ValidateShapeConstraints`**：运行时约束违反返回 `Status::InvalidArgument`，不执行 kernel
