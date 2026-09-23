# Weight 数据概念与生命周期

- **状态**: Current（建议性词表；只写仓库事实）
- **版本**: 1.2
- **日期**: 2026-09-23
- **关联代码**: 见 §2 各阶段"权威文件"列
- **上游依赖**: [AGENTS.md](../../../AGENTS.md) 模块边界、[01-model-loader.md](01-model-loader.md)（加载期 resolve）
- **关联文档**: [cpu-gemm-packed-weight.md](../../operators/gemm/cpu-gemm-packed-weight.md)（打包现状与 GEMM 演进）、[architecture_overview.md](../architecture/architecture_overview.md)

## 1. 背景与目标

weight 相关概念横跨 graph / model / compiler / backend / execution / inference 六个模块（packing、binding、resolver、store、prepack 等多组近似命名并存），历史上出现过 model 层直接实例化 `CpuWeightPrepacker` 的越界依赖。本文是**唯一权威词表**：

1. 六阶段定位每个 weight 概念的职责、层次与权威文件；
2. 固定命名动词约定，新符号必须落入表中某一阶段；
3. 冻结依赖红线，防止概念再跨层漂移。

## 2. 六阶段词表

数据流按生命周期分为六段，依次为：**resolve → bind → request → pack → store → specialize**。

| 阶段（动词） | 概念 | 权威文件 | 所属层 |
|---|---|---|---|
| **resolve**（按张量名） | `hf::ResolveWeights`：HF 张量名 → 逻辑权重树（含 tied embedding 别名）；`ResolvedModelWeights`；`RawWeightView`/`RawStorage`（借用视图 + 共享 backing） | [hf_tensor_resolver.h](../../../include/aethermind/model/formats/hf/hf_tensor_resolver.h)、[resolved_model_weights.h](../../../include/aethermind/model/resolved_model_weights.h)、[raw_weight.h](../../../include/aethermind/model/raw_weight.h) | model |
| **bind**（按结构化角色） | `WeightBinding`/`WeightBindingSpec`（direct/qkv/gate-up，纯数据类型）；`ResolveWeightBinding`：结构化 binding → `RawWeightView`（tied lm-head 回退的单一权威） | [graph_types.h](../../../include/aethermind/graph/graph_types.h)、[weight_packing.h](../../../include/aethermind/model/weight/weight_packing.h) | graph（类型）/ model（映射） |
| **request**（图 → 请求） | `WeightPackingRequest`（类型，含 `recipe` 字段）；`BuildWeightPackingRequests`：artifact 的 kWeight 值 → 请求（纯数据映射，不触 backend，唯一生产来源）；recipe 由编排层在 prepare 期经 `Backend::GetPackingRecipe`（复用 descriptor eligibility）注入 | [weight_packing.h](../../../include/aethermind/model/weight/weight_packing.h)、[packing_request_builder.h](../../../include/aethermind/compiler/packing_request_builder.h) | model（类型）/ compiler（生产）/ inference（注入） |
| **pack**（执行） | `KernelDescriptor::packing_recipe`（per-op layout 声明）；`Backend::PackWeights`/`GetPackingRecipe`（抽象契约，默认 Unimplemented）；`PrepackWeightRequests`（要求显式 recipe，校验产物 recipe 一致）；`PackingRecipe`/`PackedWeights`（backend 纯数据契约）；`CpuWeightPrepacker`（按 recipe 分派：identity 与 `cpu_bpanel_f32_v1_avx2`；常量分别在 `cpu_weight_prepacker.h` 与 `cpu_bpanel_packing.h`）；`packed_weight_utils.h`（消费侧闸口：identity + bpanel） | [backend.h](../../../include/aethermind/backend/backend.h)、[kernel_descriptor.h](../../../include/aethermind/backend/kernel_descriptor.h)、[weight_packing.cpp](../../../src/model/weight/weight_packing.cpp)、[packed_weights.h](../../../include/aethermind/backend/packed_weights.h)、[cpu_weight_prepacker.h](../../../include/aethermind/backend/cpu/cpu_weight_prepacker.h)、[cpu_bpanel_packing.h](../../../include/aethermind/backend/cpu/cpu_bpanel_packing.h) | backend（契约+实现）/ model（编排） |
| **store**（归位） | `PackedWeightStore`（`Store`/`Find` 按完整 `WeightArtifactKey` 幂等存储与精确查找）；`WeightArtifactKey{source_id, value_index, binding, selector, recipe}` | [weight_packing.h](../../../include/aethermind/model/weight/weight_packing.h) | model |
| **specialize**（执行期） | `ExternalTensorBindings`/`ExternalReadOnlyValueBinding`/`ExternalWritableValueBinding`（外部绑定契约）；`PrepareExecutionBindings`/`PreparedExecutionBindings`/`ComputeExternalReadRequirements`（plan 冷路径特化）；`WeightBindingStorage`（immutable 绑定 shape/stride 元数据所有权） | [execution_bindings.h](../../../include/aethermind/execution/execution_bindings.h)、[weight_binding_storage.h](../../../include/aethermind/inference/weight_binding_storage.h) | execution / inference |

## 3. 命名约定

1. 新增 weight 相关公开符号，必须能归入 §2 某一阶段；归不进的先在本文档补阶段再写代码。
2. 动词前缀固定五个：`Resolve*`（按名/按角色）、`Binding`（结构化身份）、`PackingRequest`/`Build*Requests`（请求）、`Pack*`/`Prepack*`（执行）、`*Bindings`（执行期特化）。同义近名必须与 §2 对齐。
3. 两个 resolver 的区分：`hf::ResolveWeights` 按**张量名**（加载期，formats/hf）；`ResolveWeightBinding` 按**结构化角色**（准备期，model/weight）。新增 resolver 时必须注明驱动方式。

## 4. 依赖红线（与 AGENTS.md 一致，禁止漂移）

- model 权重组装（`WeightPackingRequest`/`PrepackWeightRequests`/`PackedWeightStore`）可依赖 backend 纯数据契约（`PackedWeights`/`PackingRecipe`），**打包执行必须经 `Backend::PackWeights` 抽象**，不得 include backend/cpu 具体实现（`CpuWeightPrepacker` 等）。
- **layout 权威在 backend**：component 校验、composite 物化、对齐与分配全部由 backend 落实；model 层只做视图字节数防御校验与编排。
- 请求构建（`BuildWeightPackingRequests`）归 compiler（读 `LoweredGraph`）；请求执行（`PrepackWeightRequests`）归 model/weight；二者通过 `WeightPackingRequest` 类型衔接。
- 执行期（execution）只消费 `PackedWeightStore`/`WeightArtifactKey`，不参与打包决策。

## 5. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-23 | 1.0 | 初版：六阶段词表、命名动词约定、依赖红线（Backend::PackWeights 抽象落地后冻结） |
| 2026-09-23 | 1.1 | 文件合并后同步：`weight_binding_resolver`/`packed_weight_store` 并入 `weight_packing.h/.cpp`（model/weight 三件套合一）；`identity_packing.h` 常量并入 `cpu_weight_prepacker.h`；`external_bindings.h` 回退并入 `execution_bindings.h` |
| 2026-09-23 | 1.2 | recipe 链闭环同步：request 阶段补 `recipe` 字段与编排层注入（`Backend::GetPackingRecipe`）；pack 阶段补 `KernelDescriptor::packing_recipe` 声明、packer 按 recipe 分派 identity/`cpu_bpanel_f32_v1_avx2`（常量头 `cpu_bpanel_packing.h`）与 identity/bpanel 双消费闸口 |