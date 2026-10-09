# Weight 数据概念与生命周期

- **状态**: Current（建议性词表；只写仓库事实）
- **版本**: 1.5
- **日期**: 2026-10-09
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

`WeightPackingRequest` 统一以非空 `components` 承载 raw views：直接绑定为一个组件，QKV 为 Q/K/V 三个组件，Gate-Up 为 Gate/Up 两个组件。组件顺序由逻辑 binding 确定，各 view 持有共享 backing；compiler 生成组件列表，model/weight 在打包前拒绝空列表并验证各组件，backend 负责 composite 物化与物理 layout。空请求批仍返回未绑定的空 collection。

`Backend::PackWeights(op, components, selector, recipe)` 是唯一打包入口；packing 为可选能力，默认返回 `Unimplemented`。`GetPackingRecipe` 保留为独立查询接口，准备层先解析 recipe 并检查 consumer 兼容性，再显式传入打包函数。具体 backend 实现四参数入口，产物 recipe 的一致性继续由 prepack/collection 校验。

| 阶段（动词） | 概念 | 权威文件 | 所属层 |
|---|---|---|---|
| **resolve**（按张量名） | `hf::ResolveWeights`：HF 张量名 → 逻辑权重树（含 tied embedding 别名）；`ResolvedModelWeights`；`RawWeightView`/`RawStorage`（借用视图 + 共享 backing） | [hf_tensor_resolver.h](../../../include/aethermind/model/formats/hf/hf_tensor_resolver.h)、[resolved_model_weights.h](../../../include/aethermind/model/resolved_model_weights.h)、[raw_weight.h](../../../include/aethermind/model/raw_weight.h) | model |
| **bind**（按结构化角色） | `WeightBinding`/`WeightBindingSpec`（direct/qkv/gate-up，纯数据类型）；`ResolveWeightBinding`：结构化 binding → `RawWeightView`（tied lm-head 回退的单一权威） | [graph_types.h](../../../include/aethermind/graph/graph_types.h)、[weight_binding_resolver.h](../../../include/aethermind/model/weight/weight_binding_resolver.h) | graph（类型）/ model（映射） |
| **request**（图 → 请求） | `WeightPackingRequest`（轻量请求数据，含待准备期填充的 `recipe`）；`BuildWeightPackingRequests`：artifact 的 kWeight 值 → 请求，不触 backend；`ResolveWeightPackingRequests`：经 `Backend::GetPackingRecipe` 注入 recipe 并合并兼容请求 | [weight_packing_request.h](../../../include/aethermind/model/weight/weight_packing_request.h)、[weight_packing_request_builder.h](../../../include/aethermind/compiler/weight_packing_request_builder.h)、[inference_internal.h](../../../src/inference/inference_internal.h) | model（类型）/ compiler（生产）/ inference（准备期解析） |
| **pack**（执行） | `PackingLayout`/`PackingRecipe`/`PackedWeight`（版本化枚举身份/recipe/产物契约）；`KernelDef::packing_recipe`、`Backend::PackWeights`/`GetPackingRecipe`；`PrepackWeightRequests` 执行批量请求并校验显式 recipe；`CpuWeightPrepacker` 实施 identity/B-panel 转换；两种布局契约独立，消费侧共用 validation | [packing_recipe.h](../../../include/aethermind/backend/packing_recipe.h)、[packed_weight.h](../../../include/aethermind/backend/packed_weight.h)、[packed_weight_collection.h](../../../include/aethermind/model/weight/packed_weight_collection.h)、[cpu_weight_prepacker.h](../../../include/aethermind/backend/cpu/cpu_weight_prepacker.h)、[cpu_identity_packing.h](../../../include/aethermind/backend/cpu/cpu_identity_packing.h)、[cpu_bpanel_packing.h](../../../include/aethermind/backend/cpu/cpu_bpanel_packing.h)、[packed_weight_validation.h](../../../include/aethermind/backend/cpu/packed_weight_validation.h) | backend（契约+实现）/ model（批量执行） |
| **store**（归位） | `WeightArtifactKey{source_id, value_index, binding, selector, recipe}` 与 `PackedWeightCollection`；`Insert` 校验 key/产物一致性并共享所有权，重复 key 返回 AlreadyExists；`Find` 按完整 key 精确查找 | [packed_weight_collection.h](../../../include/aethermind/model/weight/packed_weight_collection.h) | model |
| **specialize**（执行期） | `ExternalTensorBindings`/`ExternalReadOnlyValueBinding`/`ExternalWritableValueBinding`（外部绑定契约）；`PrepareExecutionBindings`/`PreparedExecutionBindings`/`ComputeExternalReadRequirements`（plan 冷路径特化）；`WeightBindingStorage`（immutable 绑定 shape/stride 元数据所有权） | [execution_bindings.h](../../../include/aethermind/execution/execution_bindings.h)、[weight_binding_storage.h](../../../include/aethermind/inference/weight_binding_storage.h) | execution / inference |

## 3. 命名约定

| 名称组 | 含义 | 例子 |
|---|---|---|
| `packed_weight_*` / `PackedWeight*` | 已生成的打包产物及其视图、存储和校验 | `PackedWeight`、`PackedWeightView`、`PackedWeightCollection`、`packed_weight_validation.h` |
| `weight_packing_*` / `WeightPacking*` | 生成打包产物的过程、请求及其构建/解析 | `WeightPackingRequest`、`weight_packing_request_builder.h`、`BuildWeightPackingRequests`、`ResolveWeightPackingRequests` |

`Prepack` / `Prepacker` 表示在模型准备期、执行之前完成打包的时机，属于打包过程这一组；具体布局由 recipe 指定。文件名保留完整的 weight-packing 词组，避免与其他 packing 请求混淆。

1. 新增 weight 相关公开符号，必须能归入 §2 某一阶段；归不进的先在本文档补阶段再写代码。
2. 动词前缀固定五个：`Resolve*`（按名/按角色）、`Binding`（结构化身份）、`PackingRequest`/`Build*Requests`（请求）、`Pack*`/`Prepack*`（执行）、`*Bindings`（执行期特化）。同义近名必须与 §2 对齐。
3. 两个 resolver 的区分：`hf::ResolveWeights` 按**张量名**（加载期，formats/hf）；`ResolveWeightBinding` 按**结构化角色**（准备期，model/weight）。新增 resolver 时必须注明驱动方式。

## 4. 依赖红线（与 AGENTS.md 一致，禁止漂移）

布局身份统一使用 backend 纯数据契约 `PackingLayout`：`kNone` 表示未指定，当前生产布局是 `kCpuIdentity` 与 `kCpuBPanelF32V1Avx2`。`PackingRecipe::layout`、`PackedWeightView::recipe_layout` 与 artifact key 都携带同一枚举；`ToString(PackingLayout)` 只提供静态诊断名称。修改 tile、panel 顺序或 padding 时新增布局 ID，保留原有格式身份；alignment 字段和物理布局公式仍由 backend 契约维护。未知枚举值在注册、批量构建、入库和消费校验处拒绝；低层 Collection 可保留未指定 recipe 的占位产物，生产预打包要求显式有效布局。

- model 权重组装（`WeightPackingRequest`/`PrepackWeightRequests`/`PackedWeightCollection`）可依赖 backend 纯数据契约（`PackedWeight`/`PackingRecipe`），**打包执行必须经 `Backend::PackWeights` 抽象**，不得 include backend/cpu 具体实现（`CpuWeightPrepacker` 等）。
- **layout 权威在 backend**：component 校验、composite 物化、对齐与分配全部由 backend 落实；model 层只做视图字节数防御校验与编排。
- 请求构建（`BuildWeightPackingRequests`）归 compiler（读 `LoweredGraph`）；请求执行（`PrepackWeightRequests`）归 model/weight；二者通过 `WeightPackingRequest` 类型衔接。
- 执行期（execution）只消费 `PackedWeightCollection`/`WeightArtifactKey`，不参与打包决策。

文件依赖按职责收敛：图构建仅需 binding resolver；compiler 请求构建仅需 request 数据；execution 计划组装仅需 collection；inference 的公开模型头仅需 collection，准备实现再包含 resolver 与 request 数据；批量构建入口 `PrepackWeightRequests` 与 `PackedWeightCollection` 同单元，其头文件仅前向声明 Backend 与 WeightPackingRequest。kernel 消费布局契约和 validation，不包含 prepacker 服务头。`PackingRecipe` 单独成头，使用 recipe 的 descriptor 不再引入 `PackedWeight`/`Buffer`。

## 5. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-23 | 1.0 | 初版：六阶段词表、命名动词约定、依赖红线（Backend::PackWeights 抽象落地后冻结） |
| 2026-09-23 | 1.1 | 文件合并后同步：`weight_binding_resolver`/`packed_weight_store` 并入 `weight_packing.h/.cpp`（model/weight 三件套合一）；`identity_packing.h` 常量并入 `cpu_weight_prepacker.h`；`external_bindings.h` 回退并入 `execution_bindings.h` |
| 2026-09-23 | 1.2 | recipe 链闭环同步：request 阶段补 `recipe` 字段与编排层注入（`Backend::GetPackingRecipe`）；pack 阶段补 `KernelDef::packing_recipe` 声明、packer 按 recipe 分派 identity/`cpu_bpanel_f32_v1_avx2`（常量头 `cpu_bpanel_packing.h`）与 identity/bpanel 双消费闸口 |
| 2026-10-08 | 1.3 | 按依赖拆分 model resolver/request，Store 与批量构建入口保留在同一单元；独立 PackingRecipe 与 CPU identity 契约，B-panel 声明/实现对应；请求 recipe 解析保留在 executable_model.cpp，私有声明集中到 inference_internal.h，消费侧文件统一为 packed_weight_validation；迁移现有测试并保留布局与所有权合同 |
| 2026-10-08 | 1.4 | `PackedWeightStore` 更名为 `PackedWeightCollection`，文件名同步为 `packed_weight_collection.h/.cpp`，插入接口 `Store` 更名为 `Insert`；完整 key、共享所有权与 source 绑定合同保持不变 |
| 2026-10-09 | 1.5 | layout 身份由借用字符串改为 PackingLayout 枚举，recipe/view/key 一致传递；新增未知枚举值校验，诊断名称集中到 ToString；identity/B-panel 字节格式与 alignment 合同保持 |
