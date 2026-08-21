# ModelLoader 模块设计

- **状态**: Current（描述已验证实现；只写仓库事实）
- **版本**: 1.0
- **日期**: 2026-08-21
- **关联代码**: [include/aethermind/model/model_loader.h](../../../include/aethermind/model/model_loader.h) / [src/model/model_loader.cpp](../../../src/model/model_loader.cpp)，及 [include/aethermind/model/formats/hf/](../../../include/aethermind/model/formats/hf/) 下 HF I/O 组件
- **上游依赖**: `base`（Status/StatusOr）、`formats/hf`（HfDirectoryReader/HfModelValidator/HfWeightResolver）、`ResolvedModelWeights`/`RawWeightView`
- **下游消费者**: `ModelCompiler::Compile` / `LoadAndCompile`（compiler 模块）、`ModelGraphBuilder::BuildLlamaDense`（model 模块内）
- **关联测试**: [tests/unit/model/test_model_loader.cpp](../../../tests/unit/model/test_model_loader.cpp) 及 `test_hf_*.cpp` 系列
- **架构总览**: [architecture_overview.md](../architecture/architecture_overview.md) 第三章（模型加载数据流）与第七章

## 1. 背景与目标

ModelLoader 是模型加载链路的前端唯一入口：把 HuggingFace 模型目录（`config.json` + `*.safetensors`）转换为 backend-independent 的 `LoadedModel`（config + resolved logical raw weights）。

目标：

- 职责收敛：只做 HF I/O、校验与逻辑权重 resolve；不构建图、不 resolve kernel、不 prepack（`AGENTS.md` model 模块边界）。
- 失败可诊断：所有失败通过 `StatusOr` 上报，不在加载期静默修复。
- 确定性：对同一目录的重复 Load 结果一致；权重视图仅持有 backing storage 的借用视图（`RawWeightView`），不复制权重数据。

## 2. 职责与边界

- **提供**：`ModelLoader::Load(model_dir)` → `unique_ptr<LoadedModel>`。
- **请求**：`HfDirectoryReader`（目录 I/O）、`HfModelValidator`（三阶段校验）、`hf::ResolveWeights`（tensor 名 → 逻辑视图，命名空间自由函数）。
- **所有权**：`LoadedModel` 按值持有 `HfModelConfig` 与 `ResolvedModelWeights`；`ResolvedModelWeights` 的共享 backing storage 为 `RawWeightView` 提供底层数据，生命周期随 `LoadedModel` 存活。
- **明确不做**：图构建（归 `ModelGraphBuilder`）、kernel 解析（归 execution/backend）、权重预打包（归 `WeightPrepackPlanner`/`PackedWeightStore`）、RoPE scaling type 的语义映射与拒绝（归 `ModelGraphBuilder::BuildLlamaDense` 独占）。
- **生命周期**：`LoadedModel` 构造后只读；由 `LoweredModelArtifact` 按值持有其 `unique_ptr`（见架构总览 §5 所有权表）。

## 3. 关键数据结构

| 成员 | 含义 | 同步机制 / 备注 |
|---|---|---|
| `HfModelConfig` | `config.json` 解析产物（模型类型、隐藏维度、层数、RoPE 配置等） | 构造后只读，无同步 |
| `HfRopeConfig` / `HfRopeScalingType` | RoPE 配置与 HF scaling 类型枚举（uint8_t 底层，保持结构紧凑） | 构造后只读 |
| `HfDirectoryDescriptor` | 目录布局描述（单文件/分片、safetensors 路径） | 构造后只读 |
| `ResolvedModelWeights` | 逻辑权重视图集合（`DecoderLayerRawWeights[]`：attn/mlp/norm） | 构造后只读 |
| `RawWeightView` / `RawStorage` | 权重借用视图与共享 backing storage | 视图不拥有数据；`IsValid()`/`IsAligned()` 校验辅助 |
| `ModelValidationOptions` | 校验策略开关（strict_tensor_names、allow_bias 等） | 默认值接受常见 HF 导出怪癖、拒绝不支持特性 |

## 4. 并发模型

- **单线程**：加载流程为调用线程内的顺序流水线；`ModelLoader` 是无状态 facade（`static` 方法），无共享可变状态。
- **线程安全**：`LoadedModel` 构造后只读，可安全跨线程共享（由调用方负责同步语义）。
- 无锁、无原子、无 TLS 依赖。

## 5. 接口定义

| 接口 | 签名 | 语义要点 | Hot path |
|---|---|---|---|
| `ModelLoader::Load` | `static StatusOr<std::unique_ptr<LoadedModel>> Load(const std::filesystem::path& model_dir)` | `@param model_dir` 含 config.json 与 safetensors 的目录；`@return` 持有 config 与 resolved weights 的 LoadedModel；所有失败经 StatusOr 上报 | ❌（一次性加载） |
| `HfDirectoryReader::Open` | `static StatusOr<HfDirectoryReader> Open(const std::filesystem::path& dir)` | 打开并校验目录结构 | ❌ |
| `HfDirectoryReader::InspectDirectory` | `static StatusOr<HfDirectoryDescriptor> InspectDirectory(const std::filesystem::path& model_dir)` | 仅检测布局，不加载配置与权重 | ❌ |
| `HfDirectoryReader::ParseConfig` | `StatusOr<HfModelConfig> ParseConfig() const` | 解析 config.json 到 AetherMind 使用的子集 | ❌ |
| `HfDirectoryReader::LoadRawWeightTable` | `StatusOr<RawWeightTable> LoadRawWeightTable() const` | 从已检测的布局加载原始 tensor 视图 | ❌ |
| `HfModelValidator::ValidateConfig` | `static Status ValidateConfig(const HfModelConfig&, const ModelValidationOptions&)` | 校验 config 语义字段是否符合 Phase 1 范围；不支持或不一致返回 InvalidArgument | ❌ |
| `HfModelValidator::ValidateWeightSet` | `static Status ValidateWeightSet(const HfModelConfig&, const RawWeightTable&, const ModelValidationOptions&)` | 校验 raw weight table 与模型 schema 的匹配 | ❌ |
| `HfModelValidator::ValidateResolvedModel` | `static Status ValidateResolvedModel(const HfModelConfig&, ...)` | 校验 resolved weight 完整性（tied embeddings、dtype） | ❌ |
| `hf::ResolveWeights` | `StatusOr<ResolvedModelWeights> ResolveWeights(const HfModelConfig&, const RawWeightTable&)` | HF tensor 名（含 tied embedding 别名）→ 逻辑模型权重视图 | ❌ |

## 6. 算法与流程

### 6.1 Load 流水线

```text
ModelLoader::Load(model_dir)
  ├─ HfDirectoryReader::Open          → 目录存在性与结构校验
  ├─ reader.Inspect()                 → 布局检测（单文件/分片）
  ├─ reader.ParseConfig()             → HfModelConfig
  ├─ reader.LoadRawWeightTable()      → RawWeightTable（raw tensor 视图）
  ├─ HfModelValidator 三阶段校验      → config 语义 / weight set schema / resolved 完整性
  └─ hf::ResolveWeights               → ResolvedModelWeights
       └─ make_unique<LoadedModel>(config, resolved_weights)
```

每步失败即返回对应 `Status`（`InvalidArgument` / 文件 I/O 错误等），不继续流水线。

### 6.2 复杂度

加载为一次性路径：O(目录文件数 + tensor 数量)，无隐藏 O(N²)（tensor 名解析为单遍哈希映射）。

## 7. 边界条件与错误处理

- **目录不存在/无 config.json**：`Open`/`Inspect` 失败。
- **非法 config 语义**（非 Llama-family、配置不一致）：`ValidateConfig` 返回 `InvalidArgument`。
- **schema 不匹配**（缺 tensor、多余 tensor、dtype 不一致）：`ValidateWeightSet` / `ValidateResolved` 按 `ModelValidationOptions` 策略判定（默认允许 extra tensors 与 tied lm_head 缺失，拒绝 bias/量化/适配器）。
- **RoPE scaling**：结构合法性由 `ValidateConfig`（`allow_rope_scaling` 默认 true）校验；none/linear 到 `RoPEScalingType` 的映射与 unsupported type 拒绝由 `ModelGraphBuilder::BuildLlamaDense` 独占（`LoadedModel` 不决策）。
- **所有权**：所有返回的 `StatusOr` 失败值不含部分构造产物，无资源泄漏路径（RAII）。

## 8. 风险与权衡

- **视图借用而非复制**：`RawWeightView` 不复制权重，加载期内存峰值低；代价是调用方必须保证 `LoadedModel` 在编译期存活（由 `LoweredModelArtifact` 持有保证）。
- **校验默认宽松**：默认选项接受 HF 导出怪癖（extra tensors、缺失 tied lm_head），换取对常见模型目录的兼容；严格性可通过 `ModelValidationOptions` 收紧。
- **无 graph 感知**：`LoadedModel` 不含任何 backend artifact，符合单一语义权威原则（`ModelGraphBuilder` 是 HF → 语义图唯一转换权威）；代价是加载与编译边界需要显式衔接（`ModelCompiler`）。

## 9. 测试要点

`tests/unit/model/`：

- `test_model_loader.cpp`：`LoadValidModel` 等端到端加载用例。
- `test_hf_config_parser.cpp` / `test_hf_json_reader.cpp`：config 解析边界。
- `test_hf_directory_reader.cpp`：目录布局（单文件/分片）。
- `test_hf_model_validator.cpp`：三阶段校验的正反用例（含策略开关组合）。
- `test_hf_weight_resolver.cpp`：tensor 名解析与 tied embeddings。
- `test_hf_safetensors_file.cpp` / `test_hf_safetensors_index.cpp`：safetensors 读取。
- `test_hf_real_model_integration.cpp`：真实模型目录端到端集成。

## 10. 变更记录

| 日期 | 变更 | 原因 | 关联 PR / ADR |
|---|---|---|---|
| 2026-08-21 | 初版（基于当前头文件与实现新写；早期 model_loader/ 设计已归档） | 文档系统落地 | [ADR-0001](../../decisions/0001-documentation-system.md) |
