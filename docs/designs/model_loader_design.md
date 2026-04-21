# ModelLoader 与权重预打包设计文档（Phase 1 收敛版）

## 1. 文档目标

本文档定义 AetherMind Phase 1 中 **模型加载（ModelLoader）** 与 **权重预打包（Weight Prepack）** 的收敛设计。

本文档的目标不是重新发明一套与现有代码并列的模型运行时对象，而是把模型加载主链明确收敛到当前仓库已经落下来的主线上：

- `ModelInstance`
- `BackendSidecar`
- `PackedWeights`
- `CpuWeightPrepacker`
- `ExecutionPlanBuilder`

本文档遵循当前已冻结的 Phase 1 约束：

- CPU-first
- 单进程
- 单模型
- 单请求
- 同步执行
- Llama-family dense decoder-only
- 静态 KV cache
- greedy decoding
- steady-state zero allocation
- token ids in / token ids out

---

## 2. 设计结论

Phase 1 的模型加载模块负责把外部模型工件一次性转换为**运行期可直接消费的 `ModelInstance`**。

加载完成后，运行期不再做以下工作：

- 不再做权重转置
- 不再做 layout 重排
- 不再做首次 lazy prepack
- 不再做热路径 kernel 选择
- 不再为权重相关对象进行动态分配

也就是说，ModelLoader 的本质是：

> 将原始模型工件一次性构造成只读、已验证、已预打包、与当前执行主线兼容的 `ModelInstance`。

---

## 3. Phase 1 支持范围

### 3.1 支持范围

Phase 1 当前设计覆盖以下范围：

- 单模型加载
- CPU backend
- Llama-family dense decoder-only 模型
- 本地 Hugging Face 风格模型目录
- `config.json + safetensors` 输入工件
- 线性层权重预打包
- embedding / norm / 少量常量按 raw 只读方式保留
- 同步阻塞式加载
- 运行期只读访问

### 3.2 非目标

Phase 1 中本模块不负责：

- tokenizer
- KV cache 分配与生命周期
- 执行期 workspace 管理
- 算子图构建
- 调度与批处理
- 多模型热切换
- 多设备模型分片
- 在线下载模型
- `trust_remote_code`
- `.bin` / pickle 权重格式
- adapter 合并
- 训练权重写回
- 运行期 lazy prepack
- 自有 packed artifact 格式

---

## 4. Hugging Face 本地目录支持策略

### 4.1 设计决策

AetherMind Phase 1 **正式支持 Hugging Face 风格本地模型目录**，但支持范围必须严格收敛。

这里的“支持”指的是：

> 支持把 Hugging Face 常见本地 checkpoint 目录作为 **模型工件输入格式**，而不是复制 Transformers `from_pretrained()` 的全语义。

内部主链保持为：

```text
ArtifactReader -> Validator -> Resolver -> Prepack -> ModelInstance
```

而不是把 Transformers 的设备分发、remote code、tokenizer、adapter 或训练生态引入引擎核心。

---

### 4.2 Phase 1 支持的目录形态

#### 形态 A：单文件权重

```text
<model_dir>/
  config.json
  model.safetensors
```

#### 形态 B：分片权重

```text
<model_dir>/
  config.json
  model.safetensors.index.json
  model-00001-of-0000N.safetensors
  ...
  model-0000N-of-0000N.safetensors
```

---

### 4.3 不支持的 Hugging Face 扩展语义

以下内容不属于 Phase 1 的 ModelLoader 支持范围：

- Hugging Face Hub 在线下载
- `trust_remote_code`
- `AutoModel` 全自动架构适配
- tokenizer 文件加载与文本前处理
- `.bin` / pickle 权重格式
- PEFT / LoRA adapter 合并
- bitsandbytes / GPTQ / AWQ 等生态格式
- `device_map="auto"`
- Accelerate 分布式大模型加载语义

Phase 1 的原则是：

> 只支持“本地静态 checkpoint + safetensors + config.json”的窄而硬路径。

---

## 5. 核心运行时对象

### 5.1 `ModelInstance` 是唯一正式终态对象

Phase 1 的模型加载结果统一为：

```cpp
StatusOr<std::unique_ptr<ModelInstance>> ModelLoader::Load(...)
```

不再引入与 `ModelInstance` 并列的另一套正式运行时模型对象。

理由：

- 当前代码主线已经围绕 `ModelInstance` 展开
- `ExecutionPlanBuilder` 已依赖 `ModelInstance`
- `PackedWeights` 已明确由 `ModelInstance` 的 backend sidecar 持有
- Phase 1 应优先收敛，不再并行发展两套模型状态表达

---

### 5.2 `BackendSidecar` 持有 packed artifacts

当前正式所有权模型为：

```text
ModelInstance
  └── BackendSidecar
        └── PackedWeights
```

这条链路是 Phase 1 模型加载设计必须遵守的既有约束：

- backend 定义 packed format
- prepacker 构建 packed artifact
- `ModelInstance` 持有 packed artifact 生命周期
- execution 只借用 packed data 指针，不拥有 packed payload

---

### 5.3 `PackedWeights` 继续作为正式 packed 表达

Phase 1 不用新的 `PackedWeightHandle` 取代 `PackedWeights` 作为持久运行时对象。

当前正式接口：

```cpp
class PackedWeights {
public:
    virtual ~PackedWeights() = default;

    virtual OpType op_type() const noexcept = 0;
    virtual const KernelSelector& selector() const noexcept = 0;
    virtual const Buffer& storage() const noexcept = 0;
};
```

这意味着：

- packed artifact 的 backend-specific 细节仍由具体 `PackedWeights` 实现承载
- `ExecutionPlanBuilder` 只需读取 `storage().data()` 作为 `packed_params`
- 不必在 Phase 1 再引入一套新的 POD handle 体系来替代当前抽象

---

## 6. 加载主链的分层职责

模型加载建议拆成五层，但最终落点统一收敛到 `ModelInstance`。

### 6.1 Artifact Reader

负责读取外部模型工件。

职责：

- 打开模型目录/文件
- 读取 `config.json`
- 识别单文件或分片 safetensors
- 建立 tensor name -> raw tensor view 索引
- 持有或转交底层只读 backing 生命周期

不负责：

- backend 语义
- 模型支持性判断
- prepack 策略

---

### 6.2 Validator

负责判断模型是否被当前引擎支持。

职责：

- architecture 检查
- tensor 完整性检查
- shape / dtype 合法性检查
- config 与实际权重一致性检查
- tied embedding / lm_head 等约束校验

---

### 6.3 Resolver

负责把字符串命名的 tensor 集合解析成强类型模型结构。

职责：

- 将 `model.layers.3.self_attn.q_proj.weight` 解析到 `layer[3].attn.q_proj`
- 构建 `ResolvedTensorIndex`
- 为后续 prepack 与 builder 提供 O(1) 强类型访问

约束：

> 解析完成后，后续阶段不允许继续依赖字符串查表来访问模型权重。

---

### 6.4 Prepack Planner + Prepacker

负责预打包权重。

职责：

- 识别哪些 tensor 需要 prepack
- 为每个 tensor 选择对应的 CPU prepack 路径
- 规划 packed arena 的 size / alignment / offset
- 执行实际 prepack
- 生成 `std::unique_ptr<PackedWeights>`

Phase 1 约束：

- 正式实现以 `CpuWeightPrepacker` 为主
- 不要求先引入通用 `PrepackKernel` 体系
- 不要求用新的 `KernelId` 主线替代现有 `KernelSelector` 主线

---

### 6.5 `ModelInstanceBuilder`

负责把 config、raw 常量视图、backing 生命周期和 packed artifacts 组装成最终的 `ModelInstance`。

输出：

```cpp
StatusOr<std::unique_ptr<ModelInstance>>
```

而不是新的并列终态对象。

---

## 7. Reader 层的 HF 适配设计

为了同时支持当前 HF 本地目录和未来可能的其他格式，Reader 层允许格式适配，但这种适配只停留在 Reader 层。

建议目录：

```text
aethermind/model/
├── model_load_options.h
├── model_loader.h/.cpp
├── model_validator.h/.cpp
├── resolved_tensor_index.h/.cpp
├── model_instance_builder.h/.cpp
├── weight_prepack_planner.h/.cpp
├── weight_prepacker.h/.cpp
└── formats/hf/
    ├── hf_directory_reader.h/.cpp
    ├── hf_config_parser.h/.cpp
    ├── hf_safetensors_index.h/.cpp
    └── hf_tensor_name_resolver.h/.cpp
```

约束：

- `formats/hf/*` 只负责 HF 本地目录读取与解析
- 不定义新的正式运行时模型对象
- 不让 HF 外部格式语义进入 runtime / execution / backend 主线

---

## 8. `ResolvedTensorIndex` 设计

Resolver 阶段建议输出按层组织的 raw tensor 索引：

```cpp
struct ResolvedAttentionRawWeights {
    RawTensorView q_proj;
    RawTensorView k_proj;
    RawTensorView v_proj;
    RawTensorView o_proj;
};

struct ResolvedFfnRawWeights {
    RawTensorView gate_proj;
    RawTensorView up_proj;
    RawTensorView down_proj;
};

struct ResolvedNormRawWeights {
    RawTensorView input_rmsnorm;
    RawTensorView post_attn_rmsnorm;
};

struct ResolvedDecoderLayerRaw {
    ResolvedNormRawWeights norm;
    ResolvedAttentionRawWeights attn;
    ResolvedFfnRawWeights ffn;
};

struct ResolvedTensorIndex {
    RawTensorView embed_tokens;
    RawTensorView final_norm;
    std::optional<RawTensorView> lm_head;
    std::vector<ResolvedDecoderLayerRaw> layers;
};
```

这一步的目标是：

- 后续 prepack 不再到处做字符串查表
- builder 可以基于强类型索引稳定构造 `ModelInstance`

---

## 9. 预打包策略

### 9.1 必须预打包的权重

Phase 1 建议以下矩阵一律在加载期 prepack：

- `q_proj.weight`
- `k_proj.weight`
- `v_proj.weight`
- `o_proj.weight`
- `gate_proj.weight`
- `up_proj.weight`
- `down_proj.weight`
- `lm_head.weight`（如果 logits 走 GEMM）

### 9.2 保持 raw 的权重

以下按 raw 只读方式保留：

- `embed_tokens.weight`
- 所有 RMSNorm 权重
- `model.norm.weight`

### 9.3 arena 规划

建议采用两阶段预打包：

```text
collect requests -> compute packed size/alignment -> allocate once -> pack all
```

目的：

- 避免 pack 过程中碎片化分配
- 确保运行期只读消费

但注意：

> packed arena 是为了构造 `PackedWeights`，不是为了引入另一套新的 packed handle 主线。

---

## 10. 与 execution 的衔接

加载完成后，execution 不关心 HF 目录，也不关心 reader 的具体细节。

execution 只看到：

- `ModelInstance`
- `PackedWeights`
- 模型静态元数据
- raw 常量权重视图

`ExecutionPlanBuilder` 继续沿用当前主线：

```cpp
const PackedWeights* packed =
    model_instance.FindPackedWeights(op_type, selector);
```

然后将：

```cpp
packed->storage().data()
```

冻结进 `ExecutionStep.packed_params`。

这意味着：

- 不需要为 execution 再设计新的模型终态对象
- 不需要在热路径重新解析权重
- 不需要在运行期做首次 prepack

---

## 11. 生命周期与所有权

### 11.1 raw tensor backing

只要 `ModelInstance` 活着，任何 raw tensor view 的 backing 都必须有效。

建议：

- Reader 负责建立 backing
- `ModelInstanceBuilder` 负责把 backing 生命周期收进最终 `ModelInstance`
- 不允许 reader 析构后 raw tensor 指针悬挂

### 11.2 packed artifacts

所有 `PackedWeights` 由 `ModelInstance` 的 `BackendSidecar` 持有。

### 11.3 execution plan

`ExecutionPlan` 不拥有 packed artifacts，只借用：

```cpp
model_instance.FindPackedWeights(...)->storage().data()
```

因此必须保证：

```text
ModelInstance lifetime > ExecutionPlan execution lifetime
```

---

## 12. 顶层加载流程

顶层接口建议为：

```cpp
StatusOr<std::unique_ptr<ModelInstance>> ModelLoader::Load(
    const ModelLoadOptions& options,
    const Backend& backend,
    const KernelRegistry& registry);
```

主流程：

```text
ModelLoader::Load()
    ↓
识别 artifact format
    ↓
HfDirectoryReader / 其他 Reader
    ↓
读取 config 与 tensors
    ↓
Validator 校验
    ↓
构建 ResolvedTensorIndex
    ↓
生成 PrepackRequest
    ↓
规划 packed arena
    ↓
调用 CpuWeightPrepacker 进行 pack
    ↓
构建 ModelInstance
```

---

## 13. 顶层伪代码骨架

```cpp
StatusOr<std::unique_ptr<ModelInstance>> ModelLoader::Load(
    const ModelLoadOptions& options,
    const Backend& backend,
    const KernelRegistry& registry) {

    AM_ASSIGN_OR_RETURN(auto reader, ModelArtifactReader::Create(options));

    AM_ASSIGN_OR_RETURN(ModelConfig config, reader->ReadConfig());
    AM_RETURN_IF_ERROR(ModelValidator::ValidateConfig(config));

    AM_ASSIGN_OR_RETURN(auto raw_tensors, reader->MapAllTensors());
    AM_RETURN_IF_ERROR(ModelValidator::ValidateTensorSet(config, raw_tensors));

    AM_ASSIGN_OR_RETURN(
        ResolvedTensorIndex resolved,
        ResolvedTensorIndexBuilder::Build(config, raw_tensors));

    AM_ASSIGN_OR_RETURN(
        auto requests,
        WeightPrepackPlanner::BuildRequests(
            config, resolved, backend, registry, options));

    AM_ASSIGN_OR_RETURN(
        PrepackPlan plan,
        WeightPrepackPlanner::Plan(requests, backend, registry));

    AM_ASSIGN_OR_RETURN(
        auto model_instance,
        ModelInstanceBuilder::Create(config, resolved, raw_tensors));

    AM_RETURN_IF_ERROR(
        WeightPrepacker::Run(
            requests,
            plan,
            *model_instance,
            backend,
            registry));

    return model_instance;
}
```

这段伪代码强调：

- 最终返回 `ModelInstance`
- prepack 结果直接进入 `ModelInstance`
- execution 后续继续走既有 `FindPackedWeights(...)` 主线

---

## 14. 错误处理

建议至少分为四类。

### 14.1 Artifact Error

- 目录不存在
- `config.json` 缺失
- `model.safetensors` / `model.safetensors.index.json` 缺失
- shard 文件缺失
- safetensors header 非法
- tensor offset 越界

### 14.2 Validation Error

- architecture 不支持
- dense decoder-only 约束不满足
- shape 不匹配
- dtype 不支持
- tensor 不完整

### 14.3 Resolve Error

- 某层 q_proj 缺失
- 层数不完整
- 参数命名不符

### 14.4 Prepack Error

- 无法生成 prepack request
- packed size / alignment 计算失败
- `CpuWeightPrepacker` 执行失败
- `BackendSidecar` 存储失败

错误信息应尽可能带上：

- model path
- tensor name
- shape
- dtype
- op type
- backend
- layer id（如适用）

---

## 15. 推荐实现顺序

### 阶段 1：打通 HF 单文件加载

目标：

- 支持 `config.json + model.safetensors`
- 能读出配置与 tensor 表
- 能构建 raw 版 `ResolvedTensorIndex`

建议完成：

- `HfDirectoryReader::DiscoverLayout()`
- `HfConfigParser::ParseFromFile()`
- `HfTensorTable`
- `HfTensorNameResolver::ResolveLlamaDense()`

### 阶段 2：支持分片 safetensors

目标：

- 支持 `model.safetensors.index.json`
- 支持跨 shard 的统一 tensor 查找

建议完成：

- `HfSafetensorsIndex::Load()`
- 多分片 mmap / 映射
- shard 缺失与命名冲突检测

### 阶段 3：接入 CPU prepack 主链

目标：

- attention / ffn / lm_head 线性权重完成预打包
- packed arena 一次性规划与分配
- 构建可直接用于 execution 的 `ModelInstance`

建议完成：

- `WeightPrepackPlanner`
- `WeightPrepacker`
- `ModelInstanceBuilder`

### 阶段 4：增强鲁棒性与可观测性

目标：

- 更强错误信息
- 加载阶段 profiling
- 生命周期安全检查
- 更完整单元测试

---

## 16. 最终结论

本设计文档对 Phase 1 给出如下明确结论：

1. 正式支持 Hugging Face 风格本地 `config.json + safetensors` checkpoint 工件。
2. 该支持仅属于外部 artifact format 兼容层，不引入 Transformers 全语义。
3. 内部主链保持为：
   - `Reader -> Validator -> Resolver -> Prepack -> ModelInstance`
4. 权重预打包必须在加载期完成，而不是在执行期 lazy 初始化。
5. Phase 1 正式终态对象统一为当前主线中的 `ModelInstance`。
6. `PackedWeights` 继续由 `ModelInstance` 的 `BackendSidecar` 持有。

一句话总结：

> AetherMind Phase 1 应尽快支持 Hugging Face 风格本地 safetensors 模型目录，但最终运行时对象必须收敛到当前代码主线中的 `ModelInstance + BackendSidecar + PackedWeights`，而不是再发展一套并列的模型终态对象。

---

## 17. 实现 Checklist（Phase 1）

下面的 checklist 用于把本文档从设计层落到可执行实现层。所有条目都默认以当前收敛主线为前提：

- 最终对象是 `ModelInstance`
- packed 权重由 `BackendSidecar` 持有
- packed artifact 形式是 `PackedWeights`
- prepack 正式实现以 `CpuWeightPrepacker` 为主
- execution 继续通过 `ExecutionPlanBuilder -> FindPackedWeights(...)` 接入

### A. Artifact Reader（HF 本地目录支持）

- [ ] 实现 `HfDirectoryReader`，识别 `config.json + model.safetensors` 单文件目录
- [ ] 实现 `HfDirectoryReader`，识别 `config.json + model.safetensors.index.json + shards` 分片目录
- [ ] 实现 `HfSafetensorsIndex`，支持 tensor name -> shard file 映射
- [ ] 实现统一的 tensor table / raw tensor view 聚合接口，屏蔽 shard 细节
- [ ] 支持 mmap 或等价零拷贝只读映射路径
- [ ] 明确 backing 生命周期转交到后续 builder / `ModelInstance`，避免 reader 析构后悬挂指针

### B. Validator 与 Resolver

- [ ] 实现 `ModelValidator::ValidateConfig()`，校验 llama-family dense decoder-only 边界
- [ ] 实现 `ModelValidator::ValidateTensorSet()`，校验张量完整性、shape、dtype 与 config 一致性
- [ ] 实现 `ResolvedTensorIndexBuilder` 或等价 resolver，将 HF 名称映射到强类型层结构
- [ ] 确保 resolver 之后不再依赖字符串查表访问模型权重
- [ ] 覆盖 `embed_tokens` / `final_norm` / `lm_head` / qkv/o / gate/up/down / layernorm 等关键命名

### C. Prepack Planner 与 Prepacker

- [ ] 实现 `WeightPrepackPlanner::BuildRequests()`，识别所有需要 prepack 的线性层权重
- [ ] 实现 `WeightPrepackPlanner::Plan()`，在 prepack 前一次性计算 total bytes 与 max alignment
- [ ] 将 attention（Q/K/V/O）权重纳入 CPU prepack 主链
- [ ] 将 FFN（gate/up/down）权重纳入 CPU prepack 主链
- [ ] 在需要时将 `lm_head.weight` 纳入 CPU prepack 主链
- [ ] 确保 embedding / RMSNorm / final norm 等非矩阵权重按 raw 只读方式保留
- [ ] 正式接入 `CpuWeightPrepacker::Pack(...)`
- [ ] 确保 prepack 失败时返回明确错误，而不是静默 fallback 到热路径重排

### D. `ModelInstance` Builder 与所有权收敛

- [ ] 实现 `ModelInstanceBuilder::Create(...)` 或等价 builder，负责组装最终 `ModelInstance`
- [ ] 将 raw 常量权重视图绑定到 `ModelInstance` 生命周期
- [ ] 将所有 `PackedWeights` 通过 `ModelInstance::StorePackedWeights(...)` 写入 sidecar
- [ ] 确保 `BackendSidecar` 拒绝重复 `(OpType, KernelSelector)` 条目
- [ ] 确保最终 `ModelLoader::Load(...)` 的唯一正式输出为 `std::unique_ptr<ModelInstance>`
- [ ] 不引入与 `ModelInstance` 并列的新正式终态对象

### E. 与 Execution 主线对接

- [ ] 验证 `ExecutionPlanBuilder` 可通过 `ModelInstance::FindPackedWeights(op_type, selector)` 获取 packed 权重
- [ ] 验证 `ExecutionStep.packed_params` 冻结自 `PackedWeights::storage().data()`
- [ ] 确保 execution 不感知 HF 目录、Reader、Resolver 等外部工件层细节
- [ ] 确保 `ModelInstance` 生命周期覆盖任何依赖其 packed data 的 `ExecutionPlan` 执行期

### F. 错误处理与观测性

- [ ] 为 artifact error 提供精确错误：缺文件、坏 header、坏 shard map、越界 offset
- [ ] 为 validation error 提供精确错误：坏 architecture、shape 不匹配、dtype 不支持、tensor 缺失
- [ ] 为 resolve error 提供精确错误：层不完整、命名不符、关键 tensor 缺失
- [ ] 为 prepack error 提供精确错误：request 构造失败、alignment 失败、pack 执行失败、sidecar 存储失败
- [ ] 错误信息尽量包含 model path、tensor name、layer id、shape、dtype、backend 等上下文

### G. 验证门槛

- [ ] 单文件 HF 目录可成功加载并构建 `ModelInstance`
- [ ] 分片 HF 目录可成功加载并构建 `ModelInstance`
- [ ] 所有需要 prepack 的矩阵权重在加载期完成 prepack
- [ ] 加载完成后，execution 热路径中不再发生首次 prepack
- [ ] 新增 loader 相关单元测试覆盖：reader / validator / resolver / prepack / builder
- [ ] 与 `ExecutionPlanBuilder` 的接入测试通过
- [ ] 相关代码 `lsp_diagnostics` 无错误
- [ ] 相关构建与最窄测试目标通过

---

## 18. 推荐的实施顺序

如果按最小闭环推进，建议实现顺序为：

1. HF 单文件 Reader
2. `config.json` 解析 + Validator
3. `ResolvedTensorIndex` / 名称解析
4. 分片 safetensors 支持
5. CPU prepack planner + prepacker
6. `ModelInstanceBuilder`
7. 与 `ExecutionPlanBuilder` 对接
8. 错误增强与回归测试收口
