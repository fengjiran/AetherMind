# HfModelValidator 设计方案

版本：v1.0  
适用范围：AetherMind Phase 1 ModelLoader / Hugging Face 本地模型目录 / safetensors 权重加载  
目标模型：Llama-family dense decoder-only 模型  
运行边界：CPU-first、单进程、单模型、单请求、同步执行、静态 KV cache、greedy decoding、steady-state zero allocation

---

## 1. 设计定位

`HfModelValidator` 的职责不是校验底层文件格式，也不是判断当前 backend 是否有可用 kernel，而是校验：

> **当前模型配置与权重集合，是否能够构成 AetherMind Phase 1 支持的 Llama dense 推理模型。**

它位于模型加载链路中间：

```text
HfDirectoryReader / HfSafetensorsFile
    ↓
HfModelConfig + RawWeightTable
    ↓
HfModelValidator
    ↓
hf::ResolveWeights
    ↓
ModelInstanceBuilder
    ↓
WeightPrepackPlanner::BuildRequests
    ↓
WeightPrepackPlanner::PrepackAndStore
```

加载完成后，Executor 应该可以默认相信：

- 模型结构已经合法；
- 权重张量已经完整；
- 所有张量 shape 与 config 一致；
- dtype 在当前支持范围内；
- 不存在当前 Phase 1 不支持的结构，例如 MoE、LoRA、bias linear、量化权重等；
- 所有需要预打包的权重都具备合法的 rank、shape、dtype 和连续存储前提。

---

## 2. 职责边界

需要严格区分 `HfModelValidator` 与其他模块的职责。

### 2.1 不属于 HfModelValidator 的职责

| 模块 | 负责内容 |
|---|---|
| `HfSafetensorsFile` | 校验 safetensors 文件二进制格式是否合法，例如 header、dtype 字符串、shape、data_offsets、offset 越界、区间重叠、数据区空洞等 |
| `HfDirectoryReader` | 校验 Hugging Face 本地目录结构是否合法，例如 `config.json` 是否存在、`model.safetensors` 或 `model.safetensors.index.json` 是否存在、shard 文件是否存在 |
| `HfDirectoryReader::ParseConfig` | 解析 JSON 字段并归一化成内部 `HfModelConfig` |
| `hf::ResolveWeights` | 将字符串形式的 HF tensor name 映射到内部强类型结构 |
| `WeightPrepackPlanner` | 校验当前 backend/kernel registry 是否能够处理这些权重，并生成预打包计划 |
| `Executor` | 执行推理，不应重新校验模型结构与权重完整性 |

### 2.2 HfModelValidator 的核心职责

`HfModelValidator` 负责：

1. 校验 `HfModelConfig` 是否在 Phase 1 支持范围内；
2. 校验原始 tensor 集合是否完整、命名是否符合预期；
3. 校验是否存在不支持的结构性 tensor；
4. 校验 tensor rank、shape、dtype 与 Llama dense 语义是否匹配；
5. 校验 tied embedding / lm_head 语义；
6. 在进入权重预打包前建立必要的不变量。

---

## 3. 总体校验分层

当前 `HfModelValidator` 暴露两层接口；后续如果需要在 `ModelWeightIndex` 层做强语义校验，可再补充 `ValidateResolvedModel`：

```cpp
class HfModelValidator {
public:
    static Status ValidateConfig(const HfModelConfig& config);

    static Status ValidateWeightSet(
        const HfModelConfig& config,
        const RawWeightTable& weights);

    // Future extension, not present in the current implementation.
    static Status ValidateResolvedModel(
        const HfModelConfig& config,
        const ModelWeightIndex& resolved);
};
```

三层职责如下：

| 接口 | 输入 | 校验重点 |
|---|---|---|
| `ValidateConfig` | `HfModelConfig` | config 参数是否处于 Phase 1 支持范围内 |
| `ValidateWeightSet` | 原始 `RawWeightTable` | tensor 是否完整、命名是否合法、是否存在不支持 tensor |
| `ValidateResolvedModel` | `ModelWeightIndex` | 后续扩展：每个强类型权重的 shape、rank、dtype 是否严格匹配 config |

---

## 4. 推荐目录结构

建议文件组织如下：

```text
aethermind/model/
├── formats/hf/hf_model_validator.h
├── formats/hf/hf_model_validator.cpp
├── formats/hf/hf_model_config.h
├── formats/hf/hf_weight_resolver.h
├── raw_weight.h
└── model_weight_index.h
```

也可以先合并为较少文件，待模块变大后拆分。

---

## 5. 输入与输出对象

### 5.1 `HfModelConfig`

`HfModelValidator` 使用的配置应是已经由 `HfDirectoryReader::ParseConfig` 归一化后的内部结构，而不是原始 JSON。

建议结构：

```cpp
struct RopeConfig {
    float theta = 10000.0f;
    std::optional<float> scaling_factor;
    std::string scaling_type;
};

struct HfModelConfig {
    std::string model_type;       // Phase 1: "llama"
    std::string architecture;     // e.g. "LlamaForCausalLM"

    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t max_position_embeddings = 0;
    int32_t head_dim = 0;

    float rms_norm_eps = 1e-6f;
    std::string hidden_act = "silu";

    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool mlp_bias = false;

    std::optional<int32_t> bos_token_id;
    std::optional<int32_t> eos_token_id;
    std::optional<int32_t> pad_token_id;

    DataType weight_dtype_hint = DataType::kUnknown;
    RopeConfig rope;
};
```

---

### 5.2 `RawWeightTable`

原始张量表建议抽象为：

```cpp
using RawWeightTable = std::unordered_map<std::string, RawWeightView>;

struct RawWeightView {
    const std::byte* data = nullptr;
    size_t bytes = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawStorage> storage{};
};
```

`RawWeightTable` 来自 reader 层。  
对于 safetensors 文件，`RawWeightView::storage` 负责保持底层 mmap / 文件数据生命周期。

---

### 5.3 `ModelWeightIndex`

`hf::ResolveWeights` 将字符串 tensor name 解析为强类型结构：

```cpp
struct AttnRawWeights {
    RawWeightView q_proj;
    RawWeightView k_proj;
    RawWeightView v_proj;
    RawWeightView o_proj;
};

struct FFNRawWeights {
    RawWeightView gate_proj;
    RawWeightView up_proj;
    RawWeightView down_proj;
};

struct NormRawWeights {
    RawWeightView input_rmsnorm;
    RawWeightView post_attn_rmsnorm;
};

struct DecoderLayerRawWeights {
    NormRawWeights norm;
    AttnRawWeights attn;
    FFNRawWeights ffn;
};

struct ModelWeightIndex {
    RawWeightView embed_tokens;
    RawWeightView final_norm;
    std::optional<RawWeightView> lm_head;
    std::vector<DecoderLayerRawWeights> layers;
};
```

---

## 6. 校验选项

建议增加一个轻量选项结构，避免把所有策略写死。

```cpp
struct ModelValidationOptions {
    bool strict_tensor_names = false;
    bool allow_unknown_tensors = true;
    bool allow_rope_scaling = false;
    bool allow_bias = false;
    bool allow_quantized_tensors = false;
    bool allow_lora_or_adapter = false;
    bool require_lm_head_when_tied = false;
    bool require_uniform_linear_dtype = true;
};
```

Phase 1 默认建议：

```cpp
ModelValidationOptions options;
options.strict_tensor_names = false;
options.allow_unknown_tensors = true;
options.allow_rope_scaling = false;
options.allow_bias = false;
options.allow_quantized_tensors = false;
options.allow_lora_or_adapter = false;
options.require_lm_head_when_tied = false;
options.require_uniform_linear_dtype = true;
```

解释：

- **未知 tensor 默认 warning**：便于兼容部分 HF 导出多余字段；
- **结构性不支持 tensor 必须 error**：例如 bias、MoE、LoRA、quantized weights；
- **RoPE scaling 默认不支持**：避免长上下文变体被误加载；
- **linear dtype 默认要求统一**：降低 Phase 1 kernel/prepack 复杂度。

---

## 7. `ValidateConfig` 设计

`ValidateConfig` 只校验模型配置，不依赖权重文件。

### 7.1 architecture / model_type 白名单

Phase 1 只支持 Llama-family dense。

硬校验：

```text
model_type == "llama"
```

`architectures` 只能作为辅助信息，不能作为唯一判断依据。原因是部分 Hugging Face 导出目录可能缺少该字段，或者字段不完全标准。

建议：

```cpp
if (config.model_type != "llama") {
    return Unsupported(
        "Only model_type=llama is supported in AetherMind Phase 1, got: {}",
        config.model_type);
}
```

---

### 7.2 基础维度合法性

以下字段必须为正：

```text
vocab_size > 0
hidden_size > 0
intermediate_size > 0
num_hidden_layers > 0
num_attention_heads > 0
num_key_value_heads > 0
max_position_embeddings > 0
```

否则直接加载失败。

---

### 7.3 Attention 维度一致性

必须校验：

```text
hidden_size % num_attention_heads == 0
num_attention_heads % num_key_value_heads == 0
num_key_value_heads <= num_attention_heads
```

推导：

```text
head_dim = hidden_size / num_attention_heads
kv_hidden_size = num_key_value_heads * head_dim
num_query_groups = num_attention_heads / num_key_value_heads
```

如果 config 显式提供 `head_dim`，必须满足：

```text
config.head_dim == hidden_size / num_attention_heads
```

该校验是后续 q/k/v/o shape 校验的基础。

---

### 7.4 FFN 维度合法性

建议校验：

```text
intermediate_size >= hidden_size
```

对于标准 Llama dense MLP，该条件通常成立。

是否作为硬错误可配置。Phase 1 建议作为硬错误，因为 `intermediate_size < hidden_size` 很可能说明不是标准 Llama MLP 结构。

---

### 7.5 激活函数支持范围

Phase 1 若只实现 Llama SwiGLU，则 `hidden_act` 应只接受：

```text
silu
```

注意：HF Llama 配置中通常写的是 `"silu"`，但实际 MLP 结构是：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

不要要求 config 中必须出现 `"swiglu"`。

---

### 7.6 Bias 支持范围

Llama 默认无 bias。  
若当前实现不支持 bias linear，则必须要求：

```text
attention_bias == false
mlp_bias == false
```

如果 `allow_bias == false` 且 config 中出现 bias 开关为 true，应直接返回 `Unsupported`。

---

### 7.7 RMSNorm epsilon

校验：

```text
rms_norm_eps > 0
```

不要限定必须等于某个固定值，例如 `1e-6` 或 `1e-5`。不同模型变体可能不同，执行时按 config 使用。

---

### 7.8 RoPE 配置

至少校验：

```text
rope.theta > 0
max_position_embeddings > 0
```

若 Phase 1 不支持 RoPE scaling，则：

```cpp
if (!options.allow_rope_scaling && config.rope.scaling_factor.has_value()) {
    return Unsupported("RoPE scaling is not supported in Phase 1");
}
```

如果允许 rope scaling，则进一步校验：

```text
scaling_factor > 0
scaling_type in supported set
```

---

### 7.9 dtype hint

`torch_dtype` 或内部 `weight_dtype_hint` 只能作为提示，不应作为最终权重 dtype 依据。真正 dtype 以 safetensors header 为准。

可校验：

```text
weight_dtype_hint in {Unknown, F32, F16, BF16}
```

若出现 int4/int8 量化提示，但当前未实现量化，则应拒绝。

---

### 7.10 tied embedding 配置

`tie_word_embeddings` 本身不需要在 config 阶段判断对错。  
它将在 tensor set / resolved model 阶段影响 `lm_head.weight` 是否必需。

---

## 8. `ValidateWeightSet` 设计

`ValidateWeightSet` 面向原始 `RawWeightTable`，主要检查张量集合是否满足 Llama dense 命名和完整性要求。

### 8.1 必需全局 tensor

必须存在：

```text
model.embed_tokens.weight
model.norm.weight
```

`lm_head.weight` 依据 `tie_word_embeddings` 判断：

| 情况 | 要求 |
|---|---|
| `tie_word_embeddings == false` | `lm_head.weight` 必须存在 |
| `tie_word_embeddings == true` | `lm_head.weight` 可以存在，也可以缺失 |

如果 `require_lm_head_when_tied == true`，则即使 tied 也强制要求 `lm_head.weight` 存在。默认不建议开启。

---

### 8.2 每层必需 tensor

对于每一层 `i in [0, num_hidden_layers)`，必须存在：

```text
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight

model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight

model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
```

每层共 9 个 tensor：

```text
2 个 norm
4 个 attention linear
3 个 mlp linear
```

---

### 8.3 层号完整性

必须检查：

```text
layers.0 ... layers.{num_hidden_layers - 1}
```

不允许：

- 缺层；
- 跳层；
- 出现 `layers.{num_hidden_layers}` 或更大的层号；
- 出现无法解析的 layer index。

低层 reader 应已经拒绝重复 key。  
`HfModelValidator` 应处理“逻辑上多出一层”的错误。

---

### 8.4 不支持 tensor 检测

Phase 1 应显式拒绝以下类型 tensor。

#### 8.4.1 Bias tensor

若 `allow_bias == false`，拒绝：

```text
*.q_proj.bias
*.k_proj.bias
*.v_proj.bias
*.o_proj.bias
*.gate_proj.bias
*.up_proj.bias
*.down_proj.bias
lm_head.bias
```

#### 8.4.2 MoE tensor

拒绝包含以下关键词的 tensor：

```text
experts
router
moe
shared_experts
```

#### 8.4.3 LoRA / adapter tensor

若 `allow_lora_or_adapter == false`，拒绝：

```text
lora_A
lora_B
adapter
peft
```

#### 8.4.4 量化 tensor

若 `allow_quantized_tensors == false`，拒绝：

```text
qweight
qzeros
scales
g_idx
bits
group_size
```

这类字段常见于 GPTQ/AWQ 等量化 checkpoint，不是普通 dense float 权重。

---

### 8.5 tensor dtype 粗校验

对所有 AetherMind 将使用的权重 tensor，dtype 应在白名单中：

```text
F32
F16
BF16
```

如果出现：

```text
I8/U8/I32/I64/BOOL/Unknown
```

在 Phase 1 中应拒绝，除非它是明确忽略的非权重辅助 tensor。

建议策略：

- 必需权重 tensor：必须是 F32/F16/BF16；
- 未知 tensor：若 `strict_tensor_names == false`，可 warning；
- 未知 tensor 若表现为量化/MoE/LoRA/bias 结构，必须 error。

---

### 8.6 tensor rank 粗校验

在 `ValidateWeightSet` 层可做粗校验：

| tensor 类型 | rank |
|---|---:|
| embedding | 2 |
| lm_head | 2 |
| linear weight | 2 |
| norm weight | 1 |

更精确 shape 校验放在 `ValidateResolvedModel`。

---

### 8.7 禁止 empty tensor

虽然 safetensors 格式允许 empty tensor，但 Llama 权重不应为空。

所有必需权重 tensor 应满足：

```text
numel > 0
nbytes > 0
```

---

### 8.8 contiguous 校验

Phase 1 要求所有 tensor 连续存储。

```cpp
if (!tensor.spec.is_contiguous) {
    return Unsupported("Non-contiguous tensor is not supported");
}
```

对于 safetensors，reader 可以天然设置 `is_contiguous = true`。

---

### 8.9 未知 tensor 处理策略

建议默认：

```text
未知但无害 tensor：warning
未知且结构性影响执行：error
```

例如：

- `rotary_emb.inv_freq`：有些模型可能保存，可 warning 或忽略；
- `lora_A`：必须 error；
- `qweight/scales`：必须 error；
- `experts.*`：必须 error。

如果 `strict_tensor_names == true`，所有未知 tensor 均 error。

---

## 9. `ValidateResolvedModel` 设计

该阶段面对的是内部强类型结构，是最关键的语义校验。

### 9.1 通用推导变量

```cpp
const int64_t hidden = config.hidden_size;
const int64_t vocab = config.vocab_size;
const int64_t intermediate = config.intermediate_size;
const int64_t num_heads = config.num_attention_heads;
const int64_t num_kv_heads = config.num_key_value_heads;
const int64_t head_dim = hidden / num_heads;
const int64_t kv_hidden = num_kv_heads * head_dim;
```

---

### 9.2 全局权重 shape

#### Embedding

```text
model.embed_tokens.weight.shape == [vocab_size, hidden_size]
```

#### Final RMSNorm

```text
model.norm.weight.shape == [hidden_size]
```

#### LM Head

如果存在：

```text
lm_head.weight.shape == [vocab_size, hidden_size]
```

如果不存在：

```text
tie_word_embeddings 必须为 true
```

---

### 9.3 每层 Norm shape

对于每层：

```text
input_layernorm.weight.shape == [hidden_size]
post_attention_layernorm.weight.shape == [hidden_size]
```

---

### 9.4 每层 Attention shape

#### q_proj

```text
q_proj.weight.shape == [hidden_size, hidden_size]
```

#### k_proj

```text
k_proj.weight.shape == [num_key_value_heads * head_dim, hidden_size]
```

#### v_proj

```text
v_proj.weight.shape == [num_key_value_heads * head_dim, hidden_size]
```

#### o_proj

```text
o_proj.weight.shape == [hidden_size, hidden_size]
```

说明：

- 对 MHA：`num_key_value_heads == num_attention_heads`，所以 `kv_hidden == hidden_size`；
- 对 GQA/MQA：`kv_hidden < hidden_size`。

---

### 9.5 每层 MLP shape

Llama MLP 结构为：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

因此：

#### gate_proj

```text
gate_proj.weight.shape == [intermediate_size, hidden_size]
```

#### up_proj

```text
up_proj.weight.shape == [intermediate_size, hidden_size]
```

#### down_proj

```text
down_proj.weight.shape == [hidden_size, intermediate_size]
```

---

### 9.6 dtype 一致性

建议 Phase 1 采用折中策略：

#### Linear dtype

所有 linear 权重应具有统一 dtype：

```text
q_proj/k_proj/v_proj/o_proj
gate_proj/up_proj/down_proj
lm_head.weight
```

允许：

```text
F32/F16/BF16
```

若 `require_uniform_linear_dtype == true`，发现混合 dtype 则 error。

#### Embedding dtype

建议与 linear dtype 一致。  
如果不一致，Phase 1 直接拒绝。

#### Norm dtype

允许：

```text
F32/F16/BF16
```

如果 RMSNorm kernel 只支持 fp32 权重，则应在这里强制要求 F32。  
如果 RMSNorm kernel 支持加载期转换，则可允许 F16/BF16。

建议当前先允许 F32/F16/BF16，但在后续 `ModelInstanceBuilder` 或 prepack/const 构建中决定是否转换。

---

### 9.7 tied embedding 语义校验

#### `tie_word_embeddings == true`

允许：

1. `lm_head.weight` 缺失；
2. `lm_head.weight` 存在。

如果 `lm_head.weight` 存在，需要校验：

```text
lm_head.shape == embed_tokens.shape
lm_head.dtype == embed_tokens.dtype
```

不建议默认做全量数据比较，因为模型可能很大。可后续增加 debug 选项：

```cpp
bool verify_tied_embedding_content = false;
```

#### `tie_word_embeddings == false`

必须存在：

```text
lm_head.weight
```

并且：

```text
lm_head.shape == [vocab_size, hidden_size]
```

---

## 10. 与 WeightPrepackPlanner 的职责分界

`HfModelValidator` 只证明：

```text
这些 tensor 是合法的 Llama dense 权重。
```

它不证明：

```text
当前 CPU backend 是否有 kernel 能处理这些权重。
```

例如，下面内容不应由 `HfModelValidator` 判断：

- AVX2/AVX512/AMX kernel 是否可用；
- packed layout 的 block size；
- K/N/M 维度是否满足某个 kernel 的特殊约束；
- packed arena 大小；
- prepack kernel 是否存在。

这些应由 `WeightPrepackPlanner` / `KernelRegistry` 判断。

分界线：

```text
HfModelValidator:
  rank/shape/dtype/结构语义是否合法

WeightPrepackPlanner:
  backend/kernel 是否支持这些合法权重
```

---

## 11. 错误模型设计

建议定义错误类别：

```cpp
enum class ModelValidationErrorCode {
    kUnsupportedArchitecture,
    kInvalidConfig,
    kInvalidAttentionConfig,
    kInvalidRopeConfig,
    kMissingTensor,
    kUnexpectedTensor,
    kUnsupportedTensor,
    kInvalidTensorShape,
    kInvalidTensorRank,
    kInvalidTensorDType,
    kInvalidTensorStorage,
    kLayerCountMismatch,
    kTiedEmbeddingMismatch,
};
```

如果当前已有统一 `Status` / `StatusOr`，不一定需要独立 error class，但错误消息应携带足够上下文。

---

## 12. 错误信息规范

错误信息必须可定位。

### 12.1 shape 错误示例

```text
Invalid tensor shape:
tensor=model.layers.7.self_attn.k_proj.weight,
expected=[1024,4096],
actual=[4096,4096],
config.hidden_size=4096,
config.num_attention_heads=32,
config.num_key_value_heads=8,
head_dim=128
```

### 12.2 缺失 tensor 示例

```text
Missing required tensor:
tensor=model.layers.13.mlp.down_proj.weight,
layer=13,
model_type=llama,
num_hidden_layers=32
```

### 12.3 不支持结构示例

```text
Unsupported tensor found:
tensor=model.layers.0.mlp.experts.0.up_proj.weight,
reason=MoE weights are not supported in AetherMind Phase 1
```

### 12.4 dtype 错误示例

```text
Invalid tensor dtype:
tensor=model.layers.0.self_attn.q_proj.weight,
expected one of [F32,F16,BF16],
actual=I8
```

---

## 13. 辅助函数设计

建议将常用检查封装成辅助函数。

```cpp
Status ExpectRank(const RawWeightView& tensor, int expected_rank);

Status ExpectShape(
    const RawWeightView& tensor,
    std::initializer_list<int64_t> expected);

Status ExpectDTypeIn(
    const RawWeightView& tensor,
    absl::Span<const DataType> allowed);

Status ExpectContiguous(const RawWeightView& tensor);

Status ExpectNonEmpty(const RawWeightView& tensor);

Status ExpectWeightTensor(
    const RawWeightView& tensor,
    int expected_rank,
    std::initializer_list<int64_t> expected_shape,
    absl::Span<const DataType> allowed_dtypes);
```

也可以增加 shape 打印工具：

```cpp
std::string ShapeToString(absl::Span<const int64_t> shape);
std::string TensorDebugString(const RawWeightView& tensor);
```

这与之前 kernel registry 中增强 `ToString()` 的思路一致，有助于提高错误可诊断性。

---

## 14. 头文件接口草案

```cpp
#pragma once

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/raw_weight.h"

namespace aethermind {

class HfModelValidator final {
public:
    HfModelValidator() = delete;

    static Status ValidateConfig(const HfModelConfig& config);

    static Status ValidateWeightSet(
        const HfModelConfig& config,
        const RawWeightTable& weights);
};

}  // namespace aethermind
```

---

## 15. `ValidateConfig` 伪代码

```cpp
Status HfModelValidator::ValidateConfig(
    const HfModelConfig& config,
    const ModelValidationOptions& options) {
    if (config.model_type != "llama") {
        return Unsupported("Only model_type=llama is supported");
    }

    RETURN_IF_ERROR(ExpectPositive(config.vocab_size, "vocab_size"));
    RETURN_IF_ERROR(ExpectPositive(config.hidden_size, "hidden_size"));
    RETURN_IF_ERROR(ExpectPositive(config.intermediate_size, "intermediate_size"));
    RETURN_IF_ERROR(ExpectPositive(config.num_hidden_layers, "num_hidden_layers"));
    RETURN_IF_ERROR(ExpectPositive(config.num_attention_heads, "num_attention_heads"));
    RETURN_IF_ERROR(ExpectPositive(config.num_key_value_heads, "num_key_value_heads"));
    RETURN_IF_ERROR(ExpectPositive(config.max_position_embeddings, "max_position_embeddings"));

    if (config.hidden_size % config.num_attention_heads != 0) {
        return InvalidArgument("hidden_size must be divisible by num_attention_heads");
    }

    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        return InvalidArgument("num_attention_heads must be divisible by num_key_value_heads");
    }

    if (config.num_key_value_heads > config.num_attention_heads) {
        return InvalidArgument("num_key_value_heads must be <= num_attention_heads");
    }

    const int32_t inferred_head_dim =
        config.hidden_size / config.num_attention_heads;

    if (config.head_dim != 0 && config.head_dim != inferred_head_dim) {
        return InvalidArgument("head_dim mismatch with hidden_size / num_attention_heads");
    }

    if (config.intermediate_size < config.hidden_size) {
        return Unsupported("intermediate_size < hidden_size is not supported for Llama dense MLP");
    }

    if (config.hidden_act != "silu") {
        return Unsupported("Only hidden_act=silu is supported for Llama MLP");
    }

    if (!options.allow_bias && (config.attention_bias || config.mlp_bias)) {
        return Unsupported("Bias linear is not supported in Phase 1");
    }

    if (config.rms_norm_eps <= 0.0f) {
        return InvalidArgument("rms_norm_eps must be positive");
    }

    if (config.rope.theta <= 0.0f) {
        return InvalidArgument("rope.theta must be positive");
    }

    if (!options.allow_rope_scaling && config.rope.scaling_factor.has_value()) {
        return Unsupported("RoPE scaling is not supported in Phase 1");
    }

    return OkStatus();
}
```

---

## 16. `ValidateResolvedModel` 伪代码

```cpp
Status HfModelValidator::ValidateResolvedModel(
    const HfModelConfig& config,
    const ModelWeightIndex& resolved,
    const ModelValidationOptions& options) {
    const int64_t hidden = config.hidden_size;
    const int64_t vocab = config.vocab_size;
    const int64_t intermediate = config.intermediate_size;
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden = config.num_key_value_heads * head_dim;

    RETURN_IF_ERROR(ExpectShape(resolved.embed_tokens, {vocab, hidden}));
    RETURN_IF_ERROR(ExpectShape(resolved.final_norm, {hidden}));

    if (resolved.lm_head.has_value()) {
        RETURN_IF_ERROR(ExpectShape(*resolved.lm_head, {vocab, hidden}));
    } else if (!config.tie_word_embeddings) {
        return InvalidArgument(
            "lm_head.weight is required when tie_word_embeddings=false");
    }

    if (resolved.layers.size() !=
        static_cast<size_t>(config.num_hidden_layers)) {
        return InvalidArgument("Layer count mismatch");
    }

    for (size_t i = 0; i < resolved.layers.size(); ++i) {
        const auto& layer = resolved.layers[i];

        RETURN_IF_ERROR(ExpectShape(layer.norm.input_rmsnorm, {hidden}));
        RETURN_IF_ERROR(ExpectShape(layer.norm.post_attn_rmsnorm, {hidden}));

        RETURN_IF_ERROR(ExpectShape(layer.attn.q_proj, {hidden, hidden}));
        RETURN_IF_ERROR(ExpectShape(layer.attn.k_proj, {kv_hidden, hidden}));
        RETURN_IF_ERROR(ExpectShape(layer.attn.v_proj, {kv_hidden, hidden}));
        RETURN_IF_ERROR(ExpectShape(layer.attn.o_proj, {hidden, hidden}));

        RETURN_IF_ERROR(ExpectShape(layer.ffn.gate_proj, {intermediate, hidden}));
        RETURN_IF_ERROR(ExpectShape(layer.ffn.up_proj, {intermediate, hidden}));
        RETURN_IF_ERROR(ExpectShape(layer.ffn.down_proj, {hidden, intermediate}));
    }

    RETURN_IF_ERROR(ValidateDTypeConsistency(config, resolved, options));

    return OkStatus();
}
```

---

## 17. 推荐校验流程

完整加载链路中，建议按以下顺序调用：

```text
1. HfDirectoryReader::ParseConfig()
2. HfModelValidator::ValidateConfig(config)

3. HfDirectoryReader / HfSafetensorsFile 构建 RawWeightTable
4. HfModelValidator::ValidateWeightSet(config, raw_weights)

5. hf::ResolveWeights(config, raw_weights)

6. ModelInstanceBuilder::Create(...)
7. WeightPrepackPlanner::BuildRequests(...)
8. WeightPrepackPlanner::PrepackAndStore(...)
```

这样可以尽早失败，并且保证每个阶段的职责清晰。

---

## 18. 单元测试设计

### 18.1 `ValidateConfig` 测试

覆盖：

- 合法 Llama config；
- 非 llama `model_type`；
- `hidden_size` 不能整除 `num_attention_heads`；
- `num_attention_heads` 不能整除 `num_key_value_heads`；
- `num_key_value_heads > num_attention_heads`；
- `hidden_act != silu`；
- `attention_bias == true`；
- `rope.theta <= 0`；
- rope scaling 默认拒绝。

### 18.2 `ValidateWeightSet` 测试

覆盖：

- 完整 tensor set 通过；
- 缺少 `model.embed_tokens.weight`；
- 缺少某层 `q_proj.weight`；
- 多出 layer index；
- `tie_word_embeddings=false` 但缺少 `lm_head.weight`；
- 出现 bias tensor；
- 出现 LoRA tensor；
- 出现 MoE tensor；
- 出现量化 tensor；
- 未知 tensor 在非 strict 模式 warning；
- 未知 tensor 在 strict 模式 error。

### 18.3 `ValidateResolvedModel` 测试

覆盖：

- 合法 shape 通过；
- embedding shape 错误；
- final norm shape 错误；
- q_proj shape 错误；
- GQA 下 k/v shape 错误；
- MLP gate/up/down shape 错误；
- tied embedding 下 lm_head 缺失通过；
- non-tied embedding 下 lm_head 缺失失败；
- linear dtype 混用失败；
- norm dtype 在允许范围内通过。

### 18.4 真实 safetensors smoke test

建议使用小型 Llama safetensors 测试模型，例如：

```text
peft-internal-testing/tiny-random-LlamaForCausalLM
```

测试链路：

```text
HfDirectoryReader
  -> HfDirectoryReader::ParseConfig
  -> HfSafetensorsFile
  -> HfModelValidator::ValidateConfig
  -> HfModelValidator::ValidateWeightSet
  -> hf::ResolveWeights
  -> HfModelValidator::ValidateResolvedModel
```

该测试不要求推理结果正确，只验证模型工件结构可以被正确解析和校验。

---

## 19. 后续扩展策略

### 19.1 支持更多 Llama-family 变体

可以通过白名单逐步扩展：

```text
llama
mistral
qwen2
gemma
```

但每一种都应有独立的 tensor name resolver 和 shape validator，不建议用一个巨大的 if-else 混合处理。

### 19.2 支持 bias linear

扩展点：

- `ModelValidationOptions::allow_bias = true`
- 解析 bias tensor；
- 修改 `ModelWeightIndex`；
- 修改 Linear kernel / prepack path；
- 更新 shape 校验。

### 19.3 支持量化权重

量化格式不应直接复用 dense float 校验逻辑。

建议新增：

```text
QuantizedTensorValidator
QuantizedWeightResolver
QuantizedPrepackPlanner
```

### 19.4 支持 MoE

MoE 不是简单多几个 tensor，而是执行图、路由逻辑和内存访问模式都不同。  
应作为独立架构支持，不应混入 dense Llama Validator。

---

## 20. 最终结论

`HfModelValidator` 的核心职责可以概括为：

> **在进入权重预打包之前，证明当前 checkpoint 确实是 AetherMind Phase 1 支持的 Llama dense 模型，并且所有权重的名称、形状、dtype、层数和 tied embedding 语义与 config 完全一致。**

推荐最终分层：

```text
ValidateConfig:
  配置是否合法，是否处于 Phase 1 支持范围

ValidateWeightSet:
  原始 tensor 集合是否完整，是否存在不支持结构

ValidateResolvedModel:
  解析后的 Llama 权重 shape/dtype 是否严格匹配 config

WeightPrepackPlanner:
  当前 backend/kernel 是否能处理这些合法权重
```

该设计可以保证：

- 加载期尽早失败；
- 错误信息可定位；
- Executor 热路径无需重复校验；
- Prepack 阶段不再处理模型语义问题；
- 后续扩展其他模型格式和架构时，职责边界清晰。
