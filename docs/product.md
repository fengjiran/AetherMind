# AetherMind 产品需求

> 产品范围、验收标准与非目标的唯一权威。系统架构见 [architecture.md](architecture.md)，模块设计见 [modules/](modules/)。

- **核心目标**：交付生产级桌面/服务器 CPU 本地推理引擎，并为服务化、异构计算和分布式演进保留可验证的架构边界。
- **关键技术**：C++20、Safetensors、FP32 / W16-A32、Static KV Cache；INT8/INT4 为待交付目标。

---

## 效力说明与设计约束

本文档采用**双层结构**：

- **当前产品合同（本文档主体）**：可执行的详细产品合同，聚焦桌面/服务器 CPU 本地推理运行时
- **长期演进方向（附录 A）**：战略愿景，仅作为方向指引，**不构成当前交付承诺**

阶段编号不再作为规范性产品分层。产品边界以“当前承诺 / 当前不承诺 / 长期演进方向”和可验证的 capability 状态表达，避免编号被误读为实现状态或时间承诺。

主体中的需求与目标继续构成交付要求；实现状态和验证证据另列于 §1.4 与 §5。标注“已实现”仅说明已有生产入口，验证结论限定于所引用记录的源码身份、平台、模型、精度和测试区间。专项回归通过、历史集成矩阵通过和全部产品验收完成必须分别记录。

**核心设计约束**：

- 当前产品范围是**独立可交付的产品**，而非临时原型
- 向后兼容通过**版本化 C ABI** 实现，而非永久稳定承诺
- 当前不承诺的长期能力必须基于当前产品的实现证据重新评估，必要时独立成册

---

## 1. 当前产品概述

### 1.1 定位与边界

**当前产品 = 桌面/服务器 CPU 本地推理引擎**

```
当前产品边界（本文档）
├── 单进程、同步单请求；每次请求绑定一个模型
├── Token IDs 输入 / Token IDs 输出（无文本分词器）
├── 仅 CPU 后端（GPU 后端属于长期演进方向）
├── 调用线程内同步执行；共享 Runtime 由宿主串行化（无调度器/批处理）
├── 贪婪采样唯一（无 temperature/top-k/top-p）
├── 静态预分配 KV Cache（无 PagedAttention）
└── Llama 家族 Dense 模型（无 MoE）
```

**当前明确不承诺**：

- ❌ HTTP/gRPC 服务接口
- ❌ 连续批处理 / Chunked Prefill
- ❌ Prefix Caching / PagedAttention
- ❌ GPU/CUDA 后端
- ❌ 文本分词器（Tokenizer）
- ❌ MoE / 专家并行
- ❌ 投机解码 / Multi-LoRA
- ❌ FP8 执行

### 1.2 核心价值主张

| 价值点 | 当前产品承诺 | 可能的长期演进 |
|--------|--------------|--------------|
| **本地部署** | 面向桌面/服务器 CPU 的 INT8/INT4 量化推理，支持 7B 模型运行 | GPU 卸载与异构执行 |
| **确定性输出** | 贪婪采样 + 数值稳定的 CPU 参考内核 | temperature/top-p 等采样策略 |
| **内存可预测性** | 静态 KV、分阶段 payload 预算，完成准备后的 Decode 稳态零分配 | PagedAttention 与动态缓存管理 |
| **API 演进** | Runtime C ABI v1.0 已由 M5 冻结、M6 实现并完成限定范围的 C 调用验收；发行与全部产品门禁独立记录 | HTTP/API 服务层 |

### 1.3 目标用户

- **桌面应用开发者**：需要在本地工作站或开发机集成 LLM 推理能力
- **服务器端开发者**：需要在 CPU 服务器环境中部署本地推理能力
- **AI Agent 框架构建者**：需要确定性、可预测的本地推理能力
- **研究人员**：需要数值稳定、可复现的推理基线

### 1.4 当前能力与验证状态

| 能力 | 当前实现与支持范围 | 边界 |
|------|--------------------|------|
| C++ FP32 Llama Dense | CPU O1/plain、O2/packed；同步 Prefill/Decode/Generate | tiny HF / 生成 checkpoint 的限定验证；不代表大模型质量 |
| C++ FP32 compatibility | 显式将 F16/BF16 权重拓宽为 FP32；CPU O1/plain | 不保留 16-bit 矩阵 |
| C++ W16/A32 | 保留 F16 或 BF16 矩阵，仅转换 Norm；FP32 activation、累加、KV、logits；CPU O1/plain、O2/identity-packed | Linear 当前为 reference 路径 |
| 独立 CLI / C ABI v1.0 | 已实现，继续采用 CPU strict FP32 source admission | 不开放 C++ reduced-weight policy |
| 静态 KV 与资源预算 | KV 默认 F32，C++ retained-W16 可显式选择 F16/BF16；支持模型、packed、KV、workspace 与请求 payload 限额及资源查询 | 见 [资源预算指南](guides/resource-budgeting.md) |
| INT8/INT4 仅权重量化 | 保留为当前产品交付需求；尚无端到端量化执行链 | loader 接受 dtype/metadata 不等于 quantized graph、packing、kernel 与生成已接通 |
| FP16/BF16 KV | C++ CPU W16/A32 的四种 W×KV 组合、O1/plain 与 O2/identity-packed reference 已接通 | 限定 fixture 验收；不代表任意模型/平台质量或加速保证 |
| 算子内并行 | 尚未建立 OpenMP/线程池生产执行与验收证据 | 需另行明确线程所有权、确定性、零分配与 TSAN 合同 |
| 7B 量化模型与目标性能 | 目标见 §4.1，尚未完成代表性模型验收 | tiny fixture 或 microbenchmark 不能替代大模型质量、冷启动、RSS 与吞吐证据 |

当前模型级证据主要来自 Linux/x86_64 的 tiny HF 与生成 checkpoint。其他 compiler、OS、architecture、物理无 AVX 平台及生产规模模型的覆盖，按各专题验证记录解释。FP16/BF16 activations（A16）也尚未交付，其支持范围需独立定义。

---

## 2. 当前产品架构（严格边界内）

### 2.1 分层与边界

当前产品采用 `inference → compiler/model/graph/execution/backend/runtime/kv_cache` 的分层调用。唯一权威分层图、依赖方向与模块地图见 [architecture.md](architecture.md)，模块细节见 [modules/](modules/)。

产品级硬约束：

- 外部入口只有 Token IDs：独立 CLI、C++ API、C ABI v1；tokenizer、HTTP、异步/流式 API 不在当前范围。
- 单进程、同步、单请求；不承诺 request scheduler、批处理或多 Session 并发 Generate。
- 单 active KV session slot；KV 静态预分配、不自动扩容；Paged KV 为长期方向。
- 准备期选核，执行期只调用冻结 kernel；Decode 稳态零分配，`Generate` 整体不是零分配接口。
- 模块所有权以 [AGENTS.md](../AGENTS.md) §2.1 为准；产品边界不以 Concepts 或模板等语言特性定义。
- 编译错误必须先于 KV 分配；Runtime 从 Prepare 起保持地址稳定，并长于 model、plan 与 session。

### 2.2 API 边界定义

**核心原则：Token IDs 是唯一数据边界**

`InferenceSession::Generate` 与 Runtime C ABI v1.0 均已实现。C ABI 合同由 M5 冻结、M6 接通，继续限定 CPU strict FP32。C++ 通过 `ExecutableModel::Load(Runtime&, directory, ModelLoadOptions)` 显式选择权重策略，或通过分阶段 Load/Compile 后调用 `ExecutableModel::Prepare`。精确声明、预算与 lifetime 以 [公共 API](api/public-api.md)、[c_api.h](../include/c_api.h) 和 [Runtime C ABI v1](api/runtime-c-abi.md) 为准。

| `ModelLoadOptions::weights` | 权重处理 | CPU preset |
|---------------------------|----------|------------|
| `WeightPolicy::kRequireFp32`（默认） | 源权重要求 scalar FP32 | O1/plain、O2/packed |
| `WeightPolicy::kConvertToFp32` | 保留 FP32 view，F16/BF16 必需权重显式拓宽为 FP32 | O1/plain |
| `WeightPolicy::kKeepFp16WeightsFp32Activation` | 保留 F16 矩阵，仅转换 Norm 为 FP32 | O1/plain、O2/identity-packed |
| `WeightPolicy::kKeepBf16WeightsFp32Activation` | 保留 BF16 矩阵，仅转换 Norm 为 FP32 | O1/plain、O2/identity-packed |

W16/A32 要求 Embedding、Linear/LM head 等矩阵的实际 dtype 匹配选定 policy，Norm 源可为 scalar F32/F16/BF16；activation、residual、logits 为 FP32，W16 projection 使用 FP32 累加。KV 默认 FP32，retained-W16 policy 可显式选择 F16/BF16，与矩阵精度独立，所有层 K/V 同型。O2 的 W16 identity/composite artifacts 保留两字节元素，并产生额外 packed storage。使用与数值合同见 [FP32 compatibility](guides/fp32-compatible-inference.md) 和 [W16/A32](guides/w16-fp32-inference.md)。这些 C++ policy 不改变 CLI/C ABI 的 strict FP32 准入。

```cpp
// C++ API
struct GenerationConfig {
    size_t max_new_tokens = 0;
    std::optional<uint32_t> eos_token_id{};
};

class InferenceSession {
public:
    static StatusOr<InferenceSession> Create(
        Runtime& runtime,
        std::shared_ptr<const ExecutableModel> model);

    // 返回仅包含本次新生成的 token IDs；遇到 EOS 时包含 EOS 后停止。
    StatusOr<std::vector<uint32_t>> Generate(
        std::span<const uint32_t> prompt_tokens,
        const GenerationConfig& config) const;
};

// C ABI v1 冻结签名；M6 已提供正式定义。
// 输入/输出均为 uint32_t 数组，无文本处理
am_status_code am_session_generate(
    am_session_t session,
    const uint32_t* input_tokens,
    size_t input_len,
    uint32_t* output_tokens,
    size_t output_capacity,
    size_t* output_len,
    am_error_handle* error
);
```

**明确排除**：

- ❌ 字符串输入/输出
- ❌ Tokenizer 集成
- ❌ 流式回调（callbacks）
- ❌ 异步/Promise/Future 接口

---

## 3. 当前产品功能需求

### 3.1 模型加载与管理

#### [P0] Safetensors 加载器

- **输入**：HuggingFace 格式目录（`config.json` + `*.safetensors`）
- **验证**：
  - 仅接受 Llama-family Dense 模型
  - 显式拒绝：MoE、编码器-解码器、滑动窗口注意力
  - 必需字段：model_type, hidden_size, num_hidden_layers, num_attention_heads, etc.
- **重排**：在 semantic graph 优化后，按具体 `WeightBinding` 将权重物化为内部 CPU 优化布局；模型加载阶段不执行预打包。
- **量化目标**：支持 INT8/INT4 仅权重量化加载与执行；需接通 quantization metadata、typed graph/lowering、packing、kernel 与生成数值门禁。当前可执行范围见 §1.4，不能用 loader 的格式接受能力替代端到端验收。

#### [P0] 模型配置契约

| 配置项 | 当前合同 / 目标 | 准入与状态 |
|--------|-----------------|------------|
| model_type | 当前 `model_type = "llama"` 的 Dense 模型 | `ParseModelArchitecture` 为唯一家族判定入口；Llama 2/3 是模型代际名称，不作为额外的 model_type 字符串接受 |
| attention | GQA、MQA、MHA | sliding window、MoE、encoder-decoder 拒绝 |
| activation | `hidden_act = "silu"`，构建 SwiGLU | 其他 activation 拒绝，新增支持须贯通语义、graph、kernel 与模型测试 |
| norm | RMSNorm | 不提供 LayerNorm 模型执行 |
| weight precision | strict FP32、显式 FP32 compatibility、显式 F16/BF16 W16/A32 | 按 §2.2 的 policy 和实际 resolved weight dtype 准入；不隐式转换 |
| quantization | 目标：INT8 per-channel、INT4 group-wise（group_size=128） | 待端到端实现与 scheme-specific 验收；其他 scheme 不承诺 |

### 3.2 内存管理

#### [P0] 静态预分配策略

```text
当前高层执行的存储组成：
Model weights：FP32，或 W16 矩阵 + FP32 Norm
    └─ O2 按 recipe 另持有 packed artifacts
KV cache：默认 FP32；C++ retained-W16 可显式选 F16/BF16 K/V planes，Runtime 构造时静态分配
    └─ [layer][head][token][head_dim]，物理 padding/strides 由 layout 决定
Request：workspace + max(Prefill activation, Decode activation)
    └─ 两个 activation arena 分阶段存活；Decode bindings/params 准备后复用
```

KV capacity 由调用方显式设置并受模型 context、checked layout bytes 和预算约束；4k 是长序列验收目标，不是统一的硬编码容量上限。请求内不自动扩容 KV。FP16/BF16 KV 的当前数值验收限定于 P3 记录列出的 fixture 与 2048-token 范围；更大上下文独立验收。INT8/INT4 权重及约 3.5 GB 的 7B INT4 逻辑 payload 属于目标估算，不能据此推导总内存占用已达标。

**资源预算与失败行为**：

- `KVCacheRuntimeOptions::max_bytes` 在 Runtime 分配 K/V 前检查物理 plane payload；`ModelLimits` 在准备期约束已知 owned backing、inline constants 与 packed payload。
- `WorkspaceRuntimeOptions::default_workspace_limit_bytes` 与 `InferenceSessionOptions::max_request_payload_bytes` 约束单请求 workspace 与 activation peak。有限预算在相应 tensor payload 分配前检查，容量、busy、metadata 和 OOM 错误保持各自状态语义。
- `InferenceSession::PlanResources` 提供 concrete request 的只读资源查询，不预约 KV、不分配 tensor Buffer；cold-path metadata 可以分配。`Generate` 仍独立执行准入。
- 高层 limit 的 `0` 表示 unlimited。payload 账本不等于 allocator rounded capacity、进程 RSS 或全局多模型配额；mapped/external backing 单独报告，未知 backing 不能当作零。

字节口径、owner 去重、转换峰值、有限 cap 与零 token 请求的精确规则见 [资源预算指南](guides/resource-budgeting.md)、[FP32 compatibility](guides/fp32-compatible-inference.md) 和 [W16/A32](guides/w16-fp32-inference.md)。

**验收判定依据**：

- [ ] 完成 Decode binding/params preparation 后，成功的重复执行区间中 malloc-family 调用次数为 0，包含 `malloc`/`calloc`/`realloc`/`free` 与对齐分配族；准备与清理区间单独记录
- [ ] 内存占用在目标硬件范围内（见 4.1 节）
- [ ] 无地址/UB 错误及泄漏（ASan/UBSan 与实际启用的 LSan 检查分别留存证据）

以上为交付检查项，局部通过范围见对应验证记录。`Generate` 每次重新准备请求资源并返回 owning token vector，整个 API 调用不承诺零分配。

### 3.3 执行引擎

#### [P0] 同步执行流程

```
Prompt Tokens
      │
      ▼
┌─────────────┐
│   Prefill   │ ──► 计算 KV Cache，输出第一个 token
│  (全序列)    │
└─────────────┘
      │
      ▼
┌─────────────┐
│    Decode   │ ──► 逐个生成 token，直到 eos/max_tokens
│  (逐个token) │     贪婪采样：argmax(logits)
└─────────────┘
      │
      ▼
Output Tokens
```

**执行约束**：

- **控制流同步阻塞**：单推理请求在调用线程内同步执行；共享 Runtime 的访问由宿主串行化，包含不同 Session 的 Generate。每次请求重新 Prefill，返回前释放 KV reservation。
- **算子并行状态**：当前未交付 OpenMP/线程池算子执行。后续为满足多核性能目标引入内部并行时，须先明确线程所有权、同步、确定性、workspace 和稳态分配合同，并通过算子开发流程与 TSAN 门禁；不因此开放并发请求或调度器。
- 贪婪采样唯一：`next_token = argmax(logits)`
- 确定性：在固定硬件、build、有效 CPU features、weight policy、preset 与 kernel 精度下，相同模型和输入产生一致输出。跨平台位对等仅作为 reference kernel 的验证目标，须按具体算子合同取得证据；与 HF 及不同累加策略的对照采用冻结的数值容差。

#### [P0] 算子分发

- **准备期解析**：compiler 仅生成纯数据 `KernelSelector`。execution 通过 `Backend::PrepareKernel` 解析；CPU 根据 op/selector、`effective_features ⊇ cpu_requirements` 与 descriptor priority 选核，冻结 entry point、params builder、workspace requirement 和 exact packing recipe。
- **执行期复用**：每个 step 使用缓存的 kernel function pointer 与 prepared params，不重新查询 registry 或调用 backend 选核。kernel 内 shape 特化与 prepared conversion table 依各自合同实现。
- **数值基准**：reference kernel 的累加精度按算子合同定义。当前 FP32 Linear reference 使用 double 累加，W16 projection reference 按 K 升序进行独立 FP32 product/add。CPU FP32 RoPE 使用 double 计算频率、角度、三角函数和旋转乘加，最终写回 FP32；与固定版本 HF 按误差门禁对照，见 [RoPE 算子契约](operators/rope/RoPE算子契约.md) 和 [W16/A32 数值合同](guides/w16-fp32-inference.md)。

#### [P0] 内核列表（最小集）

| 算子 | 当前实现 / 交付目标 | 验证方式 |
|------|--------------------|----------|
| Embedding Lookup | FP32 或 W16 行读取，FP32 output | scalar/selected 对照、边界测试与模型数值 |
| RMSNorm | FP32 Norm 与 activation | 独立数值基准、epsilon/alias/shape 边界 |
| RoPE | typed RoPE 语义与 CPU FP32 执行；具体算法覆盖以验证记录为准 | 与固定 HF 参考及算子合同对照 |
| Attention (GQA) | FP32 scaled dot-product，静态 FP32 KV | scalar oracle、HF logits/KV 对照 |
| Linear / LM head | 当前 FP32、W16/A32；O2 按 exact recipe 消费 packed artifacts | 独立 oracle、stride/alias/zero-size、prepared 与模型回归 |
| Linear (INT8/INT4) | 待交付：仅权重量化 GEMM | 保留误差 < 1% vs FP32 的目标；误差统计量、workload 与质量门禁需在量化专题冻结 |
| SwiGLU | SiLU 与 gating；semantic fusion 和 kernel 实现分别验证 | 算子及模型数值对照 |
| Softmax | Attention 的 stable softmax 数值需求；独立 op/kernel 的支持状态单独记录 | 概率归一化、极值与 Attention 数值对照 |

### 3.4 C ABI 规范

#### 版本化策略

- **当前产品目标是发布 ABI v1.0**。M5 冻结 [公共 C 声明](../include/c_api.h) 与 [完整合同](api/runtime-c-abi.md)，M6 已提供 Runtime/model/session 与版本查询定义及纯 C 调用证据。实现完成不等于发行或全部产品门禁通过。
- `AM_ABI_VERSION = (major << 16) | minor`，v1.0 为 `0x00010000`；`am_get_abi_version` 返回库实际实现的版本。
- `am_config_t` / `am_session_config_t` 使用 `struct_size`、`abi_version` 与固定宽度字段；v1 大小分别为 48 / 24 bytes，原布局固定。拒绝不支持的 major、更高 minor、尺寸不符和非法字段，不隐式猜测默认配置。
- v1.x 保持旧结构、字段 offset、常量数值和函数签名；扩展使用另命名的结构/函数，破坏性变更通过 `_v2` 显式区分。

#### 核心接口

```c
#include "c_api.h"

// 生命周期；所有 create/load 的 error out 可为 NULL。
am_runtime_t am_runtime_create(const am_config_t* config, am_error_handle* error);
void am_runtime_destroy(am_runtime_t runtime);

// 模型
am_model_t am_model_load(am_runtime_t runtime, const char* path, am_error_handle* error);
void am_model_unload(am_model_t model);

// 会话
am_session_t am_session_create(am_model_t model, const am_session_config_t* config,
                               am_error_handle* error);
void am_session_destroy(am_session_t session);
am_status_code am_session_generate(
    am_session_t session,
    const uint32_t* input_tokens, size_t input_len,
    uint32_t* output_tokens, size_t output_capacity, size_t* output_len,
    am_error_handle* error
);

// 已实现的错误接口继续复用。
am_error_handle am_error_create(am_status_code code, const char* message);
void am_error_destroy(am_error_handle error);
am_status_code am_error_code(am_error_handle error);
const char* am_error_message(am_error_handle error);
```

Runtime 创建时显式给出 KV 层数、KV heads、逻辑 head_dim、capacity、FP32 dtype 与 CPU preset；此时立即分配 KV。模型 Load→共享 FP32 准入→Compile 后，由 adapter 唯一的跨对象 helper 校验所有层 K/V state dtype、几何及 context 上限，再 Prepare/发布 model handle，不修改已有 Runtime/KV。

model/session 保留必要 owner，支持外部 runtime/model handle 先释放；共享 Runtime 的所有访问由宿主串行化。destroy/unload(NULL) 为 no-op，非空 handle 仅释放一次，不承诺诊断伪造或已释放的 handle。

Generate 的 output_len 必填；正数 N 要求 output_capacity 至少 N，短 buffer 在执行前返回 OutOfRange，output_len=N 表示安全容量上界。容量足够时才执行，成功后一次复制完整新 IDs 并写实际长度；其他失败长度为 0、token buffer 不变。N=0 仍验证 prompt/EOS/model context，但不预约/执行。错误对象是可选诊断：自身 OOM 时创建返回 NULL、error 为 NULL；Generate 仍返回原失败状态码，无异常越过 C 边界。完整的检查顺序、null、内存与所有权约束以 [冻结合同](api/runtime-c-abi.md) 为准。

---

## 4. 非功能需求

### 4.1 性能门禁与基准指标

**交付门禁（Hard Gates，未达成则当前产品不可交付）**：

| 指标 | 目标 | 测试条件与验证入口 |
|------|------|--------------------|
| 功能正确性 | 必需用例 100% 通过 | FP32 独立 HF logits/KV、greedy 与 context-boundary 对照使用 `ctest --preset validation-numerical`；CLI/C ABI、精度扩展与错误恢复按 §5 映射另行覆盖 |
| 稳态零分配 | 指定执行区间的 malloc-family 调用全部为 0 | §3.2 定义的 prepared Decode 区间；普通 Linux/glibc interposer 构建验证，FP32 与 W16 的专用测试分别记录 |
| 确定性 | 固定执行配置下结果 100% 一致 | 保留相同输入 100 次重复与 hash 一致的目标；现有重复请求回归之外，完整 100 次记录需单独取得 |
| 无数据竞争 | 适用测试无 TSAN 报告 | 配置/构建后 `ctest --preset validation-tsan`；覆盖同步引擎与 thread primitives，不证明共享 Runtime 可并发调用 |
| 内存安全与无泄漏 | 必需范围无 ASan/UBSan/LSan 报告 | 配置/构建后 `ctest --preset validation-asan-ubsan`；泄漏通过必须具有实际开启 leak detection 且正常结束的证据 |

执行前按根 [README](../README.md) 准备 fixtures、configure 和 build。CTest presets 的 `noTestsAction=error` 防止空发现；[最终验证驱动](../tools/run_final_validation.py) 进一步核对冻结 inventory 与实际 JUnit 的集合、数量、skip 和退出状态。新增精度/算子测试是否纳入某个 preset，须核对 [测试分类](../cmake/AetherMindTests.cmake) 与 [冻结清单](../tests/fixtures/final_validation/policy.json)，不能由 suite 存在或历史通过推定。

TSAN 在 `main` 前因主机环境启动失败须单独记录；sanitizer 不兼容的 interposer 测试须由普通构建补足。设置 `detect_leaks=0`、LSan 因 ptrace 环境退出、跳过或未执行，都不能记为泄漏门禁通过。现有五批集成验证与 W16 专项验证的范围和限制分别保留，不构成当前全部产品 hard gates 已通过的声明。

**目标基准（Target Benchmarks，允许基于量化误差进行经评审的小幅校准）**：

| 指标 | 目标 | 测试条件 | 备注 |
|------|------|----------|------|
| 吞吐量 | ≥ 10 tok/s | 7B INT4, 4-core x86, 2k context | 目标而非承诺 |
| 内存占用 | ≤ 4GB | 同上 | 包含运行时开销 |
| 启动时间 | ≤ 2s | 从 `load()` 到首次 `generate()` | 冷启动 |
| TTFT | ≤ 500ms | 2k tokens prefill | 单请求 |

上述数值继续作为目标，尚未取得 7B 量化模型的完整验收证据。采集时须冻结模型 revision、attention/KV geometry、quantization scheme、build/ISA、线程与 affinity、warmup、repetitions、缓存状态、计时区间和内存计量单位。总内存包含实际驻留权重、额外 packed artifacts、KV、request resources 与运行时开销，不能仅用 7B × 4 bits 的逻辑权重估算代替。

现有 model benchmark 区分 Load、转换、Compile、Runtime、Prepare、内部 prepared Prefill/Decode 与公开 Generate。artifact driver 在执行前读取模型以收集身份，故当前 load timing 不能验收冷文件缓存启动目标。吞吐、TTFT 与启动目标需采用各自完整的测量区间；方法与证据规范见 [算子开发流程](guides/operator-development-workflow.md) 与根 [README](../README.md)。

### 4.2 工程规范

编码与注释规则引用 [AGENTS.md](../AGENTS.md)、[C++ 风格指南](guides/cpp_coding_style_guidelines.md)、[注释指南](guides/cpp_comment_guidelines.md) 与 [测试指南](guides/test_writing_guidelines.md)。运行时公共契约以当前头文件和以下规则为准：

- 使用 C++20、RAII、明确 ownership/lifetime 与 const correctness；命名遵循所在模块已有接口，不引入平行规范。
- 可恢复业务错误使用项目 [`Status` / `StatusOr<T>`](../include/aethermind/base/status.h)。`AM_CHECK`/`AM_DCHECK` 服务于程序不变量，用户输入、资源不足与支持范围错误不得用 abort 代替状态返回。
- 分配异常在对外约定的边界转换为 `ResourceExhausted`；诊断 OOM 允许空 message。C adapter 捕获异常，任何异常不得越过 C ABI。
- 不为本运行时接口引入 C++23 `std::expected` 依赖或另一套 `Expected` 返回合同。现有异常工具的存在不改变运行时的 Status API。
- Backend/Allocator 等准备与资源接口允许 virtual；执行 hot path 消费 prepared 函数指针。Concepts 按类型约束需要使用，不要求替换全部虚接口。
- 不使用 C++20 Modules；公共 C++ 合同改变时消费方需重新编译，C ABI 兼容按 §3.4 管理。
- 算子语义、reference/optimized kernels、packing/layout、fusion 和 threading 变更必须遵循 [算子开发流程](guides/operator-development-workflow.md) 的适用 O0–O4 证据门禁。性能结论需有对应 prepared production path 或端到端证据，microbenchmark 仅作诊断。

---

## 5. 验收标准与测试矩阵

### 5.1 当前测试入口与待完成门禁

下表是能力到验收入口的映射，不声明某次工作树已执行通过。

| 范围 | 验收入口 | 边界 |
|------|----------|------|
| HF 加载/准入 | `ModelLoader_*`、`HfFixedTrajectory` | parser/loader 与固定轨迹；不代表所有模型 |
| 执行/Session | `TinyLlamaExecution`、`DirectPrefillDecode` | scalar oracle、Generate、OOM/重试、KV 生命周期 |
| 精度策略 | `Fp32CompatibleInference`、`W16*`、`KV16*` | 转换/保留策略的数值、预算与零分配 |
| KV/资源 | `KVCache*`、`*ResourceLimits` | generation/commit、payload cap；不替代 RSS |
| C ABI/CLI | `CAbi*`、`InferenceCliIntegration` | strict FP32、buffer/error/owner |
| ISA/sanitizer | baseline/AVX2、ASan/UBSan/LSan、TSAN presets | 每个候选与 fallback 的适用覆盖 |
| 待完成 | INT8/INT4、A16、代表性 4k/7B、100 次确定性、其他平台 | 需实际实现与可执行门禁 |

既有五批集成入口：

```bash
python3 tools/prepare_validation_assets.py
python3 tools/run_final_validation.py --jobs 4 --build-concurrency 2
```

第一条默认离线校验/准备已有源资产，缺少源 checkpoint 或数值 bundle 时按 [最终验证指南](guides/final-validation.md) 显式准备。第二条运行既有七个 profile；只有全部七项通过、源码与资产身份一致且清单无缺失时，才能记录该五批集成里程碑完成。局部 profile 成功时 `milestone_complete` 仍为 false。新增或扩展的专项覆盖须按实际分类与报告另列；七个 profile 的历史通过不自动覆盖后续增加的测试。

聚焦独立 HF 数值门禁可在准备资产后运行：

```bash
cmake --preset validation-avx2
cmake --build --preset validation-avx2
ctest --preset validation-numerical
```

此入口要求相应可用 ISA 与已验证资产；baseline、完整集成及 sanitizer 的范围见根 [README](../README.md)。单独运行 GTest filter 时必须检查实际发现/执行数量，不能仅凭退出码判定验收。

### 5.2 集成测试

| 测试项 | 描述 | 通过标准与当前边界 |
|--------|------|--------------------|
| FP32 Greedy Generation | 完整 prompt → Prefill → Decode | 固定 HF bundle 的 logits/KV 满足容差，所列 greedy token 序列一致；不泛化为所有 checkpoint 的质量验收 |
| W16/A32 生成 | 同一 reduced checkpoint 与 FP32 compatibility 对照 | 使用 W16 专项冻结的绝对/相对误差与 argmax margin 条件；不要求不同源精度或累加策略无条件 token 逐位一致 |
| 长序列稳定性 | 保留 4k context 完整生成目标 | 无崩溃、无持续内存增长；当前 near-context-limit fixture 仅证明其配置范围，代表性 4k 模型待独立验收 |
| 错误恢复 | 无效模型、短 buffer、预算/OOM、执行失败 | 正确状态、无部分结果发布，资源/owner 按各接口合同恢复；无泄漏需实际检测证据 |
| Session 隔离 | 同一 Runtime 上串行使用多个 Session / 重复请求 | 每次重新 Prefill，KV reservation 与状态不跨请求泄漏；不承诺共享 Runtime 并发 Generate |

### 5.3 性能基准

| 基准项 | 当前入口 / 待实现工作 | 度量与证据要求 |
|--------|---------------------|----------------|
| 模型准备与生成 | Linux `aethermind_model_benchmark` + `tools/run_model_benchmark.py` | 分开记录准备各阶段、prepared Prefill/Decode、公开 Generate、RSS 与 payload；支持明确的 weight policy/preset |
| 算子生产路径 | `aethermind_benchmark` 与 [算子记录](operators/README.md) | 以 registered/resolved kernel + prepared params 为基线，区分 microkernel 与模型收益 |
| KV 与分配器 | 使用当前 benchmark executable 的实际发现清单选择对应 case | latency、payload 与 allocation/data movement 按 workload 留档，不使用不存在的规划 benchmark 名称充当执行入口 |
| 量化 Linear | 待实现 INT8/INT4 kernel/packing 与量化 benchmark | 保留与 FP32/适用 BLAS baseline 的误差和性能对比目标；冻结形状、scheme、线程与 ISA 后才作结论 |

方法、raw artifact、baseline/candidate 身份和性能结论必须满足 [算子开发流程](guides/operator-development-workflow.md) 与根 [README](../README.md)。测量合同测试通过仅验证工具行为，不表示性能目标达成。

---

## 附录 A: 长期演进方向（战略方向，非承诺）

> **警告**：以下内容仅作为战略方向指引，不属于当前产品交付合同。具体实施计划必须基于已验证需求、当前产品实现证据和独立评审重新确定，必要时另行成册。

### 服务化与异构执行方向

**目标方向**：GPU 服务化、连续批处理、HTTP API

**考虑范围**（非承诺）：

- CUDA 后端 + GPU HAL
- PagedAttention + Block-based KV Cache
- 连续批处理 (Continuous Batching)
- Chunked Prefill（与 Decode 混合调度）
- HTTP/gRPC API (OpenAI 兼容)
- Prefix Caching (RadixAttention)

**关键决策点**（进入专项立项前评估）：

- 是否保留 C++ API 作为服务层底层？
- C ABI v1.0 是否需要 v2.0 以支持异步接口？
- 是否引入第三方依赖（gRPC, libevent）？

### 分布式执行方向（Multi-Node）

**目标方向**：张量并行、专家并行、多机调度

**考虑范围**（非承诺）：

- 张量并行 (TP) + NCCL
- 专家并行 (EP) + All-to-All
- C++20 Coroutines 异步调度
- 分层存储 (GPU/CPU/Disk)

### 高级推理与智能体方向（Agentic）

**目标方向**：高级解码、多模态、生产治理

**考虑范围**（非承诺）：

- 投机解码 (Speculative Decoding)
- Multi-LoRA 动态挂载
- 结构化输出 (JSON Schema)
- 多模态输入（图像/音频）
- 自动扩缩容、A/B 测试

---

## 附录 B: 参考文档

- [仓库工程与模块边界](../AGENTS.md)
- [构建、配置与使用入口](../README.md)
- [公共 API](api/public-api.md) 与 [Runtime C ABI v1](api/runtime-c-abi.md)
- [FP32 compatibility](guides/fp32-compatible-inference.md) 与 [W16/A32](guides/w16-fp32-inference.md)
- [CPU Runtime 配置](guides/runtime-configuration.md) 与 [资源预算](guides/resource-budgeting.md)
- [Allocator 公共接口](../include/aethermind/memory/allocator.h) 与 [基础库模块设计](modules/foundation.md)
- [算子开发与证据流程](guides/operator-development-workflow.md)
- [最终验证指南](guides/final-validation.md)

---
