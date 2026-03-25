# AetherMind 推理引擎产品需求文档 (PRD v1.1)

| 文档版本 | V1.1 |
|----------|----------------|
| **项目名称** | **AetherMind** |
| **核心目标** | 交付生产级桌面/服务器 CPU 推理引擎 (Phase 1)，并为服务化/分布式演进预留架构空间 |
| **关键技术** | C++20 (Concepts), Safetensors, INT8/INT4 Quantization, Static KV Cache |
| **更新日期** | 2026年3月 |

---

## 效力说明与设计约束

本文档采用**双层结构**：
- **Phase 1（本文档主体）**：可执行的详细产品合同，聚焦桌面/服务器 CPU 本地推理运行时
- **长期路线图（附录 A）**：战略愿景，Phase 2+ 能力仅作为方向指引，**不构成交付承诺**

**核心设计约束**：
- Phase 1 是**独立可交付的产品**，而非临时原型
- 向后兼容通过**版本化 C ABI** 实现，而非永久稳定承诺
- Phase 2+ 需基于 Phase 1 经验重新评估，可能独立成册

---

## 1. Phase 1 产品概述

### 1.1 定位与边界

**Phase 1 = 桌面/服务器 CPU 推理引擎**

```
Phase 1 边界（本文档）
├── 单进程、单线程、单模型、单请求
├── Token IDs 输入 / Token IDs 输出（无文本分词器）
├── 仅 CPU 后端（GPU 后端在 Phase 2 规划）
├── 同步执行（无调度器/批处理）
├── 贪婪采样唯一（无 temperature/top-k/top-p）
├── 静态预分配 KV Cache（无 PagedAttention）
└── Llama 家族 Dense 模型（无 MoE）
```

**明确排除在 Phase 1 之外**：
- ❌ HTTP/gRPC 服务接口
- ❌ 连续批处理 / Chunked Prefill
- ❌ Prefix Caching / PagedAttention
- ❌ GPU/CUDA 后端
- ❌ 文本分词器（Tokenizer）
- ❌ MoE / 专家并行
- ❌ 投机解码 / Multi-LoRA
- ❌ FP8（仅 INT8/INT4）

### 1.2 核心价值主张

| 价值点 | Phase 1 实现 | 后续阶段演进 |
|--------|--------------|--------------|
| **本地部署** | 面向桌面/服务器 CPU 的 INT8/INT4 量化推理，支持 7B 模型运行 | Phase 2: GPU 卸载 |
| **确定性输出** | 贪婪采样 + 数值稳定的 CPU 参考内核 | Phase 2+: 支持 temperature/top-p |
| **低延迟启动** | 静态预分配，预热后稳态零分配 | Phase 2+: PagedAttention 动态管理 |
| **API 演进** | 定义中：版本化 C ABI (v1.0)，优先实现核心推理接口 | Phase 2+: HTTP API |

### 1.3 目标用户

- **桌面应用开发者**：需要在本地工作站或开发机集成 LLM 推理能力
- **服务器端开发者**：需要在 CPU 服务器环境中部署本地推理能力
- **AI Agent 框架构建者**：需要确定性、可预测的本地推理能力
- **研究人员**：需要数值稳定、可复现的推理基线

---

## 2. Phase 1 架构（严格边界内）

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      API 层 (API Layer)                         │
│  ┌─────────────┐ ┌─────────────┐                               │
│  │   C ABI     │ │   C++ API   │                               │
│  │ (Draft v1.0)│ │  (Header)   │                               │
│  └─────────────┘ └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    运行时层 (Runtime Layer)                      │
│  ┌────────────┐ ┌──────────────────────────────┐ ┌────────────┐│
│  │   Model    │ │      执行器 (Executor)        │ │  KV Cache  ││
│  │  Manager   │ │ ┌──────────┐    ┌──────────┐ │ │  Manager   ││
│  │ (Weights/  │ │ │ Prefill  │    │  Decode  │ │ │ (Static    ││
│  │  Metadata) │ │ └──────────┘    └──────────┘ │ │   Pool)    ││
│  └────────────┘ └──────────────────────────────┘ └────────────┘│
│  ┌────────────┐ ┌────────────┐                                 │
│  │  Loader    │ │ Dispatch   │                                 │
│  │Safetensors │ │ Table      │                                 │
│  │(Repack)    │ │(Static)    │                                 │
│  └────────────┘ └────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    算子层 (Operator Layer)                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │   Kernels  │ │    GEMM    │ │  Attention │ │  Sampling  │  │
│  │(Reference) │ │(BLAS/Int8) │ │(RoPE/GQA)  │ │  (Greedy)  │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                硬件抽象层 (HAL - CPU Backend)                    │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │  X86_64    │ │   ARM64    │ │  Threads   │                  │
│  │(AVX/AMX)   │ │   (NEON)   │ │   (Pool)   │                  │
│  └────────────┘ └────────────┘ └────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

**架构执行准则**：

- 核心计算模型：Phase 1 以 **decoder-only Transformer** 为执行核心，运行时显式区分 **Prefill** 与 **Decode** 两个阶段。
- 核心组件：Phase 1 架构由 `Runtime`（生命周期与资源管理）、`Executor`（同步执行流控）与 `KVCacheManager`（静态 KV 内存池管理）构成。
- 无请求调度器：Phase 1 不引入 `Request Scheduler`，不承担请求排队、批处理、连续批处理或多会话仲裁职责。
- 无虚函数开销：使用 C++20 Concepts + 静态分发
- 无动态内存：稳态零分配（推理预热完成后，Decode 路径排除权重映射与 KV Cache 静态扩容外，无堆内存申请）
- 内存池化：中间工作区与 KV 缓存采用池化管理；Phase 1 使用静态 KV 布局，Paged KV 作为后续演进能力。
- 无跨层依赖：HAL 仅暴露 Concepts，不暴露模板实现

### 2.2 API 边界定义

**核心原则：Token IDs 是唯一数据边界**

> 注：以下 C++ API / C ABI 为 **Phase 1 目标接口草案**，用于冻结功能边界与验收口径；在 v1.0 接口冻结前，其作为开发目标契约，不代表仓库当前已完整实现。

```cpp
// C++ API 伪代码
class Session {
public:
    // 输入: token IDs (uint32_t*), 输出: token IDs (uint32_t*)
    std::vector<uint32_t> Generate(
        const std::vector<uint32_t>& prompt_tokens,
        const GenerationConfig& config  // max_tokens, eos_token_id only
    );
};

// C ABI 伪代码
typedef struct am_session* am_session_t;

// 输入/输出均为 uint32_t 数组，无文本处理
int am_session_generate(
    am_session_t session,
    const uint32_t* input_tokens,
    size_t input_len,
    uint32_t* output_tokens,
    size_t output_capacity,
    size_t* output_len,
    am_error_t* error
);
```

**明确排除**：
- ❌ 字符串输入/输出
- ❌ Tokenizer 集成
- ❌ 流式回调（callbacks）
- ❌ 异步/Promise/Future 接口

---

## 3. Phase 1 功能需求

### 3.1 模型加载与管理

#### [P0] Safetensors 加载器
- **输入**：HuggingFace 格式目录（`config.json` + `*.safetensors`）
- **验证**：
  - 仅接受 Llama-family Dense 模型
  - 显式拒绝：MoE、编码器-解码器、滑动窗口注意力
  - 必需字段：model_type, hidden_size, num_hidden_layers, num_attention_heads, etc.
- **重排**：将权重重排为内部 CPU 优化布局（NCHW/Blocked 格式）
- **量化**：支持 INT8/INT4 仅权重量化加载

#### [P0] 模型配置契约
| 配置项 | 支持 | 拒绝策略 |
|--------|------|----------|
| model_type | llama, llama2, llama3 | 非 llama 家族报错 |
| attention | GQA, MQA, MHA | sliding window 报错 |
| activation | SwiGLU | 其他报错 |
| norm | RMSNorm | LayerNorm 报错 |
| quantization | INT8 per-channel, INT4 group-wise (group_size=128) | 其他 scheme 报错 |

### 3.2 内存管理

#### [P0] 静态预分配策略

```
内存布局（预热后固定）：
┌─────────────────────────────────────────┐
│ Model Weights (INT4/INT8/FP16)          │
│ ~3.5GB for 7B INT4                      │
├─────────────────────────────────────────┤
│ KV Cache (Static,FP16/BF16 format)      │
│ [layer][head][seq_len][head_dim]        │
│ Max 4k tokens, pre-allocated            │
├─────────────────────────────────────────┤
│ Workspace (Scratch)                     │
│ ammalloc 管理的瞬时工作区                  │
│ Decode 稳态零分配                         │
└─────────────────────────────────────────┘
```

**验收判定依据**：
- [ ] 稳态零分配验证（预热后 `malloc`/`free` 调用次数为 0，通过钩子验证）
- [ ] 内存占用在目标硬件范围内（见 4.1 节）
- [ ] 无内存泄漏（ASAN/Valgrind 通过）

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
- **控制流同步阻塞**：单推理请求在调用线程内同步执行。算子层（GEMM/Attention）支持基于 OpenMP 的并行加速以满足性能门禁。
- 贪婪采样唯一：`next_token = argmax(logits)`
- 确定性：在相同硬件平台与算子精度（Reference/Optimized）下，相同输入产生一致输出；跨平台位对等（Bit-identical）仅作为参考内核（Reference Kernels）的验证目标。

#### [P0] 算子分发
- **静态分发**：编译期确定 `op_id + cpu_feature + quant_scheme → kernel_ptr`
- **无运行时查找**：通过模板特化/函数指针缓存避免虚函数
- **参考内核**：Float32 累积，数值稳定实现作为正确性基准

#### [P0] 内核列表（最小集）

| 算子 | 实现要求 | 验证方式 |
|------|----------|----------|
| Embedding Lookup | 向量 gather | golden test |
| RMSNorm | 在线计算 1/sqrt(mean(x^2)+eps) | 数值奇偶校验 |
| RoPE | 旋转位置编码 | 与 HF 实现对比 |
| Attention (GQA) | 标准 scaled dot-product | golden test |
| Linear (INT8/INT4) | 仅权重量化 GEMM | 误差 < 1% vs FP32 |
| SwiGLU | Swish + Gating | 数值奇偶校验 |
| Softmax | 在线 stable softmax | 数值奇偶校验 |

### 3.4 C ABI 规范

#### 版本化策略
- **Phase 1 目标是发布 ABI v1.0**
- 稳定性承诺以最终发布的 runtime C ABI 为准；在接口冻结前，本文中的函数签名视为目标契约而非既成实现
- 向前兼容：v1.x 保持向后兼容，新增功能通过扩展接口
- 破坏性变更：通过 `_v2` 后缀显式区分，不隐式兼容

#### 核心接口

```c
// 生命周期
am_runtime_t am_runtime_create(const am_config_t* config);
void am_runtime_destroy(am_runtime_t runtime);

// 模型
am_model_t am_model_load(am_runtime_t runtime, const char* path, am_error_t* error);
void am_model_unload(am_model_t model);

// 会话
am_session_t am_session_create(am_model_t model, const am_session_config_t* config);
void am_session_destroy(am_session_t session);
int am_session_generate(
    am_session_t session,
    const uint32_t* input_tokens, size_t input_len,
    uint32_t* output_tokens, size_t output_capacity, size_t* output_len,
    am_error_t* error
);

// 错误处理
const char* am_error_message(am_error_t error);
void am_error_free(am_error_t error);
```

**ABI 稳定性目标（v1.x 内，针对最终发布接口）**：
- 结构体大小不变（仅添加字段到末尾）
- 函数签名不变
- 枚举值不变（新增追加到末尾）

---

## 4. 非功能需求

### 4.1 性能门禁与基准指标

**交付门禁（Hard Gates，未达成则 Phase 1 交付失败）**：

| 指标 | 目标 | 测试条件 | 验证命令 |
|------|------|----------|----------|
| 功能正确性 | 100% | 与 HF 参考实现对比 | `--gtest_filter=LlamaDecode.*` |
| 稳态零分配 | 100% | Decode 循环 (排除 weight mapping) | 自定义 malloc 钩子统计 |
| 确定性 | 100% | 同平台相同输入输出 | 100 次重复测试 hash 一致 |
| 无数据竞争 | 100% | TSAN 检测 | `cmake -DENABLE_TSAN=ON` |
| 内存无泄漏 | 100% | ASAN/LSAN 检测 | `cmake -DENABLE_ASAN=ON` |

**目标基准（Target Benchmarks，允许基于量化误差进行经评审的小幅校准）**：

| 指标 | 目标 | 测试条件 | 备注 |
|------|------|----------|------|
| 吞吐量 | ≥ 10 tok/s | 7B INT4, 4-core x86, 2k context | 目标而非承诺 |
| 内存占用 | ≤ 4GB | 同上 | 包含运行时开销 |
| 启动时间 | ≤ 2s | 从 `load()` 到首次 `generate()` | 冷启动 |
| TTFT | ≤ 500ms | 2k tokens prefill | 单请求 |

### 4.2 工程规范

**编码规范直接引用 AGENTS.md**：
- 头文件保护：使用 `#ifndef AETHERMIND_XXX_H`（非 `#pragma once`）
- 命名约定：PascalCase 类型、snake_case 函数、kPrefix 枚举
- 错误处理：`AM_THROW`, `AM_CHECK`, `AM_DCHECK`
- 性能宏：`AM_LIKELY`, `AM_UNLIKELY`, `AM_ALWAYS_INLINE`

**代码规范例外说明**：
- 使用 C++20 Concepts 替代虚函数（与 AGENTS.md 不冲突）
- 错误处理机制：优先尝试 std::expected (C++23) 或编译器补丁版本。若环境受限，必须回退至项目内置的 `aethermind::Expected<T, E>`，禁止直接使用异常抛出作为业务错误路径。
- 不使用 C++20 Modules（与 AGENTS.md 一致）

---

## 5. 验收标准与测试矩阵

### 5.1 单元测试

> 注：以下为 **目标测试套件**，按里程碑逐步落地并纳入验收。

| 测试套件 | 覆盖范围 | 验收阶段 |
|----------|----------|----------|
| `InferenceRuntimeContract.*` | API 边界与生命周期 | M1 |
| `ModelConfigContract.*` | 配置解析与拒绝策略 | M1 |
| `SafetensorsLoader.*` | 加载器与重排 | M1 |
| `ReferenceCpuKernels.*` | 参考内核数值正确性 | M1/M2 |
| `QuantLinearInt8.*` | INT8 量化奇偶校验 | M2 |
| `QuantLinearInt4.*` | INT4 量化奇偶校验 | M2 |
| `KVCache.*` | 静态缓存管理 | M2 |
| `LlamaDecode.*` | 端到端解码正确性 | M2 |
| `CAbiRuntime.*` | C ABI 功能与内存安全 | M2 |
| `InferenceRuntimeThreading.*` | TSAN 无竞争验证 | M2 |

### 5.2 集成测试

| 测试项 | 描述 | 通过标准 |
|--------|------|----------|
| 端到端 Greedy Generation | 完整 prompt → decode 流程 | 与 HF 参考输出 token 序列一致 |
| 长序列稳定性 | 4k context 完整生成 | 无崩溃、无内存增长 |
| 错误恢复 | 加载无效模型 | 干净报错、无内存泄漏 |
| 多会话隔离 | 多个独立 session | 状态不互相干扰 |

### 5.3 性能基准

| 基准项 | 度量指标 | 目标 |
|--------|----------|------|
| `BM_LlamaDecode` | tok/s | 测量并记录基线 |
| `BM_KVCache` | 访问延迟 | 测量并记录基线 |
| `BM_QuantLinear` | GEMM GFLOPS | 与 OpenBLAS 对比 |
| `BM_MemoryPool` | 分配/释放延迟 | ammalloc 性能验证 |

---

## 6. Phase 1 里程碑

### M1: 基础运行时 (Week 1-4)

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 1.1 运行时契约 | `include/runtime/`（目标布局）头文件 | 编译通过，无外部依赖 |
| 1.2 模型配置 | Llama config 解析器 | 支持/拒绝测试通过 |
| 1.3 执行上下文 | CPU 运行时 + ammalloc 集成 | 预热后零分配验证 |
| 1.4 算子分发 | 静态分发表示 | 所有 V1 算子可解析 |

### M2: 内核与集成 (Week 5-8)

| 任务 | 交付物 | 验收标准 |
|------|--------|----------|
| 1.5 参考内核 | RMSNorm, RoPE, Attention, SwiGLU | 数值奇偶校验 |
| 1.6 量化内核 | INT8/INT4 GEMM | 误差 < 1% |
| 1.7 KV 缓存 | 静态预分配实现 | 单请求确定性 |
| 1.8 解码执行 | 完整 Llama 解码 | 端到端 Golden Test |
| 1.9 C ABI | v1.0 接口 | ABI 稳定性测试 |
| 1.10 工程加固 | CI + 消毒器 + 基准 | 全部通过 |

---

## 附录 A: 长期路线图（战略方向，非承诺）

> **警告**：以下内容仅作为战略方向指引，具体实施计划需在 Phase 1 完成后重新评估，可能独立成册。

### Phase 2: 服务化引擎（GPU-First）
**目标方向**：GPU 服务化、连续批处理、HTTP API

**考虑范围**（非承诺）：
- CUDA 后端 + GPU HAL
- PagedAttention + Block-based KV Cache
- 连续批处理 (Continuous Batching)
- Chunked Prefill（与 Decode 混合调度）
- HTTP/gRPC API (OpenAI 兼容)
- Prefix Caching (RadixAttention)

**关键决策点**（Phase 1 后评估）：
- 是否保留 C++ API 作为服务层底层？
- C ABI v1.0 是否需要 v2.0 以支持异步接口？
- 是否引入第三方依赖（gRPC, libevent）？

### Phase 3: 分布式引擎（Multi-Node）
**目标方向**：张量并行、专家并行、多机调度

**考虑范围**（非承诺）：
- 张量并行 (TP) + NCCL
- 专家并行 (EP) + All-to-All
- C++20 Coroutines 异步调度
- 分层存储 (GPU/CPU/Disk)

### Phase 4: 智能体引擎（Agentic）
**目标方向**：高级解码、多模态、生产治理

**考虑范围**（非承诺）：
- 投机解码 (Speculative Decoding)
- Multi-LoRA 动态挂载
- 结构化输出 (JSON Schema)
- 多模态输入（图像/音频）
- 自动扩缩容、A/B 测试

---

## 附录 B: 参考文档

- **Phase 1 详细实施计划**: `.sisyphus/plans/cpu-first-llama-runtime-v1.md`
- **编码规范**: `AGENTS.md`
- **分配器规范**: `ammalloc/AGENTS.md`

---

**文档所有者**: AetherMind 架构团队  
**当前状态**: ✅ Phase 1 技术交付契约已核准  
**下次更新**: M1 里程碑完成后补充实践经验
