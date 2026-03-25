# AetherMind 推理引擎整体架构设计文档

**版本**: v1.1  
**日期**: 2026-03-25  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [设计目标](#2-设计目标)
3. [整体架构](#3-整体架构)
4. [核心模块职责](#4-核心模块职责)
5. [关键数据流](#5-关键数据流)
6. [内存架构](#6-内存架构)
7. [并发模型](#7-并发模型)
8. [确定性策略](#8-确定性策略)
9. [实现级 Contract 补充](#9-实现级-contract-补充)
10. [目录与模块布局建议](#10-目录与模块布局建议)
11. [Phase 1 与 Phase 2 的边界](#11-phase-1-与-phase-2-的边界)
12. [关键架构决策](#12-关键架构决策)
13. [结语](#13-结语)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind 大模型推理引擎的整体技术架构，面向后续实现阶段提供统一的分层模型、模块职责划分、数据流约束、实现级 contract 与演进边界。

本文档的目标不是一次性覆盖所有未来能力，而是在 Phase 1 内优先冻结：

- 运行时边界
- 对象模型
- 执行与内存契约
- 算子签名语义
- 可验证的正确性基线

### 1.2 Phase 1 定位

Phase 1 目标是交付一个**面向桌面与服务器 CPU 的本地推理运行时**，其范围严格限定为：

- 单进程（single-process）
- 单模型（single-model）
- 单请求（single-request）
- 同步阻塞执行（synchronous execution）
- Token IDs 输入 / Token IDs 输出
- Llama-family dense 模型
- 静态 KV Cache
- 贪婪采样（greedy decoding）

### 1.3 核心原则

| 原则 | 说明 |
|------|------|
| **边界优先** | 先冻结 runtime contract，再扩展实现 |
| **正确性优先** | 先交付 reference path，再引入 optimized kernels |
| **稳态零分配** | Decode 稳态路径禁止新的堆分配 |
| **静态分发** | 热路径不依赖虚函数或装箱调度 |
| **窄接口设计** | runtime 核心仅负责 token-to-token 推理 |
| **对象分层** | 区分外部模型资产、内部可执行表示与运行时句柄 |
| **实现可验证** | Reference 与 Optimized 路径必须具备稳定对照关系 |

---

## 2. 设计目标

### 2.1 功能目标

- 支持 HuggingFace `config.json` + `*.safetensors` 的模型加载路径
- 支持 Llama-family decoder-only dense 模型执行
- 支持 Prefill + Decode 推理流程
- 支持静态 KV Cache 管理
- 支持 C++ API 与薄 C ABI
- 支持至少一条可验证的量化线性路径

### 2.2 性能目标

- 推理预热完成后，Decode 路径满足**稳态零分配**
- 算子层支持 CPU feature-aware 优化
- 热路径采用缓存函数指针，避免重复分发开销
- Reference 内核作为正确性基线，Optimized 内核作为性能路径
- 线程策略受统一 Runtime Context 管理，避免 oversubscription

### 2.3 非目标

Phase 1 不承担以下能力：

- HTTP / gRPC 服务接口
- 连续批处理（continuous batching）
- Chunked Prefill
- Prefix Caching / PagedAttention
- GPU / CUDA 后端
- Tokenizer 集成到 runtime 核心
- MoE / 多模型调度
- Speculative Decoding / Multi-LoRA
- Request-level scheduler

---

## 3. 整体架构

### 3.1 分层架构图

```text
┌──────────────────────────────────────────────┐
│              Application Layer              │
│ embedding app / agent framework / adapter   │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│                  API Layer                   │
│  C++ API (Session)    C ABI (opaque handles) │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│                Runtime Layer                 │
│ Runtime Context | Model Manager | Errors     │
│ 全局资源初始化、线程策略、模型句柄与错误边界管理 │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│               Execution Layer                │
│ ┌──────────────────────────────────────────┐ │
│ │            Executor (同步执行器)          │ │
│ │  Prefill Path  <──>  Decode Path (Sync)  │ │
│ └──────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────┐ │
│ │       KV Cache Manager / KVCacheView     │ │
│ │  Static Storage | Layout Contract        │ │
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│                 Model Layer                  │
│ Config Parser | Safetensors Loader           │
│ Model Assets | Weight Manifest | Repack      │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│               Operator Layer                 │
│ Decoder-only Transformer Kernels (Ref/Opt)  │
│ Embedding | Attention | RMSNorm | Linear    │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│           Dispatch / Kernel Layer            │
│ op_id + dtype + quant + cpu_feature          │
│        -> cached function pointer            │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│             HAL / Platform Layer             │
│ CPU feature detect | threading | timing      │
│ mmap / file IO | alignment | memory policy   │
└──────────────────────────────────────────────┘
```

### 3.2 分层设计原则

1. **API 层**只暴露集成接口，不承载推理实现细节。
2. **Runtime 层**是系统装配与生命周期中枢，负责全局资源、线程策略、模型句柄与错误管理。
3. **Execution 层**组织 Prefill / Decode 流程，并以 `Executor + KVCacheManager` 为核心，但不直接承担模型加载职责。
4. **Model 层**负责从外部模型资产构建内部可执行表示。
5. **Operator 层**提供算子语义实现。
6. **Dispatch 层**负责运行前绑定热路径函数指针。
7. **HAL 层**只抽象必要的平台能力，不做泛化过度的设备框架。

---

## 4. 核心模块职责

### 4.1 API Layer

#### C++ API

面向桌面应用与服务器端本地集成场景，提供类型安全的 runtime / loaded-model / session 封装。

职责：

- 管理对象生命周期
- 封装 generation config
- 保持 token-based 输入输出边界
- 为测试与示例提供自然接口

#### C ABI

面向跨语言绑定与稳定边界。

职责：

- 使用 opaque handle 暴露 runtime / model / session
- 显式 create / load / generate / destroy / error API
- 避免暴露 STL、模板和内部对象布局

### 4.2 Runtime Layer

#### Runtime Context

职责：

- CPU 特性检测
- 线程策略配置
- allocator policy 配置
- workspace sizing
- dispatch table 初始化

#### Model Manager

职责：

- 管理模型加载流程的统一入口
- 协调 Config Parser、Safetensors Loader 与 Repack Store
- 构建并返回可执行模型句柄
- 管理模型级资源的生命周期

#### Loaded Model

职责：

- 持有只读的模型运行句柄
- 持有解析后的模型配置
- 持有内部重排后的权重表示
- 持有已绑定完成的 dispatch table
- 暴露只读运行元数据
- 不承担模型加载职责

#### Session

职责：

- 持有单次生成状态
- 管理 KV Cache 生命周期
- 跟踪当前 decode position 与输出 token
- 承载 GenerationConfig

#### Error Subsystem

职责：

- 定义结构化错误码
- 管理错误消息与上下文
- 桥接 C++ 错误与 C ABI 错误对象

### 4.3 Model Layer

#### Config Parser

职责：

- 解析 `config.json`
- 校验 llama-family dense 合法性
- 拒绝不支持的模型结构与配置组合

#### Safetensors Loader

职责：

- 解析 Safetensors header 与 shard 信息
- 校验 tensor metadata
- 建立加载清单与原始权重视图

#### Repack Store

职责：

- 将外部权重转换为内部 CPU 友好布局
- 为 INT8 / INT4 kernel 提供稳定输入格式
- 避免 decode 时做临时格式转换

#### 内部对象模型

为降低 Phase 1 的实现复杂度，同时避免 “Model” 一词同时表示外部模型资产、内部可执行表示与运行时句柄，模型相关对象在 V1 中优先收敛为两类核心实体：

1. **Model Assets**
   - 表示来自外部模型目录的原始资产视图
   - 包括 `config.json`、Safetensors metadata 与原始 tensor 映射信息
   - 负责初始化校验与元数据提取，不直接参与热路径推理

2. **Loaded Model**
   - 表示经过校验、重排、量化适配并绑定 Dispatch Table 的内部只读可执行表示
   - 是推理热路径依赖的唯一核心模型对象
   - 生命周期由 Runtime / Model Manager 管理，并对外提供稳定访问边界

若后续版本确有需要，可再将 `Loaded Model` 进一步拆分为“内部可执行对象”与“外部只读句柄”；但 Phase 1 不要求预先固化三层对象体系。

### 4.4 Execution Layer（执行层）

本层是推理计算的组织核心，聚焦于 **decoder-only Transformer** 的单次同步推理，**不承担请求调度职责**。

#### 1. Executor（执行器）

职责：

- 同步流控：驱动从 Prefill 到 Decode 的完整状态机转换
- Prefill 阶段：一次性处理 Prompt 全序列，填充初始 KV Cache 并产出首个 Token
- Decode 阶段：逐 Token 循环执行，直到触发停止条件
- 单请求阻塞：在单次推理任务内保持同步阻塞，不涉及任务切换或并发请求合并

#### 2. KV Cache Manager（KV 缓存管理器）

职责：

- 管理预先分配的 KV 底层存储
- 提供 Session 级 KV 生命周期管理
- 提供逻辑布局约束下的高效索引能力
- 为执行层暴露稳定的 `KVCacheView`

#### 3. 解码状态管理 (Greedy Sampler)

职责：

- 贪婪采样：仅支持基于 Argmax 的确定性 Token 选择
- 停止判定：监控 EOS Token 与 Max Tokens 约束
- 输出管理：维护输出 token 序列与解码终止状态
- 简化说明：在 Phase 1 中，该逻辑通常作为 `Executor` 的私有组件实现，不建议额外扩展为独立控制层

### 4.5 Operator Layer

Phase 1 最小算子集：

- Embedding Lookup
- RMSNorm
- RoPE
- Attention
- Softmax
- SwiGLU
- Residual Add
- Linear（严格遵循 PRD 约束：支持 INT8 per-channel 与 INT4 group-wise 量化方案）
- Argmax

Operator Layer 分为两类实现：

| 类型 | 目标 | 特征 |
|------|------|------|
| **Reference Kernels** | 正确性基线 | FP32 累积、数值稳定、可对照验证 |
| **Optimized Kernels** | 性能路径 | 向量化、blocking、量化优化、受控误差 |

**量化执行契约**：量化算子必须能够处理经 `Repack Store` 转换后的目标布局数据；支持范围与误差门禁以 PRD 为准，即优先支持 INT8 / INT4 方案，并满足相对 FP32 的精度约束。

### 4.6 Dispatch / Kernel Layer

分发核心：

```text
(op_id, cpu_feature, dtype, quant_scheme) -> kernel_fn
```

要求：

- 初始化阶段完成解析与绑定
- 热路径直接调用缓存函数指针
- 不使用虚函数
- 不在 decode 中途进行复杂查找
- 不支持的组合在初始化阶段明确失败

### 4.7 HAL / Platform Layer

职责：

- CPU feature detect
- cache line / alignment 信息暴露
- timing / profiling hooks
- mmap / file IO primitives
- threading policy bridge

HAL 的目标是**暴露必要能力**，而不是抽象一个过度泛化的多后端运行时。

---

## 5. 关键数据流

### 5.1 模型加载流

```text
HF model dir
  ├─ config.json
  └─ *.safetensors
        │
        ▼
Config Parser
        │
        ▼
Safetensors Loader
        │
        ▼
Model Assets
        │
        ▼
Weight Manifest
        │
        ▼
Repack Store
        │
        ▼
Loaded Model
```

### 5.2 推理执行流

```text
input token ids
      │
      ▼
Session Create
      │
      ▼
Session Preparation
      │
      ├─ KV Cache Reserve
      ├─ Workspace Bind
      └─ Output State Init
      ▼
Prefill
      │
      ▼
KV Cache Filled
      │
      ▼
Decode Loop
      │
      ├─ layer stack
      ├─ attention with KV
      ├─ logits
      ├─ argmax
      └─ stop check
      ▼
output token ids
```

补充说明：

- `Engine Warmup` 若存在，应视为 runtime / model 级一次性行为
- `Session Preparation` 是单次生成前的准备阶段
- Phase 1 不要求每次生成都执行额外的 kernel warmup

### 5.3 数据边界约束

- Runtime 核心边界固定为 **token ids in / token ids out**
- 文本分词与反分词不进入 Phase 1 runtime 核心
- Streaming / callback / async future 不属于当前执行模型

---

## 6. 内存架构

### 6.1 内存分区

建议固定为三类：

#### 1. Model Weights

- 生命周期：模型级
- 来源：mmap 或加载后驻留
- 特征：只读、大块、长期驻留

#### 2. KV Cache

- 生命周期：session 级
- 特征：静态预分配
- 按 token 位置递增写入，不做动态扩容

#### 3. Workspace / Scratch

- 生命周期：运行时执行期
- 特征：预留、复用、稳态零分配
- 可由 `ammalloc` 管理瞬时工作区

### 6.2 稳态零分配定义

本设计中的“零分配”定义为：

- 允许模型加载阶段执行 mmap / 初始化分配
- 允许 Session 创建阶段执行一次性预分配
- **Decode 稳态路径禁止新的堆分配**

该定义用于约束热路径行为，而非要求整个进程在所有阶段完全不分配内存。

### 6.3 内存布局原则

- 权重布局服从 kernel-friendly repack 需求
- KV Cache 布局服从顺序写入与按层访问需求
- Workspace 布局服从算子复用与 cache locality 需求
- 持久态与瞬时态内存必须分离管理

### 6.4 KV Cache Layout Contract

Phase 1 的 KV Cache 采用**静态预分配 + 线性地址空间**的实现策略，但算子接口不应直接硬编码某一具体线性实现细节。为兼顾 V1 落地效率与后续演进，建议采用如下抽象边界：

#### 1. 抽象建议（非强制）

- **KVCacheStorage**
  - 推荐作为底层内存持有与释放的逻辑边界
  - 在 Phase 1 中可直接实现为连续静态缓冲区
- **KVCacheView**
  - 作为执行层与 Attention 算子访问 KV 的推荐接口
  - 提供按 layer / kv_head / seq_pos 的逻辑访问能力
- **KVLayoutContract**
  - 用于描述布局、stride、alignment 与 append 规则
  - 作为未来 Paged KV 扩展时的兼容边界

**Phase 1 简化准则**：V1 可先实现一个高效的静态 `KVCacheManager`，只要其对外接口能够稳定支持按 `layer/head/pos` 的逻辑索引即可。

#### 2. Phase 1 布局要求

- K 与 V 使用独立存储区域
- 每个 Session 的 KV 空间在创建阶段一次性预留
- 写入顺序按 token position 单调递增
- 逻辑访问维度至少包含：
  - `layer_idx`
  - `kv_head_idx`
  - `seq_pos`
  - `head_dim_idx`

#### 3. 接口要求

- 执行层与算子层通过 `KVCacheView` 访问 KV 数据
- 算子签名中不得暴露底层物理布局常量
- `seq_pos` 表示逻辑 token 位置，不直接等同于未来扩展场景下的物理地址偏移

#### 4. 演进兼容要求

Phase 2 若引入 Paged KV Cache，应优先替换 `KVCacheStorage` 实现，并保持 `KVCacheView` 的上层访问语义尽量稳定。

---

## 7. 并发模型

### 7.1 Phase 1 执行模型

Phase 1 采用：

> **single-request synchronous runtime + optional intra-op parallel compute**

即：

- 控制流：单请求、同步阻塞
- 算子层：允许 intra-op parallelism
- 不提供 request scheduler
- 不支持多请求批处理

### 7.2 多线程范围

允许：

- GEMM/GEMV 并行
- Attention 局部并行
- 由 OpenMP 或等价机制驱动的算子级加速

不允许：

- 改变 API 的同步语义
- 引入 request-level 并发调度
- 引入多会话共享调度器复杂度

线程实施指南：

- 默认策略：Phase 1 默认采用算子内部（intra-op）并行
- 外部库约束：若依赖 OpenMP 或第三方并行数学库，应通过 `RuntimeContext` 提供统一线程数配置，避免多层嵌套并行导致 oversubscription
- 简化实现：V1 允许直接在算子循环中采用简单并行策略，无需预先构建复杂的自定义线程调度器

---

## 8. 确定性策略

### 8.1 分层确定性定义

#### Contract Level

- 同平台 / 同构建 / 同输入 -> 输出一致

#### Reference Level

- Reference kernels 作为正确性基线
- 尽量保持 bit-stable 或高度稳定的数值行为

#### Optimized Level

- Optimized kernels 与 reference path 在定义容差内一致
- 不以跨平台 bit-identical 作为通用承诺

### 8.2 为什么需要分层定义

若 Phase 1 后续引入：

- SIMD 特化
- 不同 BLAS 路径
- OpenMP 并行优化

则“所有平台 bit-identical”不再是合理的工程承诺。因此，必须将**契约确定性**与**实现路径确定性**区分开。

---

## 9. 实现级 Contract 补充

### 9.1 Tensor / Buffer / View Contract

Phase 1 不引入完整深度学习框架式 Tensor 系统，但必须冻结最小可执行的数据视图契约。

#### 1. 目标

- 为 Operator Layer 提供统一输入输出描述
- 为 Reference / Optimized 两类 kernel 提供稳定签名
- 避免在热路径上传递高层容器或不透明对象

#### 2. 最小约束

每个 Tensor / Buffer 视图至少应具备以下元信息：

- data pointer
- dtype
- rank
- shape
- stride
- alignment
- mutability

#### 3. 所有权约束

Phase 1 约定：

- Execution / Operator 热路径主要处理 **borrowed view**
- 持久态数据由 Model Weights / KV Cache / Workspace 等上层对象持有
- 算子本身不拥有输入输出 buffer 的生命周期

#### 4. 设计原则

- 热路径优先使用轻量 view，而非复杂 owning tensor
- shape / stride contract 必须显式可见
- 不在算子接口中隐式推断布局

### 9.2 Execution Contract

Phase 1 的 Execution Layer 不承担请求调度，但必须冻结执行阶段的最小数据契约。

建议至少稳定以下概念：

- **ExecutorContext**
  - 持有 runtime context、loaded model handle、workspace binding、threading policy
- **SessionState**
  - 持有 prompt tokens、output tokens、current position、stop state、generation config
- **KVCacheView**
  - 提供当前 Session 的 KV 访问视图
- **OpContext**
  - 向算子传递 workspace、threading、profiling 与 dispatch 所需上下文
- **DecodeStepOutput**
  - 表示单步 decode 的 logits、next token 与 stop 状态

该契约的目标是避免将执行器实现演化为一个参数不断膨胀的“大函数”，并为 Prefill / Decode 共享执行基础设施提供稳定边界。

### 9.3 Operator Signature Contract

为保证 Reference Kernels 与 Optimized Kernels 可对照验证，Phase 1 要求关键算子共享一致的逻辑签名语义。

#### 1. 基本原则

- 算子接口描述逻辑输入输出，而不是绑定具体内存实现
- Reference / Optimized 路径在同一签名语义下工作
- 是否 in-place、是否需要 workspace、误差容忍范围必须可定义

#### 2. 建议优先冻结的算子签名

- Embedding
- RMSNorm
- RoPE
- Linear
- Attention
- Argmax

#### 3. 每个算子至少应明确

- 输入 view
- 输出 view
- 所需 workspace
- 是否允许 in-place
- 对齐要求
- 数值语义要求

#### 4. 特别说明

Attention 算子必须通过 `KVCacheView` 访问 KV，不得直接耦合到底层线性静态缓冲区实现。

### 9.4 Threading Policy Contract

Phase 1 允许算子级并行，但应限制线程策略复杂度。

#### 1. 主原则

- Phase 1 应优先保持单一主导的 intra-op threading runtime
- 应避免多个线程运行时在热路径上叠加造成 oversubscription

#### 2. 推荐约束

- 若选择 OpenMP，则外部 BLAS / kernel 库的线程化能力必须受统一配置约束
- 不允许 request-level threading 与 intra-op threading 混用
- 不允许在 Phase 1 中引入多会话共享线程调度器

#### 3. 工程目标

- 保持 API 同步语义不变
- 控制线程数、绑定策略与调度开销
- 避免同一执行路径中出现 nested parallelism 失控问题

### 9.5 测试 Contract

Phase 1 的测试体系至少分为以下三层：

#### 1. Operator Correctness

- 对关键算子执行单元测试
- Reference kernel 作为主要正确性基线

#### 2. Reference vs Optimized Consistency

- 同输入下比较 Reference 与 Optimized 路径
- 采用按算子定义的容差策略，而非统一追求 bit-identical

#### 3. End-to-End Generation Regression

- 对 Prefill + Decode 的完整生成流程做回归测试
- 验证 greedy generation 的输出稳定性
- 验证 EOS / Max Tokens 等停止条件

#### 4. Memory Behavior

- 检查 Session 创建阶段允许的一次性预分配
- 检查 Decode 稳态路径无新增堆分配

---

## 10. 目录与模块布局建议

建议代码目录逐步向下述结构收敛：

```text
include/aethermind/
  runtime/
    runtime.h
    loaded_model.h
    session.h
    config.h
    error.h
    generation_config.h
    execution_context.h
  model/
    model_config.h
    config_parser.h
    safetensors_loader.h
    model_assets.h
    weight_manifest.h
    repack_store.h
    executable_model.h
  execution/
    executor.h
    prefill_path.h
    decode_path.h
    kv_cache.h
    kv_cache_view.h
    generation_loop.h
  ops/
    embedding.h
    rmsnorm.h
    rope.h
    attention.h
    softmax.h
    swiglu.h
    linear.h
    argmax.h
  dispatch/
    dispatch_key.h
    kernel_registry.h
    dispatch_table.h
  platform/
    cpu_features.h
    threading.h
    memory_map.h
  c_api/
    am_runtime.h

src/
  runtime/
  model/
  execution/
  ops/
  dispatch/
  platform/
  c_api/
```

该布局表达的是**目标模块边界**，不表示当前仓库已完整实现该结构。

---

## 11. Phase 1 与 Phase 2 的边界

### 11.1 Phase 1 必须具备

- token-based runtime
- config + safetensors loader
- static KV cache
- reference kernels
- deterministic greedy generation
- C++ API + thin C ABI
- 至少一条可验证的量化线性路径（优先 INT8，可扩展到 INT4）

### 11.2 Phase 1 明确不包含

- tokenizer in runtime core
- HTTP / gRPC serving
- scheduler
- batching
- prefix cache
- paged attention
- GPU backend
- speculative decoding
- LoRA

### 11.3 Phase 2+ 演进路径

Phase 1 的静态架构为以下能力预留扩展点，但在 Phase 1 交付前不引入：

- `Scheduler`：引入请求队列、执行编排与连续批处理（Continuous Batching）调度逻辑
- `Paged KV Cache`：将静态 `KVCacheManager` 升级为支持分页与非连续存储的管理模型
- 更完整的 Quantization 体系：在现有 INT8 / INT4 基础上扩展校准、格式与执行策略
- GPU HAL / Backend：引入可插拔的非 CPU 设备后端
- Tokenizer Adapter / Serving Adapter：在 runtime 核心之外补充上层接入能力

---

## 12. 关键架构决策

在 Phase 1 中，以下决策必须优先冻结：

1. **Runtime 边界：token ids in / token ids out**
2. **执行模型：single-request synchronous**
3. **内存策略：static KV + steady-state zero allocation**
4. **算子策略：reference first, optimized second**
5. **分发策略：init-time resolve, hot-path function pointer**
6. **对象模型：Model Assets / Loaded Model 两层分离**
7. **KV 访问边界：Execution / Operator 通过 KVCacheView 访问 KV**
8. **算子契约：Reference 与 Optimized 路径共享统一逻辑签名**

这些决策一旦冻结，后续 loader、session、kernel、ABI 与测试体系都可以围绕其稳定展开。

---

## 13. 结语

本架构的核心不是一开始覆盖所有推理场景，而是先建立一个**边界清晰、执行稳定、可验证、可扩展**的桌面/服务器 CPU 本地推理底座。

Phase 1 的成功标准，不是“功能铺得足够广”，而是：

- runtime contract 稳定
- reference path 正确
- optimized path 可验证
- 内存与执行模型可控
- 为后续 Phase 2+ 演进保留清晰扩展点

这也是 AetherMind 推理引擎架构设计的出发点。
